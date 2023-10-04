# coding=utf-8
from __future__ import absolute_import, division, print_function

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import argparse
import glob
import logging
import random
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizerFast,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup, AutoTokenizer,
)

from layoutlm import LayoutlmConfig, LayoutlmForSequenceClassification
from layoutlm.data.layout_utils import *

from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_auc_score
from transformers.models.layoutlm import tokenization_layoutlm

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizerFast),
    "roberta": (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForSequenceClassification, BertTokenizerFast),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def train(args, train_dataset, eval_dataset, model, tokenizer):  # noqa C901
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = (
        RandomSampler(train_dataset)
        if args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = (
            args.max_steps
            // (len(train_dataloader) // args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // args.gradient_accumulation_steps
            * args.num_train_epochs
        )

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(args.num_train_epochs))
    print(
        "  Instantaneous batch size per GPU = {}".format(args.per_gpu_train_batch_size)
    )
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = {}".
            format(args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    )
    print("  Gradient Accumulation steps = {}".format(args.gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(t_total))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #     int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    # )
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    if args.exp_mode == 0:  # stance
        label_id = 3
    else:  # persuasiveness
        label_id = 4

    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0
    best_macro_f1 = 0.0
    best_auc_score = 0.0


    for epoch_num in range(int(args.num_train_epochs)):
        # epoch_iterator = tqdm(
        #     train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        # )
        running_corrects = 0
        for step, batch in enumerate(train_dataloader):
            model.train()
            if args.model_type != "layoutlm":
                batch = batch[:4]
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[label_id]
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": labels,
            }
            if args.model_type == "layoutlm":
                inputs["bbox"] = batch[5]
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "layoutlm"] else None
            )  # RoBERTa don't use segment_ids
            outputs = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            preds = logits.detach()
            preds = torch.argmax(preds, dim=1)
            running_corrects += torch.sum(preds == labels.reshape(-1))

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


            if args.max_steps > 0 and global_step > args.max_steps:
                # epoch_iterator.close()
                break

        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train acc: {:.4f}'.format(epoch_acc))

        out_label_ids, preds, (epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) = evaluate(args, model, eval_dataset, "val")

        if best_macro_f1 < macro_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = eval_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch_num
            best_macro_f1 = macro_f1
            best_auc_score = auc_score

            if (args.local_rank in [-1, 0]):
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "epoch-{}".format(best_epoch_num)
                )
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                # torch.save(args, os.path.join(output_dir, "training_args.bin"))
                # tokenizer.save_pretrained(output_dir)
                print("Saving model to %s", output_dir)

        print(
            'val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
                eval_loss, epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall, macro_f1, auc_score))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}, epoch{}'.format(best_loss,
                                                                                                              best_acc,
                                                                                                              best_f1,
                                                                                                              best_precision,
                                                                                                              best_recall,
                                                                                                              best_macro_f1,
                                                                                                              best_auc_score,
                                                                                                              best_epoch_num + 1))
        print(classification_report(out_label_ids, preds, digits=4))

        if args.max_steps > 0 and global_step > args.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_epoch_num


def evaluate(args, model, eval_dataset, mode, prefix=""):

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
    )

    # Eval!
    print("***** Running evaluation {} *****".format(prefix))
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(args.eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    if args.exp_mode == 0:  # stance
        label_id = 3
    else:  # persuasiveness
        label_id = 4

    for _, batch in enumerate(eval_dataloader):
        model.eval()
        if args.model_type != "layoutlm":
            batch = batch[:4]
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "labels": batch[label_id],
            }
            if args.model_type == "layoutlm":
                inputs["bbox"] = batch[5]
            inputs["token_type_ids"] = (
                batch[2] if args.model_type in ["bert", "layoutlm"] else None
            )  # RoBERTa don"t use segment_ids
            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0
            )

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=1)

    epoch_metrics = classification_report(out_label_ids, preds, output_dict=True, digits=4)
    epoch_f1 = epoch_metrics["1"]['f1-score']
    epoch_precision = epoch_metrics["1"]['precision']
    epoch_recall = epoch_metrics["1"]['recall']
    epoch_acc = epoch_metrics["accuracy"]

    macro_f1 = (epoch_metrics["1"]['f1-score'] + epoch_metrics["0"]['f1-score']) / 2
    auc_score = roc_auc_score(out_label_ids, preds)

    if mode == "test":
        print(
            'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
                eval_loss, epoch_acc,
                epoch_f1,
                epoch_precision,
                epoch_recall, macro_f1, auc_score))
        print(classification_report(out_label_ids, preds, digits=4))


    return out_label_ids, preds, (epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name",
    )

    ## Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_test", action="store_true", help="Whether to run test on the test set."
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight deay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=22, help="random seed for initialization"
    )

    parser.add_argument(
        "--tpu",
        action="store_true",
        help="Whether to run on the TPU defined in the environment variables",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        '--exp_mode', default=0, choices=[0, 1], type=int, help='0: stance; 1: persuasive'
    )

    args = parser.parse_args()


    # create experiment dirs
    exp_name = get_exp_name_layout(args)
    args.output_dir = f"./experiments/{exp_name}"
    make_dir(args.output_dir)
    sys.stdout = Logger(os.path.join(args.output_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.output_dir, "error.log"), sys.stderr)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device


    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    print(
        "Process rank: {}, device: {}, n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
            args.local_rank,
            device,
            args.n_gpu,
            bool(args.local_rank != -1),
            args.fp16
        )
    )

    # Set seed
    set_seed(args)

    processor = LayoutProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
    )

    for dataset_name in ['gun_control', 'abortion']:
        print(f"\n##################### {dataset_name} ##########################\n")

        args.output_dir = f"./experiments/{exp_name}/{dataset_name}"
        make_dir(args.output_dir)

        if args.model_name_or_path == "microsoft/layoutlm-base-uncased":
            tokenizer = tokenizer_class.from_pretrained(
                args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                do_lower_case=args.do_lower_case,
            )
        else:
            # microsoft/layoutlm-base-cased
            tokenizer = AutoTokenizer.from_pretrained('microsoft/layoutlm-base-cased')

        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
        )

        if args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(args.device)

        print("Training/evaluation parameters {}".format(args))

        df = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_train.csv'), index_col=0)
        df = shuffle(df, random_state=args.seed)
        dataset_len = len(df)
        train_annotation = df[:int(dataset_len * 0.8)]
        train_annotation = train_annotation.reset_index()
        val_annotation = df[int(dataset_len * 0.8):]
        val_annotation = val_annotation.reset_index()
        # annotation = df.reset_index()

        def encode_annotation(annotation):
            for idx in range(len(annotation)):
                annotation.loc[idx, 'stance'] = encode_stance(annotation.loc[idx, 'stance'])
                annotation.loc[idx, 'persuasiveness'] = encode_persuasiveness(annotation.loc[idx, 'persuasiveness'])
            return annotation

        train_annotation = encode_annotation(train_annotation)
        val_annotation = encode_annotation(val_annotation)

        # oversample
        # if args.exp_mode == 0:
        #     labels = np.array(train_annotation)[:, -3]
        # else:
        #     labels = np.array(train_annotation)[:, -2]
        # features = np.array(range(len(train_annotation))).reshape(-1, 1)
        # oversampler = SMOTE(random_state=args.seed)
        # os_features, os_labels = oversampler.fit_resample(features, labels)
        # print("after oversample: ")
        # print(np.array(list(map(int,os_labels))).sum() / len(os_labels))
        # train_annotation = pd.concat([train_annotation.loc[i] for i in os_features]).reset_index(drop=True)
        # assert len(train_annotation) == len(os_labels)

        df_test = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_dev.csv'), index_col=0)
        test_annotation = df_test.reset_index()
        test_annotation = encode_annotation(test_annotation)
        test_tweet_id = np.array(test_annotation)[:, 0]

        train_dataset = load_and_cache_examples(args, tokenizer, train_annotation, dataset_name, mode="train")
        val_dataset = load_and_cache_examples(args, tokenizer, val_annotation, dataset_name, mode="val")
        test_dataset = load_and_cache_examples(args, tokenizer, test_annotation, dataset_name, mode="test")

        best_epoch_num = -1
        # Training
        if args.do_train:
            global_step, tr_loss, best_epoch_num = train(args, train_dataset, val_dataset, model, tokenizer)
            # print(" global_step = {}, average loss = {}".format(global_step, tr_loss))

        # # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        # if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        #     # Create output directory if needed
        #     if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        #         os.makedirs(args.output_dir)
        #
        #     print("Saving model checkpoint to {}".format(args.output_dir))
        #     # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        #     # They can then be reloaded using `from_pretrained()`
        #     model_to_save = (
        #         model.module if hasattr(model, "module") else model
        #     )  # Take care of distributed/parallel training
        #     model_to_save.save_pretrained(args.output_dir)
        #     tokenizer.save_pretrained(args.output_dir)
        #
        #     # Good practice: save your training arguments together with the trained model
        #     torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
        #
        #     # Load a trained model and vocabulary that you have fine-tuned
        #     model = model_class.from_pretrained(args.output_dir)
        #     tokenizer = tokenizer_class.from_pretrained(
        #         args.output_dir, do_lower_case=args.do_lower_case
        #     )
        #     model.to(args.device)


        # Evaluation
        if args.do_eval and args.local_rank in [-1, 0]:
            if best_epoch_num != -1:
                model_dir = args.output_dir + "/epoch-" + str(best_epoch_num)
            else:
                model_dir = args.output_dir
            # tokenizer = tokenizer_class.from_pretrained(
            #     model_dir, do_lower_case=args.do_lower_case
            # )
            checkpoints = [model_dir]
            # if args.eval_all_checkpoints:
            #     checkpoints = list(
            #         os.path.dirname(c)
            #         for c in sorted(
            #             glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            #         )
            #     )
            print("Evaluate the following checkpoints: {}".format(checkpoints))
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = (
                    checkpoint.split("/")[-1]
                    if checkpoint.find("epoch") != -1 and args.eval_all_checkpoints
                    else ""
                )

                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                out_label_ids, preds, (epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) = evaluate(args, model, val_dataset, mode="val", prefix=prefix)

                print(
                    'best eval loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
                        eval_loss, epoch_acc,
                        epoch_f1,
                        epoch_precision,
                        epoch_recall, macro_f1, auc_score))


        if args.do_test and args.local_rank in [-1, 0]:
            if best_epoch_num != -1:
                model_dir = args.output_dir + "/epoch-" + str(best_epoch_num)
            else:
                model_dir = args.output_dir
            # tokenizer = tokenizer_class.from_pretrained(
            #     model_dir, do_lower_case=args.do_lower_case
            # )
            checkpoints = [model_dir]
            # if args.eval_all_checkpoints:
            #     checkpoints = list(
            #         os.path.dirname(c)
            #         for c in sorted(
            #             glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)
            #         )
            #     )
            print("Evaluate the following checkpoints: {}".format(checkpoints))
            for checkpoint in checkpoints:
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = (
                    checkpoint.split("/")[-1]
                    if checkpoint.find("epoch") != -1 and args.eval_all_checkpoints
                    else ""
                )

                model = model_class.from_pretrained(checkpoint)
                model.to(args.device)
                out_label_ids, preds, (epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) = evaluate(args, model, test_dataset, mode="test", prefix=prefix)

                predict_df = pd.DataFrame(
                    {"ids": test_tweet_id, "gold_labels": out_label_ids, "predicted_labels": preds,
                     # "probabilities": predicted_probs
                     })
                predict_df.to_csv(os.path.join(args.output_dir, f"test_best_results.csv"), index=False)



if __name__ == "__main__":
    main()
