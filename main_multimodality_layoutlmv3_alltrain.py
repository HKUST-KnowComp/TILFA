#!/usr/bin/env python
# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import logging
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import ClassLabel, load_dataset, load_metric

import transformers

from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    get_linear_schedule_with_warmup
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
import pandas as pd
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, roc_auc_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from layoutlmv3.data import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.5.0")

from timm.data.constants import \
    IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision import transforms
import torch
import random
from models import MultiModelLayout

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    text_model_name_or_path: str = field(
        default="bert-base-uncased", metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    cross_attn_type: int = field(default=-1, metadata={"help": "cross attention type"})
    use_forget_gate: int = field(default=1, metadata={"help": "use forget gate or not"})
    use_margin_ranking_loss: int = field(default=0, metadata={"help": "use margin ranking loss or not"})
    margin: float = field(default=0.1, metadata={"help": "margin for margin ranking loss"})
    use_pooler: int = field(default=0, metadata={"help": "use pooler or not"})

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: Optional[str] = field(default="ner", metadata={"help": "The name of the task (ner, pos...)."})
    dataset_name: Optional[str] = field(
        default='funsd', metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a csv or JSON file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate on (a csv or JSON file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to predict on (a csv or JSON file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_val_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of validation examples to this "
            "value if set."
        },
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of test examples to this "
            "value if set."
        },
    )
    label_all_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether to put the label for one word on all tokens of generated by that word or just on the "
            "one (in which case the other tokens will have a padding index)."
        },
    )
    return_entity_level_metrics: bool = field(
        default=False,
        metadata={"help": "Whether to return all the entity levels during evaluation or just the overall ones."},
    )
    visual_embed: bool = field(default=True)
    data_dir: Optional[str] = field(default=None)
    input_size: int = field(default=224, metadata={"help": "images input size for backbone"})
    second_input_size: int = field(default=112, metadata={"help": "images input size for discrete vae"})
    train_interpolation: str = field(
        default='bicubic', metadata={"help": "Training interpolation (random, bilinear, bicubic)"})
    second_interpolation: str = field(
        default='lanczos', metadata={"help": "Interpolation for discrete vae (random, bilinear, bicubic)"})
    imagenet_default_mean_and_std: bool = field(default=False, metadata={"help": ""})
    exp_mode: int = field(default=0, metadata={"help": "0: stance; 1: persuasive"})
    use_wordnet: int = field(default=0, metadata={"help": "0: not use; 1: use"})


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def train(model_args, data_args, training_args, train_dataset, eval_dataset, model, data_collator):  # noqa C901
    """ Train the model """

    train_batch_size = training_args.per_gpu_train_batch_size * max(1, training_args.n_gpu)

    train_sampler = (
        RandomSampler(train_dataset)
        if training_args.local_rank == -1
        else DistributedSampler(train_dataset)
    )
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=train_batch_size, collate_fn=data_collator
    )

    if training_args.max_steps > 0:
        t_total = training_args.max_steps
        training_args.num_train_epochs = (
            training_args.max_steps
            // (len(train_dataloader) // training_args.gradient_accumulation_steps)
            + 1
        )
    else:
        t_total = (
            len(train_dataloader)
            // training_args.gradient_accumulation_steps
            * training_args.num_train_epochs
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
            "weight_decay": training_args.weight_decay,
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
        optimizer_grouped_parameters, lr=training_args.learning_rate, eps=training_args.adam_epsilon
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=training_args.warmup_steps, num_training_steps=t_total
    )
    if training_args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training."
            )
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=training_args.fp16_opt_level
        )

    # multi-gpu training (should be after apex fp16 initialization)
    if training_args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if training_args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[training_args.local_rank],
            output_device=training_args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    print("***** Running training *****")
    print("  Num examples = {}".format(len(train_dataset)))
    print("  Num Epochs = {}".format(training_args.num_train_epochs))
    print(
        "  Instantaneous batch size per GPU = {}".format(training_args.per_gpu_train_batch_size)
    )
    print(
        "  Total train batch size (w. parallel, distributed & accumulation) = {}".
            format(train_batch_size
        * training_args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if training_args.local_rank != -1 else 1))
    )
    print("  Gradient Accumulation steps = {}".format(training_args.gradient_accumulation_steps))
    print("  Total optimization steps = {}".format(t_total))

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    # train_iterator = trange(
    #     int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    # )
    set_seed(training_args)  # Added here for reproductibility (even between python 2 and 3)

    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0
    best_macro_f1 = 0.0
    best_auc_score = 0.0

    for epoch_num in range(int(training_args.num_train_epochs)):
        # epoch_iterator = tqdm(
        #     train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0]
        # )
        running_corrects = 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            model.train()
            batch = {k: v.to(training_args.device) for k, v in batch.items()}
            labels = batch["labels"]
            outputs = model(**batch)
            del batch
            loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
            logits = outputs[1]
            preds = logits.detach()
            if model_args.use_margin_ranking_loss == 0:
                preds = torch.argmax(preds, dim=1)
            else:
                preds = (np.sign(preds.cpu())+1)/2
                preds = preds.int().reshape([-1]).cuda()
            running_corrects += torch.sum(preds == labels.reshape(-1))

            if training_args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if training_args.gradient_accumulation_steps > 1:
                loss = loss / training_args.gradient_accumulation_steps

            if training_args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), training_args.max_grad_norm
                )
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), training_args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1


            if training_args.max_steps > 0 and global_step > training_args.max_steps:
                # epoch_iterator.close()
                break

        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train acc: {:.4f}'.format(epoch_acc))

        out_label_ids, preds, (epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) \
            = evaluate(model_args, data_args, training_args, model, eval_dataset, "val", data_collator)

        is_best_epoch = False
        if best_macro_f1 < macro_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = eval_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch_num
            best_macro_f1 = macro_f1
            best_auc_score = auc_score

            is_best_epoch = True

            if (training_args.local_rank in [-1, 0]):
                # Save model checkpoint
                # output_dir = os.path.join(
                #     training_args.output_dir, "epoch-{}".format(best_epoch_num)
                # )
                # if not os.path.exists(output_dir):
                #     os.makedirs(output_dir)

                checkpoint_name = os.path.join(training_args.output_dir, f'model_epoch_{epoch_num + 1}.pth.tar')
                save_checkpoint(training_args, {
                    'epoch': epoch_num + 1,
                    'state_dict': model.state_dict(),
                    'best_f1': best_f1,
                    'optimizer': optimizer.state_dict(),
                }, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

                print("Saving model to %s", training_args.output_dir)

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

        if training_args.max_steps > 0 and global_step > training_args.max_steps:
            # train_iterator.close()
            break

    return global_step, tr_loss / global_step, best_epoch_num


def evaluate(model_args, data_args, training_args, model, eval_dataset, mode, data_collator):

    if not os.path.exists(training_args.output_dir) and training_args.local_rank in [-1, 0]:
        os.makedirs(training_args.output_dir)

    eval_batch_size = training_args.per_gpu_eval_batch_size * max(1, training_args.n_gpu)

    eval_dataloader = DataLoader(
        eval_dataset, batch_size=eval_batch_size, collate_fn=data_collator
    )

    # Eval!
    print("  Num examples = {}".format(len(eval_dataset)))
    print("  Batch size = {}".format(eval_batch_size))
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None

    for _, batch in enumerate(eval_dataloader):
        model.eval()
        batch = {k: v.to(training_args.device) for k, v in batch.items()}

        with torch.no_grad():
            outputs = model(**batch)
            tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = batch["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(
                out_label_ids, batch["labels"].detach().cpu().numpy(), axis=0
            )
        del batch

    eval_loss = eval_loss / nb_eval_steps
    if model_args.use_margin_ranking_loss == 0:
        preds = np.argmax(preds, axis=1)
    else:
        preds = (np.sign(preds)+1)/2
        preds = preds.astype(int).reshape([-1])

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
    # See all possible arguments in layoutlmv3/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = [model_args, data_args, training_args]

    # create experiment dirs
    exp_name = get_exp_name_layout_multi_alltrain(args)
    training_args.output_dir = f"/home/data/zwanggy/2023/image_arg_experiments/{exp_name}"
    make_dir(training_args.output_dir)
    sys.stdout = Logger(os.path.join(training_args.output_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(training_args.output_dir, "error.log"), sys.stderr)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    # logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    print(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    # if is_main_process(training_args.local_rank):
    #     transformers.utils.logging.set_verbosity_info()
    #     transformers.utils.logging.enable_default_handler()
    #     transformers.utils.logging.enable_explicit_format()
    print(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args)

    processor = LayoutProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    for dataset_name in ['gun_control', 'abortion']:
        print(f"\n################################# {dataset_name} ######################################\n")

        training_args.output_dir = f"/home/data/zwanggy/2023/image_arg_experiments/{exp_name}/{dataset_name}"
        make_dir(training_args.output_dir)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        text_tokenizer = AutoTokenizer.from_pretrained(model_args.text_model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            tokenizer_file=None,  # avoid loading from a cached file of the pre-trained model in another machine
            cache_dir=model_args.cache_dir,
            use_fast=True,
            add_prefix_space=True,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = MultiModelLayout(model_args=model_args, num_labels=num_labels)

        # Tokenizer check: this script requires a fast tokenizer.
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise ValueError(
                "This example script only works for models that have a fast tokenizer. Checkout the big table of models "
                "at https://huggingface.co/transformers/index.html#bigtable to find the model types that meet this "
                "requirement"
            )

        if training_args.local_rank == 0:
            torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

        model.to(training_args.device)

        # Preprocessing the dataset
        # Padding strategy
        padding = "max_length" if data_args.pad_to_max_length else False

        if data_args.visual_embed:
            imagenet_default_mean_and_std = data_args.imagenet_default_mean_and_std
            mean = IMAGENET_INCEPTION_MEAN if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_MEAN
            std = IMAGENET_INCEPTION_STD if not imagenet_default_mean_and_std else IMAGENET_DEFAULT_STD
            common_transform = Compose([
                # transforms.ColorJitter(0.4, 0.4, 0.4),
                # transforms.RandomHorizontalFlip(p=0.5),
                RandomResizedCropAndInterpolationWithTwoPic(
                    size=data_args.input_size, interpolation=data_args.train_interpolation),
            ])

            patch_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])

        def encode_annotation(annotation):
            for idx in range(len(annotation)):
                annotation.loc[idx, 'stance'] = encode_stance(annotation.loc[idx, 'stance'])
                annotation.loc[idx, 'persuasiveness'] = encode_persuasiveness(annotation.loc[idx, 'persuasiveness'])
            return annotation

        dataset_cache_name = get_dataset_cache_name_alltrain(model_args, data_args, dataset_name, data_args.exp_mode)
        if os.path.exists(dataset_cache_name) and not data_args.overwrite_cache:
            print("Loading features from cached file {}".format(dataset_cache_name))
            [train_dataset, eval_dataset] = torch.load(dataset_cache_name)
            # df_test = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_dev.csv'), index_col=0)
            # test_annotation = df_test.reset_index()
            # test_annotation = encode_annotation(test_annotation)
            # test_tweet_id = np.array(test_annotation)[:, 0]
        else:
            print("Creating features from dataset file at {}".format(data_args.data_dir))
            # df = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_train.csv'), index_col=0)
            if data_args.exp_mode == 0:  # stance
                if data_args.use_wordnet == 0:  # not use
                    df = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_train_after_stance_augmentation_new.csv'), index_col=0)
                else:  # use
                    df = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_train_after_stance_augmentation_wordnet.csv'), index_col=0)
            else:
                if data_args.use_wordnet == 0:  # not use
                    df = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_train_after_persuasiveness_augmentation_new.csv'), index_col=0)
                else:  # use
                    df = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_train_after_persuasiveness_augmentation_wordnet.csv'), index_col=0)
            df = df.dropna(axis=0, how="any")
            df = shuffle(df, random_state=training_args.seed)
            # dataset_len = len(df)
            # train_annotation = df[:int(dataset_len * 0.8)]
            # train_annotation = train_annotation.reset_index()
            # val_annotation = df[int(dataset_len * 0.8):]
            # val_annotation = val_annotation.reset_index()
            train_annotation = df.reset_index()
            df_val = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_dev.csv'), index_col=0)
            val_annotation = df_val.reset_index()

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

            # df_test = pd.read_csv(os.path.join(data_args.data_dir, dataset_name + '_dev.csv'), index_col=0)
            # test_annotation = df_test.reset_index()
            # test_annotation = encode_annotation(test_annotation)
            # test_tweet_id = np.array(test_annotation)[:, 0]


            train_dataset = processor.get_examples(data_args.data_dir,
                                                    os.path.join(data_args.data_dir, 'images/' + dataset_name),
                                                    data_args.exp_mode, train_annotation, dataset_name, "train")
            eval_dataset = processor.get_examples(data_args.data_dir,
                                                    os.path.join(data_args.data_dir, 'images/' + dataset_name),
                                                    data_args.exp_mode, val_annotation, dataset_name, "val")
            # test_dataset = processor.get_examples(data_args.data_dir,
            #                                         os.path.join(data_args.data_dir, 'images/' + dataset_name),
            #                                         data_args.exp_mode, test_annotation, dataset_name, "test")



            # Tokenize all texts and align the labels with them.
            def tokenize_and_align_labels(examples, augmentation=False):
                tokenized_text_inputs = tokenize_text(examples["tweet_texts"], text_tokenizer)
                tokenized_inputs = tokenizer(
                    examples["tokens"],
                    padding=False,
                    truncation=True,
                    return_overflowing_tokens=False,
                    # We use this argument because the texts in our dataset are lists of words (with a label for each word).
                    is_split_into_words=True,
                )

                labels = []
                bboxes = []
                images = []
                for batch_index in tqdm(range(len(tokenized_inputs["input_ids"]))):
                    word_ids = tokenized_inputs.word_ids(batch_index=batch_index)

                    label = examples["labels"][batch_index]
                    bbox = examples["bboxes"][batch_index]
                    bbox_inputs = []
                    for word_idx in word_ids:
                        if word_idx is None:
                            bbox_inputs.append([0, 0, 0, 0])
                        else:
                            bbox_inputs.append(bbox[word_idx])
                    labels.append(label)
                    bboxes.append(bbox_inputs)

                    if data_args.visual_embed:
                        ipath = examples["image_path"][batch_index]
                        img = pil_loader(ipath)
                        for_patches, _ = common_transform(img, augmentation=augmentation)
                        patch = patch_transform(for_patches)
                        images.append(patch)

                tokenized_inputs["labels"] = labels
                tokenized_inputs["bbox"] = bboxes
                if data_args.visual_embed:
                    tokenized_inputs["images"] = images
                tokenized_inputs["text_input_ids"] = tokenized_text_inputs[0]
                tokenized_inputs["text_attention_mask"] = tokenized_text_inputs[1]

                return tokenized_inputs

            train_dataset = tokenize_and_align_labels(train_dataset)
            eval_dataset = tokenize_and_align_labels(eval_dataset)
            # test_dataset = tokenize_and_align_labels(test_dataset)

            if training_args.local_rank in [-1, 0]:
                print("Saving features into cached file {}".format(dataset_cache_name))
                torch.save([train_dataset, eval_dataset], dataset_cache_name)

        train_dataset = MultiLayoutlmDataset(train_dataset)
        eval_dataset = MultiLayoutlmDataset(eval_dataset)
        # test_dataset = MultiLayoutlmDataset(test_dataset)


        # Data collator
        data_collator = DataCollatorForKeyValueExtraction(
            tokenizer,
            pad_to_multiple_of=8 if training_args.fp16 else None,
            padding=padding,
            max_length=512,
        )

        # Training
        if training_args.do_train:
            global_step, tr_loss, best_epoch_num = train(model_args, data_args, training_args, train_dataset, eval_dataset, model, data_collator)

        # Evaluation
        # if training_args.do_eval and training_args.local_rank in [-1, 0]:
        #     print("#############evaluation#############")
        #
        #     checkpoint = torch.load(os.path.join(training_args.output_dir, f'model_best.pth.tar'))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     model.to(training_args.device)
        #     out_label_ids, preds, (
        #     epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) = evaluate(model_args,
        #         data_args, training_args, model, eval_dataset, "val", data_collator)
        #
        #     print(
        #         'best eval loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
        #             eval_loss, epoch_acc,
        #             epoch_f1,
        #             epoch_precision,
        #             epoch_recall, macro_f1, auc_score))

        # Predict
        # if training_args.do_predict:
        #     print("#############predict#############")
        #
        #     checkpoint = torch.load(os.path.join(training_args.output_dir, f'model_best.pth.tar'))
        #     model.load_state_dict(checkpoint['state_dict'])
        #     model.to(training_args.device)
        #     out_label_ids, preds, (
        #     epoch_f1, epoch_acc, eval_loss, epoch_precision, epoch_recall, macro_f1, auc_score) = evaluate(model_args,
        #         data_args, training_args, model, test_dataset, "test", data_collator)
        #
        #     predict_df = pd.DataFrame(
        #         {"ids": test_tweet_id, "gold_labels": out_label_ids, "predicted_labels": preds,
        #          # "probabilities": predicted_probs
        #          })
        #     predict_df.to_csv(os.path.join(training_args.output_dir, f"test_best_results.csv"), index=False)

            del model
            torch.cuda.empty_cache()


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
