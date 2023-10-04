#!/usr/bin/env python
# coding=utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
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

def evaluate(model_args, data_args, training_args, model, eval_dataset, data_collator, output_name):

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

    # eval_loss = eval_loss / nb_eval_steps
    if model_args.use_margin_ranking_loss == 0:
        predicted_probs = torch.sigmoid(torch.tensor(preds))[:,1].numpy()
        preds = np.argmax(preds, axis=1)
    else:
        preds = (np.sign(preds)+1)/2
        preds = preds.astype(int).reshape([-1])

    # epoch_metrics = classification_report(out_label_ids, preds, output_dict=True, digits=4)
    # epoch_f1 = epoch_metrics["1"]['f1-score']
    # epoch_precision = epoch_metrics["1"]['precision']
    # epoch_recall = epoch_metrics["1"]['recall']
    # epoch_acc = epoch_metrics["accuracy"]
    #
    # macro_f1 = (epoch_metrics["1"]['f1-score'] + epoch_metrics["0"]['f1-score']) / 2
    # # auc_score = roc_auc_score(out_label_ids, preds)
    #
    # print(
    #     'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}'.format(
    #         eval_loss, epoch_acc,
    #         epoch_f1,
    #         epoch_precision,
    #         epoch_recall, macro_f1))
    # print(classification_report(out_label_ids, preds, digits=4))

    # predict_df = pd.DataFrame(
    #     {"tweet_id": test_tweet_id, "gold_labels": out_label_ids, "predicted_labels": preds, "probabilities": predicted_probs})
    predict_df = pd.DataFrame(
        {"tweet_id": test_tweet_id, "predicted_labels": preds, "probabilities": predicted_probs})
    predict_df.to_csv(f"./output/{output_name}", index=False)



if __name__ == '__main__':

    dataset_name = 'gun_control'
    model_name = 'stance_alltrain_multi_layoutlmv3-base_bert-large-uncased_lr1e-05_bs8*2_warmup0_cross-type4_augmentation_useWordNet1_usePooler1'
    output_name = 'stance_alltrain_multi_layoutlmv3-base_bert-large-uncased_lr1e-05_bs8*2_warmup0_cross-type4_augmentation_useWordNet1_usePooler1_gun.csv'

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    args = [model_args, data_args, training_args]

    set_seed(training_args)

    processor = LayoutProcessor()
    label_list = processor.get_labels()
    num_labels = len(label_list)

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
    model.to(training_args.device)

    padding = "max_length" if data_args.pad_to_max_length else False
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


    print("Creating features from dataset file at {}".format(data_args.data_dir))

    # df_test = pd.read_csv(f"./data/{dataset_name}_dev.csv", index_col=0)
    df_test = pd.read_csv(f"./data/{dataset_name}_test.csv", index_col=0)
    df_test.insert(2, 'stance', 'oppose')
    df_test.insert(3, 'persuasiveness', 'no')
    test_annotation = df_test.reset_index()
    test_annotation = encode_annotation(test_annotation)
    test_tweet_id = np.array(test_annotation)[:, 0]

    test_dataset = processor.get_examples(data_args.data_dir,
                                          os.path.join(data_args.data_dir, 'images/' + dataset_name),
                                          data_args.exp_mode, test_annotation, dataset_name, "test")


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

    test_dataset = tokenize_and_align_labels(test_dataset)
    test_dataset = MultiLayoutlmDataset(test_dataset)

    # Data collator
    data_collator = DataCollatorForKeyValueExtraction(
        tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        padding=padding,
        max_length=512,
    )

    checkpoint = torch.load(os.path.join(f"/home/data/zwanggy/2023/image_arg_experiments/{model_name}/{dataset_name}",
                                          f'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(training_args.device)
    evaluate(model_args, data_args, training_args, model, test_dataset, data_collator, output_name)


