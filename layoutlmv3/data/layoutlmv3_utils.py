
import os
import re
from transformers import DataProcessor
from tqdm import tqdm
from lxml import html
from multiprocessing import Pool
from torch.utils.data import Dataset
import preprocessor
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION, preprocessor.OPT.HASHTAG)
from layoutlmv3.data.image_utils import *
import sys
import pandas as pd
import torch
import shutil

def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


def encode_persuasiveness(label):
    if label == "yes":
        label = 1
    else:
        label = 0
    return label


def encode_stance(label):
    if label == "oppose":
        label = 0
    else:
        label = 1
    return label

def get_longest_length(sentences):
    return max([len(s) for s in sentences])

def tokenize_text(sentences, tokenizer):
    input_ids = []
    attention_masks = []

    max_length = get_longest_length(sentences)
    # print("tokenizing...")
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt')
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

    input_ids = torch.stack(input_ids, dim=0)
    attention_masks = torch.stack(attention_masks, dim=0)

    return input_ids, attention_masks


class LayoutlmDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):

        return {
            "input_ids": self.dataset["input_ids"][idx],
            "attention_mask": self.dataset["attention_mask"][idx],
            "labels": self.dataset["labels"][idx],
            "bbox": self.dataset["bbox"][idx],
            "images": self.dataset["images"][idx],
        }


class MultiLayoutlmDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset["input_ids"])

    def __getitem__(self, idx):

        return {
            "text_input_ids": self.dataset["text_input_ids"][idx],
            "text_attention_mask": self.dataset["text_attention_mask"][idx],
            "input_ids": self.dataset["input_ids"][idx],
            "attention_mask": self.dataset["attention_mask"][idx],
            "labels": self.dataset["labels"][idx],
            "bbox": self.dataset["bbox"][idx],
            "images": self.dataset["images"][idx],
        }


class LayoutProcessor(DataProcessor):
    """Processor for the data set."""

    def worker(self, idx):
        tweet_id = str(self.annotation.loc[idx, 'tweet_id'])
        tweet_text = self.annotation.loc[idx, 'tweet_text']
        stance_label = self.annotation.loc[idx, 'stance']
        persuasiveness_label = self.annotation.loc[idx, 'persuasiveness']
        text, bbox = self.read_hocr_file(tweet_id)
        image_path = os.path.join(self.root_dir, f'{tweet_id}.jpg')
        image, _ = load_image(image_path)
        return [tweet_text, text, bbox, stance_label, persuasiveness_label, image, image_path]

    def get_examples(self, data_dir, root_dir, exp_mode, annotation, dataset_name, mode):
        annotation["tweet_text"] = annotation["tweet_text"].apply(lambda x: preprocessor.clean(x))
        self.data_dir = data_dir
        self.root_dir = root_dir
        self.dataset_name = dataset_name
        self.annotation = annotation
        # with open(os.path.join(data_dir, "labels", "{}.txt".format(mode))) as f:
        #     lines = f.readlines()
        examples = []
        for idx in tqdm(range(len(annotation))):
            example = self.worker(idx)
            examples.append(example)
        return self._create_examples(examples, exp_mode)

    def read_hocr_file(self, tweet_id):
        hocr_file = os.path.join(self.data_dir, "hocr", self.dataset_name, tweet_id + ".html")
        text_buffer = []
        bbox_buffer = []
        try:
            doc = html.parse(hocr_file)
        except AssertionError:
            print(f"{hocr_file} is empty or its format is unacceptable. Skipped.")
            return [], []
        for page in doc.xpath("//*[@class='ocr_page']"):
            page_bbox = [int(x) for x in get_prop(page, "bbox").split()]
            width, height = page_bbox[2], page_bbox[3]
            for word in doc.xpath("//*[@class='ocrx_word']"):
                textnodes = word.xpath(".//text()")
                s = "".join([text for text in textnodes])
                text = re.sub(r"\s+", " ", s).strip()
                if text:
                    text_buffer.append(text)
                    bbox = [int(x) for x in get_prop(word, "bbox").split()]
                    bbox = [
                        bbox[0] / width,
                        bbox[1] / height,
                        bbox[2] / width,
                        bbox[3] / height,
                    ]
                    bbox = [int(x * 1000) for x in bbox]
                    bbox_buffer.append(bbox)
        return text_buffer, bbox_buffer

    def get_labels(self):
        return list(map(str, list(range(2))))

    def _create_examples(self, lines, exp_mode):
        """Creates examples for the training and dev sets."""
        tweet_texts = []
        tokens = []
        bboxes = []
        labels = []
        images = []
        image_paths = []
        for line in lines:
            tweet_text = line[0]
            token = line[1]
            bbox = line[2]
            stance_label = line[3]
            persuasiveness_label = line[4]
            image = line[5]
            image_path = line[6]

            if exp_mode == 0:
                label = stance_label
            else:  # 1
                label = persuasiveness_label

            tweet_texts.append(tweet_text)
            tokens.append(token)
            bboxes.append(bbox)
            labels.append(label)
            images.append(image)
            image_paths.append(image_path)

        examples = {
                "tweet_texts": tweet_texts,
                "tokens": tokens,
                "bboxes": bboxes,
                "labels": labels,
                "image": images,
                "image_path": image_paths
            }

        return examples

def get_exp_name_layout(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    exp_name = f"{exp_mode}_{args[0].model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
               f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
               f"_warmup{args[2].warmup_steps}_no-oversample_cleaned_augmentation"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name

def get_exp_name_layout_alltrain(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    exp_name = f"{exp_mode}_alltrain_{args[0].model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
               f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
               f"_warmup{args[2].warmup_steps}"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


def get_exp_name_layout_multi(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score
    if args[0].use_margin_ranking_loss == 0:
        exp_name = f"{exp_mode}_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}"
    else:
        exp_name = f"{exp_mode}_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_mr-loss{args[0].margin}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}"
    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name

def get_exp_name_layout_multi_alltrain(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score
    if args[0].use_margin_ranking_loss == 0:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}"
    else:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_mr-loss{args[0].margin}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}"
    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name

def get_exp_name_layout_multi_alltrain_translation(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score
    if args[0].use_margin_ranking_loss == 0:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}_trans"
    else:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_mr-loss{args[0].margin}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}_trans"
    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name

def get_exp_name_layout_multi_alltrain_wordnet(args, is_print=True):
    ## declare experimental name
    if args[1].exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score
    if args[0].use_margin_ranking_loss == 0:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}_pure"
    else:
        exp_name = f"{exp_mode}_alltrain_multi_{args[0].model_name_or_path.split('/')[-1]}_{args[0].text_model_name_or_path.split('/')[-1]}_lr{args[2].learning_rate}" \
                   f"_bs{args[2].per_gpu_train_batch_size}*{args[2].gradient_accumulation_steps}" \
                   f"_warmup{args[2].warmup_steps}_cross-type{args[0].cross_attn_type}_mr-loss{args[0].margin}_augmentation_useWordNet{args[1].use_wordnet}_usePooler{args[0].use_pooler}_pure"
    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(os.path.join(path, "train.log")):
        os.remove(os.path.join(path, "train.log"))

    if os.path.exists(os.path.join(path, "error.log")):
        os.remove(os.path.join(path, "error.log"))

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# def get_dataset_cache_name(model_args, data_args, dataset_name, exp_mode):
#     return os.path.join(
#             data_args.data_dir,
#             "cached_{}_{}_exp-mode{}_no-oversample_cleaned".format(
#                 list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
#                 dataset_name,
#                 exp_mode
#             ),
#         )

def get_dataset_cache_name_layout_alltrain(model_args, data_args, dataset_name, exp_mode):
    return os.path.join(
            data_args.data_dir,
            "cached_{}_{}_{}_exp-mode{}_layout_alltrain".format(
                list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                model_args.text_model_name_or_path.split('/')[-1],
                dataset_name,
                exp_mode
            ),
        )


def get_dataset_cache_name(model_args, data_args, dataset_name, exp_mode):
    if data_args.use_wordnet == 0:
        return os.path.join(
                data_args.data_dir,
                "cached_{}_{}_{}_exp-mode{}_no-oversample_cleaned_augmentation".format(
                    list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                    model_args.text_model_name_or_path.split('/')[-1],
                    dataset_name,
                    exp_mode
                ),
            )
    else:
        return os.path.join(
            data_args.data_dir,
            "cached_{}_{}_{}_exp-mode{}_no-oversample_cleaned_augmentation_wordnet".format(
                list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                model_args.text_model_name_or_path.split('/')[-1],
                dataset_name,
                exp_mode
            ),
        )


def get_dataset_cache_name_alltrain(model_args, data_args, dataset_name, exp_mode):
    if data_args.use_wordnet == 0:
        return os.path.join(
                data_args.data_dir,
                "cached_{}_{}_{}_exp-mode{}_augmentation_alltrain".format(
                    list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                    model_args.text_model_name_or_path.split('/')[-1],
                    dataset_name,
                    exp_mode
                ),
            )
    else:
        return os.path.join(
            data_args.data_dir,
            "cached_{}_{}_{}_exp-mode{}_augmentation_alltrain_wordnet".format(
                list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                model_args.text_model_name_or_path.split('/')[-1],
                dataset_name,
                exp_mode
            ),
        )

def get_dataset_cache_name_alltrain_pure(model_args, data_args, dataset_name, exp_mode):
    if data_args.use_wordnet == 0:
        return os.path.join(
                data_args.data_dir,
                "cached_{}_{}_{}_exp-mode{}_pure_alltrain".format(
                    list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                    model_args.text_model_name_or_path.split('/')[-1],
                    dataset_name,
                    exp_mode
                ),
            )
    else:
        return os.path.join(
            data_args.data_dir,
            "cached_{}_{}_{}_exp-mode{}_pure_alltrain_wordnet".format(
                list(filter(None, model_args.model_name_or_path.split("/"))).pop(),
                model_args.text_model_name_or_path.split('/')[-1],
                dataset_name,
                exp_mode
            ),
        )

def save_checkpoint(args, state, filename='model_checkpoint.pth.tar', is_best=False, save_best_only=False):
    if save_best_only == True:
        if is_best:
            torch.save(state, os.path.join(args.output_dir, f'model_best.pth.tar'))
    else:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.output_dir, f'model_best.pth.tar'))