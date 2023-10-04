# coding=utf-8
import copy
import json
import logging
import os
import re
from multiprocessing import Pool
import sys
import torch
from lxml import html
from torch.utils.data import TensorDataset
from tqdm import tqdm
from transformers import DataProcessor
import preprocessor
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION, preprocessor.OPT.HASHTAG)
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import numpy as np


def get_text(node):
    textnodes = node.xpath(".//text()")
    s = "".join([text for text in textnodes])
    return re.sub(r"\s+", " ", s).strip()


def get_prop(node, name):
    title = node.get("title")
    props = title.split(";")
    for prop in props:
        (key, args) = prop.split(None, 1)
        args = args.strip('"')
        if key == name:
            return args
    return None


class DocExample(object):
    def __init__(self, guid, text_a, bbox=None, stance_label=None, persuasiveness_label=None):
        self.guid = guid
        self.text_a = text_a
        self.bbox = bbox
        self.stance_label = stance_label
        self.persuasiveness_label = persuasiveness_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def encode_persuasiveness(label):
    # if label >= args.persuasive_label_threshold:
    if label == "yes":
        label = "1"
    else:
        label = "0"
    return label


def encode_stance(label):
    if label == "oppose":
        label = "0"
    else:
        label = "1"
    return label


class LayoutProcessor(DataProcessor):
    """Processor for the data set."""

    def worker(self, idx):
        tweet_id = str(self.annotation.loc[idx, 'tweet_id'])
        stance_label = self.annotation.loc[idx, 'stance']
        persuasiveness_label = self.annotation.loc[idx, 'persuasiveness']
        text, bbox = self.read_hocr_file(tweet_id)
        return [text, bbox, stance_label, persuasiveness_label]

    def get_examples(self, data_dir, annotation, dataset_name, mode):
        annotation["tweet_text"] = annotation["tweet_text"].apply(lambda x: preprocessor.clean(x))
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.annotation = annotation
        # with open(os.path.join(data_dir, "labels", "{}.txt".format(mode))) as f:
        #     lines = f.readlines()
        examples = []
        with tqdm(range(len(annotation)), desc="Gettting {} examples".format(mode)) as t, Pool(24) as p:
            for example in p.imap(self.worker, range(len(annotation))):
                examples.append(example)
                t.update()
        return self._create_examples(examples, mode)

    # def _get_examples(self, data_dir, mode):
    #     with open(os.path.join(data_dir, "labels", "{}.txt".format(mode))) as f:
    #         lines = []
    #         for line in tqdm(f.readlines(), desc="Gettting {} examples".format(mode)):
    #             file, label = line.split()
    #             text, bbox = self.read_hocr_file(data_dir, file)
    #             lines.append([text, bbox, label])
    #     return self._create_examples(lines, mode)

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

    def _create_examples(self, lines, mode):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (mode, i)
            text = line[0]
            bbox = line[1]
            stance_label = line[2]
            persuasiveness_label = line[3]
            examples.append(
                DocExample(guid=guid, text_a=text, bbox=bbox, stance_label=stance_label, persuasiveness_label=persuasiveness_label)
            )
        return examples


class DocFeature(object):
    def __init__(self, input_ids, bboxes, attention_mask, token_type_ids, stance_label, persuasiveness_label):
        assert (
            0 <= all(bboxes) <= 1000
        ), "Error with input bbox ({}): the coordinate value is not between 0 and 1000".format(
            bboxes
        )
        self.input_ids = input_ids
        self.bboxes = bboxes
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.stance_label = stance_label
        self.persuasiveness_label = persuasiveness_label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def convert_examples_to_features(
    examples,
    tokenizer,
    max_length=512,
    label_list=None,
    pad_on_left=False,
    pad_token="[PAD]",
    pad_token_id=0,
    pad_token_segment_id=0,
    mask_padding_with_zero=True,
):

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens = []
        bboxes = []

        if len(example.text_a) == 0:
            bboxes.append([0, 0, 0, 0])
            tokens.append(pad_token)

        for token, bbox in zip(example.text_a, example.bbox):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                bboxes.append(bbox)
                tokens.append(sub_token)

        tokens = tokens[: max_length - 2]
        bboxes = bboxes[: max_length - 2]
        bboxes = [[0, 0, 0, 0]] + bboxes + [[1000, 1000, 1000, 1000]]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        token_type_ids = [0] * len(input_ids)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_length - len(input_ids)
        if pad_on_left:
            input_ids = ([pad_token_id] * padding_length) + input_ids
            bboxes = ([[0, 0, 0, 0]] * padding_length) + bboxes
            attention_mask = (
                [0 if mask_padding_with_zero else 1] * padding_length
            ) + attention_mask
            token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
        else:
            input_ids = input_ids + ([pad_token_id] * padding_length)
            bboxes = bboxes + ([[0, 0, 0, 0]] * padding_length)
            attention_mask = attention_mask + (
                [0 if mask_padding_with_zero else 1] * padding_length
            )
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids) == max_length, "Error with input length {} vs {}".format(
            len(input_ids), max_length
        )
        assert len(bboxes) == max_length, "Error with input length {} vs {}".format(
            len(bboxes), max_length
        )
        assert (
            len(attention_mask) == max_length
        ), "Error with input length {} vs {}".format(len(attention_mask), max_length)
        assert (
            len(token_type_ids) == max_length
        ), "Error with input length {} vs {}".format(len(token_type_ids), max_length)

        stance_label = label_map[example.stance_label]
        persuasiveness_label = label_map[example.persuasiveness_label]

        if ex_index < 2:
            print("*** Example ***")
            print(f"guid: {example.guid}")
            print("input_ids: {}".format(" ".join([str(x) for x in input_ids])))
            print("bboxes: {}".format(" ".join([str(x) for x in bboxes])))
            print("attention_mask: {}".format(" ".join([str(x) for x in attention_mask])))
            print("token_type_ids: {}".format(" ".join([str(x) for x in token_type_ids])))
            print("stance_label: {} (id = {})".format(example.stance_label, stance_label))
            print("persuasiveness_label: {} (id = {})".format(example.persuasiveness_label, persuasiveness_label))

        features.append(
            DocFeature(
                input_ids=input_ids,
                bboxes=bboxes,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                stance_label=stance_label,
                persuasiveness_label=persuasiveness_label
            )
        )
    return features


def load_and_cache_examples(args, tokenizer, annotation, dataset_name, mode="train"):
    if args.local_rank not in [-1, 0] and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = LayoutProcessor()
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}_exp-mode{}_no-oversample_cleaned".format(
            mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            dataset_name,
            args.exp_mode
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        print("Loading features from cached file {}".format(cached_features_file))
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at {}".format(args.data_dir))
        label_list = processor.get_labels()

        examples = processor.get_examples(args.data_dir, annotation, dataset_name, mode)
        features = convert_examples_to_features(
            examples,
            tokenizer,
            label_list=label_list,
            max_length=args.max_seq_length,
            pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_token=tokenizer.pad_token,
            pad_token_id=tokenizer.pad_token_id,
            pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
        )
        if args.local_rank in [-1, 0]:
            print("Saving features into cached file {}".format(cached_features_file))
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and mode == "train":
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_bboxes = torch.tensor([f.bboxes for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor(
        [f.attention_mask for f in features], dtype=torch.long
    )
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in features], dtype=torch.long
    )

    all_stance_labels = torch.tensor([f.stance_label for f in features], dtype=torch.long)
    all_persuasiveness_labels = torch.tensor([f.persuasiveness_label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_attention_mask, all_token_type_ids, all_stance_labels, all_persuasiveness_labels,
        all_bboxes
    )
    return dataset


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(os.path.join(path, "train.log")):
        os.remove(os.path.join(path, "train.log"))

    if os.path.exists(os.path.join(path, "error.log")):
        os.remove(os.path.join(path, "error.log"))


def get_exp_name_layout(args, is_print=True):
    ## declare experimental name
    if args.exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    exp_name = f"{exp_mode}_{args.model_name_or_path.split('/')[-1]}_lr{args.learning_rate}" \
               f"_bs{args.per_gpu_train_batch_size * args.gradient_accumulation_steps}" \
               f"_warmup{args.warmup_steps}_no-oversample_cleaned"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# if __name__ == "__main__":
#     import argparse
#     from transformers import BertTokenizerFast
#
#     parser = argparse.ArgumentParser()
#     args = parser.parse_args()
#     args.local_rank = -1
#     args.data_dir = "data"
#     args.model_name_or_path = "bert-base-uncased"
#     args.max_seq_length = 512
#     args.model_type = "bert"
#     args.overwrite_cache = True
#     tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     dataset = load_and_cache_examples(args, tokenizer, mode="test")
#     print(len(dataset))



