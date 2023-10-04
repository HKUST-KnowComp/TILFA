import torch
import ast
import argparse
import shutil
import os
import sys
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from transformers import BertTokenizer, AutoTokenizer
import torchvision.models as models

def encode_image_content_type(label):
    if label == "symbolic":
        label = 0
    elif label == "anecdote":
        label = 1
    elif label == "slogan":
        label = 2
    elif label == "scene":
        label = 3
    elif label == "statistics":
        label = 4
    elif label == "testimony":
        label = 5
    else:
        label = 6  # other
    return label


def encode_persuasive_mode(labels):
    logos = 0.0
    pathos = 0.0
    ethos = 0.0
    if type(labels) == str and len(labels) > 0:
        # labels = labels.replace("\'","\"")
        # labels = json.loads(labels)
        labels = ast.literal_eval(labels)
        if labels["logos"] == "yes":
            logos = 1.0
        if labels["pathos"] == "yes":
            pathos = 1.0
        if labels["ethos"] == "yes":
            ethos = 1.0
    return [logos, pathos, ethos]


def encode_persuasiveness(label, args):
    # if label >= args.persuasive_label_threshold:
    if label == "yes":
        label = 1.0
    else:
        label = 0.0
    return label


def encode_stance(label):
    if label == "oppose":
        label = 0.0
    else:
        label = 1.0
    return label


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_longest_length(sentences):
    return max([len(s) for s in sentences])


def bert_tokenizer(sentences, args):
    if args.text_model_name == 'bert-large-uncased':
        tokenizer = BertTokenizer.from_pretrained(args.text_model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.text_model_name)

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


def reset_weights(model):
    # if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
    #     m.reset_parameters()
    for name, module in model.named_modules():
        if hasattr(module, 'reset_parameters'):
            print('Resetting ', name)
            module.reset_parameters()


def save_checkpoint(args, state, fold, filename='model_checkpoint.pth.tar', is_best=False, save_best_only=False):
    if save_best_only == True:
        if is_best:
            torch.save(state, os.path.join(args.exp_dir, f'fold_{fold}_model_best.pth.tar'))
    else:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.exp_dir, f'fold_{fold}_model_best.pth.tar'))


def save_checkpoint_nofold(args, state, filename='model_checkpoint.pth.tar', is_best=False, save_best_only=False):
    if save_best_only == True:
        if is_best:
            torch.save(state, os.path.join(args.exp_dir, f'model_best.pth.tar'))
    else:
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(args.exp_dir, f'model_best.pth.tar'))


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    if os.path.exists(os.path.join(path, "train.log")):
        os.remove(os.path.join(path, "train.log"))

    if os.path.exists(os.path.join(path, "error.log")):
        os.remove(os.path.join(path, "error.log"))


def plot_image(args, img_id_list, prob_list, option="", is_show=False):
    fig = plt.figure(figsize=(10, 2.5))
    columns = 5
    rows = 1
    for i in range(columns * rows):
        img_id = img_id_list[i]
        img = mpimg.imread(f'./data/images/{img_id}.jpg')
        img_resized = cv2.resize(img, (4000, 4000))
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img_resized)
        plt.axis('off')
        plt.title(f"id {img_id}\nprob {round(prob_list[i], 3)}", fontsize=8)
    fig.text(.5, .1, f'{option} text_id: {args.text_id}', ha='center')
    if is_show:
        plt.show()
    plt.savefig(os.path.join(args.exp_dir, f"demo_{args.text_id}.jpg"))


def get_exp_name(args, is_print=True):
    ## declare experimental name
    if args.exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    if args.data_mode == 0:
        data_mode = "text"
    elif args.data_mode == 1:
        data_mode = "image"
    else:
        data_mode = "multimodality"

    if args.img_model == 0:
        img_model = "resnet50"
    elif args.img_model == 1:
        img_model = "resnet101"
    else:
        img_model = "vgg16"

    if args.exp_mode == 1:
        exp_name = f"{exp_mode}_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_persuasive_threshold_{args.persuasive_label_threshold}_lr{args.lr}_agumentation"
    else:
        exp_name = f"{exp_mode}_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_lr{args.lr}_agumentation"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


def get_exp_name_nofold(args, is_print=True):
    ## declare experimental name
    if args.exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    if args.data_mode == 0:
        data_mode = "text"
    elif args.data_mode == 1:
        data_mode = "image"
    else:
        data_mode = "multimodality"

    if args.img_model == 0:
        img_model = "resnet50"
    elif args.img_model == 1:
        img_model = "resnet101"
    else:
        img_model = "vgg16"

    if args.exp_mode == 1:
        exp_name = f"{exp_mode}_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_persuasive_threshold_{args.persuasive_label_threshold}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}"
    else:
        exp_name = f"{exp_mode}_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


def get_exp_name_nofold_alltrain(args, is_print=True):
    ## declare experimental name
    if args.exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    if args.data_mode == 0:
        data_mode = "text"
    elif args.data_mode == 1:
        data_mode = "image"
    else:
        data_mode = "multimodality"

    if args.img_model == 0:
        img_model = "resnet50"
    elif args.img_model == 1:
        img_model = "resnet101"
    else:
        img_model = "vgg16"

    if args.exp_mode == 1:
        exp_name = f"{exp_mode}_alltrain_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_persuasive_threshold_{args.persuasive_label_threshold}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}"
    else:
        exp_name = f"{exp_mode}_alltrain_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


def get_exp_name_nofold_alltrain_wordnet(args, is_print=True):
    ## declare experimental name
    if args.exp_mode == 0:
        exp_mode = "stance"
    else:  # 1
        exp_mode = "persuasiveness"  # persuasiveness score

    if args.data_mode == 0:
        data_mode = "text"
    elif args.data_mode == 1:
        data_mode = "image"
    else:
        data_mode = "multimodality"

    if args.img_model == 0:
        img_model = "resnet50"
    elif args.img_model == 1:
        img_model = "resnet101"
    else:
        img_model = "vgg16"

    if args.exp_mode == 1:
        exp_name = f"{exp_mode}_alltrain_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_persuasive_threshold_{args.persuasive_label_threshold}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}_pure"
    else:
        exp_name = f"{exp_mode}_alltrain_{data_mode}_{img_model}_{args.text_model_name.split('/')[-1]}_lr{args.lr}_bs{args.batch_size}_augmentation_wordnet{args.use_wordnet}_pooler{args.use_pooler}_pure"

    if is_print:
        print(f"Experiment {exp_name}")
    return exp_name


def get_argparser():
    parser = argparse.ArgumentParser(description='Persuasiveness')
    parser.add_argument('--data-dir', default='./data', help='path to data')
    parser.add_argument('--exp-dir', default='./experiments/debug', help='path save experimental results')
    parser.add_argument('--exp-mode', default=0, choices=[0, 1], type=int, help='0: stance; 1: persuasive')
    parser.add_argument('--num-epochs', default=10, type=int, help='number of running epochs')
    parser.add_argument('--data-mode', default=2, choices=[0, 1, 2], type=int, help='0: text; 1: image; 2: image+text')
    parser.add_argument('--seed', default=22, type=int, help='random seed number')
    parser.add_argument('--batch-size', default=16, type=int, help='number of samples per batch')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--persuasive-label-threshold', default=0.6, type=float,
                        help='threshold to categorize persuasive labels')
    parser.add_argument('--kfold', default=5, help='number of fold validation')
    parser.add_argument('--img-model', default=0, choices=[0, 1, 2], type=int,
                        help='0: Resnet50; 1: Resnet101; 2: VGG16')
    # parser.add_argument('--save-checkpoint', default=0, choices=[0, 1], type=int,
    #                     help='0:do not save checkpoints; 1:save checkpoints')
    parser.add_argument('--text-model-name', default='bert-large-uncased', type=str, help='name of the text model')
    parser.add_argument("--freeze-model", action='store_true', help='freeze model?')
    parser.add_argument("--use-wordnet", default=0, choices=[0, 1], help='use wordnet?', type=int)
    parser.add_argument("--use-pooler", default=0, choices=[0, 1], help='use pooler?', type=int)
    return parser


class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
