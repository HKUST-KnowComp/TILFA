import os


os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from torchvision import transforms
import os.path
from sklearn.utils import shuffle
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, TensorDataset, random_split, SubsetRandomSampler, ConcatDataset
import torch.nn.init
import pandas as pd
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.init
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import copy
import time
from utils import *
from models import *
from dataloader import *

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def print_model(model):
    # 查看网络的结构
    print(model)

    # 打印模型参数
    for param in model.parameters():
        print(param)

    # 打印模型名称与shape
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())

    print("\n\n")
    print(get_parameter_number(model))


def predict(model, test_dataloaders, criterion, output_name):
    print("\n######## test ########")

    model.eval()
    predicted_labels = []
    predicted_probs = []
    predicted_text_ids = []

    with torch.no_grad():
        for i, (text_ids, input_ids, attention_masks) in enumerate(test_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)

            logits = model(input_ids, attention_masks)
            outputs = torch.sigmoid(logits)

            preds = outputs.reshape(-1).round()

            predicted_text_ids += list(text_ids)
            predicted_labels += preds.detach().cpu().tolist()
            predicted_probs += outputs.reshape(-1).detach().cpu().tolist()

    predict_df = pd.DataFrame(
        {"tweet_id": predicted_text_ids, "predicted_labels": predicted_labels,
         "probabilities": predicted_probs})
    predict_df.to_csv(f"./output/{output_name}", index=False)

    # model.eval()
    # running_loss = 0.0
    # running_corrects = 0
    # predicted_labels = []
    # predicted_probs = []
    # predicted_text_ids = []
    # gold_labels = []
    #
    # with torch.no_grad():
    #     for i, (text_ids, input_ids, attention_masks, labels) in enumerate(test_dataloaders):
    #         input_ids = input_ids.to(device)
    #         attention_masks = attention_masks.to(device)
    #         labels = labels.to(device)
    #
    #         logits = model(input_ids, attention_masks)
    #         loss = criterion(logits, labels)
    #         outputs = torch.sigmoid(logits)
    #
    #         preds = outputs.reshape(-1).round()
    #
    #         running_loss += loss.item() * input_ids.size(0)
    #         running_corrects += torch.sum(preds == labels.reshape(-1))
    #
    #         predicted_text_ids += list(text_ids)
    #         predicted_labels += preds.detach().cpu().tolist()
    #         predicted_probs += outputs.reshape(-1).detach().cpu().tolist()
    #         gold_labels += labels.reshape(-1).detach().cpu().tolist()
    #
    # epoch_loss = running_loss / len(test_dataset)
    # # epoch_acc = running_corrects.double() / len(val_dataset)
    #
    # epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
    # epoch_f1 = epoch_metrics["1.0"]['f1-score']
    # epoch_precision = epoch_metrics["1.0"]['precision']
    # epoch_recall = epoch_metrics["1.0"]['recall']
    # epoch_acc = epoch_metrics["accuracy"]
    #
    # macro_f1 = (epoch_metrics["1.0"]['f1-score'] + epoch_metrics["0.0"]['f1-score']) / 2
    # auc_score = roc_auc_score(gold_labels, predicted_labels)
    #
    # predict_df = pd.DataFrame(
    #     {"tweet_id": predicted_text_ids, "gold_labels": gold_labels, "predicted_labels": predicted_labels,
    #      "probabilities": predicted_probs})
    # predict_df.to_csv(f"./output/{output_name}", index=False)
    #
    # print(
    #     'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
    #         epoch_loss, epoch_acc,
    #         epoch_f1,
    #         epoch_precision,
    #         epoch_recall, macro_f1, auc_score))
    #
    # print(classification_report(gold_labels, predicted_labels, digits=4))



if __name__ == '__main__':

    dataset_name = 'abortion'
    model_name = 'stance_alltrain_text_resnet50_deberta-v3-large_lr1e-05_bs8_augmentation_wordnet0_pooler0'
    output_name = 'stance_alltrain_text_resnet50_deberta-v3-large_lr1e-05_bs8_augmentation_wordnet0_pooler0_abortion.csv'

    args = get_argparser().parse_args()

    df_test = pd.read_csv(f"./data/{dataset_name}_test.csv", index_col=0)
    # df_test = pd.read_csv(f"./data/{dataset_name}_dev.csv", index_col=0)
    test_annotation = df_test.reset_index()
    test_dataset = TextTestDataset(args, annotation=test_annotation, root_dir=os.path.join(args.data_dir, f'images/{dataset_name}'))
    # test_dataset = TextDataset(args, annotation=test_annotation, root_dir=os.path.join(args.data_dir, f'images/{dataset_name}'))

    test_dataloaders = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    model1 = TextModel(text_model_name=args.text_model_name, out_dim=1, freeze_model=args.freeze_model)
    criterion = nn.BCEWithLogitsLoss()
    checkpoint1 = torch.load(os.path.join(f"/home/data/zwanggy/2023/image_arg_experiments/{model_name}/{dataset_name}", f'model_best.pth.tar'))
    # checkpoint1 = torch.load(os.path.join(f"./experiments/{model_name}/{dataset_name}", f'model_best.pth.tar'))
    model1.load_state_dict(checkpoint1['state_dict'])
    model1.to(device)

    # print_model(model1)

    predict(model1, test_dataloaders, criterion, output_name)
