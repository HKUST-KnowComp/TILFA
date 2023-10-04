import os


os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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
from dataloader import TextDataset

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

def predict(model, test_dataloaders, criterion):
    print("\n################## test ########################")

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    predicted_labels = []
    predicted_probs = []
    predicted_text_ids = []
    gold_labels = []

    with torch.no_grad():
        for i, (text_ids, input_ids, attention_masks, labels) in enumerate(test_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            preds = outputs.reshape(-1).round()

            running_loss += loss.item() * input_ids.size(0)
            running_corrects += torch.sum(preds == labels.reshape(-1))

            predicted_text_ids += list(text_ids)
            predicted_labels += preds.detach().cpu().tolist()
            predicted_probs += outputs.reshape(-1).detach().cpu().tolist()
            gold_labels += labels.reshape(-1).detach().cpu().tolist()

    epoch_loss = running_loss / len(test_dataset)

    epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
    epoch_f1 = epoch_metrics["1.0"]['f1-score']
    epoch_precision = epoch_metrics["1.0"]['precision']
    epoch_recall = epoch_metrics["1.0"]['recall']
    epoch_acc = epoch_metrics["accuracy"]

    macro_f1 = (epoch_metrics["1.0"]['f1-score'] + epoch_metrics["0.0"]['f1-score']) / 2
    auc_score = roc_auc_score(gold_labels, predicted_labels)

    print(
        'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(epoch_loss, epoch_acc,
                                                                                              epoch_f1,
                                                                                              epoch_precision,
                                                                                              epoch_recall, macro_f1, auc_score))
    print(classification_report(gold_labels, predicted_labels, digits=4))



args = get_argparser().parse_args()

sys.stdout = Logger("./train.log", sys.stdout)
sys.stderr = Logger("./error.log", sys.stderr)

df_test = pd.read_csv("./data/gun_control_dev.csv", index_col=0)
test_annotation = df_test.reset_index()
test_dataset = TextDataset(args, annotation=test_annotation, root_dir=os.path.join(args.data_dir, 'images/gun_control'))
test_dataloaders = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(args.seed)

model1 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=True)
checkpoint1 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_39.pth.tar'))
model1.load_state_dict(checkpoint1['state_dict'])
model1.to(device)
criterion = nn.BCEWithLogitsLoss()

# print_model(model1)

predict(model1, test_dataloaders, criterion)

print("###########################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=False)
checkpoint2 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_21.pth.tar'))
model2.load_state_dict(checkpoint2['state_dict'])
model2.to(device)
criterion = nn.BCEWithLogitsLoss()

# print_model(model2)

predict(model2, test_dataloaders, criterion)

# print("###########################################################")
#
# if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in zip(model1.bert_model.parameters(), model2.bert_model.parameters())]) == False :
#     print("model1.bert == model2.bert")
# else:
#     print("model1.bert != model2.bert")
#
# if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in zip(model1.fc.parameters(), model2.fc.parameters())]) == False :
#     print("model1.fc == model2.fc")
# else:
#     print("model1.fc != model2.fc")

print("###########################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=False)
checkpoint2 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_20.pth.tar'))
model2.load_state_dict(checkpoint2['state_dict'])
model2.to(device)
criterion = nn.BCEWithLogitsLoss()

predict(model2, test_dataloaders, criterion)

print("###########################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=False)
checkpoint2 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_19.pth.tar'))
model2.load_state_dict(checkpoint2['state_dict'])
model2.to(device)
criterion = nn.BCEWithLogitsLoss()

predict(model2, test_dataloaders, criterion)

print("###########################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=False)
checkpoint2 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_17.pth.tar'))
model2.load_state_dict(checkpoint2['state_dict'])
model2.to(device)
criterion = nn.BCEWithLogitsLoss()

predict(model2, test_dataloaders, criterion)

print("###########################################################")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = TextModel(text_model_name="bert-large-uncased", out_dim=1, freeze_model=False)
checkpoint2 = torch.load(os.path.join("./experiments/stance_text_resnet50_bert-large-uncased_freeze-False_lr1e-05_bs16_nofold/gun_control", f'model_epoch_10.pth.tar'))
model2.load_state_dict(checkpoint2['state_dict'])
model2.to(device)
criterion = nn.BCEWithLogitsLoss()

predict(model2, test_dataloaders, criterion)