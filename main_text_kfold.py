import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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


def train_model_binary_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, num_epochs=5):
    best_acc = 0.0
    best_loss = 0.0
    best_f1 = 0.0
    best_precision = 0.0
    best_recall = 0.0
    best_epoch_num = 0

    for epoch in range(num_epochs):
        print('Fold {} Epoch {}/{}'.format(fold + 1, epoch + 1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, input_ids, attention_masks, labels) in enumerate(train_dataloaders):
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            labels = labels.to(device)

            logits = model(input_ids, attention_masks)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            optimizer.zero_grad()

            # total_num = sum(p.numel() for p in model.parameters())
            # trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # print({'Total': total_num, 'Trainable': trainable_num})
            #
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")


            loss.backward()
            optimizer.step()

            # print("=============更新之后===========")
            # for name, parms in model.named_parameters():
            #     print('-->name:', name)
            #     print('-->para:', parms)
            #     print('-->grad_requirs:', parms.requires_grad)
            #     print('-->grad_value:', parms.grad)
            #     print("===")
            # print(optimizer)
            #
            # if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
            #         zip(init_model.bert_model.parameters(), model.bert_model.parameters())]) == False:
            #     print("model_new.bert == model_origin.bert")
            # if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
            #         zip(init_model.fc.parameters(), model.fc.parameters())]) == False:
            #     print("model_new.fc == model_origin.fc")

            preds = outputs.reshape(-1).round()
            running_loss += loss.item() * input_ids.size(0)
            running_corrects += torch.sum(preds == labels.reshape(-1))

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print('train loss: {:.4f}, acc: {:.4f}'.format(epoch_loss, epoch_acc))

        ##################### validation ##########################
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        predicted_labels = []
        predicted_probs = []
        predicted_text_ids = []
        gold_labels = []

        if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
                zip(init_model.bert_model.parameters(), model.bert_model.parameters())]) == False:
            print("model_new.bert == model_origin.bert")
        if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
                zip(init_model.fc.parameters(), model.fc.parameters())]) == False:
            print("model_new.fc == model_origin.fc")

        with torch.no_grad():
            for i, (text_ids, input_ids, attention_masks, labels) in enumerate(val_dataloaders):
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

        epoch_loss = running_loss / len(val_dataset)
        # epoch_acc = running_corrects.double() / len(val_dataset)

        epoch_metrics = classification_report(gold_labels, predicted_labels, output_dict=True, digits=4)
        epoch_f1 = epoch_metrics["1.0"]['f1-score']
        epoch_precision = epoch_metrics["1.0"]['precision']
        epoch_recall = epoch_metrics["1.0"]['recall']
        epoch_acc = epoch_metrics["accuracy"]

        is_best_epoch = False
        if best_f1 <= epoch_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch

            is_best_epoch = True
            predict_df = pd.DataFrame(
                {"ids": predicted_text_ids, "gold_labels": gold_labels, "predicted_labels": predicted_labels,
                 "probabilities": predicted_probs})
            predict_df.to_csv(os.path.join(args.exp_dir, f"fold_{fold}_results.csv"), index=False)

        checkpoint_name = os.path.join(args.exp_dir, f'fold_{fold}_model_epoch_{epoch + 1}.pth.tar')
        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_f1': best_f1,
            'optimizer': optimizer.state_dict(),
        }, fold=fold, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

        print(
            'val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch_loss, epoch_acc,
                                                                                                  epoch_f1,
                                                                                                  epoch_precision,
                                                                                                  epoch_recall))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, epoch{}'.format(best_loss,
                                                                                                              best_acc,
                                                                                                              best_f1,
                                                                                                              best_precision,
                                                                                                              best_recall,
                                                                                                              best_epoch_num + 1))
        print(classification_report(gold_labels, predicted_labels, digits=4))

    return best_f1, best_precision, best_recall, best_acc, best_epoch_num


def predict(init_model, test_dataloaders, criterion, best_fold, best_epoch_num):
    print("\n######## test ########")

    model = copy.deepcopy(init_model)

    checkpoint = torch.load(os.path.join(args.exp_dir, f'fold_{best_fold}_model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
            zip(init_model.bert_model.parameters(), model.bert_model.parameters())]) == False:
        print("model_new.bert == model_origin.bert")
    if any([p1.data.ne(p2.data).sum() > 0 for p1, p2 in
            zip(init_model.fc.parameters(), model.fc.parameters())]) == False:
        print("model_new.fc == model_origin.fc")

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

    predict_df = pd.DataFrame(
        {"ids": predicted_text_ids, "gold_labels": gold_labels, "predicted_labels": predicted_labels,
         "probabilities": predicted_probs})
    predict_df.to_csv(os.path.join(args.exp_dir, f"test_best-fold_{best_fold}_best-epoch_{best_epoch_num + 1}_results.csv"), index=False)


    print(
        'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(epoch_loss, epoch_acc,
                                                                                              epoch_f1,
                                                                                              epoch_precision,
                                                                                              epoch_recall, macro_f1, auc_score))
    print(classification_report(gold_labels, predicted_labels, digits=4))


if __name__ == '__main__':

    args = get_argparser().parse_args()
    print(args.text_model_name)
    print(args.freeze_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # create experiment dirs
    exp_name = get_exp_name(args)
    old_path = args.exp_dir
    args.exp_dir = f"{old_path}/{exp_name}"
    make_dir(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "error.log"), sys.stderr)

    # initial model and optimizer
    # binary classification
    init_model = TextModel(text_model_name=args.text_model_name, out_dim=1, freeze_model=args.freeze_model)
    criterion = nn.BCEWithLogitsLoss()

    # results
    f1_list = []
    precision_list = []
    recall_list = []
    acc_list = []


    kfold = KFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    for dataset_name in ['gun_control', 'abortion']:
        print(f"\n##################### {dataset_name} ##########################\n")

        args.exp_dir = f"{old_path}/{exp_name}/{dataset_name}"
        make_dir(args.exp_dir)

        df = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_train.csv'), index_col=0)
        df = shuffle(df, random_state=args.seed)

        df_test = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_dev.csv'), index_col=0)
        test_annotation = df_test.reset_index()
        test_dataset = TextDataset(args, annotation=test_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name))
        test_dataloaders = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

        best_f1 = 0.
        best_fold = 0
        best_epoch_num = 0

        for fold, (train_idx, val_idx) in enumerate(kfold.split(df)):
            print('Running fold {}...'.format(fold + 1))

            train_annotation = df.iloc[train_idx].reset_index()
            val_annotation = df.iloc[val_idx].reset_index()
            train_dataset = TextDataset(args, annotation=train_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name))
            val_dataset = TextDataset(args, annotation=val_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name))
            train_dataloaders = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
            val_dataloaders = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)

            init_model.to(device)
            model = copy.deepcopy(init_model)
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            fold_f1, fold_precision, fold_recall, fold_acc, epoch_num = train_model_binary_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, args.num_epochs)

            f1_list.append(fold_f1)
            precision_list.append(fold_precision)
            recall_list.append(fold_recall)
            acc_list.append(fold_acc)

            if best_f1 <= fold_f1:
                best_fold = fold
                best_epoch_num = epoch_num

        predict(init_model, test_dataloaders, criterion, best_fold, best_epoch_num)

        m_f1, m_precision, m_recall, m_acc = np.round(np.mean(f1_list), 4), np.round(np.mean(precision_list), 4), np.round(
            np.mean(recall_list), 4), np.round(np.mean(acc_list), 4)
        print(f"{exp_name} {args.kfold} fold validation...")
        print(f"{'f1' : <10}{'precision' : <10}{'recall' : <10}{'acc' : <10}")
        print(f"{m_f1 : <10}{m_precision : <10}{m_recall : <10}{m_acc : <10}")

        with open(os.path.join(args.exp_dir, "report.txt"), "w") as f:
            f.write(f"{exp_name} {args.kfold} fold validation...\n")
            f.write(f"{'f1' : <10}{'precision' : <10}{'recall' : <10}{'acc' : <10}\n")
            f.write(f"{m_f1 : <10}{m_precision : <10}{m_recall : <10}{m_acc : <10}\n")
