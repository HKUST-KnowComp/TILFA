import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

import os.path
from sklearn.utils import shuffle
import torch
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader,TensorDataset,random_split,SubsetRandomSampler, ConcatDataset
from torchvision import transforms
import torch.nn.init
import pandas as pd
from torch import nn
import numpy as np
import torch.optim as optim
import torch.nn.init
from sklearn.metrics import roc_auc_score
import copy
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
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        ##################### train ##########################
        model.train()
        running_loss = 0.0
        running_corrects = 0

        for i, (_, image, labels) in enumerate(train_dataloaders):
            labels = labels.to(device)
            image = image.to(device)

            logits = model(image)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            preds = outputs.reshape(-1).round()
            running_loss += loss.item() * labels.size(0)
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

        with torch.no_grad():
            for i, (text_ids, image, labels) in enumerate(val_dataloaders):
                labels = labels.to(device)
                image = image.to(device)

                logits = model(image)
                loss = criterion(logits, labels)
                outputs = torch.sigmoid(logits)

                preds = outputs.reshape(-1).round()

                running_loss += loss.item() * labels.size(0)
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
        if  best_f1 <= epoch_f1:
            best_f1 = epoch_f1
            best_acc = epoch_acc
            best_loss = epoch_loss
            best_precision = epoch_precision
            best_recall = epoch_recall
            best_epoch_num = epoch

            is_best_epoch = True
            predict_df = pd.DataFrame({"ids":predicted_text_ids, "gold_labels":gold_labels, "predicted_labels":predicted_labels, "probabilities": predicted_probs})
            predict_df.to_csv(os.path.join(args.exp_dir, f"results.csv"), index=False)

            checkpoint_name = os.path.join(args.exp_dir, f'model_epoch_{epoch+1}.pth.tar')
            save_checkpoint_nofold(args, {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_f1': best_f1,
                'optimizer': optimizer.state_dict(),
            }, filename=checkpoint_name, is_best=is_best_epoch, save_best_only=True)

        print('val loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}'.format(epoch_loss, epoch_acc, epoch_f1, epoch_precision, epoch_recall))
        print('best loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, epoch {}'.format(best_loss, best_acc, best_f1, best_precision, best_recall, best_epoch_num+1))
        print(classification_report(gold_labels, predicted_labels, digits=4))

    return best_epoch_num


def predict(model, test_dataloaders, criterion, best_epoch_num):
    print("\n######## test ########")

    checkpoint = torch.load(os.path.join(args.exp_dir, f'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    model.eval()
    running_loss = 0.0
    running_corrects = 0
    predicted_labels = []
    predicted_probs = []
    predicted_text_ids = []
    gold_labels = []

    with torch.no_grad():
        for i, (text_ids, image, labels) in enumerate(test_dataloaders):
            labels = labels.to(device)
            image = image.to(device)

            logits = model(image)
            loss = criterion(logits, labels)
            outputs = torch.sigmoid(logits)

            preds = outputs.reshape(-1).round()

            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels.reshape(-1))

            predicted_text_ids += list(text_ids)
            predicted_labels += preds.detach().cpu().tolist()
            predicted_probs += outputs.reshape(-1).detach().cpu().tolist()
            gold_labels += labels.reshape(-1).detach().cpu().tolist()

    epoch_loss = running_loss / len(test_dataset)
    # epoch_acc = running_corrects.double() / len(val_dataset)

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
    predict_df.to_csv(os.path.join(args.exp_dir, f"test_best-epoch_{best_epoch_num + 1}_results.csv"), index=False)

    print(
        'test loss: {:.4f}, acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
            epoch_loss, epoch_acc,
            epoch_f1,
            epoch_precision,
            epoch_recall, macro_f1, auc_score))


    print(classification_report(gold_labels, predicted_labels, digits=4))


if __name__ == '__main__':

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.RandomResizedCrop((224, 224)),
        # transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    args = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # create experiment dirs
    exp_name = get_exp_name_nofold_alltrain(args)
    old_path = args.exp_dir
    args.exp_dir = f"{old_path}/{exp_name}"
    make_dir(args.exp_dir)
    sys.stdout = Logger(os.path.join(args.exp_dir, "train.log"), sys.stdout)
    sys.stderr = Logger(os.path.join(args.exp_dir, "error.log"), sys.stderr)

    # initial model and optimizer
    # binary classification
    if args.img_model == 0:
        init_model = ImageModelResNet50(out_dim=1, freeze_model=args.freeze_model)
    elif args.img_model == 1:
        init_model = ImageModelResNet101(out_dim=1, freeze_model=args.freeze_model)
    else:
        init_model = ImageModelVGG16(out_dim=1, freeze_model=args.freeze_model)
    criterion = nn.BCEWithLogitsLoss()

    # results
    f1_list = []
    precision_list = []
    recall_list = []
    acc_list = []

    for dataset_name in ['gun_control', 'abortion']:
        print(f"\n##################### {dataset_name} ##########################\n")

        args.exp_dir = f"{old_path}/{exp_name}/{dataset_name}"
        make_dir(args.exp_dir)

        df = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_train.csv'), index_col=0)
        df = shuffle(df, random_state=args.seed)

        df_test = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_test.csv'), index_col=0)
        test_annotation = df_test.reset_index()
        test_dataset = ImageDataset(args, annotation=test_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name), transform=val_transform)
        test_dataloaders = DataLoader(test_dataset, collate_fn=collate_fn, batch_size=args.batch_size)


        df_dev = pd.read_csv(os.path.join(args.data_dir, dataset_name + '_dev.csv'), index_col=0)
        val_annotation = df_dev.reset_index()
        val_dataset = ImageDataset(args, annotation=val_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name), transform=val_transform)
        val_dataloaders = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=args.batch_size)
        train_annotation = df.reset_index()
        train_dataset = ImageDataset(args, annotation=train_annotation, root_dir=os.path.join(args.data_dir, 'images/' + dataset_name), transform=train_transform)
        train_dataloaders = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=args.batch_size)


        model = copy.deepcopy(init_model)
        model.to(device)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        best_epoch_num = train_model_binary_classification(model, train_dataloaders, val_dataloaders, criterion, optimizer, args.num_epochs)
