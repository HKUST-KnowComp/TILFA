"""
=========================================================
Title: ImageArg Shared Task Code - Evaluation
---------------------------------------------------------
Warning: The downloaded dataset should only be used for
participating in ImageArg Shared Task. Any other use is
explicitly prohibited. Any participants are not allowed
to redistribute the dataset per Twitter Developer Policy:
https://developer.twitter.com/en/developer-terms/policy.
---------------------------------------------------------
Notice: This code is managed by ImageArg Shared Task
(https://imagearg.github.io/).
---------------------------------------------------------
Data: 2023-08-03
=========================================================
"""

import pandas as pd
import argparse
import requests
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score


def get_gold_file(args):
    headers = {'Accept': 'application/json'}
    lines = requests.get(args.meta_data, headers=headers).json()
    df = pd.DataFrame(lines)
    return df


def get_prediction_file(args):
    if not os.path.exists(args.file_path):
        print(f"{args.file_path} not exist!")
        sys.exit()

    df = pd.read_csv(args.file_path)
    return df


def get_task_predictions(args):
    df_gold = get_gold_file(args)
    df_gold = df_gold[df_gold["topic"] == args.topic]
    df_pred = get_prediction_file(args)
    assert len(df_gold) == len(df_pred)
    df = pd.merge(df_pred, df_gold, on="tweet_id")
    df = df[df["tweet_id"] != 1321141824300306433]  # remove this tweet as it is not available online
    # if len(df) != 299:
    #     print(f"Error! You only has {len(df)} predictions!")
    #     sys.exit()

    # stance prediction
    if "stance_x" in df.columns:
        print(f"you are evaluating stance predictions:\n{args.file_path}.\ntopic: {args.topic}")
        pred_list = df["stance_x"].tolist()
        gold_list = df["stance_y"].tolist()
    # elif "persuasiveness_x" in df.columns:
    #     print(f"you are evaluating persuasiveness predictions:\n{args.file_path}.\n")
    #     pred_list = df["persuasiveness_x"].tolist()
    #     gold_list = df["persuasiveness_y"].tolist()
    else:
        print("please run 'check_submission_format.py' to check your submission format!")
        sys.exit()

    label_dict = {'support': 1, 'oppose': 0, 'yes': 1, 'no': 0}
    pred_list = [label_dict[x] for x in pred_list]
    gold_list = [label_dict[x] for x in gold_list]

    return pred_list, gold_list


def get_evaluation_report(pred, gold):
    auc_score = roc_auc_score(gold, pred)
    rp = classification_report(gold, pred, output_dict=False, digits=4)
    print(rp)
    print("auc: {:.4f}".format(auc_score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ImageArg shared Task')
    parser.add_argument('-f', '--file-path', default="", required=True,
                        help='path to submission file')
    parser.add_argument('--meta-data', default='https://people.cs.pitt.edu/~zhexiong/data/meta_data_test.json',
                        help='meta data path')
    parser.add_argument('--topic', default='gun_control', help='topic name')
    args = parser.parse_args()

    pred_list, gold_list = get_task_predictions(args)
    get_evaluation_report(pred_list, gold_list)
