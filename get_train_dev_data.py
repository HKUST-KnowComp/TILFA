"""
=========================================================
Title: ImageArg Shared Task Code - Dataset Downloader
---------------------------------------------------------
Warning: The downloaded dataset should only be used for
participating in ImageArg Shared Task. Any other use is
explicitly prohibited. Any participants are not allowed
to redistribute the dataset per Twitter Developer Policy:
https://developer.twitter.com/en/developer-terms/policy.
---------------------------------------------------------
Notice: This code is managed by ImageArg Shared Task
(https://imagearg.github.io/). Parts of the code are
adopted from https://github.com/markowanga/stweet.
---------------------------------------------------------
Data: 2023-05-15
=========================================================
"""

from tqdm import tqdm
import pandas as pd
import stweet as st
import requests
import argparse
import json
import os


def try_tweet_by_id_scrap(tweet_id):
    id_task = st.TweetsByIdTask(tweet_id)
    output_print = st.PrintRawOutput()
    output_collector = st.CollectorRawOutput()
    st.TweetsByIdRunner(tweets_by_id_task=id_task,
                        raw_data_outputs=[output_print, output_collector]).run()
    return output_collector


def run(topic="abortion"):
    if not os.path.exists(args.data_dir):
        os.mkdir(args.data_dir)
    if not os.path.exists(os.path.join(args.data_dir, "images")):
        os.mkdir(os.path.join(args.data_dir, "images"))
    if not os.path.exists(os.path.join(args.data_dir, "images", f"{topic}")):
        os.mkdir(os.path.join(args.data_dir, "images", f"{topic}"))

    text_exist_ids = []
    df_train_exist = pd.DataFrame()
    df_dev_exist = pd.DataFrame()
    if os.path.exists(os.path.join(f"{args.data_dir}", f"{topic}_train.csv")):
        df_train_exist = pd.read_csv(os.path.join(f"{args.data_dir}", f"{topic}_train.csv"))
        text_exist_ids += df_train_exist["tweet_id"].tolist()
    if os.path.exists(os.path.join(f"{args.data_dir}", f"{topic}_train.csv")):
        df_dev_exist = pd.read_csv(os.path.join(f"{args.data_dir}", f"{topic}_dev.csv"))
        text_exist_ids += df_dev_exist["tweet_id"].tolist()

    # if not os.path.exists(args.meta_data):
    #     raise "No meta data found! Please download meta data..."
    # with open(args.meta_data) as f:
    #     lines = json.load(f)

    headers = {'Accept': 'application/json'}
    lines = requests.get(args.meta_data, headers=headers).json()
    lines = [line for line in lines if line["topic"] == topic]

    tweetid_list = []
    tweeturl_list = []
    tweettext_list = []
    stance_list = []
    persuasiveness_list = []
    split_list = []
    for item in tqdm(lines):
        tweetid = item['tweet_id']
        stance = item["stance"]
        persuasiveness = item["persuasiveness"]
        tweeturl = item["tweet_url"]
        split = item["split"]

        # save time in case the images are downloaded
        image_exist_path = os.path.join(args.data_dir, "images", f"{topic}", f"{tweetid}.jpg")
        if os.path.exists(image_exist_path) and (tweetid in text_exist_ids):
            continue

        try:
            output_collector = try_tweet_by_id_scrap(item['tweet_id'])
            jsonline = json.loads(output_collector.get_raw_list()[0].to_json_line())
            text = jsonline['raw_value']['legacy']['full_text']

            img_url = jsonline['raw_value']['legacy']['entities']['media'][0]['media_url_https']
            img_data = requests.get(img_url, timeout=10).content
            with open(os.path.join(args.data_dir, "images", f"{topic}", f"{tweetid}.jpg"), 'wb') as handler:
                handler.write(img_data)

            tweettext_list.append(text)
            tweetid_list.append(tweetid)
            tweeturl_list.append(tweeturl)
            stance_list.append(stance)
            persuasiveness_list.append(persuasiveness)
            split_list.append(split)

            df = pd.DataFrame()
            df['tweet_id'] = tweetid_list
            df["tweet_url"] = tweeturl_list
            df['tweet_text'] = tweettext_list
            df['stance'] = stance_list
            df["persuasiveness"] = persuasiveness_list
            df["split"] = split_list

            df_train = df[df["split"] == "train"]
            df_dev = df[df["split"] == "dev"]

            if len(df_train_exist) > 0:
                df_train = pd.concat([df_train_exist, df_train])
            if len(df_dev_exist) > 0:
                df_dev = pd.concat([df_dev_exist, df_dev])

            df_train.to_csv(os.path.join(f"{args.data_dir}", f"{topic}_train.csv"), index=False)
            df_dev.to_csv(os.path.join(f"{args.data_dir}", f"{topic}_dev.csv"), index=False)

        except Exception as e:
            print("Skip Error", e)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Persuasiveness')
    parser.add_argument('--meta-data', default='https://people.cs.pitt.edu/~zhexiong/data/meta_data.json',
                        help='meta data path')
    parser.add_argument('--data-dir', default='./data', help='path to save data')
    args = parser.parse_args()
    for topic in ["gun_control", "abortion"]:
        run(topic=topic)
    print("Done!")
