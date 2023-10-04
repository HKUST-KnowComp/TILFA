
import json
import pandas as pd
import copy

origin_file_path = "../abortion_train_for_pure_wordnet.csv"
input_file_path = "./final/abortion_train_pure_wordnet.json"
new_file_name = '../abortion_train_pure_wordnet.csv'

with open(input_file_path) as fin:
    wordnet_data = [json.loads(line) for line in fin]
origin_df = pd.read_csv(origin_file_path)

for line in wordnet_data:
    synonym_list = line["synonym_list"]
    if synonym_list == []:
        continue
    id = line["id"]
    tweet_id = line["tweet_id"]
    if tweet_id != str(origin_df.loc[id, "tweet_id"]):
        print(tweet_id)
        print(str(origin_df.loc[id, "tweet_id"]))
    token_list = line["token_list"]
    instance = line["instance"]
    for synonym in synonym_list:
        new_token_list = [synonym if i == instance else i for i in token_list]
        new_tweet_text = " ".join(new_token_list)
        tweet_line = copy.deepcopy(origin_df.loc[id])
        tweet_line["tweet_text"] = new_tweet_text
        origin_df = pd.concat([origin_df, pd.DataFrame(dict(tweet_line), index=[0])], axis=0, ignore_index=True)

origin_df.to_csv(new_file_name,index=False)
