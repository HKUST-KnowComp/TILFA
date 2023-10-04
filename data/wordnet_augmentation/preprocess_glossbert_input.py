import pandas as pd
import spacy
import random
from nltk.corpus import wordnet as wn
import json
nlp = spacy.load('en')

old_file_name = '../gun_control_train_for_pure_wordnet.csv'
output_file_name = './gun_control_train_pure_wordnet.json'

old_df = pd.read_csv(old_file_name)
tweet_texts = []
for idx in range(len(old_df)):
    tweet_text = old_df.loc[idx, 'tweet_text']
    tweet_id = str(old_df.loc[idx, 'tweet_id'])
    try:
        doc = nlp(tweet_text)
    except:
        print("1")
        print(tweet_text)
        continue
    token_list = [str(token) for token in doc]
    instance_list =[]
    for pos, token in enumerate(doc):
        if token.pos_ == "NOUN":
            instance_list.append({"instance": str(token), "start_idx": pos, "end_idx": pos, "head_idx": pos})
    try:
        instance_idx = random.sample(range(len(instance_list)), 1)[0]
        instance = instance_list[instance_idx]
        synset_list = [i.name() for i in wn.synsets(instance["instance"])]
    except:
        print("2")
        print(tweet_text)
        continue
    tweet_texts.append(dict({"event_text": tweet_text, "token_list": token_list, "synset_list": synset_list, "source": "wn", "id": idx, "tweet_id": tweet_id}, **instance))
with open(output_file_name, 'w') as f:
    for dict in tweet_texts:
        f.write(json.dumps(dict)+"\n")
