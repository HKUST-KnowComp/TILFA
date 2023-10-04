import requests

from utils.AuthV3Util import addAuthParams

# 您的应用ID
APP_KEY = '5330683f5fd526bf'
# 您的应用密钥
APP_SECRET = 'qS7HosfgmdpZfIO13mn7A0hKcePcbRv1'

import json

import time

def createRequest(q, lang_from, lang_to):
    '''
    note: 将下列变量替换为需要请求的参数
    '''
    # q = '待翻译文本'
    # lang_from = '源语言语种'
    # lang_to = '目标语言语种'
    # vocab_id = '您的用户词表ID'

    data = {'q': q, 'from': lang_from, 'to': lang_to}

    addAuthParams(APP_KEY, APP_SECRET, data)

    header = {'Content-Type': 'application/x-www-form-urlencoded'}
    time.sleep(5)
    res = doCall('https://openapi.youdao.com/api', header, data, 'post')
    try:
        translation = json.loads(res.content)["translation"]
        return " ".join(translation)
    except:
        print("fail!")
        return None


def doCall(url, header, params, method):
    if 'get' == method:
        return requests.get(url, params)
    elif 'post' == method:
        return requests.post(url, params, header)

# 网易有道智云翻译服务api调用demo
# api接口: https://openapi.youdao.com/api


# Import python library
import pandas as pd
import nltk
import preprocessor
preprocessor.set_options(preprocessor.OPT.URL, preprocessor.OPT.EMOJI, preprocessor.OPT.MENTION, preprocessor.OPT.HASHTAG)


# Read file
file_name = '../../abortion_dev.csv'
# Read file using pandas
df = pd.read_csv(file_name)
df["tweet_text"] = df["tweet_text"].apply(lambda x: preprocessor.clean(x))

# Function for augmenting data using langauge translation
# Could not found free service for langauge translation, Use paid service like Azure, Google translator etc

from textblob.translate import NotTranslated
import random
sr = random.SystemRandom()
import requests

language = ["es", "de", "fr", "ar", "te", "hi", "ja", "fa", "sq", "bg", "nl", "gu", "ig", "kk", "mt", "ps"]



def data_augmentation(tweet, language, aug_range=1):
    translate_tweet_text_df = pd.DataFrame()
    tweet_text = tweet["tweet_text"]
    if hasattr(tweet_text, "decode"):
        tweet_text = tweet_text.decode("utf-8")

    for j in range(0,aug_range) :
        new_tweet_df = tweet.copy()
        # text = TextBlob(tweet_text)
        try:
            lang = sr.choice(language)
            tweet_text = createRequest(tweet_text, "en", lang)
            if tweet_text is None:
                continue
            tweet_text = createRequest(tweet_text, lang, "en")
            if tweet_text is None:
                continue
            new_tweet_df["tweet_text"] = tweet_text
            translate_tweet_text_df = translate_tweet_text_df.append(new_tweet_df)
        except NotTranslated:
            pass

    return translate_tweet_text_df

# Dictionary for intent count
# Intent is column name
persuasiveness = df.persuasiveness.value_counts().to_dict()

# Get max intent count to match other minority classes through data augmentation
import operator
max_intent_count = max(persuasiveness.items(), key=operator.itemgetter(1))[1]

# Loop to interate all messages
import numpy as np
import math
import tqdm

newdf = pd.DataFrame()
for persuasiveness_i, count in persuasiveness.items():
    count_diff = max_intent_count - count  ## Difference to fill
    multiplication_count = math.ceil(
        (count_diff) / count)  ## Multiplying a minority classes for multiplication_count times
    if (multiplication_count):
        old_tweet_text_df = pd.DataFrame()
        new_tweet_text_df = pd.DataFrame()
        orig_augment_df = df[df["persuasiveness"] == persuasiveness_i]
        for idx in tqdm.tqdm(orig_augment_df.index):
            # Extracting existing minority class batch
            tweet = orig_augment_df.loc[idx]
            tweet_text = tweet["tweet_text"]
            old_tweet_text_df = old_tweet_text_df.append(tweet)

            # Creating new augmented batch from existing minority class
            translate_tweet_text_df = data_augmentation(tweet.copy(), language, multiplication_count)

            new_tweet_text_df = new_tweet_text_df.append(translate_tweet_text_df)

        # Select random data points from augmented data
        new_tweet_text_df = new_tweet_text_df.take(np.random.permutation(len(new_tweet_text_df))[:count_diff])

        # Merge existing and augmented data points
        newdf = newdf.append([old_tweet_text_df, new_tweet_text_df])
    else:
        newdf = newdf.append(df[df["persuasiveness"] == persuasiveness_i])

# Print count of all new data points
newdf.persuasiveness.value_counts()

# Save newdf back to file
newdf.to_csv('../../abortion_dev_after_persuasiveness_augmentation.csv')
