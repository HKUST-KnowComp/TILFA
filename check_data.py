import pandas as pd
import os

train = pd.read_csv(os.path.join('./data/gun_control_train.csv'), index_col=0).reset_index()
dev = pd.read_csv(os.path.join('./data/gun_control_dev.csv'), index_col=0).reset_index()

train_ids = [train.loc[idx, "tweet_id"] for idx in range(len(train))]
dev_ids = [dev.loc[idx, "tweet_id"] for idx in range(len(dev))]

num = 0
for id in dev_ids:
    if num in train_ids:
        num += 1

print(num)
