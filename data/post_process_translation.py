
import pandas as pd

old_file_name = './abortion_dev.csv'
new_file_name = './abortion_dev_after_stance_augmentation.csv'

old_df = pd.read_csv(old_file_name)
new_df = pd.read_csv(new_file_name)

for index in new_df.index:
    origin_index = new_df.loc[index, 'Unnamed: 0']
    tweet_id = str(old_df.loc[origin_index, 'tweet_id'])
    new_df.loc[index, 'tweet_id'] = tweet_id

new_df = new_df.drop(['Unnamed: 0'], axis=1)
new_df = new_df.sample(frac=1)
print(new_df.stance.value_counts().to_dict())
new_df.to_csv('./abortion_dev_after_stance_augmentation_new.csv',index=False)


