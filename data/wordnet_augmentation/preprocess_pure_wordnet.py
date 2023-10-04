import pandas as pd

old_file_name = '../gun_control_train_after_persuasiveness_augmentation_new.csv'
output_file_name = '../gun_control_train_for_pure_wordnet.csv'

old_df = pd.read_csv(old_file_name)
print(len(old_df))

old_df.drop_duplicates(subset=['tweet_id'], keep='first', inplace=True)
print(len(old_df))

old_df.to_csv(output_file_name,index=False)
