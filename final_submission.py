import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import torch

def decode_persuasiveness(label):
    if label == 1:
        label = "yes"
    else:
        label = "no"
    return label


def decode_stance(label):
    if label == 0:
        label = "oppose"
    else:
        label = "support"
    return label

def concat_datasets(file_name_gun, file_name_abortion, file_name_final):
    gun_output = pd.read_csv(f"./output/{file_name_gun}", index_col=0).reset_index()
    abortion_output = pd.read_csv(f"./output/{file_name_abortion}", index_col=0).reset_index()
    if "gold_labels" in gun_output.columns:
        gun_output = gun_output.drop('gold_labels', axis=1)
    if "gold_labels" in abortion_output.columns:
        abortion_output = abortion_output.drop('gold_labels', axis=1)
    output_list = [gun_output, abortion_output]
    concat_output = pd.concat(output_list, ignore_index=True)
    concat_output = concat_output.drop('probabilities', axis=1)

    # if "TaskA" in file_name_gun:
    label_name = "stance"
    concat_output.rename(columns={'predicted_labels': label_name}, inplace=True)
    for idx in range(len(concat_output)):
        concat_output.loc[idx, 'stance'] = decode_stance(concat_output.loc[idx, 'stance'])

    # else:
    #     label_name = "persuasiveness"
    #     concat_output.rename(columns={'predicted_labels': label_name}, inplace=True)
    #     for idx in range(len(concat_output)):
    #         concat_output.loc[idx, 'persuasiveness'] = decode_persuasiveness(concat_output.loc[idx, 'persuasiveness'])

    concat_output.to_csv(f"./final/{file_name_final}", index=False)

def post_process_datasets(file_name, file_name_final):
    output = pd.read_csv(f"./output/{file_name}", index_col=0).reset_index()
    if "gold_labels" in output.columns:
        output = output.drop('gold_labels', axis=1)
    output = output.drop('probabilities', axis=1)

    label_name = "stance"
    output.rename(columns={'predicted_labels': label_name}, inplace=True)
    for idx in range(len(output)):
        output.loc[idx, 'stance'] = decode_stance(output.loc[idx, 'stance'])

    output.to_csv(f"./final/{file_name_final}", index=False)

def ensemble_two(file1_name, file2_name, file_name_ensemble):
    output1 = pd.read_csv(f"./output/{file1_name}", index_col=0).reset_index()
    output2 = pd.read_csv(f"./output/{file2_name}", index_col=0).reset_index()
    assert len(output1) == len(output2)
    out_label_ids = []
    preds = []
    for idx in range(len(output1)):
        assert output1.loc[idx, 'tweet_id'] == output2.loc[idx, 'tweet_id']
        probabilities1 = output1.loc[idx, 'probabilities']
        probabilities2 = output2.loc[idx, 'probabilities']
        probabilities = (probabilities1 + probabilities2) / 2
        predicted_labels = probabilities.round()
        output1.loc[idx, 'probabilities'] = probabilities
        output1.loc[idx, 'predicted_labels'] = predicted_labels
        out_label_ids.append(output1.loc[idx, 'gold_labels'])
        preds.append(predicted_labels)

    epoch_metrics = classification_report(out_label_ids, preds, output_dict=True, digits=4)
    epoch_f1 = epoch_metrics["1.0"]['f1-score']
    epoch_precision = epoch_metrics["1.0"]['precision']
    epoch_recall = epoch_metrics["1.0"]['recall']
    epoch_acc = epoch_metrics["accuracy"]
    macro_f1 = (epoch_metrics["1.0"]['f1-score'] + epoch_metrics["0.0"]['f1-score']) / 2
    auc_score = roc_auc_score(out_label_ids, preds)

    print(
        'test acc: {:.4f}, f1: {:.4f}, precision: {:.4f}, recall: {:.4f}, macro_f1: {:.4f}, auc_score: {:.4f}'.format(
            epoch_acc,
            epoch_f1,
            epoch_precision,
            epoch_recall, macro_f1, auc_score))
    print(classification_report(out_label_ids, preds, digits=4))

    # output1.to_csv(f"./output/{file_name_ensemble}", index=False)

if __name__ == '__main__':
    # file_name_gun = "1top_stance_alltrain_multi_layoutlmv3-base_deberta-v3-large_lr1e-05_bs8*2_warmup0_cross-type-1_augmentation_useWordNet1_usePooler0_gun.csv"
    # file_name_abortion = "1top_stance_alltrain_multi_layoutlmv3-base_deberta-v3-large_lr1e-05_bs8*1_warmup0_cross-type-1_augmentation_useWordNet1_usePooler1_abortion.csv"
    # file_name_final = "final_top_score_87.79.csv"
    # concat_datasets(file_name_gun, file_name_abortion, file_name_final)

    file_name = "stance_alltrain_multi_layoutlmv3-base_bert-large-uncased_lr1e-05_bs8*2_warmup0_cross-type4_augmentation_useWordNet1_usePooler1_gun.csv"
    file_name_final = file_name
    post_process_datasets(file_name, file_name_final)

    # file1_name = "KnowComp.Resnet50Deberta.TaskB.1_abortion4800_dev.csv"
    # file2_name = "KnowComp.Resnet50Deberta.TaskB.1_abortion4800_dev.csv"
    # file_name_ensemble = ""
    # ensemble_two(file1_name, file2_name, file_name_ensemble)

