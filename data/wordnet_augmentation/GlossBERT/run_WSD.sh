root=/home/data/zwanggy/2023/gloss_bert

for source in wikipedia
do
  CUDA_VISIBLE_DEVICES=1 python run_classifier_WSD_sent.py \
  --eval_data_dir ../gloss_input/gun_control_train_pure_wordnet.csv \
  --output_dir ../gloss_score/gun_control_train_pure_wordnet \
  --bert_model ${root}/Sent_CLS_WS \
  --task_name WSD --do_test --do_lower_case --max_seq_length 512 \
  --eval_batch_size 128 --seed 42
done

for source in wikipedia
do
  CUDA_VISIBLE_DEVICES=1 python run_classifier_WSD_sent.py \
  --eval_data_dir ../gloss_input/abortion_train_pure_wordnet.csv \
  --output_dir ../gloss_score/abortion_train_pure_wordnet \
  --bert_model ${root}/Sent_CLS_WS \
  --task_name WSD --do_test --do_lower_case --max_seq_length 512 \
  --eval_batch_size 128 --seed 42
done
