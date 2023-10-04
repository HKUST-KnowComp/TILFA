python3 main_text_alltrain.py --num-epochs=25 --batch-size=4 --exp-mode=0 --data-mode=0 --lr=5e-6 --img-model=0 --text-model-name=microsoft/deberta-v3-large

python3 main_image_alltrain.py --num-epochs=25 --batch-size=16 --exp-mode=0 --data-mode=1 --lr=1e-6 --img-model=0 --text-model-name=bert-large-uncased

python3 main_multimodality_alltrain.py --num-epochs=25 --batch-size=4 --exp-mode=0 --data-mode=2 --lr=1e-5 --img-model=1 --text-model-name=microsoft/deberta-v3-large --use-pooler=0 --use-wordnet=1


python3 main_layoutlmv3_alltrain.py --data_dir=./data --output_dir=/home/data/zwanggy/2023/image_arg_experiments --do_train --do_eval --do_predict --model_name_or_path=microsoft/layoutlmv3-base --visual_embed --num_train_epochs=25 --input_size=224 --learning_rate=1e-5 --per_gpu_train_batch_size=8 --per_gpu_eval_batch_size=8 --seed=22 --gradient_accumulation_steps=1 --text_model_name_or_path=microsoft/deberta-v3-large

python3 main_multimodality_layoutlmv3_alltrain.py --data_dir=./data --output_dir=/home/data/zwanggy/2023/image_arg_experiments --do_train --do_eval --model_name_or_path=microsoft/layoutlmv3-base --visual_embed --num_train_epochs=25 --input_size=224 --learning_rate=1e-5 --per_gpu_train_batch_size=4 --per_gpu_eval_batch_size=4 --seed=22  --gradient_accumulation_steps=1 --text_model_name_or_path=microsoft/deberta-v3-large --exp_mode=0  --use_wordnet=1 --use_pooler=1 --cross_attn_type=4


