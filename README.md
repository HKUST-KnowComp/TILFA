# TILFA: A Unified Framework for Text, Image, and Layout Fusion in Argument Mining

This repository is the official implementation of TILFA: A Unified Framework for Text, Image, and Layout Fusion in Argument Mining.

The paper is accepted to the Proceedings of the 10th Workshop on Argument Mining 2023.

## Abstract

A main goal of Argument Mining (AM) is to analyze an author's stance. 
Unlike previous AM datasets focusing only on text, 
the shared task at the 10th Workshop on Argument Mining 
introduces a [dataset](https://aclanthology.org/2022.argmining-1.1.pdf) including both text and images. 
Importantly, these images contain both visual elements 
and optical characters. Our new framework, **TILFA** 
(A Unified Framework for **T**ext, **I**mage, and **L**ayout **F**usion 
in **A**rgument Mining), is designed to handle this mixed data. It 
excels at not only understanding text but also detecting optical 
characters and recognizing layout details in images.
Our model significantly outperforms existing baselines, 
earning our team, KnowComp, the [**1st**](https://imagearg.github.io/) place in the leaderboard
of Argumentative Stance Classification subtask in this shared task.


## An Overview of Our Method

 ![](./method_figure.png)

## Requirements

Python version is 3.7

requirements:
```
apex==0.9.10dev
boto3==1.28.10
botocore==1.31.10
datasets==2.3.2
detectron2==0.6+cu111
imbalanced_learn==0.10.1
imblearn==0.0
inflect==7.0.0
lxml==4.9.2
matplotlib==3.5.3
nltk==3.8.1
numpy==1.21.6
opencv_python==4.8.0.74
pandas==1.1.5
Pillow==9.5.0
Pillow==10.0.1
preprocessor==1.1.3
ptvsd==4.3.2
pytesseract==0.3.10
Requests==2.31.0
scikit_learn==1.0.2
spacy==2.2.1
stweet==2.1.1
tensorflow==2.14.0
textblob==0.17.1
timm==0.4.12
torch==1.10.0+cu111
torchvision==0.11.1+cu111
tqdm==4.65.0
transformers==4.12.5
tweet_preprocessor==0.6.0
websocket_client==1.6.3
```

You can install all requirements with the command
```
pip install -r requirements.txt
```

## Run TILFA Framework

### Step 1: Training
- training examples can be found in ./run.sh
#### pure text
```angular2html
python3 main_text_alltrain.py 
--exp-dir=YOUR_EXPERIMENT_PATH
--num-epochs=25 
--batch-size=16 
--exp-mode=0 
--data-mode=0 
--lr=5e-6 
--img-model=0 
--text-model-name=microsoft/deberta-v3-large
```

#### pure image
```angular2html
python3 main_image_alltrain.py 
--exp-dir=YOUR_EXPERIMENT_PATH
--num-epochs=25 
--batch-size=16 
--exp-mode=0 
--data-mode=1 
--lr=1e-6 
--img-model=0 
--text-model-name=microsoft/deberta-v3-large
```

#### original multimodality
```angular2html
python3 main_multimodality_alltrain.py 
--exp-dir=YOUR_EXPERIMENT_PATH
--num-epochs=25 
--batch-size=16 
--exp-mode=0 
--data-mode=2 
--lr=1e-5 
--img-model=1 
--text-model-name=microsoft/deberta-v3-large 
--use-pooler=0 
--use-wordnet=1
```

#### pure layout
```angular2html
python3 main_layoutlmv3_alltrain.py 
--data_dir=./data 
--output_dir=YOUR_EXPERIMENT_PATH 
--do_train 
--do_eval 
--do_predict 
--model_name_or_path=microsoft/layoutlmv3-base 
--visual_embed 
--num_train_epochs=25 
--input_size=224 
--learning_rate=1e-5 
--per_gpu_train_batch_size=8 
--per_gpu_eval_batch_size=8 
--seed=22 
--gradient_accumulation_steps=1 
--text_model_name_or_path=microsoft/deberta-v3-large
```

#### layout multimodality
```angular2html
python3 main_multimodality_layoutlmv3_alltrain.py 
--data_dir=./data 
--output_dir=/home/data/zwanggy/2023/image_arg_experiments 
--do_train 
--do_eval 
--model_name_or_path=microsoft/layoutlmv3-base 
--visual_embed 
--num_train_epochs=25 
--input_size=224 
--learning_rate=1e-5 
--per_gpu_train_batch_size=4 
--per_gpu_eval_batch_size=4 
--seed=22  
--gradient_accumulation_steps=1 
--text_model_name_or_path=microsoft/deberta-v3-large 
--exp_mode=0  
--use_wordnet=1 
--use_pooler=0 
--cross_attn_type=-1
```

### Step 2: Predict
predict_test_origin_text.py is for pure text
predict_test_origin_image.py is for pure image
predict_test_origin_multi.py is for original multimodality
predict_test_layout.py is for pure layout
predict_test_layout_multi.py is for layout multimodality

You should change the model name in the code to the one you want to predict with.
Other parameters are consistent with the training part.

### Step 3: Post Process

You should change the file name in the code to the one you want to process.
```angular2html
python3 final_submission.py
```

### Step 4: Evaluate And Get The Score

If you want to get the score across topic:
```angular2html
python3 get_evaluation.py
-f=YOUR_FILE_PATH
```

If you want to get the score within topic:
```angular2html
python3 get_evaluation_within_topic.py
-f=YOUR_FILE_PATH
--topic=choose one in [gun_control, abortion]
```

## Others

### Address Data Imbalance
- code used to address data imbalance is in path ./data/TranslateDemo 
- a stands for abortion, g stands for gun_control
- s stands for stance, p stands for persuasiveness

```angular2html
cd data/TranslateDemo
python3 TranslateDemo_a_s.py
```

### Data Augmentation
- code used to do data augmentation is in path ./data/wordnet_augmentation

```angular2html
cd data/wordnet_augmentation
python3 preprocess_glossbert_input.py
python3 build_gloss_bert_input.py
cd GlossBERT
./run_WSD.sh
cd ..
python3 incorporate_score.py
```
