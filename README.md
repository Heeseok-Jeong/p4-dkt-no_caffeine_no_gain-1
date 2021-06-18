## [목차]

* [\[Dialogue State Tracking 소개\]](#dialogue-state-tracking-소개)
- [\[Deep Knowledge Tracing 소개]\](#deep-knowledge-tracing-소개)
- [[Installation]](#installation)
  * [Dependencies](#dependencies)
- [[Usage]](#usage)
  * [Dataset](#dataset)
  * [Train](#train)
  * [Inference](#inference)
  * [Arguments](#arguments)
- [[File Structure]](#file-structure)
  * [LSTM](#lstm)
  * [LSTMATTN](#lstmattn)
  * [BERT](#bert)
  * [LGBM](#lgbm)
  * [SAINT](#saint)
  * [LastQuery](#lastquery)
  * [TABNET](#tabnet)
- [[Input CSV File]](#input-csv-file)
- [[Feature]](#feature)
- [[Contributors]](#contributors)
- [[Collaborative Works]](#collaborative-works)
  * [📝 Notion](#notion)
- [[Reference]](#reference)
  * [Papers](#papers)
  * [Dataset](#dataset-1)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

<br>
<br>

## [Deep Knowledge Tracing 소개]

**DKT**는 **Deep Knowledge Tracing**의 약자로 우리의 "지식 상태"를 추적하는 딥러닝 방법론입니다.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122540925-10143980-d064-11eb-8afc-ccdb1e76114c.png' height='250px '/>
</div>
<br/>


대회에서는 학생 개개인의 이해도를 가리키는 지식 상태를 예측하는 일보다는, 주어진 문제를 맞출지 틀릴지 예측하는 것에 집중합니다.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541124-42be3200-d064-11eb-8e60-0f7d82a15df9.png' height='250px '/>
</div>
<br/>

<br>

## [Installation]

### Dependencies

- torch
- pandas
- sklearn
- pycaret
- tqdm
- wandb
- easydict
- pytorch-tabnet

```bash
pip install -r requirements.txt
```

<br>
<br>

## [Usage]

### Dataset

학습에 필요한 데이터를 만들기 위해 두 개의 `.py` 파일을 순차적으로 실행해야 합니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python make_elapsed.py
$ p4-dkt-no_caffeine_no_gain# python make_fixed_data.py
```

### Train

모델을 학습하기 위해서는 `train.py` 를 실행시킵니다.

아래 Arguments 에 있는 argument 중 필요한 argumet 를 바꿔 사용하면 됩니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python train.py
```

총 7가지의 모델을 선택할 수 있습니다.

- **TABNET**
- **LASTQUERY**
- **SAINT**
- **LGBM**
- **BERT**
- **LSTMATTN**
- **LSTM**

### Inference

학습된 모델로 추론하기 위해서는 `inference.py` 를 실행시킵니다.

필요한 argument 는 `—-model_name` 과 `—-model_epoch` 입니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python inference.py --model_name "학습한 모델 폴더 이름" --model_epoch "사용하고픈 모델의 epoch"
```

### Arguments

train 과 inference 에서 필요한 argument 입니다.

```python
# Basic
--model: model type (default:'lstm')
--scheduler: scheduler type (default:'plateau')
--device: device to use (defualt:'cpu')
--data_dir: data directory (default:'/opt/ml/input/data/train_dataset')
--asset_dir: asset directory (default:'asset/')
--train_file_name: train file name (default:'add_FE_fixed_train.csv')
--valid_file_name: validation file name (default:'add_FE_fixed_valid.csv')
--test_file_name: test file name (default:'add_FE_fixed_test.csv')
--model_dir: model directory (default:'models/')
--num_workers: number of workers (default:1)
--output_dir: output directory (default:'output/')
--output_file: output file name (default:'output')
--model_name: model folder name (default:'')
--model_epoch: model epoch to use (default:1)

# Hyperparameters
--seed: random state (default:42)
--optimizer: optimizer type (default:'adamW')
--max_seq_len: max sequence length (default:20)
--hidden_dim: hidden dimension size (default:64)
--n_layers: number of layers (default:2)
--n_epochs: number of epochs (default:20)
--batch_size: batch size (default:64)
--lr: learning rate (default:1e-4)
--clip_grad: clip grad (default:10)
--patience: for early stopping (default:5)
--drop_out: drop out rate (default:0.2)
--dim_div: hidden dimension dividor in model to prevent too be large scale (default:3)

# Transformer
--n_heads: number of heads (default:2)
--is_decoder: use transformer decoder (default:True)

# TabNet
--tabnet_pretrain: Using TabNet pretrain (default:False)
--use_test_to_train: to training includes test data (default:False)
--tabnet_scheduler: TabNet scheduler (default:'steplr')
--tabnet_optimizer: TabNet optimizer (default:'adam')
--tabnet_lr: TabNet learning rate (default:2e-2)
--tabnet_batchsize: TabNet batchsize (default:16384)
--tabnet_n_step: TabNet n step(not log step) (default:5)
--tabnet_gamma: TabNet gamma (default:1.7)
--tabnet_mask_type: TabNet mask type (default:'saprsemax')
--tabnet_virtual_batchsize: TabNet virtual batchsize (default:256)
--tabnet_pretraining_ratio: TabNet pretraining ratio (default:0.8)

# Sliding Window
--window: Using Sliding Window augmentation (default:False)
--shuffle: shuffle Sliding Window (default:False)
--stride: Sliding Window stride (default:20)
--shuffle_n: Shuffle times (default:1)

# T-Fixup
--Tfixup: Using T-Fixup (default:False)
--layer_norm: T-Fixup with layer norm (default:False)

# Pseudo Labeling
--use_pseudo: Using Pseudo Labeling (default:False)
--pseudo_label_file: file path for Pseudo Labeling (default:'')

# log
--log_steps: print log per n steps (default:50)

# wandb
--use_wandb: if you want to use wandb (default:True)
```

## [File Structure]

전체적인 File Structure 입니다.

```
code
├── README.md
├── .gitignore
├── args.py
├── make_custom_data
│   ├── make_elapsed.py - time 관련 feature 생성
│   ├── make_fixed_data.py - user 정답률 기반으로 valid 생성
│   └── make_original_fixed_data.py - shuffle해서 valid 생성
│
├── dkt
│   ├── criterion.py
│   ├── dataloader.py
│   ├── metric.py
│   ├── model.py
│   ├── optimizer.py
│   ├── scheduler.py
│   ├── trainer.py
│   └── utils.py
├── ensemble.py
├── inference.py
├── requirements.txt - dependencies
└── train.py

```

<br>
<br>

### LSTM

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541256-66817800-d064-11eb-92ea-7fc9ae8b0cce.png' height='250px '/>
</div>
<br/>

- sequence data를 다루기 위한 LSTM 모델입니다.
- **구현**

    ```
    model.py
    ├── class LSTM
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### LSTMATTN

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541497-a6485f80-d064-11eb-8c97-61b7b9d25954.png' height='250px '/>
</div>
<br/>


- LSTM 모델에 Self-Attention을 추가한 모델입니다.
- **구현**

    ```
    model.py
    ├── class LSTMATTN
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### BERT

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541560-b6f8d580-d064-11eb-94ac-73c0acafc796.png' height='250px '/>
</div>
<br/>


- `Huggingface` 에서 BERT 구조를 가져와서 사용합니다. 다만, pre-trained 모델이 아니기 때문에 Transformer-encoder 와 같습니다.
- 현재 모델에서는 bert_config 의 is_decoder 를 True 로 주어 Transformer-decoder 로 사용하고 있습니다.
- **구현**

    ```
    model.py
    ├── class Bert
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    ├── args.is_decoder(default : True)
    └── args.hidden_dim(default : 64)
    ****
    ```

<br>
<br>

### LGBM

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541790-03dcac00-d065-11eb-9464-3f4c890bccda.png' height='250px '/>
</div>
<br/>


- tabular data에서 좋은 성능을 보이는 Machine Learning 모델입니다.

<br>
<br>

### SAINT

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541828-0d661400-d065-11eb-9028-d7b1a0d6adce.png' height='250px '/>
</div>
<br/>



- Kaggle Riiid AIEd Challenge 2020의 [Host가 제시한 solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/193250) 입니다.
- Transformer와 비슷한 구조의 모델로 Encoder와 Decoder를 가지고 있습니다.
- 인코더는 feature 임베딩 스트림에 self-attention 레이어를 적용하고 디코더에서 self-attention 레이어와 인코더-디코더 attention 레이어를 응답 임베딩과 인코더의 출력 스트림에 번갈아 적용하는 구조입니다.
- **Paper Review** : [[Saint 모델 분석]](https://www.notion.so/Saint-507d13692825492ba05128f4548c2da7)
- **구현**

    ```
    model.py
    ├── class Saint
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    └── args.hidden_dim(default : 64)
    ```

<br>
<br>

### LastQuery

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541905-1eaf2080-d065-11eb-995e-3e7fa03907d3.png' height='250px '/>
</div>
<br/>


- Kaggle Riiid AIEd Challenge 2020의 [1st place solution](https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/218318)입니다.
- transformer encoder의 입력으로 sequence의 마지막 query만 사용하여 시간복잡도를 줄이고, encoder의 output을 LSTM에 넣어 학습하는 방식의 모델입니다.
- **Paper Review :**  [[Last Query Transformer RNN for knowledge tracing 리뷰]](https://www.notion.so/Last-Query-Transformer-RNN-for-knowledge-tracing-e0930bfff69b4d2e852de4cbd8e44678)
- **구현**

    ```
    model.py
    ├── class LastQuery
    │   ├── init()
    └── └── forward() : return predicts

    args.py
    ├── args.max_seq_len(default : 20)
    ├── args.n_layers(default : 2)
    ├── args.n_heads(default : 2)
    ├── args.hidden_dim(default : 64)
    └── args.Tfixup(default : False)
    ```

<br>

<br>

### TABNET

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122541964-2b337900-d065-11eb-9cad-f8d9c86c5f4e.png' height='250px '/>
</div>
<br/>


- tabular data에서 ML모델보다 더 우수한 성능을 보이는 Deep-learning model입니다.
- data에서 Sparse instance-wise feature selection을 사용하여 자체적으로 중요한 feature 선별해낸 후 학습하는 방식을 사용하며, feature 선별시 non-linear한 processing을 사용하여 learning capacity를 향상시킵니다.
- Sequential한 multi-step architecture를  가지고있으며, feature masking으로 Unsupervised 학습도 가능합니다.
- **Paper Review : [[Tabnet 논문 리뷰]](https://www.notion.so/Tabnet-298eca48c26a4486a4df8e1586cba2ed)**

- **구현**

    ```
    model.py
    ├── class TabNet
    │   ├── TabNetPreTrainer
    │   ├── TabNetClassifier
    │   ├── get_scheduler()
    │   ├── get_optimizer()
    └── └── forward() : return models

    trainer.py
    ├── tabnet_run(args, train_data, valid_data)
    ├── get_tabnet_model(args)
    └── tabnet_inference(args, test_data)

    train.py
    └── tabnet_run()

    args.py
    ├── args.tabnet_pretrain(default : False)
    ├── args.use_test_to_train(default : False)
    ****├── args.tabnet_scheduler(default:'steplr')
    ****├── args.tabnet_optimizer(default:'adam')
    ****├── args.tabnet_lr(default:2e-2)
    ****├── args.tabnet_batchsize(default:16384)
    ****├── args.tabnet_n_step(default:5)
    ****├── args.tabnet_gamma(default:1.7)
    ├── args.tabnet_mask_type(default:'saprsemax')
    ├── args.tabnet_virtual_batchsize(default:256)
    └── args.tabnet_pretraining_ratio(default:0.8)
    ****
    ```

<br>

## [Input CSV File]

데이터는 아래와 같은 형태이며, 한 행은 한 사용자가 한 문항을 풀었을 때의 정보와 그 문항을 맞췄는지에 대한 정보가 담겨져 있습니다. 데이터는 모두 Timestamp 기준으로 정렬되어 있습니다.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542042-3d151c00-d065-11eb-8278-be19e177037e.png' height='250px '/>
</div>
<br/>


- `userID` 사용자의 고유번호입니다. 총 7,442명의 고유 사용자가 있으며, train/test셋은 이 `userID`를 기준으로 90/10의 비율로 나누어졌습니다.
- `assessmentItemID` 문항의 고유번호입니다. 총 9,454개의 고유 문항이 있습니다.
- `testId` 시험지의 고유번호입니다. 문항과 시험지의 관계는 아래 그림을 참고하여 이해하시면 됩니다. 총 1,537개의 고유한 시험지가 있습니다.

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542102-49997480-d065-11eb-9957-c84bc1ab77d5.png' height='250px '/>
</div>
<br/>

- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터이며 0은 사용자가 해당 문항을 틀린 것, 1은 사용자가 해당 문항을 맞춘 것입니다.
- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.
- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다. 태그 자체의 정보는 비식별화 되어있지만, 문항을 군집화하는데 사용할 수 있습니다. 912개의 고유 태그가 존재합니다.

## [Feature]

```python
elapsed: 유저가 문제를 푸는데에 소요한 시간

time_bin: 문제를 푼 시간대(아침, 점심, 저녁, 새벽)

classification: 대분류(학년)

paperNum: 시험지 번호

problemNum: 문제 번호

user_total_acc: 유저의 총 정답률

test_acc: 각 시험지의 평균 정답률

assessment_acc: 각 문제의 평균 정답률

tag_acc: 각 태그의 평균 정답률

total_used_time: 유저가 하나의 시험지를 다 푸는데에 소요한 시간

past_correct: 유저별 과거 맞춘 문제의 수

past_content_count: 유저-문제별 과거에 동일 문제를 만난 횟수

correct_per_hour: 시간(hours)별 정답률

same_tag: 동일 태그를 연속으로 풀었는지 유무(T/F)

cont_tag: 연속으로 푼 동일 태그 개수(0~)

etc...
```

<br>
<br>


## [Contributors]

- **정희석** ([Heeseok-Jeong](https://github.com/Heeseok-Jeong))
- **이애나** ([Anna-Lee](https://github.com/ceanna93))
- **이창우** ([changwoomon](https://github.com/changwoomon))
- **안유진** ([dkswndms4782](https://github.com/dkswndms4782))
- **선재우** ([JAEWOOSUN](https://github.com/JAEWOOSUN))

<br>
<br>

## [Collaborative Works]

**Gitflow 브랜치 전략**

→ 92개의 Commits, 26개의 Pull Requests

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542173-59b15400-d065-11eb-9c49-56c091e4fc9f.gif' height='250px '/>
</div>
<br/>


**Github issues & projects 로 일정 관리**

→ 28개의 Issues

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542239-6e8de780-d065-11eb-9f6f-821372f4bbbf.gif' height='250px '/>
</div>
<br/>


→ Modeling Project 에서 관리

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542265-751c5f00-d065-11eb-8465-97d2a6fbfebf.gif' height='250px '/>
</div>
<br/>


**Notion 실험노트로 실험 공유**

→ 39개의 실험노트

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542297-81082100-d065-11eb-96de-713440c9544b.gif' height='250px '/>
</div>
<br/>


Notion 제출기록으로 제출 내역 공유
→ 155개의 제출기록

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542323-86fe0200-d065-11eb-8949-8aa146157d73.gif' height='250px '/>
</div>


### 📝 Notion


[DKT-10조-No_Caffeine_No_Gain](https://www.notion.so/DKT-10-No_Caffeine_No_Gain-dcc1e3823ec849578ab5ae0bcf117145)


<br>
<br>

## [Reference]

### Papers

- [Deep Knowledge Tracing (Piech et al., arXiv 2015)](https://arxiv.org/pdf/1506.05908.pdf)
- [Last Query Transformer RNN for Knowledge Tracing (Jeon, S., arXiv 2021)](https://arxiv.org/abs/2102.05038)
- [Towards an Appropriate Query, Key, and Value Computation for Knowledge Tracing (Choi et al., arXiv 2021)](https://arxiv.org/abs/2002.07033)
- [How to Fine-Tune BERT for Text Classification? (Sun et al., arXiv 2020)](https://arxiv.org/pdf/1905.05583.pdf)
- [Improving Transformer Optimization Through Better Initialization (Huang et al., ICML 2020)](https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf)

### Dataset
- i-Scream edu Dataset
<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122542423-9ed58600-d065-11eb-9c4e-8c8efa83de80.png' height='250px '/>
</div>
