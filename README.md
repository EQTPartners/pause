# PAUSE: Positive and Annealed Unlabeled Sentence Embedding

This repo contains source code to reproduce the baseline results in the paper: [**PAUSE: Positive and Annealed Unlabeled Sentence Embedding**](https://aclanthology.org/2021.emnlp-main.791).

Sentence embedding refers to a set of effective and versatile techniques for converting raw text into numerical vector representations that can be used in a wide range of natural language processing (NLP) applications. The majority of these techniques are either supervised or unsupervised. Compared to the unsupervised methods, the supervised ones make less assumptions about optimization objectives and usually achieve better results. However, the training requires a large amount of labeled sentence pairs, which is not available in many industrial scenarios. To that end, we propose a generic and end-to-end approach -- PAUSE (Positive and Annealed Unlabeled Sentence Embedding), capable of learning high-quality sentence embeddings from a partially labeled dataset, which effectively learns sentence embeddings from PU datasets by jointly optimizing the supervised and PU loss. The main highlights of PAUSE include:
- good sentence embeddings can be learned from datasets with only a few positive labels;
- it can be trained in an end-to-end fashion;
- it can be directly applied to any dual-encoder model architecture;
- it is extended to scenarios with an arbitrary number of classes;
- polynomial annealing of the PU loss is proposed to stabilize the training;
- our experiments (reproduction steps are illustrated below) show that PAUSE constantly outperforms baseline methods.

This repository contains Tensorflow implementation of PAUSE to reproduce the experimental results. Upon using this repo for your work, please cite:
```bibtex
@inproceedings{cao-etal-2021-pause,
    title = "{PAUSE}: Positive and Annealed Unlabeled Sentence Embedding",
    author = "Cao, Lele  and
      Larsson, Emil  and
      von Ehrenheim, Vilhelm  and
      Cavalcanti Rocha, Dhiana Deva  and
      Martin, Anna  and
      Horn, Sonja",
    booktitle = "Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2021",
    address = "Online and Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.emnlp-main.791",
    pages = "10096--10107",
}
```

## Prerequisites
Install virtual environment first to avoid breaking your native environment. 
If you use [Anaconda](https://www.anaconda.com/distribution/), do
```
conda update conda
conda create --name py37-pause python=3.7
conda activate py37-pause
```

Then install the dependent libraries:
```
pip install -r requirements.txt
```

## Unsupervised STS
Models are trained on a combination of the SNLI and Multi-Genre NLI datasets, which contain one million sentence pairs annotated with three labels: entailment, contradiction and neutral. The trained model is tested on the STS 2012-2016, STS benchmark, and SICK-Relatedness (SICK-R) datasets, which have labels between 0 and 5 indicating the semantic relatedness of sentence pairs.

### Training
Example 1: train PAUSE-small using 5% labels for 10 epochs
```bash
python train_nli.py \
  --batch_size=1024 \
  --train_epochs=10 \
  --model=small \
  --pos_sample_prec=5
```
Example 2: train PAUSE-base using 30% labels for 20 epochs
```bash
python train_nli.py \
  --batch_size=1024 \
  --train_epochs=20 \
  --model=base \
  --pos_sample_prec=30
```

To check the parameters, run
```bash
python train_nli.py --help
```
which will print the usage as follows.
```
usage: train_nli.py [-h] [--model MODEL]
                    [--pretrained_weights PRETRAINED_WEIGHTS]
                    [--train_epochs TRAIN_EPOCHS] [--batch_size BATCH_SIZE]
                    [--train_steps_per_epoch TRAIN_STEPS_PER_EPOCH]
                    [--max_seq_len MAX_SEQ_LEN] [--prior PRIOR]
                    [--train_lr TRAIN_LR] [--pos_sample_prec POS_SAMPLE_PREC]
                    [--log_dir LOG_DIR] [--model_dir MODEL_DIR]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         The tfhub link for the base embedding model
  --pretrained_weights PRETRAINED_WEIGHTS
                        The pretrained model if any
  --train_epochs TRAIN_EPOCHS
                        The max number of training epoch
  --batch_size BATCH_SIZE
                        Training mini-batch size
  --train_steps_per_epoch TRAIN_STEPS_PER_EPOCH
                        Step interval of evaluation during training
  --max_seq_len MAX_SEQ_LEN
                        The max number of tokens in the input
  --prior PRIOR         Expected ratio of positive samples
  --train_lr TRAIN_LR   The maximum learning rate
  --pos_sample_prec POS_SAMPLE_PREC
                        The percentage of sampled positive examples used in
                        training; should be one of 1, 10, 30, 50, 70
  --log_dir LOG_DIR     The path where the logs are stored
  --model_dir MODEL_DIR
                        The path where models and weights are stored
```

### Testing
After the model is trained, you will be prompted to where the model is saved, e.g. `./artifacts/model/20210517-131724`, where the directory name (`20210517-131724`) is the model ID. To test the model with that ID, run

```bash
python test_sts.py --model=20210517-131724
```

The test result on STS datasets will be printed on console and also saved in file `./artifacts/test/sts_20210517-131724.txt`

## Supervised STS
### Train
You can continue to finetune a pertained model on supervised STSb. For example, assume we have trained a PAUSE model based on **small** BERT (say located at `./artifacts/model/20210517-131725`), if we want to finetune the model on STSb for 2 epochs, we can run

```bash
python ft_stsb.py \
  --model=small \
  --train_epochs=2 \
  --pretrained_weights=./artifacts/model/20210517-131725
```
Note that it is important to match the model size (`--model`) with the pretrained model size (`--pretrained_weights`).

### Testing
After the model is finetuned, you will be prompted to where the model is saved, e.g. `./artifacts/model/20210517-131726`, where the directory name (`20210517-131726`) is the model ID. To test the model with that ID, run

```bash
python ft_stsb_test.py --model=20210517-131726
```

## SentEval evaluation

To evaluate the PAUSE embeddings using [SentEval](https://github.com/facebookresearch/SentEval) (preferably using GPU), you need to download the data first:
```bash
cd ./data/downstream
./get_transfer_data.bash
cd ../..
```
Then, run the `sent_eval.py` script:

```bash
python sent_eval.py \
  --data_path=./data \
  --model=20210328-212801
```
where the `--model` parameter specifies the ID of the model you want to evaluate. By default, the model should exist in folder `./artifacts/model/embed`. If you want to evaluate a trained model in our public GCS (`gs://motherbrain-pause/model/...`), please run (e.g. `PAUSE-NLI-base-50%`):
```bash
python sent_eval.py \
  --data_path=./data \
  --model_location=gcs \
  --model=20210329-065047
```
We provide the following models for demonstration purposes:
|        Model        |      Model ID    |
|---------------------|:----------------:|
| PAUSE-NLI-base-100% |  20210414-162525 |
| PAUSE-NLI-base-70%  |  20210328-212801 |
| PAUSE-NLI-base-50%  |  20210329-065047 |
| PAUSE-NLI-base-30%  |  20210329-133137 |
| PAUSE-NLI-base-10%  |  20210329-180000 |
| PAUSE-NLI-base-5%   |  20210329-205354 |
| PAUSE-NLI-base-1%   |  20210329-195024 |