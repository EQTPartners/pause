#!/bin/bash
#
# Copyright (C) eqtgroup.com Ltd 2021
# https://github.com/EQTPartners/pause
# License: MIT, https://github.com/EQTPartners/pause/LICENSE.md
#
# Download and tokenize data with MOSES tokenizer
# This is adapted from https://github.com/facebookresearch/SentEval


data_path=.
preprocess_exec=./tokenizer.sed

# Get MOSES
echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git
SCRIPTS=mosesdecoder/scripts
MTOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LOWER=$SCRIPTS/tokenizer/lowercase.perl

if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

PTBTOKENIZER="sed -f tokenizer.sed"

mkdir $data_path

TREC='http://cogcomp.cs.illinois.edu/Data/QA/QC'
BINCLASSIF='https://dl.fbaipublicfiles.com/senteval/senteval_data/datasmall_NB_ACL12.zip'
SSTbin='https://raw.githubusercontent.com/PrincetonML/SIF/master/data'

# MRPC is a special case (we use "cabextract" to extract the msi file on Linux, see below)
MRPC='https://download.microsoft.com/download/D/4/6/D46FF87A-F6B9-4252-AA8B-3604ED519838/MSRParaphraseCorpus.msi'


### Get Stanford Sentiment Treebank (SST) binary classification task
# SST binary
mkdir -p $data_path/SST/binary
for split in train dev test
do
    curl -Lo $data_path/SST/binary/sentiment-$split $SSTbin/sentiment-$split
done


### download TREC
mkdir $data_path/TREC

for split in train_5500 TREC_10
do
    urlname=$TREC/$split.label
    curl -Lo $data_path/TREC/$split.label $urlname
    sed -i -e "s/\`//g" $data_path/TREC/$split.label
    sed -i -e "s/'//g" $data_path/TREC/$split.label
done


### download MR CR SUBJ MPQA
# Download and unzip file
curl -Lo $data_path/data_classif.zip $BINCLASSIF
unzip $data_path/data_classif.zip -d $data_path/data_bin_classif
rm $data_path/data_classif.zip

# MR
mkdir $data_path/MR
cat -v $data_path/data_bin_classif/data/rt10662/rt-polarity.pos | $PTBTOKENIZER > $data_path/MR/rt-polarity.pos
cat -v $data_path/data_bin_classif/data/rt10662/rt-polarity.neg | $PTBTOKENIZER > $data_path/MR/rt-polarity.neg

# CR
mkdir $data_path/CR
cat -v $data_path/data_bin_classif/data/customerr/custrev.pos | $PTBTOKENIZER > $data_path/CR/custrev.pos
cat -v $data_path/data_bin_classif/data/customerr/custrev.neg | $PTBTOKENIZER > $data_path/CR/custrev.neg

# SUBJ
mkdir $data_path/SUBJ
cat -v $data_path/data_bin_classif/data/subj/subj.subjective | $PTBTOKENIZER > $data_path/SUBJ/subj.subjective
cat -v $data_path/data_bin_classif/data/subj/subj.objective | $PTBTOKENIZER > $data_path/SUBJ/subj.objective

# MPQA
mkdir $data_path/MPQA
cat -v $data_path/data_bin_classif/data/mpqa/mpqa.pos | $PTBTOKENIZER > $data_path/MPQA/mpqa.pos
cat -v $data_path/data_bin_classif/data/mpqa/mpqa.neg | $PTBTOKENIZER > $data_path/MPQA/mpqa.neg

# CLEAN-UP
rm -r $data_path/data_bin_classif


### download MRPC
mkdir $data_path/MRPC
curl -Lo $data_path/MRPC/msr_paraphrase_train.txt https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt
curl -Lo $data_path/MRPC/msr_paraphrase_test.txt https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt

# remove moses folder
rm -rf mosesdecoder
