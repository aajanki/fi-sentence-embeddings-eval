#!/bin/sh

### DATASET ###

# Finnish Internet Parsebank word2vec vectors
# http://bionlp.utu.fi/finnish-internet-parsebank.html
mkdir -p data/fin-word2vec
wget -P data/fin-word2vec/ http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin

# Word frequencies on Finnish Internet Parseback
# https://turkunlp.org/finnish_nlp.html
mkdir -p data/finnish_vocab
wget -P data/finnish_vocab/ http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz

# Opusparcus: Open Subtitles Paraphrase Corpus for Six Languages
# http://urn.fi/urn:nbn:fi:lb-2018021221
mkdir -p data/opusparcus
wget -P data/opusparcus/ https://korp.csc.fi/download/opusparcus/opusparcus_fi.zip
wget -P data/opusparcus/ https://korp.csc.fi/download/opusparcus/README.txt
unzip -od data/opusparcus/ data/opusparcus/opusparcus_fi.zip

# Ylilauta
# http://urn.fi/urn:nbn:fi:lb-2016101210
mkdir -p data/ylilauta
wget -P data/ylilauta/ https://korp.csc.fi/download/Ylilauta/ylilauta_20150304.zip
unzip -od data/ylilauta/ data/ylilauta/ylilauta_20150304.zip
python fiSentenceEmbeddingEval/prepare-ylilauta.py

# Eduskunta: Vastaukset kirjallisiin kysymyksiin
mkdir -p data/eduskunta-vkk
wget -P data/eduskunta-vkk https://github.com/aajanki/eduskunta-vkk/raw/1.0/vkk/train.csv.bz2
wget -P data/eduskunta-vkk https://github.com/aajanki/eduskunta-vkk/raw/1.0/vkk/dev.csv.bz2
wget -P data/eduskunta-vkk https://github.com/aajanki/eduskunta-vkk/raw/1.0/vkk/test.csv.bz2


### PRE-TRAINED MODELS ###

# word2vec embeddings for Finnish
mkdir -p pretrained/fin-word2vec/
wget -P pretrained/fin-word2vec/ http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin

# FastText Finnish model
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
mkdir -p pretrained/fasttext-fi
wget -P pretrained/fasttext-fi/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz
gunzip pretrained/fasttext-fi/cc.fi.300.bin.gz

# BERT
# https://github.com/google-research/bert/blob/master/multilingual.md
mkdir -p pretrained/bert
wget -P pretrained/bert https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip -od pretrained/bert pretrained/bert/multi_cased_L-12_H-768_A-12.zip

# LASER
# https://github.com/facebookresearch/LASER
#
# Checkout the source code here because we need to run scripts from
# the source root
git clone https://github.com/facebookresearch/LASER.git
export LASER=`pwd`/LASER
LASER/install_external_tools.sh
LASER/install_models.sh
