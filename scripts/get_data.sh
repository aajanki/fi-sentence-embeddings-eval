#!/bin/sh

# Finnish Internet Parsebank word2vec vectors
# http://bionlp.utu.fi/finnish-internet-parsebank.html
mkdir -p data/fin-word2vec
wget -P data/fin-word2vec/ http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin

# FastText Finnish model
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
mkdir -p data/fasttext-fi
wget -P data/fasttext-fi/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz
gunzip data/fasttext-fi/cc.fi.300.bin.gz

# BERT
# https://github.com/google-research/bert/blob/master/multilingual.md
mkdir -p data/bert
wget -P data/bert https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip -d data/bert data/bert/multi_cased_L-12_H-768_A-12.zip
