#!/bin/sh

# Finnish Internet Parsebank word2vec vectors
# http://bionlp.utu.fi/finnish-internet-parsebank.html
mkdir -p data/fin-word2vec
wget -P data/fin-word2vec/ http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin

# Word frequencies on Finnish Internet Parseback
# https://turkunlp.org/finnish_nlp.html
mkdir -p data/finnish_vocab
wget -P data/finnish_vocab/ http://bionlp-www.utu.fi/.jmnybl/finnish_vocab.txt.gz

# FastText Finnish model
# https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md
mkdir -p pretrained/fasttext-fi
wget -P pretrained/fasttext-fi/ https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.fi.300.bin.gz
gunzip pretrained/fasttext-fi/cc.fi.300.bin.gz

# BERT
# https://github.com/google-research/bert/blob/master/multilingual.md
mkdir -p pretrained/bert
wget -P pretrained/bert https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip
unzip -d pretrained/bert pretrained/bert/multi_cased_L-12_H-768_A-12.zip
