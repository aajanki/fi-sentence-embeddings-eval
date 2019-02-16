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
