#!/bin/sh

# Finnish Internet Parsebank word2vec vectors
# http://bionlp.utu.fi/finnish-internet-parsebank.html
mkdir -p data/fin-word2vec
wget -O data/fin-word2vec/fin-word2vec.bin http://bionlp-www.utu.fi/fin-vector-space-models/fin-word2vec.bin
