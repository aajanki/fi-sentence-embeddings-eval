# Comparing Finnish sentence embedding methods

The purpose of this repository is to compare sentence embedding models
for Finnish and understand if the methods, which are known to perform
well on English language, are useful on Finnish, too.

[Sentence embeddings](https://en.wikipedia.org/wiki/Sentence_embedding) are
natural language processing algorithms that map textual sentences into
numerical vectors. Vectors are supposed to capture the meaning of the
sentence. The embeddings can be used to compare sentences: if two
sentences express a similar idea using different words, the
corresponding embedding vectors should still be close to each other.
Sentence embeddings are crucial on many NLP tasks, such as sentiment
analysis and machine translation.

The training of the embedding models usually requires very large text
corpora and significant computing power. Researchers have, however,
published pre-trained models which can be adapted to various
downstream tasks with reasonable low effort. Pre-trained sentence
embeddings are typically used as input features to a neural network
(or other machine learning model). Only the task-specific model is
trained while the sentence embedding model if kept fixed.

Researchers have so far focused mostly on English and other most
spoken languages. However, there have been a few pre-trained models
published for Finnish (or rather multilingual models that include
Finnish). This analysis will compare the published Finnish models.

Models included in the comparison:
* TF-IDF
* Average-pooled [word2vec](https://en.wikipedia.org/wiki/Word2vec) trained on the [Finnish Internet Parsebank](http://bionlp.utu.fi/finnish-internet-parsebank.html)
* Average-pooled multilingual [FastText](https://github.com/facebookresearch/fastText/blob/master/docs/crawl-vectors.md)
* Multilingual [BERT](https://github.com/google-research/bert/blob/master/multilingual.md)
* [Smoothed Inverse Frequency weighting](https://openreview.net/forum?id=SyK00v5xx) of word embeddings
* [Bag of embedding projections](https://openreview.net/forum?id=BkgPajAcY7)

## Download datasets and pre-trained models

```
./scripts/get_data.sh
```

## Run

```
pipenv run scripts/run.sh
```
