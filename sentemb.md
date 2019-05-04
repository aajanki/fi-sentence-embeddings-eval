---
title: Evaluating Finnish sentence embedding models
author:
- name: Antti Ajanki
  email: antti.ajanki@iki.fi
date: May 2019
abstract: >
  This study compares commonly used pre-trained NLP sentence
  representation models on the Finnish language. The selected models are
  compared on four sentence (or sentence-pair) classification tasks.
  Surprisingly, I find that on Finnish average pooled or frequency
  weighted word embeddings tend to perform better than more advanced
  contextual sentence representation models.
...


## Introduction

* sentence representation
* Sentence embedding applications
* Pre-training
* Finnish

Most of the sentence representation research has so far focused on
English. However, some researcher groups have published multi-lingual
variants of their models. Such models are trained on documents written
in dozens of different languages, and are able to generate sentence
representations on all languages included in the training corpus.
Pre-trained LASER and BERT models support Finnish.

Most evaluations of the previously published multi-lingual models
still emphasize the performance on English. One goal of this study is
to present a more comprehensive evaluation on Finnish corpora.

## Related work

### Models

* TF-IDF bag-of-words
* word embeddings based models: word2vec [@mikolov2013], fastText [@bojanowski2017], SIF [@arora2017], BOREP [@wieting2019]
* contextual embeddings: BERT [@devlin2018], LASER [@artetxe2018]

### Transfer learning

### Feature extraction vs tuning

## Experiments

The models were compared on the following Finnish sentence
classification tasks:

**Eduskunta-VKK** [@eduskuntavkk]: The corpus is based on cabinet
ministers' answers to written questions by the members of Finnish
parliament.

The task is straight-forward sentence classification. The sentences
are extracted from the answers and the correct class are is the
ministry which gave the answer to the question. There are 15 classes
(ministries) and 49693 sentences as training data and 2000 sentences
as test data. The evaluation measure is the F1 score.

**Opusparcus** [@creutz2018]: Sentence pair paraphrasing task. The
corpus is based on sentences from open movie subtitle datasets. The
size of the training dataset is about 20 million sentence pairs and
the testset is about 2000 pairs. The training data consists of
sentence pairs and a statistical estimate on how likely the two
sentences mean the same thing. The estimate is based on how often
sentences are aligned with similar translations over subtitles on
multiple languages. Model's task is to learn to predict this alignment
score.

The development and test sets consists of sentence pairs that have
been labeled by human annotators on the scale 1 (sentences do not mean
the same thing) to 5 (the sentences are paraphrases). The evaluation
measure is the correlation between model's prediction and the human
score. The model is evaluated only on the Finnish portion of the
corpus.

**Ylilauta** [@ylilautacorpus]: Sentences extracted from Ylilauta
discussion forum. The task is to predict if a given pair of sentences
are consecutive sentences in the original text or not. The size of the
training dataset is 5000 sentence pairs and the test dataset is 2000
pairs. Half of the sentence pairs are in reality consecutive and half
are randomly paired sentences. The evaluation measure is the
classification accuracy.

**TDT categories** [@pyysalo2015]: The dataset contains Finnish
sentences extracted from many sources: blogs, Wikinews, Europarl,
student magazine articles, etc. The task is to predict the source of a
sentence. There are 8 classes, and about 8000 training and 1000 test
sentences. The evaluation measure is the F1 score.

### Classifier architecture

The embeddings from a given model are used as an input for a single
hidden layer neural network classifier. The classifier is trained on
training data, the pre-trained embedding models are not trained. The
classifier parameters are learned on a development dataset for each
task and embedding model separately. See the source code for the
details.

## Results

![Model performances on the evaluation tasks](images/scores.svg)

## Discussion

* Average pooled word2vec or SIF (= frequency weighted average of
  word2vec) should be the first choices because they are among the top
  performers in all tested tasks.
* Word embedding models perform much better than TF-IDF
* More advanced models BERT and LASER, which incorporate word context,
  are not much better than SIF and often worse. This is in contrast to
  general experiences on English. Maybe the BERT/LASER training
  corpuses for Finnish are too small?


## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
