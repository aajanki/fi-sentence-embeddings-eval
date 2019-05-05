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

A crucial component in most natural language processing applications
is finding an expressive representation for text. Typically, sentences
are mapped to numerical vectors which attempt to capture the semantic
content of the text. If two sentences express a similar idea using
different words, their representations should be similar to each
other. Mathematically, a representation is constructed by embedding
textual information into a low dimensional vector space.

A good sentence representation would enhance the performance of many
NLP applications. Comparing the meaning of snippets of text is a core
component of web search engines. Machine translation systems use text
representations as an "intermediate language" when translating between
two human languages. Applications that automatically classify, tag or
other wise organize text documents benefits from better a
representation. With better representations it is possible to build
applications that react more naturally to the sentiment and topic of
written text.

Researchers have studied many models for constructing sentence
representations. Many researchers have published their pre-trained
models open for anyone to use.

Most of the research has so far focused mainly on the English language
and other major languages. The purpose of this study is to present a
more comprehensive evaluation on Finnish texts, and to see if models
that are known to perform well on English also manage to maintain
their edge on Finnish texts.

## Related work

### Sentence embedding models

This section introduces the sentence embedding models that a compared
in this study. I have selected models which have a pre-trained Finnish
variant publicly available (or which require only minimal training).

The most important sentence embedding models can be divided into three
categories: baseline bag-of-word models, models that are based on
merging embeddings for the individual words in a sentence, and models
that process a full sentence at once.

**Bag-of-words**. I'm using TF-IDF (term frequency-inverse document
frequency) vectors as a baseline. A TF-IDF vector is a sparse vector
with one dimension per each unique word in the vocabulary. The value
is a count of a particular word in a sentence multiplied by a factor
that is inversely proportional to the overall frequency of that word
in the whole corpus. The latter factor is meant to diminish the effect
of very common words which are unlikely to tell much about the actual
content of the sentence.

As a preprocessing, words are converted to their dictionary form
(lemmatized) and all unigrams and bigrams occurring more than four
times are selected. Typically (depending on the corpus) this results
in dimensionality (vocabulary size) around 5000.

**Pooled word embeddings**.

* word2vec [@mikolov2013]
* fastText [@bojanowski2017]
* SIF [@arora2017]
* BOREP [@wieting2019]

**Contextula full sentence embeddings**.

* BERT [@devlin2018]
* LASER [@artetxe2018]

Some researcher groups have published multi-lingual variants of their
models. Such models are trained on documents written in dozens of
different languages, and are able to generate sentence representations
on all languages included in the training corpus. Pre-trained LASER
and BERT models support Finnish.

### Pre-training

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

The source code is available at
<https://github.com/aajanki/fi-sentence-embeddings-eval>

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
