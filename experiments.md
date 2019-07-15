---
title: Experiments
section:
- title: Results
  href: results.html
- title: Sentence embeddings
  href: sentemb.html
- title: Experiments
  href: experiments.html
...

## Sentence classifier architecture

The evaluation tasks are either sentence classification tasks (given a
sentence output the class it belongs to) or sentence pair comparison
tasks (given a pair of sentences output a binary yes/no judgment: are
the two sentences paraphrases or do they belong to the same document).

The evaluation architecture for the single sentence tasks is as
follows: the sentence embedding model under evaluation converts the
sentence text into a sentence embedding vector that is used as input
layer to a dense neural network consisting of one hidden layer and a
softmax output layer. The classifier part is trained on the
development dataset (separately for each task and embedding model) but
the pre-trained sentence embedding model is keep fixed during the
whole experiment.

For sentence pair tasks, I'm using a technique introduced
by [@tai2015]: first, sentence embeddings are generated for the two
sentences separately, and then they are merged into a single feature
vector that represents the pair. Let the embeddings for the two
sentences be called $u$ and $v$. The feature vector for the pair is
then generated as a concatenation of the elementwise product $u \odot v$
and the elementwise absolute distance $|u-v|$. The concatenated
feature vector is then used as the input for the classification part,
like above. The BERT model is an exception. It has an integrated way
of handling sentence pair tasks (see below).

The pre-trained sentence embedding models are treated as black box
feature extractors that output embedding vectors. An alternative
approach is to fine-tune a pre-trained embedding model by optimizing
the full pipeline (usually with a small learning rate for the
embedding part). Feature extraction is computationally cheaper, but
fine-tuning better adapts pre-trained representations to many
different tasks. The relative performance of feature extraction and
fine-tuning approaches depends on the similarity of the pre-training
and target tasks [@peters2019] and is worth more careful study. The
current evaluation focuses only on the feature extraction to
understand what can be expected on relatively low computational
resources.

### Pre-training

### Transfer learning

## Sentence embedding models

This section introduces the sentence embedding models that a compared
in this study. I have selected models which have a pre-trained Finnish
variant publicly available (or which require only minimal training).

The most important sentence embedding models can be divided into three
categories: baseline bag-of-word models, models that are based on
aggregating embeddings for the individual words in a sentence, and
models that build a representation for a full sentence.

**Bag-of-words** (BoW). I'm using [TF-IDF (term frequency-inverse
document frequency)](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)
vectors as a baseline. A TF-IDF vector is a sparse vector with one
dimension per each unique word in the vocabulary. The value is a count
of a particular word in a sentence multiplied by a factor that is
inversely proportional to the overall frequency of that word in the
whole corpus. The latter factor is meant to diminish the effect of
very common words which are unlikely to tell much about the actual
content of the sentence. The vectors are L2-normalized to reduce the
effect of different sentence lengths.

As a preprocessing, words are converted to their dictionary form
(lemmatized) and all unigrams and bigrams occurring more than four
times are selected. Typically (depending on the corpus) this results
in dimensionality (i.e. the vocabulary size) around 5000.

BoW models treat each word as a homogenous entity. They don't consider
the semantic similarity of words. A BoW model doesn't understand that
a *horse* and a *pony* are more similar than a *horse* and a *shovel*.
Next, we'll move on to word embedding models that have been proposed
to include semantic information.

**Pooled word embeddings**. A [word
embedding](https://en.wikipedia.org/wiki/Word_embedding) is a vector
representation of a word. Words, which often occur in similar context
(a *horse* and a *pony*), are assigned vector that are close by each
other and words, which rarely occur in a similar context (a *horse*
and a *shovel*), dissimilar vectors. The embedding vectors are dense
fixed-sized relatively low dimensional (typically 50-300 dimensions)
vectors.

The embedding vectors are typically trained on a language modeling
task: predict the presentation for a word given the representation of
a few previous (or previous and following) words. This is unsupervised
or, rather, self-supervised task as the supervision signal is the
order of the word in a document. The training requires just a large
corpus of text documents. It is well established that word embeddings
learned on a language modeling task generalize well to other
downstream tasks: classification, part-of-speech tagging and so on.

One of the earliest and still widely used word embedding model is the
word2vec model [@mikolov2013]. I'm using the Finnish word2vec vectors
trained on the [Finnish Internet
Parsebank](https://turkunlp.org/finnish_nlp.html#parsebank) data by
the [Turku NLP research group](https://turkunlp.org/). FastText
[@bojanowski2017] extended the idea by generating embeddings for
subword units (character n-grams). By utilizing the subword structure,
FastText aims to provide better embeddings for rare and
out-of-vocabulary words. The authors of FastText have published a
Finnish model trained on Finnish subsets of [Common
Crawl](http://commoncrawl.org/) and
[Wikipedia](https://www.wikipedia.org/).

There are several ways to get from word representations to sentence
representations. Perhaps, the most straight-forward idea is to
aggregate the embeddings of each word that appears in a sentence, for
example, by taking an element-wise average, minimum or maximum. These
strategies are called average-pooling, min-pooling and max-pooling,
respectively. Still another alternative is the concatenation of min-
and max-pooled vectors. In this work, I'm comparing average-pooled
word2vec and FastText models. Average-pooling performed better than
the alternatives on a brief preliminary study on the TDT dataset. As
this might be dataset-dependent, it's a good idea to experiment with
different pooling strategies.

Some researchers have proposed slightly more advanced aggregation
methods that still require little or no training. I'm evaluating two
such proposals here.

[@arora2017] introduced a model called smooth inverse frequency (SIF).
They propose taking a weighted average (weighted by a term related to
the inverse document frequency) of the word embedding in a sentence
and then removing the projection of the first singular vector. This is
derived from an assumption that the sentence has been generated by a
random walk of a discourse vector on a latent word embedding space,
and by including smoothing terms for frequent words.

The idea of BOREP or bag-of-random-embedding-projections
[@wieting2019] is to project word embeddings to a random
high-dimensional space and pool the projections there. The intuition
is that casting things into a higher dimensional space tends to make
them more likely to be linearly separable thus making the text
classification easier. In the evaluations, I'm projecting to a
4096 dimensional space following the lead of the paper authors.

While word embeddings capture some aspects of semantics, pooling
embeddings is quite a crude way of aggregating the meaning of a
sentence. The current state-of-the-art in NLP are deep learning models
that take a whole sentence as an input and are thus capable of
processing the full context of the sentence. Let's take a look at two
such models.

**Contextual full sentence embeddings**. BERT (Bidirectional Encoder
Representations from Transformers) [@devlin2018] processes a full
sentence to generate a sentence embeddings. It uses self-attention to
decide the relevant context for each word in the sentence. Another
major technical contribution in the model was a bi-directional
transformer architecture, which allows the model to consider word's
left and right context when making predictions.

In the evaluation, I'll use the value of the second-to-last hidden
layer for the special \[CLS\] token as the sentence embedding. BERT is
trained to aggregate information about the whole sequence to the
\[CLS\] token in classification tasks. I have selected the
second-to-last hidden layer because both the paper and my brief
preliminary study showed that it gives slightly better classification
performance than the \[CLS\] output layer. BERT also produces output
embeddings for each word, but these are not used in the evaluations.
In the paraphrase and consecutive sentence evaluation tasks that
directly compare two sentences, both sentences are fed as input
separated by a separator token to match how BERT was trained.

I'm using the pre-trained multilingual (Bert-base, multilingual cased)
variant published by the BERT authors. It has been trained on 104
languages, including Finnish. The embedding dimensionality is 768.

As the second contextual sentence embedding method, I'll evaluate
LASER (Language-Agnostic SEntence Representations) by [@artetxe2018].
It is specifically meant to learn language-agnostic sentence
embeddings. It has a similar deep bi-directional architecture as BERT,
but uses LSTM encoders instead of transformer blocks like BERT. I'm
using the pre-trained model published by the LASER authors. It
produces 1024 dimensional embedding vectors.

## Evaluation tasks

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

## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
