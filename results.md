---
title: Evaluating Finnish sentence embedding models
author:
- Antti Ajanki:
  name: Antti Ajanki
  email: antti.ajanki@iki.fi
date: 21.7.2019
section:
- title: Results
  href: results.html
- title: Evaluation tasks
  href: tasks.html
- title: Embedding models
  href: models.html
...

This study compares modern sentence classification models on the
Finnish language. The goal is to understand which sentence
classification models should be considered as quick baselines without
needing a long and expensive training runs. This means that only
publicly available pre-trained Finnish models are included in the
evaluation. The models are evaluated on four datasets with sentence
classification and paraphrasing tasks. Surprisingly, I find that on
Finnish average pooled or frequency weighted word embeddings tend to
perform better than BERT and other advanced contextual sentence
representations.

## Sentence embedding

A crucial component in most natural language processing (NLP)
applications is finding an expressive representation for text. Modern
methods typically map a sentence onto a numerical vector which
attempts to capture the semantic content of the text. If two sentences
express a similar idea using different words, their representations
(vectors) should still be similar to each other. The vectors are
called sentence embeddings as the textual information in a sentence is
embedded in a low dimensional numerical space.

Several methods for constructing these embeddings have been proposed
in the literature. Interestingly, learned embeddings tend generalize
quite well to other material and NLP tasks besides the ones they were
trained on. This is fortunate, because it allows us to use pre-trained
models and avoid costly training.

Some application areas where sentence embeddings are applied include
information retrieval (comparing the meaning of text snippets),
machine translation (sentence embeddings as an “intermediate language”
when translating between two human languages) and various
classification and tagging applications. With better representations
it is possible to build applications that react more naturally to the
sentiment and topic of written text.

## Evaluation results

The evaluation consists of four classification tasks, each on a
separate dataset. In two cases the task is to assign a class for a
sentence (Eduskunta-VKK, TDT categories), in one case to detect if two
sentences are consecutive or not (Ylilauta), and in one the task is to
identify if two sentences are paraphrases (Opusparcus). See [the task
page](tasks.html) for further details.

The classification models can be divided in three groups:

* Models based on aggregated word embeddings: Pooled word2vec, pooled FastText, SIF, BOREP
* Full sentence embeddings: BERT, LASER
* TF-IDF as a baseline

These models have been shown to perform well on English text. The aim
of this investigation is to see how well they (or their multilingual
variant) perform on Finnish material. A more detailed description of
the models is given on [the separate page](models.html).

The evaluation results are shown below:

![Model performances on the evaluation tasks](images/scores.svg)

The [source code for replicating the
results](https://github.com/aajanki/fi-sentence-embeddings-eval) is
available.

## Key findings

Average pooled word2vec or frequency weighted average of word2vec
(SIF) should be the first choices because they are simple to implement
and are among the top performers in all tested tasks. I was unable to
replicate the finding by [@wieting2019] that pooled random projections
(BOREP) would be consistently better than plain pooled embeddings
(pooled word2vec).

These results reinforce previous findings in the literature that in
general word embeddings perform better better than older bag-of-word
models (TF-IDF). The average pooled word2vec beats TF-IDF on there
tasks out of four.

More advanced models BERT and LASER, which incorporate the sentence
context, are not much better than SIF and often worse. This is in
contrast to general experiences on the English language, where BERT
was the state-of-the-art in the beginning of 2019. Additionally, BERT
and LASER are more complicated, have more hyperparameters that require
tuning, and are slower than pooled word embeddings. So there is no
reason to prefer BERT or LASER on Finnish documents. The situation
could of course change if one would train the models with
task-specific material, but that requires serious computational
resources and a large dataset.

The evaluation could be made more comprehensive by including different
kinds of NLP tasks, such as question answering, natural language
inference or sentiment analysis, but I'm not aware of suitable public
Finnish datasets. This study compares only pre-trained models. It
would be interesting to find out how much performance improves when a
sentence embedding model is trained specifically for a certain task.

## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
