---
title: Evaluating Finnish sentence embedding models
author:
- Antti Ajanki:
  name: Antti Ajanki
  email: antti.ajanki@iki.fi
date: 28.7.2020
section:
- title: Results
  href: index.html
- title: Embedding models
  href: models.html
- title: Evaluation tasks
  href: tasks.html
license:
  name: CC BY 4.0
  href: https://creativecommons.org/licenses/by/4.0/
...

This study compares modern sentence classification models on the
Finnish language. The goal is to understand which sentence
classification models should be considered as quick baselines in
various natural language processing (NLP) tasks without having to
resort to a long and costly training. Therefore, only publicly
available pre-trained Finnish models are included in the evaluation.
The models are evaluated on four datasets with sentence classification
and paraphrasing tasks.

FinBERT attains the highest accuracy scores on all tested tasks.

#### Change history

22.7.2019 The initial version\
29.8.2019 Using macro F1 score as the evaluation metric because it
makes more sense on unbalanced classification. The change makes SIF
stand out even more clearly. (git commit
[8eb451f6](https://github.com/aajanki/fi-sentence-embeddings-eval/tree/8eb451f6db888af6c48e931109d6d2ee0cd56ea0))\
21.9.2019 Grid search over the TF-IDF minimum document frequency.
TF-IDF now beats other methods on one task. (git commit
[fb5ac7bb](https://github.com/aajanki/fi-sentence-embeddings-eval/tree/fb5ac7bba3da7b18db444d476757cfc2363b344e))\
28.7.2020 Included FinBERT, dropped the multilingual BERT. Updated the eduskunta-vkk dataset to version v2. (git commit [41230ed](https://github.com/aajanki/fi-sentence-embeddings-eval/tree/41230ed79e84e660e9674b2b1279023df9ff23f4))

## Evaluation results

The evaluation consists of four classification tasks, each on a
separate dataset. In two cases the task is to assign a sentence to a
class (Eduskunta-VKK, TDT categories), in one case to detect if two
sentences are consecutive or not (Ylilauta), and in one the task is to
identify if two sentences are paraphrases (Opusparcus). See [the task
page](tasks.html) for further details.

The classification models can be divided in three groups:

* Models based on aggregated word embeddings: Pooled word2vec, pooled FastText, SIF, BOREP
* Full sentence embeddings: FinBERT, LASER
* TF-IDF as a baseline

These models have been shown to perform well on English text. The aim
of this investigation is to see how well they (or their multilingual
variants) perform on the Finnish language. A more detailed description
of the models is given on [the embedding models page](models.html).

The performance of the sentence classification models (the colored
bars) on the evaluation tasks (the four panels) is shown below. The
models are evaluated on test data not seen during the training. The
presented accuracy scores are averages over three random
initialization.

![Performance of the evaluated models (the colored bars) on the
evaluation tasks (the four panels)](images/scores.svg)

You can download the results in the [CSV format from
here](https://github.com/aajanki/fi-sentence-embeddings-eval/blob/master/scores/scores.csv).

The [source code for replicating the
results](https://github.com/aajanki/fi-sentence-embeddings-eval) is
available.

## Key findings

FinBERT gives the best performance on all tested tasks, and it should
be the first choice when the highest accuracy is required.

In some cases it might make sense to consider one the simpler models,
too. The frequency weighted average of word2vec (SIF) is the second
best model behind FinBERT on three tasks out of four. Forming the
embedding vectors and training the classifier takes 1.5-5 times as
long for FinBERT as for SIF in these experiments. The pooled word2vec
is even easier to implement and still performs quite well.

These results reinforce the previous findings in the literature that
in general word embeddings perform better better than simpler
bag-of-word models (TF-IDF). SIF and the average pooled word2vec beat
TF-IDF on three tasks out of four.

LASER, which incorporate the sentence context, performs considerably
worse than FinBERT despite the similarities in the architecture. This
could indicate the importance of monolingual training on the target
language and show weakness on LASER's multilingual approach.

The evaluation could be made more comprehensive by including more
varied NLP tasks, such as question answering, natural language
inference or sentiment analysis, but I'm not aware of suitable public
Finnish datasets. This study compares only pre-trained models to limit
the required computational effort. It would be interesting to find out
how much performance improves if an embedding model is fine-tuned for
the task under evaluation. This could benefit especially BERT and
other advanced models.
