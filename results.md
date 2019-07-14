---
title: Evaluation results
section:
- title: Results
  href: results.html
- title: Sentence embeddings
  href: sentemb.html
- title: Experiments
  href: experiments.html
...

I compared the performance of a few sentence classification algorithms
on the Finnish language.

The evaluation consisted of four classification tasks on four separate
datasets. In two cases the task was to assign a class for a sentence
(Eduskunta-VKK, TDT categories), in one case to detect if two
sentences are consecutive or not (Ylilauta), and in one the task was
to identify if two sentences are paraphrases (Opusparcus). See [the
experiments page](experiments.html) for further details.

The classification models can be divided in three groups:

* Models based on aggregated word embeddings: Pooled word2vec, pooled FastText, SIF, BOREP
* Full sentence embeddings: BERT, LASER
* TF-IDF as a baseline

These models have been shown to perform well on English text. The aim
of this investigation is to see how well they (or their multilingual
variant, to be more precise) perform on Finnish material. Only models
with publicly available pre-trained Finnish or multilingual
implementation are included. This means that some state-of-the-art
models without a multilingual variant, such as GPT-2 and XLNet, are
not included in this evaluation. A more detailed description of the
models is given on [the experiments page](experiments.html).

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

## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
