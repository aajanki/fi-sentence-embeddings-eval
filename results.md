---
title: Results
author:
- name: Antti Ajanki
  email: antti.ajanki@iki.fi
section:
- title: Results
  href: results.html
- title: Sentence embeddings
  href: sentemb.html
- title: Experiments
  href: experiments.html
...

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
