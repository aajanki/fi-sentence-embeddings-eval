---
title: Evaluation tasks
section:
- title: Results
  href: results.html
- title: Sentence embeddings
  href: sentemb.html
- title: Evaluation tasks
  href: tasks.html
- title: Experiments
  href: experiments.html
...

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
