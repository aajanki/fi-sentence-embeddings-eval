---
title: Evaluating Finnish sentence embedding models
author:
- name: Antti Ajanki
  email: antti.ajanki@iki.fi
date: May 2019
section:
- title: Results
  href: results.html
- title: Sentence embeddings
  href: sentemb.html
- title: Evaluation tasks
  href: tasks.html
- title: Embedding models
  href: models.html
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


## References
\setlength{\parindent}{-0.2in}
\setlength{\leftskip}{0.2in}
\setlength{\parskip}{8pt}
\vspace*{-0.2in}
\noindent
