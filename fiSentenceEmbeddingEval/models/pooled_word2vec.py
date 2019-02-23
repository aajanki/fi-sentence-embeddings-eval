import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


class PooledWord2Vec:
    split_whitespace_re = re.compile(r"(?u)\b\w\w+\b")

    def __init__(self, data_filename):
        self.name = 'Pooled word2vec'
        self.model = KeyedVectors.load_word2vec_format(data_filename,
                                                       binary=True)

    def fit(self, sentences):
        # nothing to do
        pass

    def describe(self):
        return '\n'.join([
            f'Embedding dimensionality: {self.model.vector_size}',
            f'Vocabulary size: {len(self.model.vocab)}'
        ])

    def transform(self, sentences):
        return np.array([self.embedding(s) for s in sentences])

    def embedding(self, sentence):
        vecs = [self.word_embedding(w) for w in self.tokenize(sentence)]
        if vecs:
            return np.atleast_2d(np.array(vecs)).mean(axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def tokenize(self, sentence):
        for w in PooledWord2Vec.split_whitespace_re.findall(sentence):
            yield w

    def word_embedding(self, word):
        if word in self.model.vocab:
            return self.model.word_vec(word)
        elif word.lower() in self.model.vocab:
            return self.model.word_vec(word.lower())
        else:
            return np.zeros(self.model.vector_size)

