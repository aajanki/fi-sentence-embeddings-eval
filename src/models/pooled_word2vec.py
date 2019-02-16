import re
import numpy as np
from gensim.models.keyedvectors import KeyedVectors


_split_whitespace_re = re.compile(r"(?u)\b\w\w+\b")


class PooledWord2Vec:
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
        num_words = 0
        vecs = []
        for w in _split_whitespace_re.findall(sentence):
            if w in self.model.vocab:
                vecs.append(self.model.word_vec(w))
            elif w.lower() in self.model.vocab:
                vecs.append(self.model.word_vec(w.lower()))
            else:
                # implicitly assume a zero vector for out-of-vocabulary words
                pass

            num_words += 1

        if num_words > 0 and len(vecs) > 0:
            return np.sum(vecs, axis=0)/num_words
        else:
            return np.zeros(self.model.vector_size)
