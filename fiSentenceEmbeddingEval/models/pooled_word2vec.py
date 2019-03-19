import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from .sentenceembedding import SentenceEmbeddingModel


class PooledWord2Vec(SentenceEmbeddingModel):
    def __init__(self, name, data_filename):
        super().__init__(name)
        self.model = KeyedVectors.load_word2vec_format(data_filename,
                                                       binary=True)

    def describe(self):
        return '\n'.join([
            self.name,
            f'Embedding dimensionality: {self.model.vector_size}',
            f'Vocabulary size: {len(self.model.vocab)}'
        ])

    def embedding(self, sentence):
        vecs = [self.word_embedding(w) for w in self.tokenize(sentence)]
        if vecs:
            return np.atleast_2d(np.array(vecs)).mean(axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def word_embedding(self, word):
        if word in self.model.vocab:
            return self.model.word_vec(word)
        elif word.lower() in self.model.vocab:
            return self.model.word_vec(word.lower())
        else:
            return np.zeros(self.model.vector_size)

