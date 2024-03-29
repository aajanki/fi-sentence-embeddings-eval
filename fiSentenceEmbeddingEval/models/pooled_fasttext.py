import numpy as np
from gensim.models.fasttext import load_facebook_vectors
from .sentenceembedding import SentenceEmbeddingModel


class PooledFastText(SentenceEmbeddingModel):
    def __init__(self, name, data_filename):
        super().__init__(name)
        self.model = load_facebook_vectors(data_filename)

    def describe(self):
        return '\n'.join([
            self.name,
            f'Embedding dimensionality: {self.model.vector_size}',
            f'Vocabulary size: {len(self.model.wv.vocab)}'
        ])

    def embedding(self, sentence):
        vecs = [self.word_embedding(w) for w in self.tokenize(sentence)]
        if len(vecs) > 0:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def word_embedding(self, word):
        try:
            return self.model.wv[word]
        except KeyError:
            # We end up here if no ngrams are present in the FastText
            # model
            return np.zeros(self.model.vector_size)
