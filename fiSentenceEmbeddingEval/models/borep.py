import numpy.random
import numpy as np
from .pooled_word2vec import PooledWord2Vec


class BOREP(PooledWord2Vec):
    """Bag of random embedding projections

    Project word embeddings to a random high-dimensional space with an
    average pooling in the high dimensional projection space.

    John Wieting, Douwe Kiela: No Training Required: Exploring Random
    Encoders for Sentence Classification, ICLR 2019
    """
    def __init__(self, name, word2vec_filename, projection_dim):
        super().__init__(name, word2vec_filename)

        input_dim = self.model.vector_size
        self.W = self.generate_projection(input_dim, projection_dim)

    def fit(self, sentences):
        self.W = self.generate_projection(self.W.shape[1], self.W.shape[0])

    def generate_projection(self, input_dim, projection_dim):
        a = 1.0/np.sqrt(input_dim)
        return numpy.random.uniform(-a, a, (projection_dim, input_dim))

    def describe(self):
        return '\n'.join([
            self.name,
            f'Projection dimensionality: {self.W.shape[0]}',
            f'Original embedding dimensionality: {self.W.shape[1]}',
            f'Vocabulary size: {len(self.model.vocab)}'
        ])

    def embedding(self, sentence):
        vecs = [self.W.dot(self.word_embedding(w))
                for w in self.tokenize(sentence)]
        if vecs:
            return np.atleast_2d(np.array(vecs)).mean(axis=0)
        else:
            return np.zeros(self.W.shape[0])
