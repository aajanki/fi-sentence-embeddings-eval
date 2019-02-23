import gzip
import numpy as np
from sklearn.decomposition import TruncatedSVD
from gensim.models.keyedvectors import KeyedVectors
from .pooled_word2vec import PooledWord2Vec


class SIF(PooledWord2Vec):
    """Smoothed Inverse Frequency weighting scheme

    Sanjeev Arora and Yingyu Liang and Tengyu Ma: A Simple but
    Tough-to-Beat Baseline for Sentence Embeddings, ICLR 2017
    """

    def __init__(self, name, word_freq_filename, word2vec_filename, a=5e-4):
        super().__init__(name, word2vec_filename)

        self.p = self.load_unigram_probabilities(word_freq_filename)
        self.a = a
        self.pc = None

    def load_unigram_probabilities(self, filename):
        freqs = []
        tokens = []

        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            for line in f.readlines():
                arr = line.strip().split(' ', 1)
                freqs.append(int(arr[0]))
                tokens.append(arr[1])

        freqs = np.array(freqs)
        p = freqs/freqs.sum()
        return dict(zip(tokens, p))

    def fit(self, sentences):
        X = np.array([self.embedding(s) for s in sentences])
        self.pc = self.compute_first_pc(X)

    def transform(self, sentences):
        X = np.array([self.embedding(s) for s in sentences])
        return self.remove_pc(X, self.pc)

    def embedding(self, sentence):
        vecs = []
        for w in self.tokenize(sentence):
            emb = self.word_embedding(w)
            w = self.a/(self.a + self.p.get(w, 0.0))
            vecs.append(w*emb)

        if vecs:
            return np.atleast_2d(np.array(vecs)).mean(axis=0)
        else:
            return np.zeros(self.model.vector_size)

    def compute_first_pc(self, X):
        svd = TruncatedSVD(n_components=1, n_iter=7, random_state=0)
        svd.fit(X)
        return svd.components_

    def remove_pc(self, X, pc):
        return X - X.dot(pc.transpose()) * pc
