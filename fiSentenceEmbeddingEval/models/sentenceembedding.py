import re
import numpy as np
from scipy import sparse


class SentenceEmbeddingModel:
    """The base class for sentence embedding models"""

    split_whitespace_re = re.compile(r"(?u)\b\w\w+\b")

    def __init__(self, name):
        self.name = name

    def fit(self, sentences):
        """Override to learn model hyperparameters

        Can be left unimplemented if the model requires no training.
        """
        pass

    def describe(self):
        """Human-readable description of the model"""
        return ''

    def transform(self, sentences):
        """Transform a list of sentences to a 2D array of embedding vectors"""
        return np.array([self.embedding(s) for s in sentences])

    def transform_pairs(self, sentence_pairs):
        """Compute a feature vector for a sentence pair

        The default implementation is a concatenation of u*v and
        |u - v|, where u is the sentence embedding for the first
        sentence and v for the second sentence.

        This technique was introduced by Kai Sheng Tai, Richard Socher,
        Christopher D. Manning: "Improved Semantic Representations From
        Tree-Structured Long Short-Term Memory Networks"
        """

        embeddings1 = self.transform(sentence_pairs.iloc[:, 0])
        embeddings2 = self.transform(sentence_pairs.iloc[:, 1])

        if sparse.issparse(embeddings1):
            embeddings1 = np.asarray(embeddings1.todense())
        if sparse.issparse(embeddings2):
            embeddings2 = np.asarray(embeddings2.todense())

        return np.concatenate((
            np.multiply(embeddings1, embeddings2),
            np.abs(embeddings1 - embeddings2)
        ), axis=1)

    def embedding(self, sentence):
        """Transform a single sentence to an embedding vector"""
        raise NotImplementedError('embedding() must be implemented')

    def tokenize(self, sentence):
        """Split a sentence (a string) into a list of tokens"""
        for w in SentenceEmbeddingModel.split_whitespace_re.findall(sentence):
            yield w
