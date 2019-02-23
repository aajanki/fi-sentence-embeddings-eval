import re
import numpy as np


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

    def embedding(self, sentence):
        """Transform a single sentence to an embedding vector"""
        raise NotImplementedError('embedding() must be implemented')

    def tokenize(self, sentence):
        """Split a sentence (a string) into a list of tokens"""
        for w in SentenceEmbeddingModel.split_whitespace_re.findall(sentence):
            yield w
