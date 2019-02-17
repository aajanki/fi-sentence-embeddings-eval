import re
import numpy as np
from gensim.models import FastText


_split_whitespace_re = re.compile(r"(?u)\b\w\w+\b")


class PooledFastText:
    def __init__(self, data_filename):
        self.name = 'Pooled FastText'
        self.model = FastText.load_fasttext_format(data_filename,
                                                   full_model=False)

    def fit(self, sentences):
        # nothing to do
        pass

    def describe(self):
        return '\n'.join([
            f'Embedding dimensionality: {self.model.vector_size}',
            f'Vocabulary size: {len(self.model.wv.vocab)}'
        ])

    def transform(self, sentences):
        return np.array([self.embedding(s) for s in sentences])

    def embedding(self, sentence):
        # FastText will generate embeddings even for out-of-vocabulary
        # words, therefore there is no need to handle them separately
        vecs = [self.model.wv[w] for w in _split_whitespace_re.findall(sentence)]
        if len(vecs) > 0:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(self.model.vector_size)
