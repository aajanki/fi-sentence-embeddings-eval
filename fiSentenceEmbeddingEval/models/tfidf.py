import re
from sklearn.feature_extraction.text import TfidfVectorizer
from .sentenceembedding import SentenceEmbeddingModel


class TfidfVectors(SentenceEmbeddingModel):
    def __init__(self, name, voikko):
        super().__init__(name)
        tokenizer = self.build_voikko_tokenizer(voikko)
        self.vectorizer = TfidfVectorizer(lowercase=True,
                                          tokenizer=tokenizer,
                                          ngram_range=(1, 2),
                                          min_df=4,
                                          max_features=20000)

    def fit(self, sentences):
        self.vectorizer.fit(sentences)

    def describe(self):
        return f'Vocabulary size: {len(self.vectorizer.vocabulary_)}'

    def transform(self, sentences):
        return self.vectorizer.transform(sentences)

    def build_voikko_tokenizer(self, voikko):
        split_whitespace_re = re.compile(r"(?u)\b\w\w+\b")

        def tokenizer(text):
            tokens = []
            for w in split_whitespace_re.findall(text):
                analyzed = voikko.analyze(w)

                if analyzed:
                    token = analyzed[0].get('BASEFORM', w)
                else:
                    token = w

                tokens.append(token.lower())

            return tokens

        return tokenizer
