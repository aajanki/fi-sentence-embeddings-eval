import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from voikko import libvoikko
from .preprocess import load_UD, source_type_percentages


def main():
    voikko = libvoikko.Voikko('fi')
    
    df_train, df_test = load_UD('data/UD_Finnish-TDT')

    print(f'{df_train.shape[0]} train samples')
    print(f'{df_test.shape[0]} test samples')
    print(f"{len(df_train['source_type'].unique())} classes")
    print('Class proportions:')
    print(source_type_percentages(df_train, df_test))

    tfidf_train, tfidf_test = tfidf_features(df_train, df_test, voikko)

    tfidf_model = train(tfidf_train, df_train['source_type'])
    tfidf_score = evaluate(tfidf_model, tfidf_test, df_test['source_type'])

    print('F1 scores:')
    print(f'TFIDF: {tfidf_score:.2f}')


def train(X, y):
    clf = LogisticRegression(multi_class='multinomial',
                             solver='lbfgs',
                             max_iter=1000)
    clf.fit(X, y)
    return clf


def evaluate(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print('Confusion matrix')
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                       columns=clf.classes_, index=clf.classes_))

    return f1_score(y_test, y_pred, average='micro')


def tfidf_features(df_train, df_test, voikko):
    tokenizer = build_voikko_tokenizer(voikko)
    vectorizer = TfidfVectorizer(lowercase=True,
                                 tokenizer=tokenizer,
                                 max_features=20000)
    features_train = vectorizer.fit_transform(df_train['sentence'])

    print('TFIDF vectorizer summary:')
    print(f'Vocabulary size: {len(vectorizer.vocabulary_)}')

    features_test = vectorizer.transform(df_test['sentence'])

    return(features_train, features_test)


def build_voikko_tokenizer(voikko):
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


if __name__ == '__main__':
    main()
