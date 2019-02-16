import re
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from voikko import libvoikko
from .preprocess import load_UD, source_type_percentages
from .models.tfidf import TfidfVectors
from .models.pooled_word2vec import PooledWord2Vec


def main():
    voikko = libvoikko.Voikko('fi')
    
    df_train, df_test = load_UD('data/UD_Finnish-TDT')

    print(f'{df_train.shape[0]} train samples')
    print(f'{df_test.shape[0]} test samples')
    print(f"{len(df_train['source_type'].unique())} classes")
    print('Class proportions:')
    print(source_type_percentages(df_train, df_test).to_string(
        float_format=zero_decimals))

    models = [
        TfidfVectors(voikko),
        PooledWord2Vec('data/fin-word2vec/fin-word2vec.bin')
    ]

    scores = []
    for model in models:
        print()
        print(f'*** {model.name} ***')
        print()

        train_features, test_features = \
            sentence_embeddings(model, df_train, df_test)

        clf = train_classifier(train_features, df_train['source_type'])
        score = evaluate(clf, test_features, df_test['source_type'])

        print(f'F1 {model.name}: {score:.2f}')

        scores.append((model.name, score))

    print('F1 score summary:')
    print(pd.DataFrame(scores).to_string(index=False, header=False,
                                         float_format=two_decimals))

def train_classifier(X, y):
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


def sentence_embeddings(embeddings_model, df_train, df_test):
    embeddings_model.fit(df_train['sentence'])

    print(embeddings_model.describe())

    features_train = embeddings_model.transform(df_train['sentence'])
    features_test = embeddings_model.transform(df_test['sentence'])
    return features_train, features_test


def zero_decimals(x):
    return f'{x:.0f}'


def two_decimals(x):
    return f'{x:.2f}'


if __name__ == '__main__':
    main()
