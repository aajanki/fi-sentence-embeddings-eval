import argparse
import os
import os.path
import pandas as pd
import numpy as np
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from voikko import libvoikko
from .preprocess import load_UD, source_type_percentages
from .models import *


def main():
    voikko = libvoikko.Voikko('fi')
    args = parse_args()
    
    df_train, df_test = load_UD('data/UD_Finnish-TDT', args.dev_set)
    print_data_summary(df_train, df_test)

    models = [
        TfidfVectors(voikko),
        PooledWord2Vec('pretrained/fin-word2vec/fin-word2vec.bin'),
        PooledFastText('pretrained/fasttext-fi/cc.fi.300.bin'),
        SIF('data/finnish_vocab/finnish_vocab.txt.gz',
            'pretrained/fin-word2vec/fin-word2vec.bin'),
        BOREP('pretrained/fin-word2vec/fin-word2vec.bin', 4096),
        Bert('pretrained/bert/multi_cased_L-12_H-768_A-12', 1),
    ]

    scores = []
    for k in range(args.num_trials):
        if args.num_trials > 1:
            print(f'Trial {k+1}/{args.num_trials}')

        scores.append(evaluate_models(models, df_train, df_test, args.logreg))
    scores = pd.concat(scores).groupby('model').mean()

    print('F1 score summary:')
    print(scores.to_string(float_format=two_decimals))

    save_results(scores, args.resultdir)


def print_data_summary(df_train, df_test):
    print(f'{df_train.shape[0]} train samples')
    print(f'{df_test.shape[0]} test samples')
    print(f"{len(df_train['source_type'].unique())} classes")
    print('Class proportions:')
    print(source_type_percentages(df_train, df_test).to_string(
        float_format=zero_decimals))


def evaluate_models(models, df_train, df_test, logreg):
    scores = []
    for model in models:
        print()
        print(f'*** {model.name} ***')
        print()

        train_features, test_features = \
            sentence_embeddings(model, df_train, df_test)

        clf = train_classifier(train_features, df_train['source_type'], logreg)
        score = evaluate(clf, test_features, df_test['source_type'])

        print(f'F1 {model.name}: {score:.2f}')

        scores.append((model.name, score))

    return pd.DataFrame(scores, columns=['model', 'score'])


def train_classifier(X, y, logreg):
    scaler = StandardScaler(with_mean=not sparse.issparse(X))

    if logreg:
        clf = LogisticRegression(multi_class='multinomial',
                                 solver='lbfgs',
                                 max_iter=1000)
    else:
        clf = KerasClassifier(build_fn=nn_classifier_model,
                              input_dim=X.shape[1],
                              num_classes=len(np.unique(y)),
                              epochs=200,
                              batch_size=8,
                              verbose=0)
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', clf)
    ])
    pipeline.fit(X, y)
    return pipeline


def nn_classifier_model(input_dim=300, num_classes=3):
    model = Sequential()
    model.add(Dropout(0.3, input_shape=(input_dim, )))
    model.add(Dense(128, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='tanh'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


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


def save_results(scores, resultdir):
    os.makedirs(resultdir, exist_ok=True)
    filename = os.path.join(resultdir, 'scores.csv')
    scores.to_csv(filename)


def zero_decimals(x):
    return f'{x:.0f}'


def two_decimals(x):
    return f'{x:.2f}'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-trials', type=int, default=1,
                        help='Number of times the training is repeated on '
                        'random intialization. The final scores are '
                        'averages of the trials')
    parser.add_argument('--dev-set', action='store_true',
                        help='Evaluate on the development set')
    parser.add_argument('--logreg', action='store_true',
                        help='Use logistic regression as the final classifier')
    parser.add_argument('--resultdir', default='results',
                        help='Name of the directory where the results will '
                        'be saved')
    return parser.parse_args()


if __name__ == '__main__':
    main()
