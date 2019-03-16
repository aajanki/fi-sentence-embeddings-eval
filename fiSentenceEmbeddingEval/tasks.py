import os.path
import numpy as np
import pandas as pd
from keras import regularizers
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .preprocess import load_UD, source_type_percentages


def zero_decimals(x):
    return f'{x:.0f}'


def sentence_pair_features(embeddings, sentences1, sentences2):
    """Compute a feature vector for a sentence pair

    The vector is a concatenation of u*v and |u - v|, where u is the
    sentence embedding for the first sentence and v for the second
    sentence.

    This technique was introduced by Kai Sheng Tai, Richard Socher,
    Christopher D. Manning: "Improved Semantic Representations From
    Tree-Structured Long Short-Term Memory Networks"
    """
    
    embeddings1 = embeddings.transform(sentences1)
    embeddings2 = embeddings.transform(sentences2)

    if sparse.issparse(embeddings1):
        embeddings1 = np.asarray(embeddings1.todense())
    if sparse.issparse(embeddings2):
        embeddings2 = np.asarray(embeddings2.todense())

    return np.concatenate((
        np.multiply(embeddings1, embeddings2),
        np.abs(embeddings1 - embeddings2)
    ), axis=1)


class TDTCategoryClassificationTask:
    """Category classification

    The data contains sentences from many Internet sites: blogs,
    Wikinews, Europarl, student magazine articles, etc.

    The task is to predict the original source of a sentence.

    Data reference: Turku Dependency Treebank (TDT) 2.2
    https://universaldependencies.org/treebanks/fi_tdt/index.html
    """

    def __init__(self, name, datadir, use_dev_set=False, verbose=False):
        self.name = name
        self.score_label = 'F1 score'
        self.verbose = verbose
        self.df_train, self.df_test = load_UD(datadir, use_dev_set)
        self.print_data_summary(self.df_train, self.df_test)

    def prepare_data(self, embeddings):
        X_train, X_test = \
            self.sentence_embeddings(embeddings, self.df_train, self.df_test)

        y_train = self.df_train['source_type']
        y_test = self.df_test['source_type']

        return X_train, y_train, X_test, y_test

    def evaluate(self, embeddings, hyperparameters):
        X_train, y_train, X_test, y_test = self.prepare_data(embeddings)
        clf = self.train_classifier(X_train, y_train, hyperparameters)
        return self.compute_score(clf, X_test, y_test)

    def train_classifier(self, X, y, params):
        scaler = StandardScaler(with_mean=not sparse.issparse(X))

        if params.get('logreg', False):
            clf = LogisticRegression(multi_class='multinomial',
                                     solver='lbfgs',
                                     max_iter=1000)
        else:
            nnparams = params.copy()
            if 'logreg' in nnparams:
                del nnparams['logreg']
            clf = KerasClassifier(build_fn=self.nn_classifier_model,
                                  input_dim=X.shape[1],
                                  num_classes=len(np.unique(y)),
                                  epochs=200,
                                  batch_size=8,
                                  verbose=1 if self.verbose else 0,
                                  **nnparams)
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', clf)
        ])
        pipeline.fit(X, y)
        return pipeline

    def nn_classifier_model(self, input_dim=300, num_classes=3,
                            hidden_dim1=128, hidden_dim2=32,
                            dropout_prop=0.3):
        model = Sequential()
        model.add(Dropout(dropout_prop, input_shape=(input_dim, )))
        model.add(Dense(int(hidden_dim1), activation='tanh'))
        model.add(Dropout(dropout_prop))
        model.add(Dense(int(hidden_dim2), activation='tanh'))
        model.add(Dropout(dropout_prop))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        if self.verbose:
            print(model.summary())

        return model

    def compute_score(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test)

        print(classification_report(y_test, y_pred))
        print('Confusion matrix')
        print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                           columns=clf.classes_, index=clf.classes_))

        f1 = f1_score(y_test, y_pred, average='micro')
        print(f'F1 score: {f1:.2f}')

        return f1

    def sentence_embeddings(self, embeddings, df_train, df_test):
        embeddings.fit(df_train['sentence'])

        print(embeddings.describe())

        features_train = embeddings.transform(df_train['sentence'])
        features_test = embeddings.transform(df_test['sentence'])
        return features_train, features_test

    def print_data_summary(self, df_train, df_test):
        print(self.name)
        print(f'{df_train.shape[0]} train samples')
        print(f'{df_test.shape[0]} test samples')
        print(f"{len(df_train['source_type'].unique())} classes")
        print('Class proportions:')
        print(source_type_percentages(df_train, df_test).to_string(
            float_format=zero_decimals))
        print()


class OpusparcusTask:
    """Paraphrase detection

    The data consists of movie subtitles. Multiple sets of subtitles
    per movie provide potential paraphrases. The similarity of
    subtitles occurring at the same time code are assessed by hand
    (test set) or by statistical methods (training set).

    The learning task is to predict if a pair of sentences is a
    paraphrase.

    Data reference: Mathias Creutz: Open Subtitles Paraphrase Corpus
    for Six Languages, LREC 2018
    """
    
    def __init__(self, name, datadir, num_sample=10000, use_dev_set=False,
                 verbose=False):
        self.name = name
        self.score_label = "Pearson's coefficient"
        self.verbose = verbose

        train_filename = os.path.join(datadir, 'fi/train/fi-train.txt.bz2')
        self.df_train = self.load_train_data(train_filename, num_sample)

        if use_dev_set:
            test_filename = os.path.join(datadir, 'fi/dev/fi-dev.txt')
        else:
            test_filename = os.path.join(datadir, 'fi/test/fi-test.txt')
        self.df_test = self.load_test_data(test_filename)

        self.print_data_summary(self.df_train, self.df_test)

    def prepare_data(self, embeddings):
        all_sentences = (self.df_train['sentence1']
                         .append(self.df_train['sentence2']))
        embeddings.fit(all_sentences)

        X_train = sentence_pair_features(embeddings,
                                         self.df_train['sentence1'],
                                         self.df_train['sentence2'])
        y_train = self.train_class_probabilities(self.df_train)

        X_test = sentence_pair_features(embeddings,
                                        self.df_test['sentence1'],
                                        self.df_test['sentence2'])
        y_test = self.df_test['score']

        return X_train, y_train, X_test, y_test
        
    def evaluate(self, embeddings, hyperparameters):
        X_train, y_train, X_test, y_test = self.prepare_data(embeddings)
        clf = self.train_classifier(X_train, y_train, hyperparameters)
        return self.compute_score(clf, X_test, y_test)

    def train_classifier(self, X, y, params):
        nnparams = params.copy()
        if 'logreg' in nnparams:
            del nnparams['logreg']

        clf = KerasClassifier(self.build_classifier,
                              input_dim=X.shape[1],
                              num_classes=y.shape[1],
                              epochs=200,
                              batch_size=8,
                              verbose=1 if self.verbose else 0,
                              **nnparams)
        clf.fit(X, y)

        return clf

    def compute_score(self, clf, X_test, y_test):
        y_proba = clf.predict_proba(X_test)
        y_pred = y_proba.dot(np.arange(1, 6))
        corr = np.corrcoef(y_test, y_pred)[0, 1]

        print(f'Correlation: {corr:.2f}')
        
        return corr

    def train_class_probabilities(self, df):
        # Split the total_pmi variable in 5 bins. (The bin boundaries
        # are chosen quite arbitrarily.)
        qs = stats.mstats.mquantiles(df['total_pmi'], [0, 0.2, 0.5, 0.8, 0.95, 1])
        qs[-1] = 1000
        ps = np.zeros((df.shape[0], 5))
        for i, q in enumerate(zip(qs, qs[1:])):
            # Apply some label smoothing because the total_pmi values
            # are noisy
            if  i == 0:
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i] = 0.95
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i+1] = 0.05
            elif i == 4:
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i-1] = 0.05
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i] = 0.95
            else:
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i-1] = 0.1
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i] = 0.8
                ps[(q[0] <= df['total_pmi']) & (df['total_pmi'] < q[1]), i+1] = 0.1

        return ps

    def build_classifier(self, input_dim, num_classes,
                         hidden_dim1=64, l2reg=1e-4):
        model = Sequential()
        model.add(Dense(int(hidden_dim1),
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(l2reg),
                        bias_regularizer=regularizers.l2(l2reg),
                        input_shape=(input_dim, )))
        model.add(Dense(num_classes,
                        activation='softmax',
                        kernel_regularizer=regularizers.l2(l2reg),
                        bias_regularizer=regularizers.l2(l2reg)))
        model.compile(loss='kullback_leibler_divergence',
                      optimizer='adam')
        if self.verbose:
            print(model.summary())

        return model

    def load_train_data(self, filename, num_sample):
        names = ['id', 'sentence1', 'sentence2', 'total_pmi',
                 'expected_back_translations', 'lang_common_translations',
                 'edit_distance']
        df = pd.read_csv(filename, sep='\t', header=None, names=names)
        df = df[df['edit_distance'] > 5]
        df = df.sample(n=num_sample, random_state=42).reset_index()
        return df[['sentence1', 'sentence2', 'total_pmi']]

    def load_test_data(self, filename):
        names = ['id', 'sentence1', 'sentence2', 'score']
        return pd.read_csv(filename, sep='\t', header=None, names=names)

    def print_data_summary(self, df_train, df_test):
        print(self.name)
        print(f'{df_train.shape[0]} train samples')
        print(f'{df_test.shape[0]} test samples')
        print()


class YlilautaConsecutiveSentencesTask:
    """Predict if two sentence were originally consecutive or not

    The data is messages from the discussion forum Ylilauta.

    Data reference: http://urn.fi/urn:nbn:fi:lb-2016101210
    """

    def __init__(self, name, datadir, use_dev_set=False, verbose=False):
        self.name = name
        self.score_label = 'Accuracy'
        self.verbose = verbose

        train_filename = os.path.join(datadir, 'train.tab')
        self.df_train = self.load_data(train_filename)

        if use_dev_set:
            test_filename = os.path.join(datadir, 'dev.tab')
        else:
            test_filename = os.path.join(datadir, 'test.tab')
        self.df_test = self.load_data(test_filename)

        self.print_data_summary(self.df_train, self.df_test)

    def prepare_data(self, embeddings):
        all_sentences = (self.df_train['sentence1']
                         .append(self.df_train['sentence2']))
        embeddings.fit(all_sentences)

        X_train = sentence_pair_features(embeddings,
                                   self.df_train['sentence1'],
                                   self.df_train['sentence2'])
        y_train = (self.df_train['label'] == 1).astype(int)

        X_test = sentence_pair_features(embeddings,
                                        self.df_test['sentence1'],
                                        self.df_test['sentence2'])
        y_test = (self.df_test['label'] == 1).astype(int)

        return X_train, y_train, X_test, y_test
        
    def evaluate(self, embeddings, hyperparameters):
        X_train, y_train, X_test, y_test = self.prepare_data(embeddings)
        clf = self.train_classifier(X_train, y_train, hyperparameters)
        return self.compute_score(clf, X_test, y_test)

    def train_classifier(self, X, y, params):
        nnparams = params.copy()
        if 'logreg' in nnparams:
            del nnparams['logreg']

        clf = KerasClassifier(self.build_classifier,
                              input_dim=X.shape[1],
                              epochs=200,
                              batch_size=8,
                              verbose=1 if self.verbose else 0,
                              **nnparams)
        clf.fit(X, y)
        return clf

    def build_classifier(self, input_dim, hidden_dim1=64, l2reg=1e-5):
        model = Sequential()
        model.add(Dense(int(hidden_dim1),
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(l2reg),
                        bias_regularizer=regularizers.l2(l2reg),
                        input_shape=(input_dim, )))
        model.add(Dense(1,
                        activation='sigmoid',
                        kernel_regularizer=regularizers.l2(l2reg),
                        bias_regularizer=regularizers.l2(l2reg)))
        model.compile(loss='binary_crossentropy', optimizer='adam',
                      metrics=['accuracy'])
        if self.verbose:
            print(model.summary())

        return model

    def compute_score(self, clf, X_test, y_test):
        y_pred = clf.predict(X_test).squeeze()
        test_acc = np.mean(y_test == y_pred)

        print(f'Accuracy: {test_acc:.2f}')

        return test_acc

    def load_data(self, filename):
        return pd.read_csv(filename, sep='\t', header=0)

    def print_data_summary(self, df_train, df_test):
        print(self.name)
        print(f'{df_train.shape[0]} train samples')
        print(f'{df_test.shape[0]} test samples')
        print()
