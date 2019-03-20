import json
import os
import itertools
from hyperopt import fmin, tpe, Trials, hp
from voikko import libvoikko
from .models import *
from .tasks import *


def tune():
    voikko = libvoikko.Voikko('fi')
    tasks = [
        TDTCategoryClassificationTask('TDT categories',
                                      'data/UD_Finnish-TDT',
                                      use_dev_set=True),
        OpusparcusTask('Opusparcus',
                       'data/opusparcus/opusparcus_v1',
                       use_dev_set=True),
        YlilautaConsecutiveSentencesTask('Ylilauta',
                                         'data/ylilauta',
                                         use_dev_set=True)
    ]

    def model_w2v():
        return PooledWord2Vec(
            'Pooled word2vec',
            'pretrained/fin-word2vec/fin-word2vec.bin')

    def model_fasttext():
        return PooledFastText(
            'Pooled FastText',
            'pretrained/fasttext-fi/cc.fi.300.bin')

    def model_bert():
        return Bert(
            'BERT multilingual',
            'pretrained/bert/multi_cased_L-12_H-768_A-12', [-3])

    def model_tfidf():
        return TfidfVectors('TF-IDF', voikko)

    def model_sif():
        return SIF(
            'SIF',
            'data/finnish_vocab/finnish_vocab.txt.gz',
            'pretrained/fin-word2vec/fin-word2vec.bin')

    def model_borep():
        return BOREP(
            'BOREP',
            'pretrained/fin-word2vec/fin-word2vec.bin',
            4096)

    evaluations = itertools.chain(
        evaluations_for_model(model_w2v, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        }),
        evaluations_for_model(model_fasttext, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        }),
        evaluations_for_model(model_bert, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 30, 768, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        }),
        evaluations_for_model(model_tfidf, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        }),
        evaluations_for_model(model_sif, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        }),
        evaluations_for_model(model_borep, tasks, {
            'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
            'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
        })
    )

    os.makedirs('results', exist_ok=True)

    best_params = {}
    for kv in evaluations:
        task = kv['task']
        embedding_model = kv['embedding_model']
        X_train, y_train, X_test, y_test = \
            task.prepare_data(embedding_model)

        def objective(space):
            print(embedding_model.name)
            print(space)

            clf = task.train_classifier(X_train, y_train, space)
            f1 = task.compute_score(clf, X_test, y_test)
            return -f1

        trials = Trials()
        best = fmin(fn=objective,
                    space=kv['space'],
                    algo=tpe.suggest,
                    max_evals=40,
                    trials=trials)
        best_score = -np.min(trials.losses())
        print(f'best score for {embedding_model.name} in task {task.name}: {best_score}')
        print('parameters:')
        print(best)

        best_params.setdefault(task.name, {})[embedding_model.name] = {
            'parameters': best,
            'score': best_score
        }

        with open('results/hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)


def evaluations_for_model(embedding_model_builder, tasks, space):
    def inner():
        model = embedding_model_builder()
        return ({
            'task': task,
            'embedding_model': model,
            'space': space
        } for task in tasks)

    return itertools.chain.from_iterable(inner() for _ in range(1))


if __name__ == '__main__':
    tune()
