import json
import os
from hyperopt import fmin, tpe, Trials, hp
from voikko import libvoikko
from .models import *
from .tasks import *


def tune():
    voikko = libvoikko.Voikko('fi')
    task_tdt = TDTCategoryClassificationTask('TDT categories',
                                             'data/UD_Finnish-TDT',
                                             use_dev_set=True)
    task_opusparcus = OpusparcusTask('Opusparcus',
                                     'data/opusparcus/opusparcus_v1',
                                     use_dev_set=True)
    task_ylilauta = YlilautaConsecutiveSentencesTask('Ylilauta',
                                                     'data/ylilauta',
                                                     use_dev_set=True)
    model_w2v = PooledWord2Vec(
        'Pooled word2vec',
        'pretrained/fin-word2vec/fin-word2vec.bin')
    model_fasttest = PooledFastText(
        'Pooled FastText',
        'pretrained/fasttext-fi/cc.fi.300.bin')
    model_bert = Bert(
        'BERT multilingual',
        'pretrained/bert/multi_cased_L-12_H-768_A-12', [-3])
    model_tfidf = TfidfVectors('TF-IDF', voikko)
    model_sif = SIF(
        'SIF',
        'data/finnish_vocab/finnish_vocab.txt.gz',
        'pretrained/fin-word2vec/fin-word2vec.bin')
    model_borep = BOREP(
        'BOREP',
        'pretrained/fin-word2vec/fin-word2vec.bin',
        4096)
    evaluations = [
        {
            'task': task_tdt,
            'embedding_model': model_w2v,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 4, 100, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_tdt,
            'embedding_model': model_fasttest,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 4, 100, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_tdt,
            'embedding_model': model_bert,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 768, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 20, 300, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_tdt,
            'embedding_model': model_tfidf,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 20, 300, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_tdt,
            'embedding_model': model_sif,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 10, 100, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_tdt,
            'embedding_model': model_borep,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 20, 300, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_w2v,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_fasttest,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_bert,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 768, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_tfidf,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_sif,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_opusparcus,
            'embedding_model': model_borep,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_w2v,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_fasttest,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_bert,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 768, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_tfidf,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_sif,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
        {
            'task': task_ylilauta,
            'embedding_model': model_borep,
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 30, 1000, 10),
                'dropout_prop': hp.uniform('dropout_prop', 0.2, 0.8),
            }
        },
    ]

    os.makedirs('results', exist_ok=True)

    best_params = {}
    for kv in evaluations:
        task = kv['task']
        X_train, y_train, X_test, y_test = \
            task.prepare_data(kv['embedding_model'])

        def objective(space):
            print(kv['embedding_model'].name)
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
        print(f'best parameters for {kv["embedding_model"].name} in task {task.name}')
        print(best)

        best_params.setdefault(task.name, {})[kv['embedding_model'].name] = {
            'parameters': best,
            'score': -np.min(trials.losses())
        }

        with open('results/hyperparameters.json', 'w') as f:
            json.dump(best_params, f, indent=2)


if __name__ == '__main__':
    tune()
