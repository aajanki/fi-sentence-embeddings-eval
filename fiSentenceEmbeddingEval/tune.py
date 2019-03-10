import json
from hyperopt import fmin, tpe, Trials, STATUS_OK, hp
from voikko import libvoikko
from .models import *
from .tasks import *


def tune():
    voikko = libvoikko.Voikko('fi')
    evaluations = [
        {
            'embedding_model': Bert(
                'BERT multilingual',
                'pretrained/bert/multi_cased_L-12_H-768_A-12', 1
            ),
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 50, 768, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 30, 300, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0, 0.7),
            }
        },
        {
            'embedding_model': TfidfVectors('TF-IDF', voikko),
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 50, 1000, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 30, 300, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0, 0.7),
            }
        },
        {
            'embedding_model': SIF(
                'SIF',
                'data/finnish_vocab/finnish_vocab.txt.gz',
                'pretrained/fin-word2vec/fin-word2vec.bin'
            ),
            'space': {
                'hidden_dim1': hp.quniform('hidden_dim1', 10, 300, 10),
                'hidden_dim2': hp.quniform('hidden_dim2', 10, 100, 5),
                'dropout_prop': hp.uniform('dropout_prop', 0, 0.7),
            }
        },
    ]
    task = TDTCategoryClassificationTask('TDT categories',
                                         'data/UD_Finnish-TDT',
                                         use_dev_set=True)
    
    outf = open('results/hyperparameters.jsonl', 'w')
    for kv in evaluations:
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
        best['model'] = kv['embedding_model'].name

        print('best')
        print(best)

        outf.write(json.dumps(best))
        outf.write('\n')
        outf.flush()


if __name__ == '__main__':
    tune()
