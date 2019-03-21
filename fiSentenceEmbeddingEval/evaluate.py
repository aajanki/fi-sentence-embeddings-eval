import argparse
import json
import os
import os.path
import pandas as pd
import numpy as np
from voikko import libvoikko
from .models import *
from .tasks import *


class Hyperparameters:
    def __init__(self, filename):
        if filename:
            self.hyperparameters = json.load(open(filename, 'r'))
        else:
            self.hyperparameters = {}

    def set_logreg(self):
        self.hyperparameters['logreg'] = True

    def get(self, task_name, model_name):
        return self.hyperparameters.get(task_name, {}).get(model_name, {})


def main():
    voikko = libvoikko.Voikko('fi')
    args = parse_args()
    hyperparameters = Hyperparameters(args.hyperparameters)
    if args.fast:
        hyperparameters.set_logreg()

    tasks = [
        TDTCategoryClassificationTask('TDT categories', 'data/UD_Finnish-TDT',
                                      use_dev_set=args.dev_set,
                                      verbose=args.verbose),
        OpusparcusTask('Opusparcus', 'data/opusparcus/opusparcus_v1',
                      use_dev_set=args.dev_set, verbose=args.verbose),
        YlilautaConsecutiveSentencesTask('Ylilauta', 'data/ylilauta',
                                         use_dev_set=args.dev_set,
                                         verbose=args.verbose),
    ]

    models = [
        TfidfVectors('TF-IDF', voikko),
        PooledWord2Vec('Pooled word2vec',
                       'pretrained/fin-word2vec/fin-word2vec.bin'),
        PooledFastText('Pooled FastText',
                       'pretrained/fasttext-fi/cc.fi.300.bin'),
        SIF('SIF',
            'data/finnish_vocab/finnish_vocab.txt.gz',
            'pretrained/fin-word2vec/fin-word2vec.bin'),
        BOREP('BOREP',
              'pretrained/fin-word2vec/fin-word2vec.bin', 4096),
        Bert('BERT multilingual',
             'pretrained/bert/multi_cased_L-12_H-768_A-12', [-3]),
    ]

    print(f'Running evaluation on {len(tasks)} tasks and {len(models)} models')
    
    scores = []
    for k in range(args.num_trials):
        if args.num_trials > 1:
            print(f'Trial {k+1}/{args.num_trials}')

        scores.append(evaluate_models(models, tasks, hyperparameters))

    scores = (pd.concat(scores)
              .groupby(['task', 'score_label', 'model'])
              .agg([np.mean, np.std]))

    print('F1 score summary:')
    print(scores.to_string(float_format=two_decimals))

    save_results(scores, args.resultdir)


def evaluate_models(models, tasks, hyperparameters):
    scores = []
    for task in tasks:
        for model in models:
            print()
            print(f'*** Task: {task.name}, model: {model.name} ***')
            print()

            hyp = hyperparameters.get(task.name, model.name)
            print(json.dumps(hyp))

            score, duration = task.evaluate(model, hyp)

            scores.append({
                'task': task.name,
                'score_label': task.score_label,
                'model': model.name,
                'score': score,
                'train_duration': duration
            })

    return pd.DataFrame(scores)


def save_results(scores, resultdir):
    os.makedirs(resultdir, exist_ok=True)
    filename = os.path.join(resultdir, 'scores.csv')
    flattened = scores.reset_index()
    flattened.columns = ['_'.join(x).rstrip('_')
                         for x in flattened.columns.values]
    flattened.to_csv(filename, index=False)


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
    parser.add_argument('--fast', action='store_true',
                        help='Use simpler final classifier. Faster but less '
                        'accurate. Good for debugging')
    parser.add_argument('--hyperparameters',
                        help='Input file that contains the hyperparameters.')
    parser.add_argument('--verbose', action='store_true',
                        help='Show verbose output')
    parser.add_argument('--resultdir', default='results',
                        help='Name of the directory where the results will '
                        'be saved')
    return parser.parse_args()


if __name__ == '__main__':
    main()
