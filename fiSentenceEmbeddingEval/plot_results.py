import argparse
import math
import os.path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    df = load_results(args.resultdir)
    df = select_scores_to_plot(df)
    g = draw_plots(df, 'score_mean')
    save_plot(g, os.path.join(args.resultdir, 'scores.svg'))
    save_plot(g, os.path.join(args.resultdir, 'scores.png'))
    plt.close()

    g = draw_plots(df, 'train_duration_mean', 'Duration (s)')
    save_plot(g, os.path.join(args.resultdir, 'duration.svg'))
    save_plot(g, os.path.join(args.resultdir, 'duration.png'))
    plt.close()


def load_results(resultdir):
    filename = os.path.join(resultdir, 'scores.csv')
    return pd.read_csv(filename)


def select_scores_to_plot(df):
    main_score_labels = [
        ('Eduskunta-VKK', 'F1 score'),
        ('Opusparcus', "Pearson's coefficient"),
        ('TDT categories', 'F1 score'),
        ('Ylilauta', 'Accuracy'),
    ]

    task_selectors = [
        (df['task'] == x[0]) & (df['score_label'] == x[1])
        for x in main_score_labels
    ]

    selected = df[pd.concat(task_selectors, axis=1).any(axis=1)]

    for task, score_label in main_score_labels:
        task_rows = selected[(selected['task'] == task) &
                             (selected['score_label'] == score_label)]
        if task_rows.empty:
            print(f'Warning: No score {score_label} found for {task}!')

    return selected


def draw_plots(df, value_column, ylabel=None):
    models = df.model.unique()
    model_colors = ['C' + str(i + 1) for i in range(len(models))]
    task_score_labels = (
        df[['task', 'score_label']]
        .drop_duplicates()
        .to_dict('records'))
    task_score_labels = sorted(task_score_labels,
                               key=lambda x: (x['task'], x['score_label']))

    nrows = 2
    ncols = math.ceil(len(task_score_labels)/nrows)
    fig, ax = plt.subplots(nrows, ncols)
    if ax.ndim < 2:
        ax = ax[:, np.newaxis]

    fig.set_size_inches(8, 8)
    for i, labels in enumerate(task_score_labels):
        tdata = df[(df['task'] == labels['task']) &
                   (df['score_label'] == labels['score_label'])]
        y = [tdata[tdata['model'] == m][value_column].values[0]
             for m in models]

        a = ax[math.floor(i/ncols), i % ncols]
        a.bar(range(len(models)), y, tick_label=models, color=model_colors)
        a.set_title(labels['task'])
        a.set_xlabel('')
        a.set_ylabel(ylabel or labels['score_label'])
        a.set_xticklabels(a.get_xticklabels(), rotation=90)
        a.spines['top'].set_visible(False)
        a.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig


def save_plot(g, filename):
    plt.savefig(filename)
    print(f'Result plot saved to {filename}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resultdir', default='results',
                        help='Name of the directory where the results will '
                        'be saved')
    return parser.parse_args()


if __name__ == '__main__':
    main()
