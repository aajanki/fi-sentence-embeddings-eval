import argparse
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    df = load_results(args.resultdir)
    g = plot_facets(df, 'score_mean', 'Score')
    save_plot(g, os.path.join(args.resultdir, 'scores.png'))

    g = plot_facets(df, 'train_duration_mean', 'Duration (s)')
    save_plot(g, os.path.join(args.resultdir, 'duration.png'))


def load_results(resultdir):
    filename = os.path.join(resultdir, 'scores.csv')
    return pd.read_csv(filename)


def plot_facets(df, y, ylabel):
    models = df.model.unique()
    g = sns.FacetGrid(df, col='task', hue='model', sharey=False,
                      height=4, aspect=0.6)
    g.map(sns.barplot, 'model', y, order=models)
    g.set_xlabels('')
    g.set_ylabels(ylabel)
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    return g


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
