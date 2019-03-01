import argparse
import os.path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    df = load_results(args.resultdir)
    g = plot_facets(df)
    save_plot(g, args.resultdir)


def load_results(resultdir):
    filename = os.path.join(resultdir, 'scores.csv')
    return pd.read_csv(filename)


def plot_facets(df):
    models = df.model.unique()
    g = sns.FacetGrid(df, col='task', hue='model', sharey=False,
                      height=4, aspect=0.6)
    g.map(sns.barplot, 'model', 'score_mean', order=models)
    g.set_xlabels('')
    g.set_ylabels('Score')
    g.set_xticklabels(rotation=90)
    plt.tight_layout()
    return g


def save_plot(g, resultdir):
    filename = os.path.join(resultdir, 'scores.png')
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
