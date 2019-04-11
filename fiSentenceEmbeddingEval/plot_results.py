import argparse
import os.path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    args = parse_args()
    df = load_results(args.resultdir)
    g = draw_plots(df, 'score_mean')
    save_plot(g, os.path.join(args.resultdir, 'scores.png'))

    g = draw_plots(df, 'train_duration_mean', 'Duration (s)')
    save_plot(g, os.path.join(args.resultdir, 'duration.png'))


def load_results(resultdir):
    filename = os.path.join(resultdir, 'scores.csv')
    return pd.read_csv(filename)


def draw_plots(df, ycol, ylabel=None):
    models = df.model.unique()
    model_colors = ['C' + str(i + 1) for i in range(len(models))]
    task_score_labels = {
        x['task']: x['score_label']
        for x in df[['task', 'score_label']]
            .drop_duplicates()
            .to_dict('records')
    }

    fig, ax = plt.subplots(1, len(task_score_labels))
    fig.set_size_inches(16, 6)
    for i, (t, task_ylabel) in enumerate(task_score_labels.items()):
        tdata = df[df['task'] == t]
        y = [tdata[tdata['model'] == m][ycol].values[0]
             for m in models]

        ax[i].bar(range(len(models)), y, tick_label=models, color=model_colors)
        ax[i].set_title(t)
        ax[i].set_xlabel('')
        ax[i].set_ylabel(ylabel or task_ylabel)
        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=90)
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

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
