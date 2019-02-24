import pandas as pd
import seaborn as sb
from matplotlib import pyplot as plt
from scipy import stats


def plot_pmi_distribution():
    filename = '../data/opusparcus/opusparcus_v1/fi/train/fi-train.txt.bz2'
    names = ['id', 'sentence1', 'sentence2', 'total_pmi',
             'expected_back_translations', 'lang_common_translations',
             'edit_distance']

    df = pd.read_csv(filename, sep='\t', header=None, names=names)
    df = df[df['edit_distance'] > 5]
    df = df.sample(n=100000, random_state=42).reset_index()

    qs = stats.mstats.mquantiles(df['total_pmi'], [0, 0.3, 0.5, 0.8, 0.95, 1])
    qs[-1] = 1000
    ranges = list(zip(qs, qs[1:]))
    for r in ranges:
        X = df[(df['total_pmi'] >= r[0]) & (df['total_pmi'] < r[1])]

        print(f'Range: {r}')
        print(X[['sentence1', 'sentence2', 'total_pmi']].sample(10).to_string(index=False))
        print('-'*80)

    sb.distplot(df['total_pmi'],kde=False)
    plt.show()


if __name__ == '__main__':
    plot_pmi_distribution()
