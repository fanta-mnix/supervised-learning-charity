import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['transpose',
           'plot_income_ratio_by',
           'pie_chart_for',
           'percentualize',
           'visualize_distribution_of',
           'plot_tsne',
           'most_important']


def transpose(nested_list):
    return zip(*nested_list)


def plot_income_ratio_by(data, column):
    assert column in data, "DataFrame does not have a column named '{}'".format(column)

    table = pd.crosstab(data[column], data['income'])
    col_sum = table.sum(axis=1)
    high_income_ratio = table.divide(col_sum, axis=0)['>50K']

    plt.figure(figsize=(16, 4))
    plt.xlabel(column)
    plt.ylabel('High Income Ratio')
    plt.xticks(np.arange(len(table.index)), table.index, rotation=90)
    plt.plot(np.arange(len(table.index)), high_income_ratio.values)
    plt.axhline(y=high_income_ratio.values.mean(), c='red')
    plt.grid()
    plt.show()


def pie_chart_for(data, column):
    assert column in data, "DataFrame does not have a column named '{}'".format(column)

    freqs = transpose(np.unique(data[column], return_counts=True))
    labels, counts = transpose(sorted(freqs, key=lambda x: x[1], reverse=True))

    patches, _ = plt.pie(counts)
    plt.legend(patches, labels, loc="best")
    plt.title(column)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def percentualize(arr):
    assert isinstance(arr, np.ndarray), 'Exoected a numpy array'
    return 100 * arr.astype(float) / arr.sum()


def visualize_distribution_of(data, column):
    assert column in data, "DataFrame does not have a column named '{}'".format(column)

    plt.title(column)
    plt.hist(data[column])
    plt.tight_layout()
    plt.show()


def plot_tsne(X, y=None, perplexity=30.0):
    from sklearn.manifold.t_sne import TSNE
    import scipy.sparse as sp

    print('Embedding data into 2D, takes some time...')
    if sp.issparse(X):
        X = X.todense()
    Xt = TSNE(perplexity=perplexity).fit_transform(X, y)

    if y is None:
        plt.plot(list(Xt[:, 0]), list(Xt[:, 1]), marker='o', linestyle='')
    else:
        df = pd.DataFrame({'x': Xt[:, 0], 'y': Xt[:, 1], 'label': y})
        groups = df.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(list(group.x), list(group.y), marker='o', linestyle='', label=name)

        ax.legend()

    plt.title('2D visualization of data')
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def most_important(learner, feature_names, n):
    assert len(feature_names) == len(learner.feature_importances_), 'Size mismatch for feature names and importances'
    return sorted(zip(feature_names, learner.feature_importances_), key=lambda x:x[1], reverse=True)[:n]