import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ['transpose', 'pie_chart_for', 'percentualize', 'visualize_distribution_of', 'plot_tsne']


def transpose(nested_list):
    return zip(*nested_list)


def pie_chart_for(data, column):
    freqs = transpose(np.unique(data[column], return_counts=True))
    labels, counts = transpose(sorted(freqs, key=lambda x: x[1], reverse=True))

    patches, _ = plt.pie(counts)
    plt.legend(patches, labels, loc="best")
    plt.title(column)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def percentualize(arr):
    return 100 * arr.astype(float) / arr.sum()


def visualize_distribution_of(data, column):
    plt.title(column)
    plt.hist(data[column])
    plt.tight_layout()
    plt.show()


def plot_tsne(plt, X, y=None, perplexity=30.0):
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
