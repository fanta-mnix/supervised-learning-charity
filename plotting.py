import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("census.csv")

income_raw = data['income']
features_raw = data.drop('income', axis = 1)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

features = pd.get_dummies(features_raw)
income = (income_raw == '>50K').astype(int)


def plot_tsne(plt, X, y=None, perplexity=30.0):
    from sklearn.manifold.t_sne import TSNE
    import scipy.sparse as sp

    print('Projecting...')
    if sp.issparse(X):
        X = X.todense()
    Xt = TSNE(perplexity=perplexity).fit_transform(X, y)

    print('Plotting...')
    if y is None:
        plt.plot(list(Xt[:, 0]), list(Xt[:, 1]), marker='o', linestyle='')
    else:
        df = pd.DataFrame({'x': Xt[:, 0], 'y': Xt[:, 1], 'label': y})
        groups = df.groupby('label')

        fig, ax = plt.subplots()
        for name, group in groups:
            ax.plot(list(group.x), list(group.y), marker='o', linestyle='', label=name)

        ax.legend()


indices = np.random.choice(np.arange(len(features)), 2000, replace=False)
plot_tsne(plt, features.iloc[indices], income[indices])
plt.show()
