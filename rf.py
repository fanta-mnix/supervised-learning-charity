import numpy as np
import pandas as pd

# from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer


def get_mean_score(score_tuple):
    return score_tuple.mean_validation_score


def get_parameters(score_tuple):
    return score_tuple.parameters


def best_scores(grid_search, n=10):
    top = sorted(grid_search.grid_scores_, key=lambda x: get_mean_score(x), reverse=True)[:n]
    return [(get_mean_score(entry), get_parameters(entry)) for entry in top]


def worst_scores(grid_search, n=10):
    bottom = sorted(grid_search.grid_scores_, key=lambda x: get_mean_score(x))[:n]
    return [(get_mean_score(entry), get_parameters(entry)) for entry in bottom]


data = pd.read_csv("census.csv")

income_raw = data['income']
features_raw = data.drop('income', axis=1)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# scaler = MinMaxScaler()
# numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
# features_raw[numerical] = scaler.fit_transform(data[numerical])

features = pd.get_dummies(features_raw)
income = (income_raw == '>50K').astype(int)


def most_important(rf, n=10):
    named_importances = zip(features.columns, rf.feature_importances_)
    return sorted(named_importances, key=lambda x: x[1], reverse=True)[:n]


scorer = make_scorer(fbeta_score, beta=0.5)

param_grid = {
}

folds = StratifiedKFold(income, 5, True, 42)
grid_clf = GridSearchCV(RandomForestClassifier(120), param_grid, scorer, n_jobs=1, cv=folds, verbose=3)
grid_clf.fit(features, income)

most_important(grid_clf.best_estimator_)


# [(0.6667743809193829, {'metric': 'manhattan', 'n_neighbors': 15}),
#  (0.6657641583102326, {'metric': 'manhattan', 'n_neighbors': 10}),
#  (0.6654424371525302, {'metric': 'euclidean', 'n_neighbors': 15}),
#  (0.6644493964009611, {'metric': 'euclidean', 'n_neighbors': 10}),
#  (0.6432909758153206, {'metric': 'manhattan', 'n_neighbors': 5}),
#  (0.6405380507795196, {'metric': 'euclidean', 'n_neighbors': 5}),
#  (0.6017199020171297, {'metric': 'manhattan', 'n_neighbors': 2}),
#  (0.6010725593824304, {'metric': 'euclidean', 'n_neighbors': 2}),
#  (0.5837037757781743, {'metric': 'manhattan', 'n_neighbors': 1}),
#  (0.5833985324541671, {'metric': 'euclidean', 'n_neighbors': 1})]