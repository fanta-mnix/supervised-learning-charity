import numpy as np
import pandas as pd

# from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import accuracy_score, fbeta_score, make_scorer
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("census.csv")

# table = pd.crosstab(data['native-country'], data['income'])
# col_sum = table.sum(axis=1)
# high_income_ratio = table.divide(col_sum,axis=0)['>50K'].sort_values()
# poor_countries = set(high_income_ratio[high_income_ratio < 0.225].index)
# data['poor_country'] = data['native-country'].map(lambda x: x in poor_countries)

income_raw = data['income']
# features_raw = data.drop(['income', 'native-country'], axis=1)
features_raw = data.drop(['income', 'education_level', 'native-country'], axis=1)

skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

features = pd.get_dummies(features_raw)
income = (income_raw == '>50K').astype(int)

scorer = make_scorer(fbeta_score, beta=0.5)

param_grid = {
    'n_neighbors': [2, 4, 8, 16],
    'metric': ['manhattan', 'euclidean']
}

learner = KNeighborsClassifier(algorithm='auto')

folds = StratifiedKFold(income, 5, True, 42)

grid_clf = GridSearchCV(learner, param_grid, scorer, n_jobs=1, cv=folds, verbose=3)
grid_clf.fit(features, income)

# [(0.676058456332389,
#   {'kneighborsclassifier__metric': 'manhattan',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 10,
#    'pca__whiten': True}),
#  (0.673659799539837,
#   {'kneighborsclassifier__metric': 'manhattan',
#    'kneighborsclassifier__n_neighbors': 15,
#    'pca__n_components': 10,
#    'pca__whiten': True}),
#  (0.6713354687671972,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 10,
#    'pca__whiten': True}),
#  (0.6707042739579605,
#   {'kneighborsclassifier__metric': 'manhattan',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 25,
#    'pca__whiten': True}),
#  (0.6695902082631678,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 25,
#    'pca__whiten': True}),
#  (0.6692942984426612,
#   {'kneighborsclassifier__metric': 'manhattan',
#    'kneighborsclassifier__n_neighbors': 15,
#    'pca__n_components': 25,
#    'pca__whiten': True}),
#  (0.6688630929015739,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 15,
#    'pca__n_components': 10,
#    'pca__whiten': True}),
#  (0.6666811904669329,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 15,
#    'pca__n_components': 25,
#    'pca__whiten': True}),
#  (0.6593575521246106,
#   {'kneighborsclassifier__metric': 'manhattan',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 50,
#    'pca__whiten': True}),
#  (0.6586928654269536,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 10,
#    'pca__n_components': 50,
#    'pca__whiten': True})]


# [(0.5687737818684054,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 75,
#    'pca__whiten': False}),
#  (0.5693535692893172,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 50,
#    'pca__whiten': False}),
#  (0.5700735625207975,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 25,
#    'pca__whiten': False}),
#  (0.570633817616456,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 10,
#    'pca__whiten': False}),
#  (0.5712923201402436,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 75,
#    'pca__whiten': True}),
#  (0.5771492445333235,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 50,
#    'pca__whiten': True}),
#  (0.5822589130087794,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 25,
#    'pca__whiten': True}),
#  (0.5896769768230536,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 1,
#    'pca__n_components': 10,
#    'pca__whiten': True}),
#  (0.5988150993990716,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 2,
#    'pca__n_components': 75,
#    'pca__whiten': True}),
#  (0.5992246633108541,
#   {'kneighborsclassifier__metric': 'euclidean',
#    'kneighborsclassifier__n_neighbors': 2,
#    'pca__n_components': 10,
#    'pca__whiten': False})]