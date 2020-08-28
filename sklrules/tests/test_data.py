import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer
from sklrules import OneHotEncoder, RuleNetworkClassifier, TypeSelector

pipe = make_pipeline(
    TypeSelector(np.number, False),
    OneHotEncoder(sparse=False, handle_unknown='ignore'),
)


def test():
    datasets = ['adult', 'airline-passenger-satisfaction', 'alpha-bank', 'bank',
                'breast-cancer', 'credit-g', 'hepatitis', 'kr-vs-kp',
                'mushroom', 'vote']
    for idx, dataset in enumerate(datasets):
        df = pd.read_csv('data/' + dataset + '.csv')
        df = df.replace('?', np.NaN)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1:]
        X = pipe.fit_transform(X).astype(bool)
        lb = LabelBinarizer()
        y = lb.fit_transform(y).ravel().astype(bool)
        print(cross_val_score(RuleNetworkClassifier(), X, y, cv=2))
