import numpy as np
import pandas as pd

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
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
        y = df.iloc[:, -1:].values.ravel()
        X = pipe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': pipe['onehotencoder'].attributes_,
            'attribute_lengths': pipe['onehotencoder'].attribute_lengths_,
            'features': pipe['onehotencoder'].features_,
            'target': list(df)[-1]
        }
        print(cross_val_score(RuleNetworkClassifier(), X, y, cv=2,
                              fit_params=fit_params))
