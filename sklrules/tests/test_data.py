import numpy as np
import pandas as pd
import wittgenstein as lw

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklrules import OneHotEncoder, RuleNetworkClassifier, TypeSelector

datasets = ['adult', 'airline-passenger-satisfaction', 'alpha-bank', 'bank',
            'breast-cancer', 'credit-g', 'hepatitis', 'kr-vs-kp',
            'mushroom', 'vote']
RANDOM_STATE = 0
POS_CLASS_METHOD = 'least-frequent'

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=5)


def test():
    for idx, dataset in enumerate(datasets):
        print('\nDataset', dataset)
        X, y, target = __process_dataset(dataset)
        X = ts.fit_transform(X)
        X = ohe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }
        print('RNC(probabilistic):', cross_val_score(RuleNetworkClassifier(
            pos_class_method=POS_CLASS_METHOD), X, y, cv=skf,
            fit_params=fit_params))


def test_ripper():
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    for idx, dataset in enumerate(datasets):
        print('\nDataset', dataset)
        X, y, target = __process_dataset(dataset)
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = classes[class_counts.argmax()]
        else:
            pos_class = classes[class_counts.argmin()]
        X = ts.fit_transform(X)
        ripper.fit(X, y, pos_class=pos_class)
        fit_params = {'pos_class': pos_class}
        print('RIPPER:', cross_val_score(ripper, X, y, cv=skf,
                                         fit_params=fit_params))
        X = ohe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }
        print('RNC(RIPPER):', cross_val_score(RuleNetworkClassifier(
            init_method='ripper', ripper_model=ripper.ruleset_,
            pos_class_method=POS_CLASS_METHOD), X, y, cv=skf,
            fit_params=fit_params))


def __process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target
