import numpy as np
import pandas as pd
import wittgenstein as lw

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklrules import OneHotEncoder, RuleNetworkClassifier, TypeSelector

pipe = make_pipeline(
    TypeSelector(np.number, False),
    OneHotEncoder(sparse=False, handle_unknown='ignore'),
)

datasets = ['adult', 'airline-passenger-satisfaction', 'alpha-bank', 'bank',
            'breast-cancer', 'credit-g', 'hepatitis', 'kr-vs-kp',
            'mushroom', 'vote']
RANDOM_STATE = 0
POS_CLASS_METHOD = 'least-frequent'


def test():
    for idx, dataset in enumerate(datasets):
        X, y, target = __process_dataset(dataset)
        X = pipe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': pipe['onehotencoder'].attributes_,
            'attribute_lengths': pipe['onehotencoder'].attribute_lengths_,
            'features': pipe['onehotencoder'].features_,
            'target': target
        }
        print(cross_val_score(RuleNetworkClassifier(
            pos_class_method=POS_CLASS_METHOD), X, y, cv=2,
            fit_params=fit_params))


def test_ripper():
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    for idx, dataset in enumerate(datasets):
        X, y, target = __process_dataset(dataset)
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = classes[class_counts.argmax()]
        else:
            pos_class = classes[class_counts.argmin()]
        ripper.fit(X, y, pos_class=pos_class)
        X = pipe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': pipe['onehotencoder'].attributes_,
            'attribute_lengths': pipe['onehotencoder'].attribute_lengths_,
            'features': pipe['onehotencoder'].features_,
            'target': target
        }
        print(cross_val_score(RuleNetworkClassifier(
            init_method='ripper', ripper_model=ripper.ruleset_,
            pos_class_method=POS_CLASS_METHOD), X, y,
            cv=2, fit_params=fit_params))


def __process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target
