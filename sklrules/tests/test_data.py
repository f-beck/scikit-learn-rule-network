import numpy as np
import pandas as pd
import wittgenstein as lw

from sklearn.model_selection import cross_validate, StratifiedKFold
from sklrules import OneHotEncoder, RuleNetworkClassifier, TypeSelector
from tabulate import tabulate

datasets = ['adult', 'airline-passenger-satisfaction', 'alpha-bank', 'bank',
            'breast-cancer', 'credit-g', 'hepatitis', 'kr-vs-kp',
            'mushroom', 'vote']
datasets = ['vote']
RANDOM_STATE = 0
POS_CLASS_METHOD = 'least-frequent'

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=5)


def test():
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    for idx, dataset in enumerate(datasets):
        print('\nDataset', dataset)
        X, y, target = __process_dataset(dataset)
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = classes[class_counts.argmax()]
        else:
            pos_class = classes[class_counts.argmin()]
        metrics = {}
        X = ts.fit_transform(X)
        ripper.fit(X, y, pos_class=pos_class)
        fit_params = {'pos_class': pos_class}
        metrics_ripper = cross_validate(ripper, X, y, cv=skf,
                                        fit_params=fit_params)
        X = ohe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }
        metrics_rnc_prob = cross_validate(RuleNetworkClassifier(
            pos_class_method=POS_CLASS_METHOD), X, y, cv=skf,
            fit_params=fit_params)
        metrics_rnc_ripper = cross_validate(RuleNetworkClassifier(
            init_method='ripper', ripper_model=ripper.ruleset_,
            pos_class_method=POS_CLASS_METHOD), X, y, cv=skf,
            fit_params=fit_params)
        for key in metrics_ripper:
            metrics[key + '_RIPPER'] = metrics_ripper[key]
            metrics[key + '_RNC_PROB'] = metrics_rnc_prob[key]
            metrics[key + '_RNC_RIPPER'] = metrics_rnc_ripper[key]
        print(tabulate(metrics, headers='keys'))


def __process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target
