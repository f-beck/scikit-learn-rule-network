import numpy as np
import pandas as pd
import wittgenstein as lw

from collections import defaultdict
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklrules import OneHotEncoder, RuleNetworkClassifier, TypeSelector
from tabulate import tabulate

datasets = ['adult', 'airline-passenger-satisfaction', 'alpha-bank', 'bank',
            'breast-cancer', 'credit-g', 'hepatitis', 'kr-vs-kp',
            'mushroom', 'vote']
RANDOM_STATE = 0
POS_CLASS_METHOD = 'least-frequent'

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=10)


def test():
    metrics = defaultdict(lambda: np.ndarray([0]))
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    metrics['dataset'] = np.array(datasets)
    for idx, dataset in enumerate(datasets):
        X, y, target = __process_dataset(dataset)
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = classes[class_counts.argmax()]
        else:
            pos_class = classes[class_counts.argmin()]
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
        metrics_rnc_supp = cross_validate(RuleNetworkClassifier(
            init_method='support',
            max_flips=5, pos_class_method=POS_CLASS_METHOD), X, y,
            cv=skf, fit_params=fit_params)
        for key in ('fit_time', 'score_time', 'test_score'):
            metrics[key + '\nRIPPER'] = np.append(
                metrics[key + '\nRIPPER'], np.average(metrics_ripper[key]))
            metrics[key + '\nRNC_PROB'] = np.append(
                metrics[key + '\nRNC_PROB'], np.average(metrics_rnc_prob[key]))
            metrics[key + '\nRNC_RIPPER'] = np.append(
                metrics[key + '\nRNC_RIPPER'], np.average(metrics_rnc_ripper[
                                                              key]))
            metrics[key + '\nRNC_SUPP'] = np.append(
                metrics[key + '\nRNC_SUPP'], np.average(metrics_rnc_supp[key]))
    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')


def __process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target
