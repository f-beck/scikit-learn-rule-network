import cProfile
import itertools
import logging
import numpy as np
import pandas as pd
import sys

from collections import defaultdict
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklrules import DeepRuleNetworkClassifier, OneHotEncoder, \
    RuleNetworkClassifier, TypeSelector
from tabulate import tabulate


def _setup_logging():
    # set up logging ...
    f = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                          datefmt='%Y-%m-%d %H:%M:%S')
    # ... for drnc
    fh = logging.FileHandler('drnc.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(f)
    logger = logging.getLogger(
        'sklrules._deep_rule_network.DeepRuleNetworkClassifier')
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('drnc_models.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(f)
    logger.addHandler(fh)
    # ... for rnc
    fh = logging.FileHandler('rnc.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(f)
    logger = logging.getLogger(
        'sklrules._rule_network.RuleNetworkClassifier')
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)


_setup_logging()

RANDOM_STATE = 0
N_REPETITIONS = 1
POS_CLASS_METHOD = 'boolean'  # 'boolean', 'least-frequent' or 'most-frequent'
AVG_RULE_LENGTH_DRNC = 2
AVG_RULE_LENGTH_RNC_1 = 2
AVG_RULE_LENGTH_RNC_2 = 5
HIDDEN_LAYER_SIZES = [32, 16, 8, 4, 2]
N_RULES = 200

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

metrics = defaultdict(lambda: np.ndarray([0]))
datasets = ['parity_1000']


def test():
    X = pd.DataFrame(itertools.product([False, True], repeat=10),
                     columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    X = ts.fit_transform(X)
    X = ohe.fit_transform(X).astype(bool)
    fit_params = {
        'attributes': ohe.attributes_,
        'attribute_lengths': ohe.attribute_lengths_,
        'features': ohe.features_,
    }

    seed = 0
    for _ in range(N_REPETITIONS):
        pos_ratio = 0
        while pos_ratio > 0.8 or pos_ratio < 0.2:
            seed += 1
            drnc1 = DeepRuleNetworkClassifier(
                hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                avg_rule_length=AVG_RULE_LENGTH_DRNC,
                pos_class_method=POS_CLASS_METHOD, random_state=seed)
            drnc1.fit(X[:2], [False, True], ohe.attributes_,
                      ohe.attribute_lengths_, ohe.features_)
            y = drnc1.predict(X)
            pos_ratio = np.count_nonzero(y) / len(y)
        metrics['seed'] = np.append(metrics['seed'], seed)
        metrics['positive\nratio'] = \
            np.append(metrics['positive\nratio'], pos_ratio)

        drnc2 = DeepRuleNetworkClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES,
            avg_rule_length=AVG_RULE_LENGTH_DRNC,
            pos_class_method=POS_CLASS_METHOD, random_state=seed + 1)
        metrics_drnc = cross_validate(drnc2, X, y, cv=skf,
                                      fit_params=fit_params)
        _add_metrics(metrics_drnc, 'DRNC')

        rnc1 = RuleNetworkClassifier(
            n_rules=N_RULES, avg_rule_length=AVG_RULE_LENGTH_RNC_1,
            pos_class_method=POS_CLASS_METHOD, random_state=RANDOM_STATE)
        metrics_rnc = cross_validate(rnc1, X, y, cv=skf, fit_params=fit_params)
        _add_metrics(metrics_rnc, 'RNC1')

        rnc2 = RuleNetworkClassifier(
            n_rules=N_RULES, avg_rule_length=AVG_RULE_LENGTH_RNC_2,
            pos_class_method=POS_CLASS_METHOD, random_state=RANDOM_STATE)
        metrics_rnc = cross_validate(rnc2, X, y, cv=skf, fit_params=fit_params)
        _add_metrics(metrics_rnc, 'RNC2')

    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')
    drnc1.print_model(style='tree')


def _add_metrics(metrics_learner, name):
    for key in ('fit_time', 'test_score'):
        metrics[key + '\n' + name] = np.append(
            metrics[key + '\n' + name], np.average(metrics_learner[key]))


def _process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target


with cProfile.Profile() as pr:
    test()

sys.stdout = open('test.prof', 'w')
pr.print_stats(sort='tottime')
