import logging
import numpy as np
import pandas as pd
import wittgenstein as lw

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
POS_CLASS_METHOD = 'boolean'  # 'boolean', 'least-frequent' or 'most-frequent'
DRNC = True
RIPPER = False
RNC_PROB = True
RNC_RIPPER = False
RNC_SUPP = False

AVG_RULE_LENGTH_DRNC = 2
AVG_RULE_LENGTH_RNC = 2

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

metrics = defaultdict(lambda: np.ndarray([0]))
datasets = ['parity_1000']


def test():
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    metrics['dataset'] = np.array(datasets)

    for idx, dataset in enumerate(datasets):
        X, y, target = _process_dataset(dataset)
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = str(classes[class_counts.argmax()])
        elif POS_CLASS_METHOD == 'least-frequent':
            pos_class = str(classes[class_counts.argmin()])
        elif POS_CLASS_METHOD == 'boolean':
            pos_class = 'True'
        else:  # use least-frequent
            pos_class = str(classes[class_counts.argmin()])
        X = ts.fit_transform(X)
        if RIPPER or RNC_RIPPER:
            ripper.fit(X, y, pos_class=pos_class)
        fit_params = {'pos_class': pos_class}
        if RIPPER:
            metrics_ripper = cross_validate(ripper, X, y, cv=skf,
                                            fit_params=fit_params)
            _add_metrics(metrics_ripper, 'RIPPER')
        X = ohe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }
        if DRNC:
            metrics_drnc = cross_validate(DeepRuleNetworkClassifier(
                hidden_layer_sizes=[10, 5, 2],
                avg_rule_length=AVG_RULE_LENGTH_DRNC,
                pos_class_method=POS_CLASS_METHOD, plot_accuracies=True,
                random_state=RANDOM_STATE), X, y, cv=skf,
                fit_params=fit_params, error_score='raise')
            _add_metrics(metrics_drnc, 'DRNC')
        if RNC_PROB:
            metrics_rnc_prob = cross_validate(RuleNetworkClassifier(
                avg_rule_length=AVG_RULE_LENGTH_RNC,
                pos_class_method=POS_CLASS_METHOD,
                random_state=RANDOM_STATE), X, y, cv=skf, fit_params=fit_params)
            _add_metrics(metrics_rnc_prob, 'RNC_PROB')
        if RNC_RIPPER:
            metrics_rnc_ripper = cross_validate(RuleNetworkClassifier(
                init_method='ripper', ripper_model=ripper.ruleset_,
                pos_class_method=POS_CLASS_METHOD,
                random_state=RANDOM_STATE), X, y, cv=skf, fit_params=fit_params)
            _add_metrics(metrics_rnc_ripper, 'RNC_RIPPER')
        if RNC_SUPP:
            metrics_rnc_supp = cross_validate(RuleNetworkClassifier(
                init_method='support', max_flips=5,
                pos_class_method=POS_CLASS_METHOD,
                random_state=RANDOM_STATE), X, y, cv=skf, fit_params=fit_params)
            _add_metrics(metrics_rnc_supp, 'RNC_SUPP')
    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')


def _add_metrics(metrics_learner, name):
    for key in ('fit_time', 'score_time', 'test_score'):
        metrics[key + '\n' + name] = np.append(
            metrics[key + '\n' + name], np.average(metrics_learner[key]))


def _process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target
