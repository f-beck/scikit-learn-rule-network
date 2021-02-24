import cProfile
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
import sys
import wittgenstein as lw

from collections import defaultdict
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklrules import DeepRuleNetworkClassifier, OneHotEncoder, TypeSelector
from sympy.logic.boolalg import to_dnf
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


def test():
    metrics['dataset'] = np.array(DATASETS)

    for idx, dataset in enumerate(DATASETS):
        X, y, target = _process_dataset(dataset)
        X, y = X[-MAX_SAMPLES:][:], y[-MAX_SAMPLES:][:]

        # make class attribute binary
        classes, class_counts = np.unique(y, return_counts=True)
        if POS_CLASS_METHOD == 'most-frequent':
            pos_class = str(classes[class_counts.argmax()])
        elif POS_CLASS_METHOD == 'least-frequent':
            pos_class = str(classes[class_counts.argmin()])
        elif POS_CLASS_METHOD == 'boolean':
            pos_class = 'True'
        else:  # use least-frequent
            pos_class = str(classes[class_counts.argmin()])
        pos_ratio = np.count_nonzero(y) / len(y) if y.dtype == 'bool' else \
            np.count_nonzero(y == pos_class) / len(y)
        metrics['positive\nratio'] = \
            np.append(metrics['positive\nratio'], pos_ratio)

        X = ts.fit_transform(X)
        X = ohe.fit_transform(X).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }
        fig, ax = [None] * N_FOLDS, [None] * N_FOLDS

        for hidden_layer_sizes in HIDDEN_LAYER_SIZES:
            if len(hidden_layer_sizes) == 1:
                for avg_rule_length in AVG_RULE_LENGTHS_RNC:
                    rnc = DeepRuleNetworkClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        avg_rule_length=avg_rule_length,
                        pos_class_method=POS_CLASS_METHOD,
                        random_state=RANDOM_STATE)
                    fig, ax = _test_estimator(
                        rnc, f'RNC_{hidden_layer_sizes}_{avg_rule_length}',
                        X, y, fit_params, fig, ax)
            else:
                for avg_rule_length in AVG_RULE_LENGTHS_DRNC:
                    for init_prob in INIT_PROBS:
                        drnc = DeepRuleNetworkClassifier(
                            hidden_layer_sizes=hidden_layer_sizes,
                            avg_rule_length=avg_rule_length,
                            init_prob=init_prob,
                            pos_class_method=POS_CLASS_METHOD,
                            random_state=RANDOM_STATE)
                        fig, ax = _test_estimator(
                            drnc, f'DRNC_{hidden_layer_sizes}_'
                                  f'{avg_rule_length}_{init_prob}', X,
                            y, fit_params, fig, ax)

        ripper = lw.RIPPER(random_state=RANDOM_STATE)
        metrics_ripper = cross_validate(ripper, X, y, cv=skf,
                                        fit_params={'pos_class': pos_class})
        _add_metrics(metrics_ripper, 'RIPPER')

        decision_tree = DecisionTreeClassifier()
        metrics_decision_tree = cross_validate(decision_tree, X, y, cv=skf)
        _add_metrics(metrics_decision_tree, 'Tree')

        if PLOT_ACCURACIES and all(fig):
            for fold in range(N_FOLDS):
                # noinspection PyUnresolvedReferences
                fig[fold].savefig(f'{dataset}_{fold}.png', bbox_inches='tight')
                plt.close('all')
        print()
        print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')

    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')


def _test_estimator(estimator, name, X, y, fit_params, fig, ax):
    metrics_drnc = cross_validate(
        estimator, X, y, cv=skf, return_estimator=True, fit_params=fit_params)
    _add_metrics(metrics_drnc, name)
    for fold in range(N_FOLDS):
        _print_estimator_model(metrics_drnc['estimator'][fold])
        if PLOT_ACCURACIES:
            fig[fold], ax[fold] = metrics_drnc['estimator'][
                fold].plot_accuracy_graph(fig[fold], ax[fold], name)
    return fig, ax


def _add_metrics(metrics_learner, name):
    for key in KEYS:
        metrics[key + '\n' + name] = np.append(
            metrics[key + '\n' + name], np.average(metrics_learner[key]))


def _print_estimator_model(metrics_learner):
    model = metrics_learner.print_model(style='tree')
    if PLOT_COMPLEX_MODEL:
        print(_to_sympy_syntax(model))
    if PLOT_SIMPLE_MODEL:
        print(_get_simplified_model(_to_sympy_syntax(model)))


def _get_simplified_model(model):
    return str(to_dnf(model, True, True))


def _to_sympy_syntax(f):
    f = re.sub(r'(\S+)=True', r'\1', f)
    f = re.sub(r'(\S+)=False', r'~\1', f)
    return f


def _process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target


RANDOM_STATE = 0  # only used for StratifiedKFold
N_FOLDS = 2
DATASETS = ['car-evaluation-binary', 'connect-4-binary', 'kr-vs-kp',
            'mushroom', 'tic-tac-toe', 'vote']
MAX_SAMPLES = 100000
POS_CLASS_METHOD = 'most-frequent'  # 'boolean', 'least-frequent' or
# 'most-frequent'
NOISE = 0.0

HIDDEN_LAYER_SIZES_SIMULATED_NETWORK = [32, 16, 8, 4, 2]
HIDDEN_LAYER_SIZES = [[32, 16, 8, 4, 2], [32, 8, 2], [20]]
AVG_RULE_LENGTHS_DRNC = [1, 2, 3]
AVG_RULE_LENGTHS_RNC = [2, 3, 4, 5]
INIT_PROBS = [0.05]

PLOT_COMPLEX_MODEL = False
PLOT_SIMPLE_MODEL = False
PLOT_ACCURACIES = True
# subset of ['fit_time', 'score_time', 'test_score']
KEYS = ['fit_time', 'test_score']

ts = TypeSelector(np.number, False)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

metrics = defaultdict(lambda: np.ndarray([0]))


if __name__ == '__main__':
    _setup_logging()

    with cProfile.Profile() as pr:
        test()

    sys.stdout = open('test.prof', 'w')
    pr.print_stats(sort='tottime')
