import cProfile
import itertools
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
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
    fh = logging.FileHandler('logs/drnc.log')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(f)
    logger = logging.getLogger(
        'sklrules._deep_rule_network.DeepRuleNetworkClassifier')
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('logs/drnc_models.log')
    fh.setLevel(logging.INFO)
    fh.setFormatter(f)
    logger.addHandler(fh)


def test_hierarchical_data():
    metrics['dataset'] = np.array(DATASETS)

    for dataset in DATASETS:
        X, y, target = _process_dataset(dataset)
        X, y = X[-MAX_SAMPLES:][:], y[-MAX_SAMPLES:][:]
        X_ohe = ts.fit_transform(X)
        X_ohe = ohe.fit_transform(X_ohe).astype(bool)
        fit_params = {
            'attributes': ohe.attributes_,
            'attribute_lengths': ohe.attribute_lengths_,
            'features': ohe.features_,
            'target': target
        }

        pos_ratio, pos_class = _calculate_pos_ratio(y)
        metrics['positive\nratio'] = \
            np.append(metrics['positive\nratio'], pos_ratio)

        fig, ax = _test_all_estimators(X, X_ohe, y, fit_params,
                                       POS_CLASS_METHOD, pos_class,
                                       RANDOM_STATE)
        _generate_plots(fig, ax, dataset)

        print()
        print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')

    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')


def test_simulated_network():
    X = pd.DataFrame(itertools.product([False, True], repeat=10),
                     columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'])
    X_ohe = ts.fit_transform(X)
    X_ohe = ohe.fit_transform(X_ohe).astype(bool)
    fit_params = {
        'attributes': ohe.attributes_,
        'attribute_lengths': ohe.attribute_lengths_,
        'features': ohe.features_,
    }

    seed = START_SEED
    fig, ax = [[None] * N_FOLDS] * N_REPETITIONS, \
              [[None] * N_FOLDS] * N_REPETITIONS
    for i in range(N_REPETITIONS):
        simulated_network, y, seed, pos_ratio = _simulate_network(X_ohe, seed)
        metrics['dataset'] = np.append(metrics['dataset'], seed)
        metrics['positive\nratio'] = \
            np.append(metrics['positive\nratio'], pos_ratio)
        _print_estimator_model(simulated_network)

        fig[i], ax[i] = _test_all_estimators(X, X_ohe, y, fit_params,
                                             'boolean', True, seed + 1)
        _generate_plots(fig[i], ax[i], seed)

        print()
        print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')

    _generate_avg_plot(ax, 0, 0, 'average_0_0')
    _generate_avg_plot(ax, 0, 1, 'average_0_1')
    _generate_avg_plot(ax, 1, 0, 'average_1_0')
    _generate_avg_plot(ax, 1, 1, 'average_1_1')

    print('\n', tabulate(metrics, headers='keys', floatfmt='.4f'), sep='')


def _process_dataset(dataset):
    df = pd.read_csv('data/' + dataset + '.csv')
    df = df.replace('?', np.NaN)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1:].values.ravel()
    target = list(df)[-1]
    return X, y, target


def _calculate_pos_ratio(y):
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
    return pos_ratio, pos_class


def _simulate_network(X, seed):
    simulated_network = None
    y = None
    pos_ratio = 0
    while pos_ratio > 0.8 or pos_ratio < 0.2:
        seed += 1
        simulated_network = DeepRuleNetworkClassifier(
            hidden_layer_sizes=HIDDEN_LAYER_SIZES_SIMULATED_NETWORK,
            avg_rule_length=2, pos_class_method='boolean', random_state=seed)
        simulated_network.fit(X[:2], [False, True], ohe.attributes_,
                              ohe.attribute_lengths_, ohe.features_)
        y = simulated_network.predict(X)
        noise_samples = random.sample(range(len(y)), round(NOISE * len(y)))
        y[noise_samples] = ~y[noise_samples]
        pos_ratio = np.count_nonzero(y) / len(y)
    return simulated_network, y, seed, pos_ratio


def _generate_plots(fig, ax, seed):
    if PLOT_ACCURACIES and all(fig):
        for fold in range(N_FOLDS):
            _format_plot(fig[fold], ax[fold], f'{seed}_{fold}')


def _generate_avg_plot(ax, exclude_first, exclude_last, name='average'):
    n_graphs = len(HIDDEN_LAYER_SIZES)
    if N_REPETITIONS and N_FOLDS and n_graphs:
        labels = [ax[0][0].lines[i].get_label() for i in range(n_graphs)]
        # find max number of data points in graph
        n_data = max([len(ax[i][j].lines[graph].get_xdata()) for graph in range(
            n_graphs) for j in range(N_FOLDS) for i in range(N_REPETITIONS)])
        # fill graphs having less data points with their last value
        y_data = [[[np.pad(ax[i][j].lines[graph].get_ydata(),
                           (0, n_data - len(ax[i][j].lines[graph].get_ydata())),
                    'edge') for graph in range(n_graphs)] for j in range(
            N_FOLDS)] for i in range(N_REPETITIONS)]
        avg_accuracies = np.mean(y_data, axis=(0, 1))
        print(avg_accuracies)

        avg_fig, avg_ax = plt.subplots()
        avg_ax.set(xlabel='Mini-batch', ylabel='Accuracy',
                   title='Average accuracy over number of mini-batches')
        labels = ['DRNC(5)', 'DRNC(3)', 'RNC']
        for (graph_data, graph_label) in zip(avg_accuracies, labels):
            if exclude_last:
                avg_ax.plot(range(exclude_first, n_data - exclude_last),
                            graph_data[exclude_first:-exclude_last],
                            label=graph_label, linewidth='1')
            else:
                avg_ax.plot(range(exclude_first, n_data),
                            graph_data[exclude_first:],
                            label=graph_label, linewidth='1')
        _format_plot(avg_fig, avg_ax, name)
        plt.close(avg_fig)


def _format_plot(fig, ax, name):
    for line, marker, linestyle in zip(
            ax.lines, itertools.cycle('so^PD'),
            itertools.cycle(["-", "--", "-.", ":"])):
        line.set_marker(marker)
        line.set_linestyle(linestyle)
    ax.legend(loc='lower right')
    fig.savefig(f'plots/{name}.png', bbox_inches='tight')
    plt.close('all')


def _test_all_estimators(X, X_ohe, y, fit_params, pos_class_method, pos_class,
                         random_state):
    fig, ax = [None] * N_FOLDS, [None] * N_FOLDS
    for hidden_layer_sizes in HIDDEN_LAYER_SIZES:
        n_layers = len(hidden_layer_sizes)
        if n_layers == 1:
            for arl in AVG_RULE_LENGTHS[1]:
                rnc = DeepRuleNetworkClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    avg_rule_length=arl, pos_class_method=pos_class_method,
                    random_state=random_state)
                fig, ax = _test_estimator(
                    rnc, f'RNC_{hidden_layer_sizes}_{arl}', X_ohe, y,
                    fit_params, fig, ax)
        else:
            for arl in AVG_RULE_LENGTHS[n_layers]:
                for ip in INIT_PROBS:
                    drnc = DeepRuleNetworkClassifier(
                        hidden_layer_sizes=hidden_layer_sizes,
                        avg_rule_length=arl, init_prob=ip,
                        pos_class_method=pos_class_method,
                        random_state=random_state)
                    fig, ax = _test_estimator(
                        drnc, f'DRNC_{hidden_layer_sizes}_{arl}_{ip}',
                        X_ohe, y, fit_params, fig, ax)

    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    metrics_ripper = cross_validate(ripper, X_ohe, y, cv=skf,
                                    fit_params={'pos_class': pos_class})
    _add_metrics(metrics_ripper, 'RIPPER')
    metrics_ripper = cross_validate(ripper, X, y, cv=skf,
                                    fit_params={'pos_class': pos_class})
    _add_metrics(metrics_ripper, 'RIPPER(orig. data)')

    decision_tree = DecisionTreeClassifier(random_state=RANDOM_STATE)
    metrics_decision_tree = cross_validate(decision_tree, X_ohe, y, cv=skf)
    _add_metrics(metrics_decision_tree, 'Tree')

    return fig, ax


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


# hyperparameters for hierarchical data test
DATASETS = ['car-evaluation-binary', 'connect-4-binary', 'kr-vs-kp',
            'mushroom', 'tic-tac-toe', 'vote']
MAX_SAMPLES = 4000
POS_CLASS_METHOD = 'most-frequent'  # 'boolean', 'least-frequent' or
# 'most-frequent'

# hyperparameters for simulated network test
START_SEED = 0
N_REPETITIONS = 20
HIDDEN_LAYER_SIZES_SIMULATED_NETWORK = [32, 16, 8, 4, 2]

# hyperparameters for both tests
RANDOM_STATE = 0
N_FOLDS = 2
NOISE = 0.0

HIDDEN_LAYER_SIZES = [[32, 16, 8, 4, 2], [32, 8, 2], [20]]
AVG_RULE_LENGTHS = {1: [5], 3: [3], 5: [2]}
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
        test_hierarchical_data()
        test_simulated_network()

    sys.stdout = open('logs/test.prof', 'w')
    pr.print_stats(sort='tottime')
