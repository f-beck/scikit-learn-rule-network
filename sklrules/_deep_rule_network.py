"""
This is a module holding the rule network class
"""
import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, \
    check_random_state


class DeepRuleNetworkClassifier(BaseEstimator, ClassifierMixin):
    """ A classifier which uses a network structure to learn rule sets.

    Parameters
    ----------
    init_method : {'probabilistic'},
    default='probabilistic'
        The method used to initialize the rules. Must be 'probabilistic' to
        initialize each attribute of a rule with a fixed probability. Further
        initialization methods are planned.

    hidden_layer_sizes : list[int], default=[10]
        The number of nodes per layer in the network. Does not include
        size for the input and output layer which will be set automatically.

    first_layer_conjunctive: bool, default=True
        The type of the first layer in the network. Must be True to emulate a
        conjunctive ('and') behavior or False for a disjunctive ('or') behavior.

    avg_rule_length : int, default=3
        The average number of conditions in an initial rule. Each attribute
        is set to a random value and added to the rule with a probability of
        3/|A| with |A| being the number of attributes. Only has an effect if
        init_method='probabilistic'.

    batch_size : int, default=50
        The number of samples per mini-batch.

    max_flips : int, default=2
        The maximum number of flips per mini-batch.

    pos_class_method : {'least-frequent', 'most-frequent'},
    default='least-frequent'
        The class chosen to be converted to True and to be the head of the
        generated rules.

    interim_train_accuracies : bool, default=True
        If True, after each batch the accuracy value on the training set will
        be measured and stored. Set False to save runtime.

    plot_accuracies : bool, default=False
        If True, after fit method the accuracy development will be plotted.

    random_state : int, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Attributes
    ----------
    n_layers : int
        The number of layers in the network. Includes input and output layer.

    last_layer_conjunctive : bool
        The type of the last layer in the network. Will be set automatically
        depending on 'first_layer' and n_layers since conjunctive and
        disjunctive layers alternate.

    X_ : ndarray, shape (n_samples, n_features)
        The categorical, one-hot-encoded features of the training samples
        passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    n_attributes_ : int
        The number of attributes passed during :meth:`fit`.

    attributes_ : list[str] of shape (n_attributes,)
        String names for (non-boolean) attributes passed during :meth:`fit`.

    attribute_lengths_ : list[int], shape (n_attributes,)
        The number of unique values per attribute, passed during :meth:`fit`.

    attribute_lengths_cumsum_ : list[int], shape (n_attributes,)
        The cumulative sum of attribute_lengths_, used as indexes for X_.

    n_features_ : int
        The number of features seen at :meth:`fit`.

    features_ : list[str], shape (n_features,)
        String names for (boolean) features passed during :meth:`fit`.

    n_classes_ : int
        The number of classes seen at :meth:`_preprocess_classes`.

    classes_ : list[str], shape (n_classes,)
        String names for the classes seen at :meth:`_preprocess_classes`.

    output_feature_ : str
        String name for output feature.

    coefs_ : list[ndarray], shape (n_layers - 1,)
        The ith element in the list represents the weight matrix
        corresponding to layer i, i.e. coefs_[i][j][k] represents if node j
        in layer i passes its output to node k in layer i+1.

    n_batches_ : int
        The number of batches used during :meth:`fit`.

    batch_accuracies_ : ndarray, shape (n_batches + 2,)
        The accuracies on the mini-batch after optimization on it. The first
        element is the accuracy on the training set after initialization and
        the last one the accuracy on the training set after optimization.

    train_accuracies_ : ndarray, shape (n_batches + 2,)
        The accuracies on the training set after optimization on a
        mini-batch. The first element is the accuracy on the training set
        after initialization and the last one the accuracy on the training
        set after optimization.

    random_state_ : int
        A random number generator instance to define the state of the
        random permutations generator.
    """

    def __init__(self, init_method='probabilistic', hidden_layer_sizes=None,
                 first_layer_conjunctive=True, avg_rule_length=3,
                 batch_size=50, max_flips=5, pos_class_method='least-frequent',
                 interim_train_accuracies=True,
                 plot_accuracies=False, random_state=None):
        if hidden_layer_sizes is None:
            hidden_layer_sizes = [10]
        self._class_logger = logging.getLogger(__name__).getChild(
            self.__class__.__name__)
        self.init_method = init_method
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_layers = len(hidden_layer_sizes) + 2
        self.first_layer_conjunctive = first_layer_conjunctive
        if self.n_layers % 2:
            self.last_layer_conjunctive = not first_layer_conjunctive
        else:
            self.last_layer_conjunctive = first_layer_conjunctive
        self.avg_rule_length = avg_rule_length
        self.batch_size = batch_size
        self.max_flips = max_flips
        self.pos_class_method = pos_class_method
        self.interim_train_accuracies = interim_train_accuracies
        self.plot_accuracies = plot_accuracies
        self.random_state = random_state

    def fit(self, X, y, attributes=None, attribute_lengths=None,
            features=None, target='class'):
        """ The fitting function creates binary layers adjusted to the
        size of the input (n_features). It learns a model by flipping
        the boolean values to create suitable rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.
        attributes : list[str], default=None
            The names of each attribute (in order of the original features in
            X).
        attribute_lengths : list[int], default=None
            The cardinality of the attributes.
        features : list[str], default=None
            The names of each column after in order of the features in X.
        target : str, default='class'
            The name of the target.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns fitted classifier.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Create dummy names/lengths for attributes and features if None were
        # given
        if attributes is None:
            if attribute_lengths is None:
                attributes = ['x%d' % i for i in range(X.shape[1])]
            else:
                attributes = ['x%d' % i for i in range(len(attribute_lengths))]
        if attribute_lengths is None:
            attribute_lengths = [1] * len(attributes)
        if features is None:
            features = attributes

        # Check additional fit parameters
        if len(attributes) != len(attribute_lengths):
            raise ValueError('%s attributes, but %s attribute lengths are given'
                             % (len(attributes), len(attribute_lengths)))
        if len(features) != sum(attribute_lengths):
            raise ValueError('%s features given, but attribute lengths sum up '
                             'to %s)' % (len(features), sum(attribute_lengths)))

        # Preprocess classes
        pos_class = self._preprocess_classes(y)
        self.output_feature_ = target + '=' + pos_class

        # Initialize attributes
        self.X_ = X
        self.n_attributes_ = len(attributes)
        self.attributes_ = attributes
        self.attribute_lengths_ = attribute_lengths
        self.attribute_lengths_cumsum_ = np.cumsum(self.attribute_lengths_)
        self.n_features_ = len(features)
        self.features_ = features
        self.n_outputs_ = 1

        # Calculate number of batches
        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]
            self._class_logger.warning('Batch size was higher than number of '
                                       'training samples. Use single batch of '
                                       'size %s for training.', self.batch_size)
        self.n_batches_ = X.shape[0] // self.batch_size

        # Initialize rule network layers
        self._init_layers()

        # Initialize arrays for storing accuracies
        self.batch_accuracies_ = np.empty(shape=(self.n_batches_ + 3,))
        self.batch_accuracies_[0] = accuracy_score(y, self.predict(X))
        if self.interim_train_accuracies:
            self.train_accuracies_ = np.empty_like(self.batch_accuracies_)
            self.train_accuracies_[0] = self.batch_accuracies_[0]

        self._class_logger.info('Training network...')
        for batch in range(self.n_batches_):
            self._class_logger.debug('Processing mini-batch %s of %s...',
                                     batch + 1, self.n_batches_)
            X_mini = X[batch * self.batch_size:(batch + 1) * self.batch_size]
            y_mini = y[batch * self.batch_size:(batch + 1) * self.batch_size]
            self.batch_accuracies_[batch + 1] = self._optimize_coefs(X_mini,
                                                                     y_mini)
            if self.interim_train_accuracies:
                self.train_accuracies_[batch + 1] = accuracy_score(
                    y, self.predict(X))

        # iterate an additional time over all samples in a single batch
        self.batch_accuracies_[self.n_batches_ + 1] = self._optimize_coefs(X, y)
        if self.interim_train_accuracies:
            self.train_accuracies_[self.n_batches_ + 1] = \
                self.batch_accuracies_[self.n_batches_ + 1]

        # optimize the last layer
        self.batch_accuracies_[self.n_batches_ + 2] = \
            self._optimize_last_layer(X, y)
        if self.interim_train_accuracies:
            self.train_accuracies_[self.n_batches_ + 2] = \
                self.batch_accuracies_[self.n_batches_ + 2]


        self._class_logger.info('Training finished.')
        self.print_model()

        if self.plot_accuracies:
            self._plot_accuracy_graph()

        # Return the classifier
        return self

    def _plot_accuracy_graph(self):
        batch_range = range(self.n_batches_ + 2)
        plt.plot(batch_range, self.batch_accuracies_,
                 label='mini-batch', linewidth='0.5')
        if self.interim_train_accuracies:
            plt.plot(batch_range, self.train_accuracies_,
                     label='train set', linewidth='1')
        plt.xlabel('Mini-batch')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')
        plt.show()

    def predict(self, X):
        """ Predict output y for given input X by checking if any rule covers
        the sample. Output will be inverse transformed to the original classes.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            An array that holds True iff the corresponding sample in X is
            covered by at least one rule.
        """
        # Check if fit had been called
        check_is_fitted(self)

        # Input validation
        check_array(X, dtype='bool')

        # Initialize layers
        activations = [X]
        for i in range(self.n_layers - 1):
            activations.append(np.empty((X.shape[0], self.layer_units[i + 1]),
                                        dtype=bool))

        # Forward propagate
        self._forward_pass(activations)
        y_pred = activations[-1]

        return self.class_transformer_.inverse_transform(y_pred)

    def print_model(self, style='prolog'):
        """ Print all rules in the model.

        Parameters
        ----------
        style : {'prolog', 'tree'}
            The style of the printed model. Must be 'prolog' to show a flat
            logical structure or 'tree' to show a hierarchical structure.
        """

        # Check if fit had been called
        check_is_fitted(self)

        header = '\n--- MODEL ---\n'
        model_string = ''
        if style == 'prolog':
            conjunctive = self.first_layer_conjunctive
            for i in range(1, self.n_layers):
                model_string += '\n'
                connector = ', ' if conjunctive else '; '
                for k in range(self.layer_units[i]):
                    first = True
                    model_string += 'l' + str(i) + 'n' + str(k + 1) + ' :- '
                    for j in range(self.layer_units[i - 1]):
                        if self.coefs_[i - 1][j][k]:
                            if first:
                                first = False
                            else:
                                model_string += connector
                            if i == 1:
                                model_string += self.features_[j]
                            else:
                                model_string += 'l' + str(i - 1) + 'n' + \
                                                str(j + 1)
                    model_string += '.\n'
                conjunctive = not conjunctive
        elif style == 'tree':
            model_string += self._get_tree_string(self.n_layers - 2, 0,
                                                  self.last_layer_conjunctive)
        self._class_logger.info(header + model_string)
        return model_string

    def _flip_feature(self, j, k, dependent_features=np.array([])):
        """ Flip the value in the first layer depending whether a feature
        contributes or does not contribute to a node. In the case that
        first_layer_conjunctive = True, the features of the same attribute
        are flipped accordingly to ensure that only one feature per attribute
        is True, i.e. contributes to a node.

        Parameters
        ----------
        j : int
            The index of the feature to flip.
        k : int
            The index of the node to adjust.
        dependent_features : ndarray, shape (1,)
            Features of the same attribute set to True (should only be one).

        Returns
        -------
        dependent_features : ndarray, shape (1,)
            Features of the same attribute flipped from True to False (should
            only be one).
        """
        if self.first_layer_conjunctive:
            if not self.coefs_[0][j][k]:
                attribute = np.searchsorted(self.attribute_lengths_cumsum_, j,
                                            side='right')
                first_j, last_j = self._get_features_of_attribute(attribute)
                dependent_features = \
                    np.where(self.coefs_[0][first_j:last_j, k])[0] + first_j
            if dependent_features.size:
                for dependent_feature in dependent_features:
                    self.coefs_[0][dependent_feature][k] = \
                        not self.coefs_[0][dependent_feature][k]
        self.coefs_[0][j][k] = not self.coefs_[0][j][k]
        return dependent_features

    def _forward_pass(self, activations):
        """ Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer. Use
        uniform NAND activations to mimic alternating conjunction and
        disjunction layers.
        Parameters
        ----------
        activations : list[ndarray], length = n_layers
            The ith element of the list holds the values of the ith layer.
        """

        # For the first layer, invert output if conjunctive
        if self.first_layer_conjunctive:
            activations[1] = safe_sparse_dot(~activations[0], self.coefs_[0])
        else:
            activations[1] = safe_sparse_dot(activations[0], self.coefs_[0])

        # Iterate over the hidden layers
        for i in range(1, self.n_layers - 1):
            activations[i + 1] = safe_sparse_dot(~activations[i],
                                                 self.coefs_[i])

        # For the last layer, invert output if conjunctive
        if self.last_layer_conjunctive:
            activations[-1] = ~activations[-1]

        return activations

    def _get_features_of_attribute(self, attribute):
        """ Return the indices of the first and the last feature of an
        attribute.

        Parameters
        ----------
        attribute : int
            The index of the attribute to determine the features of.

        Returns
        -------
        first_feature : int
            The index of the first feature of the attribute (including).
        last_feature : int
            The index of the last feature of the attribute (excluding).
        """
        if attribute:
            first_feature = self.attribute_lengths_cumsum_[attribute - 1]
        else:  # first attribute starts at 0
            first_feature = 0
        last_feature = self.attribute_lengths_cumsum_[attribute]
        return first_feature, last_feature

    def _get_tree_string(self, i, k, conjunctive):
        """ Compute recursively a formatted string for the model. The output
        will have a tree structure.

        Parameters
        ----------
        i: int
            The current layer.
        k: int
            The current node in the successor layer
        conjunctive: bool
            Flag if the current layer is conjunctive.

        Returns
        -------
        model_string: String
            The model as a formatted string.
        """

        if i < 0:
            model_string = self.features_[k]
        else:
            connector = ' & ' if conjunctive else ' | '
            model_string = '(\n' + '  ' * (self.n_layers - 1 - i)
            first = True
            layer_string = ''
            for j in range(self.layer_units[i]):
                if self.coefs_[i][j][k]:
                    if first:
                        first = False
                    else:
                        layer_string += connector
                    layer_string += self._get_tree_string(i - 1, j,
                                                          not conjunctive)
            if not layer_string:
                layer_string = 'True'
            model_string += layer_string
            model_string += '\n' + '  ' * (self.n_layers - 2 - i) + ')'

        return model_string

    def _init_layers(self):
        """ Initialize the layers of the network according to the
        initialization parameter settings.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns classifier with initialized network.
        """
        self._class_logger.info('Initializing network...')
        self.random_state_ = check_random_state(self.random_state)

        self.layer_units = ([self.n_features_] + self.hidden_layer_sizes + [
            self.n_outputs_])
        self.coefs_ = [np.zeros((n_fan_in_, n_fan_out_), dtype=bool) for
                       n_fan_in_, n_fan_out_ in zip(self.layer_units[:-1],
                                                    self.layer_units[1:])]
        if self.init_method == 'probabilistic':
            if self.avg_rule_length > self.n_attributes_:
                self.avg_rule_length = self.n_attributes_
                self._class_logger.warning('Average rule length was higher '
                                           'than number of attributes. Use '
                                           'number of attributes as average '
                                           'rule length.')

            # initialize first layer according to attribute logic if it is
            # conjunctive
            if self.first_layer_conjunctive:
                for k in range(self.layer_units[1]):
                    for attribute in range(self.n_attributes_):
                        if self.random_state_.random() < (
                                self.avg_rule_length / self.n_attributes_):
                            random_feature = self.random_state_.randint(
                                self.attribute_lengths_[attribute])
                            self.coefs_[0][self.attribute_lengths_cumsum_[
                                               attribute] -
                                           random_feature - 1][k] = True

            # initialize last layer with True
            for attribute in range(self.layer_units[-2]):
                self.coefs_[self.n_layers - 2][attribute][0] = True

            # initialize all layers in between (starting from 1 if first
            # layer is conjunctive and from 0 if it is disjunctive)
            for i in range(self.first_layer_conjunctive * 1, self.n_layers - 2):
                for attribute in range(self.layer_units[i]):
                    # guarantee at least one outgoing edge set to True
                    k = self.random_state_.randint(self.layer_units[i + 1])
                    self.coefs_[i][attribute][k] = True
                    for k in range(self.layer_units[i + 1]):
                        if self.random_state_.random() < 0.2:
                            self.coefs_[i][attribute][k] = True

        self._class_logger.info('Initialization finished.')
        return self

    def _optimize_coefs(self, X, y):
        """ This function optimizes the layers of the network to the
        current mini-batch (X, y) by flipping at most max_flips times
        the boolean values.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        best_accuracy : RuleNetworkClassifier
            Returns accuracy on mini-batch (X, y).
        """
        base_accuracy = accuracy_score(y, self.predict(X))
        best_accuracy = base_accuracy
        best_i = None
        best_j = None
        best_k = None
        optimal = False
        flip_count = 0

        while not optimal and flip_count < self.max_flips:
            optimal = True

            # evaluate flips in the first layer
            for k in range(self.layer_units[1]):
                for attribute in range(self.n_attributes_):
                    first_j, last_j = self._get_features_of_attribute(attribute)
                    for j in range(first_j, last_j):
                        dependent_j = self._flip_feature(j, k)
                        accuracy = accuracy_score(y, self.predict(X))
                        if accuracy > best_accuracy:
                            optimal = False
                            best_accuracy = accuracy
                            best_i = 0
                            best_j = j
                            best_k = k
                        self._flip_feature(j, k, dependent_j)

            # evaluate flips in all other layers (except the last one)
            for i in range(1, self.n_layers - 2):
                for j in range(self.layer_units[i]):
                    for k in range(self.layer_units[i + 1]):
                        self.coefs_[i][j][k] = not self.coefs_[i][j][k]
                        accuracy = accuracy_score(y, self.predict(X))
                        if accuracy > best_accuracy:
                            optimal = False
                            best_accuracy = accuracy
                            best_i = i
                            best_j = j
                            best_k = k
                        self.coefs_[i][j][k] = not self.coefs_[i][j][k]

            if not optimal:
                # specific output for first layer
                if best_i == 0:
                    dependent_j = self._flip_feature(best_j, best_k)
                    if dependent_j.size:
                        self._class_logger.debug(
                            'Rule %s - replaced %s with %s - accuracy '
                            'increased from %s to %s', best_k + 1,
                            self.features_[dependent_j[0]],
                            self.features_[best_j],
                            base_accuracy, best_accuracy)
                    elif self.coefs_[0][best_j][best_k]:
                        self._class_logger.debug(
                            'Rule %s - added %s - accuracy increased from %s '
                            'to %s', best_k + 1, self.features_[best_j],
                            base_accuracy, best_accuracy)
                    else:
                        self._class_logger.debug(
                            'Rule %s - removed %s - accuracy increased from '
                            '%s to %s', best_k + 1, self.features_[best_j],
                            base_accuracy, best_accuracy)

                # general output for all other layers
                else:
                    self.coefs_[best_i][best_j][best_k] = \
                        not self.coefs_[best_i][best_j][best_k]
                    if self.coefs_[best_i][best_j][best_k]:
                        self._class_logger.debug(
                            'Layer %s - added connection from node %s to next '
                            'layer\'s node %s - accuracy increased from %s '
                            'to %s', best_i + 1, best_j + 1, best_k + 1,
                            base_accuracy, best_accuracy)
                    else:
                        self._class_logger.debug(
                            'Layer %s - removed connection from node %s to '
                            'next layer\'s node %s - accuracy increased from '
                            '%s to %s', best_i + 1, best_j + 1, best_k + 1,
                            base_accuracy, best_accuracy)

                flip_count += 1
                base_accuracy = best_accuracy

        return best_accuracy

    def _optimize_last_layer(self, X, y):
        """ This function optimizes the existing network by setting the coefs
        in the last layer to False and then greedily adding rules to the set
        until no improvement is achieved.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        best_accuracy : RuleNetworkClassifier
            Returns accuracy on training set (X, y).
        """
        self._class_logger.info('Optimizing last layer...')

        # set all coefs in the last layer to False
        for j in range(self.layer_units[self.n_layers - 2]):
            self.coefs_[self.n_layers - 2][j][0] = False

        # evaluate all flips to True in last layer
        base_accuracy = accuracy_score(y, self.predict(X))
        best_accuracy = base_accuracy
        best_j = None
        optimal = False
        while not optimal:
            optimal = True
            for j in range(self.layer_units[self.n_layers - 2]):
                if not self.coefs_[self.n_layers - 2][j][0]:
                    self.coefs_[self.n_layers - 2][j][0] = True
                    accuracy = accuracy_score(y, self.predict(X))
                    if accuracy > best_accuracy:
                        optimal = False
                        best_accuracy = accuracy
                        best_j = j
                    self.coefs_[self.n_layers - 2][j][0] = False
            if not optimal:
                self.coefs_[self.n_layers - 2][best_j][0] = True
                self._class_logger.debug(
                    'Connection/Rule %s added - accuracy increased from %s to '
                    '%s', best_j + 1, base_accuracy, best_accuracy)
                base_accuracy = best_accuracy
        self._class_logger.info('Training accuracy: %s', best_accuracy)
        return best_accuracy

    def _preprocess_classes(self, y):
        """ Store the classes seen during fit and choose target class based
        on parameter pos_class_method.

        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            The target values. An array of str.

        Returns
        -------
        pos_class : str
            The class value that will be converted to True (all others to
            False).
        """
        self.classes_, class_counts = np.unique(y, return_counts=True)
        self.n_classes_ = len(self.classes_)
        if self.pos_class_method == 'least-frequent':
            pos_class = self.classes_[class_counts.argmin()]
        elif self.pos_class_method == 'most-frequent':
            pos_class = self.classes_[class_counts.argmax()]
        else:
            pos_class = self.classes_[class_counts.argmin()]
        self.class_transformer_ = LabelBinarizer()
        self.y_ = self.class_transformer_.fit_transform(y).astype(bool)

        # Adjust y and class order in label binarizer for correct inverse
        # transformation
        if self.classes_[0] != pos_class:
            self.y_ = ~self.y_
            self.class_transformer_.classes_ = self.classes_[::-1]

        return str(pos_class)
