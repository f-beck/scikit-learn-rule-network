"""
This is a module holding the rule network class
"""
import logging
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted, \
    check_random_state


class RuleNetworkClassifier(BaseEstimator, ClassifierMixin):
    """ A classifier which uses a network structure to learn rule sets.

    Parameters
    ----------
    init_method : {'probabilistic', 'ripper', 'support'},
    default='probabilistic'
        The method used to initialize the rules. Must be 'probabilistic' to
        initialize each attribute of a rule with a fixed probability,
        'ripper' to initialize the rule set with the one learned by RIPPER or
        'support' if each rule should cover a fixed percentage of the samples.

    n_rules : int, default=10
        The maximum number of rules in the network. Will be set automatically
        when init_method='ripper'.

    avg_rule_length : int, default=3
        The average number of conditions in an initial rule. Each attribute
        is set to a random value and added to the rule with a probability of
        3/|A| with |A| being the number of attributes. Only has an effect if
        init_method='probabilistic'.

    ripper_model : wittgenstein.base.Ruleset, default=None
        A ruleset computed by the RIPPER implementation wittgenstein. Only
        has an effect if init_method='ripper'.

    min_support : int, default=0.01
        The minimum percentage of samples covered by an initial rule. Only
        has an effect if init_method='support'.

    batch_size : int, default=50
        The number of samples per mini-batch.

    max_flips : int, default=2
        The maximum number of flips per mini-batch.

    max_rule_set_size : int, default=10
        The maximum number of rules in the final rule set.

    pos_class_method : {'least-frequent', 'most-frequent'},
    default='least-frequent'
        The class chosen to be converted to True and to be the head of the
        generated rules.

    interim_train_accuracies : bool, default=True
        If True, after each batch the accuracy value on the training set will
        be measured and stored. Set False to save runtime.

    random_state : int, default=None
        A random number generator instance to define the state of the
        random permutations generator.

    Attributes
    ----------
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
        The number of classes seen at :meth:`__preprocess_classes`.

    classes_ : list[str], shape (n_classes,)
        String names for the classes seen at :meth:`__preprocess_classes`.

    output_feature_ : str
        String name for output feature.

    n_batches_ : int
        The number of batches used during :meth:`fit`.

    and_layer_ : ndarray, shape (n_features, n_rules)
        The rules learned at :meth:`fit` as array.

    or_layer_ : ndarray, shape (n_rules,)
        The rule set learned at :meth:`fit` as array.

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

    def __init__(self, init_method='probabilistic', n_rules=10,
                 avg_rule_length=3, ripper_model=None, min_support=0.01,
                 batch_size=50, max_flips=2, max_rule_set_size=10,
                 pos_class_method='least-frequent',
                 interim_train_accuracies=True, random_state=None):
        self._class_logger = logging.getLogger(__name__).getChild(
            self.__class__.__name__)
        self.init_method = init_method
        self.n_rules = n_rules
        self.avg_rule_length = avg_rule_length
        self.ripper_model = ripper_model
        self.min_support = min_support
        self.batch_size = batch_size
        self.max_flips = max_flips
        self.max_rule_set_size = max_rule_set_size
        self.pos_class_method = pos_class_method
        self.interim_train_accuracies = interim_train_accuracies
        self.random_state = random_state

        # Override number of rules if network is initialized with RIPPER model
        if init_method == 'ripper' and ripper_model is not None:
            self.n_rules = len(self.ripper_model)

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
        pos_class = self.__preprocess_classes(y)
        self.output_feature_ = target + '=' + pos_class

        # Initialize attributes
        self.X_ = X
        self.n_attributes_ = len(attributes)
        self.attributes_ = attributes
        self.attribute_lengths_ = attribute_lengths
        self.attribute_lengths_cumsum_ = np.cumsum(self.attribute_lengths_)
        self.n_features_ = len(features)
        self.features_ = features

        # Calculate number of batches
        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]
            self._class_logger.warning('Batch size was higher than number of '
                                       'training samples. Use single batch of '
                                       'size %s for training.', self.batch_size)
        self.n_batches_ = X.shape[0] // self.batch_size

        # Initialize rule network layers
        pos_mask = y == pos_class
        X_pos = X[pos_mask]
        self.__init_layers(X_pos)

        # Initialize arrays for storing accuracies
        self.batch_accuracies_ = np.empty(shape=(self.n_batches_ + 2,))
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
            self.batch_accuracies_[batch + 1] = self.__optimize_rules(X_mini,
                                                                      y_mini)
            if self.interim_train_accuracies:
                self.train_accuracies_[batch + 1] = accuracy_score(
                    y, self.predict(X))

        self.batch_accuracies_[self.n_batches_ + 1] = \
            self.__optimize_rule_set(X, y)
        if self.interim_train_accuracies:
            self.train_accuracies_[self.n_batches_ + 1] = \
                self.batch_accuracies_[self.n_batches_ + 1]

        self._class_logger.info('Training finished.')
        self.print_model()

        # Return the classifier
        return self

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
        X = check_array(X, dtype='bool')

        y = (~(~X @ self.and_layer_) @ self.or_layer_).ravel()
        return self.class_transformer_.inverse_transform(y)

    def print_model(self):
        """ Print all rules in the model. """

        # Check if fit had been called
        check_is_fitted(self)

        model_string = '\n--- MODEL ---'
        for rule in range(self.n_rules):
            if self.or_layer_[rule]:
                model_string += '\nRule {} - {} :- '.format(
                    rule + 1, self.output_feature_)
                first = True
                for feature in range(self.n_features_):
                    if self.and_layer_[feature][rule]:
                        if first:
                            first = False
                        else:
                            model_string += ', '
                        model_string += self.features_[feature]
                model_string += '.'
        self._class_logger.info(model_string)

    def __init_layers(self, X=None):
        """ Initialize the layers of the network according to the
        initialization parameter settings.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The positive input samples, used to determine rules with minimal
            support.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns classifier with initialized network.
        """
        self._class_logger.info('Initializing network...')
        self.random_state_ = check_random_state(self.random_state)
        self.and_layer_ = np.zeros((self.n_features_, self.n_rules), dtype=bool)
        if self.init_method == 'probabilistic':
            if self.avg_rule_length > self.n_attributes_:
                self.avg_rule_length = self.n_attributes_
                self._class_logger.warning('Average rule length was higher '
                                           'than number of attributes. Use '
                                           'number of attributes as average '
                                           'rule length.')
            for j in range(self.n_rules):
                for i in range(self.n_attributes_):
                    if self.random_state_.random() < (self.avg_rule_length /
                                                      self.n_attributes_):
                        random_feature = self.random_state_.randint(
                            self.attribute_lengths_[i])
                        self.and_layer_[self.attribute_lengths_cumsum_[i] -
                                        random_feature - 1][j] = True
        elif self.init_method == 'ripper':
            for j, rule in enumerate(self.ripper_model):
                for cond in rule.conds:
                    try:
                        feature = self.features_.index(str(cond))
                        self.and_layer_[feature][j] = True
                    except ValueError:
                        pass  # ignore missing features
        elif self.init_method == 'support':
            for j in range(self.n_rules):
                attribute_order = self.random_state_.permutation(
                    self.n_attributes_)
                for i in attribute_order:
                    feature_order = self.random_state_.permutation(
                        self.attribute_lengths_[i])
                    for feature in feature_order:
                        self.and_layer_[self.attribute_lengths_cumsum_[i] -
                                        feature - 1][j] = True
                        self._class_logger.debug(self.features_[
                                     self.attribute_lengths_cumsum_[i]
                                     - feature - 1])
                        self._class_logger.debug(self.__get_rule_support(X, j))
                        if self.__get_rule_support(X, j) > self.min_support:
                            break
                        else:
                            self.and_layer_[self.attribute_lengths_cumsum_[i]
                                            - feature - 1][j] = False
        self.or_layer_ = np.ones((self.n_rules, 1), dtype=bool)
        self.print_model()
        self._class_logger.info('Initialization finished.')
        return self

    def __preprocess_classes(self, y):
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

    def __optimize_rules(self, X, y):
        """ This function optimizes the existing rules of the network to the
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
        best_feature = None
        best_rule = None
        optimal = False
        flip_count = 0
        while not optimal and flip_count < self.max_flips:
            optimal = True
            for rule in range(self.n_rules):
                for attribute in range(self.n_attributes_):
                    first_feature, last_feature = \
                        self.__get_features_of_attribute(attribute)
                    for feature in range(first_feature, last_feature):
                        dependent_features = self.__flip_feature(rule, feature)
                        accuracy = accuracy_score(y, self.predict(X))
                        if accuracy > best_accuracy:
                            optimal = False
                            best_accuracy = accuracy
                            best_feature = feature
                            best_rule = rule
                        self.__flip_feature(rule, feature, dependent_features)
            if not optimal:
                dependent_features = self.__flip_feature(best_rule,
                                                         best_feature)
                flip_count += 1
                if dependent_features.size:
                    self._class_logger.debug('Rule %s - replaced %s with %s - '
                                             'accuracy increased from %s to %s',
                                             best_rule + 1, self.features_
                                             [dependent_features[0]],
                                             self.features_[best_feature],
                                             base_accuracy, best_accuracy)
                elif self.and_layer_[best_feature][best_rule]:
                    self._class_logger.debug('Rule %s - added %s - accuracy '
                                             'increased from %s to %s',
                                             best_rule + 1,
                                             self.features_[best_feature],
                                             base_accuracy, best_accuracy)
                else:
                    self._class_logger.debug('Rule %s - removed %s - accuracy '
                                             'increased from %s to %s',
                                             best_rule + 1,
                                             self.features_[best_feature],
                                             base_accuracy, best_accuracy)
                base_accuracy = best_accuracy
        self._class_logger.debug('Rule optimization finished.')
        return best_accuracy

    def __optimize_rule_set(self, X, y):
        """ This function optimizes the existing rule set by creating an
        empty rule set and greedily adding rules to the set until either no
        improvement is achieved or max_rule_set_size is reached.

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
        self._class_logger.debug('Optimizing rule set...')
        self.or_layer_.fill(False)
        base_accuracy = accuracy_score(y, self.predict(X))
        best_accuracy = base_accuracy
        best_rule = None
        optimal = False
        rule_count = 0
        while not optimal and rule_count < self.max_rule_set_size:
            optimal = True
            for rule in range(self.n_rules):
                self.__flip_rule(rule)
                accuracy = accuracy_score(y, self.predict(X))
                if accuracy > best_accuracy:
                    optimal = False
                    best_accuracy = accuracy
                    best_rule = rule
                self.__flip_rule(rule)
            if not optimal:
                self.__flip_rule(best_rule)
                rule_count += 1
                if self.or_layer_[best_rule]:
                    self._class_logger.debug('Rule %s added - accuracy '
                                             'increased from %s to %s',
                                             best_rule + 1, base_accuracy,
                                             best_accuracy)
                else:
                    self._class_logger.debug('Rule %s added - accuracy '
                                             'increased from %s to %s',
                                             best_rule + 1, base_accuracy,
                                             best_accuracy)
                base_accuracy = best_accuracy
        self._class_logger.info('Training accuracy: %s', best_accuracy)
        return best_accuracy

    def __get_rule_support(self, X, rule):
        """ Calculates the percentage of samples in X that are covered by a
        rule.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        rule : int
            The index of the rule in and_layer_.

        Returns
        -------
        y : float
            The support of the rule on set X.
        """
        return np.count_nonzero(~(~X @ self.and_layer_[:, rule])) / X.shape[0]

    def __flip_feature(self, rule, feature, dependent_features=np.array([])):
        """ Flips the value in the and layer if a feature belongs or does not
        belong to a rule. Other features of the same attribute are flipped
        accordingly to ensure that only one feature per attribute is True.

        Parameters
        ----------
        rule : int
            The index of the rule to adjust.
        feature : int
            The index of the feature to flip.
        dependent_features : ndarray, shape (1,)
            Features of the same attribute set to True (should only be one).

        Returns
        -------
        dependent_features : ndarray, shape (1,)
            Features of the same attribute flipped from True to False (should
            only be one).
        """
        if not self.and_layer_[feature][rule]:
            attribute = np.searchsorted(self.attribute_lengths_cumsum_, feature,
                                        side='right')
            first_feature, last_feature = self.__get_features_of_attribute(
                attribute)
            dependent_features = \
                np.where(self.and_layer_[first_feature:last_feature, rule])[
                    0] + first_feature
        if dependent_features.size:
            for dependent_feature in dependent_features:
                self.and_layer_[dependent_feature][rule] = \
                    not self.and_layer_[dependent_feature][rule]
        self.and_layer_[feature][rule] = not self.and_layer_[feature][rule]
        return dependent_features

    def __get_features_of_attribute(self, attribute):
        """ Returns the indices of the first and the last feature of an
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

    def __flip_rule(self, rule):
        """ Flips the value in the or layer if a rule belongs or does not
        belong to the final rule set.

        Parameters
        ----------
        rule : int
            The index of the rule to be flipped.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns classifier with new rule set.
        """
        self.or_layer_[rule] = not self.or_layer_[rule]
        return self
