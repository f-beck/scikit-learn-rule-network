"""
This is a module holding the rule network class
"""
import logging
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class RuleNetworkClassifier(BaseEstimator, ClassifierMixin):
    """ A classifier which uses a network structure to learn rule sets.

    Parameters
    ----------
    init_method : {'probabilistic', 'support'}, default='support'
        The method used to initialize the rules. Must be 'probabilistic' to
        initialize each attribute of a rule with a fixed probability or
        'support' if each rule should cover a fixed percentage of the samples.

    n_rules : int, default=10
        The maximum number of rules in the network.

    min_support : int, default=10
        The minimum number of samples covered by an initial rule. Only has an
        effect if init_method='support'.

    avg_rule_length : int, default=3
        The average number of conditions in an initial rule. Each attribute
        is set to a random value and added to the rule with a probability of
        3/|A| with |A| being the number of attributes. Only has an effect if
        init_method='probabilistic'.

    batch_size : int, default=50
        The number of samples per mini-batch.

    max_flips : int, default=2
        The maximum number of flips per mini-batch.

    max_rule_set_size : int, default=10
        The maximum number of rules in the final rule set.

    rule_head_class : {'least-frequent', 'most-frequent'},
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
        The input passed during :meth:`fit`.

    y_ : ndarray, shape (n_samples,)
        The labels passed during :meth:`fit`.

    classes_ : ndarray, shape (n_classes,)
        The classes seen at :meth:`fit`.

    n_batches_ : int
        The number of batches used during :meth:`fit`.

    n_attributes_ : int
        The number of attributes seen at :meth:`fit`.

    attribute_lengths_ : ndarray, shape (n_attributes,)
        The number of unique values per attribute.

    attribute_lengths_cumsum_ : ndarray, shape (n_attributes,)
        The cumulative sum of attribute_lengths_, used as indexes for X_.

    n_features_ : int
        The number of features seen at :meth:`fit`.

    feature_names_ : list of str of shape (n_features,)
        String names for (boolean) features.

    target_class_name_ : str
        String name for output feature.

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
    """

    def __init__(self, init_method='support', n_rules=10, min_support=10,
                 avg_rule_length=3, batch_size=50, max_flips=2,
                 max_rule_set_size=10, rule_head_class='least-frequent',
                 interim_train_accuracies=True, random_state=None):
        self.init_method = init_method
        self.n_rules = n_rules
        self.min_support = min_support
        self.avg_rule_length = avg_rule_length
        self.batch_size = batch_size
        self.max_flips = max_flips
        self.max_rule_set_size = max_rule_set_size
        self.init_method = init_method
        self.rule_head_class = rule_head_class
        self.interim_train_accuracies = interim_train_accuracies
        self.random_state = random_state

    def _more_tags(self):
        return {'allow_nan': True,
                'binary_only': True}

    def fit(self, X, y):
        """ The fitting function creates binary layers adjusted to the
        size of the input (n_features). It learns a model by flipping
        the boolean values to create suitable rules.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_attributes)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns fitted classifier.
        """

        # Check that X and y have correct shape
        X, y = check_X_y(X, y)

        # Calculate number of batches
        if self.batch_size > X.shape[0]:
            self.batch_size = X.shape[0]
            logging.warning('Batch size was higher than number of training '
                            'samples. Use single batch of size %s for '
                            'training.', self.batch_size)
        self.n_batches_ = X.shape[0] // self.batch_size

        # Preprocess X and y
        X = self.__preprocess_X(X)
        y = self.__preprocess_y(y)

        # Initialize rule network layers
        self.__init_layers(X)

        # Initialize arrays for storing accuracies
        self.batch_accuracies_ = np.empty(shape=(self.n_batches_ + 2,))
        self.batch_accuracies_[0] = accuracy_score(y, self.predict(X))
        if self.interim_train_accuracies:
            self.train_accuracies_ = np.empty_like(self.batch_accuracies_)
            self.train_accuracies_[0] = self.batch_accuracies_[0]

        logging.info('Training network...')
        for batch in range(self.n_batches_):
            logging.debug('Processing mini-batch %s of %s...', batch + 1,
                          self.n_batches_)
            X_mini = X[batch * self.batch_size:(batch + 1) * self.batch_size]
            y_mini = y[batch * self.batch_size:(batch + 1) * self.batch_size]
            self.batch_accuracies_[batch + 1] = self.__optimize_rules(X_mini,
                                                                      y_mini)
            if self.interim_train_accuracies:
                self.train_accuracies_[batch + 1] = self.predict(X)

        self.batch_accuracies_[self.n_batches_ + 2] = \
            self.__optimize_rule_set(X, y)
        if self.interim_train_accuracies:
            self.train_accuracies_[self.n_batches_ + 2] = \
                self.batch_accuracies_[self.n_batches_ + 2]

        logging.info('Training finished.')

        # Return the classifier
        return self

    def predict(self, X):
        """ A reference implementation of a prediction for a classifier.

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
        # Check is fit had been called
        check_is_fitted(self)

        # Input validation
        X = check_array(X, dtype='bool')

        return ~(~X @ self.and_layer_) @ self.or_layer_

    def __init_layers(self, X=None):
        """ Initialize the layers of the network according to the
        initialization parameter settings.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples, used to determine rules with minimal support.

        Returns
        -------
        self : RuleNetworkClassifier
            Returns classifier with initialized network.
        """
        logging.info('Initializing network...')
        self.and_layer_ = np.zeros((self.n_features_, self.n_rules), dtype=bool)
        if self.init_method == 'support':
            for j in range(self.n_rules):
                attribute_order = np.random.permutation(self.n_attributes_)
                for i in attribute_order:
                    feature_order = np.random.permutation(
                        self.attribute_lengths_[i])
                    for feature in feature_order:
                        self.and_layer_[self.attribute_lengths_cumsum_[i] -
                                        feature - 1][j] = True
                        if self.__get_rule_support(X, j) > self.min_support:
                            break
                        else:
                            self.and_layer_[self.attribute_lengths_cumsum_[i]
                                            - feature - 1][j] = False
        elif self.init_method == 'probabilistic':
            if self.avg_rule_length > self.n_attributes_:
                self.avg_rule_length = self.n_attributes_
                logging.warning('Average rule length was higher than number of '
                                'attributes. Use number of attributes as '
                                'average rule length.', self.batch_size)
            for j in range(self.n_rules):
                for i in range(self.n_attributes_):
                    if np.random.random() < (self.avg_rule_length /
                                             self.n_attributes_):
                        random_feature = np.random.randint(
                            self.attribute_lengths_[i])
                        self.and_layer_[self.attribute_lengths_cumsum_[i] -
                                        random_feature - 1][j] = True
        self.or_layer_ = np.ones((self.n_rules, 1), dtype=bool)
        logging.info('Initialization finished.')
        return self

    def __preprocess_X(self, X):
        """ One-hot-encode all non-numeric columns and store attribute and
        feature names. Expects X to be a pandas DataFrame.

        Parameters
        ----------
        X : pd.DataFrame, shape (n_samples, n_features)
            The training input samples.

        Returns
        -------
        X_ : ndarray, shape (n_samples, n_features)
            The categorical, one-hot-encoded features of the training samples.
        """
        assert isinstance(X, pd.DataFrame)
        X = X.select_dtypes(exclude='number').astype('object')
        input_features = list(X)
        self.n_attributes_ = X.shape[1]
        one_hot_encoder = OneHotEncoder(sparse=False, dtype=np.bool,
                                        handle_unknown='ignore')
        self.X_ = one_hot_encoder.fit_transform(X)
        self.n_features_ = X.shape[1]
        self.feature_names_ = [feature.replace('_', '=') for feature in
                               one_hot_encoder.get_feature_names(
                                   input_features)]
        self.attribute_lengths_ = list(map(len, one_hot_encoder.categories_))
        self.attribute_lengths_cumsum_ = np.cumsum(self.attribute_lengths_)
        return X

    def __preprocess_y(self, y):
        """ Store the classes seen during fit and choose target class based
        on parameter rule_head_class. Expects y to be a pandas DataFrame.

        Parameters
        ----------
        y : array-like, shape (n_samples,)
            The target values. An array of int.

        Returns
        -------
        y_ : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        assert isinstance(y, pd.DataFrame)
        output_feature = list(y)
        self.classes_, class_counts = np.unique(y, return_counts=True)
        if self.rule_head_class == 'least-frequent':
            target_class = self.classes_[class_counts.argmin()]
        elif self.rule_head_class == 'most-frequent':
            target_class = self.classes_[class_counts.argmax()]
        else:
            target_class = self.classes_[class_counts.argmin()]
        self.y_ = y == target_class
        self.target_class_name_ = output_feature[0] + '=' + str(target_class)

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
                    logging.debug('Rule %s - replaced %s with %s - accuracy '
                                  'increased from %s to %s', best_rule + 1,
                                  self.feature_names_[dependent_features[0]],
                                  self.feature_names_[best_feature],
                                  base_accuracy, best_accuracy)
                elif self.and_layer_[best_feature][best_rule]:
                    logging.debug('Rule %s - added %s - accuracy increased '
                                  'from %s to %s', best_rule + 1,
                                  self.feature_names_[best_feature],
                                  base_accuracy, best_accuracy)
                else:
                    logging.debug('Rule %s - removed %s - accuracy increased '
                                  'from %s to %s', best_rule + 1,
                                  self.feature_names_[best_feature],
                                  base_accuracy, best_accuracy)
                base_accuracy = best_accuracy
        logging.debug('Rule optimization finished.')
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
        logging.debug('Optimizing rule set...')
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
                    logging.debug('Rule %s added - accuracy increased from '
                                  '%s to %s', best_rule + 1, base_accuracy,
                                  best_accuracy)
                else:
                    logging.debug('Rule %s added - accuracy increased from '
                                  '%s to %s', best_rule + 1, base_accuracy,
                                  best_accuracy)
                base_accuracy = best_accuracy
        logging.info('Training accuracy: %s', best_accuracy)
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
