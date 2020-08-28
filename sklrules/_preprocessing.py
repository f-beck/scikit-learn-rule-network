import pandas as pd
import re

from itertools import compress
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder as SklearnOneHotEncoder


class OneHotEncoder(SklearnOneHotEncoder):
    """
    Encode categorical features as a one-hot numeric array. Wrapper class of
    sklearn OneHotEncoder with handling of NaN columns and additional
    attributes.

    Attributes
    ----------
    attributes_ : list of str
        The names of each column (= attribute) determined during fitting (in
        order of the attributes in X).

    attribute_lengths_ : list of int
        The cardinality of the attributes, derived from ``categories_``
        attribute.

    features_ : list of str
        The names of each new column (=feature) determined after transforming
        (in order of the attributes in X and corresponding with the output of
        ``transform``).
    """

    def __init__(self, **kwargs):
        super(OneHotEncoder, self).__init__(**kwargs)

    def fit(self, X, **kwargs):
        """
        Fit OneHotEncoder to X. Additionally use Imputer to replace NaN with
        None values.
        """
        self.attributes_ = list(X)
        X = SimpleImputer(strategy='constant', copy=False).fit_transform(X)
        return super().fit(X)

    def transform(self, X, **kwargs):
        """
        Transform X using one-hot encoding. Additionally remove NaN indicator
        columns.
        """
        X = super(OneHotEncoder, self).transform(X)

        # remove NaN indicator columns
        idx = list(map(lambda x: bool(re.match(r'.*(?<!_missing_value)$', x)),
                       self.get_feature_names(self.attributes_)))
        feature_names_with_nan = [feature.replace('_', '=') for feature in
                                  self.get_feature_names(self.attributes_)]
        filtered_categories = list(map(lambda x: x[x != 'missing_value'],
                                       self.categories_))

        # update attributes accordingly
        self.attribute_lengths_ = list(map(len, filtered_categories))

        self.features_ = list(compress(feature_names_with_nan, idx))
        return X[:, idx]


class TypeSelector(BaseEstimator, TransformerMixin):
    """
    Simple selector class to filter columns of specific type of a
    pd.DataFrame. Compatible with :class:`sklearn.pipeline.Pipeline`.

    Parameters
    ----------
    dtype : type
        Desired dtype to filter.

    include : bool, default=True
        If True, the output will contain all columns of the specified dtype.
        If False, the output will contain all columns of other dtypes.
    """

    def __init__(self, dtype, include=True):
        self.dtype = dtype
        self.include = include

    # noinspection PyUnusedLocal
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        if self.include:
            return X.select_dtypes(include=[self.dtype])
        else:
            return X.select_dtypes(exclude=[self.dtype])
