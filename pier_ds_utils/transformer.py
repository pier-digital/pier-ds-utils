import operator
import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class BaseCustomTransformer(BaseEstimator, TransformerMixin):
    def set_output(self, transform: str = "pandas") -> BaseEstimator:
        return self


class CustomDiscreteCategorizer(BaseCustomTransformer):
    def __init__(
        self,
        column: str,
        categories: typing.List[typing.List[typing.Any]],
        labels: typing.List[typing.Any],
        default_value: typing.Any = None,
        output_column: typing.Optional[str] = None,
    ):
        """
        Transformer to categorize a column into custom categories.

        Parameters
        ----------
        column: str
            Name of the column to be transformed
        categories: list of lists
            List of categories to be used for categorization. Each category must be a list of elements.
        labels: list
            List of labels to be used for categorization. Must have the same length as categories.
        default_value: any
            Value to be used for missing values. If None, missing values will be kept as NaN.
        output_column: str
            Name of the output column. If None, the original column will be overwritten.
        """
        if len(categories) != len(labels):
            raise ValueError(
                "Number of categories must be the same as number of labels"
            )

        for category in categories:
            if not isinstance(category, list):
                raise TypeError("Each category must be a list")

        self._column = column
        self._categories = categories
        self._labels = labels
        self._default_value = default_value
        self._output_column = output_column

    @property
    def categories_(self) -> typing.List[typing.List[typing.Any]]:
        return self._categories

    @property
    def labels_(self) -> typing.List[typing.Any]:
        return self._labels

    @classmethod
    def from_dict(cls, categories: typing.Dict, **kwargs):
        return cls(list(categories.values()), list(categories.keys()), **kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "categories": self.categories_,
            "labels": self.labels_,
            "default_value": self._default_value,
            "output_column": self._output_column,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = X[self._column].copy()
        output = pd.Series(np.nan, index=X.index, dtype="object")

        for category, label in zip(self._categories, self._labels):
            output.loc[values.isin(category)] = label

        if self._default_value is not None:
            output.fillna(self._default_value, inplace=True)

        output_column = self._output_column or self._column
        X[output_column] = output

        return X


class CustomIntervalCategorizer(BaseCustomTransformer):
    def __init__(
        self,
        column: str,
        intervals: typing.List[typing.Tuple[typing.Union[int, float]]],
        labels: typing.List[typing.Any],
        default_value: typing.Any = None,
        output_column: typing.Optional[str] = None,
        output_type: typing.Union[str, type] = "object",
    ):
        """
        Custom transformer to categorize a numeric column into intervals.

        Parameters
        ----------
        column: str
            Name of the column to be transformed
        intervals: list of tuples
            List of intervals to be used for categorization. Each interval must be a tuple with two elements.
            The first element must be smaller than the second. The comparison is inclusive for the first element (>=)
            and exclusive for the second (<).
        labels: list
            List of labels to be used for categorization. Must have the same length as intervals.
        default_value: any
            Value to be used for missing values. If None, missing values will be kept as NaN.
        output_column: str
            Name of the output column. If None, the original column will be overwritten.
        output_type: str or type
            Dtype of the output column, any value accepted by pandas.Series.astype (e.g. "object", "category",
            "float", "string"). Defaults to "object". Unmatched values without a default_value stay NaN, so
            integer types that cannot hold NaN (e.g. "int") will raise.
        """
        if len(intervals) != len(labels):
            raise ValueError("Number of intervals must be the same as number of labels")

        for interval in intervals:
            if not isinstance(interval, tuple):
                raise TypeError("Each interval must be a tuple")

            if len(interval) != 2:
                raise ValueError("Each interval must have two elements")

            if not isinstance(interval[0], (int, float)) or not isinstance(
                interval[1], (int, float)
            ):
                raise TypeError("Each interval element must be a number")

            if interval[0] >= interval[1]:
                raise ValueError(
                    "Each interval must have the first element smaller than the second"
                )

        self._column = column
        self._intervals = intervals
        self._labels = labels
        self._default_value = default_value
        self._output_column = output_column
        self._output_type = output_type

    @property
    def column_(self) -> str:
        return self._column

    @property
    def intervals_(self) -> typing.List[typing.Tuple[typing.Union[int, float]]]:
        return self._intervals

    @property
    def labels_(self) -> typing.List[typing.Any]:
        return self._labels

    @property
    def default_value_(self) -> typing.Any:
        return self._default_value

    @property
    def output_column_(self) -> str:
        return self._output_column

    @property
    def output_type_(self) -> typing.Union[str, type]:
        return self._output_type

    def get_output_column(self) -> str:
        return self.output_column_ or self.column_

    @classmethod
    def from_dict(cls, intervals: typing.Dict, **kwargs):
        return cls(list(intervals.values()), list(intervals.keys()), **kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "intervals": self.intervals_,
            "labels": self.labels_,
            "default_value": self.default_value_,
            "output_column": self.output_column_,
            "column": self.column_,
            "output_type": self.output_type_,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        values = X[self.column_].astype(float).copy()
        output = pd.Series(np.nan, index=X.index, dtype="object")

        for interval, label in zip(self.intervals_, self.labels_):
            output.loc[(values >= interval[0]) & (values < interval[1]),] = label

        if self.default_value_ is not None:
            output.fillna(self.default_value_, inplace=True)

        X[self.get_output_column()] = output.astype(self.output_type_)

        return X


class CustomIntervalCategorizerByCategory(BaseCustomTransformer):
    def __init__(
        self,
        category_column: str,
        interval_categorizers: typing.Dict[str, CustomIntervalCategorizer],
        default_categorizer: typing.Optional[CustomIntervalCategorizer] = None,
        default_value: typing.Any = None,
        output_column: typing.Optional[str] = None,
    ):
        """
        Custom transformer to categorize a numeric column into intervals given a categorical column.

        Parameters
        ----------
        category_column: str
            Name of the column to be used for categorization
        interval_categorizers: dict
            Dictionary of interval categorizers to be used for categorization. Keys must be the categories of the
            category_column and values must be CustomIntervalCategorizer.
        default_categorizer: CustomIntervalCategorizer
            Categorizer to be used when value does not match any categorizer in interval_categorizers.
        default_value: any
            Value to be used for missing values. If None, missing values will be kept as NaN.
        output_column: str
            Name of the output column. If None, the original column will be overwritten.
        """
        if not isinstance(interval_categorizers, dict):
            raise TypeError("interval_categorizers must be a dict")

        for key, value in interval_categorizers.items():
            if not isinstance(key, str):
                raise TypeError("Keys of interval_categorizers must be strings")

            if not isinstance(value, CustomIntervalCategorizer):
                raise TypeError(
                    "Values of interval_categorizers must be CustomIntervalCategorizer"
                )

        self._category_column = category_column
        self._interval_categorizers = interval_categorizers
        self._default_categorizer = default_categorizer
        self._default_value = default_value
        self._output_column = output_column

    @property
    def category_column_(self) -> str:
        return self._category_column

    @property
    def interval_categorizers_(self) -> typing.Dict[str, CustomIntervalCategorizer]:
        return self._interval_categorizers

    @property
    def default_categorizer_(self) -> typing.Optional[CustomIntervalCategorizer]:
        return self._default_categorizer

    @classmethod
    def from_dict(cls, **kwargs):
        return cls(**kwargs)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "category_column": self.category_column_,
            "interval_categorizers": self.interval_categorizers_,
            "default_categorizer": self.default_categorizer_,
            "default_value": self._default_value,
            "output_column": self._output_column,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        output = pd.Series(np.nan, index=X.index, dtype="object")

        for category, interval_categorizer in self.interval_categorizers_.items():
            output.loc[
                X[self._category_column] == category
            ] = interval_categorizer.transform(
                X.loc[X[self._category_column] == category]
            )[interval_categorizer.get_output_column()]

        if self._default_categorizer is not None:
            output.loc[
                ~(X[self.category_column_].isin(self.interval_categorizers_.keys()))
            ] = self._default_categorizer.transform(
                X.loc[
                    ~(X[self.category_column_].isin(self.interval_categorizers_.keys()))
                ]
            )[self._default_categorizer.get_output_column()]

        if self._default_value is not None:
            output.fillna(self._default_value, inplace=True)

        output_column = self._output_column or self._category_column
        X[output_column] = output

        return X


_MATH_OPERATIONS = {
    "addition": operator.add,
    "subtraction": operator.sub,
    "multiplication": operator.mul,
    "division": operator.truediv,
    "exponentiation": operator.pow,
}


class CustomMathOperation(BaseCustomTransformer):
    _OPERATIONS = _MATH_OPERATIONS

    def __init__(
        self,
        operation: str,
        column_a: str,
        column_b: str,
        output_column: str,
    ):
        """
        Transformer to apply a math operation between two columns.

        Parameters
        ----------
        operation: str
            Operation to apply. One of "addition", "subtraction",
            "multiplication", "division", "exponentiation".
        column_a: str
            Name of the first operand column.
        column_b: str
            Name of the second operand column.
        output_column: str
            Name of the output column.
        """
        if operation not in self._OPERATIONS:
            raise ValueError(
                f"operation must be one of {list(self._OPERATIONS)}, got {operation!r}"
            )

        self._operation = operation
        self._column_a = column_a
        self._column_b = column_b
        self._output_column = output_column

    @property
    def operation_(self) -> str:
        return self._operation

    @property
    def column_a_(self) -> str:
        return self._column_a

    @property
    def column_b_(self) -> str:
        return self._column_b

    @property
    def output_column_(self) -> str:
        return self._output_column

    def get_output_column(self) -> str:
        return self._output_column

    def get_params(self, deep: bool = True) -> dict:
        return {
            "operation": self._operation,
            "column_a": self._column_a,
            "column_b": self._column_b,
            "output_column": self._output_column,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        op = self._OPERATIONS[self._operation]
        X[self.get_output_column()] = op(X[self._column_a], X[self._column_b])
        return X


class CustomMathOperationByConstant(BaseCustomTransformer):
    _OPERATIONS = _MATH_OPERATIONS

    def __init__(
        self,
        operation: str,
        column: str,
        constant: typing.Union[int, float],
        output_column: str,
        invert_order: bool = False,
    ):
        """
        Transformer to apply a math operation between a column and a constant.

        Parameters
        ----------
        operation: str
            Operation to apply. One of "addition", "subtraction",
            "multiplication", "division", "exponentiation".
        column: str
            Name of the operand column.
        constant: int or float
            Constant value to apply the operation with.
        output_column: str
            Name of the output column.
        invert_order: bool
            If True, applies `constant op column` instead of the default
            `column op constant`. Useful for non-commutative operations
            (subtraction, division, exponentiation). For exponentiation,
            True gives `constant ** column`. Defaults to False.
        """
        if operation not in self._OPERATIONS:
            raise ValueError(
                f"operation must be one of {list(self._OPERATIONS)}, got {operation!r}"
            )

        self._operation = operation
        self._column = column
        self._constant = constant
        self._output_column = output_column
        self._invert_order = invert_order

    @property
    def operation_(self) -> str:
        return self._operation

    @property
    def column_(self) -> str:
        return self._column

    @property
    def constant_(self) -> typing.Union[int, float]:
        return self._constant

    @property
    def output_column_(self) -> str:
        return self._output_column

    @property
    def invert_order_(self) -> bool:
        return self._invert_order

    def get_output_column(self) -> str:
        return self._output_column

    def get_params(self, deep: bool = True) -> dict:
        return {
            "operation": self._operation,
            "column": self._column,
            "constant": self._constant,
            "output_column": self._output_column,
            "invert_order": self._invert_order,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        op = self._OPERATIONS[self._operation]
        if self._invert_order:
            X[self.get_output_column()] = op(self._constant, X[self._column])
        else:
            X[self.get_output_column()] = op(X[self._column], self._constant)
        return X


class LogTransformer(BaseCustomTransformer):
    """Calculates the natural logarithm of the input data. This transformer is useful for transforming skewed data into a more normal distribution."""

    def fit(self, X, y=None):
        # No fitting needed for this transformer
        return self

    def transform(self, X):
        # Apply log transformation (ensure values are positive for log)
        return X.apply(np.log, axis=1)


class BoundariesTransformer(BaseCustomTransformer):
    def __init__(
        self,
        lower_bound: float,
        upper_bound: float,
        lower_value: float = None,
        upper_value: float = None,
    ):
        """
        Transformer to apply lower and upper boundaries to the data.

        This transformer replaces values that fall outside the specified
        lower and/or upper boundaries. If a replacement value is not provided,
        the corresponding boundary itself will be used as the replacement.

        Parameters
        ----------
        lower_bound : float
            Lower boundary (cut-off condition). Values strictly less than this
            threshold will be replaced.

        upper_bound : float
            Upper boundary (cut-off condition). Values strictly greater than this
            threshold will be replaced.

        lower_value : float, optional (default=None)
            Replacement value when ``X < lower_bound``.
            If None, uses ``lower_bound`` itself.

        upper_value : float, optional (default=None)
            Replacement value when ``X > upper_bound``.
            If None, uses ``upper_bound`` itself.

        Returns
        -------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Transformed data with values capped or floored according
            to the specified boundaries.
        """
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound
        self._lower_value = lower_value
        self._upper_value = upper_value

    def get_params(self, deep=True):
        return {
            "lower_bound": self._lower_bound,
            "upper_bound": self._upper_bound,
            "lower_value": self._lower_value,
            "upper_value": self._upper_value,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        replacement = (
            self._lower_value if self._lower_value is not None else self._lower_bound
        )
        X[X < self._lower_bound] = replacement

        replacement = (
            self._upper_value if self._upper_value is not None else self._upper_bound
        )
        X[X > self._upper_bound] = replacement

        return X


_BINARIZER_CONDITIONS = {
    ">": operator.gt,
    ">=": operator.ge,
    "=": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
}


class CustomBinarizer(BaseCustomTransformer):
    _CONDITIONS = _BINARIZER_CONDITIONS

    def __init__(
        self,
        threshold: float = 0.0,
        condition: str = ">",
        true_value: typing.Any = 1,
        false_value: typing.Any = 0,
    ):
        """
        Transformer to binarize data according to a threshold and a condition.

        Works like scikit-learn's Binarizer, but allows choosing the
        condition and the values assigned when it is met or not.

        Parameters
        ----------
        threshold : float, optional (default=0.0)
            Value the data is compared against.
        condition : str, optional (default=">")
            Condition applied as `X condition threshold`. One of ">", ">=",
            "=", "<", "<=".
        true_value : any, optional (default=1)
            Value assigned where the condition is met.
        false_value : any, optional (default=0)
            Value assigned where the condition is not met. Missing values
            never meet the condition, so they receive this value.

        Returns
        -------
        X : DataFrame of shape (n_samples, n_features)
            Binarized data, with the same index and columns as the input.
        """
        if condition not in self._CONDITIONS:
            raise ValueError(
                f"condition must be one of {list(self._CONDITIONS)}, got {condition!r}"
            )

        self._threshold = threshold
        self._condition = condition
        self._true_value = true_value
        self._false_value = false_value

    @property
    def threshold_(self) -> float:
        return self._threshold

    @property
    def condition_(self) -> str:
        return self._condition

    @property
    def true_value_(self) -> typing.Any:
        return self._true_value

    @property
    def false_value_(self) -> typing.Any:
        return self._false_value

    def get_params(self, deep: bool = True) -> dict:
        return {
            "threshold": self._threshold,
            "condition": self._condition,
            "true_value": self._true_value,
            "false_value": self._false_value,
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mask = self._CONDITIONS[self._condition](X, self._threshold)
        return pd.DataFrame(
            np.where(mask, self._true_value, self._false_value),
            index=X.index,
            columns=X.columns,
        )
