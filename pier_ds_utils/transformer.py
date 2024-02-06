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
        X.loc[:, output_column] = output

        return X


class CustomIntervalCategorizer(BaseCustomTransformer):
    def __init__(
        self,
        column: str,
        intervals: typing.List[typing.Tuple[typing.Union[int, float]]],
        labels: typing.List[typing.Any],
        default_value: typing.Any = None,
        output_column: typing.Optional[str] = None,
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

        X.loc[:, self.get_output_column()] = output

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
        X.loc[:, output_column] = output

        return X
