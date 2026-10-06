import typing
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pier_ds_utils.transformer import BaseCustomTransformer
from pier_ds_utils.prep import add_constant_column
from sklearn.base import BaseEstimator


class GLMWrapper(BaseCustomTransformer):
    def __init__(
        self, add_constant: bool = True, os_factor: np.float64 = 1.0, **init_params
    ):
        self._add_constant = add_constant
        self.os_factor = os_factor
        self.init_params = init_params

    def get_params(self, deep=True):
        return {
            **self.init_params,
            **{"add_constant": self._add_constant, "os_factor": self.os_factor},
        }

    def fit(self, X, y, **fit_params):
        if self._add_constant:
            X = add_constant_column(
                X, prepend=True, constant_value=1.0, column_name="const"
            )

        self.model_ = sm.GLM(endog=y, exog=X, **self.init_params)
        fit_method = fit_params.pop("fit_method", "fit")
        self.results_ = getattr(self.model_, fit_method)(**fit_params)
        return self

    def predict(self, X, **predict_params):
        if self._add_constant:
            X = add_constant_column(
                X, prepend=True, constant_value=1.0, column_name="const"
            )

        return self.results_.predict(exog=X, **predict_params) * self.os_factor


class PredictProbaSelector(BaseCustomTransformer):
    def __init__(self, model: BaseEstimator, column: typing.Union[str, int] = None):
        self.model = model
        self.column = column

    def predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self.model.predict_proba(X, **kwargs)[:, self.column].tolist()

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        return self.model.fit(X, y, **kwargs)

    def __getattr__(self, __name: str) -> Any:
        if not hasattr(super(), __name):
            return getattr(self.model, __name)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "model": self.model.get_params(deep=deep) if deep else self.model,
            "column": self.column,
        }


class ClusterLabelMapper(BaseCustomTransformer):
    def __init__(self, estimator: BaseEstimator, cluster_map: typing.Dict[int, str]):
        """
        Wraps a clustering estimator and translates the cluster indexes it
        predicts into human readable labels.

        Parameters
        ----------
        estimator : BaseEstimator
            Estimator whose `predict` returns integer cluster indexes.
        cluster_map : dict
            Mapping of cluster index (int) to label (str).
        """
        self._check_cluster_map(cluster_map)
        self.estimator = estimator
        self.cluster_map = cluster_map

    @staticmethod
    def _check_cluster_map(cluster_map: typing.Dict[int, str]) -> None:
        if not isinstance(cluster_map, dict):
            raise ValueError("cluster_map must be a dictionary.")

        if not cluster_map:
            raise ValueError("cluster_map cannot be empty.")

        if any(not isinstance(k, (int, np.integer)) for k in cluster_map):
            raise ValueError("All keys in cluster_map must be integers.")

        if any(not isinstance(v, str) for v in cluster_map.values()):
            raise ValueError("All values in cluster_map must be strings.")

    def _translate(self, indexes: typing.Any) -> np.ndarray:
        indexes = np.asarray(indexes)
        unmapped = sorted(set(indexes.tolist()) - set(self.cluster_map))
        if unmapped:
            raise ValueError(f"Cluster indexes missing from cluster_map: {unmapped}")

        return np.array([self.cluster_map[i] for i in indexes.tolist()], dtype=object)

    def fit(self, X, y=None, **fit_params) -> "ClusterLabelMapper":
        self.estimator.fit(X, y, **fit_params)
        if hasattr(self.estimator, "labels_"):
            self._translate(np.unique(self.estimator.labels_))
        self.fitted_ = True
        return self

    def predict(self, X, **predict_params) -> np.ndarray:
        return self._translate(self.estimator.predict(X, **predict_params))

    def fit_predict(self, X, y=None, **fit_params) -> np.ndarray:
        return self.fit(X, y, **fit_params).predict(X)

    def __getattr__(self, name: str) -> Any:
        # Only called when normal lookup fails; delegate fitted attributes
        # (e.g. cluster_centroids_) to the wrapped estimator.
        if name == "estimator" or name.startswith("__"):
            raise AttributeError(name)
        return getattr(self.estimator, name)
