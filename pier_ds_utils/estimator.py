import typing
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm

from pier_ds_utils.transformer import BaseCustomTransformer
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
            X["const"] = 1

        self.model_ = sm.GLM(endog=y, exog=X, **self.init_params)
        fit_method = fit_params.pop("fit_method", "fit")
        self.results_ = getattr(self.model_, fit_method)(**fit_params)
        return self

    def predict(self, X, **predict_params):
        if self._add_constant:
            X["const"] = 1

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
