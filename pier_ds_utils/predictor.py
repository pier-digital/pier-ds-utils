import pandas as pd
import numpy as np
import typing
from pier_ds_utils.transformer import BaseCustomTransformer
import functools


class StaticGLM(BaseCustomTransformer):
    def __init__(
        self,
        coefficients_map: typing.Dict[str, typing.Union[int, float]],
        constant: typing.Optional[typing.Union[int, float]] = 0.0,
        os_factor: typing.Optional[typing.Union[int, float]] = 1.0,
    ) -> None:
        """
        A static Generalized Linear Model (GLM) created from a dictionary of coefficients.
        The model is used for making predictions based on a linear combination of features.

        Parameters
        ----------
        coefficients_map : dict
            A dictionary mapping feature names to their corresponding coefficients.
        constant : float, optional
            A constant term added to the linear combination of features. Default is 0.0.
        os_factor : float, optional
            A scaling factor applied to the predictions. Default is 1.0.
        """
        super().__init__()
        self.coefficients_map = coefficients_map
        self.constant = constant
        self.os_factor = os_factor
        self.fitted_ = True

    @property
    def coefficients_map(self) -> typing.Dict[str, float]:
        """
        Get the coefficients map.
        """
        return self._coefficients_map

    @coefficients_map.setter
    def coefficients_map(self, value: typing.Dict[str, float]) -> None:
        """
        Set the coefficients map.
        """
        if not isinstance(value, dict):
            raise ValueError("coefficients_map must be a dictionary.")

        if not value:
            raise ValueError("coefficients_map cannot be empty.")

        if any(not isinstance(k, str) for k in value.keys()):
            raise ValueError("All keys in coefficients_map must be strings.")

        if any(k == "" for k in value.keys()):
            raise ValueError("All keys in coefficients_map must be non-empty strings.")

        self._coefficients_map = {}

        for key, val in value.items():
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"Value for {key} in coefficients_map must be a number."
                )
            self._coefficients_map[key] = float(val)

    @property
    def constant(self) -> float:
        """
        Get the constant.
        """
        return self._constant

    @constant.setter
    def constant(self, value: float) -> None:
        """
        Set the constant.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("constant must be a number.")
        self._constant = float(value)

    @property
    def os_factor(self) -> float:
        """
        Get the os_factor.
        """
        return self._os_factor

    @os_factor.setter
    def os_factor(self, value: float) -> None:
        """
        Set the os_factor.
        """
        if not isinstance(value, (int, float)):
            raise ValueError("os_factor must be a number.")
        self._os_factor = float(value)

    @functools.cached_property
    def feature_names(self) -> typing.List[str]:
        """
        Get the feature names.
        """
        return list(self.coefficients_map.keys())

    @functools.cached_property
    def coefficients_(self) -> np.ndarray:
        """
        Get the coefficients.
        """
        return np.array(list(self.coefficients_map.values()), dtype=np.float64)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "StaticGLM":
        """
        Fit the model.
        """
        return self

    @staticmethod
    def _check_prediction_input(
        X: pd.DataFrame, required_cols: typing.List[str]
    ) -> None:
        """
        Check if the input DataFrame contains all required columns.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame.")

        missing_cols = set(required_cols) - set(X.columns)
        if missing_cols:
            raise ValueError(
                f"Input DataFrame is missing the following columns: {missing_cols}"
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict using the model.
        """
        self._check_prediction_input(X, self.feature_names)

        X = X.copy()

        # Ensure the DataFrame contains only the required columns
        X = X[self.feature_names]

        # Calculate the linear combination of features and coefficients
        linear_combination = np.dot(X.values, self.coefficients_)

        # Apply the os_factor and constant
        predictions = self.os_factor * (linear_combination + self.constant)

        return predictions
