from pier_ds_utils.predictors import StaticGLM
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
import pytest


def test_create_static_glm():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = 2.0
    os_factor = 1.0

    glm = StaticGLM(
        coefficients_map=coefficients_map, constant=constant, os_factor=os_factor
    )

    assert glm.coefficients_map == coefficients_map
    assert glm.constant == constant
    assert glm.os_factor == os_factor
    assert glm.feature_names == list(coefficients_map.keys())
    assert isinstance(glm.coefficients_, np.ndarray)
    assert glm.coefficients_.tolist() == list(coefficients_map.values())
    assert glm.coefficients_.dtype == np.float64
    assert glm.coefficients_.shape == (2,)


@pytest.mark.parametrize(
    "invalid_coefficients_map, expected_message",
    [
        ("", "coefficients_map must be a dictionary."),
        ([], "coefficients_map must be a dictionary."),
        (None, "coefficients_map must be a dictionary."),
        ({}, "coefficients_map cannot be empty."),
        ({"feature1": 0.5, 1: 1.5}, "All keys in coefficients_map must be strings."),
        (
            {"feature1": 0.5, "": 1.5},
            "All keys in coefficients_map must be non-empty strings.",
        ),
        (
            {"feature1": 0.5, "feature2": None},
            "Value for feature2 in coefficients_map must be a number.",
        ),
        (
            {"feature1": "1", "feature2": 0.5},
            "Value for feature1 in coefficients_map must be a number.",
        ),
    ],
)
def test_create_static_glm_with_invalid_coefficients(
    invalid_coefficients_map, expected_message
):
    with pytest.raises(ValueError, match=expected_message):
        StaticGLM(coefficients_map=invalid_coefficients_map)


def test_create_static_glm_with_invalid_constant():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = "invalid"

    with pytest.raises(ValueError, match="constant must be a number."):
        StaticGLM(coefficients_map=coefficients_map, constant=constant)


def test_create_static_glm_with_invalid_os_factor():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    os_factor = "invalid"

    with pytest.raises(ValueError, match="os_factor must be a number."):
        StaticGLM(coefficients_map=coefficients_map, os_factor=os_factor)


def test_static_glm_fit():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = 2.0
    os_factor = 1.0

    glm = StaticGLM(
        coefficients_map=coefficients_map, constant=constant, os_factor=os_factor
    )

    assert glm.fit(None, None) is glm


def test_static_glm_check_prediction_input_with_invalid_type():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = 2.0
    os_factor = 1.0

    glm = StaticGLM(
        coefficients_map=coefficients_map, constant=constant, os_factor=os_factor
    )

    with pytest.raises(ValueError, match="Input must be a pandas DataFrame."):
        glm._check_prediction_input("invalid_input", ["example"])


def test_static_glm_check_prediction_input_with_missing_columns():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = 2.0
    os_factor = 1.0

    glm = StaticGLM(
        coefficients_map=coefficients_map, constant=constant, os_factor=os_factor
    )

    df = pd.DataFrame({"feature1": [1, 2], "feature3": [3, 4]})

    with pytest.raises(
        ValueError,
        match="Input DataFrame is missing the following columns: {'feature2'}",
    ):
        glm._check_prediction_input(df, ["feature1", "feature2"])


def test_static_glm_predict():
    coefficients_map = {"feature1": 0.5, "feature2": 1.5}
    constant = 2.0
    os_factor = 1.0

    glm = StaticGLM(
        coefficients_map=coefficients_map, constant=constant, os_factor=os_factor
    )

    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    expected_prediction = np.array([7.0, 9.0])  # (1*0.5 + 3*1.5 + 2, 2*0.5 + 4*1.5 + 2)

    prediction = glm.predict(df)

    assert isinstance(prediction, np.ndarray)
    assert prediction.tolist() == expected_prediction.tolist()
    assert prediction.dtype == np.float64

    assert prediction.shape == (2,)


def test_static_glm_inside_sklearn_pipeline():
    """
    Test the StaticGLM inside a sklearn pipeline.
    """
    glm = StaticGLM(coefficients_map={"feature1": 0.5, "feature2": 1.5})

    pipeline = Pipeline([("glm", glm)])

    df = pd.DataFrame({"feature1": [1, 2], "feature2": [3, 4]})
    expected_prediction = np.array([5.0, 7.0])  # (1*0.5 + 3*1.5, 2*0.5 + 4*1.5)

    prediction = pipeline.predict(df)

    assert isinstance(prediction, np.ndarray)
    assert prediction.tolist() == expected_prediction.tolist()
    assert prediction.dtype == np.float64
