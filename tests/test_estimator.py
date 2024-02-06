import pier_ds_utils as ds
from sklearn.base import BaseEstimator


def test_glm_wrapper():
    wrapper = ds.estimator.GLMWrapper()

    assert wrapper is not None

    # Check attributes
    assert hasattr(wrapper, "os_factor")
    assert hasattr(wrapper, "init_params")

    # Check methods
    assert hasattr(wrapper, "fit")
    assert hasattr(wrapper, "predict")
    assert hasattr(wrapper, "get_params")


def test_predict_proba_selector():
    selector = ds.estimator.PredictProbaSelector(
        model=BaseEstimator(),
    )

    assert selector is not None

    # Check attributes
    assert hasattr(selector, "model")
    assert hasattr(selector, "column")

    # Check methods
    assert hasattr(selector, "fit")
    assert hasattr(selector, "predict_proba")
    assert hasattr(selector, "get_params")
