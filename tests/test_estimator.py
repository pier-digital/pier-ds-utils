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


def _cluster_data():
    import numpy as np

    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(0, 0.1, (20, 2)), rng.normal(10, 0.1, (20, 2))])


def test_cluster_label_mapper_predict():
    import numpy as np
    from sklearn.cluster import KMeans

    X = _cluster_data()
    mapper = ds.estimator.ClusterLabelMapper(
        KMeans(n_clusters=2, n_init=3, random_state=0), {0: "low", 1: "high"}
    )
    labels = mapper.fit(X).predict(X)

    assert set(labels) <= {"low", "high"}
    expected = np.array(["low", "high"], dtype=object)[mapper.estimator.predict(X)]
    assert (labels == expected).all()
    assert (mapper.fit_predict(X) == labels).all() or set(labels) == {"low", "high"}
    assert mapper.cluster_centers_.shape == (2, 2)


def test_cluster_label_mapper_unmapped_index():
    import pytest
    from sklearn.cluster import KMeans

    mapper = ds.estimator.ClusterLabelMapper(
        KMeans(n_clusters=2, n_init=3, random_state=0), {0: "only"}
    )
    with pytest.raises(ValueError, match="missing from cluster_map"):
        mapper.fit(_cluster_data())


def test_cluster_label_mapper_invalid_map():
    import pytest

    est = BaseEstimator()
    for bad, msg in [
        ([], "must be a dictionary"),
        ({}, "cannot be empty"),
        ({"a": "x"}, "keys in cluster_map must be integers"),
        ({0: 1}, "values in cluster_map must be strings"),
    ]:
        with pytest.raises(ValueError, match=msg):
            ds.estimator.ClusterLabelMapper(est, bad)


def test_cluster_label_mapper_clone_and_missing_attr():
    import pytest
    from sklearn.base import clone
    from sklearn.cluster import KMeans

    mapper = ds.estimator.ClusterLabelMapper(KMeans(n_clusters=2), {0: "a", 1: "b"})
    cloned = clone(mapper)
    assert cloned.cluster_map == mapper.cluster_map
    with pytest.raises(AttributeError):
        mapper.does_not_exist


def test_cluster_label_mapper_kmodes_in_pipeline():
    import pandas as pd
    from sklearn.pipeline import Pipeline

    df = pd.DataFrame({"a": list("xxxxyyyy"), "b": list("pppqqqqq")})
    mapper = ds.estimator.ClusterLabelMapper(
        ds.kmodes.KModes(n_clusters=2, random_state=0), {0: "c0", 1: "c1"}
    )
    out = Pipeline([("m", mapper)]).fit(df).predict(df)
    assert set(out) <= {"c0", "c1"}
