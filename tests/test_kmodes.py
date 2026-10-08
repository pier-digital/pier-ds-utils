import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

import pier_ds_utils as ds
from pier_ds_utils.kmodes.dissim import euclidean_dissim, matching_dissim, ng_dissim
from pier_ds_utils.kmodes.util import (
    apply_encoding,
    decode_categorical,
    encode_categorical,
    resolve_columns,
)


def make_df(n=240, seed=0):
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 3, n)

    def noisy(p=0.9):
        return np.where(rng.random(n) < p, g, rng.integers(0, 3, n)).astype(str)

    df = pd.DataFrame(
        {
            "a": noisy(),
            "b": noisy(),
            "x": g * 5 + rng.normal(size=n),
            "y": g * 3 + rng.normal(size=n),
        }
    )
    return df, g


def purity(g, labels):
    return pd.crosstab(g, labels).values.max(axis=1).sum() / len(g)


@pytest.mark.parametrize("init", ["Huang", "Cao", "random"])
def test_kmodes_init_methods(init):
    df, g = make_df()
    km = ds.kmodes.KModes(3, init=init, n_init=3, random_state=0)
    km.fit(df[["a", "b"]])
    assert km.labels_.shape == (len(df),)
    assert km.cluster_centroids_.shape == (3, 2)
    assert km.epoch_costs_[-1] <= km.epoch_costs_[0]
    assert purity(g, km.labels_) > 0.8


def test_kmodes_uses_all_columns():
    df, _ = make_df()
    X = df[["a", "b"]]
    from_df = ds.kmodes.KModes(3, random_state=0).fit(X)
    from_array = ds.kmodes.KModes(3, random_state=0).fit(X.to_numpy())
    assert from_df.categorical_idx_ == [0, 1]
    assert from_df.numeric_idx_ == []
    assert (from_df.labels_ == from_array.labels_).all()
    assert (from_df.cluster_centroids_ == from_array.cluster_centroids_).all()
    assert (from_df.predict(X) == from_df.labels_).all()
    with pytest.raises(TypeError):
        ds.kmodes.KModes(3, categorical_columns=["a", "b"])


def test_kmodes_reproducible_and_parallel():
    df, _ = make_df()
    X = df[["a", "b"]]
    one = ds.kmodes.KModes(3, random_state=3, n_init=4, n_jobs=1).fit(X)
    two = ds.kmodes.KModes(3, random_state=3, n_init=4, n_jobs=2).fit(X)
    assert (one.labels_ == two.labels_).all()
    assert one.cost_ == two.cost_


def test_kmodes_ng_dissim_and_array_init():
    df, _ = make_df()
    X = df[["a", "b"]]
    km = ds.kmodes.KModes(3, cat_dissim=ng_dissim, random_state=0).fit(X)
    assert len(km.labels_) == len(X)
    init = km.cluster_centroids_
    again = ds.kmodes.KModes(3, init=init, random_state=0).fit(X)
    assert again.cost_ <= again.epoch_costs_[0]


def test_kmodes_predict_unseen_category():
    df, _ = make_df()
    km = ds.kmodes.KModes(3, random_state=0).fit(df[["a", "b"]])
    new = pd.DataFrame({"a": ["zzz", "0"], "b": ["zzz", "0"]})
    assert km.predict(new).shape == (2,)


def test_kmodes_errors():
    df, _ = make_df()
    X = df[["a", "b"]]
    with pytest.raises(ValueError, match="between 1"):
        ds.kmodes.KModes(1000).fit(X)
    with pytest.raises(ValueError, match="init must be"):
        ds.kmodes.KModes(2, init="nope").fit(X)
    with pytest.raises(ValueError, match="at least 1"):
        ds.kmodes.KModes(2, max_iter=0).fit(X)
    with pytest.raises(ValueError, match="distinct"):
        ds.kmodes.KModes(3).fit(pd.DataFrame({"a": ["u", "v", "u", "v"]}))
    with pytest.raises(ValueError, match="Missing"):
        ds.kmodes.KModes(2).fit(X.assign(a=[None] * len(X)))
    with pytest.raises(ValueError, match="2D"):
        ds.kmodes.KModes(2).fit(np.arange(5))
    with pytest.raises(ValueError, match="shape"):
        ds.kmodes.KModes(2, init=[["0", "0"]]).fit(X)
    with pytest.raises(ValueError, match="not present"):
        ds.kmodes.KModes(2, init=[["q", "0"], ["0", "0"]]).fit(X)
    km = ds.kmodes.KModes(2, random_state=0).fit(X)
    with pytest.raises(ValueError, match="features"):
        km.predict(X.assign(c="1"))
    with pytest.raises(ValueError, match="do not match"):
        km.predict(X.rename(columns={"a": "z"}))


def test_kmodes_sklearn_api():
    df, _ = make_df()
    X = df[["a", "b"]]
    km = ds.kmodes.KModes(3, random_state=0)
    cloned = clone(km)
    assert cloned.get_params() == km.get_params()
    pipe = Pipeline([("km", km)]).fit(X)
    assert len(pipe.predict(X)) == len(X)
    assert len(ParameterGrid({"n_clusters": [2, 3]})) == 2
    assert list(km.fit_predict(X)) == list(km.labels_)
    assert list(km.feature_names_in_) == ["a", "b"]


def test_resolve_columns():
    assert resolve_columns(["b", "a"], 3, ["a", "b", "c"]) == [0, 1]
    assert resolve_columns([2, 0], 3) == [0, 2]
    with pytest.raises(ValueError, match="DataFrame"):
        resolve_columns(["a"], 3)
    with pytest.raises(ValueError, match="not found"):
        resolve_columns(["zz"], 3, ["a", "b", "c"])
    with pytest.raises(ValueError, match="either all names"):
        resolve_columns(["a", 1], 3, ["a", "b", "c"])
    with pytest.raises(ValueError, match="out of range"):
        resolve_columns([5], 3)
    with pytest.raises(ValueError, match="empty"):
        resolve_columns([], 3)
    with pytest.raises(ValueError, match="duplicates"):
        resolve_columns([1, 1], 3)


def test_encoding_roundtrip_and_mixed_types():
    data = np.array([["b", 1], ["a", 2], ["b", "x"]], dtype=object)
    codes, cats = encode_categorical(data)
    assert (decode_categorical(codes, cats) == data).all()
    unseen = apply_encoding(np.array([["zzz", 1]], dtype=object), cats)
    assert unseen[0, 0] == len(cats[0])


def test_dissimilarities():
    pts = np.array([[0, 1], [1, 1]])
    cents = np.array([[0, 1], [1, 0]])
    assert matching_dissim(pts, cents).tolist() == [[0, 2], [1, 1]]
    assert euclidean_dissim(pts.astype(float), cents.astype(float)).tolist() == [
        [0, 2],
        [1, 1],
    ]
    assert ng_dissim(pts, cents).tolist() == matching_dissim(pts, cents).tolist()
