import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import pier_ds_utils as ds
from tests.test_kmodes import make_df, purity


@pytest.mark.parametrize("init", ["Huang", "Cao", "random"])
def test_kprototypes_init_methods(init):
    df, g = make_df()
    kp = ds.kmodes.KPrototypes(
        3, init=init, n_init=3, random_state=0, categorical=["a", "b"]
    ).fit(df)
    assert kp.cluster_centroids_.shape == (3, 4)
    assert kp.epoch_costs_[-1] <= kp.epoch_costs_[0]
    assert purity(g, kp.labels_) > 0.8
    assert kp.gamma_ > 0


def test_kprototypes_names_equal_indices():
    df, _ = make_df()
    by_name = ds.kmodes.KPrototypes(3, random_state=1, categorical=["a", "b"])
    by_index = ds.kmodes.KPrototypes(3, random_state=1, categorical=[0, 1])
    by_name.fit(df)
    by_index.fit(df.to_numpy())
    assert (by_name.labels_ == by_index.labels_).all()
    assert by_name.cost_ == by_index.cost_
    assert (by_name.cluster_centroids_ == by_index.cluster_centroids_).all()
    assert (by_name.predict(df) == by_name.labels_).all()


def test_kprototypes_categorical_in_the_middle_and_fit_override():
    df, _ = make_df()
    shuffled = df[["x", "a", "y", "b"]]
    kp = ds.kmodes.KPrototypes(3, random_state=1)
    kp.fit(shuffled, categorical=["a", "b"])
    assert kp.categorical_idx_ == [1, 3]
    assert kp.numeric_idx_ == [0, 2]
    assert kp.cluster_centroids_[0, 0] != kp.cluster_centroids_[0, 0] + 1
    assert isinstance(kp.cluster_centroids_[0, 1], str)


def test_kprototypes_gamma_array_init_and_parallel():
    df, _ = make_df()
    kp = ds.kmodes.KPrototypes(
        3, gamma=1.5, random_state=0, categorical=["a", "b"], n_init=2, n_jobs=2
    ).fit(df)
    assert kp.gamma_ == 1.5
    again = ds.kmodes.KPrototypes(
        3, init=kp.cluster_centroids_, categorical=["a", "b"], random_state=0
    ).fit(df)
    assert again.cost_ <= again.epoch_costs_[0]


def test_kprototypes_errors():
    df, _ = make_df()
    with pytest.raises(ValueError, match="must list"):
        ds.kmodes.KPrototypes(2).fit(df)
    with pytest.raises(ValueError, match="DataFrame"):
        ds.kmodes.KPrototypes(2, categorical=["a"]).fit(df.to_numpy())
    with pytest.raises(ValueError, match="not found"):
        ds.kmodes.KPrototypes(2, categorical=["nope"]).fit(df)
    with pytest.raises(ValueError, match="either all names"):
        ds.kmodes.KPrototypes(2, categorical=["a", 1]).fit(df)
    with pytest.raises(ValueError, match="use KModes"):
        ds.kmodes.KPrototypes(2, categorical=["a", "b", "x", "y"]).fit(df)
    with pytest.raises(ValueError, match="numeric"):
        ds.kmodes.KPrototypes(2, categorical=["a"]).fit(
            df.assign(x=df["x"].astype(str) + "k")
        )


def test_kprototypes_clone_and_names():
    df, _ = make_df()
    kp = ds.kmodes.KPrototypes(3, categorical=["a", "b"], random_state=0)
    assert clone(kp).get_params() == kp.get_params()
    kp.fit(df)
    assert list(kp.feature_names_in_) == list(df.columns)
    kp.fit(df.to_numpy()[:, :4], categorical=[0, 1])
    assert not hasattr(kp, "feature_names_in_")
    ints = pd.DataFrame(np.c_[df.to_numpy()], columns=range(4))
    kp.fit(ints, categorical=[0, 1])
    assert not hasattr(kp, "feature_names_in_")
