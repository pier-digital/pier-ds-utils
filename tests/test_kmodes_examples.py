"""Validation of the kmodes port with generated datasets and with iris.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.metrics import adjusted_rand_score

import pier_ds_utils as ds

CATEGORICAL_BEST_COST = 152.0
MIXED_BEST_COST = 1498.106
IRIS_KMODES_COST = 100.0
IRIS_KPROTOTYPES_COST = 49.816


def purity(classes, labels):
    return pd.crosstab(np.asarray(classes), labels).values.max(axis=1).sum() / len(
        labels
    )


def make_categorical_data(seed=0, sizes=(10, 10, 10, 17), n_features=35, keep=0.85):
    """Integer coded categorical attributes around one prototype per class."""
    rng = np.random.default_rng(seed)
    n_levels = rng.integers(2, 5, size=n_features)
    classes = np.repeat(np.arange(len(sizes)), sizes)
    prototypes = rng.integers(0, n_levels, size=(len(sizes), n_features))
    noise = rng.integers(0, n_levels, size=(len(classes), n_features))
    keep_cell = rng.random(noise.shape) < keep
    X = np.where(keep_cell, prototypes[classes], noise)
    return pd.DataFrame(X), pd.Series(classes, name="class")


def make_mixed_data(seed=0, sizes=(12, 20, 30, 38), keep=0.9):
    """Market cap (numeric), sector and country (categorical) in 4 groups."""
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(len(sizes)), sizes)
    sectors = np.array(["tech", "nrg", "fin", "cons"])
    countries = np.array(["USA", "CN", "NL", "BR"])
    center = np.array([10.0, 25.0, 40.0, 55.0])  # market cap per group

    def noisy(values):
        picked = rng.choice(values, size=len(groups))
        return np.where(rng.random(len(groups)) < keep, values[groups], picked)

    X = pd.DataFrame(
        {
            "market_cap": center[groups] + rng.normal(0, 4.0, len(groups)),
            "sector": noisy(sectors),
            "country": noisy(countries),
        },
        index=[f"SYM{i}" for i in range(len(groups))],
    )
    return X, pd.Series(groups, index=X.index, name="group")


@pytest.mark.parametrize("init", ["Huang", "Cao"])
def test_categorical_kmodes(init):
    X, y = make_categorical_data()
    assert X.shape == (47, 35)
    km = ds.kmodes.KModes(4, init=init, n_init=10, random_state=0).fit(X)

    assert km.cluster_centroids_.shape == (4, 35)
    assert sorted(np.bincount(km.labels_)) == [10, 10, 10, 17]
    assert km.epoch_costs_[-1] <= km.epoch_costs_[0]
    assert km.cost_ == CATEGORICAL_BEST_COST
    assert purity(y, km.labels_) == 1.0
    assert (km.predict(X) == km.labels_).all()


def test_mixed_kprototypes_by_name_and_index():
    X, groups = make_mixed_data()
    assert X.shape == (100, 3)
    params = dict(n_clusters=4, init="Cao", n_init=10, random_state=0)

    by_name = ds.kmodes.KPrototypes(**params, categorical=["sector", "country"])
    by_name.fit(X)
    by_index = ds.kmodes.KPrototypes(**params)
    by_index.fit(X.to_numpy(dtype=object), categorical=[1, 2])

    assert (by_name.labels_ == by_index.labels_).all()
    assert (by_name.cluster_centroids_ == by_index.cluster_centroids_).all()
    assert by_name.cost_ == by_index.cost_
    assert by_name.cost_ == pytest.approx(MIXED_BEST_COST, abs=1e-3)
    assert purity(groups, by_name.labels_) >= 0.95
    assert (by_name.predict(X) == by_name.labels_).all()
    assert list(by_name.feature_names_in_) == ["market_cap", "sector", "country"]


PLANTED_SIZES = (30, 50, 70, 90)
INITS = ["Huang", "Cao", "random"]


def make_planted_categorical(sizes=PLANTED_SIZES, n_features=8, noise=0.0, seed=0):
    """Predefined clusters: cluster c has the prototype ``(c + j) % n_clusters``.

    Every pair of prototypes differs in every column. Each cell is replaced by a
    random level with probability ``noise``. Returns the data, the true labels
    and the prototypes.
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(sizes)
    prototypes = (
        np.arange(n_clusters)[:, None] + np.arange(n_features)[None, :]
    ) % n_clusters
    labels = np.repeat(np.arange(n_clusters), sizes)
    random_cells = rng.integers(0, n_clusters, size=(len(labels), n_features))
    X = np.where(
        rng.random(random_cells.shape) < noise, random_cells, prototypes[labels]
    )
    columns = [f"cat{j}" for j in range(n_features)]
    return pd.DataFrame(X, columns=columns), labels, prototypes


def make_planted_mixed(
    sizes=PLANTED_SIZES, n_cat=8, n_num=3, cat_noise=0.1, spread=10.0, seed=0
):
    """Predefined clusters with numeric centers ``spread * c`` and unit noise.

    ``cat_noise=1.0`` makes the categorical block uninformative and
    ``spread=0.0`` makes the numeric block uninformative.
    """
    rng = np.random.default_rng(seed)
    X, labels, prototypes = make_planted_categorical(sizes, n_cat, cat_noise, seed=seed)
    centers = spread * np.repeat(np.arange(len(sizes))[:, None], n_num, axis=1)
    numeric = centers[labels] + rng.normal(size=(len(labels), n_num))
    for j in range(n_num):
        X[f"num{j}"] = numeric[:, j]
    return X, labels, prototypes, centers


def centroids_by_cluster(true, labels, centroids):
    """Centroid found for each predefined cluster (via its majority label)."""
    found = [np.bincount(labels[true == c]).argmax() for c in np.unique(true)]
    return np.asarray(centroids)[found]


def categorical_names(n_cat=8):
    return [f"cat{j}" for j in range(n_cat)]


@pytest.mark.parametrize("init", INITS)
@pytest.mark.parametrize("noise, min_ari", [(0.0, 1.0), (0.15, 0.95)])
def test_kmodes_recovers_planted_clusters(init, noise, min_ari):
    X, true, prototypes = make_planted_categorical(noise=noise)
    km = ds.kmodes.KModes(4, init=init, n_init=10, random_state=0).fit(X)

    assert adjusted_rand_score(true, km.labels_) >= min_ari
    found = centroids_by_cluster(true, km.labels_, km.cluster_centroids_)
    assert (found.astype(int) == prototypes).all()


@pytest.mark.parametrize("init", INITS)
def test_kprototypes_recovers_planted_clusters(init):
    X, true, prototypes, centers = make_planted_mixed()
    kp = ds.kmodes.KPrototypes(
        4, init=init, n_init=10, random_state=0, categorical=categorical_names()
    ).fit(X)

    assert adjusted_rand_score(true, kp.labels_) >= 0.95
    found = centroids_by_cluster(true, kp.labels_, kp.cluster_centroids_)
    assert (found[:, :8].astype(int) == prototypes).all()
    assert np.abs(found[:, 8:].astype(float) - centers).max() < 0.5


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize(
    "informative, kwargs, gamma",
    [
        # the default gamma is derived from the numeric spread, so it is tiny
        # when only the categorical block carries information
        ("categorical", dict(spread=0.0), 5.0),
        ("numeric", dict(cat_noise=1.0), None),
    ],
)
def test_kprototypes_uses_both_parts(informative, kwargs, gamma, seed):
    X, true, _, _ = make_planted_mixed(seed=seed, **kwargs)
    kp = ds.kmodes.KPrototypes(
        4,
        init="Cao",
        n_init=10,
        random_state=seed,
        gamma=gamma,
        categorical=categorical_names(),
    ).fit(X)
    assert adjusted_rand_score(true, kp.labels_) >= 0.95, informative


@pytest.mark.parametrize("seed", range(8))
def test_kmodes_random_init_escapes_duplicate_centroids(seed):
    # noise-free clusters are made of identical rows: two random seeds taken
    # from the same cluster give identical centroids, which must not leave
    # another pair of clusters merged
    X, true, _ = make_planted_categorical(noise=0.0, seed=seed)
    km = ds.kmodes.KModes(4, init="random", n_init=10, random_state=seed).fit(X)
    assert adjusted_rand_score(true, km.labels_) == 1.0
    assert km.cost_ == 0.0


def test_predict_recovers_planted_clusters_on_new_data():
    X, true, _ = make_planted_categorical(noise=0.15, seed=0)
    new, new_true, _ = make_planted_categorical(noise=0.15, seed=1)
    km = ds.kmodes.KModes(4, init="Cao", random_state=0).fit(X)
    assert adjusted_rand_score(new_true, km.predict(new)) >= 0.95

    X, true, _, _ = make_planted_mixed(seed=0)
    new, new_true, _, _ = make_planted_mixed(seed=1)
    kp = ds.kmodes.KPrototypes(
        4, init="Cao", random_state=0, categorical=categorical_names()
    ).fit(X)
    assert adjusted_rand_score(new_true, kp.predict(new)) >= 0.95


def test_planted_clusters_robust_to_noise_level():
    scores = []
    for noise in (0.0, 0.2, 0.4):
        X, true, _ = make_planted_categorical(noise=noise)
        km = ds.kmodes.KModes(4, init="Cao", n_init=10, random_state=0).fit(X)
        scores.append(adjusted_rand_score(true, km.labels_))
    assert scores[0] == 1.0
    assert scores == sorted(scores, reverse=True)
    assert scores[1] >= 0.9


def iris_binned():
    iris = load_iris(as_frame=True)
    binned = iris.data.apply(
        lambda col: pd.qcut(col, 3, labels=["low", "mid", "high"]).astype(str)
    )
    return iris.data, binned, iris.target


def test_iris_kmodes():
    _, binned, species = iris_binned()
    km = ds.kmodes.KModes(3, init="Cao", n_init=10, random_state=0).fit(binned)

    assert sorted(set(km.labels_)) == [0, 1, 2]
    assert km.cost_ == IRIS_KMODES_COST
    assert purity(species, km.labels_) >= 0.85
    assert km.cluster_centroids_.shape == (3, 4)


def test_iris_kprototypes_by_name():
    numeric, binned, species = iris_binned()
    petals = ["petal length (cm)", "petal width (cm)"]
    mixed = numeric.assign(**{c: binned[c] for c in petals})
    params = dict(n_clusters=3, init="Cao", n_init=10, random_state=0)

    by_name = ds.kmodes.KPrototypes(**params, categorical=petals).fit(mixed)
    by_index = ds.kmodes.KPrototypes(**params, categorical=[2, 3]).fit(
        mixed.to_numpy(dtype=object)
    )

    assert (by_name.labels_ == by_index.labels_).all()
    assert by_name.cost_ == pytest.approx(IRIS_KPROTOTYPES_COST, abs=1e-3)
    assert purity(species, by_name.labels_) >= 0.85
    assert by_name.gamma_ == pytest.approx(0.315, abs=1e-3)
    assert (by_name.predict(mixed) == by_name.labels_).all()
