"""Timing tests for KModes / KPrototypes.

Opt-in: they are deselected by default and run with ``make performance``.
Budgets are generous multiples of the times measured on a laptop so they only
fail on a real slowdown, not on a slower CI machine.
"""

import time

import numpy as np
import pandas as pd
import pytest

import pier_ds_utils as ds

pytestmark = pytest.mark.performance

N_ROWS = 100_000
N_PREDICT_ROWS = 500_000
N_CLUSTERS = 8
MAX_ITER = 5
N_SINGLE_PREDICTIONS = 500

KMODES_FIT_BUDGET = 50.0  # measured ~6s
KPROTOTYPES_FIT_BUDGET = 150.0  # measured ~17s
PREDICT_BUDGET = 5.0  # measured ~0.5s
SINGLE_PREDICT_BUDGET = 0.01  # seconds per call; measured ~0.5-0.9ms
MAX_SCALING_RATIO = 3.5  # time per iteration for 2n rows / for n rows


def make_categorical(n_rows, n_features=10, n_levels=5, keep=0.6, seed=0):
    """Clustered categorical data: noisy copies of N_CLUSTERS prototypes."""
    rng = np.random.default_rng(seed)
    prototypes = rng.integers(0, n_levels, size=(N_CLUSTERS, n_features))
    groups = rng.integers(0, N_CLUSTERS, size=n_rows)
    noise = rng.integers(0, n_levels, size=(n_rows, n_features))
    X = np.where(rng.random(noise.shape) < keep, prototypes[groups], noise)
    columns = [f"cat{i}" for i in range(n_features)]
    return pd.DataFrame(X.astype(str), columns=columns), groups


def make_mixed(n_rows, n_cat=5, n_num=5, seed=0):
    X, groups = make_categorical(n_rows, n_cat, seed=seed)
    rng = np.random.default_rng(seed + 1)
    centers = rng.normal(0, 3, size=(N_CLUSTERS, n_num))
    numeric = centers[groups] + rng.normal(size=(n_rows, n_num))
    for j in range(n_num):
        X[f"num{j}"] = numeric[:, j]
    return X, [c for c in X.columns if c.startswith("cat")]


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return result, time.perf_counter() - start


def kmodes(**params):
    defaults = dict(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, random_state=0)
    return ds.kmodes.KModes(**{**defaults, "n_init": 1, **params})


def kprototypes(categorical, **params):
    defaults = dict(n_clusters=N_CLUSTERS, max_iter=MAX_ITER, random_state=0)
    return ds.kmodes.KPrototypes(
        categorical=categorical, **{**defaults, "n_init": 1, **params}
    )


@pytest.mark.parametrize("init", ["Huang", "Cao"])
def test_kmodes_fit_time(init):
    X, _ = make_categorical(N_ROWS)
    model, elapsed = timed(lambda: kmodes(init=init).fit(X))
    print(f"KModes[{init}] fit {N_ROWS} rows: {elapsed:.2f}s")
    assert model.n_iter_ >= 2
    assert elapsed < KMODES_FIT_BUDGET


def test_kprototypes_fit_time():
    X, categorical = make_mixed(N_ROWS)
    model, elapsed = timed(lambda: kprototypes(categorical).fit(X))
    print(f"KPrototypes fit {N_ROWS} rows: {elapsed:.2f}s")
    assert model.n_iter_ >= 2
    assert elapsed < KPROTOTYPES_FIT_BUDGET


def test_predict_time():
    X, _ = make_categorical(N_ROWS)
    km = kmodes().fit(X)
    new, _ = make_categorical(N_PREDICT_ROWS, seed=1)
    labels, elapsed = timed(lambda: km.predict(new))
    print(f"KModes predict {N_PREDICT_ROWS} rows: {elapsed:.2f}s")
    assert labels.shape == (N_PREDICT_ROWS,)
    assert elapsed < PREDICT_BUDGET

    mixed, categorical = make_mixed(N_ROWS)
    kp = kprototypes(categorical).fit(mixed)
    new_mixed, _ = make_mixed(N_PREDICT_ROWS, seed=1)
    labels, elapsed = timed(lambda: kp.predict(new_mixed))
    print(f"KPrototypes predict {N_PREDICT_ROWS} rows: {elapsed:.2f}s")
    assert labels.shape == (N_PREDICT_ROWS,)
    assert elapsed < PREDICT_BUDGET


def per_iteration_time(n_rows):
    X, _ = make_categorical(n_rows)
    model, elapsed = timed(lambda: kmodes().fit(X))
    return elapsed / model.n_iter_


def test_fit_scales_roughly_linearly():
    small = per_iteration_time(N_ROWS // 2)
    large = per_iteration_time(N_ROWS)
    print(f"time per iteration: {small:.3f}s -> {large:.3f}s (2x rows)")
    assert large / small < MAX_SCALING_RATIO


def test_n_jobs_does_not_change_result():
    X, _ = make_categorical(N_ROWS // 4)
    serial = kmodes(n_init=4, n_jobs=1).fit(X)
    parallel = kmodes(n_init=4, n_jobs=2).fit(X)
    assert (serial.labels_ == parallel.labels_).all()
    assert serial.cost_ == parallel.cost_


def average_call_time(fn, inputs):
    """Average seconds per call of ``fn`` over ``inputs`` (after one warm-up)."""
    fn(inputs[0])
    start = time.perf_counter()
    for item in inputs:
        fn(item)
    return (time.perf_counter() - start) / len(inputs)


def fitted_for_single_row(name):
    if name == "KModes":
        X, _ = make_categorical(N_ROWS // 20)
        return kmodes().fit(X), X
    X, categorical = make_mixed(N_ROWS // 20)
    return kprototypes(categorical).fit(X), X


@pytest.mark.parametrize("name", ["KModes", "KPrototypes"])
def test_predict_single_row_latency(name):
    model, X = fitted_for_single_row(name)
    rows = [X.iloc[[i]] for i in range(N_SINGLE_PREDICTIONS)]

    average = average_call_time(model.predict, rows)
    print(f"{name} predict on one row: {average * 1000:.2f}ms on average")

    batch = model.predict(X.iloc[:N_SINGLE_PREDICTIONS])
    for i in (0, 1, N_SINGLE_PREDICTIONS - 1):
        assert model.predict(rows[i]).tolist() == [batch[i]]
    assert average < SINGLE_PREDICT_BUDGET
