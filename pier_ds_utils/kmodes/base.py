"""Shared machinery of :class:`KModes` and :class:`KPrototypes`.

Both estimators are the same algorithm: K-Modes is K-Prototypes without numeric
columns, so a single implementation (with an empty numeric block) serves both.

"""

import logging
import typing

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import check_is_fitted

from pier_ds_utils.kmodes import init as centroid_init
from pier_ds_utils.kmodes.dissim import euclidean_dissim, matching_dissim
from pier_ds_utils.kmodes.util import (
    apply_encoding,
    decode_categorical,
    encode_categorical,
)

logger = logging.getLogger(__name__)

INIT_METHODS = ("Huang", "Cao", "random")


class _ClusterState:
    """Memberships plus the statistics needed to update centroids online."""

    def __init__(self, Xnum, Xcat, n_values, labels, cnum, ccat):
        n_clusters = len(cnum)
        self.Xnum, self.Xcat, self.labels = Xnum, Xcat, labels
        self.cnum, self.ccat = cnum.copy(), ccat.copy()
        self.sizes = np.bincount(labels, minlength=n_clusters)
        self.freq = [
            [
                np.bincount(Xcat[labels == c, j], minlength=n_values[j])
                for j in range(Xcat.shape[1])
            ]
            for c in range(n_clusters)
        ]
        self.num_sum = np.stack(
            [Xnum[labels == c].sum(axis=0) for c in range(n_clusters)]
        )
        for c in range(n_clusters):
            self.refresh(c)

    def refresh(self, c):
        """Recompute the centroid of cluster ``c`` (kept as is when empty)."""
        if self.sizes[c] == 0:
            return
        self.ccat[c] = [f.argmax() for f in self.freq[c]]
        self.cnum[c] = self.num_sum[c] / self.sizes[c]

    def move(self, i, new):
        """Move point ``i`` to cluster ``new`` and refresh both centroids."""
        old = self.labels[i]
        for j, code in enumerate(self.Xcat[i]):
            self.freq[old][j][code] -= 1
            self.freq[new][j][code] += 1
        self.num_sum[old] -= self.Xnum[i]
        self.num_sum[new] += self.Xnum[i]
        self.sizes[old] -= 1
        self.sizes[new] += 1
        self.labels[i] = new
        self.refresh(old)
        self.refresh(new)

    def reseed_empty(self, rng):
        """Give every empty cluster a random point of the largest cluster."""
        for c in np.flatnonzero(self.sizes == 0):
            donor = self.sizes.argmax()
            if self.sizes[donor] < 2:
                return
            self.move(rng.choice(np.flatnonzero(self.labels == donor)), c)


def _to_object_2d(X) -> typing.Tuple[np.ndarray, typing.Optional[list]]:
    if isinstance(X, pd.DataFrame):
        return X.to_numpy(dtype=object), list(X.columns)
    data = np.asarray(X, dtype=object)
    if data.ndim != 2:
        raise ValueError(f"X must be 2D, got an array with {data.ndim} dimensions.")
    return data, None


def _to_float(data: np.ndarray) -> np.ndarray:
    try:
        return data.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Numeric columns must hold numeric values.") from exc


class _BaseKModes(ClusterMixin, BaseEstimator):
    """Base class; subclasses define parameters and ``_resolve_indices``."""

    def _resolve_indices(self, columns, n_features, override=None):
        raise NotImplementedError

    def _gamma(self, Xnum):
        return 1.0

    # ----------------------------------------------------------------- data
    def _split(self, X, override=None, fitting=True):
        data, columns = _to_object_2d(X)
        if fitting:
            cat_idx, num_idx = self._resolve_indices(columns, data.shape[1], override)
            self.categorical_idx_, self.numeric_idx_ = cat_idx, num_idx
            self.n_features_in_ = data.shape[1]
            self._set_feature_names(columns)
        else:
            self._check_input_matches(data, columns)
        used = data[:, self.categorical_idx_ + self.numeric_idx_]
        if pd.isna(used).any():
            raise ValueError("Missing values are not supported; impute them first.")
        return _to_float(data[:, self.numeric_idx_]), data[:, self.categorical_idx_]

    def _set_feature_names(self, columns):
        if columns is not None and all(isinstance(c, str) for c in columns):
            self.feature_names_in_ = np.asarray(columns, dtype=object)
        elif hasattr(self, "feature_names_in_"):
            del self.feature_names_in_

    def _check_input_matches(self, data, columns):
        if data.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {data.shape[1]} features, but {type(self).__name__} "
                f"was fitted with {self.n_features_in_} features."
            )
        names = getattr(self, "feature_names_in_", None)
        if names is not None and columns is not None and list(names) != columns:
            raise ValueError("The columns of X do not match the ones seen in fit.")

    # ------------------------------------------------------------ dissimilarity
    def _dissim(self, Xnum, Xcat, cnum, ccat, gamma, freq=None, sizes=None):
        cat_dissim = self.cat_dissim or matching_dissim
        out = gamma * cat_dissim(Xcat, ccat, freq, sizes)
        if Xnum.shape[1]:
            num_dissim = getattr(self, "num_dissim", None) or euclidean_dissim
            out = out + num_dissim(Xnum, cnum)
        return out

    def _state_dissim(self, st, rows, gamma):
        return self._dissim(
            st.Xnum[rows], st.Xcat[rows], st.cnum, st.ccat, gamma, st.freq, st.sizes
        )

    def _cost(self, st, gamma):
        d = self._state_dissim(st, slice(None), gamma)
        return float(d[np.arange(len(d)), st.labels].sum())

    # -------------------------------------------------------------- single run
    def _initial_centroids(self, Xnum, Xcat, init, rng):
        if isinstance(init, str) and init == "Huang":
            ccat = centroid_init.init_huang(Xcat, self.n_clusters, rng)
        elif isinstance(init, str) and init == "Cao":
            ccat = centroid_init.init_cao(Xcat, self.n_clusters)
        elif isinstance(init, str):
            seeds = rng.choice(len(Xcat), self.n_clusters, replace=False)
            return Xnum[seeds], Xcat[seeds]
        else:
            return init
        return centroid_init.init_numeric(Xnum, self.n_clusters, rng), ccat

    def _online_pass(self, st, gamma, rng):
        moves = 0
        for i in range(len(st.labels)):
            d = self._state_dissim(st, slice(i, i + 1), gamma)[0]
            new = int(d.argmin())
            if new == st.labels[i]:
                continue
            st.move(i, new)
            st.reseed_empty(rng)
            moves += 1
        return moves

    def _single_run(self, Xnum, Xcat, n_values, init, gamma, seed):
        rng = check_random_state(seed)
        cnum, ccat = self._initial_centroids(Xnum, Xcat, init, rng)
        labels = self._dissim(Xnum, Xcat, cnum, ccat, gamma).argmin(axis=1)
        st = _ClusterState(Xnum, Xcat, n_values, labels, cnum, ccat)
        st.reseed_empty(rng)

        cost = self._cost(st, gamma)
        epoch_costs, n_iter = [cost], 0
        for n_iter in range(1, self.max_iter + 1):
            moves = self._online_pass(st, gamma, rng)
            new_cost = self._cost(st, gamma)
            epoch_costs.append(new_cost)
            converged = moves == 0 or new_cost >= cost
            cost = new_cost
            if self.verbose:
                logger.info("iteration %d: cost %.4f, moves %d", n_iter, cost, moves)
            if converged:
                break
        return st, cost, n_iter, epoch_costs

    # --------------------------------------------------------------------- fit
    def _check_params(self, n_samples, Xnum, Xcat):
        if self.n_clusters < 1 or self.n_clusters > n_samples:
            raise ValueError("n_clusters must be between 1 and the number of rows.")
        if self.max_iter < 1 or self.n_init < 1:
            raise ValueError("max_iter and n_init must be at least 1.")
        if isinstance(self.init, str) and self.init not in INIT_METHODS:
            raise ValueError(f"init must be one of {INIT_METHODS} or an array.")
        if isinstance(self.init, str) and self.init != "random":
            unique = np.unique(np.hstack([Xnum, Xcat]).astype(float), axis=0)
            if len(unique) < self.n_clusters:
                raise ValueError("n_clusters exceeds the number of distinct rows.")

    def _array_init(self, init, categories):
        init = np.asarray(init, dtype=object)
        used = sorted(self.categorical_idx_ + self.numeric_idx_)
        if init.shape != (self.n_clusters, len(used)):
            raise ValueError(
                f"init must have shape {(self.n_clusters, len(used))}, "
                f"got {init.shape}."
            )
        pos = {col: p for p, col in enumerate(used)}
        cat = apply_encoding(
            init[:, [pos[c] for c in self.categorical_idx_]], categories
        )
        if any((cat[:, j] >= len(cats)).any() for j, cats in enumerate(categories)):
            raise ValueError("init holds categories not present in X.")
        num = _to_float(init[:, [pos[c] for c in self.numeric_idx_]])
        return num, cat

    def _fit(self, X, override=None):
        Xnum, Xcat_obj = self._split(X, override)
        Xcat, self.categories_ = encode_categorical(Xcat_obj)
        self._check_params(len(Xcat), Xnum, Xcat)
        n_values = [len(c) for c in self.categories_]
        self.gamma_ = self._gamma(Xnum)

        init, n_init = self.init, self.n_init
        if not isinstance(init, str):
            init, n_init = self._array_init(init, self.categories_), 1
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=n_init)
        runs = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(Xnum, Xcat, n_values, init, self.gamma_, s)
            for s in seeds
        )
        st, cost, n_iter, epoch_costs = min(runs, key=lambda r: r[1])

        self.labels_, self.cost_ = st.labels, cost
        self.n_iter_, self.epoch_costs_ = n_iter, epoch_costs
        self._cnum, self._ccat = st.cnum, st.ccat
        self.cluster_centroids_ = self._decode_centroids()
        return self

    def _decode_centroids(self):
        used = sorted(self.categorical_idx_ + self.numeric_idx_)
        cat = decode_categorical(self._ccat, self.categories_)
        out = np.empty((self.n_clusters, len(used)), dtype=object)
        for p, col in enumerate(used):
            if col in self.categorical_idx_:
                out[:, p] = cat[:, self.categorical_idx_.index(col)]
            else:
                out[:, p] = self._cnum[:, self.numeric_idx_.index(col)]
        return out

    def predict(self, X):
        """Assign each row of ``X`` to its closest cluster."""
        check_is_fitted(self, "cluster_centroids_")
        Xnum, Xcat_obj = self._split(X, fitting=False)
        Xcat = apply_encoding(Xcat_obj, self.categories_)
        d = self._dissim(Xnum, Xcat, self._cnum, self._ccat, self.gamma_)
        return d.argmin(axis=1)
