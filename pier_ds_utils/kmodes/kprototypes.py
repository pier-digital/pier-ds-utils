"""K-Prototypes clustering for mixed numeric / categorical data (Huang 1997).
"""

import typing

import numpy as np

from pier_ds_utils.kmodes.base import _BaseKModes
from pier_ds_utils.kmodes.util import resolve_columns


class KPrototypes(_BaseKModes):
    """K-Prototypes clustering.

    Parameters
    ----------
    n_clusters, max_iter, cat_dissim, init, n_init, verbose, random_state, n_jobs
        See :class:`KModes`.
    num_dissim: callable, optional
        Dissimilarity for the numeric columns. Defaults to squared euclidean.
    gamma: float, optional
        Weight of the categorical part. Defaults to half the mean standard
        deviation of the numeric columns.
    categorical: list of str or int
        Categorical columns, given as names (DataFrame input) or positional
        indices. Names and indices cannot be mixed. Every other column is
        treated as numeric.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        num_dissim: typing.Optional[typing.Callable] = None,
        cat_dissim: typing.Optional[typing.Callable] = None,
        gamma: typing.Optional[float] = None,
        init: typing.Union[str, typing.Any] = "Cao",
        n_init: int = 10,
        verbose: int = 0,
        random_state=None,
        n_jobs: int = 1,
        categorical: typing.Optional[typing.List[typing.Union[str, int]]] = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.num_dissim = num_dissim
        self.cat_dissim = cat_dissim
        self.gamma = gamma
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.categorical = categorical

    def _gamma(self, Xnum):
        if self.gamma is not None:
            return self.gamma
        return 0.5 * float(np.mean(Xnum.std(axis=0)))

    def _resolve_indices(self, columns, n_features, override=None):
        spec = override if override is not None else self.categorical
        if spec is None:
            raise ValueError("`categorical` must list the categorical columns.")
        cat = resolve_columns(spec, n_features, columns)
        num = [i for i in range(n_features) if i not in cat]
        if not num:
            raise ValueError("All columns are categorical; use KModes instead.")
        return cat, num

    def fit(self, X, y=None, categorical=None):
        """Compute K-Prototypes clustering of ``X``.

        ``categorical`` overrides the constructor argument when given.
        """
        return self._fit(X, categorical)
