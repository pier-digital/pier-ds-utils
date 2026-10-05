"""K-Modes clustering for categorical data (Huang 1998).
"""

import typing

from pier_ds_utils.kmodes.base import _BaseKModes


class KModes(_BaseKModes):
    """K-Modes clustering.

    Every column of ``X`` is treated as categorical; select the columns to
    cluster on beforehand (e.g. ``df[["a", "b"]]``).

    Parameters
    ----------
    n_clusters: int
        Number of clusters.
    max_iter: int
        Maximum number of passes over the data in a single run.
    cat_dissim: callable, optional
        Dissimilarity ``f(points, centroids, cl_attr_freq, cl_sizes)`` returning
        a ``(n_points, n_clusters)`` matrix. Defaults to ``matching_dissim``.
    init: {"Huang", "Cao", "random"} or array-like
        Initialisation method, or an array of shape ``(n_clusters, n_columns)``.
    n_init: int
        Number of runs with different initialisations; the best is kept.
    verbose: int
        Log the cost of each iteration when non-zero.
    random_state: int, RandomState or None
        Seed for reproducibility.
    n_jobs: int
        Number of parallel jobs used for the ``n_init`` runs.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        max_iter: int = 100,
        cat_dissim: typing.Optional[typing.Callable] = None,
        init: typing.Union[str, typing.Any] = "Cao",
        n_init: int = 10,
        verbose: int = 0,
        random_state=None,
        n_jobs: int = 1,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.cat_dissim = cat_dissim
        self.init = init
        self.n_init = n_init
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _resolve_indices(self, columns, n_features, override=None):
        return list(range(n_features)), []

    def fit(self, X, y=None):
        """Compute K-Modes clustering of ``X``."""
        return self._fit(X)
