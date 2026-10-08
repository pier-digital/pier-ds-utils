"""Centroid initialisation strategies (Huang and Cao).
"""

import numpy as np

from pier_ds_utils.kmodes.dissim import matching_dissim


def _to_first_free_row(X, centroids, ik):
    """Replace centroid ``ik`` with the closest row not already a centroid."""
    order = np.argsort(matching_dissim(X, centroids[ik : ik + 1])[:, 0], kind="stable")
    for idx in order:
        if not (X[idx] == centroids).all(axis=1).any():
            centroids[ik] = X[idx]
            return


def init_huang(Xcat, n_clusters, rng):
    """Sample every attribute from its empirical distribution (Huang 1998)."""
    centroids = np.stack(
        [rng.choice(Xcat[:, j], n_clusters) for j in range(Xcat.shape[1])], axis=1
    )
    for ik in range(n_clusters):
        _to_first_free_row(Xcat, centroids, ik)
    return centroids


def init_cao(Xcat, n_clusters):
    """Density and distance based initialisation (Cao et al. 2009)."""
    n_points, n_attrs = Xcat.shape
    dens = np.zeros(n_points)
    for j in range(n_attrs):
        dens += np.bincount(Xcat[:, j])[Xcat[:, j]] / n_points / n_attrs

    centroids = np.empty((n_clusters, n_attrs), dtype=Xcat.dtype)
    centroids[0] = Xcat[np.argmax(dens)]
    for ik in range(1, n_clusters):
        dd = matching_dissim(Xcat, centroids[:ik]) * dens[:, None]
        centroids[ik] = Xcat[np.argmax(dd.min(axis=1))]
    return centroids


def init_numeric(Xnum, n_clusters, rng):
    """Random numeric centroids around the mean, scaled by the std."""
    noise = rng.standard_normal((n_clusters, Xnum.shape[1]))
    return Xnum.mean(axis=0) + noise * Xnum.std(axis=0)
