"""Dissimilarity measures.

Every function receives ``points`` (p, m) and ``centroids`` (k, m) and returns a
(p, k) matrix. Categorical measures also receive, while fitting, the per-cluster
category frequencies and cluster sizes.

"""

import numpy as np


def matching_dissim(points, centroids, cl_attr_freq=None, cl_sizes=None):
    """Number of attributes in which points and centroids differ."""
    return np.stack([(points != c).sum(axis=1) for c in centroids], axis=1).astype(
        float
    )


def euclidean_dissim(points, centroids, cl_attr_freq=None, cl_sizes=None):
    """Squared euclidean distance."""
    return np.stack([((points - c) ** 2).sum(axis=1) for c in centroids], axis=1)


def ng_dissim(points, centroids, cl_attr_freq=None, cl_sizes=None):
    """Frequency based dissimilarity of Ng et al. (2007).

    Falls back to matching dissimilarity when no frequencies are available
    (e.g. initial assignment and prediction).
    """
    if cl_attr_freq is None:
        return matching_dissim(points, centroids)
    out = np.zeros((len(points), len(centroids)))
    for c, freqs in enumerate(cl_attr_freq):
        size = max(cl_sizes[c], 1)
        for j, freq in enumerate(freqs):
            out[:, c] += 1.0 - freq[points[:, j]] / size
    return out
