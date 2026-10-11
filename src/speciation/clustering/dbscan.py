"""DBSCAN on a controller-built pairwise distance matrix."""

from __future__ import annotations

import numpy as np


def dbscan_precomputed(dist: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Labels ``0..k-1`` for density clusters, ``-1`` for noise."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got {dist.shape}")
    n = dist.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if min_samples < 1 or eps < 0:
        raise ValueError(f"invalid eps={eps} min_samples={min_samples}")

    neighbors = [np.flatnonzero(dist[i] <= eps) for i in range(n)]
    is_core = np.array([len(nbs) >= min_samples for nbs in neighbors], dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        seed = list(neighbors[i])
        labels[i] = cid
        j = 0
        while j < len(seed):
            q = int(seed[j])
            j += 1
            if labels[q] == -1:
                labels[q] = cid
            if is_core[q]:
                for r in neighbors[q]:
                    r = int(r)
                    if labels[r] == -1:
                        labels[r] = cid
                        seed.append(r)
        cid += 1
    return labels
