r"""Embedding dissimilarity \(d_E\) on L2-normalized prompt embeddings.

\[
d_E(u,v) = \\frac{1 - \\mathrm{clip}(e_u^\\top e_v, -1, 1)}{2} \\in [0, 1].
\]

``semantic_distance`` is the unnormalized cosine form in ``[0, 2]``
(``1 - cos``); ``embedding_distance`` is the canonical ``[0, 1]`` form.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def semantic_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine distance on L2-normalized embeddings: ``1 - clip(dot)`` ∈ [0, 2]."""
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    if not (np.isclose(norm_e1, 1.0) and np.isclose(norm_e2, 1.0)):
        raise ValueError(f"Embeddings must be L2-normalized. Got norms: {norm_e1}, {norm_e2}")

    cosine_similarity = np.clip(np.dot(e1, e2), -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def semantic_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Batch cosine distances; same unit-norm precondition as ``semantic_distance``."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    query_norm = np.linalg.norm(query_embedding)
    if not np.isclose(query_norm, 1.0):
        raise ValueError(f"Query embedding must be L2-normalized. Got norm: {query_norm}")
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(embedding_norms, 1.0):
        bad = np.where(~np.isclose(embedding_norms, 1.0))[0]
        raise ValueError(
            f"Embeddings must be L2-normalized. Non-unit norms at indices {bad.tolist()}: "
            f"{embedding_norms[bad].tolist()}"
        )

    cosine_similarities = np.clip(embeddings @ query_embedding, -1.0, 1.0)
    return 1.0 - cosine_similarities


def embedding_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """Normalized cosine dissimilarity in ``[0, 1]``: ``(1 - cos) / 2``."""
    return float(semantic_distance(e1, e2) / 2.0)


def embedding_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    return semantic_distances_batch(query_embedding, embeddings) / 2.0


def pairwise_embedding_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Vectorized pairwise ``embedding_distance`` matrix (symmetric, zero diagonal).

    ``embeddings`` must be ``(n, d)`` L2-normalized rows.
    """
    if embeddings.ndim != 2:
        raise ValueError(f"embeddings must be 2-D, got shape {embeddings.shape}")
    n = embeddings.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    sims = np.clip(embeddings @ embeddings.T, -1.0, 1.0)
    dist = (1.0 - sims) / 2.0
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    return dist.astype(np.float64, copy=False)


def unit_normalize_rows(embeddings: np.ndarray) -> np.ndarray:
    """L2-normalize rows; zeros become small epsilon-safe unit vectors."""
    emb = np.asarray(embeddings, dtype=np.float64)
    if emb.ndim == 1:
        n = np.linalg.norm(emb)
        return emb / max(n, 1e-12)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return emb / norms


__all__ = [
    "semantic_distance",
    "semantic_distances_batch",
    "embedding_distance",
    "embedding_distances_batch",
    "pairwise_embedding_matrix",
    "unit_normalize_rows",
]
