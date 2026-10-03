r"""Objective (phenotype) dissimilarity \(d_P\) on moderation score vectors.

\[
d_P(u,v) = \\min\\bigl(1,\\; \\|p_u - p_v\\|_2 / \\sqrt{k}\\bigr) \\in [0, 1].
\]

Missing vectors contribute distance ``1.0``. This is evaluator-score geometry,
not prompt geometry.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils import get_custom_logging
from utils.evaluator_profiles import (
    GOOGLE_PHENOTYPE_SCORE_ORDER,
    resolve_evaluator,
)

get_logger, _, _, _ = get_custom_logging()

PHENOTYPE_SCORE_ORDER = list(GOOGLE_PHENOTYPE_SCORE_ORDER)


def _backend_key_from_genome(genome: dict) -> str:
    evaluator = genome.get("evaluator")
    if evaluator in ("google", "openai"):
        return evaluator
    mr = genome.get("moderation_result") or {}
    if isinstance(mr, dict):
        if "openai" in mr:
            return "openai"
        if "google" in mr:
            return "google"
    return "google"


def extract_phenotype_vector(genome: dict, logger=None) -> Optional[np.ndarray]:
    """Build the objective score vector from a genome's moderation result."""
    if logger is None:
        logger = get_logger("ObjectiveDistance")

    if not genome or "moderation_result" not in genome:
        return None

    from utils.population_io import get_moderation_scores

    scores = get_moderation_scores(genome)
    if not scores:
        return None

    backend_key = _backend_key_from_genome(genome)
    profile = resolve_evaluator(backend_key)
    score_order = profile.phenotype_score_order

    phenotype = np.array(
        [float(scores.get(score_name, 0.0)) for score_name in score_order],
        dtype=np.float32,
    )

    if not np.all((phenotype >= 0.0) & (phenotype <= 1.0)):
        invalid_indices = np.where((phenotype < 0.0) | (phenotype > 1.0))[0]
        logger.warning(f"Phenotype scores out of [0,1] range: indices {invalid_indices}")
        phenotype = np.clip(phenotype, 0.0, 1.0)

    return phenotype


# Aliases matching older call sites / docs
extract_objective_vector = extract_phenotype_vector


def objective_distance(p1: Optional[np.ndarray], p2: Optional[np.ndarray]) -> float:
    """Normalized Euclidean distance on objective vectors; missing → ``1.0``."""
    if p1 is None or p2 is None:
        return 1.0

    p1 = np.asarray(p1, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    euclidean_dist = float(np.linalg.norm(p1 - p2))
    max_distance = float(np.sqrt(len(p1)))
    return float(min(euclidean_dist / max_distance, 1.0))


def objective_distances_batch(
    query_objective: Optional[np.ndarray],
    objectives: np.ndarray,
) -> np.ndarray:
    """Batch objective distances from one query to many targets."""
    if query_objective is None:
        if objectives.ndim == 1:
            return np.array([1.0])
        return np.ones(len(objectives))

    if objectives.ndim == 1:
        objectives = objectives.reshape(1, -1)

    query = np.asarray(query_objective, dtype=np.float32)
    diff = objectives - query
    euclidean_dists = np.linalg.norm(diff, axis=1)
    max_distance = float(np.sqrt(query.shape[0]))
    return np.clip(euclidean_dists / max_distance, 0.0, 1.0)


def pairwise_objective_matrix(objectives: np.ndarray) -> np.ndarray:
    """Pairwise ``objective_distance`` matrix (symmetric, zero diagonal).

    Rows with all-NaN are treated as missing (distance 1 to everyone, 0 to self).
    """
    obj = np.asarray(objectives, dtype=np.float64)
    if obj.ndim != 2:
        raise ValueError(f"objectives must be 2-D, got shape {obj.shape}")
    n = obj.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    missing = np.any(~np.isfinite(obj), axis=1)
    # Replace non-finite with 0 for matmul; mark missing separately
    clean = np.nan_to_num(obj, nan=0.0, posinf=0.0, neginf=0.0)
    # ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a·b
    sq = np.sum(clean * clean, axis=1, keepdims=True)
    gram = clean @ clean.T
    dist_sq = np.maximum(sq + sq.T - 2.0 * gram, 0.0)
    dist = np.sqrt(dist_sq)
    max_distance = float(np.sqrt(obj.shape[1])) if obj.shape[1] > 0 else 1.0
    dist = np.clip(dist / max_distance, 0.0, 1.0)
    np.fill_diagonal(dist, 0.0)

    if np.any(missing):
        for i in range(n):
            if missing[i]:
                dist[i, :] = 1.0
                dist[:, i] = 1.0
                dist[i, i] = 0.0
    return dist.astype(np.float64, copy=False)


# Backward-compatible names
phenotype_distance = objective_distance
phenotype_distances_batch = objective_distances_batch


__all__ = [
    "PHENOTYPE_SCORE_ORDER",
    "extract_phenotype_vector",
    "extract_objective_vector",
    "objective_distance",
    "objective_distances_batch",
    "pairwise_objective_matrix",
    "phenotype_distance",
    "phenotype_distances_batch",
]
