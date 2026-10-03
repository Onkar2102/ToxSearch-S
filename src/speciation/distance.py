r"""Distance controller for speciation.

Select via ``SpeciationConfig.distance_method`` / CLI ``--distance-method``:

- ``embedding`` — prompt embedding cosine dissimilarity \(d_E \in [0,1]\)
- ``objective`` — moderation score-vector Euclidean \(d_P \in [0,1]\)
- ``nli`` — symmetrized entailment dissimilarity \(d_N \in [0,1]\)
- ``nli+embedding`` — hybrid \(d_H = \alpha d_E + (1-\alpha) d_N\)
- ``embedding+objective`` — legacy blend \(w_g d_E + w_p d_P\)

Atomic metrics live in ``speciation.distance_metrics``
(``embedding`` / ``objective`` / ``nli``).
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from .distance_metrics import (
    PHENOTYPE_SCORE_ORDER,
    clear_nli_cache,
    embedding_distance,
    embedding_distances_batch,
    entailment_probability,
    extract_objective_vector,
    extract_phenotype_vector,
    nli_distance,
    nli_distances_batch,
    objective_distance,
    objective_distances_batch,
    pairwise_embedding_matrix,
    pairwise_nli_matrix,
    pairwise_objective_matrix,
    phenotype_distance,
    phenotype_distances_batch,
    semantic_distance,
    semantic_distances_batch,
    set_entailment_scorer,
    unit_normalize_rows,
)

# Canonical method names (singles + controller-managed combos)
DISTANCE_METHODS = (
    "embedding",
    "objective",
    "nli",
    "nli+embedding",
    "embedding+objective",
)


def normalize_distance_method(method: Optional[str]) -> str:
    """Normalize aliases / punctuation to a canonical ``DISTANCE_METHODS`` value."""
    raw = (method or "embedding").strip().lower().replace("-", "_").replace(" ", "")
    aliases = {
        "emb": "embedding",
        "genotype": "embedding",
        "semantic": "embedding",
        "phenotype": "objective",
        "obj": "objective",
        "scores": "objective",
        "entailment": "nli",
        "nli_emb": "nli+embedding",
        "nli_embedding": "nli+embedding",
        "nli_hybrid": "nli+embedding",
        "embedding_nli": "nli+embedding",
        "emb_nli": "nli+embedding",
        "hybrid": "nli+embedding",
        "ensemble": "embedding+objective",
        "genotype_phenotype": "embedding+objective",
        "embedding_objective": "embedding+objective",
        "embedding_phenotype": "embedding+objective",
        "objective_embedding": "embedding+objective",
        # plus-sign forms
        "nli+emb": "nli+embedding",
        "nli+embedding": "nli+embedding",
        "emb+nli": "nli+embedding",
        "embedding+nli": "nli+embedding",
        "embedding+objective": "embedding+objective",
        "genotype+phenotype": "embedding+objective",
        "embedding+phenotype": "embedding+objective",
        "objective+embedding": "embedding+objective",
    }
    if raw in aliases:
        return aliases[raw]
    plus_form = raw.replace("_", "+")
    if plus_form in aliases:
        return aliases[plus_form]
    if raw in DISTANCE_METHODS:
        return raw
    if plus_form in DISTANCE_METHODS:
        return plus_form
    raise ValueError(
        f"distance_method must be one of {DISTANCE_METHODS}, got {method!r}"
    )


def parse_distance_components(method: str) -> Tuple[str, ...]:
    """Return the atomic components used by ``method`` (order preserved)."""
    m = normalize_distance_method(method)
    if m in ("embedding", "objective", "nli"):
        return (m,)
    if m == "nli+embedding":
        return ("embedding", "nli")
    if m == "embedding+objective":
        return ("embedding", "objective")
    raise ValueError(f"Unknown distance_method={method!r}")


def _blend(parts: Sequence[float], weights: Sequence[float]) -> float:
    if len(parts) != len(weights):
        raise ValueError("blend parts/weights length mismatch")
    return float(sum(w * p for w, p in zip(weights, parts)))


def pair_distance(
    method: str,
    *,
    embedding_a: Optional[np.ndarray] = None,
    embedding_b: Optional[np.ndarray] = None,
    objective_a: Optional[np.ndarray] = None,
    objective_b: Optional[np.ndarray] = None,
    text_a: Optional[str] = None,
    text_b: Optional[str] = None,
    alpha: float = 0.7,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
) -> float:
    """Pairwise distance under the selected method (always in ``[0, 1]``)."""
    m = normalize_distance_method(method)

    def _emb() -> float:
        if embedding_a is None or embedding_b is None:
            return 1.0
        return embedding_distance(embedding_a, embedding_b)

    def _obj() -> float:
        return objective_distance(objective_a, objective_b)

    def _nli() -> float:
        return nli_distance(text_a or "", text_b or "", logger=logger)

    if m == "embedding":
        return _emb()
    if m == "objective":
        return _obj()
    if m == "nli":
        return _nli()
    if m == "nli+embedding":
        a = float(np.clip(alpha, 0.0, 1.0))
        return _blend((_emb(), _nli()), (a, 1.0 - a))
    # embedding+objective
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}"
        )
    return _blend((_emb(), _obj()), (w_genotype, w_phenotype))


def distances_to_query(
    method: str,
    *,
    query_embedding: Optional[np.ndarray] = None,
    embeddings: Optional[np.ndarray] = None,
    query_objective: Optional[np.ndarray] = None,
    objectives: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    query_text: Optional[str] = None,
    texts: Optional[Sequence[str]] = None,
    alpha: float = 0.7,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
) -> np.ndarray:
    """Distances from one query to many targets under ``method``."""
    m = normalize_distance_method(method)

    def _n_targets() -> int:
        if embeddings is not None:
            arr = np.asarray(embeddings)
            return 1 if arr.ndim == 1 else len(arr)
        if objectives is not None:
            return len(objectives) if isinstance(objectives, list) else (
                1 if np.asarray(objectives).ndim == 1 else len(objectives)
            )
        if texts is not None:
            return len(texts)
        return 0

    def _emb_batch() -> np.ndarray:
        if query_embedding is None or embeddings is None:
            return np.ones(_n_targets(), dtype=np.float64)
        return embedding_distances_batch(query_embedding, embeddings)

    def _obj_batch() -> np.ndarray:
        if objectives is None:
            return np.zeros(0, dtype=np.float64)
        if isinstance(objectives, list):
            n = len(objectives)
            out = np.full(n, 1.0, dtype=np.float64)
            valid_idx = []
            valid_rows = []
            for i, p in enumerate(objectives):
                if p is not None:
                    valid_idx.append(i)
                    valid_rows.append(p)
            if query_objective is not None and valid_rows:
                batch = objective_distances_batch(
                    query_objective, np.asarray(valid_rows, dtype=np.float32)
                )
                for j, i in enumerate(valid_idx):
                    out[i] = batch[j]
            return out
        return objective_distances_batch(query_objective, np.asarray(objectives))

    def _nli_batch() -> np.ndarray:
        if texts is None:
            return np.zeros(0, dtype=np.float64)
        return nli_distances_batch(query_text or "", texts, logger=logger)

    if m == "embedding":
        return _emb_batch()
    if m == "objective":
        return _obj_batch()
    if m == "nli":
        return _nli_batch()
    if m == "nli+embedding":
        a = float(np.clip(alpha, 0.0, 1.0))
        return a * _emb_batch() + (1.0 - a) * _nli_batch()
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}"
        )
    return w_genotype * _emb_batch() + w_phenotype * _obj_batch()


def pairwise_distance_matrix(
    method: str,
    *,
    embeddings: Optional[np.ndarray] = None,
    objectives: Optional[np.ndarray] = None,
    texts: Optional[Sequence[str]] = None,
    alpha: float = 0.7,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
) -> np.ndarray:
    """Full pairwise matrix for DBSCAN / bulk clustering."""
    m = normalize_distance_method(method)

    def _emb_mat() -> np.ndarray:
        if embeddings is None:
            return np.zeros((0, 0), dtype=np.float64)
        emb = unit_normalize_rows(np.asarray(embeddings, dtype=np.float64))
        return pairwise_embedding_matrix(emb)

    def _obj_mat() -> np.ndarray:
        if objectives is None:
            return np.zeros((0, 0), dtype=np.float64)
        return pairwise_objective_matrix(np.asarray(objectives, dtype=np.float64))

    def _nli_mat() -> np.ndarray:
        return pairwise_nli_matrix(list(texts or []), logger=logger)

    if m == "embedding":
        return _emb_mat()
    if m == "objective":
        return _obj_mat()
    if m == "nli":
        return _nli_mat()
    if m == "nli+embedding":
        a = float(np.clip(alpha, 0.0, 1.0))
        return a * _emb_mat() + (1.0 - a) * _nli_mat()
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(
            f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}"
        )
    return w_genotype * _emb_mat() + w_phenotype * _obj_mat()


def ensemble_distance(
    e1: np.ndarray,
    e2: np.ndarray,
    p1: Optional[np.ndarray] = None,
    p2: Optional[np.ndarray] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
) -> float:
    """Legacy name for ``embedding+objective`` pairwise distance."""
    return pair_distance(
        "embedding+objective",
        embedding_a=e1,
        embedding_b=e2,
        objective_a=p1,
        objective_b=p2,
        w_genotype=w_genotype,
        w_phenotype=w_phenotype,
    )


def ensemble_distances_batch(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    query_phenotype: Optional[np.ndarray] = None,
    phenotypes: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
) -> np.ndarray:
    """Legacy name for ``embedding+objective`` batch distances."""
    return distances_to_query(
        "embedding+objective",
        query_embedding=query_embedding,
        embeddings=embeddings,
        query_objective=query_phenotype,
        objectives=phenotypes,
        w_genotype=w_genotype,
        w_phenotype=w_phenotype,
    )


__all__ = [
    "DISTANCE_METHODS",
    "normalize_distance_method",
    "parse_distance_components",
    "pair_distance",
    "distances_to_query",
    "pairwise_distance_matrix",
    # re-exports
    "semantic_distance",
    "semantic_distances_batch",
    "embedding_distance",
    "embedding_distances_batch",
    "pairwise_embedding_matrix",
    "unit_normalize_rows",
    "PHENOTYPE_SCORE_ORDER",
    "extract_phenotype_vector",
    "extract_objective_vector",
    "objective_distance",
    "objective_distances_batch",
    "pairwise_objective_matrix",
    "phenotype_distance",
    "phenotype_distances_batch",
    "nli_distance",
    "nli_distances_batch",
    "pairwise_nli_matrix",
    "set_entailment_scorer",
    "clear_nli_cache",
    "entailment_probability",
    "ensemble_distance",
    "ensemble_distances_batch",
]
