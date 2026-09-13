"""Unit tests for speciation.distance."""

from __future__ import annotations

import numpy as np
import pytest

from speciation.distance import ensemble_distance, semantic_distance, semantic_distances_batch


def _unit(v: list[float]) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def test_semantic_distance_identical() -> None:
    e = _unit([1.0, 0.0, 0.0])
    assert semantic_distance(e, e) == pytest.approx(0.0)


def test_semantic_distance_opposite() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([-1.0, 0.0])
    assert semantic_distance(e1, e2) == pytest.approx(2.0)


def test_semantic_distances_batch_matches_scalar() -> None:
    q = _unit([1.0, 0.0, 0.0])
    others = np.stack([_unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0])])
    batch = semantic_distances_batch(q, others)
    assert batch[0] == pytest.approx(semantic_distance(q, others[0]))
    assert batch[1] == pytest.approx(semantic_distance(q, others[1]))


def test_ensemble_distance_requires_unit_embeddings() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    d = ensemble_distance(e1, e2, w_genotype=0.7, w_phenotype=0.3)
    assert 0.0 <= d <= 1.0


def test_ensemble_weights_must_sum_to_one() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    with pytest.raises(ValueError):
        ensemble_distance(e1, e2, w_genotype=0.6, w_phenotype=0.3)
