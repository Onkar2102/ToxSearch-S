"""Unit tests for speciation.distance."""

from __future__ import annotations

import numpy as np
import pytest

from speciation.distance import (
    embedding_distance,
    embedding_distances_batch,
    ensemble_distance,
    semantic_distance,
    semantic_distances_batch,
)


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


def test_embedding_distance_is_half_semantic() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    assert embedding_distance(e1, e2) == pytest.approx(semantic_distance(e1, e2) / 2.0)
    assert 0.0 <= embedding_distance(e1, e2) <= 1.0


def test_embedding_distances_batch_matches_scalar() -> None:
    q = _unit([1.0, 0.0, 0.0])
    others = np.stack([_unit([1.0, 0.0, 0.0]), _unit([0.0, 1.0, 0.0])])
    batch = embedding_distances_batch(q, others)
    assert batch[0] == pytest.approx(embedding_distance(q, others[0]))
    assert batch[1] == pytest.approx(embedding_distance(q, others[1]))


def test_normalize_distance_method_aliases() -> None:
    from speciation.distance import normalize_distance_method

    assert normalize_distance_method("emb") == "embedding"
    assert normalize_distance_method("phenotype") == "objective"
    assert normalize_distance_method("nli_hybrid") == "nli+embedding"
    assert normalize_distance_method("nli+embedding") == "nli+embedding"
    assert normalize_distance_method("ensemble") == "embedding+objective"
    with pytest.raises(ValueError):
        normalize_distance_method("cosine")


def test_nli_plus_embedding_hybrid() -> None:
    from speciation.distance import pair_distance, set_entailment_scorer, clear_nli_cache

    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    d_e = embedding_distance(e1, e2)

    # Inject deterministic entailment: always 0.0 → d_N = 1.0
    set_entailment_scorer(lambda a, b: 0.0)
    try:
        d = pair_distance(
            "nli+embedding",
            embedding_a=e1,
            embedding_b=e2,
            text_a="hello",
            text_b="world",
            alpha=0.7,
        )
        assert d == pytest.approx(0.7 * d_e + 0.3 * 1.0)
    finally:
        set_entailment_scorer(None)
        clear_nli_cache()


def test_config_distance_method() -> None:
    from speciation.config import SpeciationConfig

    cfg = SpeciationConfig(distance_method="nli+embedding", distance_alpha=0.5)
    assert cfg.distance_method == "nli+embedding"
    assert cfg.distance_alpha == 0.5
    with pytest.raises((AssertionError, ValueError)):
        SpeciationConfig(distance_method="nope")
