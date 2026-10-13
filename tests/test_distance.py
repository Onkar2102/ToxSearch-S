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


def test_ensemble_distance_in_unit_interval() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    p = np.zeros(2, dtype=np.float32)
    d = ensemble_distance(e1, e2, p, p, w_genotype=0.7, w_phenotype=0.3)
    assert 0.0 <= d <= 1.0


def test_ensemble_distance_requires_objectives() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    with pytest.raises(ValueError, match="objective"):
        ensemble_distance(e1, e2, w_genotype=0.7, w_phenotype=0.3)


def test_pair_distance_embedding_requires_embeddings() -> None:
    from speciation.distance import pair_distance

    with pytest.raises(ValueError, match="embedding"):
        pair_distance("embedding", embedding_a=None, embedding_b=_unit([1.0, 0.0]))


def test_ensemble_weights_must_sum_to_one() -> None:
    e1 = _unit([1.0, 0.0])
    e2 = _unit([0.0, 1.0])
    p = np.zeros(2, dtype=np.float32)
    with pytest.raises(ValueError):
        ensemble_distance(e1, e2, p, p, w_genotype=0.6, w_phenotype=0.3)


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


def test_pure_nli_distance_no_embeddings() -> None:
    from speciation.distance import (
        clear_nli_cache,
        method_uses_embeddings,
        nli_distance,
        pair_distance,
        pairwise_distance_matrix,
        set_entailment_scorer,
    )

    assert method_uses_embeddings("nli") is False
    assert method_uses_embeddings("embedding") is True

    # Legacy entailment inject maps remainder to contradiction → same as old 1-P_E.
    # Symmetric entailment 1.0 → d_N = 0; identical texts already 0
    set_entailment_scorer(lambda a, b: 1.0 if a != b else 1.0)
    try:
        assert nli_distance("same", "same") == pytest.approx(0.0)
        assert pair_distance("nli", text_a="a", text_b="a") == pytest.approx(0.0)
        # Entailment 0 both ways → d_N = 1
        set_entailment_scorer(lambda a, b: 0.0)
        assert pair_distance("nli", text_a="hello", text_b="world") == pytest.approx(1.0)
        # Entailment 0.5 both ways → d_N = 0.5
        set_entailment_scorer(lambda a, b: 0.5)
        assert pair_distance("nli", text_a="hello", text_b="world") == pytest.approx(0.5)
        mat = pairwise_distance_matrix("nli", texts=["x", "y", "x"])
        assert mat.shape == (3, 3)
        assert mat[0, 0] == pytest.approx(0.0)
        assert mat[0, 2] == pytest.approx(0.0)  # identical text
        assert mat[0, 1] == pytest.approx(0.5)
    finally:
        set_entailment_scorer(None)
        clear_nli_cache()


def test_nli_distance_qiu_expected_squared() -> None:
    """d_N = mean(P_C) + 0.5 * mean(P_N) (Qiu Eq. 9, symmetrized)."""
    from speciation.distance import clear_nli_cache, nli_distance, set_nli_scorer

    # Pure contradiction both ways → d_N = 1
    set_nli_scorer(lambda a, b: (0.0, 0.0, 1.0))
    try:
        assert nli_distance("a", "b") == pytest.approx(1.0)
        # Pure neutral both ways → d_N = 0.5
        set_nli_scorer(lambda a, b: (0.0, 1.0, 0.0))
        assert nli_distance("a", "b") == pytest.approx(0.5)
        # Pure entailment both ways → d_N = 0
        set_nli_scorer(lambda a, b: (1.0, 0.0, 0.0))
        assert nli_distance("a", "b") == pytest.approx(0.0)
        # Mixed / asymmetric: ab=(0.2,0.3,0.5), ba=(0.4,0.1,0.5)
        # d_ab = 0.5 + 0.5*0.3 = 0.65; d_ba = 0.5 + 0.5*0.1 = 0.55; mean = 0.6
        def _mixed(a: str, b: str):
            if a == "u" and b == "v":
                return (0.2, 0.3, 0.5)
            if a == "v" and b == "u":
                return (0.4, 0.1, 0.5)
            return (1.0 / 3.0,) * 3

        set_nli_scorer(_mixed)
        assert nli_distance("u", "v") == pytest.approx(0.6)
        assert nli_distance("same", "same") == pytest.approx(0.0)
    finally:
        set_nli_scorer(None)
        clear_nli_cache()


def test_nli_requires_non_empty_text() -> None:
    from speciation.distance import clear_nli_cache, pair_distance, set_entailment_scorer

    set_entailment_scorer(lambda a, b: 0.5)
    try:
        with pytest.raises(ValueError, match="text"):
            pair_distance("nli", text_a="hello", text_b="")
        # pure nli must NOT require embeddings
        d = pair_distance("nli", text_a="hello", text_b="world", embedding_a=None, embedding_b=None)
        assert 0.0 <= d <= 1.0
    finally:
        set_entailment_scorer(None)
        clear_nli_cache()


def test_apply_distance_runtime_config_sets_nli_model() -> None:
    from speciation.distance import (
        apply_distance_runtime_config,
        get_nli_model_name,
        set_nli_model_name,
    )

    prev = get_nli_model_name()
    try:
        m = apply_distance_runtime_config(
            distance_method="nli",
            nli_model="test/mnli-dummy",
        )
        assert m == "nli"
        assert get_nli_model_name() == "test/mnli-dummy"
    finally:
        set_nli_model_name(prev)


def test_config_distance_method() -> None:
    from speciation.config import SpeciationConfig
    from speciation.distance import get_nli_model_name, set_nli_model_name

    prev = get_nli_model_name()
    try:
        cfg = SpeciationConfig(
            distance_method="nli+embedding",
            distance_alpha=0.5,
            nli_model="org/custom-mnli",
        )
        assert cfg.distance_method == "nli+embedding"
        assert cfg.distance_alpha == 0.5
        assert cfg.nli_model == "org/custom-mnli"
        assert get_nli_model_name() == "org/custom-mnli"
        with pytest.raises((AssertionError, ValueError)):
            SpeciationConfig(distance_method="nope")
    finally:
        set_nli_model_name(prev)


@pytest.mark.parametrize(
    "method,uses_emb,uses_txt,uses_obj",
    [
        ("embedding", True, False, False),
        ("objective", False, False, True),
        ("nli", False, True, False),
        ("nli+embedding", True, True, False),
        ("embedding+objective", True, False, True),
    ],
)
def test_method_uses_flags(method, uses_emb, uses_txt, uses_obj) -> None:
    from speciation.distance import (
        method_uses_embeddings,
        method_uses_objectives,
        method_uses_texts,
    )

    assert method_uses_embeddings(method) is uses_emb
    assert method_uses_texts(method) is uses_txt
    assert method_uses_objectives(method) is uses_obj


def test_pair_distance_objective_requires_vectors() -> None:
    from speciation.distance import pair_distance

    with pytest.raises(ValueError, match="objective"):
        pair_distance("objective", objective_a=None, objective_b=np.zeros(2))


def test_pair_distance_objective_unit_interval() -> None:
    from speciation.distance import extract_objective_vector, pair_distance

    def _g(tox: float) -> dict:
        return {
            "moderation_result": {
                "google": {
                    "scores": {
                        "toxicity": tox,
                        "threat": 0.1,
                        "profanity": 0.1,
                        "sexually_explicit": 0.1,
                        "identity_attack": 0.1,
                        "flirtation": 0.1,
                        "insult": 0.1,
                        "severe_toxicity": 0.1,
                    }
                }
            }
        }

    d = pair_distance(
        "objective",
        objective_a=extract_objective_vector(_g(0.9)),
        objective_b=extract_objective_vector(_g(0.1)),
    )
    assert 0.0 <= d <= 1.0
    assert d > 0.0
