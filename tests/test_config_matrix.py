"""Parametrized smoke: every clustering_method × distance_method on synthetic genomes."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from speciation.clustering import cluster
from speciation.clustering.inc_dbscan import seed_from_incremental
from speciation.config import SpeciationConfig
from speciation.distance import (
    DISTANCE_METHODS,
    clear_nli_cache,
    extract_objective_vector,
    method_uses_embeddings,
    method_uses_objectives,
    method_uses_texts,
    pair_distance,
    pairwise_distance_matrix,
    set_entailment_scorer,
)
from speciation.species import SpeciesIdGenerator
from utils.evaluator_profiles import set_active_evaluator, set_active_north_star


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _genome(gid, emb, fitness=0.5, prompt=None):
    google_scores = {
        "toxicity": fitness,
        "threat": max(0.0001, fitness * 0.5),
        "profanity": max(0.0001, fitness * 0.4),
        "sexually_explicit": max(0.0001, fitness * 0.3),
        "identity_attack": max(0.0001, fitness * 0.2),
        "flirtation": max(0.0001, fitness * 0.1),
        "insult": max(0.0001, fitness * 0.45),
        "severe_toxicity": max(0.0001, fitness * 0.25),
    }
    return {
        "id": gid,
        "prompt": prompt or f"prompt text for genome {gid}",
        "prompt_embedding": emb.tolist(),
        "fitness": fitness,
        "toxicity": fitness,
        "generation": 0,
        "status": "complete",
        "moderation_result": {"google": {"scores": google_scores}},
    }


@pytest.fixture(autouse=True)
def _evaluator_and_nli():
    set_active_evaluator("google")
    set_active_north_star("toxicity")
    set_entailment_scorer(lambda a, b: 0.5 if a != b else 1.0)
    yield
    clear_nli_cache()
    set_entailment_scorer(None)


def _population():
    vecs = [
        _unit([1, 0, 0]),
        _unit([0.99, 0.1, 0]),
        _unit([0, 1, 0]),
        _unit([0.1, 0.99, 0]),
        _unit([0, 0, 1]),
        _unit([0.05, 0.05, 0.99]),
    ]
    return [_genome(i + 1, vecs[i], fitness=1.0 - 0.1 * i) for i in range(len(vecs))]


@pytest.mark.parametrize("distance_method", DISTANCE_METHODS)
def test_pair_distance_all_methods(distance_method: str) -> None:
    g0, g1 = _population()[:2]
    emb_a = np.asarray(g0["prompt_embedding"], dtype=np.float64)
    emb_b = np.asarray(g1["prompt_embedding"], dtype=np.float64)
    kwargs = {}
    if method_uses_embeddings(distance_method):
        kwargs["embedding_a"] = emb_a
        kwargs["embedding_b"] = emb_b
    if method_uses_texts(distance_method):
        kwargs["text_a"] = g0["prompt"]
        kwargs["text_b"] = g1["prompt"]
    if method_uses_objectives(distance_method):
        kwargs["objective_a"] = extract_objective_vector(g0)
        kwargs["objective_b"] = extract_objective_vector(g1)
    d = pair_distance(distance_method, alpha=0.7, w_genotype=0.7, w_phenotype=0.3, **kwargs)
    assert 0.0 <= d <= 1.0 + 1e-9
    mat = pairwise_distance_matrix(
        distance_method,
        embeddings=np.stack([emb_a, emb_b]) if method_uses_embeddings(distance_method) else None,
        texts=[g0["prompt"], g1["prompt"]] if method_uses_texts(distance_method) else None,
        objectives=[kwargs.get("objective_a"), kwargs.get("objective_b")]
        if method_uses_objectives(distance_method)
        else None,
        alpha=0.7,
        w_genotype=0.7,
        w_phenotype=0.3,
    )
    assert mat.shape == (2, 2)
    assert mat[0, 1] == pytest.approx(d, abs=1e-6)


@pytest.mark.parametrize("distance_method", DISTANCE_METHODS)
def test_leader_follower_all_distance_methods(distance_method: str) -> None:
    SpeciesIdGenerator.reset(0)
    genomes = _population()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        temp_p = td_path / "temp.json"
        (td_path / "elites.json").write_text("[]")
        state_p = td_path / "speciation_state.json"
        temp_p.write_text(json.dumps(genomes))
        state_p.write_text("{}")
        cfg = SpeciationConfig(
            clustering_method="leader_follower",
            distance_method=distance_method,
            theta_sim=0.25,
            theta_merge=0.25,
            distance_alpha=0.7,
        )
        species, touched = cluster(
            "leader_follower",
            temp_path=str(temp_p),
            speciation_state_path=str(state_p),
            current_generation=0,
            config=cfg,
        )
        assert isinstance(species, dict)
        assert isinstance(touched, set)


@pytest.mark.parametrize("distance_method", DISTANCE_METHODS)
def test_inc_dbscan_all_distance_methods(distance_method: str) -> None:
    SpeciesIdGenerator.reset(0)
    genomes = _population()
    state, species, ev = seed_from_incremental(
        genomes,
        eps=0.15,
        min_samples=2,
        current_generation=0,
        distance_method=distance_method,
        distance_alpha=0.7,
    )
    assert state is not None
    assert isinstance(species, dict)
    assert ev.get("event") == "inc_dbscan_seed"
