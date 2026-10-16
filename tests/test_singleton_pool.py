"""Singleton/noise pool (species_id=0), individuals_from_genomes filtering, redistribute rules."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from speciation.clustering._io import (
    ARCHIVE_SPECIES_ID,
    SINGLETON_SPECIES_ID,
    individuals_from_genomes,
)
from speciation.clustering.inc_dbscan import seed_from_incremental
from speciation.clustering.leader_follower import leader_follower_clustering
from speciation.distance import set_entailment_scorer, clear_nli_cache
from speciation.genome_tracker import GenomeTracker
from speciation.species import SpeciesIdGenerator
from utils.evaluator_profiles import set_active_evaluator, set_active_north_star


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _genome(gid, emb=None, fitness=0.5, prompt=None, *, with_embedding=True, with_scores=True):
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
    g = {
        "id": gid,
        "prompt": prompt if prompt is not None else f"prompt text for genome {gid}",
        "fitness": fitness,
        "toxicity": fitness,
        "generation": 0,
        "status": "complete",
    }
    if with_embedding:
        if emb is None:
            emb = _unit([1.0, 0.0, 0.0])
        g["prompt_embedding"] = np.asarray(emb, dtype=np.float64).tolist()
    if with_scores:
        g["moderation_result"] = {"google": {"scores": google_scores}}
    return g


@pytest.fixture(autouse=True)
def _iso():
    set_active_evaluator("google")
    set_active_north_star("toxicity")
    SpeciesIdGenerator.reset(0)
    set_entailment_scorer(lambda a, b: 0.5 if a != b else 1.0)
    yield
    clear_nli_cache()
    set_entailment_scorer(None)
    SpeciesIdGenerator.reset(0)


class TestIndividualsFromGenomes:
    def test_nli_keeps_text_without_embedding(self):
        genomes = [_genome(1, with_embedding=False, prompt="hello world")]
        inds = individuals_from_genomes(genomes, distance_method="nli")
        assert len(inds) == 1
        assert inds[0].embedding is None

    def test_embedding_drops_missing_embedding(self):
        genomes = [_genome(1, with_embedding=False, prompt="hello")]
        inds = individuals_from_genomes(genomes, distance_method="embedding")
        assert inds == []

    def test_objective_drops_missing_scores(self):
        genomes = [_genome(1, with_scores=False)]
        inds = individuals_from_genomes(genomes, distance_method="objective")
        assert inds == []

    def test_objective_keeps_scores_without_text(self):
        genomes = [_genome(1, prompt="")]
        # empty prompt still has embedding + scores
        inds = individuals_from_genomes(genomes, distance_method="objective")
        assert len(inds) == 1


class TestLeaderFollowerSingletonPool:
    def test_far_genome_joins_shared_pool_zero(self):
        close_a, close_b = _unit([1, 0, 0]), _unit([0.99, 0.1, 0])
        far = _unit([0, 0, 1])
        genomes = [
            _genome(1, close_a, 0.9),
            _genome(2, close_b, 0.8),
            _genome(3, far, 0.1),
        ]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_p = td_path / "temp.json"
            (td_path / "elites.json").write_text("[]")
            state_p = td_path / "speciation_state.json"
            temp_p.write_text(json.dumps(genomes))
            state_p.write_text("{}")
            species, _ = leader_follower_clustering(
                temp_path=str(temp_p),
                speciation_state_path=str(state_p),
                theta_sim=0.2,
                current_generation=0,
                distance_method="embedding",
            )
            updated = json.loads(temp_p.read_text())
            by_id = {g["id"]: g for g in updated}
            assert SINGLETON_SPECIES_ID not in species
            assert all(sid > 0 for sid in species)
            assert by_id[1]["species_id"] == by_id[2]["species_id"] > 0
            assert by_id[3]["species_id"] == SINGLETON_SPECIES_ID

    def test_two_close_genomes_mature_from_pool(self):
        a, b = _unit([1, 0]), _unit([0.99, 0.05])
        genomes = [_genome(1, a, 0.9), _genome(2, b, 0.8)]
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_p = td_path / "temp.json"
            (td_path / "elites.json").write_text("[]")
            state_p = td_path / "speciation_state.json"
            temp_p.write_text(json.dumps(genomes))
            state_p.write_text("{}")
            species, touched = leader_follower_clustering(
                temp_path=str(temp_p),
                speciation_state_path=str(state_p),
                theta_sim=0.25,
                distance_method="embedding",
            )
            assert len(species) == 1
            assert sorted(touched) == sorted(species.keys())
            assert min(species.keys()) >= 1
            updated = json.loads(temp_p.read_text())
            assert all(g["species_id"] > 0 for g in updated)


class TestIncDBSCANSingletonPool:
    def test_size1_mature_labels_demoted_to_pool(self):
        """min_samples=1 can mint size-1 clusters; they must demote to pool 0."""
        genomes = [
            _genome(1, _unit([1, 0, 0]), 0.9),
            _genome(2, _unit([0, 1, 0]), 0.8),
        ]
        state, species, ev = seed_from_incremental(
            genomes, eps=0.01, min_samples=1, distance_method="embedding"
        )
        assert len(species) == 0
        assert ev["singletons"] >= 2 or all(
            int(g["species_id"]) == SINGLETON_SPECIES_ID for g in genomes
        )
        assert all(int(lab) == SINGLETON_SPECIES_ID for lab in state.labels)
        assert all(int(g["species_id"]) == SINGLETON_SPECIES_ID for g in genomes)

class TestRedistributeSingletonAndArchive:
    def test_sid_zero_goes_to_elites_sid_minus_one_to_archive(self, monkeypatch):
        import importlib
        import sys

        # Package __init__ may shadow the submodule name with the run_speciation function.
        rs = sys.modules.get("speciation.run_speciation")
        if rs is None or not hasattr(rs, "redistribute_genomes"):
            rs = importlib.import_module("speciation.run_speciation")

        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            g_mature = _genome(1, _unit([1, 0]), 0.9)
            g_mature["species_id"] = 1
            g_pool = _genome(2, _unit([0, 1]), 0.5)
            g_pool["species_id"] = SINGLETON_SPECIES_ID
            g_arch = _genome(3, _unit([0, 0, 1]), 0.1)
            g_arch["species_id"] = ARCHIVE_SPECIES_ID

            (out / "temp.json").write_text(json.dumps([g_mature, g_pool, g_arch]))
            (out / "elites.json").write_text("[]")
            (out / "archive.json").write_text("[]")

            tracker = GenomeTracker()
            tracker.register(1, 1, 0)
            tracker.register(2, SINGLETON_SPECIES_ID, 0)
            tracker.register(3, ARCHIVE_SPECIES_ID, 0)

            logger = MagicMock()
            state = {
                "logger": logger,
                "_genome_tracker": tracker,
                "north_star_metric": "toxicity",
                "config": MagicMock(),
                "species": {},
            }

            monkeypatch.setattr(rs, "get_outputs_path", lambda: out)
            monkeypatch.setattr(rs, "_get_state", lambda: state)
            # Avoid writing tracker to disk during test
            monkeypatch.setattr(tracker, "save", lambda *a, **k: None)

            stats = rs.redistribute_genomes(temp_path=str(out / "temp.json"), current_generation=0)
            elites = json.loads((out / "elites.json").read_text())
            archive = json.loads((out / "archive.json").read_text())
            elite_ids = {g["id"] for g in elites}
            archive_ids = {g["id"] for g in archive}

            assert 1 in elite_ids
            assert 2 in elite_ids  # singleton pool stays in elites
            assert 3 in archive_ids
            assert 3 not in elite_ids
            assert stats["elites_moved"] == 2
            assert stats["archive_moved"] == 1
            assert json.loads((out / "temp.json").read_text()) == []


def test_process_generation_has_contiguous_stages_no_skip_stubs():
    """Regression: pipeline stages use descriptive names, not skipped stubs."""
    import inspect
    from speciation.run_speciation import process_generation

    src = inspect.getsource(process_generation)
    assert "(skipped)" not in src
    assert "Step 11" not in src
    assert "phase8_" not in src
    for stage in (
        "Existing-species update",
        "Species merge",
        "Capacity enforcement",
        "Stagnation freeze",
        "Genome redistribution",
        "Metrics and statistics",
    ):
        assert stage in src
    assert "Phase 7" not in src
    assert "Phase 8" not in src
    for n in range(1, 7):
        assert f"Phase {n}" not in src
