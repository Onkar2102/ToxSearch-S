"""IncDBSCAN must record the same cluster-quality / diversity fields as L–F."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from speciation.config import SpeciationConfig
from speciation.run_inc_dbscan import record_inc_cluster_metrics
from speciation.species import Individual, Species


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _ind(gid, sid, emb, fitness, pheno=None):
    p = np.asarray(pheno if pheno is not None else [fitness] * 8, dtype=np.float32)
    return Individual(
        id=gid, prompt=f"q{gid}?", fitness=fitness,
        embedding=_unit(emb), phenotype=p, species_id=sid,
    )


def _genome(ind: Individual) -> dict:
    return {
        "id": ind.id,
        "prompt": ind.prompt,
        "prompt_embedding": ind.embedding.tolist(),
        "species_id": ind.species_id,
        "density_label": ind.species_id,
        "fitness": ind.fitness,
        "toxicity": ind.fitness,
    }


def _species(sid, members):
    champ = max(members, key=lambda m: m.fitness)
    return Species(
        id=sid, leader=champ, members=list(members), radius=0.25,
        max_fitness=champ.fitness, species_state="active",
    )


def _write_inc_outputs(tmp: Path, genomes):
    tmp.mkdir(parents=True, exist_ok=True)
    (tmp / "temp.json").write_text(json.dumps(genomes), encoding="utf-8")
    (tmp / "run_config.json").write_text(
        json.dumps({"speciation": {"clustering_method": "dbscan"}}),
        encoding="utf-8",
    )
    return str(tmp / "temp.json")


def test_two_mature_species_writes_cluster_quality(tmp_path: Path):
    a = [_ind(1, 1, [1, 0, 0], 0.9), _ind(2, 1, [0.98, 0.1, 0], 0.8), _ind(3, 1, [0.97, 0.2, 0], 0.7)]
    b = [_ind(4, 2, [0, 1, 0], 0.6), _ind(5, 2, [0.1, 0.98, 0], 0.5), _ind(6, 2, [0.2, 0.97, 0], 0.4)]
    species = {1: _species(1, a), 2: _species(2, b)}
    genomes = [_genome(x) for x in a + b]
    temp_p = _write_inc_outputs(tmp_path, genomes)
    quality = record_inc_cluster_metrics(
        species, genomes, cfg=SpeciationConfig(), outputs_path=tmp_path,
        temp_path=temp_p, current_generation=4, logger=MagicMock(),
    )
    assert quality["cluster_quality"] is not None
    cq = quality["cluster_quality"]
    assert cq["num_clusters"] == 2
    assert cq["num_samples"] == 6
    assert "silhouette_score" in cq
    assert "davies_bouldin_index" in cq
    assert "calinski_harabasz_index" in cq
    assert "qd_score" in cq
    assert quality["inter_species_diversity"] > 0.0
    assert quality["intra_species_diversity"] >= 0.0


def test_single_mature_species_skips_sklearn_like_lf(tmp_path: Path):
    members = [
        _ind(1, 1, [1, 0, 0], 0.9),
        _ind(2, 1, [0.98, 0.1, 0], 0.8),
        _ind(3, 1, [0.97, 0.2, 0], 0.7),
        _ind(4, 1, [0.96, 0.25, 0], 0.6),
    ]
    species = {1: _species(1, members)}
    genomes = [_genome(x) for x in members]
    temp_p = _write_inc_outputs(tmp_path, genomes)
    quality = record_inc_cluster_metrics(
        species, genomes, cfg=SpeciationConfig(), outputs_path=tmp_path,
        temp_path=temp_p, current_generation=2, logger=MagicMock(),
    )
    assert quality["cluster_quality"] is None
    assert quality["inter_species_diversity"] == 0.0
    assert quality["intra_species_diversity"] > 0.0


def test_tracker_copies_inc_quality_from_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    rs = importlib.import_module("speciation.run_speciation")

    _write_inc_outputs(tmp_path, [])
    tracker_path = tmp_path / "EvolutionTracker.json"
    tracker_path.write_text(json.dumps({
        "status": "running",
        "total_generations": 1,
        "selection_mode": "default",
        "generations": [{"generation_number": 1, "speciation": {}}],
    }), encoding="utf-8")
    monkeypatch.setattr(rs, "get_outputs_path", lambda: tmp_path)

    cq = {
        "silhouette_score": 0.42,
        "davies_bouldin_index": 1.1,
        "calinski_harabasz_index": 12.0,
        "qd_score": 0.3,
        "num_samples": 6,
        "num_clusters": 2,
    }
    rs.update_evolution_tracker_with_speciation(
        str(tracker_path),
        current_generation=1,
        speciation_result={
            "inc_dbscan": True,
            "active_species_count": 2,
            "frozen_species_count": 0,
            "density_size": 6,
            "noise_count": 0,
            "singleton_species_count": 0,
            "clustered_count": 6,
            "inter_species_diversity": 0.55,
            "intra_species_diversity": 0.12,
            "cluster_quality": cq,
        },
        speciation_stats={"metrics_summary": {}},
        logger=MagicMock(),
    )
    stored = json.loads(tracker_path.read_text(encoding="utf-8"))
    spec = stored["generations"][0]["speciation"]
    assert spec["inter_species_diversity"] == pytest.approx(0.55)
    assert spec["intra_species_diversity"] == pytest.approx(0.12)
    assert spec["cluster_quality"]["silhouette_score"] == pytest.approx(0.42)


def test_inc_metrics_history_and_summary_persist(tmp_path: Path):
    from speciation.clustering._io import write_species_state
    from speciation.run_inc_dbscan import _append_inc_metrics_history

    state_path = tmp_path / "speciation_state.json"
    first = {
        "generation": 1,
        "species_count": 2,
        "best_fitness": 0.8,
        "speciation_events": 2,
        "merge_events": 1,
        "extinction_events": 0,
        "inter_species_diversity": 0.4,
        "intra_species_diversity": 0.1,
        "cluster_quality": {"silhouette_score": 0.3, "num_clusters": 2},
    }
    metrics = _append_inc_metrics_history(state_path, first)
    write_species_state(state_path, {}, generation=1, extra={"metrics": metrics})
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored["generation"] == 1
    assert len(stored["metrics"]["history"]) == 1
    assert stored["metrics"]["history"][0]["cluster_quality"]["silhouette_score"] == pytest.approx(0.3)
    assert stored["metrics"]["summary"]["total_speciation_events"] == 2
    assert stored["metrics"]["summary"]["total_merge_events"] == 1
    assert stored["metrics"]["summary"]["total_extinction_events"] == 0

    second = {
        "generation": 2,
        "species_count": 3,
        "best_fitness": 0.9,
        "speciation_events": 1,
        "merge_events": 2,
        "extinction_events": 1,
        "inter_species_diversity": 0.5,
        "intra_species_diversity": 0.12,
        "cluster_quality": {"silhouette_score": 0.42, "num_clusters": 3},
    }
    metrics = _append_inc_metrics_history(state_path, second)
    write_species_state(state_path, {}, generation=2, extra={"metrics": metrics})
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    assert stored["generation"] == 2
    assert len(stored["metrics"]["history"]) == 2
    assert stored["metrics"]["summary"]["total_generations"] == 2
    assert stored["metrics"]["summary"]["total_speciation_events"] == 3
    assert stored["metrics"]["summary"]["total_merge_events"] == 3
    assert stored["metrics"]["summary"]["total_extinction_events"] == 1
    assert stored["metrics"]["summary"]["best_fitness_ever"] == pytest.approx(0.9)
    assert stored["metrics"]["history"][-1]["cluster_quality"]["silhouette_score"] == pytest.approx(0.42)


def test_record_inc_quality_includes_event_counts(tmp_path: Path):
    a = [_ind(1, 1, [1, 0, 0], 0.9), _ind(2, 1, [0.98, 0.1, 0], 0.8), _ind(3, 1, [0.97, 0.2, 0], 0.7)]
    b = [_ind(4, 2, [0, 1, 0], 0.6), _ind(5, 2, [0.1, 0.98, 0], 0.5), _ind(6, 2, [0.2, 0.97, 0], 0.4)]
    species = {1: _species(1, a), 2: _species(2, b)}
    genomes = [_genome(x) for x in a + b]
    temp_p = _write_inc_outputs(tmp_path, genomes)
    quality = record_inc_cluster_metrics(
        species, genomes, cfg=SpeciationConfig(), outputs_path=tmp_path,
        temp_path=temp_p, current_generation=4, logger=MagicMock(),
        speciation_events=2, merge_events=1, extinction_events=0,
    )
    assert quality["speciation_events"] == 2
    assert quality["merge_events"] == 1
    assert quality["extinction_events"] == 0
    assert quality["best_fitness"] == pytest.approx(0.9)
    assert quality["cluster_quality"] is not None


def test_lf_save_state_writes_generation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    from speciation.config import SpeciationConfig
    from speciation.genome_tracker import GenomeTracker
    from speciation.metrics import SpeciationMetricsTracker

    rs = importlib.import_module("speciation.run_speciation")
    (tmp_path / "elites.json").write_text("[]", encoding="utf-8")
    (tmp_path / "archive.json").write_text("[]", encoding="utf-8")
    logger = MagicMock()
    state = {
        "config": SpeciationConfig(),
        "logger": logger,
        "species": {},
        "historical_species": {},
        "global_best": None,
        "metrics_tracker": SpeciationMetricsTracker(logger=logger),
        "_genome_tracker": GenomeTracker(logger=logger),
        "_current_gen_events": {
            "speciation": 0, "merge": 0, "extinction": 0, "moved_to_archive": 0,
        },
        "_archived_count": 0,
        "north_star_metric": "toxicity",
    }
    monkeypatch.setattr(rs, "get_outputs_path", lambda: tmp_path)
    monkeypatch.setattr(rs, "_get_state", lambda: state)
    rs.save_state(str(tmp_path / "speciation_state.json"), generation=7)
    payload = json.loads((tmp_path / "speciation_state.json").read_text(encoding="utf-8"))
    assert payload["generation"] == 7
    assert payload["noise_count"] == 0
    assert payload["singleton_species_count"] == 0


def test_tracker_lf_noise_from_speciation_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    from speciation.metrics import SpeciationMetricsTracker

    rs = importlib.import_module("speciation.run_speciation")
    _write_inc_outputs(tmp_path, [])
    (tmp_path / "speciation_state.json").write_text(json.dumps({
        "generation": 5,
        "species": {},
        "noise": {"species_id": 0, "count": 1023, "genome_ids": ["1"]},
        "noise_count": 1023,
        "singleton_species_count": 1023,
    }), encoding="utf-8")
    tracker_path = tmp_path / "EvolutionTracker.json"
    tracker_path.write_text(json.dumps({
        "status": "running",
        "total_generations": 1,
        "selection_mode": "default",
        "generations": [{"generation_number": 1, "speciation": {}}],
    }), encoding="utf-8")
    monkeypatch.setattr(rs, "get_outputs_path", lambda: tmp_path)
    monkeypatch.setattr(rs, "_get_state", lambda: {
        "metrics_tracker": SpeciationMetricsTracker(),
        "species": {},
    })
    rs.update_evolution_tracker_with_speciation(
        str(tracker_path),
        current_generation=1,
        speciation_result={
            "active_species_count": 1,
            "frozen_species_count": 0,
            "inter_species_diversity": 0.3,
            "intra_species_diversity": 0.1,
        },
        speciation_stats={"metrics_summary": {}},
        logger=MagicMock(),
    )
    stored = json.loads(tracker_path.read_text(encoding="utf-8"))
    spec = stored["generations"][0]["speciation"]
    assert spec["noise_count"] == 1023
    assert spec["singleton_species_count"] == 1023
    assert stored["speciation_summary"]["current_noise_count"] == 1023
    assert stored["speciation_summary"]["current_singleton_species_count"] == 1023


def test_inc_tracker_keeps_result_noise_not_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import importlib

    rs = importlib.import_module("speciation.run_speciation")
    _write_inc_outputs(tmp_path, [])
    (tmp_path / "speciation_state.json").write_text(json.dumps({
        "noise_count": 99,
        "singleton_species_count": 99,
    }), encoding="utf-8")
    tracker_path = tmp_path / "EvolutionTracker.json"
    tracker_path.write_text(json.dumps({
        "status": "running",
        "total_generations": 1,
        "selection_mode": "default",
        "generations": [{"generation_number": 1, "speciation": {}}],
    }), encoding="utf-8")
    monkeypatch.setattr(rs, "get_outputs_path", lambda: tmp_path)
    rs.update_evolution_tracker_with_speciation(
        str(tracker_path),
        current_generation=1,
        speciation_result={
            "inc_dbscan": True,
            "active_species_count": 2,
            "frozen_species_count": 0,
            "density_size": 10,
            "noise_count": 3,
            "singleton_species_count": 2,
            "clustered_count": 7,
        },
        speciation_stats={"metrics_summary": {}},
        logger=MagicMock(),
    )
    spec = json.loads(tracker_path.read_text(encoding="utf-8"))["generations"][0]["speciation"]
    assert spec["noise_count"] == 3
    assert spec["singleton_species_count"] == 2
