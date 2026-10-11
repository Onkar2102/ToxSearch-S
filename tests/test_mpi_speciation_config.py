"""MPI coverage for speciation config + clustering×distance dispatch."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.mpi


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _genome(gid, emb, fitness=0.5):
    return {
        "id": gid,
        "prompt": f"prompt text for genome {gid}",
        "prompt_embedding": emb.tolist(),
        "fitness": fitness,
        "toxicity": fitness,
        "generation": 0,
        "status": "complete",
        "moderation_result": {
            "google": {
                "scores": {
                    "toxicity": fitness,
                    "threat": max(0.0001, fitness * 0.5),
                    "profanity": max(0.0001, fitness * 0.4),
                    "sexually_explicit": max(0.0001, fitness * 0.3),
                    "identity_attack": max(0.0001, fitness * 0.2),
                    "flirtation": max(0.0001, fitness * 0.1),
                    "insult": max(0.0001, fitness * 0.45),
                    "severe_toxicity": max(0.0001, fitness * 0.25),
                }
            }
        },
    }


def test_broadcast_speciation_config_all_distance_methods(comm, rank, size, logger):
    """Master broadcasts each distance_method; workers validate SpeciationConfig."""
    from speciation.config import SpeciationConfig
    from speciation.distance import DISTANCE_METHODS, set_entailment_scorer, clear_nli_cache

    set_entailment_scorer(lambda a, b: 0.5 if a != b else 1.0)
    try:
        for method in DISTANCE_METHODS:
            payload = None
            if rank == 0:
                payload = {
                    "clustering_method": "leader_follower",
                    "distance_method": method,
                    "theta_sim": 0.25,
                    "theta_merge": 0.25,
                    "distance_alpha": 0.7,
                }
            payload = comm.bcast(payload, root=0)
            cfg = SpeciationConfig(**payload)
            assert cfg.distance_method == method
            assert cfg.clustering_method == "leader_follower"
            logger.info("rank %d ok distance_method=%s", rank, method)
        comm.Barrier()
    finally:
        clear_nli_cache()
        set_entailment_scorer(None)


@pytest.mark.parametrize("clustering_method", ["leader_follower", "dbscan"])
def test_mpi_cluster_dispatch_embedding(comm, rank, size, logger, clustering_method):
    """Each rank clusters a tiny synthetic population (embedding distance)."""
    from speciation.clustering import cluster
    from speciation.clustering.inc_dbscan import seed_from_batch
    from speciation.config import SpeciationConfig
    from speciation.species import SpeciesIdGenerator
    from utils.evaluator_profiles import set_active_evaluator, set_active_north_star

    set_active_evaluator("google")
    set_active_north_star("toxicity")
    SpeciesIdGenerator.reset(1000 * (rank + 1))

    genomes = [
        _genome(rank * 10 + 1, _unit([1, 0, 0]), 0.9),
        _genome(rank * 10 + 2, _unit([0.99, 0.1, 0]), 0.8),
        _genome(rank * 10 + 3, _unit([0, 1, 0]), 0.1),
    ]

    if clustering_method == "dbscan":
        state, species, ev = seed_from_batch(
            genomes, eps=0.15, min_samples=2, distance_method="embedding"
        )
        assert state is not None
        assert isinstance(species, dict)
        assert ev.get("event") == "inc_dbscan_init"
    else:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            temp_p = td_path / "temp.json"
            (td_path / "elites.json").write_text("[]")
            state_p = td_path / "speciation_state.json"
            temp_p.write_text(json.dumps(genomes))
            state_p.write_text("{}")
            cfg = SpeciationConfig(
                clustering_method="leader_follower",
                distance_method="embedding",
                theta_sim=0.25,
                theta_merge=0.25,
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

    # All ranks must finish; gather species counts for a collective check.
    n_species = len(species)
    counts = comm.gather(n_species, root=0)
    if rank == 0:
        assert len(counts) == size
        assert all(isinstance(c, int) and c >= 0 for c in counts)
        logger.info("%s species counts across ranks: %s", clustering_method, counts)
    comm.Barrier()
