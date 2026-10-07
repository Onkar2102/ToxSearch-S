"""Clustering package: leader–follower and Incremental DBSCAN behind ``cluster()``."""

from __future__ import annotations

from typing import Dict, Optional, Set, Tuple

from ..config import SpeciationConfig
from ..species import Species
from ._io import ARCHIVE_SPECIES_ID
from .dbscan import dbscan_cluster_population, dbscan_precomputed
from .leader_follower import leader_follower_clustering

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


def cluster(
    method: str,
    *,
    temp_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    include_elites: bool = False,
    logger=None,
    genome_tracker=None,
    events_tracker=None,
) -> Tuple[Dict[int, Species], Set[int]]:
    """Dispatch to leader–follower or Incremental DBSCAN using ``SpeciationConfig``.

    For ``dbscan``, prefer ``run_inc_dbscan.run_inc_dbscan_speciation`` from the
    orchestration layer. This helper remains for tests / direct calls and runs
    the IncDBSCAN generation step then returns species from state.
    """
    del include_elites  # IncDBSCAN does not use elites
    cfg = config or SpeciationConfig()
    log = logger or get_logger("Clustering")
    method = (method or cfg.clustering_method or "leader_follower").strip().lower().replace("-", "_")
    if method == "lf":
        method = "leader_follower"

    if method == "leader_follower":
        log.info("Clustering method: leader_follower (distance_method=%s)", cfg.distance_method)
        return leader_follower_clustering(
            temp_path=temp_path,
            speciation_state_path=speciation_state_path,
            theta_sim=cfg.theta_sim,
            current_generation=current_generation,
            w_genotype=cfg.w_genotype,
            w_phenotype=cfg.w_phenotype,
            distance_method=cfg.distance_method,
            distance_alpha=cfg.distance_alpha,
            min_island_size=cfg.min_island_size,
            logger=log,
            genome_tracker=genome_tracker,
            events_tracker=events_tracker,
        )

    if method == "dbscan":
        log.info(
            "Clustering method: dbscan/IncDBSCAN (eps=%s, min_samples=%d, distance_method=%s)",
            cfg.dbscan_eps if cfg.dbscan_eps is not None else cfg.theta_sim,
            cfg.dbscan_min_samples,
            cfg.distance_method,
        )
        from ..run_inc_dbscan import process_generation_inc_dbscan
        from ._io import load_species_state, default_paths
        process_generation_inc_dbscan(
            temp_path=temp_path,
            current_generation=current_generation,
            config=cfg,
        )
        _, _, state_p = default_paths(temp_path, speciation_state_path=speciation_state_path)
        eps = float(cfg.dbscan_eps if cfg.dbscan_eps is not None else cfg.theta_sim)
        species = load_species_state(state_p, eps, logger=log)
        return species, set(species.keys())

    raise ValueError(f"Unknown clustering_method={method!r}; use 'leader_follower' or 'dbscan'")


__all__ = [
    "ARCHIVE_SPECIES_ID",
    "cluster",
    "dbscan_precomputed",
    "dbscan_cluster_population",
    "leader_follower_clustering",
]
