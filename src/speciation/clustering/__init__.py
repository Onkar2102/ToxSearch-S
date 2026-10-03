"""Clustering package: leader–follower and DBSCAN behind one ``cluster()`` entry."""

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
    """Dispatch to leader–follower or DBSCAN using ``SpeciationConfig``."""
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
        eps = cfg.dbscan_eps if cfg.dbscan_eps is not None else cfg.theta_sim
        log.info(
            "Clustering method: dbscan (eps=%.4f, min_samples=%d, include_elites=%s, distance_method=%s)",
            eps, cfg.dbscan_min_samples, include_elites, cfg.distance_method,
        )
        return dbscan_cluster_population(
            temp_path=temp_path,
            speciation_state_path=speciation_state_path,
            eps=float(eps),
            min_samples=int(cfg.dbscan_min_samples),
            current_generation=current_generation,
            include_elites=include_elites,
            distance_method=cfg.distance_method,
            distance_alpha=cfg.distance_alpha,
            w_genotype=cfg.w_genotype,
            w_phenotype=cfg.w_phenotype,
            logger=log,
            genome_tracker=genome_tracker,
            events_tracker=events_tracker,
        )

    raise ValueError(f"Unknown clustering_method={method!r}; use 'leader_follower' or 'dbscan'")


__all__ = [
    "ARCHIVE_SPECIES_ID",
    "cluster",
    "dbscan_precomputed",
    "dbscan_cluster_population",
    "leader_follower_clustering",
]
