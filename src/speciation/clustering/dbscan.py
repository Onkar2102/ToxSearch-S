"""DBSCAN on a controller-built pairwise distance matrix."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..distance import normalize_distance_method, pairwise_distance_matrix
from ..species import Individual, Species, generate_species_id
from ._io import (
    default_paths,
    individuals_from_genomes,
    load_json_list,
    load_species_state,
    stack_objectives,
    stack_unit_embeddings,
    write_species_state,
)

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


def dbscan_precomputed(dist: np.ndarray, eps: float, min_samples: int = 2) -> np.ndarray:
    """Labels ``0..k-1`` for density clusters, ``-1`` for noise."""
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError(f"dist must be square, got {dist.shape}")
    n = dist.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64)
    if min_samples < 1 or eps < 0:
        raise ValueError(f"invalid eps={eps} min_samples={min_samples}")

    neighbors = [np.flatnonzero(dist[i] <= eps) for i in range(n)]
    is_core = np.array([len(nbs) >= min_samples for nbs in neighbors], dtype=bool)
    labels = np.full(n, -1, dtype=np.int64)
    cid = 0
    for i in range(n):
        if labels[i] != -1 or not is_core[i]:
            continue
        seed = list(neighbors[i])
        labels[i] = cid
        j = 0
        while j < len(seed):
            q = int(seed[j]); j += 1
            if labels[q] == -1:
                labels[q] = cid
            if is_core[q]:
                for r in neighbors[q]:
                    r = int(r)
                    if labels[r] == -1:
                        labels[r] = cid
                        seed.append(r)
        cid += 1
    return labels


def dbscan_cluster_population(
    temp_path: Optional[str] = None,
    elites_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
    eps: float = 0.25,
    min_samples: int = 2,
    current_generation: int = 0,
    include_elites: bool = True,
    distance_method: str = "embedding",
    distance_alpha: float = 0.7,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
    genome_tracker=None,
    events_tracker=None,
) -> Tuple[Dict[int, Species], Set[int]]:
    log = logger or get_logger("DBSCANClustering")
    method = normalize_distance_method(distance_method)
    temp_p, elites_p, state_p = default_paths(temp_path, elites_path, speciation_state_path)

    # Advance id generator from prior state (ignore returned species — DBSCAN rebuilds)
    load_species_state(state_p, eps, logger=log)

    seen: set = set()
    genomes: List[dict] = []
    for path in ((temp_p,) + ((elites_p,) if include_elites else ())):
        for g in load_json_list(path, log):
            key = str(g.get("id")) if g.get("id") is not None else None
            if key is None or key in seen:
                continue
            seen.add(key)
            genomes.append(g)

    individuals = individuals_from_genomes(genomes, log)
    if not individuals:
        log.warning("DBSCAN: no genomes with embeddings")
        return {}, set()

    emb = stack_unit_embeddings(individuals)
    objectives = stack_objectives(individuals) if method in ("objective", "embedding+objective") else None
    texts = [ind.prompt or "" for ind in individuals] if "nli" in method else None
    dist = pairwise_distance_matrix(
        method, embeddings=emb, objectives=objectives, texts=texts,
        alpha=distance_alpha, w_genotype=w_genotype, w_phenotype=w_phenotype, logger=log,
    )
    raw = dbscan_precomputed(dist, eps=eps, min_samples=min_samples)

    # Remap noise (-1) → singleton local labels after density clusters
    labels = raw.copy()
    next_lab = int(raw.max()) + 1 if raw.size and raw.max() >= 0 else 0
    for i, lab in enumerate(labels):
        if lab < 0:
            labels[i] = next_lab
            next_lab += 1

    groups: Dict[int, List[Individual]] = defaultdict(list)
    for ind, lab in zip(individuals, labels):
        groups[int(lab)].append(ind)

    species: Dict[int, Species] = {}
    touched: Set[int] = set()
    raw_by_id = {ind.id: int(raw[i]) for i, ind in enumerate(individuals)}

    for members in groups.values():
        members = sorted(members, key=lambda x: x.fitness, reverse=True)
        champ = members[0]
        sid = generate_species_id()
        for m in members:
            m.species_id = sid
        noise = raw_by_id.get(champ.id, 0) < 0 and len(members) == 1
        species[sid] = Species(
            id=sid, leader=champ, members=list(members), radius=eps,
            stagnation=0, max_fitness=champ.fitness, species_state="active",
            created_at=current_generation, last_improvement=current_generation,
            fitness_history=[champ.fitness], labels=[], label_history=[],
            cluster_origin="dbscan_noise" if noise else "dbscan",
            parent_ids=None, leader_distance=0.0,
        )
        touched.add(sid)
        if genome_tracker is not None:
            genome_tracker.batch_update(
                {str(m.id): sid for m in members}, current_generation, "dbscan_assignment"
            )
        if events_tracker is not None:
            for m in members:
                events_tracker.log(m.id, "dbscan_assignment", {"species_id": sid, "eps": eps})

    write_species_state(state_p, species, current_generation, extra={
        "clustering_method": "dbscan",
        "dbscan_eps": eps,
        "dbscan_min_samples": min_samples,
        "distance_method": method,
        "distance_alpha": distance_alpha,
    })
    if genome_tracker is not None:
        try:
            genome_tracker.save()
        except Exception as e:
            log.warning("DBSCAN: genome_tracker.save failed: %s", e)

    n_noise = int(np.sum(raw < 0))
    n_dense = int(raw.max() + 1) if raw.size and raw.max() >= 0 else 0
    log.info(
        "DBSCAN: n=%d eps=%.4f min_samples=%d distance_method=%s → density=%d noise=%d groups=%d",
        len(individuals), eps, min_samples, method, n_dense, n_noise, len(species),
    )
    return species, touched
