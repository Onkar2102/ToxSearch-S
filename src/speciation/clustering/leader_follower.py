"""Leader–follower clustering (fitness-ordered nearest-leader assignment)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from ..distance import distances_to_query, normalize_distance_method, pair_distance
from ..species import Individual, Species, generate_species_id
from ._io import (
    ARCHIVE_SPECIES_ID,
    default_paths,
    individuals_from_genomes,
    load_json_list,
    load_species_state,
    save_json,
    write_species_state,
)

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()

# sid, embedding, phenotype
LeaderRow = Tuple[int, np.ndarray, Optional[np.ndarray]]


def _dist_kw(alpha: float, w_g: float, w_p: float, logger) -> dict:
    return dict(alpha=alpha, w_genotype=w_g, w_phenotype=w_p, logger=logger)


def _nearest_leader(
    ind: Individual,
    leaders: List[LeaderRow],
    species: Dict[int, Species],
    method: str,
    **dkw,
) -> Tuple[Optional[int], float]:
    if not leaders:
        return None, float("inf")
    if len(leaders) == 1:
        sid, emb, ph = leaders[0]
        text = species[sid].leader.prompt if sid in species and species[sid].leader else ""
        d = pair_distance(
            method,
            embedding_a=ind.embedding, embedding_b=emb,
            objective_a=ind.phenotype, objective_b=ph,
            text_a=ind.prompt, text_b=text, **dkw,
        )
        return sid, d
    emb = np.array([e for _, e, _ in leaders])
    pheno = [p for _, _, p in leaders]
    texts = [
        (species[sid].leader.prompt if sid in species and species[sid].leader else "")
        for sid, _, _ in leaders
    ]
    dists = distances_to_query(
        method,
        query_embedding=ind.embedding, embeddings=emb,
        query_objective=ind.phenotype, objectives=pheno,
        query_text=ind.prompt, texts=texts, **dkw,
    )
    i = int(np.argmin(dists))
    return leaders[i][0], float(dists[i])


def _ensure_unique_leader(
    species: Dict[int, Species], new_leader: Individual, sid: int, logger
) -> None:
    for other_id, sp in species.items():
        if other_id == sid or not sp.leader or sp.leader.id != new_leader.id:
            continue
        if new_leader in sp.members:
            sp.members.remove(new_leader)
            new_leader.species_id = None
        if sp.members:
            sp.leader = max(sp.members, key=lambda x: x.fitness)
            if sp.leader not in sp.members:
                sp.members.insert(0, sp.leader)
            logger.info("Reassigned species %s leader → %s", other_id, sp.leader.id)
        else:
            sp.species_state = "extinct"
            sp.leader = None
            logger.info("Species %s emptied after leader reassignment — marked extinct (no incubator)", other_id)


def _maybe_promote_leader(
    sp: Species, ind: Individual, dist: float, sid: int,
    species: Dict[int, Species], leaders: List[LeaderRow], logger,
) -> None:
    if ind.fitness <= sp.leader.fitness:
        return
    if ind.fitness > sp.max_fitness:
        sp.max_fitness = ind.fitness
        sp.stagnation = 0
    _ensure_unique_leader(species, ind, sid, logger)
    sp.leader = ind
    sp.leader_distance = dist
    if sp.species_state == "frozen":
        sp.species_state = "active"
        logger.info("Frozen species %s reactivated by leader %s", sid, ind.id)
    for i, (lsid, _, _) in enumerate(leaders):
        if lsid == sid:
            leaders[i] = (sid, sp.leader.embedding, sp.leader.phenotype)
            break


def _form_singleton_species(
    ind: Individual,
    species: Dict[int, Species],
    leaders: List[LeaderRow],
    touched: Set[int],
    theta: float,
    generation: int,
    logger,
    events_tracker=None,
) -> int:
    """Create a new elite species with ``ind`` as sole leader/member."""
    sid = generate_species_id()
    _ensure_unique_leader(species, ind, sid, logger)
    ind.species_id = sid
    species[sid] = Species(
        id=sid,
        leader=ind,
        members=[ind],
        radius=theta,
        created_at=generation,
        last_improvement=generation,
        cluster_origin="natural",
        parent_ids=None,
        leader_distance=0.0,
    )
    leaders.append((sid, ind.embedding, ind.phenotype))
    touched.add(sid)
    if events_tracker:
        events_tracker.log(ind.id, "species_formed", {"species_id": sid, "size": 1})
    logger.info("Species %s formed: singleton leader %s", sid, ind.id)
    return sid


def leader_follower_clustering(
    temp_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
    theta_sim: float = 0.2,
    current_generation: int = 0,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    distance_method: str = "embedding",
    distance_alpha: float = 0.7,
    min_island_size: int = 2,
    logger=None,
    genome_tracker=None,
    events_tracker=None,
) -> Tuple[Dict[int, Species], Set[int]]:
    """Assign ``temp`` genomes to nearest leaders; unmatched genomes seed new species.

    ``min_island_size`` is accepted for API compatibility but unused: singleton
    elite species are allowed (no reserves / no minimum-size reject).
    """
    del min_island_size  # size-1 species are valid elites
    log = logger or get_logger("LeaderFollowerClustering")
    method = normalize_distance_method(distance_method)
    dkw = _dist_kw(distance_alpha, w_genotype, w_phenotype, log)
    log.info(
        "Leader-Follower distance_method=%s alpha=%.3f w_g=%.2f w_p=%.2f",
        method, distance_alpha, w_genotype, w_phenotype,
    )

    temp_p, _, state_p = default_paths(temp_path, speciation_state_path=speciation_state_path)
    genomes = load_json_list(temp_p, log)
    if not genomes:
        log.error("Temp file missing or empty: %s", temp_p)
        return {}, set()

    population = individuals_from_genomes(genomes, log)
    if not population:
        log.error("No individuals with embeddings")
        return {}, set()

    species = load_species_state(state_p, theta_sim, logger=log)
    leaders: List[LeaderRow] = [
        (sid, sp.leader.embedding, sp.leader.phenotype)
        for sid, sp in species.items()
        if sp.leader and sp.leader.embedding is not None
    ]
    touched: Set[int] = set()
    new_species = 0

    for ind in sorted(population, key=lambda x: x.fitness, reverse=True):
        sid, dist = _nearest_leader(ind, leaders, species, method, **dkw)
        if sid is not None and dist < theta_sim:
            sp = species[sid]
            sp.add_member(ind)
            ind.species_id = sid
            touched.add(sid)
            _maybe_promote_leader(sp, ind, dist, sid, species, leaders, log)
            continue

        # Farther than theta_sim from all leaders → seed a new singleton species
        # (replaces legacy reserves/archive path for unmatched temp genomes).
        _form_singleton_species(
            ind, species, leaders, touched, theta_sim, current_generation, log, events_tracker
        )
        new_species += 1

    # Persist species_id onto temp genomes + tracker (batch once)
    id_to_sid = {ind.id: ind.species_id for ind in population}
    updates = {}
    for g in genomes:
        gid = g.get("id")
        sid = id_to_sid.get(gid)
        g["species_id"] = sid
        if genome_tracker is not None:
            updates[str(gid)] = sid if sid is not None and sid > 0 else ARCHIVE_SPECIES_ID
    save_json(temp_p, genomes)
    write_species_state(state_p, species, current_generation)

    if genome_tracker is not None and updates:
        result = genome_tracker.batch_update(updates, current_generation, "leader_follower_clustering")
        if result.get("failed", 0) > 0:
            log.warning("Genome tracker batch update had %s failures", result["failed"])

    log.info(
        "Leader-Follower: %d individuals → %d species (%d new singletons this call)",
        len(population), len(species), new_species,
    )
    return species, touched
