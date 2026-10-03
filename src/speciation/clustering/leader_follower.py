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
# species_id|None, emb, pheno, seed_ind, followers
Potential = Tuple[Optional[int], np.ndarray, Optional[np.ndarray], Individual, List[Individual]]


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
            sp.species_state = "incubator"
            sp.leader = None


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


def _members_within(
    candidates: List[Individual], leader: Individual, theta: float, method: str, **dkw
) -> List[Individual]:
    keep = [leader]
    for m in candidates:
        if m.id == leader.id or m.embedding is None:
            continue
        d = pair_distance(
            method,
            embedding_a=m.embedding, embedding_b=leader.embedding,
            objective_a=m.phenotype, objective_b=leader.phenotype,
            text_a=m.prompt, text_b=leader.prompt, **dkw,
        )
        if d < theta:
            keep.append(m)
    return keep


def _try_form_species(
    pl_id: int,
    pot: Dict[int, Potential],
    species: Dict[int, Species],
    leaders: List[LeaderRow],
    touched: Set[int],
    theta: float,
    min_size: int,
    generation: int,
    method: str,
    dkw: dict,
    logger,
    genome_tracker=None,
    events_tracker=None,
) -> bool:
    """Form a species from a potential-leader bucket if large enough. Returns True if joined/formed."""
    pl_sid, pl_emb, pl_pheno, pl_ind, followers = pot[pl_id]
    if pl_sid is not None:
        return False  # caller handles join-to-existing

    all_members = [pl_ind] + followers
    if len(all_members) < min_size:
        return False
    leader = max(all_members, key=lambda x: x.fitness)
    valid = _members_within(all_members, leader, theta, method, **dkw)
    if len(valid) < min_size:
        # drop followers that fell outside the new leader's radius
        valid_ids = {m.id for m in valid}
        pot[pl_id] = (None, pl_emb, pl_pheno, pl_ind, [f for f in followers if f.id in valid_ids])
        return False

    sid = generate_species_id()
    _ensure_unique_leader(species, leader, sid, logger)
    species[sid] = Species(
        id=sid, leader=leader, members=valid, radius=theta,
        created_at=generation, last_improvement=generation,
        cluster_origin="natural", parent_ids=None, leader_distance=0.0,
    )
    for m in valid:
        m.species_id = sid
        if genome_tracker:
            genome_tracker.update_species_id(str(m.id), sid, generation, "species_formed")
        if events_tracker:
            events_tracker.log(m.id, "species_formed", {"species_id": sid})
    touched.add(sid)
    pot[pl_id] = (sid, leader.embedding, leader.phenotype, leader, followers)
    leaders.append((sid, leader.embedding, leader.phenotype))
    logger.info(
        "Species %s formed: leader %s + %d members (min=%d)",
        sid, leader.id, len(valid) - 1, min_size,
    )
    return True


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
    is_gen0 = len(species) == 0
    leaders: List[LeaderRow] = [
        (sid, sp.leader.embedding, sp.leader.phenotype)
        for sid, sp in species.items()
        if sp.leader and sp.leader.embedding is not None
    ]
    potentials: Dict[int, Potential] = {}
    touched: Set[int] = set()

    for ind in sorted(population, key=lambda x: x.fitness, reverse=True):
        assigned = False
        sid, dist = _nearest_leader(ind, leaders, species, method, **dkw)
        if sid is not None and dist < theta_sim:
            sp = species[sid]
            sp.add_member(ind)
            ind.species_id = sid
            touched.add(sid)
            _maybe_promote_leader(sp, ind, dist, sid, species, leaders, log)
            assigned = True

        if not assigned and is_gen0 and potentials:
            for pl_id, (pl_sid, pl_emb, pl_pheno, pl_ind, followers) in list(potentials.items()):
                d = pair_distance(
                    method,
                    embedding_a=ind.embedding, embedding_b=pl_emb,
                    objective_a=ind.phenotype, objective_b=pl_pheno,
                    text_a=ind.prompt, text_b=pl_ind.prompt, **dkw,
                )
                if d >= theta_sim:
                    continue
                if pl_sid is None:
                    followers.append(ind)
                    ind.species_id = ARCHIVE_SPECIES_ID
                    _try_form_species(
                        pl_id, potentials, species, leaders, touched,
                        theta_sim, min_island_size, current_generation,
                        method, dkw, log, genome_tracker, events_tracker,
                    )
                else:
                    sp = species[pl_sid]
                    sp.add_member(ind)
                    ind.species_id = pl_sid
                    touched.add(pl_sid)
                    _maybe_promote_leader(sp, ind, d, pl_sid, species, leaders, log)
                assigned = True
                break

        if not assigned:
            ind.species_id = ARCHIVE_SPECIES_ID
            if is_gen0:
                potentials[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])

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

    log.info("Leader-Follower: %d individuals → %d species", len(population), len(species))
    return species, touched
