"""Leader–follower clustering (fitness-ordered nearest-leader assignment)."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from ..distance import distances_to_query, normalize_distance_method, pair_distance
from ..species import Individual, Species, generate_species_id
from ._io import (
    ARCHIVE_SPECIES_ID,
    SINGLETON_SPECIES_ID,
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
# Provisional singleton leaders share species_id=0; keyed by list index during clustering.
ProvKey = Tuple[str, int]  # ("prov", index)


def _dist_kw(alpha: float, w_g: float, w_p: float, logger) -> dict:
    return dict(alpha=alpha, w_genotype=w_g, w_phenotype=w_p, logger=logger)


def _nearest_target(
    ind: Individual,
    leaders: List[LeaderRow],
    provisional: List[Individual],
    species: Dict[int, Species],
    method: str,
    **dkw,
) -> Tuple[Optional[Union[int, ProvKey]], float]:
    """Nearest mature leader (sid>0) or provisional singleton; return key and distance."""
    keys: List[Union[int, ProvKey]] = []
    emb_list: List[np.ndarray] = []
    pheno: List[Optional[np.ndarray]] = []
    texts: List[str] = []

    for sid, emb, ph in leaders:
        keys.append(sid)
        emb_list.append(emb)
        pheno.append(ph)
        texts.append(
            species[sid].leader.prompt if sid in species and species[sid].leader else ""
        )
    for i, prov in enumerate(provisional):
        if prov.embedding is None:
            continue
        keys.append(("prov", i))
        emb_list.append(prov.embedding)
        pheno.append(prov.phenotype)
        texts.append(prov.prompt or "")

    if not keys:
        return None, float("inf")

    if len(keys) == 1:
        d = pair_distance(
            method,
            embedding_a=ind.embedding, embedding_b=emb_list[0],
            objective_a=ind.phenotype, objective_b=pheno[0],
            text_a=ind.prompt, text_b=texts[0], **dkw,
        )
        return keys[0], d

    emb = np.array(emb_list)
    dists = distances_to_query(
        method,
        query_embedding=ind.embedding, embeddings=emb,
        query_objective=ind.phenotype, objectives=pheno,
        query_text=ind.prompt, texts=texts, **dkw,
    )
    i = int(np.argmin(dists))
    return keys[i], float(dists[i])


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


def _mature_from_provisional(
    prov: Individual,
    joiner: Individual,
    species: Dict[int, Species],
    leaders: List[LeaderRow],
    touched: Set[int],
    theta: float,
    generation: int,
    logger,
    events_tracker=None,
) -> int:
    """Two genomes near each other → mint a mature species_id (>= 1)."""
    sid = generate_species_id()
    members = sorted([prov, joiner], key=lambda x: x.fitness, reverse=True)
    champ = members[0]
    _ensure_unique_leader(species, champ, sid, logger)
    for m in members:
        m.species_id = sid
    species[sid] = Species(
        id=sid,
        leader=champ,
        members=list(members),
        radius=theta,
        created_at=generation,
        last_improvement=generation,
        cluster_origin="natural",
        parent_ids=None,
        leader_distance=0.0,
        max_fitness=champ.fitness,
    )
    leaders.append((sid, champ.embedding, champ.phenotype))
    touched.add(sid)
    if events_tracker:
        events_tracker.log(
            joiner.id, "species_matured",
            {"species_id": sid, "size": 2, "from_singleton_pool": True},
        )
        events_tracker.log(
            prov.id, "species_matured",
            {"species_id": sid, "size": 2, "from_singleton_pool": True},
        )
    logger.info(
        "Mature species %s formed from singleton pool: leaders %s + %s",
        sid, prov.id, joiner.id,
    )
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
    logger=None,
    genome_tracker=None,
    events_tracker=None,
) -> Tuple[Dict[int, Species], Set[int]]:
    """Assign ``temp`` genomes to nearest leaders; unmatched stay in singleton pool (id=0).

    Mature multi-member niches keep unique ``species_id >= 1``. Singleton / unmatched
    genomes share ``species_id=0`` so the ID counter does not grow per singleton.
    Existing elite singletons (id=0) act as provisional leaders and can mature when
    a new genome joins within ``theta_sim``.
    """
    log = logger or get_logger("LeaderFollowerClustering")
    method = normalize_distance_method(distance_method)
    dkw = _dist_kw(distance_alpha, w_genotype, w_phenotype, log)
    log.info(
        "Leader-Follower distance_method=%s alpha=%.3f w_g=%.2f w_p=%.2f",
        method, distance_alpha, w_genotype, w_phenotype,
    )

    temp_p, elites_p, state_p = default_paths(temp_path, speciation_state_path=speciation_state_path)
    genomes = load_json_list(temp_p, log)
    if not genomes:
        log.error("Temp file missing or empty: %s", temp_p)
        return {}, set()

    population = individuals_from_genomes(genomes, log)
    if not population:
        log.error("No individuals with embeddings")
        return {}, set()

    species = load_species_state(state_p, theta_sim, logger=log)
    # Drop any legacy size-1 entries from state; singletons live in pool 0 only.
    for sid in list(species.keys()):
        if sid <= 0:
            species.pop(sid, None)

    leaders: List[LeaderRow] = [
        (sid, sp.leader.embedding, sp.leader.phenotype)
        for sid, sp in species.items()
        if sid > 0 and sp.leader and sp.leader.embedding is not None
    ]

    # Prior elite singletons (species_id=0) are provisional niche seeds.
    provisional: List[Individual] = []
    elites = load_json_list(elites_p, log)
    temp_ids = {ind.id for ind in population}
    for g in elites:
        if int(g.get("species_id", ARCHIVE_SPECIES_ID) or ARCHIVE_SPECIES_ID) != SINGLETON_SPECIES_ID:
            continue
        if g.get("id") in temp_ids:
            continue
        try:
            ind = Individual.from_genome(g)
        except Exception as e:
            log.debug("Skip elite singleton id=%s: %s", g.get("id"), e)
            continue
        if ind.embedding is None:
            continue
        ind.species_id = SINGLETON_SPECIES_ID
        provisional.append(ind)

    touched: Set[int] = set()
    new_singletons = 0
    matured = 0
    # Track elite genomes whose species_id changed (singleton → mature).
    elite_sid_updates: Dict[str, int] = {}

    for ind in sorted(population, key=lambda x: x.fitness, reverse=True):
        key, dist = _nearest_target(ind, leaders, provisional, species, method, **dkw)
        if key is not None and dist < theta_sim:
            if isinstance(key, tuple) and key[0] == "prov":
                _, pidx = key
                prov = provisional.pop(pidx)
                sid = _mature_from_provisional(
                    prov, ind, species, leaders, touched, theta_sim,
                    current_generation, log, events_tracker,
                )
                matured += 1
                if str(prov.id) not in {str(x.id) for x in population}:
                    elite_sid_updates[str(prov.id)] = sid
                continue

            sid = int(key)
            sp = species[sid]
            sp.add_member(ind)
            ind.species_id = sid
            touched.add(sid)
            _maybe_promote_leader(sp, ind, dist, sid, species, leaders, log)
            continue

        # Farther than theta_sim from all mature + provisional leaders → singleton pool.
        ind.species_id = SINGLETON_SPECIES_ID
        provisional.append(ind)
        new_singletons += 1
        if events_tracker:
            events_tracker.log(
                ind.id, "singleton_pool",
                {"species_id": SINGLETON_SPECIES_ID, "generation": current_generation},
            )

    # Persist species_id onto temp genomes + tracker (batch once)
    id_to_sid = {ind.id: ind.species_id for ind in population}
    updates = dict(elite_sid_updates)
    for g in genomes:
        gid = g.get("id")
        sid = id_to_sid.get(gid)
        if sid is None:
            g["species_id"] = ARCHIVE_SPECIES_ID
            if genome_tracker is not None:
                updates[str(gid)] = ARCHIVE_SPECIES_ID
            continue
        g["species_id"] = sid
        if genome_tracker is not None:
            updates[str(gid)] = int(sid)

    # Update elites.json for former singletons that matured this call.
    if elite_sid_updates and elites:
        for g in elites:
            gid = g.get("id")
            if gid is None:
                continue
            new_sid = elite_sid_updates.get(str(gid))
            if new_sid is not None:
                g["species_id"] = new_sid
        save_json(elites_p, elites)

    save_json(temp_p, genomes)
    # Only persist mature species (sid > 0) in speciation_state.
    mature_only = {sid: sp for sid, sp in species.items() if sid > 0}
    write_species_state(state_p, mature_only, current_generation)

    if genome_tracker is not None and updates:
        result = genome_tracker.batch_update(updates, current_generation, "leader_follower_clustering")
        if result.get("failed", 0) > 0:
            log.warning("Genome tracker batch update had %s failures", result["failed"])

    log.info(
        "Leader-Follower: %d individuals → %d mature species "
        "(%d new singletons→pool0, %d matured from pool)",
        len(population), len(mature_only), new_singletons, matured,
    )
    return mature_only, touched
