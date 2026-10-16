"""Grow-only Incremental DBSCAN for ToxSearch-S (CLI ``clustering_method=dbscan``).

Gen-0 and later generations use the same local insert path (absorb / new species
from noise / merge). No deletions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..density_memory import (
    DENSITY_LABEL_KEY,
    NOISE_LABEL,
    get_density_label,
    is_labeled,
    set_density_label,
)
from ..distance import (
    method_uses_embeddings,
    method_uses_objectives,
    method_uses_texts,
    distances_to_query,
    normalize_distance_method,
)
from ..species import Individual, Species, SpeciesIdGenerator, generate_species_id
from ._io import (
    SINGLETON_SPECIES_ID,
    individuals_from_genomes,
    write_species_state,
)

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()


@dataclass
class IncDBSCANState:
    """In-memory density structure over D_t (indexed parallel to genomes)."""

    eps: float
    min_samples: int
    distance_method: str = "embedding"
    distance_alpha: float = 0.7
    w_genotype: float = 0.7
    w_phenotype: float = 0.3
    # Parallel arrays / lists keyed by point index
    embeddings: Optional[np.ndarray] = None  # (n, d) unit
    objectives: Optional[np.ndarray] = None
    texts: List[str] = field(default_factory=list)
    genome_ids: List[Any] = field(default_factory=list)
    labels: List[int] = field(default_factory=list)  # persistent species id or NOISE
    is_core: List[bool] = field(default_factory=list)
    neighbors: List[Set[int]] = field(default_factory=list)

    def n_points(self) -> int:
        return len(self.genome_ids)

    def to_serializable(self) -> dict:
        try:
            from ..distance import get_nli_model_name
            nli_model = get_nli_model_name()
        except Exception:
            nli_model = None
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "distance_method": self.distance_method,
            "distance_alpha": self.distance_alpha,
            "nli_model": nli_model,
            "w_genotype": self.w_genotype,
            "w_phenotype": self.w_phenotype,
            "genome_ids": list(self.genome_ids),
            "labels": list(self.labels),
            "is_core": list(self.is_core),
            "neighbors": [sorted(s) for s in self.neighbors],
            "embeddings": self.embeddings.tolist() if self.embeddings is not None else None,
            "objectives": self.objectives.tolist() if self.objectives is not None else None,
            "texts": list(self.texts),
        }

    @classmethod
    def from_serializable(cls, data: dict) -> "IncDBSCANState":
        emb = data.get("embeddings")
        obj = data.get("objectives")
        st = cls(
            eps=float(data["eps"]),
            min_samples=int(data["min_samples"]),
            distance_method=normalize_distance_method(data.get("distance_method", "embedding")),
            distance_alpha=float(data.get("distance_alpha", 0.7)),
            w_genotype=float(data.get("w_genotype", 0.7)),
            w_phenotype=float(data.get("w_phenotype", 0.3)),
            genome_ids=list(data.get("genome_ids", [])),
            labels=[int(x) for x in data.get("labels", [])],
            is_core=[bool(x) for x in data.get("is_core", [])],
            neighbors=[set(n) for n in data.get("neighbors", [])],
            texts=list(data.get("texts", [])),
            embeddings=np.asarray(emb, dtype=np.float64) if emb is not None else None,
            objectives=np.asarray(obj, dtype=np.float64) if obj is not None else None,
        )
        return st


def _dkw(state: IncDBSCANState, logger) -> dict:
    return dict(
        alpha=state.distance_alpha,
        w_genotype=state.w_genotype,
        w_phenotype=state.w_phenotype,
        logger=logger,
    )


def seed_from_incremental(
    genomes: List[Dict[str, Any]],
    *,
    eps: float,
    min_samples: int,
    current_generation: int = 0,
    distance_method: str = "embedding",
    distance_alpha: float = 0.7,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3,
    logger=None,
    events_tracker=None,
) -> Tuple[IncDBSCANState, Dict[int, Species], Dict[str, Any]]:
    """Bootstrap empty density memory via sequential inserts (fitness-descending).

    Same insert logic as later generations; clears any prior density labels so
    Gen-0 (or a missing state file) always rebuilds from scratch.
    """
    log = logger or get_logger("IncDBSCAN")
    method = normalize_distance_method(distance_method)
    SpeciesIdGenerator.reset(0)
    for g in genomes:
        g.pop(DENSITY_LABEL_KEY, None)

    state = IncDBSCANState(
        eps=eps,
        min_samples=min_samples,
        distance_method=method,
        distance_alpha=distance_alpha,
        w_genotype=w_genotype,
        w_phenotype=w_phenotype,
    )
    species: Dict[int, Species] = {}
    if not genomes:
        return state, species, {
            "new_species": 0,
            "noise": 0,
            "singletons": 0,
            "n": 0,
            "event": "inc_dbscan_seed",
        }

    insert_evs = process_unlabeled(
        state,
        genomes,
        current_generation=current_generation,
        species=species,
        logger=log,
        events_tracker=events_tracker,
    )
    n_noise_raw = int(sum(1 for lab in state.labels if int(lab) == NOISE_LABEL))
    n_singletons = promote_noise_to_singleton_species(
        state, genomes, species, current_generation=current_generation, logger=log,
        events_tracker=events_tracker,
    )
    n_new = sum(1 for ev in insert_evs if ev.get("event") == "inc_dbscan_new_species")
    n_merge = sum(1 for ev in insert_evs if ev.get("event") == "inc_dbscan_merge")
    events = {
        "new_species": len(species),
        "insert_new_species_events": n_new,
        "merge_events": n_merge,
        "noise": 0,
        "noise_before_singleton_promotion": n_noise_raw,
        "singletons": n_singletons,
        "n": state.n_points(),
        "event": "inc_dbscan_seed",
    }
    log.info(
        "IncDBSCAN incremental seed: n=%d eps=%.4f min_samples=%d → "
        "mature_species=%d (noise→pool0: %d)",
        events["n"], eps, min_samples, events["new_species"], n_singletons,
    )
    return state, species, events


def _build_species_from_state(
    state: IncDBSCANState,
    genomes: Sequence[Dict[str, Any]],
    individuals: Sequence[Individual],
    current_generation: int,
    eps: float,
    origin: str,
    existing: Optional[Dict[int, Species]] = None,
) -> Dict[int, Species]:
    by_id = {ind.id: ind for ind in individuals}
    groups: Dict[int, List[Individual]] = defaultdict(list)
    for gid, lab in zip(state.genome_ids, state.labels):
        # Mature species only (sid > 0). Noise (-1) and singleton pool (0) are not Species entries.
        if lab is None or int(lab) <= 0:
            ind = by_id.get(gid)
            if ind is not None and int(lab) == SINGLETON_SPECIES_ID:
                ind.species_id = SINGLETON_SPECIES_ID
            continue
        ind = by_id.get(gid)
        if ind is None:
            continue
        ind.species_id = lab
        groups[lab].append(ind)

    species: Dict[int, Species] = dict(existing or {})
    # Drop species no longer present in the density labels
    alive = set(groups)
    for sid in list(species.keys()):
        if sid not in alive:
            species.pop(sid, None)

    for sid, members in groups.items():
        members = sorted(members, key=lambda x: x.fitness, reverse=True)
        champ = members[0]
        prev = species.get(sid)
        max_hist = max((prev.max_size_historical if prev and hasattr(prev, "max_size_historical") else 0), len(members))
        if prev is None:
            sp = Species(
                id=sid, leader=champ, members=list(members), radius=eps,
                stagnation=0, max_fitness=champ.fitness, species_state="active",
                created_at=current_generation, last_improvement=current_generation,
                fitness_history=[champ.fitness], labels=[], label_history=[],
                cluster_origin=origin, parent_ids=None, leader_distance=0.0,
            )
        else:
            sp = prev
            sp.leader = champ
            sp.members = list(members)
            sp.max_fitness = max(sp.max_fitness, champ.fitness)
            if champ.fitness >= sp.max_fitness:
                sp.last_improvement = current_generation
                sp.stagnation = 0
            sp.cluster_origin = prev.cluster_origin or origin
        # Optional metadata attributes (not on dataclass by default)
        setattr(sp, "max_size_historical", max_hist)
        setattr(sp, "density_size", len(members))
        species[sid] = sp
    return species


def insert_point(
    state: IncDBSCANState,
    genome: Dict[str, Any],
    individual: Individual,
    *,
    current_generation: int,
    species: Dict[int, Species],
    logger=None,
    events_tracker=None,
) -> Dict[str, Any]:
    """Insert one labeled-or-new genome into the density structure (grow-only)."""
    log = logger or get_logger("IncDBSCAN")
    method = state.distance_method
    dkw = _dkw(state, log)
    need_emb = method_uses_embeddings(method)
    need_txt = method_uses_texts(method)

    emb_row = None
    if individual.embedding is not None:
        emb_row = np.asarray(individual.embedding, dtype=np.float64)
        nrm = float(np.linalg.norm(emb_row))
        if nrm > 0:
            emb_row = emb_row / nrm
    if need_emb and emb_row is None:
        raise ValueError(
            f"IncDBSCAN insert requires embedding for distance_method={method!r} "
            f"(genome id={individual.id})"
        )
    if need_txt and not (individual.prompt or "").strip():
        raise ValueError(
            f"IncDBSCAN insert requires prompt text for distance_method={method!r} "
            f"(genome id={individual.id})"
        )

    if state.n_points() == 0:
        # First point after empty state
        state.embeddings = emb_row.reshape(1, -1) if emb_row is not None else None
        if individual.phenotype is not None:
            state.objectives = np.asarray(individual.phenotype, dtype=np.float64).reshape(1, -1)
        state.texts = [individual.prompt or ""]
        state.genome_ids = [individual.id]
        state.labels = [NOISE_LABEL]
        state.neighbors = [{0}]
        state.is_core = [state.min_samples <= 1]
        if state.is_core[0]:
            # Size-1 cannot be mature; park in singleton pool until a second member joins.
            state.labels[0] = SINGLETON_SPECIES_ID
            set_density_label(genome, SINGLETON_SPECIES_ID)
            individual.species_id = SINGLETON_SPECIES_ID
            ev = {"event": "inc_dbscan_singleton_pool", "species_id": SINGLETON_SPECIES_ID}
        else:
            set_density_label(genome, NOISE_LABEL)
            ev = {"event": "inc_dbscan_insert_noise"}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    # Distances from new point to existing
    dists = distances_to_query(
        method,
        query_embedding=emb_row if (need_emb or state.embeddings is not None) else None,
        embeddings=state.embeddings if (need_emb or state.embeddings is not None) else None,
        query_objective=individual.phenotype,
        objectives=state.objectives,
        query_text=individual.prompt or "",
        texts=state.texts if need_txt else None,
        **dkw,
    )
    n_old = state.n_points()
    nbrs = {int(j) for j in np.flatnonzero(dists <= state.eps)}
    new_idx = n_old

    # Append point
    if state.embeddings is not None:
        if emb_row is None:
            emb_row = np.zeros((state.embeddings.shape[1],), dtype=np.float64)
        state.embeddings = np.vstack([state.embeddings, emb_row.reshape(1, -1)])
    elif emb_row is not None:
        # Resume path where prior state had no embeddings
        state.embeddings = emb_row.reshape(1, -1)
    if state.objectives is not None:
        ph = individual.phenotype
        if ph is None:
            row = np.full((1, state.objectives.shape[1]), np.nan)
        else:
            ph = np.asarray(ph, dtype=np.float64).ravel()
            row = np.full((1, state.objectives.shape[1]), np.nan)
            row[0, : min(len(ph), state.objectives.shape[1])] = ph[: state.objectives.shape[1]]
        state.objectives = np.vstack([state.objectives, row])
    state.texts.append(individual.prompt or "")
    state.genome_ids.append(individual.id)
    state.labels.append(NOISE_LABEL)
    state.neighbors.append(set(nbrs) | {new_idx})
    state.is_core.append(False)

    # Symmetric neighbor updates
    for j in nbrs:
        state.neighbors[j].add(new_idx)

    # Recompute core flags for affected points
    affected = set(nbrs) | {new_idx}
    for i in affected:
        state.is_core[i] = len(state.neighbors[i]) >= state.min_samples

    # Clusters of mature core neighbors (sid > 0 only; pool 0 / noise are not niches)
    core_cluster_ids: Set[int] = set()
    for j in nbrs:
        lab_j = int(state.labels[j])
        if state.is_core[j] and lab_j > 0:
            core_cluster_ids.add(lab_j)

    new_is_core = state.is_core[new_idx]

    if not new_is_core and not core_cluster_ids:
        # Pure noise
        set_density_label(genome, NOISE_LABEL)
        state.labels[new_idx] = NOISE_LABEL
        ev = {"event": "inc_dbscan_insert_noise"}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    if not new_is_core and core_cluster_ids:
        # Border: join nearest core's cluster (lowest id for stability)
        sid = min(core_cluster_ids)
        state.labels[new_idx] = sid
        set_density_label(genome, sid)
        _absorb_member(species, sid, individual, current_generation, state.eps)
        ev = {"event": "inc_dbscan_absorb", "species_id": sid}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    # new point is core
    if len(core_cluster_ids) == 0:
        # Expand through noise / singleton-pool neighbors; mature only if size >= 2.
        reachable = _collect_reachable(state, new_idx)
        if len(reachable) < 2:
            for i in reachable:
                state.labels[i] = SINGLETON_SPECIES_ID
            set_density_label(genome, SINGLETON_SPECIES_ID)
            individual.species_id = SINGLETON_SPECIES_ID
            ev = {
                "event": "inc_dbscan_singleton_pool",
                "species_id": SINGLETON_SPECIES_ID,
                "assigned": len(reachable),
            }
            if events_tracker:
                events_tracker.log(individual.id, ev["event"], ev)
            return ev
        sid = generate_species_id()
        assigned = _expand_cluster(state, new_idx, sid)
        set_density_label(genome, sid)
        members = [individual]
        species[sid] = Species(
            id=sid, leader=individual, members=members, radius=state.eps,
            created_at=current_generation, last_improvement=current_generation,
            cluster_origin="inc_dbscan_from_noise", max_fitness=individual.fitness,
        )
        setattr(species[sid], "pending_member_indices", assigned)
        ev = {"event": "inc_dbscan_new_species", "species_id": sid, "assigned": len(assigned)}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    if len(core_cluster_ids) == 1:
        sid = next(iter(core_cluster_ids))
        assigned = _expand_cluster(state, new_idx, sid)
        set_density_label(genome, sid)
        _absorb_member(species, sid, individual, current_generation, state.eps)
        setattr(species[sid], "pending_member_indices", assigned)
        ev = {"event": "inc_dbscan_absorb", "species_id": sid, "assigned": len(assigned)}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    # Survivor = oldest parent (lowest ID). Higher parents are absorbed.
    # Advance high-water past every parent ID so absorbed IDs are not re-issued.
    parents = sorted(core_cluster_ids)
    survivor = parents[0]
    for pid in parents:
        SpeciesIdGenerator.note_id(pid)
    assigned = _expand_cluster(state, new_idx, survivor)
    # Relabel all points of parent clusters onto the survivor
    for i, lab in enumerate(state.labels):
        if lab in core_cluster_ids:
            state.labels[i] = survivor
    state.labels[new_idx] = survivor
    set_density_label(genome, survivor)

    merged_members: List[Individual] = [individual]
    parent_ids = list(parents)
    created_at = current_generation
    for pid in parents:
        old = species.pop(pid, None)
        if old:
            if pid == survivor:
                created_at = old.created_at
            for m in old.members:
                m.species_id = survivor
                if m.id != individual.id:
                    merged_members.append(m)
    merged_members = sorted(merged_members, key=lambda x: x.fitness, reverse=True)
    champ = merged_members[0]
    species[survivor] = Species(
        id=survivor, leader=champ, members=merged_members, radius=state.eps,
        created_at=created_at, last_improvement=current_generation,
        cluster_origin="inc_dbscan_merge", parent_ids=parent_ids,
        max_fitness=champ.fitness,
    )
    setattr(species[survivor], "pending_member_indices", assigned)
    ev = {
        "event": "inc_dbscan_merge",
        "species_id": survivor,
        "parent_ids": parent_ids,
        "absorbed_ids": parents[1:],
        "assigned": len(assigned),
    }
    if events_tracker:
        events_tracker.log(individual.id, ev["event"], ev)
    log.info("IncDBSCAN merge: %s → survivor %s (absorbed %s)", parent_ids, survivor, parents[1:])
    return ev


def _is_absorbable_label(lab: int, sid: Optional[int] = None) -> bool:
    """Noise / singleton pool (lab <= 0), or already in mature ``sid``, can be walked."""
    if lab <= 0:
        return True
    return sid is not None and lab == sid


def _collect_reachable(state: IncDBSCANState, seed: int) -> Set[int]:
    """Density-reachable indices from ``seed`` through noise/singleton-pool."""
    q = deque([seed])
    seen = {seed}
    assigned: Set[int] = set()
    while q:
        i = q.popleft()
        assigned.add(i)
        if not state.is_core[i]:
            continue
        for j in state.neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if _is_absorbable_label(int(state.labels[j])):
                q.append(j)
    return assigned


def _expand_cluster(state: IncDBSCANState, seed: int, sid: int) -> Set[int]:
    """Assign density-reachable points from seed core to ``sid`` (noise + singleton pool)."""
    q = deque([seed])
    seen = {seed}
    assigned: Set[int] = set()
    while q:
        i = q.popleft()
        state.labels[i] = sid
        assigned.add(i)
        if not state.is_core[i]:
            continue
        for j in state.neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            if _is_absorbable_label(int(state.labels[j]), sid):
                q.append(j)
    return assigned


def _absorb_member(
    species: Dict[int, Species],
    sid: int,
    individual: Individual,
    generation: int,
    eps: float,
) -> None:
    individual.species_id = sid
    if sid not in species:
        species[sid] = Species(
            id=sid, leader=individual, members=[individual], radius=eps,
            created_at=generation, last_improvement=generation,
            cluster_origin="inc_dbscan_growth", max_fitness=individual.fitness,
        )
        return
    sp = species[sid]
    sp.add_member(individual)
    if individual.fitness > sp.leader.fitness:
        sp.leader = individual
        sp.leader_distance = 0.0
    if individual.fitness > sp.max_fitness:
        sp.max_fitness = individual.fitness
        sp.last_improvement = generation
        sp.stagnation = 0


def sync_genomes_from_state(genomes: List[Dict[str, Any]], state: IncDBSCANState) -> None:
    """Write density labels from state onto genome dicts (by id)."""
    id_to_label = {gid: lab for gid, lab in zip(state.genome_ids, state.labels)}
    for g in genomes:
        gid = g.get("id")
        if gid in id_to_label:
            set_density_label(g, id_to_label[gid])


def promote_noise_to_singleton_species(
    state: IncDBSCANState,
    genomes: List[Dict[str, Any]],
    species: Dict[int, Species],
    current_generation: int,
    logger=None,
    events_tracker=None,
) -> int:
    """Promote DBSCAN noise (-1) to the shared singleton pool (``species_id=0``).

    All singleton/noise genomes share id 0 so the mature-species counter (1…)
    does not grow per outlier. No per-genome ``Species`` entries are created.
    Also demotes any accidental size-1 mature labels to the pool.
    """
    log = logger or get_logger("IncDBSCAN")
    id_to_idx = {gid: i for i, gid in enumerate(state.genome_ids)}
    promoted = 0
    for g in genomes:
        lab = get_density_label(g)
        if lab is None or int(lab) != NOISE_LABEL:
            continue
        gid = g.get("id")
        set_density_label(g, SINGLETON_SPECIES_ID)
        if gid in id_to_idx:
            state.labels[id_to_idx[gid]] = SINGLETON_SPECIES_ID
        promoted += 1
        if events_tracker:
            events_tracker.log(
                gid, "inc_dbscan_singleton_pool",
                {"species_id": SINGLETON_SPECIES_ID, "generation": current_generation},
            )
    # Demote size-1 mature labels → pool 0 (keeps ID counter honest for future minting).
    demoted = _demote_size1_labels_to_pool(state, genomes, species, log)
    if promoted or demoted:
        log.info(
            "IncDBSCAN: %d noise → singleton pool 0; demoted %d size-1 mature labels",
            promoted, demoted,
        )
    return promoted + demoted


def _demote_size1_labels_to_pool(
    state: IncDBSCANState,
    genomes: List[Dict[str, Any]],
    species: Dict[int, Species],
    logger=None,
) -> int:
    """Relabel any positive species_id with only one member to singleton pool 0."""
    log = logger or get_logger("IncDBSCAN")
    counts: Dict[int, int] = defaultdict(int)
    for lab in state.labels:
        if int(lab) > 0:
            counts[int(lab)] += 1
    size1 = {sid for sid, n in counts.items() if n == 1}
    if not size1:
        return 0
    demoted = 0
    id_to_g = {g.get("id"): g for g in genomes if g.get("id") is not None}
    for i, lab in enumerate(state.labels):
        if int(lab) not in size1:
            continue
        state.labels[i] = SINGLETON_SPECIES_ID
        gid = state.genome_ids[i]
        g = id_to_g.get(gid)
        if g is not None:
            set_density_label(g, SINGLETON_SPECIES_ID)
        demoted += 1
    for sid in size1:
        species.pop(sid, None)
    if demoted:
        log.debug("Demoted %d genomes from size-1 mature labels → pool 0", demoted)
    return demoted


def process_unlabeled(
    state: IncDBSCANState,
    genomes: List[Dict[str, Any]],
    *,
    current_generation: int,
    species: Dict[int, Species],
    logger=None,
    events_tracker=None,
) -> List[Dict[str, Any]]:
    """Insert every unlabeled genome with features required by ``state.distance_method``."""
    log = logger or get_logger("IncDBSCAN")
    events: List[Dict[str, Any]] = []
    method = state.distance_method
    need_emb = method_uses_embeddings(method)
    need_txt = method_uses_texts(method)
    need_obj = method_uses_objectives(method)
    # Fitness-descending for reproducibility
    idxs = [i for i, g in enumerate(genomes) if not is_labeled(g)]
    idxs.sort(
        key=lambda i: float(
            genomes[i].get("fitness")
            or genomes[i].get("toxicity")
            or 0.0
        ),
        reverse=True,
    )
    skipped = 0
    for i in idxs:
        g = genomes[i]
        skip_reason = None
        try:
            ind = Individual.from_genome(g)
        except Exception as e:
            skip_reason = str(e)
            ind = None
        if ind is not None and need_emb and ind.embedding is None:
            skip_reason = f"no embedding (required by {method})"
        elif ind is not None and need_txt and not (ind.prompt or "").strip():
            skip_reason = f"empty prompt (required by {method})"
        elif ind is not None and need_obj and ind.phenotype is None:
            skip_reason = f"no objective vector (required by {method})"
        if skip_reason is not None:
            skipped += 1
            log.warning("IncDBSCAN skip genome id=%s: %s", g.get("id"), skip_reason)
            events.append({
                "event": "inc_dbscan_insert_skipped",
                "genome_id": g.get("id"),
                "reason": skip_reason,
                "generation": current_generation,
            })
            continue
        ev = insert_point(
            state, g, ind,
            current_generation=current_generation,
            species=species,
            logger=log,
            events_tracker=events_tracker,
        )
        events.append(ev)
    if skipped:
        log.warning(
            "IncDBSCAN process_unlabeled: skipped %d/%d unlabeled genomes",
            skipped, len(idxs),
        )
    sync_genomes_from_state(genomes, state)
    # Rebuild species members from full state
    individuals = individuals_from_genomes(genomes, log, distance_method=method)
    rebuilt = _build_species_from_state(
        state, genomes, individuals, current_generation, state.eps,
        origin="inc_dbscan_growth", existing=species,
    )
    species.clear()
    species.update(rebuilt)
    return events


def save_inc_state(path, state: IncDBSCANState) -> None:
    from pathlib import Path
    from ._io import save_json
    save_json(Path(path), state.to_serializable())


def load_inc_state(path, logger=None) -> Optional[IncDBSCANState]:
    from pathlib import Path
    import json
    p = Path(path)
    if not p.exists():
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return IncDBSCANState.from_serializable(data)
    except Exception as e:
        log = logger or get_logger("IncDBSCAN")
        log.warning("Failed to load inc_dbscan_state: %s", e)
        return None


def write_species_and_config(
    speciation_state_path,
    species: Dict[int, Species],
    current_generation: int,
    state: IncDBSCANState,
    extra: Optional[dict] = None,
) -> None:
    from pathlib import Path
    try:
        from ..distance import get_nli_model_name
        nli_model = get_nli_model_name()
    except Exception:
        nli_model = None
    from ._io import build_noise_entry

    # Non-mature = singleton pool (0) + any residual DBSCAN noise (-1).
    noise_ids = [
        gid for gid, lab in zip(state.genome_ids, state.labels)
        if lab is None or int(lab) <= 0
    ]
    noise_entry = build_noise_entry(noise_ids)
    singleton_only = sum(1 for x in state.labels if int(x) == SINGLETON_SPECIES_ID)
    residual_raw = sum(1 for x in state.labels if int(x) == NOISE_LABEL)
    payload = {
        "clustering_method": "dbscan",
        "dbscan_eps": state.eps,
        "dbscan_min_samples": state.min_samples,
        "distance_method": state.distance_method,
        "distance_alpha": state.distance_alpha,
        "nli_model": nli_model,
        "inc_dbscan": True,
        # IncDBSCAN never enforces species_capacity (grow-only density memory).
        "species_capacity_enforced": False,
        # noise = all genomes not in a mature species (min_samples for maturity).
        "noise": noise_entry,
        "noise_count": noise_entry["count"],
        "singleton_species_count": singleton_only,
        "residual_dbscan_noise_count": residual_raw,
        "density_size": state.n_points(),
    }
    if extra:
        payload.update(extra)
        # Re-assert noise after extra merge if caller did not replace it.
        if "noise" not in (extra or {}):
            payload["noise"] = noise_entry
            payload["noise_count"] = noise_entry["count"]
    write_species_state(
        Path(speciation_state_path),
        species,
        current_generation,
        extra=payload,
    )
