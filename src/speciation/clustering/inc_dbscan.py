"""Grow-only Incremental DBSCAN for ToxSearch-S (CLI ``clustering_method=dbscan``).

Gen-0 seeds via ``dbscan_precomputed``. Later genomes are inserted with local
neighborhood updates (absorb / new species from noise / merge). No deletions.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from ..density_memory import (
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
    pairwise_distance_matrix,
)
from ..species import Individual, Species, SpeciesIdGenerator, generate_species_id
from .dbscan import dbscan_precomputed
from ._io import (
    SINGLETON_SPECIES_ID,
    individuals_from_genomes,
    stack_objectives,
    stack_unit_embeddings,
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


def _rebuild_neighbors(state: IncDBSCANState, dist: np.ndarray) -> None:
    n = dist.shape[0]
    state.neighbors = [set(int(j) for j in np.flatnonzero(dist[i] <= state.eps)) for i in range(n)]
    state.is_core = [len(state.neighbors[i]) >= state.min_samples for i in range(n)]


def seed_from_batch(
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
) -> Tuple[IncDBSCANState, Dict[int, Species], Dict[str, Any]]:
    """Run batch DBSCAN once; assign mature IDs (>=1); promote noise → shared pool 0."""
    log = logger or get_logger("IncDBSCAN")
    method = normalize_distance_method(distance_method)
    individuals = individuals_from_genomes(genomes, log, distance_method=method)
    if not individuals:
        state = IncDBSCANState(eps=eps, min_samples=min_samples, distance_method=method,
                               distance_alpha=distance_alpha, w_genotype=w_genotype, w_phenotype=w_phenotype)
        return state, {}, {"new_species": 0, "noise": 0, "singletons": 0, "n": 0}

    # Align genomes list to individuals (same filter as individuals_from_genomes)
    id_to_genome = {g.get("id"): g for g in genomes if g.get("id") is not None}
    ordered_genomes = [id_to_genome[ind.id] for ind in individuals if ind.id in id_to_genome]

    need_emb = method_uses_embeddings(method)
    need_obj = method_uses_objectives(method)
    need_txt = method_uses_texts(method)
    # Keep embeddings when present even for pure nli (optional; used by GDP / hybrid resume).
    have_all_emb = all(ind.embedding is not None for ind in individuals)
    emb = stack_unit_embeddings(individuals) if (need_emb or have_all_emb) else None
    objectives = stack_objectives(individuals) if need_obj else None
    texts = [ind.prompt or "" for ind in individuals] if need_txt else [""] * len(individuals)
    dist = pairwise_distance_matrix(
        method,
        embeddings=emb,
        objectives=objectives,
        texts=texts if need_txt else None,
        alpha=distance_alpha, w_genotype=w_genotype, w_phenotype=w_phenotype, logger=log,
    )
    raw = dbscan_precomputed(dist, eps=eps, min_samples=min_samples)

    SpeciesIdGenerator.reset(0)
    persistent = [NOISE_LABEL] * len(raw)
    local_to_sid: Dict[int, int] = {}
    for lab in sorted({int(x) for x in raw if int(x) >= 0}):
        local_to_sid[lab] = generate_species_id()
    for i, lab in enumerate(raw):
        persistent[i] = NOISE_LABEL if int(lab) < 0 else local_to_sid[int(lab)]

    state = IncDBSCANState(
        eps=eps, min_samples=min_samples, distance_method=method,
        distance_alpha=distance_alpha, w_genotype=w_genotype, w_phenotype=w_phenotype,
        embeddings=emb, objectives=objectives, texts=list(texts),
        genome_ids=[ind.id for ind in individuals],
        labels=list(persistent),
        is_core=[], neighbors=[],
    )
    _rebuild_neighbors(state, dist)

    for g, lab in zip(ordered_genomes, persistent):
        set_density_label(g, lab)

    # Sync unlabeled genomes that lacked embeddings: leave unlabeled
    species = _build_species_from_state(state, ordered_genomes, individuals, current_generation, eps, origin="inc_dbscan_init")
    n_noise_raw = int(sum(1 for x in persistent if x == NOISE_LABEL))
    # Residual DBSCAN noise → shared singleton pool (species_id=0).
    n_singletons = promote_noise_to_singleton_species(
        state, ordered_genomes, species, current_generation=current_generation, logger=log
    )
    if ordered_genomes is not genomes:
        n_singletons += promote_noise_to_singleton_species(
            state, genomes, species, current_generation=current_generation, logger=log
        )
    events = {
        "new_species": len(species),
        "noise": 0,
        "noise_before_singleton_promotion": n_noise_raw,
        "singletons": n_singletons,
        "n": len(persistent),
        "event": "inc_dbscan_init",
    }
    log.info(
        "IncDBSCAN seed: n=%d eps=%.4f min_samples=%d → mature_species=%d (noise→pool0: %d)",
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
    # Drop species no longer present
    alive = set(groups)
    for sid in list(species.keys()):
        if sid not in alive:
            # Keep lineage shell as extinct if we had merges; for init just omit
            if origin == "inc_dbscan_init":
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


def validate_against_batch(state: IncDBSCANState, logger=None) -> bool:
    """Return True if IncDBSCAN partition matches batch DBSCAN up to label permutation."""
    log = logger or get_logger("IncDBSCAN")
    n = state.n_points()
    if n == 0:
        return True
    method = state.distance_method
    dist = pairwise_distance_matrix(
        method,
        embeddings=state.embeddings,
        objectives=state.objectives if method in ("objective", "embedding+objective") else None,
        texts=state.texts if "nli" in method else None,
        alpha=state.distance_alpha,
        w_genotype=state.w_genotype,
        w_phenotype=state.w_phenotype,
        logger=log,
    )
    raw = dbscan_precomputed(dist, eps=state.eps, min_samples=state.min_samples)

    def _parts_inc(labels):
        """Inc labels: >0 mature; 0 pool + -1 noise → anonymous singletons."""
        groups = defaultdict(set)
        for i, lab in enumerate(labels):
            lab = int(lab)
            if lab <= 0:
                groups[("singleton", i)].add(i)
            else:
                groups[("sid", lab)].add(i)
        return {frozenset(m) for m in groups.values()}

    def _parts_batch(labels):
        """Batch DBSCAN local labels: >=0 are clusters (0 is valid); -1 → singleton."""
        groups = defaultdict(set)
        for i, lab in enumerate(labels):
            lab = int(lab)
            if lab < 0:
                groups[("singleton", i)].add(i)
            else:
                groups[("sid", lab)].add(i)
        return {frozenset(m) for m in groups.values()}

    inc_p = _parts_inc(state.labels)
    bat_p = _parts_batch(raw)
    ok = inc_p == bat_p
    if not ok:
        log.warning(
            "IncDBSCAN validation mismatch: inc_parts=%d batch_parts=%d",
            len(inc_p), len(bat_p),
        )
    else:
        log.info("IncDBSCAN validation OK (n=%d parts=%d)", n, len(inc_p))
    return ok


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
        # After promotion, residual raw noise (-1) should be 0; pool 0 holds singletons.
        "noise_count": sum(1 for x in state.labels if x == NOISE_LABEL),
        "singleton_species_count": sum(1 for x in state.labels if int(x) == SINGLETON_SPECIES_ID),
        "density_size": state.n_points(),
    }
    if extra:
        payload.update(extra)
    write_species_state(
        Path(speciation_state_path),
        species,
        current_generation,
        extra=payload,
    )
