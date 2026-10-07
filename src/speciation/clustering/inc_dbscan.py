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
    distances_to_query,
    normalize_distance_method,
    pairwise_distance_matrix,
)
from ..species import Individual, Species, SpeciesIdGenerator, generate_species_id
from .dbscan import dbscan_precomputed
from ._io import (
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
        return {
            "eps": self.eps,
            "min_samples": self.min_samples,
            "distance_method": self.distance_method,
            "distance_alpha": self.distance_alpha,
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
    """Run batch DBSCAN once; assign persistent IDs; keep noise as ``-1``."""
    log = logger or get_logger("IncDBSCAN")
    method = normalize_distance_method(distance_method)
    individuals = individuals_from_genomes(genomes, log)
    if not individuals:
        state = IncDBSCANState(eps=eps, min_samples=min_samples, distance_method=method,
                               distance_alpha=distance_alpha, w_genotype=w_genotype, w_phenotype=w_phenotype)
        return state, {}, {"new_species": 0, "noise": 0, "n": 0}

    # Align genomes list to individuals with embeddings (same filter as individuals_from_genomes)
    id_to_genome = {g.get("id"): g for g in genomes if g.get("id") is not None}
    ordered_genomes = [id_to_genome[ind.id] for ind in individuals if ind.id in id_to_genome]

    emb = stack_unit_embeddings(individuals)
    objectives = stack_objectives(individuals) if method in ("objective", "embedding+objective") else None
    texts = [ind.prompt or "" for ind in individuals] if "nli" in method else [""] * len(individuals)
    dist = pairwise_distance_matrix(
        method, embeddings=emb, objectives=objectives, texts=texts if "nli" in method else None,
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
    events = {
        "new_species": len(species),
        "noise": int(sum(1 for x in persistent if x == NOISE_LABEL)),
        "n": len(persistent),
        "event": "inc_dbscan_init",
    }
    log.info(
        "IncDBSCAN seed: n=%d eps=%.4f min_samples=%d → species=%d noise=%d",
        events["n"], eps, min_samples, events["new_species"], events["noise"],
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
        if lab == NOISE_LABEL:
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

    emb_row = np.asarray(individual.embedding, dtype=np.float64)
    nrm = float(np.linalg.norm(emb_row))
    if nrm > 0:
        emb_row = emb_row / nrm

    if state.embeddings is None or state.n_points() == 0:
        # First point after empty state
        state.embeddings = emb_row.reshape(1, -1)
        if individual.phenotype is not None:
            state.objectives = np.asarray(individual.phenotype, dtype=np.float64).reshape(1, -1)
        state.texts = [individual.prompt or ""]
        state.genome_ids = [individual.id]
        state.labels = [NOISE_LABEL]
        state.neighbors = [{0}]
        state.is_core = [state.min_samples <= 1]
        if state.is_core[0]:
            sid = generate_species_id()
            state.labels[0] = sid
            set_density_label(genome, sid)
            species[sid] = Species(
                id=sid, leader=individual, members=[individual], radius=state.eps,
                created_at=current_generation, last_improvement=current_generation,
                cluster_origin="inc_dbscan_from_noise", max_fitness=individual.fitness,
            )
            ev = {"event": "inc_dbscan_new_species", "species_id": sid}
        else:
            set_density_label(genome, NOISE_LABEL)
            ev = {"event": "inc_dbscan_insert_noise"}
        if events_tracker:
            events_tracker.log(individual.id, ev["event"], ev)
        return ev

    # Distances from new point to existing
    dists = distances_to_query(
        method,
        query_embedding=emb_row,
        embeddings=state.embeddings,
        query_objective=individual.phenotype,
        objectives=state.objectives,
        query_text=individual.prompt or "",
        texts=state.texts if "nli" in method else None,
        **dkw,
    )
    n_old = state.n_points()
    nbrs = {int(j) for j in np.flatnonzero(dists <= state.eps)}
    new_idx = n_old

    # Append point
    state.embeddings = np.vstack([state.embeddings, emb_row.reshape(1, -1)])
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

    # Clusters of core neighbors (excluding new point for now)
    core_cluster_ids: Set[int] = set()
    for j in nbrs:
        if state.is_core[j] and state.labels[j] != NOISE_LABEL:
            core_cluster_ids.add(state.labels[j])

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
        # Form new species: expand from new_idx through cores/noise in ε-graph
        sid = generate_species_id()
        assigned = _expand_cluster(state, new_idx, sid)
        set_density_label(genome, sid)
        members = [individual]
        # Update genome labels for assigned noise indices — caller syncs genomes list
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

    # Merge multiple species: keep oldest (smallest) parent ID (Ester / IPD convention).
    parents = sorted(core_cluster_ids)
    survivor = parents[0]
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


def _expand_cluster(state: IncDBSCANState, seed: int, sid: int) -> Set[int]:
    """Assign density-reachable points from seed core to ``sid`` (noise + same seed component)."""
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
            # Absorb noise or unlabeled; also walk through points already in sid
            if state.labels[j] in (NOISE_LABEL, sid) or state.labels[j] < 0:
                q.append(j)
            elif state.is_core[j] and state.labels[j] == sid:
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


def process_unlabeled(
    state: IncDBSCANState,
    genomes: List[Dict[str, Any]],
    *,
    current_generation: int,
    species: Dict[int, Species],
    logger=None,
    events_tracker=None,
) -> List[Dict[str, Any]]:
    """Insert every unlabeled genome (with embedding) into ``state``."""
    log = logger or get_logger("IncDBSCAN")
    events: List[Dict[str, Any]] = []
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
    for i in idxs:
        g = genomes[i]
        try:
            ind = Individual.from_genome(g)
        except Exception as e:
            log.warning("IncDBSCAN skip genome id=%s: %s", g.get("id"), e)
            continue
        if ind.embedding is None:
            log.warning("IncDBSCAN skip genome id=%s: no embedding", g.get("id"))
            continue
        ev = insert_point(
            state, g, ind,
            current_generation=current_generation,
            species=species,
            logger=log,
            events_tracker=events_tracker,
        )
        events.append(ev)
    sync_genomes_from_state(genomes, state)
    # Rebuild species members from full state
    individuals = individuals_from_genomes(genomes, log)
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
    # Build maps: cluster -> frozenset of indices
    def _parts(labels):
        groups = defaultdict(set)
        noise = set()
        for i, lab in enumerate(labels):
            lab = int(lab)
            if lab < 0:
                noise.add(i)
            else:
                groups[lab].add(i)
        return set(frozenset(s) for s in groups.values()), frozenset(noise)

    inc_c, inc_n = _parts(state.labels)
    bat_c, bat_n = _parts(raw)
    ok = inc_c == bat_c and inc_n == bat_n
    if not ok:
        log.warning(
            "IncDBSCAN validation mismatch: inc_clusters=%d batch_clusters=%d inc_noise=%d batch_noise=%d",
            len(inc_c), len(bat_c), len(inc_n), len(bat_n),
        )
    else:
        log.info("IncDBSCAN validation OK (n=%d clusters=%d noise=%d)", n, len(inc_c), len(inc_n))
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
    payload = {
        "clustering_method": "dbscan",
        "dbscan_eps": state.eps,
        "dbscan_min_samples": state.min_samples,
        "distance_method": state.distance_method,
        "distance_alpha": state.distance_alpha,
        "inc_dbscan": True,
        # IncDBSCAN never enforces species_capacity (grow-only density memory).
        "species_capacity_enforced": False,
        "noise_count": sum(1 for x in state.labels if x == NOISE_LABEL),
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
