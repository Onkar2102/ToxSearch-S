"""Shared clustering I/O helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence  # Optional used by individuals_from_genomes

import numpy as np

from ..distance import extract_phenotype_vector
from ..species import Individual, Species, SpeciesIdGenerator

from utils import get_system_utils

_, _, _, get_outputs_path, _, _, _ = get_system_utils()

ARCHIVE_SPECIES_ID = -1
# Shared pool for singleton / noise genomes (not mature niches). Mature species IDs start at 1.
SINGLETON_SPECIES_ID = 0


def default_paths(
    temp_path: Optional[str] = None,
    elites_path: Optional[str] = None,
    speciation_state_path: Optional[str] = None,
) -> tuple[Path, Path, Path]:
    out = get_outputs_path()
    return (
        Path(temp_path or out / "temp.json"),
        Path(elites_path or out / "elites.json"),
        Path(speciation_state_path or out / "speciation_state.json"),
    )


def load_json_list(path: Path, logger=None) -> List[dict]:
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        if logger:
            logger.warning("Failed to load %s: %s", path, e)
        return []


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def individuals_from_genomes(
    genomes: Sequence[dict],
    logger=None,
    distance_method: Optional[str] = None,
) -> List[Individual]:
    """Build individuals that have the features required by ``distance_method``.

    When ``distance_method`` is omitted, require embeddings (historical default).
    Otherwise keep genomes that have the features used by that method
    (e.g. prompt text for ``nli``, embeddings for ``embedding``).
    """
    from ..distance import (
        method_uses_embeddings,
        method_uses_objectives,
        method_uses_texts,
        normalize_distance_method,
    )

    need_emb = True
    need_txt = False
    need_obj = False
    if distance_method is not None:
        method = normalize_distance_method(distance_method)
        need_emb = method_uses_embeddings(method)
        need_txt = method_uses_texts(method)
        need_obj = method_uses_objectives(method)

    out: List[Individual] = []
    for g in genomes:
        try:
            ind = Individual.from_genome(g)
        except Exception as e:
            if logger:
                logger.warning("Skipping genome id=%s: %s", g.get("id"), e)
            continue
        if need_emb and ind.embedding is None:
            continue
        if need_txt and not (ind.prompt or "").strip():
            continue
        if need_obj and ind.phenotype is None:
            continue
        # At least one usable signal when method omitted extras
        if not need_emb and not need_txt and not need_obj and ind.embedding is None:
            continue
        out.append(ind)
    return out


def load_species_state(path: Path, theta_sim: float, logger=None) -> Dict[int, Species]:
    """Load live species (id > 0) and restore the species-ID high-water mark."""
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except Exception as e:
        if logger:
            logger.warning("Failed to load %s: %s", path, e)
        return {}

    species: Dict[int, Species] = {}
    for sid_str, sp_dict in state.get("species", {}).items():
        sid = int(sid_str)
        if sid <= 0:
            continue
        emb = np.array(sp_dict["leader_embedding"]) if sp_dict.get("leader_embedding") else None
        pheno = None
        if sp_dict.get("leader_genome_data"):
            pheno = extract_phenotype_vector(sp_dict["leader_genome_data"], logger=logger)
        leader = Individual(
            id=sp_dict["leader_id"],
            prompt=sp_dict.get("leader_prompt", ""),
            fitness=sp_dict.get("leader_fitness", 0.0),
            embedding=emb,
            phenotype=pheno,
            species_id=sid,
        )
        species[sid] = Species(
            id=sid,
            leader=leader,
            members=[leader],
            radius=sp_dict.get("radius", theta_sim),
            stagnation=sp_dict.get("stagnation", 0),
            max_fitness=float(sp_dict.get("max_fitness", leader.fitness) or leader.fitness),
            species_state=sp_dict.get("species_state", "active"),
            created_at=sp_dict.get("created_at", 0),
            last_improvement=sp_dict.get("last_improvement", 0),
            fitness_history=sp_dict.get("fitness_history", []),
            labels=sp_dict.get("labels", []),
            label_history=sp_dict.get("label_history", []),
            cluster_origin=sp_dict.get("cluster_origin", "natural"),
            parent_ids=sp_dict.get("parent_ids"),
            leader_distance=sp_dict.get("leader_distance", 0.0),
        )
    high_water = 0
    if species:
        high_water = max(high_water, max(species))
    for sid in state.get("extinct") or []:
        try:
            high_water = max(high_water, int(sid))
        except (TypeError, ValueError):
            continue
    for sid in state.get("incubators") or []:
        try:
            high_water = max(high_water, int(sid))
        except (TypeError, ValueError):
            continue
    if state.get("next_species_id") is not None:
        SpeciesIdGenerator.restore_next(int(state["next_species_id"]))
    elif high_water > 0:
        SpeciesIdGenerator.set_min_id(high_water + 1)
    return species


def write_species_state(
    path: Path,
    species: Dict[int, Species],
    generation: int,
    extra: Optional[dict] = None,
) -> None:
    payload = {
        "species": {str(sid): sp.to_dict() for sid, sp in species.items()},
        "generation": generation,
        "next_species_id": SpeciesIdGenerator.peek_next(),
    }
    existing: dict = {}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            for key in ("global_best_id", "metrics", "archive_count", "extinct", "incubators"):
                if key in existing and key not in payload:
                    payload[key] = existing[key]
        except Exception:
            existing = {}
    if extra:
        payload.update(extra)

    # Keep extinct list = prior extinct ∪ parent IDs referenced by merges but not alive.
    extinct_ids = set()
    for sid in payload.get("extinct") or []:
        try:
            extinct_ids.add(int(sid))
        except (TypeError, ValueError):
            continue
    for sp in species.values():
        for pid in sp.parent_ids or []:
            try:
                pid_int = int(pid)
            except (TypeError, ValueError):
                continue
            if pid_int > 0 and pid_int not in species:
                extinct_ids.add(pid_int)
                SpeciesIdGenerator.note_id(pid_int)
    if extinct_ids:
        payload["extinct"] = sorted(extinct_ids)

    # Prefer generator high-water over any stale next_species_id in extras.
    payload["next_species_id"] = max(
        int(payload.get("next_species_id") or 1),
        SpeciesIdGenerator.peek_next(),
    )
    save_json(path, payload)


def stack_unit_embeddings(individuals: Sequence[Individual]) -> np.ndarray:
    emb = np.stack([np.asarray(ind.embedding, dtype=np.float64) for ind in individuals], axis=0)
    norms = np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
    return emb / norms


def stack_objectives(individuals: Sequence[Individual]) -> np.ndarray:
    rows = [np.asarray(ind.phenotype, dtype=np.float64) if ind.phenotype is not None else None
            for ind in individuals]
    max_d = max((r.shape[0] for r in rows if r is not None), default=1)
    out = np.full((len(individuals), max_d), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if r is not None:
            out[i, : r.shape[0]] = r
    return out
