"""Incremental DBSCAN speciation orchestration (no elites/archive).

``temp.json`` is the grow-only density memory D_t. Leader-follower
``run_speciation.process_generation`` is not used on this path.

Species have **no max capacity** on this path (unlike L–F Phase 4).
Stagnation freeze applies when a species was used as a parent source and
its max fitness does not improve (same ``species_stagnation`` threshold).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Set

from .clustering.inc_dbscan import (
    load_inc_state,
    process_unlabeled,
    save_inc_state,
    seed_from_batch,
    sync_genomes_from_state,
    validate_against_batch,
    write_species_and_config,
)
from .clustering._io import load_species_state
from .config import SpeciationConfig
from .density_memory import (
    NOISE_LABEL,
    is_labeled,
    load_density_memory,
    save_density_memory,
    unlabeled_indices,
)
from .embeddings import compute_and_save_embeddings
from .events_tracker import EventsTracker
from .genome_tracker import GenomeTracker
from .species import Species, SpeciesIdGenerator
from utils.population_io import _extract_north_star_score

from utils import get_custom_logging, get_system_utils

get_logger, _, _, PerformanceLogger = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


def _eps(cfg: SpeciationConfig) -> float:
    return float(cfg.dbscan_eps if cfg.dbscan_eps is not None else cfg.theta_sim)


def _load_parent_species_ids(outputs_path: Path, logger) -> Set[int]:
    parents_path = outputs_path / "parents.json"
    selected: Set[int] = set()
    if not parents_path.exists():
        return selected
    try:
        with open(parents_path, "r", encoding="utf-8") as f:
            parents = json.load(f)
        if not isinstance(parents, list):
            return selected
        for parent in parents:
            sid = parent.get("species_id")
            if sid is None:
                sid = parent.get("density_label")
            if sid is None:
                continue
            sid_int = int(sid)
            if sid_int > 0:
                selected.add(sid_int)
    except Exception as e:
        logger.warning("IncDBSCAN freeze: failed to load parents.json: %s", e)
    return selected


def _load_prev_max_fitness(state_path: Path) -> Dict[int, float]:
    if not state_path.exists():
        return {}
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return {}
    raw = data.get("prev_max_fitness") or {}
    out: Dict[int, float] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except (TypeError, ValueError):
            continue
    # Fallback: per-species max_fitness from last write
    if not out:
        for sid_str, sp in (data.get("species") or {}).items():
            try:
                out[int(sid_str)] = float((sp or {}).get("max_fitness", 0.0) or 0.0)
            except (TypeError, ValueError):
                continue
    return out


def apply_inc_dbscan_stagnation_freeze(
    species: Dict[int, Species],
    genomes: list,
    *,
    current_generation: int,
    species_stagnation: int,
    outputs_path: Path,
    state_path: Path,
    north_star_metric: str,
    logger,
) -> int:
    """Increment stagnation for parent-source species with no fitness gain; freeze at threshold.

    Does **not** enforce species capacity. Noise (label -1) is never frozen.
    Returns number of species newly frozen this generation.
    """
    if current_generation <= 0:
        return 0

    selected_ids = _load_parent_species_ids(outputs_path, logger)
    prev_max = _load_prev_max_fitness(state_path)

    # Current best fitness per cluster label from density memory
    current_best: Dict[int, float] = defaultdict(float)
    for g in genomes:
        lab = g.get("density_label", g.get("species_id"))
        if lab is None:
            continue
        sid = int(lab)
        if sid <= 0:
            continue
        try:
            score = float(_extract_north_star_score(g, north_star_metric))
        except Exception:
            score = float(g.get("fitness") or 0.0)
        if score > current_best[sid]:
            current_best[sid] = score

    frozen_now = 0
    for sid, sp in list(species.items()):
        if sid <= 0:
            continue
        cur = max(current_best.get(sid, 0.0), float(sp.max_fitness or 0.0))
        # Prefer density-memory max when members exist
        if sid in current_best:
            cur = current_best[sid]
        was_selected = sid in selected_ids
        prev = prev_max.get(sid, float(sp.max_fitness or -1.0))
        improved = cur > prev + 1e-12
        sp.record_fitness(
            current_generation,
            was_selected_as_parent=was_selected,
            max_fitness_increased=improved,
        )
        sp.max_fitness = cur
        if improved:
            sp.last_improvement = current_generation

        if (
            sp.stagnation >= species_stagnation
            and sp.species_state != "frozen"
        ):
            sp.species_state = "frozen"
            frozen_now += 1
            logger.info(
                "IncDBSCAN frozen species %s (stagnation=%d >= %d, was_parent=%s) — "
                "excluded from exploit/explore parent selection",
                sid, sp.stagnation, species_stagnation, was_selected,
            )
        elif was_selected and not improved:
            logger.debug(
                "IncDBSCAN stagnation: species %s stagnation=%d (no fitness gain after parent use)",
                sid, sp.stagnation,
            )

    return frozen_now


def process_generation_inc_dbscan(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    north_star_metric: str = "toxicity",
) -> Dict[str, Any]:
    """Embed newcomers, seed or incrementally insert, persist D_t + species state."""
    cfg = config or SpeciationConfig()
    logger = get_logger("RunIncDBSCAN", log_file)
    outputs = get_outputs_path()
    temp_p = str(temp_path or (outputs / "temp.json"))
    state_path = outputs / "speciation_state.json"
    inc_path = outputs / "inc_dbscan_state.json"
    eps = _eps(cfg)

    logger.info(
        "IncDBSCAN speciation gen=%d eps=%.4f min_samples=%d distance=%s",
        current_generation, eps, cfg.dbscan_min_samples, cfg.distance_method,
    )

    with PerformanceLogger(logger, "IncDBSCAN: embeddings"):
        compute_and_save_embeddings(
            temp_path=temp_p,
            model_name=cfg.embedding_model,
            batch_size=cfg.embedding_batch_size,
            logger=logger,
        )

    genomes = load_density_memory(temp_p, logger=logger)
    empty = {
        "success": True,
        "species_count": 0,
        "active_species_count": 0,
        "frozen_species_count": 0,
        "archive_size": 0,
        "noise_count": 0,
        "density_size": 0,
        "largest_species_size": 0,
        "average_species_size": 0.0,
        "speciation_events": 0,
        "merge_events": 0,
        "extinction_events": 0,
        "archived_count": 0,
        "genomes_updated": 0,
        "elites_moved": 0,
        "archive_moved": 0,
        "error": None,
    }
    if not genomes:
        logger.warning("IncDBSCAN: empty density memory at %s", temp_p)
        return empty

    events_tracker = EventsTracker(current_generation, logger=logger)
    genome_tracker = GenomeTracker(logger=logger)
    try:
        genome_tracker.load()
    except Exception:
        pass

    species: Dict[int, Any] = {}
    merge_events = 0
    speciation_events = 0
    genomes_updated = 0

    inc_state = load_inc_state(inc_path, logger=logger)
    do_seed = (
        current_generation == 0
        or inc_state is None
        or inc_state.n_points() == 0
    )

    if do_seed:
        with PerformanceLogger(logger, "IncDBSCAN: batch seed"):
            SpeciesIdGenerator.reset(0)
            inc_state, species, seed_ev = seed_from_batch(
                genomes,
                eps=eps,
                min_samples=int(cfg.dbscan_min_samples),
                current_generation=current_generation,
                distance_method=cfg.distance_method,
                distance_alpha=cfg.distance_alpha,
                w_genotype=cfg.w_genotype,
                w_phenotype=cfg.w_phenotype,
                logger=logger,
            )
            speciation_events = int(seed_ev.get("new_species", 0))
            genomes_updated = sum(1 for g in genomes if is_labeled(g))
            events_tracker.log("population", "inc_dbscan_init", seed_ev)
    else:
        species = load_species_state(state_path, eps, logger=logger)
        if species:
            SpeciesIdGenerator.set_min_id(max(species) + 1)
        n_unlabeled = len(unlabeled_indices(genomes))
        if n_unlabeled > 0:
            with PerformanceLogger(logger, f"IncDBSCAN: insert {n_unlabeled} newcomers"):
                evs = process_unlabeled(
                    inc_state,
                    genomes,
                    current_generation=current_generation,
                    species=species,
                    logger=logger,
                    events_tracker=events_tracker,
                )
                genomes_updated = n_unlabeled
                for ev in evs:
                    if ev.get("event") == "inc_dbscan_new_species":
                        speciation_events += 1
                    elif ev.get("event") == "inc_dbscan_merge":
                        merge_events += 1

    sync_genomes_from_state(genomes, inc_state)
    save_density_memory(genomes, temp_p, logger=logger)
    save_inc_state(inc_path, inc_state)

    # Freeze stagnant parent-source species (no capacity enforcement on IncDBSCAN).
    frozen_this_gen = apply_inc_dbscan_stagnation_freeze(
        species,
        genomes,
        current_generation=current_generation,
        species_stagnation=int(cfg.species_stagnation),
        outputs_path=outputs,
        state_path=state_path,
        north_star_metric=north_star_metric,
        logger=logger,
    )
    prev_max_payload = {
        str(sid): float(sp.max_fitness or 0.0) for sid, sp in species.items() if sid > 0
    }
    write_species_and_config(
        state_path,
        species,
        current_generation,
        inc_state,
        extra={"prev_max_fitness": prev_max_payload},
    )

    updates = {}
    for g in genomes:
        gid = g.get("id")
        if gid is None:
            continue
        lab = g.get("density_label", g.get("species_id"))
        if lab is not None:
            updates[str(gid)] = int(lab)
    if updates:
        genome_tracker.batch_update(updates, current_generation, "inc_dbscan_assignment")
        try:
            genome_tracker.save()
        except Exception as e:
            logger.warning("genome_tracker.save failed: %s", e)

    validate_every = int(getattr(cfg, "inc_dbscan_validate_every", 0) or 0)
    if validate_every > 0 and current_generation % validate_every == 0:
        ok = validate_against_batch(inc_state, logger=logger)
        events_tracker.log(
            "population",
            "inc_dbscan_validate_ok" if ok else "inc_dbscan_validate_mismatch",
            {"generation": current_generation},
        )
    try:
        events_tracker.save()
    except Exception as e:
        logger.warning("events_tracker.save failed: %s", e)

    noise_count = sum(
        1 for g in genomes
        if int(g.get("density_label", NOISE_LABEL) if g.get("density_label") is not None else NOISE_LABEL) == NOISE_LABEL
    )
    active = len([sp for sp in species.values() if sp.species_state == "active"])
    frozen = len([sp for sp in species.values() if sp.species_state == "frozen"])

    result = {
        "success": True,
        "inc_dbscan": True,
        "species_count": active + frozen,
        "active_species_count": active,
        "frozen_species_count": frozen,
        "archive_size": 0,
        "noise_count": noise_count,
        "density_size": len(genomes),
        "clustered_count": max(0, len(genomes) - noise_count),
        "largest_species_size": max((sp.size for sp in species.values()), default=0),
        "average_species_size": (
            sum(sp.size for sp in species.values()) / len(species) if species else 0.0
        ),
        "speciation_events": speciation_events,
        "merge_events": merge_events,
        "extinction_events": frozen_this_gen,
        "archived_count": 0,
        "genomes_updated": genomes_updated,
        "elites_moved": 0,
        "archive_moved": 0,
        "error": None,
    }
    logger.info(
        "IncDBSCAN done: density=%d species=%d noise=%d new=%d merges=%d frozen=%d",
        result["density_size"], result["species_count"], noise_count,
        speciation_events, merge_events, frozen_this_gen,
    )
    return result


def run_inc_dbscan_speciation(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    north_star_metric: str = "toxicity",
) -> Dict[str, Any]:
    """Public entry used by ``run_speciation`` dispatch."""
    result = process_generation_inc_dbscan(
        temp_path=temp_path,
        current_generation=current_generation,
        config=config,
        log_file=log_file,
        north_star_metric=north_star_metric,
    )
    try:
        from .run_speciation import update_evolution_tracker_with_speciation
        outputs = get_outputs_path()
        # Do not call get_speciation_statistics here: it initializes empty L–F
        # in-memory state and warns about active-count mismatch.
        update_evolution_tracker_with_speciation(
            evolution_tracker_path=str(outputs / "EvolutionTracker.json"),
            current_generation=current_generation,
            speciation_result=result,
            speciation_stats={"initialized": True, "metrics_summary": {}},
            logger=get_logger("RunIncDBSCAN", log_file),
        )
    except Exception as e:
        get_logger("RunIncDBSCAN", log_file).error(
            "Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True,
        )
    return result
