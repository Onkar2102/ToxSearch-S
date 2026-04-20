

import json
import time as _time
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

from .config import SpeciationConfig
from .species import Individual, Species, SpeciesIdGenerator
from .embeddings import compute_and_save_embeddings, remove_embeddings_from_temp
from .leader_follower import leader_follower_clustering
from .reserves import Cluster0, CLUSTER_0_ID
from .metrics import SpeciationMetricsTracker, log_generation_summary
from .events_tracker import EventsTracker
from .genome_tracker import GenomeTracker
from .validation import validate_speciation_consistency, validate_flow2_speciation, validate_metrics_from_files

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, PerformanceLogger = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()
_state: Optional[Dict[str, Any]] = None


def _init_state(config: Optional[SpeciationConfig] = None, logger=None) -> None:
    
    global _state
    if _state is None:
        _state = {
            "config": config or SpeciationConfig(),
            "logger": logger or get_logger("Speciation"),
            "species": {},
            "historical_species": {},
            "cluster0": Cluster0(
                min_cluster_size=(config or SpeciationConfig()).cluster0_min_cluster_size,
                theta_sim=(config or SpeciationConfig()).theta_sim,
                max_capacity=(config or SpeciationConfig()).cluster0_max_capacity,
                min_island_size=(config or SpeciationConfig()).min_island_size,
                w_genotype=(config or SpeciationConfig()).w_genotype,
                w_phenotype=(config or SpeciationConfig()).w_phenotype,
                logger=logger or get_logger("Speciation")
            ),
            "global_best": None,
            "metrics_tracker": SpeciationMetricsTracker(logger=logger or get_logger("Speciation")),
            "_current_gen_events": {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0},
            "_embedding_model": None,
            "_archived_count": 0
        }
        _state["logger"].info(f"Speciation initialized: theta_sim={_state['config'].theta_sim}, species_capacity={_state['config'].species_capacity}")
    else:
        if config is not None:
            old_config = _state["config"]
            _state["config"] = config
            if (old_config.theta_sim != config.theta_sim or 
                old_config.cluster0_max_capacity != config.cluster0_max_capacity or
                old_config.cluster0_min_cluster_size != config.cluster0_min_cluster_size):
                _state["cluster0"].theta_sim = config.theta_sim
                _state["cluster0"].max_capacity = config.cluster0_max_capacity
                _state["cluster0"].min_cluster_size = config.cluster0_min_cluster_size
                _state["logger"].info(f"Config updated: theta_sim={config.theta_sim}, species_stagnation={config.species_stagnation}, species_capacity={config.species_capacity}")
        if logger is not None:
            _state["logger"] = logger


def _get_state() -> Dict[str, Any]:
    
    if _state is None:
        _init_state()
    return _state


def _save_tracker_if_dirty(state: Dict[str, Any]) -> None:
    
    if "_genome_tracker" in state:
        tracker = state["_genome_tracker"]
        if tracker._dirty:
            tracker.save()
            state["logger"].debug("Saved genome tracker after critical operation")


def _validate_tracker_consistency(state: Dict[str, Any], phase_name: str) -> None:
    
    if "_genome_tracker" not in state:
        return
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    is_consistent, errors = state["_genome_tracker"].validate_consistency(
        elites_path, reserves_path, archive_path, load_archive=False
    )
    if not is_consistent:
        state["logger"].warning(f"Tracker consistency check failed after {phase_name}: {len(errors)} errors")
        for error in errors[:5]:
            state["logger"].warning(f"  - {error}")


def _validate_species_accounting(
    state: Dict[str, Any],
    phase_name: str,
    incubator_ids: Optional[List[int]] = None,
    extinct_ids: Optional[List[int]] = None,
) -> None:
    
    if "_genome_tracker" not in state:
        return
    logger = state["logger"]
    stats = state["_genome_tracker"].get_distribution_stats()
    by_sid = stats.get("by_species_id", {})
    tracker_species_ids = {int(sid) for sid in by_sid.keys() if int(sid) > 0}
    if not tracker_species_ids:
        return
    alive = {int(sid) for sid in state["species"].keys()}
    if incubator_ids is not None and extinct_ids is not None:
        incubator_set = set(int(sid) for sid in incubator_ids)
        extinct_set = set(int(sid) for sid in extinct_ids)
    else:
        incubator_set = set()
        extinct_set = set()
        for sid, sp in state.get("historical_species", {}).items():
            sid_int = int(sid)
            state_str = getattr(sp, "species_state", None)
            if state_str == "incubator":
                incubator_set.add(sid_int)
            elif state_str == "extinct":
                extinct_set.add(sid_int)
    accounted = alive | incubator_set | extinct_set
    orphan = tracker_species_ids - accounted
    if orphan:
        logger.warning(
            "Species accounting (%s): %d species have genomes in tracker but are not alive, incubator, or extinct: %s. "
            "Alive=%s, incubators=%d, extinct=%d.",
            phase_name,
            len(orphan),
            sorted(orphan),
            sorted(alive),
            len(incubator_set),
            len(extinct_set),
        )
    else:
        logger.debug(
            "Species accounting (%s): all %d tracker species accounted (alive=%d, incubator=%d, extinct=%d)",
            phase_name, len(tracker_species_ids), len(alive), len(incubator_set), len(extinct_set),
        )


def _load_json_file(file_path: Path, logger, default=None):
    
    if default is None:
        default = []
    if not file_path.exists():
        return default
    try:
        if file_path.name == "elites.json":
            from utils.population_io import load_elites
            return load_elites(str(file_path), logger=logger) or default
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load {file_path.name}: {e}")
        return default


def _update_genomes_from_tracker(genomes: List[Dict], genome_tracker, current_generation: int, 
                                  logger, file_name: str, default_species_id: int = 0) -> None:
    
    for genome in genomes:
        genome_id = str(genome.get("id")) if genome.get("id") else None
        if not genome_id:
            continue
            
        if genome_tracker.exists(genome_id):
            tracker_sid = genome_tracker.get_species_id(genome_id)
            file_sid = genome.get("species_id")
            if file_sid != tracker_sid:
                genome["species_id"] = tracker_sid
                logger.debug(f"Updated {genome_id} species_id: {file_sid} -> {tracker_sid} ({file_name})")
        else:
            file_sid = genome.get("species_id")
            if file_sid is None:
                file_sid = default_species_id
                genome["species_id"] = default_species_id
            genome_tracker.register(genome_id, file_sid, current_generation)
            logger.debug(f"Registered {genome_id} with species_id={file_sid} ({file_name})")


def _deduplicate_genomes(genomes: List[Dict]) -> List[Dict]:
    
    seen = set()
    result = []
    for genome in genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if gid and gid not in seen:
            seen.add(gid)
            result.append(genome)
    return result


def _write_json_atomic(file_path: Path, data: List[Dict], logger, file_name: str) -> None:
    
    out = data if isinstance(data, list) else []
    temp_path = file_path.with_suffix('.json.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    temp_path.replace(file_path)
    logger.info(f"Wrote {len(out)} genomes to {file_name}")


def _validate_active_count(state: Dict[str, Any], calculated_count: int, source: str) -> int:
    
    in_memory_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    if calculated_count != in_memory_count:
        state["logger"].warning(
            f"Active count mismatch: calculated={calculated_count} (from {source}), "
            f"in_memory={in_memory_count}, using in_memory as source of truth"
        )
        return in_memory_count
    return calculated_count


def _archive_individuals(individuals: List[Individual], generation: int, reason: str) -> None:
    
    if not individuals:
        return
    
    state = _get_state()
    state["_archived_count"] += len(individuals)
    logger = state["logger"]
    
    try:
        outputs_path = get_outputs_path()
        archive_path = outputs_path / "archive.json"
        
        archive = _load_json_file(archive_path, logger, [])
        if not isinstance(archive, list):
            if isinstance(archive, dict):
                logger.warning(f"archive.json is a dict (expected list), converting to list")
                archive = list(archive.values()) if len(archive) > 0 else []
            else:
                logger.warning(f"archive.json has unexpected format, initializing as empty list")
                archive = []
        
        for ind in individuals:
            if hasattr(ind, 'to_genome'):
                entry = ind.to_genome()
                if not entry:
                    entry = {}
                if "id" not in entry:
                    entry["id"] = ind.id
                if "prompt" not in entry and hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            else:
                entry = {"id": ind.id}
                if hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            
            entry["archived_at_generation"] = generation
            entry["archive_reason"] = reason
            if "generation" not in entry and hasattr(ind, 'generation'):
                entry["generation"] = ind.generation
            elif "generation" not in entry:
                entry["generation"] = generation
            if hasattr(ind, 'fitness') and "fitness" not in entry:
                entry["fitness"] = ind.fitness
            entry["species_id"] = -1
            if "initial_state" not in entry:
                if "capacity" in reason.lower():
                    entry["initial_state"] = "non-elite"
                else:
                    entry["initial_state"] = entry.get("initial_state", "elite")
            archive.append(entry)
        
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Archived {len(individuals)} individuals ({reason}) to archive.json")
        
    except Exception as e:
        logger.warning(f"Failed to archive individuals: {e}")


def _load_species_leaders_from_state(outputs_path: Path, logger) -> Dict[int, Tuple[Individual, Any, Optional[Any]]]:
    
    import numpy as np
    from .species import Individual
    
    speciation_state_path = outputs_path / "speciation_state.json"
    leaders = {}
    
    if not speciation_state_path.exists():
        logger.debug("speciation_state.json not found, cannot load leaders (this is normal for Generation 0)")
        return leaders
    
    try:
        with open(speciation_state_path, 'r', encoding='utf-8') as f:
            state_data = json.load(f)
        
        species_dict = state_data.get("species", {})
        for sid_str, sp_dict in species_dict.items():
            sid = int(sid_str)
            if sid == 0:
                continue
            
            leader_id = sp_dict.get("leader_id")
            leader_embedding_list = sp_dict.get("leader_embedding")
            leader_fitness = sp_dict.get("leader_fitness", 0.0)
            leader_prompt = sp_dict.get("leader_prompt", "")
            
            if not leader_id or not leader_embedding_list:
                logger.debug(f"Species {sid} has no leader or embedding, skipping")
                continue
            
            leader_embedding = np.array(leader_embedding_list, dtype=np.float32)
            
            leader_phenotype = None
            if sp_dict.get("leader_genome_data"):
                from .phenotype_distance import extract_phenotype_vector
                leader_phenotype = extract_phenotype_vector(sp_dict["leader_genome_data"], logger=logger)
            
            leader = Individual(
                id=leader_id,
                prompt=leader_prompt,
                fitness=leader_fitness,
                embedding=leader_embedding,
                phenotype=leader_phenotype,
                species_id=sid
            )
            
            leaders[sid] = (leader, leader_embedding, leader_phenotype)
        
        logger.info(f"Loaded {len(leaders)} species leaders from speciation_state.json")
    except Exception as e:
        logger.warning(f"Failed to load species leaders from speciation_state.json: {e}")
    
    return leaders


def _load_genomes_by_ids(genome_ids: List[str], outputs_path: Path, logger) -> List[Dict[str, Any]]:
    
    genome_ids_set = set(str(gid) for gid in genome_ids)
    loaded_genomes = []
    seen_ids = set()
    
    for file_name in ["elites.json", "temp.json", "reserves.json"]:
        file_path = outputs_path / file_name
        genomes = _load_json_file(file_path, logger, [])
        for g in genomes:
            gid = str(g.get("id")) if g.get("id") else None
            if gid and gid in genome_ids_set and gid not in seen_ids:
                loaded_genomes.append(g)
                seen_ids.add(gid)
    
    return loaded_genomes


def phase8_redistribute_genomes(temp_path: Optional[str] = None, current_generation: int = 0) -> Dict[str, int]:
    
    state = _get_state()
    logger = state["logger"]
    outputs_path = get_outputs_path()
    
    if temp_path is None:
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    
    if "_genome_tracker" not in state:
        logger.error("Genome tracker not initialized")
        return {"elites_moved": 0, "reserves_moved": 0, "archived_moved": 0, "total_processed": 0}
    
    genome_tracker = state["_genome_tracker"]
    
    stats = genome_tracker.get_distribution_stats()
    has_archived = int(stats["by_species_id"].get("-1", 0)) > 0
    
    logger.info("Phase 7: Steps 1-3 - Updating species_id from tracker")
    elites_genomes = _load_json_file(elites_path, logger, [])
    reserves_genomes = _load_json_file(reserves_path, logger, [])
    temp_genomes = _load_json_file(temp_path_obj, logger, [])
    archive_genomes = _load_json_file(archive_path, logger, [])
    
    _update_genomes_from_tracker(elites_genomes, genome_tracker, current_generation, logger, "elites.json", default_species_id=0)
    _update_genomes_from_tracker(reserves_genomes, genome_tracker, current_generation, logger, "reserves.json", default_species_id=0)
    _update_genomes_from_tracker(temp_genomes, genome_tracker, current_generation, logger, "temp.json", default_species_id=0)
    
    for genome in temp_genomes:
        if "generation" not in genome or genome.get("generation") is None:
            genome["generation"] = current_generation
    
    logger.info("Phase 7: Step 4 - Redistributing genomes to correct files")
    
    tracked_ids = set(genome_tracker.genomes.keys())
    file_sources = {
        "elites": (elites_genomes, set(str(g.get("id")) for g in elites_genomes if g.get("id"))),
        "reserves": (reserves_genomes, set(str(g.get("id")) for g in reserves_genomes if g.get("id"))),
        "temp": (temp_genomes, set(str(g.get("id")) for g in temp_genomes if g.get("id")))
    }
    
    tracked_genomes = []
    untracked_by_file = {"elites": [], "reserves": []}
    
    for file_name, (genomes, ids_set) in file_sources.items():
        for genome in genomes:
            gid = str(genome.get("id")) if genome.get("id") else None
            if gid and gid in tracked_ids:
                tracked_genomes.append(genome)
            elif gid and file_name in untracked_by_file:
                untracked_by_file[file_name].append(genome)
    
    _merged_by_id = {}
    _merge_order = []
    for genome in tracked_genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if not gid:
            continue
        if gid not in _merged_by_id:
            _merged_by_id[gid] = genome
            _merge_order.append(gid)
        else:
            canon = _merged_by_id[gid]
            if genome_tracker.exists(gid):
                canon["species_id"] = genome_tracker.get_species_id(gid)
            for _k, _v in genome.items():
                if _v is not None and (_k not in canon or canon.get(_k) is None):
                    canon[_k] = _v
    tracked_genomes = [_merged_by_id[gid] for gid in _merge_order]
    for genome in tracked_genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if gid and genome_tracker.exists(gid):
            genome["species_id"] = genome_tracker.get_species_id(gid)
    
    elites_to_save = untracked_by_file["elites"]
    reserves_to_save = untracked_by_file["reserves"]
    archive_to_save = [g for g in archive_genomes if g.get("species_id") == -1 or str(g.get("id")) not in tracked_ids]
    movements = []
    
    for genome in tracked_genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if not gid:
            continue
        
        species_id = genome.get("species_id")
        if species_id is None:
            species_id = genome_tracker.get_species_id(gid) if genome_tracker.exists(gid) else 0
            genome["species_id"] = species_id
            if not genome_tracker.exists(gid):
                genome_tracker.register(gid, 0, current_generation)
        
        if species_id > 0:
            new_file = "elites"
        elif species_id == 0:
            new_file = "reserves"
        elif species_id == -1:
            new_file = "archive"
        else:
            logger.warning(f"Genome {gid} has invalid species_id: {species_id}, skipping")
            continue
        
        old_file = next((f for f, (_, ids) in file_sources.items() if gid in ids), None)
        if old_file and old_file != new_file:
            movements.append((gid, old_file, new_file))
        
        {"elites": elites_to_save, "reserves": reserves_to_save, "archive": archive_to_save}[new_file].append(genome)
    
    for genome_list in [elites_to_save, reserves_to_save]:
        to_archive = [g for g in genome_list if g.get("archive_reason")]
        for genome in to_archive:
            _rid = genome.get("id")
            gid = str(_rid) if _rid is not None and _rid != "" else None
            if gid is not None:
                if genome_tracker.exists(gid) and genome_tracker.get_species_id(gid) != -1:
                    genome_tracker.update_species_id(gid, -1, current_generation, f"archive_reason_{genome.get('archive_reason')}")
                elif not genome_tracker.exists(gid):
                    genome_tracker.register(gid, -1, current_generation)
                genome["species_id"] = -1
                if gid not in {str(g.get("id")) for g in archive_to_save if g.get("id")}:
                    archive_to_save.append(genome)
        for g in to_archive:
            if g in genome_list:
                genome_list.remove(g)
    
    logger.info("Phase 7: Step 5 - Deduplicating and writing files")
    elites_deduped = _deduplicate_genomes(elites_to_save)
    reserves_deduped = _deduplicate_genomes(reserves_to_save)
    archive_deduped = _deduplicate_genomes(archive_to_save)
    
    _seen_prompts = set()
    _removed_dup_prompt_ids = []
    _new_elites = []
    for g in elites_deduped:
        pr = g.get("prompt")
        if pr is None:
            _new_elites.append(g)
            continue
        if not isinstance(pr, str):
            pr = str(pr)
        if pr in _seen_prompts:
            gid = g.get("id")
            if gid is not None:
                _removed_dup_prompt_ids.append(str(gid))
            logger.warning(
                "Phase 7: omitting genome id=%s from elites — duplicate prompt (case-sensitive) already retained",
                gid,
            )
            continue
        _seen_prompts.add(pr)
        _new_elites.append(g)
    elites_deduped = _new_elites
    _new_reserves = []
    for g in reserves_deduped:
        pr = g.get("prompt")
        if pr is None:
            _new_reserves.append(g)
            continue
        if not isinstance(pr, str):
            pr = str(pr)
        if pr in _seen_prompts:
            gid = g.get("id")
            if gid is not None:
                _removed_dup_prompt_ids.append(str(gid))
            logger.warning(
                "Phase 7: omitting genome id=%s from reserves — duplicate prompt (case-sensitive) already retained",
                gid,
            )
            continue
        _seen_prompts.add(pr)
        _new_reserves.append(g)
    reserves_deduped = _new_reserves
    _new_archive = []
    for g in archive_deduped:
        pr = g.get("prompt")
        if pr is None:
            _new_archive.append(g)
            continue
        if not isinstance(pr, str):
            pr = str(pr)
        if pr in _seen_prompts:
            gid = g.get("id")
            if gid is not None:
                _removed_dup_prompt_ids.append(str(gid))
            logger.warning(
                "Phase 7: omitting genome id=%s from archive — duplicate prompt (case-sensitive) already retained",
                gid,
            )
            continue
        _seen_prompts.add(pr)
        _new_archive.append(g)
    archive_deduped = _new_archive

    _elite_ids = {
        str(g.get("id"))
        for g in elites_deduped
        if g.get("id") is not None and str(g.get("id")) != ""
    }
    _n_res_before = len(reserves_deduped)
    reserves_deduped = [
        g
        for g in reserves_deduped
        if g.get("id") is None
        or str(g.get("id")) == ""
        or str(g.get("id")) not in _elite_ids
    ]
    if len(reserves_deduped) < _n_res_before:
        logger.warning(
            "Phase 7: dropped %d reserve row(s) whose id already appears in elites (placement invariant)",
            _n_res_before - len(reserves_deduped),
        )
    _living_ids = _elite_ids | {
        str(g.get("id"))
        for g in reserves_deduped
        if g.get("id") is not None and str(g.get("id")) != ""
    }
    _n_arch_before = len(archive_deduped)
    archive_deduped = [
        g
        for g in archive_deduped
        if g.get("id") is None
        or str(g.get("id")) == ""
        or str(g.get("id")) not in _living_ids
    ]
    if len(archive_deduped) < _n_arch_before:
        logger.warning(
            "Phase 7: dropped %d archive row(s) whose id already appears in elites or reserves",
            _n_arch_before - len(archive_deduped),
        )

    for _rid in _removed_dup_prompt_ids:
        if _rid in genome_tracker.genomes:
            del genome_tracker.genomes[_rid]
            genome_tracker._dirty = True
    
    _write_json_atomic(elites_path, elites_deduped, logger, "elites.json")
    _write_json_atomic(reserves_path, reserves_deduped, logger, "reserves.json")
    _write_json_atomic(archive_path, archive_deduped, logger, "archive.json")
    
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    logger.info("Cleared temp.json")
    
    if "_events_tracker" in state and movements:
        for genome_id, old_file, new_file in movements:
            state["_events_tracker"].log(genome_id, "file_redistribution", 
                {"from_file": old_file, "to_file": new_file, "generation": current_generation})
        state["_events_tracker"].save()
        logger.info(f"Logged {len(movements)} file redistribution events")
    
    logger.info("Phase 7: Final validation")
    is_consistent, errors = genome_tracker.validate_consistency(
        elites_path, reserves_path, archive_path, load_archive=has_archived
    )
    
    if not is_consistent:
        logger.warning(f"Phase 7 validation found {len(errors)} inconsistencies:")
        for error in errors[:10]:
            logger.warning(f"  - {error}")
    else:
        logger.info("Phase 7 validation passed - all genomes consistent")
    
    distribution_stats = {
        "elites_moved": len(elites_deduped),
        "reserves_moved": len(reserves_deduped),
        "archived_moved": len(archive_deduped),
        "total_processed": len(genome_tracker.genomes),
        "file_movements": len(movements),
        "validation_errors": len(errors) if not is_consistent else 0
    }
    
    logger.info(f"Phase 7 complete: {distribution_stats}")
    return distribution_stats


def process_generation(current_generation: int,
                       temp_path: Optional[str] = None,
                       config: Optional[SpeciationConfig] = None, logger=None) -> Tuple[Dict[int, Species], Cluster0]:
    
    _init_state(config, logger)
    state = _get_state()
    
    
    _process_gen_start = _time.time()
    state["logger"].info("STARTING: Speciation generation %d", current_generation)
    state["logger"].info(f"=== Speciation Generation {current_generation} ===")
    state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0}
    state["_archived_count"] = 0
    
    events_tracker = EventsTracker(current_generation, logger=state["logger"])
    state["_events_tracker"] = events_tracker
    
    from .species import Individual
    
    genome_tracker = GenomeTracker(logger=state["logger"])
    genome_tracker.load()
    
    if len(genome_tracker.genomes) == 0:
        outputs_path_check = get_outputs_path()
        elites_path_check = outputs_path_check / "elites.json"
        is_gen0 = not elites_path_check.exists() or (elites_path_check.exists() and len(json.load(open(elites_path_check, 'r', encoding='utf-8'))) == 0)
        
        if not is_gen0:
            from .migration import auto_migrate_if_needed
            auto_migrate_if_needed(logger=state["logger"])
            genome_tracker.load()
        else:
            state["logger"].debug("Generation 0 detected: files initialized but empty, skipping auto-migration")
    
    state["_genome_tracker"] = genome_tracker
    
    if current_generation > 0:
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        if Path(state_path).exists():
            load_state(state_path)
            state["logger"].info("Restored speciation state from previous generation")
    
    
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    
    has_existing_species = len(state["species"]) > 0
    
    elites_has_genomes = False
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_data = json.load(f)
                elites_has_genomes = (isinstance(elites_data, list) and len(elites_data) > 0) or \
                                     (isinstance(elites_data, dict) and len(elites_data) > 0)
        except (json.JSONDecodeError, Exception):
            elites_has_genomes = False
    
    is_generation_0 = not has_existing_species and not elites_has_genomes
    
    if has_existing_species and not elites_has_genomes:
        state["logger"].debug(f"Found {len(state['species'])} existing species in memory but elites.json is empty - will run Phase 1")
    
    if is_generation_0:
        state["logger"].info("=== Generation 0: Skipping Phase 1 (no existing species) ===")
        state["_prev_max_fitness"] = {}
        
        
        if temp_path is None:
            outputs_path = get_outputs_path()
            temp_path = str(outputs_path / "temp.json")
        
        temp_path_obj = Path(temp_path)
        if temp_path_obj.exists():
            compute_and_save_embeddings(
                temp_path=temp_path,
                model_name=state["config"].embedding_model,
                batch_size=state["config"].embedding_batch_size,
                logger=state["logger"]
            )
            
            try:
                with open(temp_path_obj, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                
                registered_count = 0
                added_to_cluster0_count = 0
                for genome in temp_genomes:
                    genome_id = str(genome.get("id")) if genome.get("id") else None
                    if genome_id and not state["_genome_tracker"].exists(genome_id):
                        state["_genome_tracker"].register(genome_id, 0, current_generation)
                        registered_count += 1
                    
                    if genome.get("prompt_embedding"):
                        try:
                            ind = Individual.from_genome(genome)
                            if ind.embedding is not None:
                                existing_ids = {cm.individual.id for cm in state["cluster0"].members}
                                if ind.id not in existing_ids:
                                    state["cluster0"].add(ind, current_generation)
                                    added_to_cluster0_count += 1
                        except Exception as e:
                            state["logger"].warning(f"Failed to add genome {genome_id} to cluster0: {e}")
                
                if registered_count > 0:
                    state["logger"].info(f"Registered {registered_count} genomes from temp.json in tracker with species_id=0 (Generation 0)")
                    state["_genome_tracker"].save()
                if added_to_cluster0_count > 0:
                    state["logger"].info(f"Added {added_to_cluster0_count} genomes to cluster0 from temp.json (Generation 0)")
            except Exception as e:
                state["logger"].warning(f"Failed to register temp.json genomes in tracker: {e}", exc_info=True)
    else:
        _phase1_start = _time.time()
        state["logger"].info("=== Phase 1: Existing Species Processing ===")

        with PerformanceLogger(state["logger"], "Speciation Phase 1: Compute prev max fitness"):
            outputs_path_prev_max = get_outputs_path()
            state["_prev_max_fitness"] = {}
            
            if "_genome_tracker" in state:
                from utils.population_io import _extract_north_star_score
                
                for sid, sp in state["species"].items():
                    species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                    
                    if species_genome_ids:
                        loaded_genomes = _load_genomes_by_ids(species_genome_ids, outputs_path_prev_max, state["logger"])
                        
                        if loaded_genomes:
                            max_fitness = 0.0
                            for g in loaded_genomes:
                                fitness = _extract_north_star_score(g, "toxicity")
                                if fitness is not None and fitness > max_fitness:
                                    max_fitness = fitness
                            state["_prev_max_fitness"][int(sid)] = max_fitness
                            
                            if abs(sp.max_fitness - max_fitness) > 0.001:
                                state["logger"].debug(
                                    f"Species {sid}: _prev_max_fitness corrected from in-memory {sp.max_fitness:.4f} "
                                    f"to tracker-based {max_fitness:.4f} ({len(loaded_genomes)} genomes)"
                                )
                        else:
                            state["_prev_max_fitness"][int(sid)] = sp.max_fitness
                    else:
                        state["_prev_max_fitness"][int(sid)] = sp.max_fitness
            else:
                state["_prev_max_fitness"] = {int(sid): sp.max_fitness for sid, sp in state["species"].items()}

        if temp_path is None:
            outputs_path = get_outputs_path()
            temp_path = str(outputs_path / "temp.json")
        
        with PerformanceLogger(state["logger"], "Speciation Phase 1: Compute and save embeddings"):
            compute_and_save_embeddings(
                temp_path=temp_path,
                model_name=state["config"].embedding_model,
                batch_size=state["config"].embedding_batch_size,
                logger=state["logger"]
            )
        
        outputs_path = get_outputs_path()
        speciation_state_path = str(outputs_path / "speciation_state.json")
        
        species_count_before_clustering = len(state["species"])
        
        with PerformanceLogger(state["logger"], "Speciation Phase 1: Leader-follower clustering (variants)"):
            state["species"], _ = leader_follower_clustering(
                temp_path=temp_path,
                speciation_state_path=speciation_state_path,
                theta_sim=state["config"].theta_sim,
                current_generation=current_generation,
                w_genotype=state["config"].w_genotype,
                w_phenotype=state["config"].w_phenotype,
                min_island_size=state["config"].min_island_size,
                skip_cluster0_outliers=True,
                logger=state["logger"],
                genome_tracker=state.get("_genome_tracker"),
                events_tracker=state.get("_events_tracker")
            )
        
        species_count_after_clustering = len(state["species"])
        new_species_from_clustering = species_count_after_clustering - species_count_before_clustering
        if new_species_from_clustering > 0:
            state["_current_gen_events"]["speciation"] += new_species_from_clustering
            state["logger"].info(f"Counted {new_species_from_clustering} new species formed during leader-follower clustering (before: {species_count_before_clustering}, after: {species_count_after_clustering})")
        
        with PerformanceLogger(state["logger"], "Speciation Phase 1: Radius enforcement"):
            from .distance import ensemble_distance
            import numpy as np
            
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            temp_path_obj = Path(temp_path)
            
            for sid in list(state["species"].keys()):
                sp = state["species"][sid]
                if sp.leader is None or sp.leader.embedding is None:
                    continue
                
                species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                
                all_member_genomes = []
                
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        all_member_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                    except Exception as e:
                        state["logger"].warning(f"Failed to load elites.json for radius cleanup: {e}")
                
                if temp_path_obj.exists():
                    try:
                        with open(temp_path_obj, 'r', encoding='utf-8') as f:
                            temp_genomes = json.load(f)
                        for g in temp_genomes:
                            if g.get("id") in species_genome_ids and not any(mg.get("id") == g.get("id") for mg in all_member_genomes):
                                all_member_genomes.append(g)
                    except Exception as e:
                        state["logger"].warning(f"Failed to load temp.json for radius cleanup: {e}")
                
                members_to_remove = []
                for genome in all_member_genomes:
                    genome_id = genome.get("id")
                    if genome_id == sp.leader.id:
                        continue
                    
                    genome_embedding = genome.get("prompt_embedding")
                    if genome_embedding is None:
                        members_to_remove.append(genome_id)
                        continue
                    
                    from .phenotype_distance import extract_phenotype_vector
                    genome_phenotype = extract_phenotype_vector(genome, logger=state["logger"])
                    leader_phenotype = sp.leader.phenotype
                    
                    dist = ensemble_distance(
                        np.array(genome_embedding), sp.leader.embedding,
                        genome_phenotype, leader_phenotype,
                        state["config"].w_genotype, state["config"].w_phenotype
                    )
                    
                    if dist >= state["config"].theta_sim:
                        members_to_remove.append(genome_id)
                
                if members_to_remove:
                    state["logger"].debug(f"Species {sid}: removing {len(members_to_remove)} members outside radius")
                    for genome_id in members_to_remove:
                        state["_genome_tracker"].update_species_id(
                            str(genome_id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves"
                        )
                    
                    updated_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                    sp.members = [m for m in sp.members if m.id in updated_member_ids]
        
        _save_tracker_if_dirty(state)
        _validate_tracker_consistency(state, "Phase 1")
        state["logger"].info("Phase 1 completed in %.2fs", _time.time() - _phase1_start)
    
    
    _phase2_start = _time.time()
    state["logger"].info("=== Phase 2: Cluster 0 Speciation (Isolated) ===")
    
    outputs_path = get_outputs_path()
    
    with PerformanceLogger(state["logger"], "Speciation Phase 2: Load cluster 0 and sync members"):
        if "_genome_tracker" not in state:
            state["logger"].warning("Genome tracker not available, cannot collect reserves")
        else:
            reserves_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(0)
            state["logger"].info(f"Found {len(reserves_genome_ids)} genomes with species_id=0 in tracker")
            
            reserves_genomes = _load_genomes_by_ids(reserves_genome_ids, outputs_path, state["logger"])
            state["logger"].debug(f"Loaded {len(reserves_genomes)} genome data entries for reserves (from {len(reserves_genome_ids)} IDs in tracker)")
            
            species_member_ids = set()
            for sp in state["species"].values():
                for member in sp.members:
                    species_member_ids.add(member.id)
            
            removed_count = 0
            cluster0_members_to_keep = []
            for cm in state["cluster0"].members:
                if cm.individual.id not in species_member_ids:
                    cluster0_members_to_keep.append(cm)
                else:
                    removed_count += 1
            
            state["cluster0"].members = cluster0_members_to_keep
            
            existing_cluster0_ids = {cm.individual.id for cm in state["cluster0"].members}
            added_count = 0
            for genome in reserves_genomes:
                genome_id = genome.get("id")
                if genome_id and genome_id not in existing_cluster0_ids and genome_id not in species_member_ids:
                    if genome.get("prompt_embedding"):
                        try:
                            outlier_ind = Individual.from_genome(genome)
                            if outlier_ind.embedding is not None:
                                state["cluster0"].add(outlier_ind, current_generation)
                                added_count += 1
                        except Exception as e:
                            state["logger"].warning(f"Failed to add genome {genome_id} to cluster0: {e}")
            
            if removed_count > 0 or added_count > 0:
                state["logger"].info(f"Synced cluster 0: removed {removed_count} (now in species), added {added_count} (from tracker with species_id=0)")
    
    cluster0_individuals = [ind for ind in state["cluster0"].individuals if getattr(ind, "embedding", None) is not None]
    
    sorted_individuals = sorted(cluster0_individuals, key=lambda x: x.fitness, reverse=True)
    
    with PerformanceLogger(state["logger"], "Speciation Phase 2: Cluster 0 speciation (isolated)"):
        new_species_from_cluster0 = cluster0_speciation_isolated(
            current_generation=current_generation,
            config=state["config"],
            logger=state["logger"],
            pre_sorted_individuals=sorted_individuals
        )
    
    newly_formed_species_ids = set()
    speciation_state_path = outputs_path / "speciation_state.json"
    
    for new_species in new_species_from_cluster0:
        state["species"][new_species.id] = new_species
        newly_formed_species_ids.add(new_species.id)
        state["_current_gen_events"]["speciation"] += 1
        state["logger"].info(f"Species {new_species.id} formed from cluster 0 ({new_species.size} members)")
        
        if "_genome_tracker" in state:
            updates = {str(m.id): new_species.id for m in new_species.members}
            result = state["_genome_tracker"].batch_update(
                updates, current_generation, f"species_formed_from_cluster0_{new_species.id}"
            )
            if result["failed"] > 0:
                state["logger"].warning(f"Tracker update failed for {result['failed']} genomes in new species {new_species.id}")
            state["_genome_tracker"].save()
        
        try:
            existing_state = {}
            file_existed = speciation_state_path.exists()
            if file_existed:
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
            
            state_dict = {
                "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                "generation": current_generation
            }
            state_dict["cluster0"] = existing_state.get("cluster0", {})
            state_dict["global_best_id"] = existing_state.get("global_best_id")
            state_dict["metrics"] = existing_state.get("metrics", {})
            state_dict["incubators"] = existing_state.get("incubators", [])
            state_dict["extinct"] = existing_state.get("extinct", [])
            
            with open(speciation_state_path, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            
            if not file_existed:
                state["logger"].debug(f"Created speciation_state.json for Generation {current_generation}")
        except Exception as e:
            state["logger"].warning(f"Failed to update speciation_state.json immediately for species {new_species.id}: {e}")
        
        if "_events_tracker" in state:
            for member in new_species.members:
                state["_events_tracker"].log(
                    member.id, "species_formed_from_cluster0",
                    {"species_id": new_species.id, "size": new_species.size}
                )
    
    state["logger"].debug("Skipping radius cleanup for newly formed species (Flow 2: no radius enforcement)")
    
    state["logger"].debug("Skipping capacity enforcement in Phase 2 (moved to Phase 4, after merging)")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 2: Save tracker and validate"):
        _save_tracker_if_dirty(state)
        _validate_tracker_consistency(state, "Phase 2")
    
    if newly_formed_species_ids:
        outputs_path = get_outputs_path()
        is_valid, errors = validate_flow2_speciation(
            outputs_path=outputs_path,
            generation=current_generation,
            newly_formed_species_ids=list(newly_formed_species_ids),
            logger=state["logger"]
        )
        if not is_valid:
            state["logger"].warning(f"Flow 2 validation found {len(errors)} errors for newly formed species")
            for error in errors[:5]:
                state["logger"].warning(f"  - {error}")
        else:
            state["logger"].debug(f"Flow 2 validation passed for {len(newly_formed_species_ids)} newly formed species")
    
    state["logger"].info("Phase 2 completed in %.2fs", _time.time() - _phase2_start)
    
    
    _phase3_start = _time.time()
    state["logger"].info("=== Phase 3: Merging + Radius Enforcement ===")
    
    
    outputs_path = get_outputs_path()
    speciation_state_path = outputs_path / "speciation_state.json"
    
    with PerformanceLogger(state["logger"], "Speciation Phase 3: Merging"):
        species_leaders = _load_species_leaders_from_state(outputs_path, state["logger"])
        
        for sid, sp in state["species"].items():
            if sid not in species_leaders and sp.leader and sp.leader.embedding is not None:
                species_leaders[sid] = (sp.leader, sp.leader.embedding, sp.leader.phenotype)
        
        species_info = {}
        for sid, (leader, embedding, phenotype) in species_leaders.items():
            if sid in state["species"]:
                sp = state["species"][sid]
                if sp.species_state != "incubator" and sp.leader:
                    species_info[sid] = {
                        "species": sp,
                        "leader": leader,
                        "embedding": embedding,
                        "phenotype": phenotype,
                        "created_at": sp.created_at
                    }
        
        from .distance import ensemble_distance
        from .merging import merge_islands
        
        species_count_before_merge = len(state["species"])
        merge_count = 0
        
        while True:
            merge_candidates = []
            species_list = list(species_info.items())
            
            for i, (id1, info1) in enumerate(species_list):
                for j, (id2, info2) in enumerate(species_list[i + 1:], start=i + 1):
                    min_stable = state["config"].min_stability_gens
                    sp1_stable = (current_generation - info1["created_at"]) >= min_stable
                    sp2_stable = (current_generation - info2["created_at"]) >= min_stable
                    
                    if not (sp1_stable and sp2_stable):
                        continue
                    
                    dist = ensemble_distance(
                        info1["embedding"], info2["embedding"],
                        info1["phenotype"], info2["phenotype"],
                        state["config"].w_genotype, state["config"].w_phenotype
                    )
                    
                    if dist < state["config"].theta_merge:
                        merge_candidates.append((id1, id2, info1, info2))
            
            if not merge_candidates:
                break
            
            id1, id2, info1, info2 = merge_candidates[0]
            sp1 = info1["species"]
            sp2 = info2["species"]
            
            merged_species, _ = merge_islands(
                sp1, sp2, current_generation,
                state["config"].theta_sim,
                state["config"].w_genotype,
                state["config"].w_phenotype,
                state["logger"]
            )
            
            del state["species"][id1]
            del state["species"][id2]
            state["species"][merged_species.id] = merged_species
            
            sp1.species_state = "extinct"
            sp2.species_state = "extinct"
            state["historical_species"][id1] = sp1
            state["historical_species"][id2] = sp2
            
            del species_info[id1]
            del species_info[id2]
            species_info[merged_species.id] = {
                "species": merged_species,
                "leader": merged_species.leader,
                "embedding": merged_species.leader.embedding,
                "phenotype": merged_species.leader.phenotype,
                "created_at": merged_species.created_at
            }
            
            if "_genome_tracker" in state:
                state["_genome_tracker"].save()
            
            if speciation_state_path.exists():
                try:
                    with open(speciation_state_path, 'r', encoding='utf-8') as f:
                        existing_state = json.load(f)
                    
                    state_dict = {
                        "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                        "generation": current_generation
                    }
                    state_dict["cluster0"] = existing_state.get("cluster0", {})
                    state_dict["global_best_id"] = existing_state.get("global_best_id")
                    state_dict["metrics"] = existing_state.get("metrics", {})
                    state_dict["incubators"] = existing_state.get("incubators", [])
                    state_dict["extinct"] = existing_state.get("extinct", [])
                    
                    with open(speciation_state_path, 'w', encoding='utf-8') as f:
                        json.dump(state_dict, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    state["logger"].warning(f"Failed to update speciation_state.json after merge: {e}")
            
            if "_events_tracker" in state:
                for member in merged_species.members:
                    state["_events_tracker"].log(
                        member.id, "species_merged",
                        {"from_species": [id1, id2], "to_species": merged_species.id}
                    )
                state["_events_tracker"].save()
            
            merge_count += 1
            state["logger"].info(f"Merge {merge_count}: {id1}+{id2}->{merged_species.id} (immediate updates completed)")
        
        species_count_after_merge = len(state["species"])
        state["_current_gen_events"]["merge"] = merge_count
        
        expected_species_after_merge = species_count_before_merge - merge_count
        if species_count_after_merge != expected_species_after_merge:
            state["logger"].warning(f"Merge count mismatch: before={species_count_before_merge}, after={species_count_after_merge}, merges={merge_count}, expected_after={expected_species_after_merge}")
        
        for sid, sp in state["species"].items():
            if sp.cluster_origin == "merge" and sp.parent_ids:
                state["logger"].debug(f"Merge verification: species {sid} from {sp.parent_ids}, origin={sp.cluster_origin}")
    
    import numpy as np
    state["logger"].info("=== Phase 3: Step 7 - Radius Enforcement (after all merging) ===")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 3: Radius enforcement"):
        for sid, sp in list(state["species"].items()):
            if sp.leader is None or sp.leader.embedding is None:
                continue
            
            if "_genome_tracker" in state:
                species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            else:
                species_genome_ids = [m.id for m in sp.members]
            
            all_member_genomes = _load_genomes_by_ids(species_genome_ids, outputs_path, state["logger"])
            
            members_to_remove = []
            for genome in all_member_genomes:
                genome_id = genome.get("id")
                if genome_id == sp.leader.id:
                    continue
                
                genome_embedding = genome.get("prompt_embedding")
                if genome_embedding is None:
                    members_to_remove.append(genome_id)
                    continue
                
                from .phenotype_distance import extract_phenotype_vector
                genome_phenotype = extract_phenotype_vector(genome, logger=state["logger"])
                
                dist = ensemble_distance(
                    np.array(genome_embedding), sp.leader.embedding,
                    genome_phenotype, sp.leader.phenotype,
                    state["config"].w_genotype, state["config"].w_phenotype
                )
                
                if dist >= state["config"].theta_sim:
                    members_to_remove.append(genome_id)
            
            if members_to_remove:
                state["logger"].debug(f"Species {sid}: removing {len(members_to_remove)} members outside radius")
                for genome_id in members_to_remove:
                    state["_genome_tracker"].update_species_id(
                        str(genome_id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves_after_merge"
                    )
                    if "_events_tracker" in state:
                        state["_events_tracker"].log(
                            str(genome_id), "radius_enforcement_after_merge",
                            {"from_species": sid, "to_species": CLUSTER_0_ID, "reason": "outside_radius"}
                        )
                
                updated_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                sp.members = [m for m in sp.members if m.id in updated_member_ids]
                
                if len(sp.members) <= 1:
                    sp.species_state = "incubator"
                    sp.members = []
                    
                    if sp.leader and state["cluster0"].size < state["config"].cluster0_max_capacity:
                        state["cluster0"].add(sp.leader, current_generation)
                        if "_genome_tracker" in state:
                            state["_genome_tracker"].update_species_id(
                                str(sp.leader.id), CLUSTER_0_ID, current_generation, "radius_enforcement_leader_to_reserves_after_merge"
                            )
                            if "_events_tracker" in state:
                                state["_events_tracker"].log(
                                    str(sp.leader.id), "radius_enforcement_after_merge",
                                    {"from_species": sid, "to_species": CLUSTER_0_ID, "reason": "species_empty_after_radius_cleanup"}
                                )
                    
                    state["logger"].info(f"Phase 3: Species {sid} became empty after radius cleanup - marked as incubator (will be processed in Phase 5)")
                elif len(sp.members) < state["config"].min_island_size:
                    state["logger"].debug(f"Phase 3: Species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in Phase 5 Step 18")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 3: Save tracker"):
        _save_tracker_if_dirty(state)
        if "_events_tracker" in state:
            state["_events_tracker"].save()
        if speciation_state_path.exists():
            try:
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
                
                state_dict = {
                    "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                    "generation": current_generation
                }
                state_dict["cluster0"] = existing_state.get("cluster0", {})
                state_dict["global_best_id"] = existing_state.get("global_best_id")
                state_dict["metrics"] = existing_state.get("metrics", {})
                state_dict["incubators"] = existing_state.get("incubators", [])
                state_dict["extinct"] = existing_state.get("extinct", [])
                
                with open(speciation_state_path, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                state["logger"].warning(f"Failed to update speciation_state.json after radius enforcement: {e}")
        
        _save_tracker_if_dirty(state)
        _validate_tracker_consistency(state, "Phase 3")
    state["logger"].info("Phase 3 completed in %.2fs", _time.time() - _phase3_start)
    
    
    _phase4_start = _time.time()
    state["logger"].info("=== Phase 4: Capacity Enforcement (species_id > 0) ===")
    
    
    all_phase4_species_ids = set(state["species"].keys())
    
    if "_genome_tracker" not in state:
        state["logger"].error("Genome tracker not available for capacity enforcement")
    else:
        tracker_stats = state["_genome_tracker"].get_distribution_stats()
        tracker_species_ids = {int(sid) for sid, count in tracker_stats["by_species_id"].items() 
                               if int(sid) > 0 and count > 0}
        
        in_memory_species_ids = {int(sid) for sid in state["species"].keys()}
        all_phase4_species_ids = tracker_species_ids | in_memory_species_ids
        
        tracker_only_species = tracker_species_ids - in_memory_species_ids
        if tracker_only_species:
            state["logger"].info(
                f"Phase 4: Found {len(tracker_only_species)} species in tracker but not in state['species']: {sorted(tracker_only_species)}"
            )
    
    outputs_path = get_outputs_path()
    
    with PerformanceLogger(state["logger"], "Speciation Phase 4: Capacity enforcement (per species)"):
        for sid in sorted(all_phase4_species_ids):
            sp = state["species"].get(sid) or state["species"].get(str(sid))
            
            if sp is None:
                state["logger"].debug(f"Phase 4: Processing tracker-only species {sid} (not in state['species'])")
            
            all_species_genomes = []
            
            if "_genome_tracker" not in state:
                state["logger"].error("Genome tracker not available for capacity enforcement")
                continue
            
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            species_genome_ids_set = set(species_genome_ids)
            
            in_memory_ids = set()
            if sp is not None:
                in_memory_ids = {str(m.id) for m in sp.members}
                
                for member in sp.members:
                    member_id_str = str(member.id)
                    if member_id_str not in species_genome_ids_set:
                        state["_genome_tracker"].register(member_id_str, sid, current_generation)
                        species_genome_ids_set.add(member_id_str)
            
            all_genome_ids = list(species_genome_ids_set | in_memory_ids)
            loaded_genomes = _load_genomes_by_ids(all_genome_ids, outputs_path, state["logger"])
            
            loaded_by_id = {str(g.get("id")): g for g in loaded_genomes if g.get("id") is not None}
            
            all_species_genomes.extend(loaded_genomes)
            
            if sp is not None:
                for member in sp.members:
                    member_id_str = str(member.id)
                    if member_id_str not in loaded_by_id:
                        genome = _individual_to_genome_dict(member, current_generation)
                        genome["species_id"] = sid
                        all_species_genomes.append(genome)
            
            if len(all_species_genomes) < len(all_genome_ids):
                missing = set(all_genome_ids) - {str(g.get("id")) for g in all_species_genomes if g.get("id")}
                if missing:
                    state["logger"].warning(
                        f"Phase 4: Species {sid} - {len(missing)} genomes from tracker not found in files: {list(missing)[:5]}"
                    )
            
            from utils.population_io import _extract_north_star_score
            valid_genomes = []
            invalid_genomes = []
            for g in all_species_genomes:
                fitness = _extract_north_star_score(g, "toxicity")
                if fitness is not None:
                    valid_genomes.append((g, fitness))
                else:
                    invalid_genomes.append(g)
                    state["logger"].warning(f"Phase 4: Genome {g.get('id')} in species {sid} has no valid fitness score, excluding from capacity enforcement")
            
            valid_genomes.sort(key=lambda x: x[1], reverse=True)
            all_species_genomes = [g for g, _ in valid_genomes] + invalid_genomes
            
            if sp is not None:
                state["logger"].debug(
                    f"Phase 4: Species {sid} (state={sp.species_state}): "
                    f"tracker={len(species_genome_ids)}, loaded={len(loaded_genomes)}, "
                    f"in_memory={len(sp.members)}, total={len(all_species_genomes)}, capacity={state['config'].species_capacity}"
                )
            else:
                state["logger"].debug(
                    f"Phase 4: Species {sid} (tracker-only, not in state['species']): "
                    f"tracker={len(species_genome_ids)}, loaded={len(loaded_genomes)}, "
                    f"total={len(all_species_genomes)}, capacity={state['config'].species_capacity}"
                )
            
            if len(all_species_genomes) > state["config"].species_capacity:
                state["logger"].info(
                    f"Phase 4: Species {sid} exceeds capacity: {len(all_species_genomes)} genomes "
                    f"(capacity: {state['config'].species_capacity}), will archive {len(all_species_genomes) - state['config'].species_capacity}"
                    + (f" (tracker-only species)" if sp is None else "")
                )
                keep_genomes = all_species_genomes[:state["config"].species_capacity]
                excess_genomes = all_species_genomes[state["config"].species_capacity:]
                
                if sp is not None:
                    keep_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
                    sp.members = [m for m in sp.members if m.id in keep_ids]
                    
                    for genome in keep_genomes:
                        gid = genome.get("id")
                        if gid and not any(m.id == gid for m in sp.members):
                            from .species import Individual
                            ind = Individual.from_genome(genome)
                            sp.members.append(ind)
                    
                    if sp.members:
                        if sp.cluster_origin == "merge":
                            current_leader_fitness = sp.leader.fitness if sp.leader else float('-inf')
                            highest_fitness_member = max(sp.members, key=lambda x: x.fitness)
                            if highest_fitness_member.fitness > current_leader_fitness:
                                state["logger"].debug(f"Phase 4: Merged species {sid} leader updated: {sp.leader.id} -> {highest_fitness_member.id} (new higher fitness genome)")
                                sp.leader = highest_fitness_member
                        else:
                            new_leader = max(sp.members, key=lambda x: x.fitness)
                            if new_leader != sp.leader:
                                state["logger"].debug(f"Phase 4: Species {sid} leader updated: {sp.leader.id if sp.leader else None} -> {new_leader.id} (after capacity enforcement)")
                            sp.leader = new_leader
                        
                        if sp.leader not in sp.members:
                            sp.members.insert(0, sp.leader)
                
                if "_genome_tracker" in state:
                    excess_ids = [str(g.get("id")) for g in excess_genomes if g.get("id") is not None]
                    if excess_ids:
                        before_count = len(state["_genome_tracker"].get_all_genomes_by_species(sid))
                        
                        updates = {gid: -1 for gid in excess_ids}
                        result = state["_genome_tracker"].batch_update(updates, current_generation, f"capacity_archived_species_{sid}")
                        
                        after_count = len(state["_genome_tracker"].get_all_genomes_by_species(sid))
                        expected_after = before_count - len(excess_ids)
                        
                        if result["failed"] > 0:
                            state["logger"].error(
                                f"CRITICAL: Genome tracker batch update failed for {result['failed']}/{result['total']} "
                                f"genomes during capacity enforcement for species {sid}"
                            )
                            state["logger"].error(f"Failed genome IDs: {result.get('failed_genome_ids', [])[:10]}")
                            for failed_id in result.get('failed_genome_ids', []):
                                try:
                                    success, _ = state["_genome_tracker"].update_species_id(
                                        failed_id, -1, current_generation, f"capacity_archived_species_{sid}_retry"
                                    )
                                    if not success:
                                        state["logger"].error(f"Retry failed for genome {failed_id}")
                                except Exception as e:
                                    state["logger"].error(f"Retry exception for genome {failed_id}: {e}")
                        
                        if after_count != expected_after:
                            state["logger"].error(
                                f"CRITICAL: Tracker update verification failed for species {sid}: "
                                f"expected {expected_after} genomes after archiving, got {after_count}. "
                                f"Before: {before_count}, Excess: {len(excess_ids)}"
                            )
                            still_in_species = [
                                gid for gid in excess_ids 
                                if state["_genome_tracker"].get_species_id(gid) == sid
                            ]
                            if still_in_species:
                                state["logger"].error(f"  Excess genomes still in species {sid}: {still_in_species[:10]}")
                        else:
                            state["logger"].info(
                                f"Phase 4: Species {sid} - archived {len(excess_ids)} genomes "
                                f"(tracker: {before_count} -> {after_count})"
                            )
                        
                        if state["_genome_tracker"]._dirty:
                            save_success = state["_genome_tracker"].save()
                            if not save_success:
                                state["logger"].error(f"CRITICAL: Failed to save tracker after capacity enforcement for species {sid}")
                            else:
                                state["logger"].debug(f"Phase 4: Saved tracker after archiving {len(excess_ids)} genomes for species {sid}")
                    else:
                        state["logger"].warning(f"Phase 4: No valid genome IDs found in excess_genomes for species {sid}")
                
                excess_individuals = []
                for genome in excess_genomes:
                    try:
                        from .species import Individual
                        ind = Individual.from_genome(genome)
                        if ind and ind.id:
                            excess_individuals.append(ind)
                        else:
                            state["logger"].warning(f"Failed to create Individual from genome id={genome.get('id')} in species {sid}")
                    except Exception as e:
                        state["logger"].error(f"Error creating Individual from genome id={genome.get('id')} in species {sid}: {e}")
                
                if excess_individuals:
                    _archive_individuals(excess_individuals, current_generation, "species_capacity_exceeded")
                
                if "_events_tracker" in state:
                    for ind in excess_individuals:
                        state["_events_tracker"].log(
                            ind.id, "capacity_archived",
                            {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity}
                        )
                
                
                state["logger"].info(f"Phase 4: Species {sid} capacity enforced ({state['config'].species_capacity}), archived {len(excess_genomes)} excess genomes from {len(all_species_genomes)} total (all generations)")
            else:
                state["logger"].debug(f"Phase 4: Species {sid} within capacity ({len(all_species_genomes)}/{state['config'].species_capacity})")
    
    leader_ids = {}
    duplicate_leaders = []
    for sid, sp in state["species"].items():
        if sp.leader:
            leader_id = sp.leader.id
            if leader_id in leader_ids:
                duplicate_leaders.append((leader_id, [leader_ids[leader_id], sid]))
            else:
                leader_ids[leader_id] = sid
    
    if duplicate_leaders:
        for leader_id, species_ids in duplicate_leaders:
            state["logger"].warning(f"Duplicate leader ID {leader_id} found in species {species_ids}, fixing...")
            for sid in species_ids[1:]:
                if sid not in state["species"]:
                    continue
                sp = state["species"][sid]
                old_leader = None
                for member in sp.members:
                    if member.id == leader_id:
                        old_leader = member
                        break
                
                if old_leader:
                    sp.members.remove(old_leader)
                    old_leader.species_id = None
                    if "_genome_tracker" in state:
                        if old_leader.species_id is not None:
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), old_leader.species_id, current_generation, "duplicate_leader_fix"
                            )
                        else:
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), CLUSTER_0_ID, current_generation, "duplicate_leader_fix_to_reserves"
                            )
                
                if len(sp.members) > 0:
                    sp.leader = max(sp.members, key=lambda x: x.fitness)
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
                    state["logger"].info(f"Reassigned species {sid} leader to genome {sp.leader.id} (fitness={sp.leader.fitness:.4f})")
                else:
                    sp.species_state = "incubator"
                    sp.leader = None
                    state["logger"].info(f"Species {sid} has no other members, marking as incubator (will be processed in Phase 5 Step 18)")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 4: Save tracker and validate"):
        _save_tracker_if_dirty(state)
        _validate_tracker_consistency(state, "Phase 4")
    
    for sid, sp in state["species"].items():
        if "_genome_tracker" in state:
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            if len(species_genome_ids) > state["config"].species_capacity:
                state["logger"].error(
                    f"CRITICAL: After Phase 4, species {sid} still has {len(species_genome_ids)} genomes "
                    f"(exceeds capacity {state['config'].species_capacity})"
                )
                state["logger"].error(f"  Genomes still in species {sid}: {species_genome_ids[:10]}")
                archived_count = len([
                    gid for gid, data in state["_genome_tracker"].genomes.items()
                    if data.get("species_id") == -1
                ])
                state["logger"].error(f"  Total archived genomes in tracker: {archived_count}")
    
    state["logger"].info("Phase 4 completed in %.2fs", _time.time() - _phase4_start)
    
    
    _phase5_start = _time.time()
    state["logger"].info("=== Phase 5: Stagnation and Incubation ===")
    
    selected_species_ids = set()
    if current_generation > 0:
        try:
            outputs_path = get_outputs_path()
            parents_path = outputs_path / "parents.json"
            if parents_path.exists():
                with open(parents_path, 'r', encoding='utf-8') as f:
                    parents = json.load(f)
                if isinstance(parents, list):
                    for parent in parents:
                        species_id = parent.get("species_id")
                        if species_id is not None and species_id != 0:
                            selected_species_ids.add(int(species_id))
                    state["logger"].debug(f"Loaded {len(selected_species_ids)} species from parents.json: {sorted(selected_species_ids)}")
                if len(parents) > 0 and len(selected_species_ids) == 0:
                    sid_vals = [p.get("species_id") for p in parents]
                    state["logger"].info(
                        "Stagnation: selected_species_ids is empty (all parents species_id in %s). "
                        "Stagnation only increments when a non-reserve species is selected and does not improve.",
                        sid_vals
                    )
        except Exception as e:
            state["logger"].warning(f"Failed to load parents.json to determine selected species: {e}")
    
    for sid, sp in state["species"].items():
        sp.max_fitness = max((m.fitness for m in sp.members), default=0.0)
        sid_int = int(sid)
        was_selected = sid_int in selected_species_ids
        prev_max = state.get("_prev_max_fitness", {}).get(sid_int, -1)
        max_fitness_increased = sp.max_fitness > prev_max
        prev_stagnation = sp.stagnation
        sp.record_fitness(current_generation, was_selected_as_parent=was_selected, max_fitness_increased=max_fitness_increased)
        if sp.stagnation != prev_stagnation:
            state["logger"].info(
                "Stagnation changed: species %s %d -> %d (was_selected=%s, max_fitness_increased=%s)",
                sid, prev_stagnation, sp.stagnation, was_selected, max_fitness_increased
            )
    
    stagnation_threshold = state["config"].species_stagnation
    for sid, sp in state["species"].items():
        state["logger"].info(
            "Stagnation: species %s state=%s stagnation=%d (threshold=%d) size=%d",
            sid, sp.species_state, sp.stagnation, stagnation_threshold, sp.size
        )
    
    state["logger"].info(
        "=== Phase 5: Step 10 - Freeze Stagnant Species (threshold=%d) ===",
        state["config"].species_stagnation
    )
    
    frozen_count = 0
    already_frozen_count = 0
    at_threshold_count = 0
    for sid, sp in list(state["species"].items()):
        at_threshold = sp.stagnation >= state["config"].species_stagnation
        if at_threshold:
            at_threshold_count += 1
        if sp.species_state == "frozen":
            already_frozen_count += 1
            state["logger"].debug(
                "Freeze skip: species %s already frozen (stagnation=%d)",
                sid, sp.stagnation
            )
            continue
        if at_threshold:
            sp.species_state = "frozen"
            frozen_count += 1
            state["_current_gen_events"]["extinction"] += 1
            state["logger"].info(
                "Frozen species %s (stagnation=%d >= %d) - excluded from parent selection, still alive and can merge",
                sid, sp.stagnation, state["config"].species_stagnation
            )
        else:
            state["logger"].debug(
                "Freeze skip: species %s stagnation=%d < threshold=%d",
                sid, sp.stagnation, state["config"].species_stagnation
            )
    
    state["logger"].info(
        "Step 10: Summary - species at stagnation>=threshold: %d; frozen this step: %d; already frozen: %d; extinction_events total: %d",
        at_threshold_count, frozen_count, already_frozen_count, state["_current_gen_events"]["extinction"]
    )
    
    if frozen_count > 0:
        _save_tracker_if_dirty(state)
        state["logger"].info(f"Step 10: Frozen {frozen_count} species (trackers updated)")
    
    state["logger"].info("=== Phase 5: Step 11 - Incubate Small Species ===")
    
    species_count_before_incubation = len(state["species"])
    cluster0_ids_before = {cm.individual.id for cm in state["cluster0"].members}
    incubator_species = {}
    moved_to_cluster0_events = []
    
    for sid, sp in list(state["species"].items()):
        if sp.species_state not in ["active", "frozen", "incubator"]:
            continue
        
        current_size = sp.size
        is_newly_merged = (sp.cluster_origin == "merge" and sp.created_at == current_generation)
        
        if sp.species_state == "incubator" or current_size < state["config"].min_island_size:
            if is_newly_merged and current_size < state["config"].min_island_size:
                state["logger"].info(f"Phase 5: Newly merged species {sid} has {current_size} members < min_island_size={state['config'].min_island_size} - dissolving to incubator")
            if state["cluster0"].size >= state["config"].cluster0_max_capacity:
                state["logger"].debug(f"Cluster 0 at capacity, cannot incubate species {sid}")
                continue
            
            moved_member_ids = []
            member_ids_set = {m.id for m in sp.members}
            
            if sp.leader and sp.leader.id not in member_ids_set:
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                    moved_member_ids.append(sp.leader.id)
            
            for member in sp.members:
                if state["cluster0"].size >= state["config"].cluster0_max_capacity:
                    break
                state["cluster0"].add(member, current_generation)
                moved_member_ids.append(member.id)
            
            if "_genome_tracker" in state:
                sid = int(sid)
                genomes_to_update = [gid for gid, gdata in state["_genome_tracker"].genomes.items() 
                                    if gdata.get("species_id") == sid]
                all_ids = set(moved_member_ids) | {sp.leader.id} if sp.leader else set() | {m.id for m in sp.members}
                genomes_to_update.extend(str(mid) for mid in all_ids if str(mid) not in genomes_to_update)
                
                if genomes_to_update:
                    result = state["_genome_tracker"].batch_update(
                        {gid: 0 for gid in genomes_to_update}, current_generation, f"incubated_species_{sid}_to_reserves"
                    )
                    if result["failed"] > 0:
                        state["logger"].warning(f"Tracker update failed for {result['failed']} genomes (species {sid})")
            
            sp.species_state = "incubator"
            sp.members = []
            incubator_species[sid] = sp
            del state["species"][sid]
            
            moved_to_cluster0_events.append({
                "generation": current_generation,
                "species_id": sid,
                "action": "incubated",
                "size": current_size,
                "moved_count": len(moved_member_ids),
                "moved_member_ids": moved_member_ids
            })
            
            state["logger"].info(
                f"Incubated species {sid} ({len(moved_member_ids)} members moved to cluster 0) - "
                f"extinction/dissolution"
            )
    
    for sid, sp in incubator_species.items():
        state["historical_species"][sid] = sp
        state["logger"].debug(f"Moved incubator species {sid} to historical_species (will be tracked by ID only in save_state)")
    
    if moved_to_cluster0_events:
        _save_tracker_if_dirty(state)
        state["_current_gen_events"]["moved_to_cluster0"] = len(moved_to_cluster0_events)
        state["logger"].info(f"Step 11: Incubated {len(moved_to_cluster0_events)} species (trackers updated)")
    
    species_count_after_incubation = len(state["species"])
    expected_species_after = species_count_before_incubation - len(moved_to_cluster0_events)
    if species_count_after_incubation != expected_species_after:
        state["logger"].warning(
            f"Incubation count mismatch: before={species_count_before_incubation}, "
            f"after={species_count_after_incubation}, incubated={len(moved_to_cluster0_events)}, "
            f"expected_after={expected_species_after}"
        )
    
    _save_tracker_if_dirty(state)
    _validate_tracker_consistency(state, "Phase 5")
    _validate_species_accounting(state, "Phase 5")
    state["logger"].info("Phase 5 completed in %.2fs", _time.time() - _phase5_start)
    
    
    _phase6_start = _time.time()
    state["logger"].info("=== Phase 6: Cluster 0 Capacity Enforcement (species_id = 0) ===")
    
    state["logger"].info("=== Phase 6: Step 12 - Cluster 0 Capacity Enforcement ===")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 6: Cluster 0 capacity enforcement"):
        if "_genome_tracker" not in state:
            state["logger"].error("Genome tracker not available for cluster 0 capacity enforcement")
        else:
            cluster0_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(CLUSTER_0_ID)
            cluster0_count = len(cluster0_genome_ids)
            
            if cluster0_count > state["config"].cluster0_max_capacity:
                state["logger"].info(
                    f"Cluster 0 capacity exceeded: {cluster0_count} genomes (capacity: {state['config'].cluster0_max_capacity})"
                )
                
                outputs_path = get_outputs_path()
                cluster0_genomes = _load_genomes_by_ids(cluster0_genome_ids, outputs_path, state["logger"])
                
                from utils.population_io import _extract_north_star_score

                capacity = state["config"].cluster0_max_capacity

                if state["config"].cluster0_selection == "nsga2":
                    import numpy as _np
                    from .reserve_selection import select_reserves_nsga2
                    from .distance import ensemble_distances_batch as _edb
                    from .phenotype_distance import extract_phenotype_vector as _epv

                    tox_vals = _np.array(
                        [_extract_north_star_score(g, "toxicity") for g in cluster0_genomes],
                        dtype=_np.float64,
                    )

                    leaders = _load_species_leaders_from_state(outputs_path, state["logger"])
                    if leaders:
                        leader_embs = _np.array([emb for _, emb, _ in leaders.values()], dtype=_np.float32)
                        leader_phenos = [pheno for _, _, pheno in leaders.values()]
                        w_g = state["config"].w_genotype
                        w_p = state["config"].w_phenotype

                        div_vals = _np.zeros(len(cluster0_genomes), dtype=_np.float64)
                        for idx, g in enumerate(cluster0_genomes):
                            emb_raw = g.get("prompt_embedding")
                            if emb_raw is None:
                                div_vals[idx] = 0.0
                                continue
                            emb = _np.array(emb_raw, dtype=_np.float32)
                            n = _np.linalg.norm(emb)
                            if n > 1e-9:
                                emb = emb / n
                            try:
                                g_pheno = _epv(g, logger=state["logger"])
                            except Exception:
                                g_pheno = None
                            dists = _edb(emb, leader_embs, g_pheno, leader_phenos, w_g, w_p)
                            div_vals[idx] = float(_np.mean(dists))
                    else:
                        state["logger"].info("No species leaders found; diversity set to 0 for all cluster 0 genomes")
                        div_vals = _np.zeros(len(cluster0_genomes), dtype=_np.float64)

                    keep_genomes, excess_genomes = select_reserves_nsga2(
                        cluster0_genomes, tox_vals, div_vals, capacity,
                    )
                    state["logger"].info(
                        f"Cluster 0 NSGA-II selection: kept {len(keep_genomes)}, archived {len(excess_genomes)} "
                        f"(leaders={len(leaders) if leaders else 0})"
                    )
                else:
                    cluster0_genomes.sort(
                        key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True,
                    )
                    keep_genomes = cluster0_genomes[:capacity]
                    excess_genomes = cluster0_genomes[capacity:]
                
                if excess_genomes:
                    excess_ids = [str(g.get("id")) for g in excess_genomes if g.get("id") is not None]
                    if excess_ids:
                        updates = {gid: -1 for gid in excess_ids}
                        result = state["_genome_tracker"].batch_update(
                            updates, current_generation, "cluster0_capacity_archived"
                        )
                        if result["failed"] > 0:
                            state["logger"].warning(
                                f"Genome tracker batch update had {result['failed']} failures "
                                f"during cluster 0 capacity enforcement"
                            )
                
                excess_individuals = []
                for genome in excess_genomes:
                    try:
                        from .species import Individual
                        ind = Individual.from_genome(genome)
                        if ind and ind.id:
                            excess_individuals.append(ind)
                    except Exception as e:
                        state["logger"].warning(f"Failed to create Individual from genome id={genome.get('id')}: {e}")
                
                if excess_individuals:
                    _archive_individuals(excess_individuals, current_generation, "cluster0_capacity_exceeded")
                
                if "_events_tracker" in state:
                    for ind in excess_individuals:
                        state["_events_tracker"].log(
                            ind.id, "capacity_archived",
                            {"reason": "cluster0_capacity", "capacity": state["config"].cluster0_max_capacity}
                        )
                
                state["logger"].info(
                    f"Cluster 0 capacity enforced: archived {len(excess_genomes)} excess genomes "
                    f"(tracker: {cluster0_count} -> {len(keep_genomes)}, capacity: {state['config'].cluster0_max_capacity})"
                )
                
                kept_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
                excess_ids_set = {g.get("id") for g in excess_genomes if g.get("id") is not None}
                state["cluster0"].members = [
                    m for m in state["cluster0"].members 
                    if m.individual.id in kept_ids or m.individual.id not in excess_ids_set
                ]
                
                _save_tracker_if_dirty(state)
                if "_events_tracker" in state:
                    state["_events_tracker"].save()
                state["logger"].info(f"Step 19: Cluster 0 capacity enforced, trackers updated immediately")
            else:
                state["logger"].debug(
                    f"Cluster 0 within capacity: {cluster0_count}/{state['config'].cluster0_max_capacity}"
                )
    
    with PerformanceLogger(state["logger"], "Speciation Phase 6: Save tracker"):
        _save_tracker_if_dirty(state)
        _validate_tracker_consistency(state, "Phase 6")
    state["logger"].info("Phase 6 completed in %.2fs", _time.time() - _phase6_start)
    
    
    _phase7_start = _time.time()
    state["logger"].info("=== Phase 7: Redistribution of Genomes ===")
    with PerformanceLogger(state["logger"], "Speciation Phase 7: Redistribution"):
        distribution_result = phase8_redistribute_genomes(
            temp_path=temp_path if temp_path else None,
            current_generation=current_generation
        )
    state["logger"].info(f"Phase 7: Distribution complete - {distribution_result.get('elites_moved', 0)} elites, {distribution_result.get('reserves_moved', 0)} reserves, {distribution_result.get('archived_moved', 0)} archived")
    
    outputs_path = get_outputs_path()
    _update_speciation_state_cluster0_size_after_distribution(outputs_path)
    
    reserves_path = outputs_path / "reserves.json"
    if reserves_path.exists() and "_genome_tracker" in state:
        try:
            reserves_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(0)
            
            reserves_genomes = _load_genomes_by_ids(reserves_genome_ids, outputs_path, state["logger"])
            
            reserves_ids_set = {str(g.get("id")) for g in reserves_genomes if g.get("id") is not None}
            state["cluster0"].members = [
                cm for cm in state["cluster0"].members 
                if str(cm.individual.id) in reserves_ids_set
            ]
            
            existing_cluster0_ids = {str(cm.individual.id) for cm in state["cluster0"].members}
            added_count = 0
            for genome in reserves_genomes:
                genome_id = genome.get("id")
                if genome_id is None:
                    continue
                genome_id_str = str(genome_id)
                
                if genome_id_str not in existing_cluster0_ids:
                    if genome.get("prompt_embedding"):
                        try:
                            from .species import Individual
                            ind = Individual.from_genome(genome)
                            if ind.embedding is not None:
                                state["cluster0"].add(ind, current_generation)
                                added_count += 1
                        except Exception as e:
                            state["logger"].debug(f"Failed to add genome {genome_id} to cluster0: {e}")
            
            if added_count > 0 or len(state["cluster0"].members) != len(reserves_genomes):
                state["logger"].debug(
                    f"Synced in-memory cluster0: {len(state['cluster0'].members)} members "
                    f"(reserves.json has {len(reserves_genomes)}, added {added_count})"
                )
        except Exception as e:
            state["logger"].warning(f"Failed to sync in-memory cluster0 with reserves.json: {e}")
    
    if "_genome_tracker" in state:
        elites_path = outputs_path / "elites.json"
        for sid, sp in list(state["species"].items()):
            tracker_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            current_member_ids = {str(m.id) for m in sp.members}
            tracker_set = set(str(mid) for mid in tracker_member_ids)
            if len(tracker_member_ids) != len(sp.members) or tracker_set != current_member_ids:
                loaded_genomes = _load_genomes_by_ids(tracker_member_ids, outputs_path, state["logger"]) if elites_path.exists() else []
                loaded_by_id = {str(g.get("id")): g for g in loaded_genomes if g.get("id") is not None}
                new_members = []
                for mid in tracker_member_ids:
                    mid_str = str(mid)
                    existing = next((m for m in sp.members if str(m.id) == mid_str), None)
                    if existing is not None:
                        new_members.append(existing)
                    elif mid_str in loaded_by_id:
                        try:
                            ind = Individual.from_genome(loaded_by_id[mid_str])
                            ind.species_id = sid
                            new_members.append(ind)
                        except Exception as e:
                            state["logger"].debug(f"Failed to create Individual for genome {mid} in species {sid}: {e}")
                if new_members:
                    best = max(new_members, key=lambda x: x.fitness)
                    sp.leader = best
                    sp.members = [best] + [m for m in new_members if m.id != best.id]
                else:
                    sp.members = []
                state["logger"].debug(
                    f"Phase 7: Synced species {sid} members from tracker: {len(sp.members)} (was {len(current_member_ids)})"
                )
    
    state["logger"].info("Phase 7 completed in %.2fs", _time.time() - _phase7_start)
    
    
    _phase8_start = _time.time()
    state["logger"].info("=== Phase 8: Metrics & Statistics ===")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 8: Update labels and record metrics"):
        from .labeling import update_species_labels
        update_species_labels(
            state["species"],
            current_generation=current_generation,
            n_words=10,
            logger=state["logger"]
        )
        
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        reserves_path = outputs_path / "reserves.json"
        
        if not elites_path.exists():
            raise FileNotFoundError(f"elites.json not found at {elites_path} - required for metrics calculation (should have been created in Phase 7)")
        if not reserves_path.exists():
            state["logger"].warning(f"reserves.json not found at {reserves_path} - using cluster0.size")
        
        metrics = state["metrics_tracker"].record_generation(
            generation=current_generation,
            species=state["species"],
            reserves_size=state["cluster0"].size,
            speciation_events=state["_current_gen_events"]["speciation"],
            merge_events=state["_current_gen_events"]["merge"],
            extinction_events=state["_current_gen_events"]["extinction"],
            cluster0=state["cluster0"],
            elites_path=str(elites_path),
            reserves_path=str(reserves_path) if reserves_path.exists() else None
        )
        
        state["logger"].debug(f"Recorded metrics for generation {current_generation} from corrected files")
        
        is_valid, errors = validate_metrics_from_files(
            outputs_path=outputs_path,
            metrics=metrics.to_dict(),
            logger=state["logger"]
        )
        if not is_valid:
            state["logger"].warning(f"Metrics validation found {len(errors)} errors")
            for error in errors[:5]:
                state["logger"].warning(f"  - {error}")
        else:
            state["logger"].debug("Metrics validation passed - all metrics match file contents")
    
    with PerformanceLogger(state["logger"], "Speciation Phase 8: Save state files"):
        save_state(str(get_outputs_path() / "speciation_state.json"))
        if "_events_tracker" in state:
            state["_events_tracker"].save()
        if "_genome_tracker" in state:
            state["_genome_tracker"].save()
    
    state["logger"].info("Phase 8 completed in %.2fs", _time.time() - _phase8_start)
    
    process_gen_elapsed = _time.time() - _process_gen_start
    state["logger"].info("COMPLETED: Speciation generation %d in %.3f seconds", current_generation, process_gen_elapsed)
    state["logger"].info("Speciation generation %d complete in %.2fs", current_generation, process_gen_elapsed)
    return state["species"], state["cluster0"]


def cluster0_speciation_isolated(
    current_generation: int, 
    config: "SpeciationConfig", 
    logger=None,
    pre_sorted_individuals: Optional[List[Individual]] = None) -> List[Species]:
    
    from .species import Species, Individual, generate_species_id
    from .distance import ensemble_distance, ensemble_distances_batch
    import numpy as np
    
    if logger is None:
        logger = get_logger("Cluster0SpeciationIsolated")
    
    if pre_sorted_individuals is not None:
        individuals = [ind for ind in pre_sorted_individuals if getattr(ind, "embedding", None) is not None]
        sorted_individuals = individuals
    else:
        state = _get_state()
        cluster0 = state.get("cluster0")
        if cluster0 is None:
            logger.debug("cluster0 not in state, no speciation possible")
            return []
        individuals = [ind for ind in cluster0.individuals if getattr(ind, "embedding", None) is not None]
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    
    if len(sorted_individuals) < config.cluster0_min_cluster_size:
        logger.debug(f"Cluster 0 has {len(sorted_individuals)} individuals with embeddings, need {config.cluster0_min_cluster_size} to attempt speciation")
        return []
    
    potential_leaders: Dict[int, Tuple[None, np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    first = sorted_individuals[0]
    potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
    remaining_individuals = sorted_individuals[1:]
    
    for ind in remaining_individuals:
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        if potential_leaders:
            leader_embeddings = []
            leader_phenotypes = []
            leader_ids = []
            for pl_id, (_, pl_emb, pl_pheno, _, _) in potential_leaders.items():
                leader_ids.append(pl_id)
                leader_embeddings.append(pl_emb)
                leader_phenotypes.append(pl_pheno)
            
            if len(leader_embeddings) > 1:
                leader_embeddings_array = np.array(leader_embeddings)
                distances = ensemble_distances_batch(
                    ind.embedding, leader_embeddings_array,
                    ind.phenotype, leader_phenotypes,
                    config.w_genotype, config.w_phenotype
                )
                min_idx = np.argmin(distances)
                min_dist = distances[min_idx]
                nearest_leader_id = leader_ids[min_idx]
            elif len(leader_embeddings) == 1:
                min_dist = ensemble_distance(
                    ind.embedding, leader_embeddings[0],
                    ind.phenotype, leader_phenotypes[0],
                    config.w_genotype, config.w_phenotype
                )
                nearest_leader_id = leader_ids[0]
            
            if nearest_leader_id is not None and min_dist < config.theta_sim:
                _, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                followers.append(ind)
                assigned = True
        
        if not assigned:
            potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
    
    new_species_list: List[Species] = []
    individuals_to_remove: List[Individual] = []
    
    for pl_id, (_, pl_emb, pl_pheno, pl_ind, followers) in potential_leaders.items():
        all_members = [pl_ind] + followers
        
        if len(all_members) >= config.min_island_size:
            new_species_id = generate_species_id()
            new_species = Species(
                id=new_species_id,
                leader=pl_ind,
                members=all_members,
                radius=config.theta_sim,
                created_at=current_generation,
                last_improvement=current_generation,
                cluster_origin="natural",
                parent_ids=None,
                leader_distance=0.0
            )
            new_species_list.append(new_species)
            individuals_to_remove.extend(all_members)
            logger.info(
                f"Cluster 0 speciation: Created species {new_species.id} from {len(all_members)} "
                f"individuals (leader={pl_ind.id}, followers={len(followers)})"
            )
        else:
            logger.debug(
                f"Cluster 0 speciation: group with {len(all_members)} members < "
                f"min_island_size {config.min_island_size} → staying in cluster 0"
            )
    
    state = _get_state()
    if individuals_to_remove and state.get("cluster0"):
        removed_count = state["cluster0"].remove_batch(individuals_to_remove)
        logger.debug(f"Removed {removed_count} individuals from in-memory cluster 0 (formed {len(new_species_list)} new species)")
    
    logger.info(f"Cluster 0 speciation isolated: formed {len(new_species_list)} new species from {len(individuals)} cluster 0 individuals")
    return new_species_list


def _individual_to_genome_dict(ind: Individual, current_generation: int) -> Dict[str, Any]:
    
    import numpy as np
    
    if ind.genome_data:
        genome = ind.genome_data.copy()
        if "generation" not in genome:
            genome["generation"] = current_generation
    else:
        genome = {
            "id": ind.id,
            "prompt": ind.prompt,
            "generation": current_generation
        }
    
    genome["species_id"] = ind.species_id
    genome["fitness"] = ind.fitness
    
    if ind.embedding is not None:
        if isinstance(ind.embedding, np.ndarray):
            genome["prompt_embedding"] = ind.embedding.tolist()
        else:
            genome["prompt_embedding"] = ind.embedding
    
    if ind.genome_data and "moderation_result" in ind.genome_data:
        genome["moderation_result"] = ind.genome_data["moderation_result"]
    
    return genome


def _update_speciation_state_cluster0_size_after_distribution(outputs_path) -> None:
    
    outputs_path = Path(outputs_path)
    state_path = outputs_path / "speciation_state.json"
    reserves_path = outputs_path / "reserves.json"
    if not state_path.exists() or not reserves_path.exists():
        return
    try:
        with open(reserves_path, 'r', encoding='utf-8') as f:
            n = len(json.load(f))
        with open(state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)
        cluster0 = state.get("cluster0")
        if isinstance(cluster0, dict):
            cluster0["size"] = n
        if "cluster0_size_from_reserves" in state:
            state["cluster0_size_from_reserves"] = n
        with open(state_path, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        try:
            _log = _get_state().get("logger")
            if _log:
                _log.debug(f"Updated speciation_state cluster0 size to {n} (from reserves.json)")
        except Exception:
            pass
    except Exception as e:
        try:
            _log = _get_state().get("logger")
            if _log:
                _log.warning(f"Failed to update speciation_state cluster0 size after distribution: {e}")
        except Exception:
            pass




def save_state(path: str) -> None:
    
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    species_sizes = {}
    species_member_ids = {}
    elites_genomes = []
    
    if elites_path.exists():
        try:
            from utils.population_io import load_elites
            elites_genomes = load_elites(str(elites_path), logger=logger)
        except Exception as e:
            logger.warning(f"Failed to load elites.json for validation: {e}")
            elites_genomes = []
    
    if "_genome_tracker" not in state:
        logger.error("Genome tracker is required for save_state() - cannot calculate species member_ids/sizes without it")
        raise RuntimeError("Genome tracker is required for save_state()")
    
    genome_tracker = state["_genome_tracker"]
    for species_id in state["species"].keys():
        member_ids = genome_tracker.get_all_genomes_by_species(species_id)
        if member_ids:
            species_member_ids[species_id] = sorted([str(mid) for mid in member_ids])
            species_sizes[species_id] = len(member_ids)
        else:
            species_member_ids[species_id] = []
            species_sizes[species_id] = 0
    
    logger.debug(f"Calculated species sizes from tracker: {len(species_sizes)} species in state['species']")
    
    species_dict = {}
    incubator_ids = []
    
    def _set_member_ids_and_size(sp_dict, sid, sp, species_member_ids, logger):
        
        sid_int = int(sid)
        if sid_int in species_member_ids:
            sp_dict["member_ids"] = species_member_ids[sid_int]
        else:
            logger.warning(f"Species {sid} not found in tracker, using in-memory member IDs ({len(sp.members)})")
            sp_dict["member_ids"] = [str(m.id) for m in sp.members]
        sp_dict["size"] = len(sp_dict["member_ids"])
    
    for sid, sp in state["species"].items():
        if sp.species_state in ["active", "frozen"]:
            sp_dict = sp.to_dict()
            _set_member_ids_and_size(sp_dict, sid, sp, species_member_ids, logger)
            
            if sp.species_state == "frozen" and sp.leader:
                if sp.leader.embedding is not None:
                    sp_dict["leader_embedding"] = sp.leader.embedding.tolist()
                elif "leader_embedding" not in sp_dict or sp_dict["leader_embedding"] is None:
                    leader_genome = next((g for g in elites_genomes if g.get("id") == sp.leader.id), None)
                    if leader_genome and "prompt_embedding" in leader_genome:
                        emb = leader_genome["prompt_embedding"]
                        sp_dict["leader_embedding"] = emb if isinstance(emb, list) else emb.tolist()
                    else:
                        logger.warning(f"Frozen species {sid} leader has no embedding - merging may fail")
                if "leader_distance" not in sp_dict:
                    sp_dict["leader_distance"] = sp.leader_distance
                if "labels" not in sp_dict:
                    sp_dict["labels"] = sp.labels
                if "label_history" not in sp_dict:
                    sp_dict["label_history"] = sp.label_history[-20:]
            
            species_dict[str(sid)] = sp_dict
    
    extinct_ids = []
    for sid, sp in state.get("historical_species", {}).items():
        if str(sid) not in species_dict:
            if sp.species_state == "extinct":
                if "_genome_tracker" in state:
                    active_member_count = len(state["_genome_tracker"].get_all_genomes_by_species(sid))
                    if active_member_count > 0:
                        logger.warning(
                            f"Extinct species {sid} has {active_member_count} genomes in genome_tracker "
                            f"(should be 0 after merge). This indicates genomes were NOT properly reassigned to merged species. "
                            f"Check merge events in events_tracker.json for details."
                        )
                    else:
                        logger.debug(f"Extinct species {sid} verified: 0 active members in genome_tracker (correctly extinct)")
                extinct_ids.append(sid)
            elif sp.species_state == "incubator":
                incubator_ids.append(sid)
    
    
    _validate_species_accounting(
        state, "save_state",
        incubator_ids=incubator_ids,
        extinct_ids=extinct_ids,
    )
    
    from collections import Counter
    leader_ids = [sp_dict.get("leader_id") for sp_dict in species_dict.values() if sp_dict.get("leader_id")]
    duplicates = {lid: count for lid, count in Counter(leader_ids).items() if count > 1}
    if duplicates:
        logger.warning(f"Duplicate leader IDs: {duplicates}")
    
    if elites_path.exists() and elites_genomes:
        for sid_str, sp_dict in species_dict.items():
            try:
                sid = int(sid_str)
                member_ids = {str(mid) for mid in sp_dict.get("member_ids", [])}
                species_genome_ids = {str(g.get("id")) for g in elites_genomes if g.get("species_id") == sid and g.get("id") is not None}
                leader_id_str = str(sp_dict.get("leader_id")) if sp_dict.get("leader_id") is not None else None
                
                if leader_id_str and leader_id_str not in member_ids:
                    logger.warning(f"Species {sid}: leader_id ({leader_id_str}) not in member_ids")
                if extra := species_genome_ids - member_ids:
                    logger.warning(f"Species {sid}: {len(extra)} genomes in elites.json not in member_ids: {sorted(list(extra))[:5]}")
            except (ValueError, KeyError) as e:
                logger.debug(f"Validation error for species {sid_str}: {e}")
                continue

    min_island_size = state["config"].min_island_size
    for sid_str, sp_dict in species_dict.items():
        mids = sp_dict.get("member_ids", [])
        n = len(mids)
        if sp_dict.get("size") != n:
            logger.warning(f"Species {sid_str}: size={sp_dict.get('size')} != len(member_ids)={n}; setting size={n}")
            sp_dict["size"] = n
        if n < min_island_size:
            logger.debug(
                "Species %s has size %d < min_island_size %d (count from tracker); "
                "Phase 5 (Stagnation and Incubation) moves such species to incubator when in-memory size drops.",
                sid_str, n, min_island_size
            )

    reserves_path = outputs_path / "reserves.json"
    reserves_genomes = _load_json_file(reserves_path, logger, [])
    actual_cluster0_size = len(reserves_genomes) if reserves_genomes else state["cluster0"].size
    
    if actual_cluster0_size != state["cluster0"].size:
        logger.warning(f"Cluster 0 size mismatch: reserves.json={actual_cluster0_size}, in-memory={state['cluster0'].size}")
    
    cluster0_max_fitness = cluster0_min_fitness = 0.0
    if reserves_genomes:
        from utils.population_io import _extract_north_star_score
        cluster0_fitnesses = [_extract_north_star_score(g, "toxicity") or 0.0 for g in reserves_genomes]
        if cluster0_fitnesses:
            cluster0_max_fitness = round(max(cluster0_fitnesses), 4)
            cluster0_min_fitness = round(min(cluster0_fitnesses), 4)
    
    cluster0_dict = state["cluster0"].to_dict()
    cluster0_dict["size"] = actual_cluster0_size
    cluster0_dict["max_fitness"] = cluster0_max_fitness
    cluster0_dict["min_fitness"] = cluster0_min_fitness
    
    if "speciation_events" in cluster0_dict:
        for evt in cluster0_dict["speciation_events"]:
            if isinstance(evt, dict) and "leader_fitness" in evt:
                try:
                    evt["leader_fitness"] = round(float(evt["leader_fitness"]), 4)
                except Exception:
                    pass
    
    state_dict = {
        "species": species_dict,
        "incubators": sorted(incubator_ids),
        "extinct": sorted(extinct_ids),
        "cluster0": cluster0_dict,
        "cluster0_size_from_reserves": actual_cluster0_size,
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict(),
        "config": state["config"].to_dict()
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
    extinct_count = len(extinct_ids)
    incubator_count = len(incubator_ids)
    
    state["logger"].info(f"Saved speciation state to {path}: {active_count} active, {frozen_count} frozen, {extinct_count} extinct (IDs only), {incubator_count} incubator (IDs only)")


def load_state(path: str) -> bool:
    
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    current_config = state["config"]
    
    state_path = Path(path)
    if not state_path.exists():
        logger.warning(f"Speciation state file not found: {path}")
        return False
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        config = current_config
        if "config" in loaded_state:
            saved_config_dict = loaded_state["config"]
            saved_config = SpeciationConfig.from_dict(saved_config_dict)
            if saved_config.species_stagnation != current_config.species_stagnation:
                logger.info(f"Config difference: saved species_stagnation={saved_config.species_stagnation}, using current={current_config.species_stagnation} (command-line argument takes precedence)")
        
        state["species"] = {}
        state["historical_species"] = {}
        max_species_id = 0
        
        for sid_str, sp_dict in loaded_state.get("species", {}).items():
            sid = int(sid_str)
            max_species_id = max(max_species_id, sid)
            
            leader_embedding = None
            if sp_dict.get("leader_embedding"):
                leader_embedding = np.array(sp_dict["leader_embedding"])
            
            leader = Individual(
                id=sp_dict["leader_id"],
                prompt=sp_dict.get("leader_prompt", ""),
                fitness=sp_dict.get("leader_fitness", 0.0),
                embedding=leader_embedding,
                species_id=sid
            )
            
            cluster_origin = sp_dict.get("cluster_origin")
            if cluster_origin is None or cluster_origin == "unknown":
                cluster_origin = "natural"
            
            members = [leader]
            member_ids = sp_dict.get("member_ids", [])
            
            if member_ids:
                outputs_path = get_outputs_path()
                try:
                    member_ids_to_load = [mid for mid in member_ids if str(mid) != str(leader.id)]
                    loaded_genomes = _load_genomes_by_ids(member_ids_to_load, outputs_path, logger)
                    
                    genome_by_id = {str(g.get("id")): g for g in loaded_genomes}
                    
                    loaded_count = 0
                    for member_id in member_ids:
                        member_id_str = str(member_id)
                        if member_id_str == str(leader.id):
                            continue
                        if member_id_str in genome_by_id:
                            member_genome = genome_by_id[member_id_str]
                            member = Individual.from_genome(member_genome)
                            members.append(member)
                            loaded_count += 1
                    
                    expected_count = len(member_ids) - (1 if str(leader.id) in [str(mid) for mid in member_ids] else 0)
                    if loaded_count != expected_count:
                        missing_ids = set(str(mid) for mid in member_ids_to_load) - set(str(g.get("id")) for g in loaded_genomes)
                        logger.warning(
                            f"Species {sid}: member loading incomplete - loaded {loaded_count}/{expected_count} members. "
                            f"Missing IDs: {sorted(missing_ids)[:10]} (may be in archive.json or not yet created)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load members for species {sid} from genome files: {e}")
            
            max_fit = max((m.fitness for m in members), default=0.0)
            species = Species(
                id=sid,
                leader=leader,
                members=members,
                radius=sp_dict.get("radius", config.theta_sim),
                stagnation=sp_dict.get("stagnation", 0),
                max_fitness=max_fit,
                species_state=sp_dict.get("species_state", "active"),
                created_at=sp_dict.get("created_at", 0),
                last_improvement=sp_dict.get("last_improvement", 0),
                fitness_history=sp_dict.get("fitness_history", []),
                labels=sp_dict.get("labels", []),
                label_history=sp_dict.get("label_history", []),
                cluster_origin=cluster_origin,
                parent_ids=sp_dict.get("parent_ids"),
                leader_distance=sp_dict.get("leader_distance", 0.0)
            )
            
            if species.leader.embedding is None:
                outputs_path = get_outputs_path()
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        leader_genome = next((g for g in elites_genomes if g.get("id") == species.leader.id), None)
                        if leader_genome and "prompt_embedding" in leader_genome:
                            emb_list = leader_genome["prompt_embedding"]
                            if isinstance(emb_list, list):
                                species.leader.embedding = np.array(emb_list, dtype=np.float32)
                                norm = np.linalg.norm(species.leader.embedding)
                                if not np.isclose(norm, 1.0, atol=1e-5) and norm > 0:
                                    species.leader.embedding = species.leader.embedding / norm
                                logger.debug(f"Loaded leader embedding for species {sid} from elites.json")
                            elif isinstance(emb_list, np.ndarray):
                                species.leader.embedding = emb_list
                    except Exception as e:
                        logger.warning(f"Failed to load leader embedding for species {sid} from elites.json: {e}")
            
            if species.species_state in ["active", "frozen"]:
                state["species"][sid] = species
            elif species.species_state == "extinct":
                state["historical_species"][sid] = species
            else:
                state["historical_species"][sid] = species
        
        incubator_ids = loaded_state.get("incubators", [])
        extinct_ids = loaded_state.get("extinct", [])
        for sid in incubator_ids:
            sid = int(sid)
            max_species_id = max(max_species_id, sid)
            state["historical_species"][sid] = SimpleNamespace(species_state="incubator")
        for sid in extinct_ids:
            sid = int(sid)
            max_species_id = max(max_species_id, sid)
            state["historical_species"][sid] = SimpleNamespace(species_state="extinct")
        if incubator_ids or extinct_ids:
            logger.debug(
                f"Restored historical_species from file: {len(incubator_ids)} incubator, {len(extinct_ids)} extinct"
            )
        
        SpeciesIdGenerator.set_min_id(max_species_id + 1)
        
        if "metrics" in loaded_state:
            metrics_dict = loaded_state["metrics"]
            state["metrics_tracker"] = SpeciationMetricsTracker.from_dict(metrics_dict, logger=logger)
        else:
            state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        
        global_best_id = loaded_state.get("global_best_id")
        if global_best_id:
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    global_best_genome = next((g for g in elites_genomes if g.get("id") == global_best_id), None)
                    if global_best_genome:
                        state["global_best"] = Individual.from_genome(global_best_genome)
                except Exception as e:
                    logger.warning(f"Failed to load global best from elites.json: {e}")
        
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        historical_count = len(state["historical_species"])
        logger.info(f"Loaded speciation state from {path}: {active_count} active, {frozen_count} frozen, {historical_count} historical species")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load speciation state: {e}", exc_info=True)
        return False


def reset_speciation_module() -> None:
    
    global _state
    if _state is not None:
        config = _state["config"]
        logger = _state["logger"]
        _state["species"] = {}
        _state["historical_species"] = {}
        _state["cluster0"] = Cluster0(
            min_cluster_size=config.cluster0_min_cluster_size,
            theta_sim=config.theta_sim,
            max_capacity=config.cluster0_max_capacity,
            min_island_size=config.min_island_size,
            w_genotype=config.w_genotype,
            w_phenotype=config.w_phenotype,
            logger=logger
        )
        _state["global_best"] = None
        _state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        _state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0}
        _state["_archived_count"] = 0
        SpeciesIdGenerator.reset()
        logger.info("Speciation module reset")
    else:
        _state = None


def run_speciation(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    north_star_metric: str = "toxicity") -> Dict[str, Any]:
    
    logger = get_logger("RunSpeciation", log_file)
    logger.info("Starting speciation: generation=%d", current_generation)

    reset_speciation_module()
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning("Temp file not found: %s - updating EvolutionTracker with current state", temp_path)
        _init_state(config, logger)
        state = _get_state()
        
        if current_generation > 0:
            outputs_path_state = get_outputs_path()
            state_path = str(outputs_path_state / "speciation_state.json")
            if Path(state_path).exists():
                load_state(state_path)
        
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        actual_reserves_size = state["cluster0"].size
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception as e:
                state["logger"].debug("Reserves size fallback: could not read reserves.json (%s), using cluster0.size=%s", e, state["cluster0"].size)
        
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        
        try:
            state_path = outputs_path / "speciation_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                   if sp.get("species_state") == "frozen"])
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json")
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
        
        total_species_count = active_count + frozen_count
        
        no_temp_result = {
            "species_count": total_species_count,
            "active_species_count": active_count,
            "frozen_species_count": frozen_count,
            "reserves_size": actual_reserves_size,
            "largest_species_size": 0,
            "average_species_size": 0.0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "elites_moved": 0,
            "reserves_moved": 0,
            "success": True,
            "error": None
        }
        
        try:
            outputs_path_tracker = get_outputs_path()
            evolution_tracker_path = str(outputs_path_tracker / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=no_temp_result,
                speciation_stats=speciation_stats,
                logger=logger
            )
            logger.info("Updated EvolutionTracker with current speciation state (temp file not found)")
        except Exception as e:
            logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        
        return no_temp_result
    
    try:
        with open(temp_path_obj, 'r', encoding='utf-8') as f:
            genomes = json.load(f)
        
        if not genomes:
            logger.warning("No genomes found in temp.json - updating EvolutionTracker with current state")
            _init_state(config, logger)
            state = _get_state()
            
            if current_generation > 0:
                outputs_path_state = get_outputs_path()
                state_path = str(outputs_path_state / "speciation_state.json")
                if Path(state_path).exists():
                    load_state(state_path)
            
            outputs_path = get_outputs_path()
            reserves_path = outputs_path / "reserves.json"
            if reserves_path.exists():
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            else:
                actual_reserves_size = state["cluster0"].size
                logger.warning(f"reserves.json not found, using cluster0.size={actual_reserves_size}")
            
            state_path = outputs_path / "speciation_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                  if sp.get("species_state") == "frozen"])
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (no genomes)")
            else:
                active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
                frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
                logger.warning("speciation_state.json not found, using in-memory counts")
            
            total_species_count = active_count + frozen_count
            sizes = [sp.size for sp in state["species"].values()]
            largest_species_size = max(sizes) if sizes else 0
            average_species_size = (sum(sizes) / len(sizes)) if sizes else 0.0
            no_genomes_result = {
                "species_count": total_species_count,
                "active_species_count": active_count,
                "frozen_species_count": frozen_count,
                "reserves_size": actual_reserves_size,
                "largest_species_size": largest_species_size,
                "average_species_size": round(average_species_size, 2),
                "speciation_events": 0,
                "merge_events": 0,
                "extinction_events": 0,
                "archived_count": 0,
                "genomes_updated": 0,
                "elites_moved": 0,
                "reserves_moved": 0,
                "success": True,
                "error": None
            }
            
            try:
                outputs_path_tracker = get_outputs_path()
                evolution_tracker_path = str(outputs_path_tracker / "EvolutionTracker.json")
                speciation_stats = get_speciation_statistics(log_file)
                update_evolution_tracker_with_speciation(
                    evolution_tracker_path=evolution_tracker_path,
                    current_generation=current_generation,
                    speciation_result=no_genomes_result,
                    speciation_stats=speciation_stats,
                    logger=logger
                )
                logger.info("Updated EvolutionTracker with current speciation state (no new genomes)")
            except Exception as e:
                logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
            
            return no_genomes_result
        
        logger.debug("Loaded %d genomes for speciation", len(genomes))
        
        _speciation_start = _time.time()
        species, cluster0 = process_generation(
            current_generation=current_generation,
            temp_path=temp_path,
            config=config,
            logger=logger
        )
        speciation_duration_seconds = round(_time.time() - _speciation_start, 3)
        
        
        
        state = _get_state()
        if "_events_tracker" in state:
            state["_events_tracker"].save()
        
        if "_genome_tracker" in state:
            state["_genome_tracker"].save()
        
        outputs_path = get_outputs_path()
        
        elites_path = str(outputs_path / "elites.json")
        reserves_path = str(outputs_path / "reserves.json")
        
        actual_reserves_size = state["cluster0"].size
        if Path(reserves_path).exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception as e:
                state["logger"].debug("Reserves size fallback: could not read reserves.json (%s), using cluster0.size=%s", e, state["cluster0"].size)
        
        log_generation_summary(current_generation, state["species"], actual_reserves_size,
                               state["_current_gen_events"], state["logger"], elites_path=elites_path)
        
        remove_embeddings_from_temp(temp_path=temp_path, logger=logger)
        
        is_valid, errors = validate_speciation_consistency(
            outputs_path, current_generation, logger=logger, expect_temp_empty=True
        )
        if not is_valid:
            logger.warning(f"Consistency validation found {len(errors)} errors")
            for error in errors[:5]:
                logger.warning(f"  - {error}")
        else:
            logger.info("Consistency validation passed after distribution")
        
        events = state["_current_gen_events"]
        
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        
        try:
            state_path = Path(outputs_path / "speciation_state.json")
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (after distribution)")
                
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                   if sp.get("species_state") == "frozen"])
                
                logger.debug(f"Calculated active_count={active_count}, frozen_count={frozen_count} from speciation_state.json")
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
        
        total_species_count = active_count + frozen_count
        
        current_metrics = None
        if state and "metrics_tracker" in state and state["metrics_tracker"].history:
            current_metrics = state["metrics_tracker"].history[-1]
            logger.debug(f"Retrieved metrics from metrics_tracker for generation {current_generation}")
        
        sizes = [sp.size for sp in state["species"].values()]
        largest_species_size = max(sizes) if sizes else 0
        average_species_size = (sum(sizes) / len(sizes)) if sizes else 0.0
        result = {
            "species_count": total_species_count,
            "active_species_count": active_count,
            "frozen_species_count": frozen_count,
            "reserves_size": actual_reserves_size,
            "largest_species_size": largest_species_size,
            "average_species_size": round(average_species_size, 2),
            "speciation_events": events.get("speciation", 0),
            "merge_events": events.get("merge", 0),
            "extinction_events": events.get("extinction", 0),
            "archived_count": state["_archived_count"],
            "genomes_updated": state["_genome_tracker"].get_distribution_stats()["total_genomes"] if "_genome_tracker" in state else 0,
            "success": True,
            "speciation_duration_seconds": speciation_duration_seconds,
        }
        
        if current_metrics:
            result["inter_species_diversity"] = round(current_metrics.inter_species_diversity, 4)
            result["intra_species_diversity"] = round(current_metrics.intra_species_diversity, 4)
            if hasattr(current_metrics, 'cluster_quality') and current_metrics.cluster_quality:
                result["cluster_quality"] = current_metrics.cluster_quality
            logger.debug(f"Added diversity metrics: inter={result['inter_species_diversity']:.4f}, intra={result['intra_species_diversity']:.4f}")
        else:
            result["inter_species_diversity"] = 0.0
            result["intra_species_diversity"] = 0.0
            result["cluster_quality"] = None
            logger.debug("No metrics available, using default diversity values (0.0)")
        
        if "_genome_tracker" in state:
            stats = state["_genome_tracker"].get_distribution_stats()
            by_sid = stats.get("by_species_id", {})
            elites_count = sum(int(v) for k, v in by_sid.items() if k.isdigit() and int(k) > 0)
            reserves_count = int(by_sid.get("0", 0))
            result.update({
                "elites_moved": elites_count,
                "reserves_moved": reserves_count
            })
        
        logger.info(
            "Speciation completed: %d active species (%d frozen), %d in reserves, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result.get("frozen_species_count", 0), result["reserves_size"],
            result["speciation_events"], result["merge_events"],
            result["extinction_events"], result["archived_count"]
        )
        
        try:
            outputs_path = get_outputs_path()
            evolution_tracker_path = str(outputs_path / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=result,
                speciation_stats=speciation_stats,
                logger=logger
            )
        except Exception as e:
            logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        
        return result
        
    except Exception as e:
        logger.error("Speciation failed: %s", e, exc_info=True)
        error_result = {
            "species_count": 0,
            "active_species_count": 0,
            "frozen_species_count": 0,
            "largest_species_size": 0,
            "average_species_size": 0.0,
            "reserves_size": 0,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "elites_moved": 0,
            "reserves_moved": 0,
            "inter_species_diversity": 0.0,
            "intra_species_diversity": 0.0,
            "cluster_quality": None,
            "success": False,
            "error": str(e)
        }
        
        try:
            outputs_path = get_outputs_path()
            evolution_tracker_path = str(outputs_path / "EvolutionTracker.json")
            speciation_stats = get_speciation_statistics(log_file)
            update_evolution_tracker_with_speciation(
                evolution_tracker_path=evolution_tracker_path,
                current_generation=current_generation,
                speciation_result=error_result,
                speciation_stats=speciation_stats,
                logger=logger
            )
            logger.info("Updated EvolutionTracker with speciation error state")
        except Exception as tracker_error:
            logger.error("Failed to update EvolutionTracker after speciation failure: %s", tracker_error, exc_info=True)
        
        return error_result


def get_speciation_statistics(log_file: Optional[str] = None) -> Dict[str, Any]:
    
    logger = get_logger("RunSpeciation", log_file)
    
    outputs_path = get_outputs_path()
    state_path = outputs_path / "speciation_state.json"
    
    if not state_path.exists():
        return {
            "initialized": False,
            "species_count": 0,
            "reserves_size": 0
        }
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        species_dict = loaded_state.get("species", {})
        file_active_species_count = len([sid for sid, sp in species_dict.items()
                                    if sp.get("species_state") == "active"])
        state = _get_state()
        if state and "species" in state:
            active_species_count = _validate_active_count(state, file_active_species_count, "get_speciation_statistics")
        else:
            active_species_count = file_active_species_count
        
        cluster0_dict = loaded_state.get("cluster0", {})
        reserves_size = cluster0_dict.get("size", 0)
        
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                reserves_size = len(reserves_genomes)
            except Exception:
                pass
        
        metrics_dict = loaded_state.get("metrics", {})
        metrics_summary = metrics_dict.get("summary", {})
        
        global_best_id = loaded_state.get("global_best_id")
        global_best_fitness = None
        if global_best_id:
            elites_path = outputs_path / "elites.json"
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    from utils.population_io import _extract_north_star_score
                    for genome in elites_genomes:
                        if genome.get("id") == global_best_id:
                            global_best_fitness = _extract_north_star_score(genome, "toxicity")
                            break
                except Exception:
                    pass
        
        return {
            "initialized": True,
            "species_count": active_species_count,
            "reserves_size": reserves_size,
            "global_best_fitness": global_best_fitness,
            "metrics_summary": metrics_summary
        }
    except Exception as e:
        logger.warning(f"Failed to read speciation statistics from file: {e}")
        state = _get_state()
        if state is None:
            return {
                "initialized": False,
                "species_count": 0,
                "reserves_size": 0
            }
        
        metrics_summary = state["metrics_tracker"].get_summary()
        active_species_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
        return {
            "initialized": True,
            "species_count": active_species_count,
            "reserves_size": state["cluster0"].size,
            "global_best_fitness": state["global_best"].fitness if state["global_best"] else None,
            "metrics_summary": metrics_summary
        }


def update_evolution_tracker_with_speciation(
    evolution_tracker_path: str,
    current_generation: int,
    speciation_result: Dict[str, Any],
    speciation_stats: Optional[Dict[str, Any]] = None,
    logger=None) -> bool:
    
    if logger is None:
        logger = get_logger("UpdateEvolutionTracker")
    
    try:
        tracker_path = Path(evolution_tracker_path)
        if not tracker_path.exists():
            logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        if speciation_stats is None:
            speciation_stats = get_speciation_statistics()
        
        metrics_summary = speciation_stats.get("metrics_summary", {})
        
        state = _get_state()
        current_metrics = None
        if state is not None and state["metrics_tracker"].history:
            current_metrics = state["metrics_tracker"].history[-1]
        
        frozen_species_count = speciation_result.get("frozen_species_count", 0)
        if frozen_species_count == 0:
            state = _get_state()
            frozen_species_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        
        active_species_count = speciation_result.get("active_species_count", 0)
        total_species_count = active_species_count + frozen_species_count
        
        speciation_summary = {
            "species_count": total_species_count,
            "active_species_count": active_species_count,
            "frozen_species_count": frozen_species_count,
            "reserves_size": speciation_result.get("reserves_size", 0),
            "largest_species_size": speciation_result.get("largest_species_size", 0),
            "average_species_size": speciation_result.get("average_species_size", 0.0),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
            "genomes_updated": speciation_result.get("genomes_updated", 0),
            "inter_species_diversity": 0.0,
            "intra_species_diversity": 0.0,
            "total_population": 0,
            "cluster_quality": None
        }
        
        best_fitness_value = 0.0
        
        if current_metrics:
            best_fitness_value = current_metrics.best_fitness
            speciation_summary.update({
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "total_population": current_metrics.total_population,
            })
            if hasattr(current_metrics, 'cluster_quality') and current_metrics.cluster_quality:
                speciation_summary["cluster_quality"] = current_metrics.cluster_quality
        else:
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            reserves_path = outputs_path / "reserves.json"
            
            total_pop = 0
            all_fitness = []
            
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    total_pop += len(elites_genomes)
                    from utils.population_io import _extract_north_star_score
                    for genome in elites_genomes:
                        fitness = _extract_north_star_score(genome, "toxicity")
                        if fitness > 0:
                            all_fitness.append(float(fitness))
                except Exception:
                    pass
            
            if reserves_path.exists():
                try:
                    with open(reserves_path, 'r', encoding='utf-8') as f:
                        reserves_genomes = json.load(f)
                    total_pop += len(reserves_genomes)
                    from utils.population_io import _extract_north_star_score
                    for genome in reserves_genomes:
                        fitness = _extract_north_star_score(genome, "toxicity")
                        if fitness > 0:
                            all_fitness.append(float(fitness))
                except Exception:
                    pass
            
            if total_pop == 0:
                for sp in state.get("species", {}).values():
                    if hasattr(sp, 'members'):
                        all_fitness.extend([m.fitness for m in sp.members])
                        total_pop += len(sp.members)
                
                cluster0 = state.get("cluster0")
                if cluster0 and hasattr(cluster0, 'individuals'):
                    all_fitness.extend([ind.fitness for ind in cluster0.individuals])
                    total_pop += len(cluster0.individuals)
            
            best_fitness_value = max(all_fitness) if all_fitness else 0.0
            
            speciation_summary.update({
                "inter_species_diversity": 0.0,
                "intra_species_diversity": 0.0,
                "total_population": total_pop,
            })
        
        
        generations = evolution_tracker.get("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        selection_mode = evolution_tracker.get("selection_mode", "default")
        
        if gen_entry:
            from utils.population_io import _ensure_generation_entry_has_all_fields
            gen_entry = _ensure_generation_entry_has_all_fields(gen_entry, current_generation, selection_mode)
        else:
            from utils.population_io import _get_standard_generation_entry_template
            gen_entry = _get_standard_generation_entry_template(current_generation, selection_mode)
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        gen_entry["speciation"] = speciation_summary
        
        
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_reserves_size": speciation_result.get("reserves_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated EvolutionTracker.json with speciation data for generation %d", current_generation)
        return True
        
    except Exception as e:
        logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        return False
