"""
run_speciation.py

Main entry point functions for Dynamic Islands speciation framework.
All functionality is provided as module-level functions (no classes).

Species Key Type Convention:
    - In-memory: state["species"] keys are int (species IDs)
    - JSON files: keys are str (JSON limitation - keys are always strings)
    - At load_state: keys are normalized to int when loading from JSON
    - At save_state: keys are converted to str when writing to JSON
    - CRITICAL: When performing lookups or set operations, always normalize
      species IDs to int to avoid type mismatch bugs (e.g., 1 != "1" in Python)
"""

import json
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

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _ = get_system_utils()

# Global state (replaces class instance)
_state: Optional[Dict[str, Any]] = None


def _init_state(config: Optional[SpeciationConfig] = None, logger=None) -> None:
    """Initialize global state."""
    global _state
    if _state is None:
        _state = {
            "config": config or SpeciationConfig(),
            "logger": logger or get_logger("Speciation"),
            "species": {},  # Active species only (state="active")
            "historical_species": {},  # Extinct (merged parents) and incubator species (preserved for reference)
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
        # Update config if provided (ensures command-line arguments are followed)
        if config is not None:
            old_config = _state["config"]
            _state["config"] = config
            # Update cluster0 parameters if config changed
            if (old_config.theta_sim != config.theta_sim or 
                old_config.cluster0_max_capacity != config.cluster0_max_capacity or
                old_config.cluster0_min_cluster_size != config.cluster0_min_cluster_size):
                _state["cluster0"].theta_sim = config.theta_sim
                _state["cluster0"].max_capacity = config.cluster0_max_capacity
                _state["cluster0"].min_cluster_size = config.cluster0_min_cluster_size
                _state["logger"].info(f"Config updated: theta_sim={config.theta_sim}, species_stagnation={config.species_stagnation}, species_capacity={config.species_capacity}")
        # Update logger if provided
        if logger is not None:
            _state["logger"] = logger


def _get_state() -> Dict[str, Any]:
    """Get global state, initializing if needed."""
    if _state is None:
        _init_state()
    return _state


def _save_tracker_if_dirty(state: Dict[str, Any]) -> None:
    """Save tracker if it has unsaved changes."""
    if "_genome_tracker" in state:
        tracker = state["_genome_tracker"]
        if tracker._dirty:
            tracker.save()
            state["logger"].debug("Saved genome tracker after critical operation")


def _validate_tracker_consistency(state: Dict[str, Any], phase_name: str) -> None:
    """Validate tracker consistency after a phase."""
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


def _load_json_file(file_path: Path, logger, default=None):
    """Load JSON file with error handling. Returns default if file doesn't exist or fails.
    Uses population_io for elites.json when possible."""
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
    """Update species_id in genomes based on tracker. Modifies genomes in-place."""
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
            # Not in tracker - register with current species_id or default
            file_sid = genome.get("species_id")
            if file_sid is None:
                file_sid = default_species_id
                genome["species_id"] = default_species_id
            genome_tracker.register(genome_id, file_sid, current_generation)
            logger.debug(f"Registered {genome_id} with species_id={file_sid} ({file_name})")


def _deduplicate_genomes(genomes: List[Dict]) -> List[Dict]:
    """Remove duplicate genomes by ID, keeping first occurrence."""
    seen = set()
    result = []
    for genome in genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if gid and gid not in seen:
            seen.add(gid)
            result.append(genome)
    return result


def _write_json_atomic(file_path: Path, data: List[Dict], logger, file_name: str) -> None:
    """Write JSON file atomically using temp file."""
    if not data:
        return
    temp_path = file_path.with_suffix('.json.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    temp_path.replace(file_path)
    logger.info(f"Wrote {len(data)} genomes to {file_name}")


def _validate_active_count(state: Dict[str, Any], calculated_count: int, source: str) -> int:
    """
    Validate and correct active species count.
    
    Compares calculated count with in-memory count and uses in-memory as source of truth.
    
    Args:
        state: Global speciation state
        calculated_count: Count calculated from file or other source
        source: Description of where calculated_count came from (for logging)
        
    Returns:
        Corrected active species count (uses in-memory if mismatch detected)
    """
    in_memory_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    if calculated_count != in_memory_count:
        state["logger"].warning(
            f"Active count mismatch: calculated={calculated_count} (from {source}), "
            f"in_memory={in_memory_count}, using in_memory as source of truth"
        )
        return in_memory_count
    return calculated_count


def _archive_individuals(individuals: List[Individual], generation: int, reason: str) -> None:
    """
    Archive individuals to archive.json.
    
    This function archives individuals to archive.json. The archive.json file is the
    authoritative source for archived genomes. Genome tracking is handled by genome_tracker.json,
    and speciation_state.json only contains species metadata, not full genome details.
    """
    if not individuals:
        return
    
    state = _get_state()
    state["_archived_count"] += len(individuals)
    logger = state["logger"]
    
    try:
        outputs_path = get_outputs_path()
        archive_path = outputs_path / "archive.json"
        
        # Load existing archive
        archive = _load_json_file(archive_path, logger, [])
        if not isinstance(archive, list):
            if isinstance(archive, dict):
                logger.warning(f"archive.json is a dict (expected list), converting to list")
                archive = list(archive.values()) if len(archive) > 0 else []
            else:
                logger.warning(f"archive.json has unexpected format, initializing as empty list")
                archive = []
        
        # Add new entries
        for ind in individuals:
            if hasattr(ind, 'to_genome'):
                entry = ind.to_genome()
                # Ensure entry has required fields
                if not entry:
                    entry = {}
                if "id" not in entry:
                    entry["id"] = ind.id
                if "prompt" not in entry and hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            else:
                # Fallback: create minimal entry
                entry = {"id": ind.id}
                if hasattr(ind, 'prompt'):
                    entry["prompt"] = ind.prompt
            
            entry["archived_at_generation"] = generation
            entry["archive_reason"] = reason
            # Preserve generation field (original generation when genome was created)
            if "generation" not in entry and hasattr(ind, 'generation'):
                entry["generation"] = ind.generation
            elif "generation" not in entry:
                # Fallback: use archived_at_generation if generation not available
                entry["generation"] = generation
            # Preserve fitness if available
            if hasattr(ind, 'fitness') and "fitness" not in entry:
                entry["fitness"] = ind.fitness
            # Always set species_id=-1 for archived genomes (don't preserve ind.species_id)
            entry["species_id"] = -1
            # Set initial_state for operator effectiveness metrics
            # Genomes archived due to capacity limits are non-elites
            if "initial_state" not in entry:
                if "capacity" in reason.lower():
                    entry["initial_state"] = "non-elite"
                else:
                    # For other reasons (extinction, etc.), preserve existing or default to elite
                    entry["initial_state"] = entry.get("initial_state", "elite")
            # Remove embeddings before archiving (save space, not needed for archived genomes)
            if "prompt_embedding" in entry:
                del entry["prompt_embedding"]
            
            archive.append(entry)
        
        # Save updated archive
        with open(archive_path, 'w', encoding='utf-8') as f:
            json.dump(archive, f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Archived {len(individuals)} individuals ({reason}) to archive.json")
        
    except Exception as e:
        logger.warning(f"Failed to archive individuals: {e}")


def _load_species_leaders_from_state(outputs_path: Path, logger) -> Dict[int, Tuple[Individual, Any, Optional[Any]]]:
    """
    Load all species leaders from speciation_state.json.
    
    Returns:
        Dict mapping species_id -> (leader_Individual, leader_embedding, leader_phenotype)
    """
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
            if sid == 0:  # Skip cluster 0
                continue
            
            # Extract leader data
            leader_id = sp_dict.get("leader_id")
            leader_embedding_list = sp_dict.get("leader_embedding")
            leader_fitness = sp_dict.get("leader_fitness", 0.0)
            leader_prompt = sp_dict.get("leader_prompt", "")
            
            if not leader_id or not leader_embedding_list:
                logger.debug(f"Species {sid} has no leader or embedding, skipping")
                continue
            
            # Convert embedding to numpy array
            leader_embedding = np.array(leader_embedding_list, dtype=np.float32)
            
            # Extract phenotype if available
            leader_phenotype = None
            if sp_dict.get("leader_genome_data"):
                from .phenotype_distance import extract_phenotype_vector
                leader_phenotype = extract_phenotype_vector(sp_dict["leader_genome_data"], logger=logger)
            
            # Create Individual for leader
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
    """Load genome data for given IDs from elites.json, temp.json, and reserves.json (archive genomes never rejoin)."""
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
    """
    Phase 7: Redistribution of Genomes
    
    Function name is phase8_redistribute_genomes() but called as Phase 7 within process_generation().
    Uses genome tracker as source of truth for distribution.
    Called within process_generation() as Phase 7, before Phase 8 (metrics).
    
    Steps:
    1. Update species_id in elites.json based on genome_tracker.json (one-by-one validation)
    2. Update species_id in reserves.json based on genome_tracker.json (one-by-one validation)
    3. Update species_id in temp.json based on genome_tracker.json (if exists)
    4. Redistribute genomes to correct files (elites.json, reserves.json, archive.json) based on updated species_id
    5. Clear temp.json
    6. Update related records (speciation_state.json, events_tracker.json)
    7. Final validation using tracker.validate_consistency()
    
    Args:
        temp_path: Path to temp.json
        current_generation: Current generation number
        
    Returns:
        Dictionary with distribution statistics
    """
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
    
    # Check if we need to load archive.json for validation (lazy loading)
    stats = genome_tracker.get_distribution_stats()
    has_archived = int(stats["by_species_id"].get("-1", 0)) > 0
    
    # Steps 1-3: Update species_id in all files based on tracker
    logger.info("Phase 7: Steps 1-3 - Updating species_id from tracker")
    elites_genomes = _load_json_file(elites_path, logger, [])
    reserves_genomes = _load_json_file(reserves_path, logger, [])
    temp_genomes = _load_json_file(temp_path_obj, logger, [])
    archive_genomes = _load_json_file(archive_path, logger, [])
    
    # Update species_id from tracker for each file (archive is NOT updated - genomes never leave archive)
    _update_genomes_from_tracker(elites_genomes, genome_tracker, current_generation, logger, "elites.json", default_species_id=0)
    _update_genomes_from_tracker(reserves_genomes, genome_tracker, current_generation, logger, "reserves.json", default_species_id=0)
    _update_genomes_from_tracker(temp_genomes, genome_tracker, current_generation, logger, "temp.json", default_species_id=0)
    
    # Ensure generation field is set for temp.json genomes
    for genome in temp_genomes:
        if "generation" not in genome or genome.get("generation") is None:
            genome["generation"] = current_generation
    
    # Step 4: Redistribute genomes to correct files based on updated species_id
    logger.info("Phase 7: Step 4 - Redistributing genomes to correct files")
    
    tracked_ids = set(genome_tracker.genomes.keys())
    file_sources = {
        "elites": (elites_genomes, set(str(g.get("id")) for g in elites_genomes if g.get("id"))),
        "reserves": (reserves_genomes, set(str(g.get("id")) for g in reserves_genomes if g.get("id"))),
        "temp": (temp_genomes, set(str(g.get("id")) for g in temp_genomes if g.get("id")))
    }
    
    # Collect all genomes for redistribution (elites, reserves, temp only - archive genomes never leave)
    tracked_genomes = []
    untracked_by_file = {"elites": [], "reserves": []}
    
    for file_name, (genomes, ids_set) in file_sources.items():
        for genome in genomes:
            gid = str(genome.get("id")) if genome.get("id") else None
            if gid and gid in tracked_ids:
                tracked_genomes.append(genome)
            elif gid and file_name in untracked_by_file:
                untracked_by_file[file_name].append(genome)
    
    # Redistribute tracked genomes by species_id
    elites_to_save = untracked_by_file["elites"]
    reserves_to_save = untracked_by_file["reserves"]
    # Preserve ALL existing archived genomes (archive is final destination - genomes never leave)
    archive_to_save = [g for g in archive_genomes if g.get("species_id") == -1 or str(g.get("id")) not in tracked_ids]
    movements = []
    
    for genome in tracked_genomes:
        gid = str(genome.get("id")) if genome.get("id") else None
        if not gid:
            continue
        
        # Get species_id from genome or tracker
        species_id = genome.get("species_id")
        if species_id is None:
            species_id = genome_tracker.get_species_id(gid) if genome_tracker.exists(gid) else 0
            genome["species_id"] = species_id
            if not genome_tracker.exists(gid):
                genome_tracker.register(gid, 0, current_generation)
        
        # Determine destination file
        if species_id > 0:
            new_file = "elites"
        elif species_id == 0:
            new_file = "reserves"
        elif species_id == -1:
            new_file = "archive"
            if "prompt_embedding" in genome:
                del genome["prompt_embedding"]
        else:
            logger.warning(f"Genome {gid} has invalid species_id: {species_id}, skipping")
            continue
        
        # Track movement
        old_file = next((f for f, (_, ids) in file_sources.items() if gid in ids), None)
        if old_file and old_file != new_file:
            movements.append((gid, old_file, new_file))
        
        # Add to destination
        {"elites": elites_to_save, "reserves": reserves_to_save, "archive": archive_to_save}[new_file].append(genome)
    
    # Handle genomes with archive_reason - always archive them
    for genome_list in [elites_to_save, reserves_to_save]:
        to_archive = [g for g in genome_list if g.get("archive_reason")]
        for genome in to_archive:
            gid = str(genome.get("id")) if genome.get("id") else None
            if gid:
                if genome_tracker.exists(gid) and genome_tracker.get_species_id(gid) != -1:
                    genome_tracker.update_species_id(gid, -1, current_generation, f"archive_reason_{genome.get('archive_reason')}")
                elif not genome_tracker.exists(gid):
                    genome_tracker.register(gid, -1, current_generation)
                if "prompt_embedding" in genome:
                    del genome["prompt_embedding"]
                genome["species_id"] = -1
                if gid not in {str(g.get("id")) for g in archive_to_save if g.get("id")}:
                    archive_to_save.append(genome)
        for g in to_archive:
            if g in genome_list:
                genome_list.remove(g)
    
    # Step 5: Deduplicate and write files atomically
    logger.info("Phase 7: Step 5 - Deduplicating and writing files")
    elites_deduped = _deduplicate_genomes(elites_to_save)
    reserves_deduped = _deduplicate_genomes(reserves_to_save)
    archive_deduped = _deduplicate_genomes(archive_to_save)
    
    _write_json_atomic(elites_path, elites_deduped, logger, "elites.json")
    _write_json_atomic(reserves_path, reserves_deduped, logger, "reserves.json")
    _write_json_atomic(archive_path, archive_deduped, logger, "archive.json")
    
    # Clear temp.json
    with open(temp_path_obj, 'w', encoding='utf-8') as f:
        json.dump([], f, indent=2, ensure_ascii=False)
    logger.info("Cleared temp.json")
    
    # Step 6: Update events tracker and validate
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
        for error in errors[:10]:  # Log first 10 errors
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
    """
    Process a single generation with full speciation pipeline.
    
    Core Logic (8 Phases):
    
    Phase 1: Existing Species Processing (only if elites.json is not empty)
      1. Compute embeddings for temp.json genomes
      2. Process variants against existing species (skip cluster 0 outliers)
      3. Radius enforcement after all processing is done
      NOTE: NO capacity enforcement in Phase 1 (saves compute resources)
      NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
            Species_id is NOT updated in elites.json/reserves.json during this phase
    
    Phase 2: Cluster 0 Speciation (when elites.json is empty, skip Phase 1)
      4. Load all genomes with species_id=0 from genome_tracker
      5. Apply isolated cluster 0 speciation (form new species from reserves)
      NOTE: NO radius enforcement or capacity enforcement in Phase 2
      NOTE: Only trackers are updated, not file genomes
    
    Phase 3: Merging + Radius Enforcement
      6. Merging of all species (iterative merging with immediate tracker updates)
         - All species_ids of both parent species changed to new merged species_id
         - Sort ALL members of both parent species by fitness (descending)
         - Select genome with highest fitness as leader (BEFORE radius enforcement)
         - Leader will NOT change until new genome with higher fitness is added
      7. Radius enforcement after all merging is complete
         - Members outside radius of leader moved to reserves (species_id=0)
      NOTE: Only trackers are updated, not file genomes
    
    Phase 4: Capacity Enforcement (species_id > 0)
      8. Capacity enforcement for ALL species (species_id > 0)
         - Top species_capacity genomes kept, excess archived (species_id=-1)
         - For merged species: leader unchanged (already highest fitness from merge)
         - For non-merged species: ensure leader is highest fitness
      NOTE: Uses genome_tracker as authoritative source
      NOTE: Does NOT update elites.json/reserves.json (only trackers)
    
    Phase 5: Stagnation and Incubation
      9. Record fitness and update stagnation counters
      10. Freeze stagnant species (excluded from parent selection, but still alive)
      11. Incubate small species (extinction/dissolution - move to cluster 0)
      NOTE: Only trackers are updated, not file genomes
    
    Phase 6: Cluster 0 Capacity Enforcement (species_id = 0)
      12. Capacity enforcement for cluster 0 (species_id = 0)
      NOTE: Uses genome_tracker as authoritative source
      NOTE: Does NOT update reserves.json (only trackers)
    
    Phase 7: Redistribution of Genomes
      13. Update species_id in temp.json, elites.json, reserves.json based on genome_tracker
      14. Distribute genomes to correct files (elites.json, reserves.json, archive.json)
      NOTE: This is the ONLY phase that updates species_id in file genomes
      NOTE: Files are synchronized with genome_tracker (authoritative source)
    
    Phase 8: Metrics & Statistics
      15. Update c-TF-IDF labels for all species
      16. Calculate and record metrics from distributed files
      17. Save all state files (speciation_state.json, events_tracker.json, genome_tracker.json)
    
    IMPORTANT: During phases 1-6, species_id is NOT updated in elites.json or reserves.json.
    Only genome_tracker.json and other metadata files are updated for quick CRUD operations.
    This saves compute resources as we don't need all genome details and avoid multiple file updates.
    File distribution happens only in Phase 7 after all speciation operations are complete.
    
    Args:
        current_generation: Current generation number
        temp_path: Optional path to temp.json
        config: Optional SpeciationConfig (uses defaults if None)
        logger: Optional logger instance
        
    Returns:
        Tuple of (species_dict, cluster0)
    """
    _init_state(config, logger)
    state = _get_state()
    
    
    state["logger"].info(f"=== Speciation Generation {current_generation} ===")
    state["_current_gen_events"] = {"speciation": 0, "merge": 0, "extinction": 0, "moved_to_cluster0": 0}
    state["_archived_count"] = 0
    
    # Initialize events tracker for audit trail
    events_tracker = EventsTracker(current_generation, logger=state["logger"])
    state["_events_tracker"] = events_tracker
    
    # Ensure Individual is available in function scope
    # This prevents UnboundLocalError when Individual is used before conditional imports later
    # Individual is imported at module level, but local imports later make it a local variable
    from .species import Individual
    
    # Initialize genome tracker (master registry)
    genome_tracker = GenomeTracker(logger=state["logger"])
    genome_tracker.load()  # Load existing tracker or start fresh
    
    # Auto-migrate if tracker is empty but source files exist and have data
    # Skip migration if this is Generation 0 (files are initialized but empty)
    if len(genome_tracker.genomes) == 0:
        outputs_path_check = get_outputs_path()
        elites_path_check = outputs_path_check / "elites.json"
        # Check if this is Generation 0 (elites.json doesn't exist or is empty)
        is_gen0 = not elites_path_check.exists() or (elites_path_check.exists() and len(json.load(open(elites_path_check, 'r', encoding='utf-8'))) == 0)
        
        if not is_gen0:
            # Not Generation 0, try migration from existing files
            from .migration import auto_migrate_if_needed
            auto_migrate_if_needed(logger=state["logger"])
            genome_tracker.load()  # Reload after migration
        else:
            # Generation 0 - files exist but are empty (initialized at start), skip migration
            state["logger"].debug("Generation 0 detected: files initialized but empty, skipping auto-migration")
    
    state["_genome_tracker"] = genome_tracker
    
    # Auto-load previous state if not first generation
    if current_generation > 0:
        outputs_path = get_outputs_path()
        state_path = str(outputs_path / "speciation_state.json")
        if Path(state_path).exists():
            load_state(state_path)
            state["logger"].info("Restored speciation state from previous generation")
    
    # ========================================================================
    # CORE LOGIC: SPECIES_ID UPDATE STRATEGY
    # ========================================================================
    # IMPORTANT: During phases 1-6, species_id is NOT updated in elites.json or reserves.json.
    # Only genome_tracker.json and other metadata files (events_tracker.json, speciation_state.json)
    # are updated for quick CRUD operations. This optimization saves compute resources because:
    # 1. We don't need all genome details for speciation operations (only IDs and species_id)
    # 2. We avoid multiple file I/O operations (updating same species_id multiple times)
    # 3. Tracker files are lightweight and fast to read/write
    #
    # Phase 7 is the ONLY phase that updates species_id in file genomes (temp.json, elites.json,
    # reserves.json) and distributes them to correct files. This ensures files are synchronized
    # with genome_tracker (authoritative source) after all speciation operations are complete.
    #
    # Enforcement rules by phase:
    # - Phase 1: Radius enforcement only, NO capacity enforcement
    # - Phase 2: NO radius enforcement, NO capacity enforcement
    # - Phase 3: Radius enforcement after merging
    # - Phase 4: Capacity enforcement for species_id > 0
    # - Phase 5: Stagnation and incubation (no enforcement)
    # - Phase 6: Capacity enforcement for species_id = 0
    # - Phase 7: File distribution (updates species_id in files)
    # - Phase 8: Metrics and statistics
    # ========================================================================
    
    # ========================================================================
    # PHASE 1: EXISTING SPECIES PROCESSING
    # ========================================================================
    # Process temp.json genomes with existing species (only if elites.json is not empty)
    # - Compute embeddings for temp.json genomes
    # - Process variants against existing species (skip cluster 0 outliers)
    # - Enforce radius after all processing is done
    # - NO capacity enforcement in Phase 1 (saves compute resources)
    # - Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # - Species_id is NOT updated in elites.json/reserves.json during this phase
    
    # Check if Generation 0 (no existing species)
    # Phase 1 should run if:
    # 1. There are existing species in memory (loaded from speciation_state.json), OR
    # 2. elites.json has genomes (from previous generation's Phase 7 distribution)
    # Phase 1 should be skipped (Generation 0) only if:
    # - No species in memory AND elites.json is empty/doesn't exist
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    
    # Check if there are existing species in memory (from speciation_state.json)
    has_existing_species = len(state["species"]) > 0
    
    # Check if elites.json has genomes
    elites_has_genomes = False
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_data = json.load(f)
                # Check if non-empty list or non-empty dict
                elites_has_genomes = (isinstance(elites_data, list) and len(elites_data) > 0) or \
                                     (isinstance(elites_data, dict) and len(elites_data) > 0)
        except (json.JSONDecodeError, Exception):
            # If file exists but can't be read, assume empty
            elites_has_genomes = False
    
    # Generation 0 = no existing species AND no genomes in elites.json
    is_generation_0 = not has_existing_species and not elites_has_genomes
    
    if has_existing_species and not elites_has_genomes:
        state["logger"].debug(f"Found {len(state['species'])} existing species in memory but elites.json is empty - will run Phase 1")
    
    if is_generation_0:
        state["logger"].info("=== Generation 0: Skipping Phase 1 (no existing species) ===")
        # Skip to Phase 2 - no snapshot needed as there are no existing species
        state["_prev_max_fitness"] = {}
        
        # For Generation 0, we still need to:
        # 1. Compute embeddings for temp.json genomes
        # 2. Register temp.json genomes in tracker with species_id=0
        # This is required for Phase 2 to work correctly
        
        if temp_path is None:
            outputs_path = get_outputs_path()
            temp_path = str(outputs_path / "temp.json")
        
        temp_path_obj = Path(temp_path)
        if temp_path_obj.exists():
            # Compute embeddings for temp.json genomes
            compute_and_save_embeddings(
                temp_path=temp_path,
                model_name=state["config"].embedding_model,
                batch_size=state["config"].embedding_batch_size,
                logger=state["logger"]
            )
            
            # Register all temp.json genomes in tracker with species_id=0
            # Also add them to cluster0 immediately so Phase 2 can use them
            try:
                with open(temp_path_obj, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                
                registered_count = 0
                added_to_cluster0_count = 0
                for genome in temp_genomes:
                    genome_id = str(genome.get("id")) if genome.get("id") else None
                    if genome_id and not state["_genome_tracker"].exists(genome_id):
                        # Register with species_id=0 (reserves/cluster 0)
                        state["_genome_tracker"].register(genome_id, 0, current_generation)
                        registered_count += 1
                    
                    # Also add to cluster0 if it has an embedding
                    if genome.get("prompt_embedding"):
                        try:
                            ind = Individual.from_genome(genome)
                            if ind.embedding is not None:
                                # Check if already in cluster0
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
        state["logger"].info("=== Phase 1: Existing Species Processing ===")

        # Snapshot max_fitness before Phase 1 (leader_follower_clustering and later steps can update it).
        # Used in Phase 5 for record_fitness(max_fitness_increased). Species created this gen are not in
        # _prev_max_fitness -> treated as max_fitness_increased=True.
        #
        # CRITICAL FIX: Compute true max fitness from tracker + loaded genomes, not just in-memory sp.members
        # In-memory members may be incomplete after load_state() (lazy loading), which causes sp.max_fitness
        # to be artificially low. This would make max_fitness_increased always True and prevent stagnation.
        outputs_path_prev_max = get_outputs_path()
        state["_prev_max_fitness"] = {}
        
        if "_genome_tracker" in state:
            from utils.population_io import _extract_north_star_score
            
            for sid, sp in state["species"].items():
                # Get all genome IDs for this species from tracker (authoritative source)
                species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                
                if species_genome_ids:
                    # Load genomes and compute true max fitness
                    loaded_genomes = _load_genomes_by_ids(species_genome_ids, outputs_path_prev_max, state["logger"])
                    
                    if loaded_genomes:
                        # Compute max fitness from loaded genomes
                        max_fitness = 0.0
                        for g in loaded_genomes:
                            fitness = _extract_north_star_score(g, "toxicity")
                            if fitness is not None and fitness > max_fitness:
                                max_fitness = fitness
                        state["_prev_max_fitness"][int(sid)] = max_fitness
                        
                        # Log if in-memory max differs significantly from true max
                        if abs(sp.max_fitness - max_fitness) > 0.001:
                            state["logger"].debug(
                                f"Species {sid}: _prev_max_fitness corrected from in-memory {sp.max_fitness:.4f} "
                                f"to tracker-based {max_fitness:.4f} ({len(loaded_genomes)} genomes)"
                            )
                    else:
                        # Fallback to in-memory if no genomes loaded
                        state["_prev_max_fitness"][int(sid)] = sp.max_fitness
                else:
                    # No genomes in tracker for this species, use in-memory
                    state["_prev_max_fitness"][int(sid)] = sp.max_fitness
        else:
            # No tracker available, fall back to in-memory max_fitness
            state["_prev_max_fitness"] = {int(sid): sp.max_fitness for sid, sp in state["species"].items()}

        # 1. Compute and save embeddings to temp.json
        if temp_path is None:
            outputs_path = get_outputs_path()
            temp_path = str(outputs_path / "temp.json")
        
        compute_and_save_embeddings(
            temp_path=temp_path,
            model_name=state["config"].embedding_model,
            batch_size=state["config"].embedding_batch_size,
            logger=state["logger"]
        )
        
        # 2. Process variants against existing species (skip cluster 0 outliers)
        outputs_path = get_outputs_path()
        speciation_state_path = str(outputs_path / "speciation_state.json")
        
        # Track species count BEFORE clustering
        species_count_before_clustering = len(state["species"])
        
        # Use skip_cluster0_outliers=True to process only against existing species
        state["species"], _ = leader_follower_clustering(
            temp_path=temp_path,
            speciation_state_path=speciation_state_path,
            theta_sim=state["config"].theta_sim,
            current_generation=current_generation,
            w_genotype=state["config"].w_genotype,
            w_phenotype=state["config"].w_phenotype,
            min_island_size=state["config"].min_island_size,
            skip_cluster0_outliers=True,  # NEW: Skip cluster 0 outliers during variant processing
            logger=state["logger"],
            genome_tracker=state.get("_genome_tracker"),
            events_tracker=state.get("_events_tracker")
        )
        
        # Count new species formed during leader_follower_clustering (should be 0 with skip_cluster0_outliers=True)
        species_count_after_clustering = len(state["species"])
        new_species_from_clustering = species_count_after_clustering - species_count_before_clustering
        if new_species_from_clustering > 0:
            state["_current_gen_events"]["speciation"] += new_species_from_clustering
            state["logger"].info(f"Counted {new_species_from_clustering} new species formed during leader-follower clustering (before: {species_count_before_clustering}, after: {species_count_after_clustering})")
        
        # 3. Radius enforcement after all processing is done
        # Load all species, get members from genome_tracker, check radius, remove outside radius
        # NOTE: This happens AFTER all variant processing is complete
        from .distance import ensemble_distance
        import numpy as np
        
        outputs_path = get_outputs_path()
        elites_path = outputs_path / "elites.json"
        temp_path_obj = Path(temp_path)
        
        for sid in list(state["species"].keys()):
            sp = state["species"][sid]
            if sp.leader is None or sp.leader.embedding is None:
                continue
            
            # Get all member IDs from genome_tracker for this species
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            
            # Load actual genome data for these IDs
            all_member_genomes = []
            
            # Load from elites.json
            if elites_path.exists():
                try:
                    with open(elites_path, 'r', encoding='utf-8') as f:
                        elites_genomes = json.load(f)
                    all_member_genomes = [g for g in elites_genomes if g.get("id") in species_genome_ids]
                except Exception as e:
                    state["logger"].warning(f"Failed to load elites.json for radius cleanup: {e}")
            
            # Load from temp.json
            if temp_path_obj.exists():
                try:
                    with open(temp_path_obj, 'r', encoding='utf-8') as f:
                        temp_genomes = json.load(f)
                    for g in temp_genomes:
                        if g.get("id") in species_genome_ids and not any(mg.get("id") == g.get("id") for mg in all_member_genomes):
                            all_member_genomes.append(g)
                except Exception as e:
                    state["logger"].warning(f"Failed to load temp.json for radius cleanup: {e}")
            
            # Check radius for each member
            members_to_remove = []
            for genome in all_member_genomes:
                genome_id = genome.get("id")
                if genome_id == sp.leader.id:
                    continue  # Leader always stays
                
                # Get embedding from genome
                genome_embedding = genome.get("prompt_embedding")
                if genome_embedding is None:
                    # No embedding - mark as species_id=0
                    members_to_remove.append(genome_id)
                    continue
                
                # Compute distance to leader
                # Extract phenotype if available
                from .phenotype_distance import extract_phenotype_vector
                genome_phenotype = extract_phenotype_vector(genome, logger=state["logger"])
                leader_phenotype = sp.leader.phenotype
                
                dist = ensemble_distance(
                    np.array(genome_embedding), sp.leader.embedding,
                    genome_phenotype, leader_phenotype,
                    state["config"].w_genotype, state["config"].w_phenotype
                )
                
                if dist >= state["config"].theta_sim:
                    # Outside radius - mark as species_id=0
                    members_to_remove.append(genome_id)
            
            # Update tracker for removed members
            if members_to_remove:
                state["logger"].debug(f"Species {sid}: removing {len(members_to_remove)} members outside radius")
                for genome_id in members_to_remove:
                    state["_genome_tracker"].update_species_id(
                        str(genome_id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves"
                    )
                
                # Update in-memory members to match tracker
                updated_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
                sp.members = [m for m in sp.members if m.id in updated_member_ids]
        
        # 4. Save tracker after Phase 1 (critical state changes)
        # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
        # NOTE: Species_id is NOT updated in elites.json/reserves.json (file distribution happens in Phase 7)
        _save_tracker_if_dirty(state)
        # Validate tracker consistency after Phase 1
        _validate_tracker_consistency(state, "Phase 1")
    
    # ========================================================================
    # PHASE 2: CLUSTER 0 SPECIATION (ISOLATED)
    # ========================================================================
    # When elites.json is empty, skip Phase 1 and start with Phase 2 directly
    # - Load all genomes with species_id=0 from genome_tracker
    # - Apply isolated cluster 0 speciation (form new species from reserves)
    # - NO radius enforcement or capacity enforcement in Phase 2
    # - Only trackers are updated, not file genomes
    
    state["logger"].info("=== Phase 2: Cluster 0 Speciation (Isolated) ===")
    
    # 4. Load cluster 0 from genome_tracker.json (all genomes with species_id=0)
    # Use genome tracker as single source of truth to collect all reserves genomes
    outputs_path = get_outputs_path()
    
    # Get all genome IDs with species_id=0 from tracker
    if "_genome_tracker" not in state:
        state["logger"].warning("Genome tracker not available, cannot collect reserves")
    else:
        reserves_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(0)
        state["logger"].info(f"Found {len(reserves_genome_ids)} genomes with species_id=0 in tracker")
        
        # Load actual genome data from multiple sources (elites.json, temp.json, reserves.json)
        reserves_genomes = _load_genomes_by_ids(reserves_genome_ids, outputs_path, state["logger"])
        state["logger"].debug(f"Loaded {len(reserves_genomes)} genome data entries for reserves (from {len(reserves_genome_ids)} IDs in tracker)")
        
        # Get IDs of individuals that are now in species (species_id > 0)
        species_member_ids = set()
        for sp in state["species"].values():
            for member in sp.members:
                species_member_ids.add(member.id)
        
        # Remove from cluster0.members any that are now in species
        removed_count = 0
        cluster0_members_to_keep = []
        for cm in state["cluster0"].members:
            if cm.individual.id not in species_member_ids:
                cluster0_members_to_keep.append(cm)
            else:
                removed_count += 1
        
        state["cluster0"].members = cluster0_members_to_keep
        
        # Add all reserves genomes to cluster0 (only those with embeddings)
        existing_cluster0_ids = {cm.individual.id for cm in state["cluster0"].members}
        added_count = 0
        for genome in reserves_genomes:
            genome_id = genome.get("id")
            if genome_id and genome_id not in existing_cluster0_ids and genome_id not in species_member_ids:
                # Only add if has embedding
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
    
    # 7. Sort all genomes with species_id=0 by fitness (descending) and apply isolated speciation
    # Get all individuals from cluster0 with embeddings
    cluster0_individuals = [ind for ind in state["cluster0"].individuals if getattr(ind, "embedding", None) is not None]
    
    # Sort by fitness (descending) - highest fitness processed first
    sorted_individuals = sorted(cluster0_individuals, key=lambda x: x.fitness, reverse=True)
    
    # Apply isolated cluster 0 speciation (two-phase: collect groups, then form species)
    new_species_from_cluster0 = cluster0_speciation_isolated(
        current_generation=current_generation,
        config=state["config"],
        logger=state["logger"],
        pre_sorted_individuals=sorted_individuals  # Pass pre-sorted individuals
    )
    
    # Add newly formed species to state and immediately update all files
    newly_formed_species_ids = set()
    speciation_state_path = outputs_path / "speciation_state.json"
    
    for new_species in new_species_from_cluster0:
        state["species"][new_species.id] = new_species
        newly_formed_species_ids.add(new_species.id)
        state["_current_gen_events"]["speciation"] += 1
        state["logger"].info(f"Species {new_species.id} formed from cluster 0 ({new_species.size} members)")
        
        # Update genome tracker for all members of new species (immediate)
        if "_genome_tracker" in state:
            updates = {str(m.id): new_species.id for m in new_species.members}
            result = state["_genome_tracker"].batch_update(
                updates, current_generation, f"species_formed_from_cluster0_{new_species.id}"
            )
            if result["failed"] > 0:
                state["logger"].warning(f"Tracker update failed for {result['failed']} genomes in new species {new_species.id}")
            # Force save tracker immediately after each species formation
            state["_genome_tracker"].save()
        
        # Update speciation_state.json immediately after each species formation
        # Create file if it doesn't exist (e.g., Generation 0)
        try:
            existing_state = {}
            file_existed = speciation_state_path.exists()
            if file_existed:
                # Read existing state to preserve other fields
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
            
            # Update with current species state
            state_dict = {
                "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                "generation": current_generation
            }
            # Preserve other fields (including incubators and extinct)
            state_dict["cluster0"] = existing_state.get("cluster0", {})
            state_dict["global_best_id"] = existing_state.get("global_best_id")
            state_dict["metrics"] = existing_state.get("metrics", {})
            state_dict["incubators"] = existing_state.get("incubators", [])
            state_dict["extinct"] = existing_state.get("extinct", [])
            
            # Save immediately (create file if it doesn't exist)
            with open(speciation_state_path, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
            
            if not file_existed:
                state["logger"].debug(f"Created speciation_state.json for Generation {current_generation}")
        except Exception as e:
            state["logger"].warning(f"Failed to update speciation_state.json immediately for species {new_species.id}: {e}")
        
        # Track speciation events (immediate)
        if "_events_tracker" in state:
            for member in new_species.members:
                state["_events_tracker"].log(
                    member.id, "species_formed_from_cluster0",
                    {"species_id": new_species.id, "size": new_species.size}
                )
    
    # 5. NO radius enforcement for newly formed species (Flow 2 requirement)
    # All members that were added as followers are kept, regardless of distance to leader
    state["logger"].debug("Skipping radius cleanup for newly formed species (Flow 2: no radius enforcement)")
    
    # 6. NO capacity enforcement in Phase 2 (moved to Phase 4, after merging)
    state["logger"].debug("Skipping capacity enforcement in Phase 2 (moved to Phase 4, after merging)")
    
    # 7. Save tracker after Phase 2 (critical state changes)
    # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # NOTE: Species_id is NOT updated in elites.json/reserves.json (file distribution happens in Phase 7)
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 2
    _validate_tracker_consistency(state, "Phase 2")
    
    # Validate Flow 2 requirements for newly formed species
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
            for error in errors[:5]:  # Log first 5 errors
                state["logger"].warning(f"  - {error}")
        else:
            state["logger"].debug(f"Flow 2 validation passed for {len(newly_formed_species_ids)} newly formed species")
    
    # ========================================================================
    # PHASE 3: MERGING + RADIUS ENFORCEMENT
    # ========================================================================
    # Merging of all species, then radius enforcement after all merging is complete
    # - Iterative merging with immediate tracker updates after each merge
    # - Radius enforcement after all merging is complete
    # - Only trackers are updated, not file genomes
    
    state["logger"].info("=== Phase 3: Merging + Radius Enforcement ===")
    
    # 6. Merging of all species (existing + newly formed)
    # Load all species leaders from speciation_state.json and perform iterative merging
    # NOTE: record_fitness() is called ONCE per generation in Phase 5 (Freeze & Incubator)
    # to avoid double-incrementing stagnation. We skip it here to prevent calling it twice.
    # This ensures stagnation only increments once per generation, preventing premature freezing.
    
    outputs_path = get_outputs_path()
    speciation_state_path = outputs_path / "speciation_state.json"
    
    # Load leaders from file
    species_leaders = _load_species_leaders_from_state(outputs_path, state["logger"])
    
    # Also include in-memory species (newly formed from Phase 2) that might not be in file yet
    for sid, sp in state["species"].items():
        if sid not in species_leaders and sp.leader and sp.leader.embedding is not None:
            species_leaders[sid] = (sp.leader, sp.leader.embedding, sp.leader.phenotype)
    
    # Build species info dict for merging
    species_info = {}
    for sid, (leader, embedding, phenotype) in species_leaders.items():
        # Get full species from in-memory state
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
    
    # Iterative merging with immediate file updates
    from .distance import ensemble_distance
    from .merging import merge_islands
    
    species_count_before_merge = len(state["species"])
    merge_count = 0
    
    while True:
        # Find merge candidates
        merge_candidates = []
        species_list = list(species_info.items())
        
        for i, (id1, info1) in enumerate(species_list):
            for j, (id2, info2) in enumerate(species_list[i + 1:], start=i + 1):
                # Check stability
                sp1_stable = (current_generation - info1["created_at"]) >= 1
                sp2_stable = (current_generation - info2["created_at"]) >= 1
                
                if not (sp1_stable and sp2_stable):
                    continue
                
                # Check distance
                dist = ensemble_distance(
                    info1["embedding"], info2["embedding"],
                    info1["phenotype"], info2["phenotype"],
                    state["config"].w_genotype, state["config"].w_phenotype
                )
                
                if dist < state["config"].theta_merge:
                    merge_candidates.append((id1, id2, info1, info2))
        
        if not merge_candidates:
            break
        
        # Merge first candidate
        id1, id2, info1, info2 = merge_candidates[0]
        sp1 = info1["species"]
        sp2 = info2["species"]
        
        # Perform merge (returns merged_species, outliers - outliers will be empty)
        merged_species, _ = merge_islands(
            sp1, sp2, current_generation,
            state["config"].theta_sim,
            state["config"].w_genotype,
            state["config"].w_phenotype,
            state["logger"]
        )
        
        # Update in-memory state
        del state["species"][id1]
        del state["species"][id2]
        state["species"][merged_species.id] = merged_species
        
        # Mark parents as extinct
        sp1.species_state = "extinct"
        sp2.species_state = "extinct"
        state["historical_species"][id1] = sp1
        state["historical_species"][id2] = sp2
        
        # Update species_info for next iteration
        del species_info[id1]
        del species_info[id2]
        species_info[merged_species.id] = {
            "species": merged_species,
            "leader": merged_species.leader,
            "embedding": merged_species.leader.embedding,
            "phenotype": merged_species.leader.phenotype,
            "created_at": merged_species.created_at
        }
        
        # IMMEDIATE FILE UPDATES after each merge
        # 1. Update genome tracker for all members (merge_islands already does this, but ensure it's saved)
        if "_genome_tracker" in state:
            state["_genome_tracker"].save()  # Force save (merge_islands already updated tracker)
        
        # 2. Update speciation_state.json
        if speciation_state_path.exists():
            try:
                with open(speciation_state_path, 'r', encoding='utf-8') as f:
                    existing_state = json.load(f)
                
                state_dict = {
                    "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                    "generation": current_generation
                }
                # Preserve other fields (including incubators and extinct so we don't lose them)
                state_dict["cluster0"] = existing_state.get("cluster0", {})
                state_dict["global_best_id"] = existing_state.get("global_best_id")
                state_dict["metrics"] = existing_state.get("metrics", {})
                state_dict["incubators"] = existing_state.get("incubators", [])
                state_dict["extinct"] = existing_state.get("extinct", [])
                
                with open(speciation_state_path, 'w', encoding='utf-8') as f:
                    json.dump(state_dict, f, indent=2, ensure_ascii=False)
            except Exception as e:
                state["logger"].warning(f"Failed to update speciation_state.json after merge: {e}")
        
        # 3. Update events tracker
        if "_events_tracker" in state:
            for member in merged_species.members:
                state["_events_tracker"].log(
                    member.id, "species_merged",
                    {"from_species": [id1, id2], "to_species": merged_species.id}
                )
            state["_events_tracker"].save()  # Force save immediately after each merge
        
        merge_count += 1
        state["logger"].info(f"Merge {merge_count}: {id1}+{id2}->{merged_species.id} (immediate updates completed)")
    
    species_count_after_merge = len(state["species"])
    state["_current_gen_events"]["merge"] = merge_count
    
    # Verify merge logic: species count should decrease by number of merges
    expected_species_after_merge = species_count_before_merge - merge_count
    if species_count_after_merge != expected_species_after_merge:
        state["logger"].warning(f"Merge count mismatch: before={species_count_before_merge}, after={species_count_after_merge}, merges={merge_count}, expected_after={expected_species_after_merge}")
    
    # Verify parent_ids are set correctly for merged species
    for sid, sp in state["species"].items():
        if sp.cluster_origin == "merge" and sp.parent_ids:
            state["logger"].debug(f"Merge verification: species {sid} from {sp.parent_ids}, origin={sp.cluster_origin}")
    
    # 7. Radius enforcement after all merging is complete
    # Check radius with leader of merged species, mark species_id=0 for removed genomes
    # NOTE: This happens AFTER all merging is complete
    import numpy as np
    state["logger"].info("=== Phase 3: Step 7 - Radius Enforcement (after all merging) ===")
    
    for sid, sp in list(state["species"].items()):
        if sp.leader is None or sp.leader.embedding is None:
            continue
        
        # Get all members from genome_tracker for this species
        if "_genome_tracker" in state:
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
        else:
            species_genome_ids = [m.id for m in sp.members]
        
        # Load actual genome data
        all_member_genomes = _load_genomes_by_ids(species_genome_ids, outputs_path, state["logger"])
        
        # Check radius for each member
        members_to_remove = []
        for genome in all_member_genomes:
            genome_id = genome.get("id")
            if genome_id == sp.leader.id:
                continue  # Leader always stays
            
            genome_embedding = genome.get("prompt_embedding")
            if genome_embedding is None:
                members_to_remove.append(genome_id)
                continue
            
            # Extract phenotype
            from .phenotype_distance import extract_phenotype_vector
            genome_phenotype = extract_phenotype_vector(genome, logger=state["logger"])
            
            dist = ensemble_distance(
                np.array(genome_embedding), sp.leader.embedding,
                genome_phenotype, sp.leader.phenotype,
                state["config"].w_genotype, state["config"].w_phenotype
            )
            
            if dist >= state["config"].theta_sim:
                members_to_remove.append(genome_id)
        
        # Update tracker for removed members (mark as species_id=0)
        if members_to_remove:
            state["logger"].debug(f"Species {sid}: removing {len(members_to_remove)} members outside radius")
            for genome_id in members_to_remove:
                state["_genome_tracker"].update_species_id(
                    str(genome_id), CLUSTER_0_ID, current_generation, "radius_enforcement_to_reserves_after_merge"
                )
                # Log radius enforcement event
                if "_events_tracker" in state:
                    state["_events_tracker"].log(
                        str(genome_id), "radius_enforcement_after_merge",
                        {"from_species": sid, "to_species": CLUSTER_0_ID, "reason": "outside_radius"}
                    )
            
            # Update in-memory members
            updated_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            sp.members = [m for m in sp.members if m.id in updated_member_ids]
            
            # Check if species is now empty or too small
            if len(sp.members) <= 1:
                # Species is empty (only leader) - move to incubator state
                sp.species_state = "incubator"
                sp.members = []  # Clear members
                
                # Move leader to cluster 0 (if leader exists)
                if sp.leader and state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                    # Update genome tracker immediately
                    if "_genome_tracker" in state:
                        state["_genome_tracker"].update_species_id(
                            str(sp.leader.id), CLUSTER_0_ID, current_generation, "radius_enforcement_leader_to_reserves_after_merge"
                        )
                        # Log radius enforcement event for leader
                        if "_events_tracker" in state:
                            state["_events_tracker"].log(
                                str(sp.leader.id), "radius_enforcement_after_merge",
                                {"from_species": sid, "to_species": CLUSTER_0_ID, "reason": "species_empty_after_radius_cleanup"}
                            )
                
                # NOTE: Do NOT delete species here - keep in state["species"] with incubator state
                # Phase 5 Step 9 will run record_fitness() for stagnation tracking
                # Phase 5 Step 11 will handle actual removal to historical_species
                state["logger"].info(f"Phase 3: Species {sid} became empty after radius cleanup - marked as incubator (will be processed in Phase 5)")
            elif len(sp.members) < state["config"].min_island_size:
                # Species too small after cleanup - mark for incubator (will be processed in Phase 5 Step 18)
                state["logger"].debug(f"Phase 3: Species {sid} size={len(sp.members)} < min_island_size={state['config'].min_island_size} after radius cleanup - will be moved to incubator in Phase 5 Step 18")
    
    # Update all files after radius enforcement
    _save_tracker_if_dirty(state)
    # Update events tracker after radius enforcement
    if "_events_tracker" in state:
        state["_events_tracker"].save()  # Force save immediately after radius enforcement
    # Update speciation_state.json after radius enforcement
    if speciation_state_path.exists():
        try:
            with open(speciation_state_path, 'r', encoding='utf-8') as f:
                existing_state = json.load(f)
            
            state_dict = {
                "species": {str(sid): sp.to_dict() for sid, sp in state["species"].items()},
                "generation": current_generation
            }
            # Preserve other fields (including incubators and extinct)
            state_dict["cluster0"] = existing_state.get("cluster0", {})
            state_dict["global_best_id"] = existing_state.get("global_best_id")
            state_dict["metrics"] = existing_state.get("metrics", {})
            state_dict["incubators"] = existing_state.get("incubators", [])
            state_dict["extinct"] = existing_state.get("extinct", [])
            
            with open(speciation_state_path, 'w', encoding='utf-8') as f:
                json.dump(state_dict, f, indent=2, ensure_ascii=False)
        except Exception as e:
            state["logger"].warning(f"Failed to update speciation_state.json after radius enforcement: {e}")
    
    # Save tracker after Phase 3 (critical state changes)
    # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # NOTE: Species_id is NOT updated in elites.json/reserves.json (file distribution happens in Phase 7)
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 3
    _validate_tracker_consistency(state, "Phase 3")
    
    # ========================================================================
    # PHASE 4: CAPACITY ENFORCEMENT (species_id > 0)
    # ========================================================================
    # Capacity enforcement for all species with species_id > 0
    # - Uses genome_tracker as authoritative source
    # - Does NOT update elites.json/reserves.json (only trackers)
    # - After Phase 4, all species have correct members (within radius, within capacity)
    
    state["logger"].info("=== Phase 4: Capacity Enforcement (species_id > 0) ===")
    
    # NOTE: Radius enforcement is done in Phase 1 (after variant processing) and Phase 3 (after merging)
    # Phase 4 only enforces capacity limits for species_id > 0
    
    # CRITICAL FIX: Process ALL species_id > 0 from tracker, not just those in state["species"]
    # Some species may be removed from state["species"] during Phase 3 (radius cleanup) but still have
    # genomes in the tracker that need capacity enforcement.
    all_phase4_species_ids = set(state["species"].keys())  # Default to in-memory species
    
    if "_genome_tracker" not in state:
        state["logger"].error("Genome tracker not available for capacity enforcement")
    else:
        # Get all species IDs from tracker that have genomes (species_id > 0)
        tracker_stats = state["_genome_tracker"].get_distribution_stats()
        tracker_species_ids = {int(sid) for sid, count in tracker_stats["by_species_id"].items() 
                               if int(sid) > 0 and count > 0}
        
        # Combine with species in state["species"]
        # CRITICAL: Normalize keys to int (state["species"] keys are int after load_state, but ensure consistency)
        in_memory_species_ids = {int(sid) for sid in state["species"].keys()}
        all_phase4_species_ids = tracker_species_ids | in_memory_species_ids
        
        # Log species not in state["species"] but in tracker (these were previously missed)
        tracker_only_species = tracker_species_ids - in_memory_species_ids
        if tracker_only_species:
            state["logger"].info(
                f"Phase 4: Found {len(tracker_only_species)} species in tracker but not in state['species']: {sorted(tracker_only_species)}"
            )
    
    # 8. Capacity enforcement for ALL species (species_id > 0)
    # CRITICAL: Capacity enforcement considers ALL genomes from genome_tracker (all generations), not just in-memory members
    # NOTE: genome_tracker.json is the authoritative source - we use it to get IDs, then fetch details as needed
    # NOTE: Does NOT update elites.json/reserves.json (only trackers are updated)
    outputs_path = get_outputs_path()
    
    # Process all species IDs (from both state["species"] and tracker)
    for sid in sorted(all_phase4_species_ids):
        # Check if this species is in state["species"] (has in-memory representation)
        # CRITICAL: Try both int and str keys for backward compatibility
        sp = state["species"].get(sid) or state["species"].get(str(sid))
        
        # Log if processing tracker-only species (not in state["species"])
        if sp is None:
            state["logger"].debug(f"Phase 4: Processing tracker-only species {sid} (not in state['species'])")
        
        # Load ALL genomes for this species (need full genome data with fitness for sorting)
        # Get genome IDs from tracker (authoritative source), then load actual genome data using helper
        all_species_genomes = []
        
        # Get all genome IDs for this species from genome_tracker (authoritative source)
        if "_genome_tracker" not in state:
            state["logger"].error("Genome tracker not available for capacity enforcement")
            continue
        
        species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
        species_genome_ids_set = set(species_genome_ids)  # Convert to set for efficient lookup
        
        # Get in-memory member IDs (current generation genomes that may not be in tracker yet)
        # Only if species has in-memory representation
        in_memory_ids = set()
        if sp is not None:
            in_memory_ids = {str(m.id) for m in sp.members}
            
            # Register any in-memory members not yet in tracker
            for member in sp.members:
                member_id_str = str(member.id)
                if member_id_str not in species_genome_ids_set:
                    state["_genome_tracker"].register(member_id_str, sid, current_generation)
                    species_genome_ids_set.add(member_id_str)
        
        # Combine all IDs and fetch genome details using helper (loads from elites.json, temp.json, reserves.json)
        # Note: We use the helper to fetch details, but don't trust species_id in those files - tracker is authoritative
        all_genome_ids = list(species_genome_ids_set | in_memory_ids)
        loaded_genomes = _load_genomes_by_ids(all_genome_ids, outputs_path, state["logger"])
        
        # Build map of loaded genomes by ID for quick lookup
        loaded_by_id = {str(g.get("id")): g for g in loaded_genomes if g.get("id") is not None}
        
        # Add all loaded genomes to all_species_genomes
        all_species_genomes.extend(loaded_genomes)
        
        # For in-memory members not found in files, convert using helper
        # Only if species has in-memory representation
        if sp is not None:
            for member in sp.members:
                member_id_str = str(member.id)
                if member_id_str not in loaded_by_id:
                    # Convert Individual to genome dict (not found in files yet)
                    genome = _individual_to_genome_dict(member, current_generation)
                    genome["species_id"] = sid
                    all_species_genomes.append(genome)
        
        # Log if genomes are missing (debugging)
        if len(all_species_genomes) < len(all_genome_ids):
            missing = set(all_genome_ids) - {str(g.get("id")) for g in all_species_genomes if g.get("id")}
            if missing:
                state["logger"].warning(
                    f"Phase 4: Species {sid} - {len(missing)} genomes from tracker not found in files: {list(missing)[:5]}"
                )
        
        # Sort ALL genomes by fitness (descending) - from all generations
        from utils.population_io import _extract_north_star_score
        # Filter out genomes without valid fitness scores (shouldn't happen, but handle gracefully)
        valid_genomes = []
        invalid_genomes = []
        for g in all_species_genomes:
            fitness = _extract_north_star_score(g, "toxicity")
            if fitness is not None:
                valid_genomes.append((g, fitness))
            else:
                invalid_genomes.append(g)
                state["logger"].warning(f"Phase 4: Genome {g.get('id')} in species {sid} has no valid fitness score, excluding from capacity enforcement")
        
        # Sort by fitness
        valid_genomes.sort(key=lambda x: x[1], reverse=True)
        all_species_genomes = [g for g, _ in valid_genomes] + invalid_genomes
        
        # Log species state for debugging
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
        
        # Keep top species_capacity, archive the rest
        if len(all_species_genomes) > state["config"].species_capacity:
            state["logger"].info(
                f"Phase 4: Species {sid} exceeds capacity: {len(all_species_genomes)} genomes "
                f"(capacity: {state['config'].species_capacity}), will archive {len(all_species_genomes) - state['config'].species_capacity}"
                + (f" (tracker-only species)" if sp is None else "")
            )
            keep_genomes = all_species_genomes[:state["config"].species_capacity]
            excess_genomes = all_species_genomes[state["config"].species_capacity:]
            
            # Update in-memory members to match kept genomes (only if species has in-memory representation)
            if sp is not None:
                keep_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
                sp.members = [m for m in sp.members if m.id in keep_ids]
                
                # Add any kept genomes that aren't in in-memory members yet (from previous generations)
                for genome in keep_genomes:
                    gid = genome.get("id")
                    if gid and not any(m.id == gid for m in sp.members):
                        # Create Individual from genome if needed
                        from .species import Individual
                        ind = Individual.from_genome(genome)
                        sp.members.append(ind)
                
                # Reselect leader as highest fitness member (after capacity enforcement)
                # For merged species: leader was already selected during merge (highest fitness from all members)
                # Only update if a new genome with higher fitness was added (not applicable in Phase 4)
                # For non-merged species: ensure leader is highest fitness
                if sp.members:
                    if sp.cluster_origin == "merge":
                        # Merged species: leader was set during merge, only update if higher fitness genome exists
                        current_leader_fitness = sp.leader.fitness if sp.leader else float('-inf')
                        highest_fitness_member = max(sp.members, key=lambda x: x.fitness)
                        if highest_fitness_member.fitness > current_leader_fitness:
                            state["logger"].debug(f"Phase 4: Merged species {sid} leader updated: {sp.leader.id} -> {highest_fitness_member.id} (new higher fitness genome)")
                            sp.leader = highest_fitness_member
                    else:
                        # Non-merged species: ensure leader is highest fitness
                        new_leader = max(sp.members, key=lambda x: x.fitness)
                        if new_leader != sp.leader:
                            state["logger"].debug(f"Phase 4: Species {sid} leader updated: {sp.leader.id if sp.leader else None} -> {new_leader.id} (after capacity enforcement)")
                        sp.leader = new_leader
                    
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
            
            # Update genome tracker FIRST: mark excess genomes as archived (species_id=-1)
            # This must happen before archiving to ensure tracker is authoritative
            if "_genome_tracker" in state:
                # Extract genome IDs directly from excess_genomes (more reliable than Individual conversion)
                excess_ids = [str(g.get("id")) for g in excess_genomes if g.get("id") is not None]
                if excess_ids:
                    # Verify current state BEFORE update
                    before_count = len(state["_genome_tracker"].get_all_genomes_by_species(sid))
                    
                    updates = {gid: -1 for gid in excess_ids}
                    result = state["_genome_tracker"].batch_update(updates, current_generation, f"capacity_archived_species_{sid}")
                    
                    # Verify update succeeded by checking tracker state AFTER update
                    after_count = len(state["_genome_tracker"].get_all_genomes_by_species(sid))
                    expected_after = before_count - len(excess_ids)
                    
                    if result["failed"] > 0:
                        state["logger"].error(
                            f"CRITICAL: Genome tracker batch update failed for {result['failed']}/{result['total']} "
                            f"genomes during capacity enforcement for species {sid}"
                        )
                        state["logger"].error(f"Failed genome IDs: {result.get('failed_genome_ids', [])[:10]}")
                        # Retry failed ones individually
                        for failed_id in result.get('failed_genome_ids', []):
                            try:
                                success, _ = state["_genome_tracker"].update_species_id(
                                    failed_id, -1, current_generation, f"capacity_archived_species_{sid}_retry"
                                )
                                if not success:
                                    state["logger"].error(f"Retry failed for genome {failed_id}")
                            except Exception as e:
                                state["logger"].error(f"Retry exception for genome {failed_id}: {e}")
                    
                    # Verify the update actually worked
                    if after_count != expected_after:
                        state["logger"].error(
                            f"CRITICAL: Tracker update verification failed for species {sid}: "
                            f"expected {expected_after} genomes after archiving, got {after_count}. "
                            f"Before: {before_count}, Excess: {len(excess_ids)}"
                        )
                        # Check which excess genomes are still in species
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
                    
                    # Force immediate save for this species to ensure persistence
                    if state["_genome_tracker"]._dirty:
                        save_success = state["_genome_tracker"].save()
                        if not save_success:
                            state["logger"].error(f"CRITICAL: Failed to save tracker after capacity enforcement for species {sid}")
                        else:
                            state["logger"].debug(f"Phase 4: Saved tracker after archiving {len(excess_ids)} genomes for species {sid}")
                else:
                    state["logger"].warning(f"Phase 4: No valid genome IDs found in excess_genomes for species {sid}")
            
            # Archive excess genomes (convert to Individual for archiving)
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
            
            # Track archival events
            if "_events_tracker" in state:
                for ind in excess_individuals:
                    state["_events_tracker"].log(
                        ind.id, "capacity_archived",
                        {"species_id": sid, "reason": "species_capacity", "capacity": state["config"].species_capacity}
                    )
            
            # NOTE: Files are updated in Phase 7 (redistribution) based on genome_tracker.
            # Tracker is authoritative - files reflect tracker state after Phase 7 distribution.
            # We do NOT update elites.json here - Phase 7 handles all file distribution based on tracker state.
            
            state["logger"].info(f"Phase 4: Species {sid} capacity enforced ({state['config'].species_capacity}), archived {len(excess_genomes)} excess genomes from {len(all_species_genomes)} total (all generations)")
        else:
            # Species within capacity - log for debugging
            state["logger"].debug(f"Phase 4: Species {sid} within capacity ({len(all_species_genomes)}/{state['config'].species_capacity})")
    
    # 14. Validate no duplicate leader IDs across all species
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
        # Fix duplicates by reassigning leaders
        for leader_id, species_ids in duplicate_leaders:
            state["logger"].warning(f"Duplicate leader ID {leader_id} found in species {species_ids}, fixing...")
            # Keep the first species, fix the others
            for sid in species_ids[1:]:
                if sid not in state["species"]:
                    continue
                sp = state["species"][sid]
                # Find the old leader in members and remove it
                old_leader = None
                for member in sp.members:
                    if member.id == leader_id:
                        old_leader = member
                        break
                
                if old_leader:
                    sp.members.remove(old_leader)
                    old_leader.species_id = None
                    # Update tracker if old leader's species_id changed
                    if "_genome_tracker" in state:
                        # If old leader is moved to cluster0 or different species, update tracker
                        if old_leader.species_id is not None:
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), old_leader.species_id, current_generation, "duplicate_leader_fix"
                            )
                        else:
                            # If species_id is None, move to cluster0
                            state["_genome_tracker"].update_species_id(
                                str(old_leader.id), CLUSTER_0_ID, current_generation, "duplicate_leader_fix_to_reserves"
                            )
                
                if len(sp.members) > 0:
                    # Reassign to next highest fitness from remaining members
                    sp.leader = max(sp.members, key=lambda x: x.fitness)
                    # Ensure new leader is in members (should be, but verify)
                    if sp.leader not in sp.members:
                        sp.members.insert(0, sp.leader)
                    state["logger"].info(f"Reassigned species {sid} leader to genome {sp.leader.id} (fitness={sp.leader.fitness:.4f})")
                else:
                    # No other members - mark for incubator (Phase 5 Step 18 will handle cleanup)
                    sp.species_state = "incubator"
                    sp.leader = None  # No leader if no members
                    state["logger"].info(f"Species {sid} has no other members, marking as incubator (will be processed in Phase 5 Step 18)")
    
    # Save tracker after Phase 4 (critical state changes)
    # CRITICAL: Save tracker immediately after capacity enforcement to ensure updates are persisted
    # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # NOTE: Species_id is NOT updated in elites.json/reserves.json (file distribution happens in Phase 7)
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 4
    _validate_tracker_consistency(state, "Phase 4")
    
    # Additional validation: Check that capacity was actually enforced
    for sid, sp in state["species"].items():
        if "_genome_tracker" in state:
            species_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            if len(species_genome_ids) > state["config"].species_capacity:
                state["logger"].error(
                    f"CRITICAL: After Phase 4, species {sid} still has {len(species_genome_ids)} genomes "
                    f"(exceeds capacity {state['config'].species_capacity})"
                )
                # Log which genomes are still in species
                state["logger"].error(f"  Genomes still in species {sid}: {species_genome_ids[:10]}")
                # Check if any excess genomes are marked as archived
                # Get the excess genomes that should have been archived
                # (This requires tracking which genomes were excess during capacity enforcement)
                # For now, just log the current state
                archived_count = len([
                    gid for gid, data in state["_genome_tracker"].genomes.items()
                    if data.get("species_id") == -1
                ])
                state["logger"].error(f"  Total archived genomes in tracker: {archived_count}")
    
    # ========================================================================
    # PHASE 5: STAGNATION AND INCUBATION
    # ========================================================================
    # Stagnation and incubation process
    # - Record fitness and update stagnation counters
    # - Freeze stagnant species (excluded from parent selection, but still alive)
    # - Incubate small species (extinction/dissolution - move to cluster 0)
    # - Only trackers are updated, not file genomes
    
    state["logger"].info("=== Phase 5: Stagnation and Incubation ===")
    
    # 9. Record fitness for ALL species (not just those with new members)
    # Stagnation only increments if species was selected as parent AND no improvement
    # Load parents.json to determine which species were selected as parents
    selected_species_ids = set()
    if current_generation > 0:  # Generation 0 has no parents
        try:
            outputs_path = get_outputs_path()
            parents_path = outputs_path / "parents.json"
            if parents_path.exists():
                with open(parents_path, 'r', encoding='utf-8') as f:
                    parents = json.load(f)
                if isinstance(parents, list):
                    for parent in parents:
                        species_id = parent.get("species_id")
                        # Only track actual species (id > 0). Cluster 0 (reserves) is not in state["species"].
                        if species_id is not None and species_id != 0:
                            selected_species_ids.add(int(species_id))
                    state["logger"].debug(f"Loaded {len(selected_species_ids)} species from parents.json: {sorted(selected_species_ids)}")
                if len(parents) > 0 and len(selected_species_ids) == 0:
                    # All parents from reserves (species_id=0) or missing species_id -> stagnation never increments
                    sid_vals = [p.get("species_id") for p in parents]
                    state["logger"].info(
                        "Stagnation: selected_species_ids is empty (all parents species_id in %s). "
                        "Stagnation only increments when a non-reserve species is selected and does not improve.",
                        sid_vals
                    )
        except Exception as e:
            state["logger"].warning(f"Failed to load parents.json to determine selected species: {e}")
    
    for sid, sp in state["species"].items():
        # max_fitness = actual max over current members only (in case members were removed in radius/capacity)
        sp.max_fitness = max((m.fitness for m in sp.members), default=0.0)
        # CRITICAL: state["species"] keys can be str (from JSON load) or int (from merge); selected_species_ids and _prev_max_fitness use int keys
        sid_int = int(sid)
        was_selected = sid_int in selected_species_ids
        # Species created this gen: sid not in _prev_max_fitness -> treat as increased
        prev_max = state.get("_prev_max_fitness", {}).get(sid_int, -1)
        max_fitness_increased = sp.max_fitness > prev_max
        if was_selected and not max_fitness_increased:
            state["logger"].info(
                "Stagnation: species %s would increment (was_selected, max_fitness not increased: %.4f vs prev %.4f)",
                sid, sp.max_fitness, prev_max
            )
        sp.record_fitness(current_generation, was_selected_as_parent=was_selected, max_fitness_increased=max_fitness_increased)
        if sp.stagnation > 0:
            state["logger"].debug(
                "Stagnation updated: species %s stagnation=%d (was_selected=%s, max_fitness_increased=%s)",
                sid, sp.stagnation, was_selected, max_fitness_increased
            )
    
    # 10. Freeze stagnant species (NOT extinction - they stay alive, can merge)
    # Frozen species cannot participate in parent selection (unless category 1 is empty),
    # but they are still alive and can merge if conditions are satisfied.
    state["logger"].info("=== Phase 5: Step 10 - Freeze Stagnant Species ===")
    
    frozen_count = 0
    for sid, sp in list(state["species"].items()):
        if sp.stagnation >= state["config"].species_stagnation and sp.species_state != "frozen":
            sp.species_state = "frozen"
            frozen_count += 1
            # CRITICAL FIX: Increment extinction_events counter when freezing a species
            # This ensures extinction_events in metrics and EvolutionTracker.json reflect actual freezes
            state["_current_gen_events"]["extinction"] += 1
            state["logger"].info(
                f"Frozen species {sid} (stagnation={sp.stagnation} >= {state['config'].species_stagnation}) - "
                f"excluded from parent selection, but still alive and can merge"
            )
    
    # Update trackers after freezing
    if frozen_count > 0:
        _save_tracker_if_dirty(state)
        state["logger"].info(f"Step 10: Frozen {frozen_count} species, extinction_events={state['_current_gen_events']['extinction']} (trackers updated)")
    
    # 11. Incubate small species (extinction/dissolution - move to cluster 0)
    # When species have less members than required (min_island_size), we incubate them.
    # Incubation = extinction/dissolution: all members and leaders get species_id=0 (cluster 0).
    state["logger"].info("=== Phase 5: Step 11 - Incubate Small Species ===")
    
    species_count_before_incubation = len(state["species"])
    cluster0_ids_before = {cm.individual.id for cm in state["cluster0"].members}
    incubator_species = {}
    moved_to_cluster0_events = []
    
    for sid, sp in list(state["species"].items()):
        # Check if species is too small (active, frozen, or already incubator)
        if sp.species_state not in ["active", "frozen", "incubator"]:
            continue
        
        current_size = sp.size
        # Check if newly merged species is too small (dissolve it)
        # A species is "newly merged" if it was created via merge in this generation
        is_newly_merged = (sp.cluster_origin == "merge" and sp.created_at == current_generation)
        
        if sp.species_state == "incubator" or current_size < state["config"].min_island_size:
            # Special handling for newly merged species that are too small
            if is_newly_merged and current_size < state["config"].min_island_size:
                state["logger"].info(f"Phase 5: Newly merged species {sid} has {current_size} members < min_island_size={state['config'].min_island_size} - dissolving to incubator")
            # Check cluster 0 capacity
            if state["cluster0"].size >= state["config"].cluster0_max_capacity:
                state["logger"].debug(f"Cluster 0 at capacity, cannot incubate species {sid}")
                continue
            
            # Move all members (including leader) to cluster 0
            moved_member_ids = []
            member_ids_set = {m.id for m in sp.members}
            
            # Move leader if exists and not in members
            if sp.leader and sp.leader.id not in member_ids_set:
                if state["cluster0"].size < state["config"].cluster0_max_capacity:
                    state["cluster0"].add(sp.leader, current_generation)
                    moved_member_ids.append(sp.leader.id)
            
            # Move all members
            for member in sp.members:
                if state["cluster0"].size >= state["config"].cluster0_max_capacity:
                    break
                state["cluster0"].add(member, current_generation)
                moved_member_ids.append(member.id)
            
            # Update tracker: find ALL genomes with this species_id (tracker is source of truth)
            if "_genome_tracker" in state:
                genomes_to_update = [gid for gid, gdata in state["_genome_tracker"].genomes.items() 
                                    if gdata.get("species_id") == sid]
                # Add in-memory members as backup
                all_ids = set(moved_member_ids) | {sp.leader.id} if sp.leader else set() | {m.id for m in sp.members}
                genomes_to_update.extend(str(mid) for mid in all_ids if str(mid) not in genomes_to_update)
                
                if genomes_to_update:
                    result = state["_genome_tracker"].batch_update(
                        {gid: 0 for gid in genomes_to_update}, current_generation, f"incubated_species_{sid}_to_reserves"
                    )
                    if result["failed"] > 0:
                        state["logger"].warning(f"Tracker update failed for {result['failed']} genomes (species {sid})")
            
            # Mark as incubator and remove from active species
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
    
    # Move incubator species to historical_species for tracking
    for sid, sp in incubator_species.items():
        state["historical_species"][sid] = sp
        state["logger"].debug(f"Moved incubator species {sid} to historical_species (will be tracked by ID only in save_state)")
    
    # Update trackers after incubation
    if moved_to_cluster0_events:
        _save_tracker_if_dirty(state)
        state["_current_gen_events"]["moved_to_cluster0"] = len(moved_to_cluster0_events)
        state["logger"].info(f"Step 11: Incubated {len(moved_to_cluster0_events)} species (trackers updated)")
    
    # Verify incubation logic
    species_count_after_incubation = len(state["species"])
    expected_species_after = species_count_before_incubation - len(moved_to_cluster0_events)
    if species_count_after_incubation != expected_species_after:
        state["logger"].warning(
            f"Incubation count mismatch: before={species_count_before_incubation}, "
            f"after={species_count_after_incubation}, incubated={len(moved_to_cluster0_events)}, "
            f"expected_after={expected_species_after}"
        )
    
    # Save tracker after Phase 5 (critical state changes)
    # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # NOTE: Species_id is NOT updated in elites.json/reserves.json (file distribution happens in Phase 7)
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 5
    _validate_tracker_consistency(state, "Phase 5")
    
    # ========================================================================
    # PHASE 6: CLUSTER 0 CAPACITY ENFORCEMENT (species_id = 0)
    # ========================================================================
    # Capacity enforcement for cluster 0 (species_id = 0)
    # - Uses genome_tracker as authoritative source
    # - Does NOT update reserves.json (only trackers)
    
    state["logger"].info("=== Phase 6: Cluster 0 Capacity Enforcement (species_id = 0) ===")
    
    # 12. Enforce cluster 0 capacity
    # Check all genomes with species_id=0 from genome_tracker (authoritative source)
    # reserves.json is not updated until Phase 7, so we use tracker to get accurate count
    state["logger"].info("=== Phase 6: Step 12 - Cluster 0 Capacity Enforcement ===")
    
    if "_genome_tracker" not in state:
        state["logger"].error("Genome tracker not available for cluster 0 capacity enforcement")
    else:
        # Get all genome IDs with species_id=0 from tracker (authoritative source)
        cluster0_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(CLUSTER_0_ID)
        cluster0_count = len(cluster0_genome_ids)
        
        if cluster0_count > state["config"].cluster0_max_capacity:
            state["logger"].info(
                f"Cluster 0 capacity exceeded: {cluster0_count} genomes (capacity: {state['config'].cluster0_max_capacity})"
            )
            
            # Fetch genome details (fitness) for all cluster 0 genomes
            outputs_path = get_outputs_path()
            cluster0_genomes = _load_genomes_by_ids(cluster0_genome_ids, outputs_path, state["logger"])
            
            # Sort by fitness (descending) - use North Star score (toxicity)
            from utils.population_io import _extract_north_star_score
            cluster0_genomes.sort(key=lambda g: _extract_north_star_score(g, "toxicity"), reverse=True)
            
            # Keep top cluster0_max_capacity, archive the rest
            keep_genomes = cluster0_genomes[:state["config"].cluster0_max_capacity]
            excess_genomes = cluster0_genomes[state["config"].cluster0_max_capacity:]
            
            # Update genome tracker: mark excess genomes as archived (species_id=-1)
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
            
            # Archive excess genomes (convert to Individual for archiving)
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
            
            # Track archival events
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
            
            # Update in-memory cluster0 to match kept genomes (for consistency)
            # Remove excess from in-memory cluster0 if they exist there
            kept_ids = {g.get("id") for g in keep_genomes if g.get("id") is not None}
            excess_ids_set = {g.get("id") for g in excess_genomes if g.get("id") is not None}
            state["cluster0"].members = [
                m for m in state["cluster0"].members 
                if m.individual.id in kept_ids or m.individual.id not in excess_ids_set
            ]
            
            # Save trackers immediately after capacity enforcement
            _save_tracker_if_dirty(state)
            if "_events_tracker" in state:
                state["_events_tracker"].save()
            state["logger"].info(f"Step 19: Cluster 0 capacity enforced, trackers updated immediately")
        else:
            state["logger"].debug(
                f"Cluster 0 within capacity: {cluster0_count}/{state['config'].cluster0_max_capacity}"
            )
    # Save tracker after Phase 6 (critical state changes)
    # NOTE: Only trackers are updated (genome_tracker.json, events_tracker.json, speciation_state.json)
    # NOTE: Species_id is NOT updated in reserves.json (file distribution happens in Phase 7)
    _save_tracker_if_dirty(state)
    # Validate tracker consistency after Phase 6
    _validate_tracker_consistency(state, "Phase 6")
    
    # ========================================================================
    # PHASE 7: REDISTRIBUTION OF GENOMES
    # ========================================================================
    # Update genomes in temp.json, elites.json, reserves.json and distribute correctly
    # - Update species_id in temp.json, elites.json, reserves.json based on genome_tracker
    # - Distribute genomes to correct files (elites.json, reserves.json, archive.json)
    # - This is the ONLY phase that updates species_id in file genomes
    # - Files are synchronized with genome_tracker (authoritative source)
    
    state["logger"].info("=== Phase 7: Redistribution of Genomes ===")
    # Distribute genomes to files based on genome_tracker (authoritative source of truth)
    # This must happen before Phase 8 (metrics) so files exist for metrics calculation
    distribution_result = phase8_redistribute_genomes(
        temp_path=temp_path if temp_path else None,
        current_generation=current_generation
    )
    state["logger"].info(f"Phase 7: Distribution complete - {distribution_result.get('elites_moved', 0)} elites, {distribution_result.get('reserves_moved', 0)} reserves, {distribution_result.get('archived_moved', 0)} archived")
    
    # Update cluster0 size in speciation_state.json to match reserves.json after distribution
    outputs_path = get_outputs_path()
    _update_speciation_state_cluster0_size_after_distribution(outputs_path)
    
    # Also update in-memory cluster0 to match reserves.json (synchronize members list)
    # This ensures in-memory state matches the file state after Phase 7 redistribution
    reserves_path = outputs_path / "reserves.json"
    if reserves_path.exists() and "_genome_tracker" in state:
        try:
            # Get all genome IDs with species_id=0 from tracker (authoritative source)
            reserves_genome_ids = state["_genome_tracker"].get_all_genomes_by_species(0)
            
            # Load actual genome data from reserves.json
            reserves_genomes = _load_genomes_by_ids(reserves_genome_ids, outputs_path, state["logger"])
            
            # Rebuild in-memory cluster0 members from reserves.json
            # Remove genomes that are no longer in reserves (species_id changed)
            reserves_ids_set = {str(g.get("id")) for g in reserves_genomes if g.get("id") is not None}
            state["cluster0"].members = [
                cm for cm in state["cluster0"].members 
                if str(cm.individual.id) in reserves_ids_set
            ]
            
            # Add genomes from reserves.json that aren't in in-memory cluster0
            existing_cluster0_ids = {str(cm.individual.id) for cm in state["cluster0"].members}
            added_count = 0
            for genome in reserves_genomes:
                genome_id = genome.get("id")
                if genome_id is None:
                    continue
                genome_id_str = str(genome_id)
                
                if genome_id_str not in existing_cluster0_ids:
                    # Check if genome has embedding (required for cluster0)
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
    
    # Sync in-memory species members from genome_tracker so sp.size == len(member_ids) == tracker count.
    # Ensures members never exceed species_capacity and saved speciation_state size matches elites.json.
    if "_genome_tracker" in state:
        elites_path = outputs_path / "elites.json"
        for sid, sp in list(state["species"].items()):
            tracker_member_ids = state["_genome_tracker"].get_all_genomes_by_species(sid)
            current_member_ids = {str(m.id) for m in sp.members}
            tracker_set = set(str(mid) for mid in tracker_member_ids)
            if len(tracker_member_ids) != len(sp.members) or tracker_set != current_member_ids:
                # Rebuild sp.members from tracker (authoritative) and elites
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
                    # Ensure leader is first and in list exactly once
                    best = max(new_members, key=lambda x: x.fitness)
                    sp.leader = best
                    sp.members = [best] + [m for m in new_members if m.id != best.id]
                else:
                    sp.members = []
                state["logger"].debug(
                    f"Phase 7: Synced species {sid} members from tracker: {len(sp.members)} (was {len(current_member_ids)})"
                )
    
    # ========================================================================
    # PHASE 8: METRICS & STATISTICS
    # ========================================================================
    # Calculate statistics, metrics and other required things
    # - Update c-TF-IDF labels for all species
    # - Calculate and record metrics from distributed files
    # - Save all state files (speciation_state.json, events_tracker.json, genome_tracker.json)
    
    state["logger"].info("=== Phase 8: Metrics & Statistics ===")
    # NOTE: This is Phase 8 within process_generation().
    # Phase 7 (redistribution) already completed, so files exist for metrics calculation.
    
    # 13. Update c-TF-IDF labels for all species
    from .labeling import update_species_labels
    update_species_labels(
        state["species"],
        current_generation=current_generation,
        n_words=10,
        logger=state["logger"]
    )
    
    # 14. Calculate and record metrics from distributed files
    # Files should exist after Phase 7 (Redistribution)
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    # Files must exist after Phase 7 (Redistribution)
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
    
    # Validate metrics are calculated correctly from files
    is_valid, errors = validate_metrics_from_files(
        outputs_path=outputs_path,
        metrics=metrics.to_dict(),
        logger=state["logger"]
    )
    if not is_valid:
        state["logger"].warning(f"Metrics validation found {len(errors)} errors")
        for error in errors[:5]:  # Log first 5 errors
            state["logger"].warning(f"  - {error}")
    else:
        state["logger"].debug("Metrics validation passed - all metrics match file contents")
    
    # 15. Save all state files
    save_state(str(get_outputs_path() / "speciation_state.json"))
    if "_events_tracker" in state:
        state["_events_tracker"].save()
    if "_genome_tracker" in state:
        state["_genome_tracker"].save()
    
    return state["species"], state["cluster0"]


def cluster0_speciation_isolated(
    current_generation: int, 
    config: "SpeciationConfig", 
    logger=None,
    pre_sorted_individuals: Optional[List[Individual]] = None) -> List[Species]:
    """
    Apply leader-follower clustering on cluster 0 in complete isolation (like generation 0).
    
    Flow 2: Two-phase approach with no leader update and no radius enforcement.
    - Phase 1: Collect all potential leader groups (no species formation)
    - Phase 2: Form species if group size >= min_island_size (keep all members, no filtering)
    
    If pre_sorted_individuals is provided, use those instead of in-memory cluster0.
    Otherwise, fall back to in-memory cluster0.individuals.
    
    Args:
        current_generation: Current generation number
        config: SpeciationConfig object with parameters
        logger: Optional logger instance
        pre_sorted_individuals: Optional list of Individual objects already sorted by fitness (descending)
        
    Returns:
        List of newly formed Species (empty list if none formed)
    """
    from .species import Species, Individual, generate_species_id
    from .distance import ensemble_distance, ensemble_distances_batch
    import numpy as np
    
    if logger is None:
        logger = get_logger("Cluster0SpeciationIsolated")
    
    # Use pre-sorted individuals if provided, otherwise fall back to in-memory cluster0
    if pre_sorted_individuals is not None:
        individuals = [ind for ind in pre_sorted_individuals if getattr(ind, "embedding", None) is not None]
        sorted_individuals = individuals  # Already sorted
    else:
        # Fallback to in-memory cluster0
        state = _get_state()
        cluster0 = state.get("cluster0")
        if cluster0 is None:
            logger.debug("cluster0 not in state, no speciation possible")
            return []
        individuals = [ind for ind in cluster0.individuals if getattr(ind, "embedding", None) is not None]
        # Sort by fitness (descending) - highest fitness processed first
        sorted_individuals = sorted(individuals, key=lambda x: x.fitness, reverse=True)
    
    if len(sorted_individuals) < config.cluster0_min_cluster_size:
        logger.debug(f"Cluster 0 has {len(sorted_individuals)} individuals with embeddings, need {config.cluster0_min_cluster_size} to attempt speciation")
        return []
    
    # PHASE 1: Collect all potential leader groups (NO species formation)
    # Potential leaders: Dict mapping leader_id -> (None, embedding, phenotype, Individual, followers_list)
    # Note: species_id is always None in Phase 1 (no species formed yet)
    # Note: leader_id is an integer (ind.id), not a string
    potential_leaders: Dict[int, Tuple[None, np.ndarray, Optional[np.ndarray], Individual, List[Individual]]] = {}
    
    # First individual becomes first potential leader
    first = sorted_individuals[0]
    potential_leaders[first.id] = (None, first.embedding, first.phenotype, first, [])
    remaining_individuals = sorted_individuals[1:]
    
    # Process ALL remaining individuals (NO early species formation)
    for ind in remaining_individuals:
        assigned = False
        min_dist = float('inf')
        nearest_leader_id = None
        
        # Check against ALL potential leaders (all are active in Phase 1)
        if potential_leaders:
            # Collect all leader embeddings and phenotypes
            leader_embeddings = []
            leader_phenotypes = []
            leader_ids = []
            for pl_id, (_, pl_emb, pl_pheno, _, _) in potential_leaders.items():
                leader_ids.append(pl_id)
                leader_embeddings.append(pl_emb)
                leader_phenotypes.append(pl_pheno)
            
            if len(leader_embeddings) > 1:
                # Vectorized distance computation
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
            
            # If within threshold, add as follower (NO species formation check here)
            if nearest_leader_id is not None and min_dist < config.theta_sim:
                _, pl_emb, pl_pheno, pl_ind, followers = potential_leaders[nearest_leader_id]
                # Add as follower (tracked but no species yet)
                followers.append(ind)
                assigned = True
        
        # If not assigned to any potential leader, become a new potential leader
        if not assigned:
            potential_leaders[ind.id] = (None, ind.embedding, ind.phenotype, ind, [])
    
    # PHASE 2: Form species from groups that meet min_island_size
    new_species_list: List[Species] = []
    individuals_to_remove: List[Individual] = []
    
    for pl_id, (_, pl_emb, pl_pheno, pl_ind, followers) in potential_leaders.items():
        all_members = [pl_ind] + followers
        
        if len(all_members) >= config.min_island_size:
            # Create species with original potential leader (NO update)
            # Keep ALL members (NO radius filtering)
            new_species_id = generate_species_id()
            new_species = Species(
                id=new_species_id,
                leader=pl_ind,  # Original potential leader, NO update
                members=all_members,  # ALL members, NO filtering
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
            # Below min_island_size → all stay in cluster 0 (no removal)
            logger.debug(
                f"Cluster 0 speciation: group with {len(all_members)} members < "
                f"min_island_size {config.min_island_size} → staying in cluster 0"
            )
    
    # Remove formed species members from in-memory cluster 0
    state = _get_state()
    if individuals_to_remove and state.get("cluster0"):
        removed_count = state["cluster0"].remove_batch(individuals_to_remove)
        logger.debug(f"Removed {removed_count} individuals from in-memory cluster 0 (formed {len(new_species_list)} new species)")
    
    logger.info(f"Cluster 0 speciation isolated: formed {len(new_species_list)} new species from {len(individuals)} cluster 0 individuals")
    return new_species_list


def _individual_to_genome_dict(ind: Individual, current_generation: int) -> Dict[str, Any]:
    """
    Convert Individual to genome dictionary format for saving to files.
    
    Preserves all original genome data and adds/updates speciation metadata.
    Ensures embeddings are preserved. Preserves original generation if available.
    
    Args:
        ind: Individual object
        current_generation: Current generation number (used as fallback if generation not in genome_data)
        
    Returns:
        Genome dictionary ready for saving
    """
    import numpy as np
    
    # Start with original genome data if available
    if ind.genome_data:
        genome = ind.genome_data.copy()
        # Preserve original generation if it exists, otherwise use current_generation
        if "generation" not in genome:
            genome["generation"] = current_generation
        # If generation exists, keep it (don't overwrite)
    else:
        # Create minimal dict with required fields
        genome = {
            "id": ind.id,
            "prompt": ind.prompt,
            "generation": current_generation
        }
    
    # Update/add speciation metadata (these can change, so always update)
    genome["species_id"] = ind.species_id
    genome["fitness"] = ind.fitness
    
    # Preserve embedding if available
    if ind.embedding is not None:
        # Convert numpy array to list for JSON serialization
        if isinstance(ind.embedding, np.ndarray):
            genome["prompt_embedding"] = ind.embedding.tolist()
        else:
            genome["prompt_embedding"] = ind.embedding
    
    # Preserve phenotype if available (moderation_result)
    if ind.genome_data and "moderation_result" in ind.genome_data:
        genome["moderation_result"] = ind.genome_data["moderation_result"]
    
    return genome


def _update_speciation_state_cluster0_size_after_distribution(outputs_path) -> None:
    """Update cluster0.size in speciation_state.json to match reserves.json after distribution.
    
    This is called after Phase 7 (redistribution) to ensure cluster0.size in speciation_state.json
    matches the actual count in reserves.json after genomes have been distributed.
    """
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
    """Save state to file.
    
    Note: Config is NOT saved here as it's passed as project arguments or fixed constants.
    The cluster0 section only contains metadata (size, speciation_events), not full member data
    since reserves.json already stores the complete genome data for cluster 0.
    
    Size calculation:
    - For active/frozen species: size = count from member_ids (derived from genome_tracker).
      Capacity enforcement considers ALL genomes from all generations via genome_tracker, sorts by fitness,
      keeps top species_capacity, and archives the rest (species_id=-1 in tracker). The genome_tracker
      is the authoritative source for species membership.
    Species storage strategy:
    - Active species (state="active") - full data saved in species dict (participate in evolution)
    - Frozen species (state="frozen") - full data saved in species dict (stagnated, excluded from parent selection, but still alive)
    - Extinct species (state="extinct") - only species ID tracked (like incubators, not really helpful but preserved for reference)
    - Incubator species (state="incubator") - only species ID tracked (moved to cluster 0, just for tracking)
    
    IMPORTANT: Members Storage:
    - Species members are NOT saved as full Individual objects in speciation_state.json
    - Only member_ids (list of genome IDs) are saved for storage efficiency
    - The members list in memory may be empty even if size > 0 - this is EXPECTED
    - Member data is loaded lazily from genome files (elites.json, etc.) when load_state() is called
    - This is a storage optimization: full member data is in genome files, only IDs are stored in state
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    
    # Calculate actual sizes and member IDs from genome_tracker for each species
    # genome_tracker is the authoritative source for species membership
    outputs_path = get_outputs_path()
    elites_path = outputs_path / "elites.json"
    species_sizes = {}
    species_member_ids = {}  # species_id -> list of all member IDs from tracker
    elites_genomes = []  # Only used for validation (leader embedding lookup)
    
    # Load elites_genomes for validation only (leader embedding lookup for frozen species)
    if elites_path.exists():
        try:
            from utils.population_io import load_elites
            elites_genomes = load_elites(str(elites_path), logger=logger)
        except Exception as e:
            logger.warning(f"Failed to load elites.json for validation: {e}")
            elites_genomes = []
    
    # REQUIRED: genome_tracker must be available for save_state
    # This ensures species membership is tracked correctly and consistently
    if "_genome_tracker" not in state:
        logger.error("Genome tracker is required for save_state() - cannot calculate species member_ids/sizes without it")
        raise RuntimeError("Genome tracker is required for save_state()")
    
    genome_tracker = state["_genome_tracker"]
    # Get member IDs from tracker for each species in state["species"] (authoritative source)
    for species_id in state["species"].keys():
        member_ids = genome_tracker.get_all_genomes_by_species(species_id)
        if member_ids:
            species_member_ids[species_id] = sorted([str(mid) for mid in member_ids])
            species_sizes[species_id] = len(member_ids)
        else:
            # Species exists in state but has no genomes in tracker (e.g. incubator with all members moved to cluster0)
            species_member_ids[species_id] = []
            species_sizes[species_id] = 0
    
    logger.debug(f"Calculated species sizes from tracker: {len(species_sizes)} species in state['species']")
    
    # Build species dict - only save full data for active and frozen species
    species_dict = {}
    incubator_ids = []
    
    def _set_member_ids_and_size(sp_dict, sid, sp, species_member_ids, logger):
        """Helper to set member_ids and size from genome_tracker (via species_member_ids) or in-memory."""
        sid_int = int(sid)
        if sid_int in species_member_ids:
            sp_dict["member_ids"] = species_member_ids[sid_int]
        else:
            # This should not happen since we iterate state["species"].keys() and populate species_member_ids for all
            logger.warning(f"Species {sid} not found in tracker, using in-memory member IDs ({len(sp.members)})")
            sp_dict["member_ids"] = [str(m.id) for m in sp.members]  # Convert to strings for consistency
        sp_dict["size"] = len(sp_dict["member_ids"])
    
    # Add active and frozen species (full data)
    for sid, sp in state["species"].items():
        if sp.species_state in ["active", "frozen"]:
            sp_dict = sp.to_dict()
            _set_member_ids_and_size(sp_dict, sid, sp, species_member_ids, logger)
            
            # Frozen species: ensure leader_embedding is preserved (needed for merging)
            if sp.species_state == "frozen" and sp.leader:
                if sp.leader.embedding is not None:
                    sp_dict["leader_embedding"] = sp.leader.embedding.tolist()
                elif "leader_embedding" not in sp_dict or sp_dict["leader_embedding"] is None:
                    # Try to load from elites.json
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
    
    # Add historical species - only extinct (merged parents) and incubator go here
    # Frozen species are NOT in historical_species - they stay in species dict
    extinct_ids = []  # Track extinct species IDs (like incubators, just IDs)
    for sid, sp in state.get("historical_species", {}).items():
        if str(sid) not in species_dict:  # Avoid duplicates
            if sp.species_state == "extinct":
                # Extinct species - just track ID (like incubators, not really helpful but preserved for reference)
                # CRITICAL: Verify extinct species truly have no active members in genome_tracker (authoritative source)
                # Note: elites.json may still show old species_id until Phase 7 redistribution, but tracker is updated immediately
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
                # Incubator species - just track ID
                incubator_ids.append(sid)
    
    # NOTE: Reconstruction block removed - species_dict is built only from state["species"] (active/frozen)
    # and state["historical_species"] (extinct/incubator IDs only). This ensures:
    # 1. All saved species have labels (update_species_labels runs on state["species"] before save_state)
    # 2. Stagnation is tracked correctly (species remain in state["species"] until Phase 5 handles them)
    # 3. Species count matches EvolutionTracker (no extra reconstructed entries)
    
    # Validate consistency
    from collections import Counter
    leader_ids = [sp_dict.get("leader_id") for sp_dict in species_dict.values() if sp_dict.get("leader_id")]
    duplicates = {lid: count for lid, count in Counter(leader_ids).items() if count > 1}
    if duplicates:
        logger.warning(f"Duplicate leader IDs: {duplicates}")
    
    if elites_path.exists() and elites_genomes:
        for sid_str, sp_dict in species_dict.items():
            try:
                sid = int(sid_str)
                # Convert member_ids to strings for comparison (member_ids are stored as strings)
                member_ids = {str(mid) for mid in sp_dict.get("member_ids", [])}
                # Convert genome IDs from elites.json to strings for comparison
                species_genome_ids = {str(g.get("id")) for g in elites_genomes if g.get("species_id") == sid and g.get("id") is not None}
                leader_id_str = str(sp_dict.get("leader_id")) if sp_dict.get("leader_id") is not None else None
                
                if leader_id_str and leader_id_str not in member_ids:
                    logger.warning(f"Species {sid}: leader_id ({leader_id_str}) not in member_ids")
                if extra := species_genome_ids - member_ids:
                    logger.warning(f"Species {sid}: {len(extra)} genomes in elites.json not in member_ids: {sorted(list(extra))[:5]}")
            except (ValueError, KeyError) as e:
                logger.debug(f"Validation error for species {sid_str}: {e}")
                continue

    # Ensure size always equals len(member_ids) (source of truth: genome_tracker)
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

    # Get cluster 0 size and fitness from reserves.json
    reserves_path = outputs_path / "reserves.json"
    reserves_genomes = _load_json_file(reserves_path, logger, [])
    actual_cluster0_size = len(reserves_genomes) if reserves_genomes else state["cluster0"].size
    
    if actual_cluster0_size != state["cluster0"].size:
        logger.warning(f"Cluster 0 size mismatch: reserves.json={actual_cluster0_size}, in-memory={state['cluster0'].size}")
    
    # Calculate fitness stats
    cluster0_max_fitness = cluster0_min_fitness = 0.0
    if reserves_genomes:
        from utils.population_io import _extract_north_star_score
        cluster0_fitnesses = [_extract_north_star_score(g, "toxicity") or 0.0 for g in reserves_genomes]
        if cluster0_fitnesses:
            cluster0_max_fitness = round(max(cluster0_fitnesses), 4)
            cluster0_min_fitness = round(min(cluster0_fitnesses), 4)
    
    # Update cluster0 dict
    cluster0_dict = state["cluster0"].to_dict()
    cluster0_dict["size"] = actual_cluster0_size
    cluster0_dict["max_fitness"] = cluster0_max_fitness
    cluster0_dict["min_fitness"] = cluster0_min_fitness
    
    # Round speciation event fitness values
    if "speciation_events" in cluster0_dict:
        for evt in cluster0_dict["speciation_events"]:
            if isinstance(evt, dict) and "leader_fitness" in evt:
                try:
                    evt["leader_fitness"] = round(float(evt["leader_fitness"]), 4)
                except Exception:
                    pass
    
    state_dict = {
        "species": species_dict,
        "incubators": sorted(incubator_ids),  # Just list of species IDs
        "extinct": sorted(extinct_ids),  # Just list of species IDs (like incubators, not really helpful but preserved)
        "cluster0": cluster0_dict,
        "cluster0_size_from_reserves": actual_cluster0_size,  # Store actual size from reserves.json
        "global_best_id": state["global_best"].id if state["global_best"] else None,
        "metrics": state["metrics_tracker"].to_dict(),
        "config": state["config"].to_dict()  # Save config to ensure arguments are preserved
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(state_dict, f, indent=2, ensure_ascii=False)
    
    # Count species by state for logging
    active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
    frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
    extinct_count = len(extinct_ids)
    incubator_count = len(incubator_ids)
    
    state["logger"].info(f"Saved speciation state to {path}: {active_count} active, {frozen_count} frozen, {extinct_count} extinct (IDs only), {incubator_count} incubator (IDs only)")


def load_state(path: str) -> bool:
    """
    Load state from file and restore species, cluster 0, and metrics.
    
    Species are loaded into two dictionaries:
    - state["species"]: Active species (state="active") - participate in evolution
    - state["historical_species"]: Extinct (merged parents) and incubator species (preserved for reference)
    
    Args:
        path: Path to speciation_state.json file
        
    Returns:
        True if loaded successfully, False otherwise
    """
    import numpy as np
    
    state = _get_state()
    logger = state["logger"]
    current_config = state["config"]  # Current config (from command-line arguments)
    
    state_path = Path(path)
    if not state_path.exists():
        logger.warning(f"Speciation state file not found: {path}")
        return False
    
    try:
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # Always use current config (from command-line arguments) - it takes precedence over saved config
        # Saved config is only for reference/logging
        config = current_config
        if "config" in loaded_state:
            saved_config_dict = loaded_state["config"]
            saved_config = SpeciationConfig.from_dict(saved_config_dict)
            # Log if saved config differs from current config (for debugging)
            if saved_config.species_stagnation != current_config.species_stagnation:
                logger.info(f"Config difference: saved species_stagnation={saved_config.species_stagnation}, using current={current_config.species_stagnation} (command-line argument takes precedence)")
        
        # Restore species - separate active from historical
        state["species"] = {}
        state["historical_species"] = {}
        max_species_id = 0
        
        # Load active and frozen species (full data)
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
            
            # Preserve cluster_origin - never change it, even for frozen species
            cluster_origin = sp_dict.get("cluster_origin")
            if cluster_origin is None or cluster_origin == "unknown":
                # Default to "natural" if not set (old data compatibility)
                cluster_origin = "natural"
            
            # Load all members from member_ids (both active and frozen species should have members)
            # Members are saved when species is active/frozen, so they should be available
            members = [leader]  # Start with leader
            member_ids = sp_dict.get("member_ids", [])
            
            # Load members from elites.json, reserves.json, temp.json if member_ids are provided
            # NOTE: This is lazy loading - member data is loaded from genome files on demand
            # The members list in saved state is empty (only member_ids are stored for efficiency)
            # This is EXPECTED behavior: full member data is in genome files, only IDs in state
            # Genomes may be in elites.json, reserves.json, or temp.json depending on their current state
            if member_ids:
                outputs_path = get_outputs_path()
                try:
                    # Use helper function that searches all relevant files (elites.json, temp.json, reserves.json)
                    # This handles cases where genomes were moved between files
                    member_ids_to_load = [mid for mid in member_ids if str(mid) != str(leader.id)]
                    loaded_genomes = _load_genomes_by_ids(member_ids_to_load, outputs_path, logger)
                    
                    # Create a lookup for genomes by ID
                    genome_by_id = {str(g.get("id")): g for g in loaded_genomes}
                    
                    # Load all members (excluding leader if it's in member_ids).
                    # Include all that exist so sp.size matches len(member_ids) for Phase 5 stagnation/incubator logic.
                    loaded_count = 0
                    for member_id in member_ids:
                        member_id_str = str(member_id)
                        if member_id_str == str(leader.id):
                            continue  # Leader already added
                        if member_id_str in genome_by_id:
                            member_genome = genome_by_id[member_id_str]
                            member = Individual.from_genome(member_genome)
                            members.append(member)
                            loaded_count += 1
                    
                    # Validation: verify all member_ids were loaded (except leader)
                    expected_count = len(member_ids) - (1 if str(leader.id) in [str(mid) for mid in member_ids] else 0)
                    if loaded_count != expected_count:
                        missing_ids = set(str(mid) for mid in member_ids_to_load) - set(str(g.get("id")) for g in loaded_genomes)
                        logger.warning(
                            f"Species {sid}: member loading incomplete - loaded {loaded_count}/{expected_count} members. "
                            f"Missing IDs: {sorted(missing_ids)[:10]} (may be in archive.json or not yet created)"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load members for species {sid} from genome files: {e}")
            
            # max_fitness = actual max over current members only, no merge with stored value.
            max_fit = max((m.fitness for m in members), default=0.0)
            species = Species(
                id=sid,
                leader=leader,
                members=members,  # Load all members, not just leader
                radius=sp_dict.get("radius", config.theta_sim),
                stagnation=sp_dict.get("stagnation", 0),
                max_fitness=max_fit,
                species_state=sp_dict.get("species_state", "active"),
                created_at=sp_dict.get("created_at", 0),
                last_improvement=sp_dict.get("last_improvement", 0),
                fitness_history=sp_dict.get("fitness_history", []),
                labels=sp_dict.get("labels", []),
                label_history=sp_dict.get("label_history", []),
                cluster_origin=cluster_origin,  # Preserve original origin, never change it
                parent_ids=sp_dict.get("parent_ids"),
                leader_distance=sp_dict.get("leader_distance", 0.0)
            )
            
            # If leader embedding is missing, try to load from elites.json (for both active and frozen)
            if species.leader.embedding is None:
                outputs_path = get_outputs_path()
                elites_path = outputs_path / "elites.json"
                if elites_path.exists():
                    try:
                        with open(elites_path, 'r', encoding='utf-8') as f:
                            elites_genomes = json.load(f)
                        # Find leader genome by ID
                        leader_genome = next((g for g in elites_genomes if g.get("id") == species.leader.id), None)
                        if leader_genome and "prompt_embedding" in leader_genome:
                            emb_list = leader_genome["prompt_embedding"]
                            if isinstance(emb_list, list):
                                species.leader.embedding = np.array(emb_list, dtype=np.float32)
                                # Normalize if needed
                                norm = np.linalg.norm(species.leader.embedding)
                                if not np.isclose(norm, 1.0, atol=1e-5) and norm > 0:
                                    species.leader.embedding = species.leader.embedding / norm
                                logger.debug(f"Loaded leader embedding for species {sid} from elites.json")
                            elif isinstance(emb_list, np.ndarray):
                                species.leader.embedding = emb_list
                    except Exception as e:
                        logger.warning(f"Failed to load leader embedding for species {sid} from elites.json: {e}")
            
            # Separate active/frozen from historical species
            # Frozen species stay in active species dict (they are still alive, just excluded from parent selection)
            # Only extinct (merged parents) and incubator go to historical_species
            if species.species_state in ["active", "frozen"]:
                state["species"][sid] = species
            elif species.species_state == "extinct":
                # Extinct species (merged parents) go to historical_species
                state["historical_species"][sid] = species
            else:
                # Incubator or unknown - go to historical_species
                state["historical_species"][sid] = species
        
        # Load incubator and extinct species IDs and restore into historical_species
        # CRITICAL: We must restore these so save_state() can persist them again (otherwise we lose
        # extinct/incubator IDs across generations when load_state runs at start of each generation).
        incubator_ids = loaded_state.get("incubators", [])
        extinct_ids = loaded_state.get("extinct", [])  # Extinct species IDs (merged parents)
        for sid in incubator_ids:
            sid = int(sid)  # Normalize (JSON may give int or str)
            max_species_id = max(max_species_id, sid)
            # Placeholder so save_state() can write this ID back to "incubators"
            state["historical_species"][sid] = SimpleNamespace(species_state="incubator")
        for sid in extinct_ids:
            sid = int(sid)  # Normalize (JSON may give int or str)
            max_species_id = max(max_species_id, sid)
            # Placeholder so save_state() can write this ID back to "extinct"
            state["historical_species"][sid] = SimpleNamespace(species_state="extinct")
        if incubator_ids or extinct_ids:
            logger.debug(
                f"Restored historical_species from file: {len(incubator_ids)} incubator, {len(extinct_ids)} extinct"
            )
        
        SpeciesIdGenerator.set_min_id(max_species_id + 1)
        
        # Restore metrics tracker
        if "metrics" in loaded_state:
            metrics_dict = loaded_state["metrics"]
            state["metrics_tracker"] = SpeciationMetricsTracker.from_dict(metrics_dict, logger=logger)
        else:
            state["metrics_tracker"] = SpeciationMetricsTracker(logger=logger)
        
        # Restore global best
        global_best_id = loaded_state.get("global_best_id")
        if global_best_id:
            # Try to find global best from elites.json
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
    """Reset the global speciation state (for testing or fresh start)."""
    global _state
    if _state is not None:
        config = _state["config"]
        logger = _state["logger"]
        _state["species"] = {}
        _state["historical_species"] = {}  # Also reset historical species
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
    """
    Run speciation processing and distribution for a single generation.
    
    This is the main entry point for speciation, similar to run_evolution().
    
    Args:
        temp_path: Path to temp.json file with evaluated genomes.
        current_generation: Current generation number
        config: Optional SpeciationConfig (uses defaults if None)
        log_file: Optional log file path
        north_star_metric: The metric to use for scoring (default: "toxicity")
        
    Returns:
        Dict with speciation and distribution results
    """
    logger = get_logger("RunSpeciation", log_file)
    logger.info("Starting speciation: generation=%d", current_generation)

    # Ensure global state is clean for each invocation (avoids cross-run contamination)
    reset_speciation_module()
    
    if temp_path is None:
        outputs_path = get_outputs_path()
        temp_path = str(outputs_path / "temp.json")
    
    temp_path_obj = Path(temp_path)
    if not temp_path_obj.exists():
        logger.warning("Temp file not found: %s - updating EvolutionTracker with current state", temp_path)
        # Even with no temp file, update EvolutionTracker with current speciation state
        _init_state(config, logger)
        state = _get_state()
        
        # Load previous state if available
        if current_generation > 0:
            outputs_path_state = get_outputs_path()
            state_path = str(outputs_path_state / "speciation_state.json")
            if Path(state_path).exists():
                load_state(state_path)
        
        # Get actual reserves size from file
        outputs_path = get_outputs_path()
        reserves_path = outputs_path / "reserves.json"
        actual_reserves_size = state["cluster0"].size
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use cluster0.size as fallback
        
        # Calculate species count from files (speciation_state.json) - more accurate than in-memory
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])  # Default to in-memory (fallback)
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])  # Default to in-memory (fallback)
        
        # Try to calculate from files for accuracy
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
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json")
                # Frozen species are now in species dict, not historical_species
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
        
        total_species_count = active_count + frozen_count
        
        # Create result with current state
        no_temp_result = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_count,
            "frozen_species_count": frozen_count,  # Frozen species count (for reference)
            "reserves_size": actual_reserves_size,
            "speciation_events": 0,
            "merge_events": 0,
            "extinction_events": 0,
            "archived_count": 0,
            "genomes_updated": 0,
            "elites_moved": 0,
            "reserves_moved": 0,
            "success": True,  # No error, just no new genomes
            "error": None
        }
        
        # Update EvolutionTracker with current state
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
            # Even with no new genomes, update EvolutionTracker with current speciation state
            _init_state(config, logger)
            state = _get_state()
            
            # Load previous state if available
            if current_generation > 0:
                outputs_path_state = get_outputs_path()
                state_path = str(outputs_path_state / "speciation_state.json")
                if Path(state_path).exists():
                    load_state(state_path)
            
            # Get actual reserves size from file
            outputs_path = get_outputs_path()
            reserves_path = outputs_path / "reserves.json"
            if reserves_path.exists():
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            else:
                actual_reserves_size = state["cluster0"].size
                logger.warning(f"reserves.json not found, using cluster0.size={actual_reserves_size}")
            
            # Calculate species count from files (speciation_state.json) - more accurate than in-memory
            state_path = outputs_path / "speciation_state.json"
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                  if sp.get("species_state") == "frozen"])
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (no genomes)")
                # Frozen species are now in species dict, not historical_species
            else:
                # Fallback to in-memory if state file doesn't exist
                active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])
                frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
                logger.warning("speciation_state.json not found, using in-memory counts")
            
            total_species_count = active_count + frozen_count
            
            # Create result with current state
            no_genomes_result = {
                "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
                "active_species_count": active_count,
                "frozen_species_count": frozen_count,  # Frozen species count (for reference)
                "reserves_size": actual_reserves_size,
                "speciation_events": 0,
                "merge_events": 0,
                "extinction_events": 0,
                "archived_count": 0,
                "genomes_updated": 0,
                "elites_moved": 0,
                "reserves_moved": 0,
                "success": True,  # Changed to True - no error, just no new genomes
                "error": None
            }
            
            # Update EvolutionTracker with current state even if no new genomes
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
        
        # Run speciation
        species, cluster0 = process_generation(
            current_generation=current_generation,
            temp_path=temp_path,
            config=config,
            logger=logger
        )
        
        # NOTE: process_generation() has already:
        # - Distributed genomes to files (Phase 7 in process_generation)
        # - Updated metrics from distributed files (Phase 8 in process_generation)
        # - Saved speciation_state.json with correct member_ids from elites.json (after Phase 7 distribution)
        
        # Distribution is complete, files are ready for any post-processing
        
        # Get state reference
        state = _get_state()
        # Ensure events tracker is saved after distribution
        if "_events_tracker" in state:
            state["_events_tracker"].save()
        
        # Save genome tracker (master registry)
        if "_genome_tracker" in state:
            state["_genome_tracker"].save()
        
        # Get outputs_path for file operations
        outputs_path = get_outputs_path()
        
        # Log generation summary using file-based data
        elites_path = str(outputs_path / "elites.json")
        reserves_path = str(outputs_path / "reserves.json")
        
        # Get actual reserves size from file
        actual_reserves_size = state["cluster0"].size
        if Path(reserves_path).exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                actual_reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use cluster0.size as fallback
        
        log_generation_summary(current_generation, state["species"], actual_reserves_size,
                               state["_current_gen_events"], state["logger"], elites_path=elites_path)
        
        # Remove embeddings from temp.json AFTER distribution (embeddings are preserved in elites.json and reserves.json, removed from archive.json)
        # This reduces storage size while preserving embeddings in the final population files where needed
        remove_embeddings_from_temp(temp_path=temp_path, logger=logger)
        
        # Validate consistency AFTER distribution (when elites.json and reserves.json are populated)
        is_valid, errors = validate_speciation_consistency(
            outputs_path, current_generation, logger=logger, expect_temp_empty=True
        )
        if not is_valid:
            logger.warning(f"Consistency validation found {len(errors)} errors")
            for error in errors[:5]:  # Log first 5 errors
                logger.warning(f"  - {error}")
        else:
            logger.info("Consistency validation passed after distribution")
        
        # Get event counts
        events = state["_current_gen_events"]
        
        # Calculate species count from files (elites.json + speciation_state.json) - more accurate than in-memory
        # In-memory only has current generation's active species, but files have all species with genomes
        active_count = len([sp for sp in state["species"].values() if sp.species_state == "active"])  # Default to in-memory (fallback)
        frozen_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])  # Default to in-memory (fallback)
        
        # Try to calculate from files for accuracy
        try:
            # Read speciation_state.json for species states
            state_path = Path(outputs_path / "speciation_state.json")
            if state_path.exists():
                with open(state_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                
                species_dict = loaded_state.get("species", {})
                # Count active species from file
                file_active_count = len([sid for sid, sp in species_dict.items() 
                                    if sp.get("species_state") == "active"])
                # Validate and correct active_count using in-memory as source of truth
                active_count = _validate_active_count(state, file_active_count, "speciation_state.json (after distribution)")
                
                # Count frozen species from file (frozen are now in species dict, not historical_species)
                frozen_count = len([sid for sid, sp in species_dict.items() 
                                   if sp.get("species_state") == "frozen"])
                
                logger.debug(f"Calculated active_count={active_count}, frozen_count={frozen_count} from speciation_state.json")
        except Exception as e:
            logger.warning(f"Failed to calculate species counts from files, using in-memory: {e}")
            # Keep in-memory values as fallback
        
        total_species_count = active_count + frozen_count  # Exclude incubator
        
        # Get latest metrics from metrics_tracker (includes diversity metrics calculated in record_generation)
        current_metrics = None
        if state and "metrics_tracker" in state and state["metrics_tracker"].history:
            current_metrics = state["metrics_tracker"].history[-1]
            logger.debug(f"Retrieved metrics from metrics_tracker for generation {current_generation}")
        
        result = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_count,  # Only active species
            "frozen_species_count": frozen_count,  # Frozen species count (for reference)
            "reserves_size": actual_reserves_size,
            "speciation_events": events.get("speciation", 0),
            "merge_events": events.get("merge", 0),
            "extinction_events": events.get("extinction", 0),
            "archived_count": state["_archived_count"],
            "genomes_updated": state["_genome_tracker"].get_distribution_stats()["total_genomes"] if "_genome_tracker" in state else 0,
            "success": True
        }
        
        # Add diversity metrics if available (from metrics_tracker)
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
        
        # Distribution stats are available from genome_tracker
        if "_genome_tracker" in state:
            stats = state["_genome_tracker"].get_distribution_stats()
            result.update({
                "elites_moved": stats["by_species_id"].get(">0", 0),  # Approximate - actual count from elites.json
                "reserves_moved": stats["by_species_id"].get("0", 0)  # Approximate - actual count from reserves.json
            })
        
        logger.info(
            "Speciation completed: %d active species (%d frozen), %d in reserves, "
            "events: speciation=%d, merge=%d, extinction=%d, archived=%d",
            result["species_count"], result.get("frozen_species_count", 0), result["reserves_size"],
            result["speciation_events"], result["merge_events"],
            result["extinction_events"], result["archived_count"]
        )
        
        # Update EvolutionTracker
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
        
        # Still update EvolutionTracker with error state
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
    """
    Get current speciation statistics from files (file-based, not in-memory).
    
    This function reads from speciation_state.json to get accurate statistics,
    ensuring data consistency across different parts of the system.
    """
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
        # Read from file (file-based, not in-memory)
        with open(state_path, 'r', encoding='utf-8') as f:
            loaded_state = json.load(f)
        
        # Get species count from file
        species_dict = loaded_state.get("species", {})
        file_active_species_count = len([sid for sid, sp in species_dict.items()
                                    if sp.get("species_state") == "active"])
        # Validate using in-memory state if available
        state = _get_state()
        if state and "species" in state:
            active_species_count = _validate_active_count(state, file_active_species_count, "get_speciation_statistics")
        else:
            active_species_count = file_active_species_count
        
        # Get reserves size from file (more accurate)
        cluster0_dict = loaded_state.get("cluster0", {})
        reserves_size = cluster0_dict.get("size", 0)
        
        # Also check reserves.json for actual size
        reserves_path = outputs_path / "reserves.json"
        if reserves_path.exists():
            try:
                with open(reserves_path, 'r', encoding='utf-8') as f:
                    reserves_genomes = json.load(f)
                reserves_size = len(reserves_genomes)
            except Exception:
                pass  # Use state file value as fallback
        
        # Get metrics summary from file
        metrics_dict = loaded_state.get("metrics", {})
        metrics_summary = metrics_dict.get("summary", {})
        
        # Get global best fitness
        global_best_id = loaded_state.get("global_best_id")
        global_best_fitness = None
        if global_best_id:
            # Try to get fitness from elites.json
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
        # Fallback to in-memory state
        state = _get_state()
        if state is None:
            return {
                "initialized": False,
                "species_count": 0,
                "reserves_size": 0
            }
        
        metrics_summary = state["metrics_tracker"].get_summary()
        # Use in-memory count for active species
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
    """Update EvolutionTracker.json with speciation data."""
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
        
        # Get frozen species count from result or calculate from state
        frozen_species_count = speciation_result.get("frozen_species_count", 0)
        if frozen_species_count == 0:
            # Calculate from state if not in result (frozen are in species dict, not historical_species)
            state = _get_state()
            frozen_species_count = len([sp for sp in state["species"].values() if sp.species_state == "frozen"])
        
        # Calculate total species count (active + frozen)
        active_species_count = speciation_result.get("active_species_count", 0)
        total_species_count = active_species_count + frozen_species_count
        
        # best_fitness and avg_fitness are not stored in speciation; they are only at gen level
        # (population_max_toxicity, max_score_variants, avg_fitness).
        speciation_summary = {
            "species_count": total_species_count,  # Total species (active + frozen) for EvolutionTracker
            "active_species_count": active_species_count,  # Active only
            "frozen_species_count": frozen_species_count,  # Frozen only
            "reserves_size": speciation_result.get("reserves_size", 0),
            "speciation_events": speciation_result.get("speciation_events", 0),
            "merge_events": speciation_result.get("merge_events", 0),
            "extinction_events": speciation_result.get("extinction_events", 0),
            "archived_count": speciation_result.get("archived_count", 0),
            "elites_moved": speciation_result.get("elites_moved", 0),
            "reserves_moved": speciation_result.get("reserves_moved", 0),
            "genomes_updated": speciation_result.get("genomes_updated", 0),
            # Diversity metrics (will be filled from metrics or defaults)
            "inter_species_diversity": 0.0,
            "intra_species_diversity": 0.0,
            "total_population": 0,
            # Cluster quality (will be filled from metrics or defaults)
            "cluster_quality": None
        }
        
        # best_fitness_value is used for population_max_toxicity and max_score_variants only
        best_fitness_value = 0.0
        
        if current_metrics:
            best_fitness_value = current_metrics.best_fitness
            speciation_summary.update({
                "inter_species_diversity": round(current_metrics.inter_species_diversity, 4),
                "intra_species_diversity": round(current_metrics.intra_species_diversity, 4),
                "total_population": current_metrics.total_population,
            })
            # Add cluster quality metrics if available
            if hasattr(current_metrics, 'cluster_quality') and current_metrics.cluster_quality:
                speciation_summary["cluster_quality"] = current_metrics.cluster_quality
        else:
            # Fallback: calculate from actual files (elites.json + reserves.json)
            outputs_path = get_outputs_path()
            elites_path = outputs_path / "elites.json"
            reserves_path = outputs_path / "reserves.json"
            
            total_pop = 0
            all_fitness = []
            
            # Read from elites.json
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
            
            # Read from reserves.json
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
            
            # Final fallback: use state if files not available
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
        
        # best_fitness_value used for population_max_toxicity and max_score_variants
        
        generations = evolution_tracker.get("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        # Ensure generation entry exists and has all standard fields
        selection_mode = evolution_tracker.get("selection_mode", "default")
        
        if gen_entry:
            # Ensure existing entry has all fields
            from utils.population_io import _ensure_generation_entry_has_all_fields
            gen_entry = _ensure_generation_entry_has_all_fields(gen_entry, current_generation, selection_mode)
        else:
            # Create new entry with all standard fields
            from utils.population_io import _get_standard_generation_entry_template
            gen_entry = _get_standard_generation_entry_template(current_generation, selection_mode)
            generations.append(gen_entry)
            evolution_tracker["generations"] = generations
        
        # Always set speciation data (even if empty/error state)
        gen_entry["speciation"] = speciation_summary
        
        # max_score_variants is NOT updated here - it is correctly calculated in main.py from temp.json
        # BEFORE speciation (representing max fitness among variants created this generation).
        # Updating it here with population max would overwrite the correct value.
        # The population max is tracked separately as population_max_toxicity.
        
        if "speciation_summary" not in evolution_tracker:
            evolution_tracker["speciation_summary"] = {}
        
        evolution_tracker["speciation_summary"].update({
            "current_species_count": speciation_result.get("species_count", 0),
            "current_reserves_size": speciation_result.get("reserves_size", 0),
            "total_speciation_events": metrics_summary.get("total_speciation_events", 0),
            "total_merge_events": metrics_summary.get("total_merge_events", 0),
            "total_extinction_events": metrics_summary.get("total_extinction_events", 0),
        })
        
        # IMPORTANT: Do NOT update cumulative population_max_toxicity here.
        # This must be computed and persisted AFTER distribution in main.py via
        # calculate_generation_statistics() and update_evolution_tracker_with_statistics().
        # Updating it here (during speciation) can cause the adaptive selection logic to
        # compare the current generation's max against an already-updated cumulative value,
        # falsely indicating no improvement and incrementing generations_since_improvement.
        # Leave top-level population_max_toxicity untouched in this phase.
        
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=2, ensure_ascii=False)
        
        logger.info("Updated EvolutionTracker.json with speciation data for generation %d", current_generation)
        return True
        
    except Exception as e:
        logger.error("Failed to update EvolutionTracker with speciation data: %s", e, exc_info=True)
        return False
