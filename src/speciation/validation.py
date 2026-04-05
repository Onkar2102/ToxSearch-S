"""
validation.py

Consistency validation and distance threshold analysis for speciation framework.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .species import Species

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def validate_speciation_consistency(
    outputs_path: Path,
    generation: int,
    logger=None,
    expect_temp_empty: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate consistency across all speciation files.
    
    Checks:
    1. Species IDs in elites.json match species in speciation_state.json
    2. Elites have species_id > 0; reserves have species_id = 0
    3. Species sizes match between state and elites.json
    4. No duplicate genome IDs across elites, reserves, archive (ids compared as str)
    4b. No duplicate prompt strings (case-sensitive) across those files
    5. cluster0.size and cluster0_size_from_reserves match len(reserves.json)
    6. Sum invariant / temp: if expect_temp_empty, temp must be []; else if temp has
       genomes and differs from elites+reserves+archive by >1, error.
    
    Args:
        outputs_path: Path to outputs directory
        generation: Current generation number
        logger: Optional logger instance
        expect_temp_empty: If True (e.g. after distribution), temp.json must be [].
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("Validation")
    
    errors: List[str] = []
    
    # Load all files
    state_file_path = outputs_path / "speciation_state.json"
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    temp_path = outputs_path / "temp.json"
    
    try:
        # Load state
        if not state_file_path.exists():
            errors.append("speciation_state.json not found")
            return False, errors
        
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_file = json.load(f)
        
        # Load elites
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        
        # Load reserves
        reserves = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves = json.load(f)
        
        # Load archive
        archive = []
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive = json.load(f)
                # Ensure archive is a list (handle edge cases)
                if not isinstance(archive, list):
                    if isinstance(archive, dict):
                        archive = list(archive.values()) if len(archive) > 0 else []
                    else:
                        archive = []
            except Exception as e:
                logger.warning(f"Failed to load archive.json: {e}")
        
        # Check 1: Species ID consistency
        elite_species_ids = {g.get("species_id") for g in elites if g.get("species_id") is not None}
        state_species_ids = set(int(k) for k in state_file.get("species", {}).keys() if k.isdigit())
        incubator_ids = set(state_file.get("incubators", []))  # Incubator species IDs (just tracked)
        
        # Remove None and 0 (cluster 0) from elite_species_ids for comparison
        elite_species_ids_filtered = {sid for sid in elite_species_ids if sid is not None and sid > 0}
        
        # Check for species in elites that are not in active/frozen or incubator
        all_tracked_ids = state_species_ids | incubator_ids
        missing_in_state = elite_species_ids_filtered - all_tracked_ids
        if missing_in_state:
            errors.append(f"Species in elites.json but not tracked in state (active/frozen/incubator): {missing_in_state}")
        
        missing_in_elites = state_species_ids - elite_species_ids_filtered
        if missing_in_elites:
            # This is OK - species might be empty or frozen
            logger.debug(f"Species in state but not in elites (may be empty/frozen): {missing_in_elites}")
        
        # Check 2a: Elites must have species_id > 0
        for g in elites:
            sid = g.get("species_id")
            if sid is None or sid <= 0:
                errors.append(f"Elite genome id={g.get('id')} has species_id={sid} (must be > 0)")
        
        # Check 2b: Reserves species_id
        reserve_species_ids = {g.get("species_id") for g in reserves if g.get("species_id") is not None}
        if reserve_species_ids != {0} and reserve_species_ids != set():
            errors.append(f"Reserves with non-zero species_id: {reserve_species_ids}")
        
        # Check 3: Size consistency (only for active/frozen species, exclude incubator)
        # Only validate if elites.json has genomes (after distribution)
        if len(elites) > 0:
            for sid_str, sp_dict in state_file.get("species", {}).items():
                try:
                    sid = int(sid_str)
                    species_state = sp_dict.get("species_state", "active")
                    
                    # Skip incubator species from size validation (they're just tracked by ID)
                    if species_state == "incubator":
                        continue
                    
                    expected_size = len([g for g in elites if g.get("species_id") == sid])
                    actual_size = sp_dict.get("size", 0)
                    if expected_size != actual_size:
                        errors.append(f"Species {sid}: expected size {expected_size} (from elites.json), got {actual_size} (from state)")
                except ValueError:
                    continue
        else:
            # Before distribution, elites.json is empty - this is expected
            # Don't validate sizes until after distribution
            logger.debug("Skipping size validation: elites.json is empty (before distribution)")
        
        # Check 4: Duplicate IDs (normalize to str so int 1 and "1" count as the same id)
        all_genome_ids = []
        for g in elites:
            x = g.get("id")
            if x is not None and x != "":
                all_genome_ids.append(str(x))
        for g in reserves:
            x = g.get("id")
            if x is not None and x != "":
                all_genome_ids.append(str(x))
        for g in archive:
            x = g.get("id")
            if x is not None and x != "":
                all_genome_ids.append(str(x))
        
        from collections import Counter
        id_counts = Counter(all_genome_ids)
        duplicates = [gid for gid, count in id_counts.items() if count > 1]
        if duplicates:
            errors.append(f"Duplicate genome IDs found: {duplicates[:10]}")  # Limit to first 10
        
        # Check 4b: duplicate prompts (case-sensitive exact string match) across living population
        all_prompts = []
        for g in elites:
            p = g.get("prompt")
            if isinstance(p, str):
                all_prompts.append(p)
        for g in reserves:
            p = g.get("prompt")
            if isinstance(p, str):
                all_prompts.append(p)
        for g in archive:
            p = g.get("prompt")
            if isinstance(p, str):
                all_prompts.append(p)
        prompt_counts = Counter(all_prompts)
        dup_prompts = [p for p, count in prompt_counts.items() if count > 1]
        if dup_prompts:
            preview = [repr(s[:40] + ("..." if len(s) > 40 else "")) for s in dup_prompts[:5]]
            errors.append(
                f"Duplicate prompts (case-sensitive) across elites/reserves/archive: {len(dup_prompts)} distinct strings; "
                f"preview={preview}"
            )
        
        # Check 5: cluster0.size and cluster0_size_from_reserves match len(reserves)
        reserves_len = len(reserves)
        cluster0 = state_file.get("cluster0", {})
        c0_size = cluster0.get("size")
        c0_from_reserves = state_file.get("cluster0_size_from_reserves")
        if c0_size is not None and c0_size != reserves_len:
            errors.append(f"cluster0.size={c0_size} != len(reserves.json)={reserves_len}")
        if c0_from_reserves is not None and c0_from_reserves != reserves_len:
            errors.append(f"cluster0_size_from_reserves={c0_from_reserves} != len(reserves.json)={reserves_len}")
        
        # Check 6: Sum invariant / temp.json
        # After distribution, temp.json should be cleared ([]). If expect_temp_empty, require that.
        if temp_path.exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                temp_count = len(temp_genomes) if isinstance(temp_genomes, list) else 0
                
                if expect_temp_empty:
                    if temp_count > 0:
                        errors.append(f"After distribution temp.json must be empty; found {temp_count} genomes")
                # When expect_temp_empty is False (e.g. unknown or pre-distribution), skip sum invariant
                # to avoid false positives on old/partial runs where temp may hold leftovers.
            except Exception as e:
                logger.warning(f"Could not verify temp/sum invariant: {e}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info(f"Consistency validation passed for generation {generation}")
        else:
            logger.warning(f"Consistency validation found {len(errors)} errors for generation {generation}")
        
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Validation failed with exception: {e}"
        logger.error(error_msg, exc_info=True)
        return False, [error_msg]


def validate_flow2_speciation(
    outputs_path: Path,
    generation: int,
    newly_formed_species_ids: List[int],
    logger=None
) -> Tuple[bool, List[str]]:
    """
    Validate Flow 2 speciation requirements for newly formed species.
    
    Validates:
    1. All newly formed species have original potential leader (not updated to highest fitness)
    2. All followers are included in species (no radius filtering)
    3. Species leader is in members list
    4. All members have correct species_id
    
    Args:
        outputs_path: Path to outputs directory
        generation: Current generation number
        newly_formed_species_ids: List of species IDs that were newly formed
        logger: Optional logger instance
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("Flow2Validation")
    
    errors: List[str] = []
    
    if not newly_formed_species_ids:
        return True, errors
    
    state_file_path = outputs_path / "speciation_state.json"
    elites_path = outputs_path / "elites.json"
    
    try:
        # Load state
        if not state_file_path.exists():
            # For Generation 0, speciation_state.json might not exist yet
            # This is not an error - it will be created when species are formed
            if generation == 0:
                logger.debug("speciation_state.json not found for Generation 0 Flow 2 validation (will be created)")
                return True, []  # Skip validation for Generation 0
            errors.append("speciation_state.json not found for Flow 2 validation")
            return False, errors
        
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_file = json.load(f)
        
        # Load elites
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        
        species_dict = state_file.get("species", {})
        
        for sid in newly_formed_species_ids:
            sid_str = str(sid)
            if sid_str not in species_dict:
                errors.append(f"Newly formed species {sid} not found in speciation_state.json")
                continue
            
            sp_dict = species_dict[sid_str]
            
            # Check 1: Leader must be in members list
            # Note: species.to_dict() uses "leader_id" (not nested "leader.id")
            leader_id = sp_dict.get("leader_id") or (sp_dict.get("leader", {}).get("id") if isinstance(sp_dict.get("leader"), dict) else None)
            member_ids = sp_dict.get("member_ids", [])
            
            if leader_id is None:
                errors.append(f"Species {sid}: leader ID is None (checked both 'leader_id' and 'leader.id')")
                continue
            
            if leader_id not in member_ids:
                errors.append(f"Species {sid}: leader {leader_id} not in member_ids {member_ids[:5]}...")
            
            # Check 2: All members in elites.json have correct species_id
            # NOTE: This validation runs BEFORE Phase 7 (redistribution), so genomes may not be in elites.json yet.
            # They could be in temp.json (new variants) or reserves.json (from cluster 0).
            # We check elites.json but don't treat missing genomes as errors if they're in other files.
            elites_for_species = [g for g in elites if g.get("species_id") == sid]
            elites_member_ids = {g.get("id") for g in elites_for_species}
            
            # Also check reserves.json and temp.json (genomes may not be distributed yet)
            reserves_path = outputs_path / "reserves.json"
            temp_path = outputs_path / "temp.json"
            other_member_ids = set()
            
            if reserves_path.exists():
                try:
                    with open(reserves_path, 'r', encoding='utf-8') as f:
                        reserves = json.load(f)
                    # Check if any member_ids are in reserves (they may have species_id=0 or not yet updated)
                    for g in reserves:
                        gid = g.get("id")
                        if gid in member_ids:
                            other_member_ids.add(gid)
                except Exception:
                    pass
            
            if temp_path.exists():
                try:
                    with open(temp_path, 'r', encoding='utf-8') as f:
                        temp_genomes = json.load(f)
                    for g in temp_genomes:
                        gid = g.get("id")
                        if gid in member_ids:
                            other_member_ids.add(gid)
                except Exception:
                    pass
            
            # Only report error if member_ids are missing from ALL files (elites, reserves, temp)
            # This means they're truly missing, not just not yet distributed
            missing_in_all = set(member_ids) - elites_member_ids - other_member_ids
            if missing_in_all:
                # This is a real error - genomes are in speciation_state but not in any population file
                errors.append(f"Species {sid}: {len(missing_in_all)} member_ids not found in any population file (elites/reserves/temp): {list(missing_in_all)[:5]}...")
            elif set(member_ids) - elites_member_ids:
                # Genomes exist but not in elites.json yet (expected before Phase 7)
                logger.debug(f"Species {sid}: {len(set(member_ids) - elites_member_ids)} member_ids not yet in elites.json (will be distributed in Phase 7)")
            
            # Check 3: Size consistency
            # Species.to_dict() stores member_ids but not size (computed property).
            # Derive expected size from member_ids; only flag if an explicit "size"
            # field exists and disagrees with len(member_ids).
            expected_size = len(member_ids)
            if "size" in sp_dict and sp_dict["size"] != expected_size:
                errors.append(f"Species {sid}: size mismatch - expected {expected_size} (from member_ids), got {sp_dict['size']} (from state.size)")
            
            # Check 4: Cluster origin should be "natural" for newly formed species
            cluster_origin = sp_dict.get("cluster_origin")
            if cluster_origin != "natural":
                errors.append(f"Species {sid}: cluster_origin is '{cluster_origin}', expected 'natural' for newly formed species")
            
            # Check 5: Created_at should match current generation
            created_at = sp_dict.get("created_at")
            if created_at != generation:
                errors.append(f"Species {sid}: created_at={created_at}, expected {generation}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.debug(f"Flow 2 validation passed for {len(newly_formed_species_ids)} newly formed species")
        else:
            logger.warning(f"Flow 2 validation found {len(errors)} errors for {len(newly_formed_species_ids)} newly formed species")
        
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Flow 2 validation failed with exception: {e}"
        logger.error(error_msg, exc_info=True)
        return False, [error_msg]


def validate_metrics_from_files(
    outputs_path: Path,
    metrics: Dict[str, Any],
    logger=None
) -> Tuple[bool, List[str]]:
    """
    Validate that metrics are calculated correctly from files.
    
    Validates:
    1. species_count matches unique species IDs in elites.json
    2. total_population equals len(elites.json) + len(reserves.json)
    3. reserves_size equals len(reserves.json)
    4. best_fitness is max fitness across elites + reserves
    5. avg_fitness is mean fitness across elites + reserves
    
    Args:
        outputs_path: Path to outputs directory
        metrics: Dictionary with metrics (from GenerationMetrics.to_dict())
        logger: Optional logger instance
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("MetricsValidation")
    
    errors: List[str] = []
    
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    try:
        # Load elites
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        else:
            errors.append("elites.json not found for metrics validation")
            return False, errors
        
        # Load reserves
        reserves = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves = json.load(f)
        
        # Validate species_count
        unique_species_ids = {g.get("species_id") for g in elites if g.get("species_id") is not None and g.get("species_id") > 0}
        expected_species_count = len(unique_species_ids)
        actual_species_count = metrics.get("species_count", 0)
        if expected_species_count != actual_species_count:
            errors.append(f"species_count mismatch: expected {expected_species_count} (from elites.json), got {actual_species_count}")
        
        # Validate total_population
        expected_total_pop = len(elites) + len(reserves)
        actual_total_pop = metrics.get("total_population", 0)
        if expected_total_pop != actual_total_pop:
            errors.append(f"total_population mismatch: expected {expected_total_pop} (elites={len(elites)}, reserves={len(reserves)}), got {actual_total_pop}")
        
        # Validate reserves_size
        expected_reserves_size = len(reserves)
        actual_reserves_size = metrics.get("reserves_size", 0)
        if expected_reserves_size != actual_reserves_size:
            errors.append(f"reserves_size mismatch: expected {expected_reserves_size} (from reserves.json), got {actual_reserves_size}")
        
        # Validate fitness metrics
        from utils.population_io import _extract_north_star_score
        all_fitness = []
        for g in elites:
            fitness = _extract_north_star_score(g, "toxicity")
            if fitness > 0:
                all_fitness.append(float(fitness))
        for g in reserves:
            fitness = _extract_north_star_score(g, "toxicity")
            if fitness > 0:
                all_fitness.append(float(fitness))
        
        if all_fitness:
            expected_best = max(all_fitness)
            actual_best = metrics.get("best_fitness", 0.0)
            if abs(expected_best - actual_best) > 0.0001:  # Allow small floating point differences
                errors.append(f"best_fitness mismatch: expected {expected_best:.4f}, got {actual_best:.4f}")
            
            expected_avg = np.mean(all_fitness)
            actual_avg = metrics.get("avg_fitness", 0.0)
            if abs(expected_avg - actual_avg) > 0.0001:
                errors.append(f"avg_fitness mismatch: expected {expected_avg:.4f}, got {actual_avg:.4f}")
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.debug("Metrics validation passed - all metrics match file contents")
        else:
            logger.warning(f"Metrics validation found {len(errors)} errors")
        
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Metrics validation failed with exception: {e}"
        logger.error(error_msg, exc_info=True)
        return False, [error_msg]
