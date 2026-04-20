

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
    
    if logger is None:
        logger = get_logger("Validation")
    
    errors: List[str] = []
    
    state_file_path = outputs_path / "speciation_state.json"
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    archive_path = outputs_path / "archive.json"
    temp_path = outputs_path / "temp.json"
    
    try:
        if not state_file_path.exists():
            errors.append("speciation_state.json not found")
            return False, errors
        
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_file = json.load(f)
        
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        
        reserves = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves = json.load(f)
        
        archive = []
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive = json.load(f)
                if not isinstance(archive, list):
                    if isinstance(archive, dict):
                        archive = list(archive.values()) if len(archive) > 0 else []
                    else:
                        archive = []
            except Exception as e:
                logger.warning(f"Failed to load archive.json: {e}")
        
        elite_species_ids = {g.get("species_id") for g in elites if g.get("species_id") is not None}
        state_species_ids = set(int(k) for k in state_file.get("species", {}).keys() if k.isdigit())
        incubator_ids = set(state_file.get("incubators", []))
        
        elite_species_ids_filtered = {sid for sid in elite_species_ids if sid is not None and sid > 0}
        
        all_tracked_ids = state_species_ids | incubator_ids
        missing_in_state = elite_species_ids_filtered - all_tracked_ids
        if missing_in_state:
            errors.append(f"Species in elites.json but not tracked in state (active/frozen/incubator): {missing_in_state}")
        
        missing_in_elites = state_species_ids - elite_species_ids_filtered
        if missing_in_elites:
            logger.debug(f"Species in state but not in elites (may be empty/frozen): {missing_in_elites}")
        
        for g in elites:
            sid = g.get("species_id")
            if sid is None or sid <= 0:
                errors.append(f"Elite genome id={g.get('id')} has species_id={sid} (must be > 0)")
        
        reserve_species_ids = {g.get("species_id") for g in reserves if g.get("species_id") is not None}
        if reserve_species_ids != {0} and reserve_species_ids != set():
            errors.append(f"Reserves with non-zero species_id: {reserve_species_ids}")
        
        if len(elites) > 0:
            for sid_str, sp_dict in state_file.get("species", {}).items():
                try:
                    sid = int(sid_str)
                    species_state = sp_dict.get("species_state", "active")
                    
                    if species_state == "incubator":
                        continue
                    
                    expected_size = len([g for g in elites if g.get("species_id") == sid])
                    actual_size = sp_dict.get("size", 0)
                    if expected_size != actual_size:
                        errors.append(f"Species {sid}: expected size {expected_size} (from elites.json), got {actual_size} (from state)")
                except ValueError:
                    continue
        else:
            logger.debug("Skipping size validation: elites.json is empty (before distribution)")
        
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
            errors.append(f"Duplicate genome IDs found: {duplicates[:10]}")
        
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
        
        reserves_len = len(reserves)
        cluster0 = state_file.get("cluster0", {})
        c0_size = cluster0.get("size")
        c0_from_reserves = state_file.get("cluster0_size_from_reserves")
        if c0_size is not None and c0_size != reserves_len:
            errors.append(f"cluster0.size={c0_size} != len(reserves.json)={reserves_len}")
        if c0_from_reserves is not None and c0_from_reserves != reserves_len:
            errors.append(f"cluster0_size_from_reserves={c0_from_reserves} != len(reserves.json)={reserves_len}")
        
        if temp_path.exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                temp_count = len(temp_genomes) if isinstance(temp_genomes, list) else 0
                
                if expect_temp_empty:
                    if temp_count > 0:
                        errors.append(f"After distribution temp.json must be empty; found {temp_count} genomes")
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
    
    if logger is None:
        logger = get_logger("Flow2Validation")
    
    errors: List[str] = []
    
    if not newly_formed_species_ids:
        return True, errors
    
    state_file_path = outputs_path / "speciation_state.json"
    elites_path = outputs_path / "elites.json"
    
    try:
        if not state_file_path.exists():
            if generation == 0:
                logger.debug("speciation_state.json not found for Generation 0 Flow 2 validation (will be created)")
                return True, []
            errors.append("speciation_state.json not found for Flow 2 validation")
            return False, errors
        
        with open(state_file_path, 'r', encoding='utf-8') as f:
            state_file = json.load(f)
        
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
            
            leader_id = sp_dict.get("leader_id") or (sp_dict.get("leader", {}).get("id") if isinstance(sp_dict.get("leader"), dict) else None)
            member_ids = sp_dict.get("member_ids", [])
            
            if leader_id is None:
                errors.append(f"Species {sid}: leader ID is None (checked both 'leader_id' and 'leader.id')")
                continue
            
            if leader_id not in member_ids:
                errors.append(f"Species {sid}: leader {leader_id} not in member_ids {member_ids[:5]}...")
            
            elites_for_species = [g for g in elites if g.get("species_id") == sid]
            elites_member_ids = {g.get("id") for g in elites_for_species}
            
            reserves_path = outputs_path / "reserves.json"
            temp_path = outputs_path / "temp.json"
            other_member_ids = set()
            
            if reserves_path.exists():
                try:
                    with open(reserves_path, 'r', encoding='utf-8') as f:
                        reserves = json.load(f)
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
            
            missing_in_all = set(member_ids) - elites_member_ids - other_member_ids
            if missing_in_all:
                errors.append(f"Species {sid}: {len(missing_in_all)} member_ids not found in any population file (elites/reserves/temp): {list(missing_in_all)[:5]}...")
            elif set(member_ids) - elites_member_ids:
                logger.debug(f"Species {sid}: {len(set(member_ids) - elites_member_ids)} member_ids not yet in elites.json (will be distributed in Phase 7)")
            
            expected_size = len(member_ids)
            if "size" in sp_dict and sp_dict["size"] != expected_size:
                errors.append(f"Species {sid}: size mismatch - expected {expected_size} (from member_ids), got {sp_dict['size']} (from state.size)")
            
            cluster_origin = sp_dict.get("cluster_origin")
            if cluster_origin != "natural":
                errors.append(f"Species {sid}: cluster_origin is '{cluster_origin}', expected 'natural' for newly formed species")
            
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
    
    if logger is None:
        logger = get_logger("MetricsValidation")
    
    errors: List[str] = []
    
    elites_path = outputs_path / "elites.json"
    reserves_path = outputs_path / "reserves.json"
    
    try:
        elites = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
        else:
            errors.append("elites.json not found for metrics validation")
            return False, errors
        
        reserves = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves = json.load(f)
        
        unique_species_ids = {g.get("species_id") for g in elites if g.get("species_id") is not None and g.get("species_id") > 0}
        expected_species_count = len(unique_species_ids)
        actual_species_count = metrics.get("species_count", 0)
        if expected_species_count != actual_species_count:
            errors.append(f"species_count mismatch: expected {expected_species_count} (from elites.json), got {actual_species_count}")
        
        expected_total_pop = len(elites) + len(reserves)
        actual_total_pop = metrics.get("total_population", 0)
        if expected_total_pop != actual_total_pop:
            errors.append(f"total_population mismatch: expected {expected_total_pop} (elites={len(elites)}, reserves={len(reserves)}), got {actual_total_pop}")
        
        expected_reserves_size = len(reserves)
        actual_reserves_size = metrics.get("reserves_size", 0)
        if expected_reserves_size != actual_reserves_size:
            errors.append(f"reserves_size mismatch: expected {expected_reserves_size} (from reserves.json), got {actual_reserves_size}")
        
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
            if abs(expected_best - actual_best) > 0.0001:
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
