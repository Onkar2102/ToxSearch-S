"""
Comprehensive validation script for EvolutionTracker.json fields.

Validates:
1. All top-level fields are updated correctly
2. All per-generation fields are updated correctly
3. Speciation block fields are updated correctly
4. Field consistency (e.g., total_population = elites_count + reserves_count)
5. Update timing and correctness
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from utils import get_custom_logging
get_logger, _, _, _ = get_custom_logging()


def validate_top_level_fields(tracker: Dict[str, Any], logger=None) -> Tuple[bool, List[str]]:
    """
    Validate all top-level EvolutionTracker fields.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("EvolutionTrackerValidation")
    
    errors = []
    
    # 1. status
    if "status" not in tracker:
        errors.append("Missing top-level field: status")
    else:
        status = tracker.get("status")
        if status not in ["not_complete", "complete"]:
            errors.append(f"Invalid status value: {status} (expected 'not_complete' or 'complete')")
    
    # 2. total_generations
    if "total_generations" not in tracker:
        errors.append("Missing top-level field: total_generations")
    else:
        total_generations = tracker.get("total_generations")
        generations = tracker.get("generations", [])
        if generations:
            max_gen_number = max(gen.get("generation_number", -1) for gen in generations)
            expected_total = max_gen_number + 1
            if total_generations != expected_total:
                errors.append(
                    f"total_generations={total_generations} doesn't match "
                    f"max(generation_number) + 1={expected_total}"
                )
    
    # 3. generations_since_improvement
    if "generations_since_improvement" not in tracker:
        errors.append("Missing top-level field: generations_since_improvement")
    else:
        gens_since_improvement = tracker.get("generations_since_improvement")
        if not isinstance(gens_since_improvement, int) or gens_since_improvement < 0:
            errors.append(f"Invalid generations_since_improvement: {gens_since_improvement} (must be non-negative int)")
    
    # 4. avg_fitness_history
    if "avg_fitness_history" not in tracker:
        errors.append("Missing top-level field: avg_fitness_history")
    else:
        avg_fitness_history = tracker.get("avg_fitness_history")
        if not isinstance(avg_fitness_history, list):
            errors.append(f"avg_fitness_history must be a list, got {type(avg_fitness_history)}")
        else:
            # Validate values match generations
            generations = tracker.get("generations", [])
            if generations and avg_fitness_history:
                # Check if values are consistent (last N generations' avg_fitness)
                recent_gens = sorted(generations, key=lambda x: x.get("generation_number", 0))[-len(avg_fitness_history):]
                for i, (hist_val, gen) in enumerate(zip(avg_fitness_history, recent_gens)):
                    gen_avg_fitness = gen.get("avg_fitness")
                    if gen_avg_fitness is not None and abs(hist_val - gen_avg_fitness) > 0.0001:
                        errors.append(
                            f"avg_fitness_history[{i}]={hist_val:.4f} doesn't match "
                            f"generation {gen.get('generation_number')} avg_fitness={gen_avg_fitness:.4f}"
                        )
    
    # 5. slope_of_avg_fitness
    if "slope_of_avg_fitness" not in tracker:
        errors.append("Missing top-level field: slope_of_avg_fitness")
    else:
        slope = tracker.get("slope_of_avg_fitness")
        if not isinstance(slope, (int, float)):
            errors.append(f"slope_of_avg_fitness must be numeric, got {type(slope)}")
    
    # 6. selection_mode
    if "selection_mode" not in tracker:
        errors.append("Missing top-level field: selection_mode")
    else:
        selection_mode = tracker.get("selection_mode")
        if selection_mode not in ["default", "exploit", "explore"]:
            errors.append(f"Invalid selection_mode: {selection_mode} (expected 'default', 'exploit', or 'explore')")
    
    # 7. population_max_toxicity
    if "population_max_toxicity" not in tracker:
        errors.append("Missing top-level field: population_max_toxicity")
    else:
        pop_max = tracker.get("population_max_toxicity")
        if not isinstance(pop_max, (int, float)) or pop_max < 0:
            errors.append(f"Invalid population_max_toxicity: {pop_max} (must be non-negative)")
        
        # Validate it's the max across all generations
        generations = tracker.get("generations", [])
        if generations:
            max_scores = []
            for gen in generations:
                max_score = gen.get("max_score_variants", 0)
                if max_score > 0:
                    max_scores.append(max_score)
            
            # Also check speciation max fitness if available
            for gen in generations:
                speciation = gen.get("speciation")
                if speciation and isinstance(speciation, dict):
                    # Note: speciation doesn't have best_fitness, but we can check if population_max_toxicity
                    # is at least as high as any max_score_variants
                    pass
            
            if max_scores:
                expected_max = max(max_scores)
                if pop_max < expected_max:
                    errors.append(
                        f"population_max_toxicity={pop_max:.4f} is less than "
                        f"max(max_score_variants)={expected_max:.4f} across generations"
                    )
    
    # 8. speciation_summary (optional but should be present after first speciation)
    speciation_summary = tracker.get("speciation_summary")
    if speciation_summary is not None:
        required_fields = ["current_species_count", "current_reserves_size"]
        for field in required_fields:
            if field not in speciation_summary:
                errors.append(f"speciation_summary missing field: {field}")
    
    # 9. cumulative_budget (optional)
    cumulative_budget = tracker.get("cumulative_budget")
    if cumulative_budget is not None:
        required_fields = ["total_llm_calls", "total_api_calls", "total_response_time", "total_evaluation_time"]
        for field in required_fields:
            if field not in cumulative_budget:
                errors.append(f"cumulative_budget missing field: {field}")
    
    # 10. cluster_quality (optional)
    cluster_quality = tracker.get("cluster_quality")
    if cluster_quality is not None:
        expected_fields = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "qd_score"]
        for field in expected_fields:
            if field not in cluster_quality:
                errors.append(f"cluster_quality missing field: {field}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_per_generation_fields(tracker: Dict[str, Any], logger=None) -> Tuple[bool, List[str]]:
    """
    Validate all per-generation fields.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("EvolutionTrackerValidation")
    
    errors = []
    generations = tracker.get("generations", [])
    
    if not generations:
        errors.append("No generations found in tracker")
        return False, errors
    
    # Track generation numbers for uniqueness check
    gen_numbers = set()
    
    for gen in generations:
        gen_num = gen.get("generation_number")
        if gen_num is None:
            errors.append("Generation entry missing generation_number")
            continue
        
        if gen_num in gen_numbers:
            errors.append(f"Duplicate generation_number: {gen_num}")
        gen_numbers.add(gen_num)
        
        # Required fields
        required_fields = [
            "generation_number",
            "genome_id",
            "max_score_variants",
            "min_score_variants",
            "avg_fitness",
            "avg_fitness_generation",
            "avg_fitness_variants",
            "avg_fitness_elites",
            "avg_fitness_reserves",
            "elites_count",
            "reserves_count",
            "archived_count",
            "total_population",
            "selection_mode",
        ]
        
        for field in required_fields:
            if field not in gen:
                errors.append(f"Generation {gen_num} missing required field: {field}")
        
        # Validate numeric fields
        numeric_fields = [
            "max_score_variants", "min_score_variants", "avg_fitness",
            "avg_fitness_generation", "avg_fitness_variants",
            "avg_fitness_elites", "avg_fitness_reserves"
        ]
        for field in numeric_fields:
            if field in gen:
                value = gen[field]
                if not isinstance(value, (int, float)) or value < 0:
                    errors.append(f"Generation {gen_num} has invalid {field}: {value}")
        
        # Validate count fields
        count_fields = ["elites_count", "reserves_count", "archived_count", "total_population"]
        for field in count_fields:
            if field in gen:
                value = gen[field]
                if not isinstance(value, int) or value < 0:
                    errors.append(f"Generation {gen_num} has invalid {field}: {value}")
        
        # Validate total_population = elites_count + reserves_count
        if "total_population" in gen and "elites_count" in gen and "reserves_count" in gen:
            total_pop = gen["total_population"]
            elites = gen["elites_count"]
            reserves = gen["reserves_count"]
            expected_total = elites + reserves
            if total_pop != expected_total:
                errors.append(
                    f"Generation {gen_num}: total_population={total_pop} != "
                    f"elites_count + reserves_count={elites + reserves}"
                )
        
        # Validate selection_mode matches tracker-level
        if "selection_mode" in gen:
            gen_selection_mode = gen["selection_mode"]
            tracker_selection_mode = tracker.get("selection_mode", "default")
            # Note: selection_mode can change during a generation, so this is just a warning
            # We'll check if it's at least a valid value
            if gen_selection_mode not in ["default", "exploit", "explore"]:
                errors.append(f"Generation {gen_num} has invalid selection_mode: {gen_selection_mode}")
        
        # Validate variant counts
        if "variants_created" in gen:
            variants_created = gen.get("variants_created", 0)
            mutation_variants = gen.get("mutation_variants", 0)
            crossover_variants = gen.get("crossover_variants", 0)
            
            if mutation_variants + crossover_variants > variants_created:
                errors.append(
                    f"Generation {gen_num}: mutation_variants + crossover_variants "
                    f"({mutation_variants + crossover_variants}) > variants_created ({variants_created})"
                )
        
        # Validate parents and top_10 (optional but should be lists if present)
        if "parents" in gen and not isinstance(gen["parents"], list):
            errors.append(f"Generation {gen_num} has invalid parents field (must be list)")
        
        if "top_10" in gen and not isinstance(gen["top_10"], list):
            errors.append(f"Generation {gen_num} has invalid top_10 field (must be list)")
        
        # Validate budget (optional)
        budget = gen.get("budget")
        if budget is not None:
            required_budget_fields = ["llm_calls", "api_calls", "total_response_time", "total_evaluation_time"]
            for field in required_budget_fields:
                if field not in budget:
                    errors.append(f"Generation {gen_num} budget missing field: {field}")
    
    # Validate generation numbers are sequential
    if gen_numbers:
        sorted_gen_nums = sorted(gen_numbers)
        for i, gen_num in enumerate(sorted_gen_nums):
            if gen_num != i:
                errors.append(f"Generation numbers not sequential: expected {i}, got {gen_num}")
                break  # Only report first mismatch
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_speciation_block(tracker: Dict[str, Any], logger=None) -> Tuple[bool, List[str]]:
    """
    Validate speciation nested object fields.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("EvolutionTrackerValidation")
    
    errors = []
    generations = tracker.get("generations", [])
    
    for gen in generations:
        gen_num = gen.get("generation_number")
        speciation = gen.get("speciation")
        
        if speciation is None:
            # Speciation block is optional for generation 0 before speciation runs
            if gen_num == 0:
                continue
            else:
                errors.append(f"Generation {gen_num} missing speciation block")
            continue
        
        if not isinstance(speciation, dict):
            errors.append(f"Generation {gen_num} has invalid speciation field (must be dict)")
            continue
        
        # Required fields in speciation block
        required_fields = [
            "species_count", "active_species_count", "frozen_species_count",
            "reserves_size", "speciation_events", "merge_events",
            "extinction_events", "archived_count", "elites_moved",
            "reserves_moved", "genomes_updated", "inter_species_diversity",
            "intra_species_diversity", "total_population"
        ]
        
        for field in required_fields:
            if field not in speciation:
                errors.append(f"Generation {gen_num} speciation missing field: {field}")
        
        # Validate numeric fields
        numeric_fields = [
            "species_count", "active_species_count", "frozen_species_count",
            "reserves_size", "speciation_events", "merge_events",
            "extinction_events", "archived_count", "elites_moved",
            "reserves_moved", "genomes_updated", "inter_species_diversity",
            "intra_species_diversity", "total_population"
        ]
        for field in numeric_fields:
            if field in speciation:
                value = speciation[field]
                if not isinstance(value, (int, float)):
                    errors.append(f"Generation {gen_num} speciation has invalid {field}: {value} (must be numeric)")
                elif field in ["species_count", "active_species_count", "frozen_species_count",
                               "reserves_size", "speciation_events", "merge_events",
                               "extinction_events", "archived_count", "elites_moved",
                               "reserves_moved", "genomes_updated", "total_population"]:
                    if value < 0:
                        errors.append(f"Generation {gen_num} speciation has negative {field}: {value}")
        
        # Validate species_count = active_species_count + frozen_species_count
        if "species_count" in speciation and "active_species_count" in speciation and "frozen_species_count" in speciation:
            species_count = speciation["species_count"]
            active_count = speciation["active_species_count"]
            frozen_count = speciation["frozen_species_count"]
            expected_total = active_count + frozen_count
            if species_count != expected_total:
                errors.append(
                    f"Generation {gen_num} speciation: species_count={species_count} != "
                    f"active_species_count + frozen_species_count={active_count + frozen_count}"
                )
        
        # Validate cluster_quality (optional)
        cluster_quality = speciation.get("cluster_quality")
        if cluster_quality is not None:
            if not isinstance(cluster_quality, dict):
                errors.append(f"Generation {gen_num} speciation.cluster_quality must be dict")
            else:
                expected_fields = ["silhouette_score", "davies_bouldin_index", "calinski_harabasz_index", "qd_score"]
                for field in expected_fields:
                    if field not in cluster_quality:
                        errors.append(f"Generation {gen_num} speciation.cluster_quality missing field: {field}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_field_consistency(tracker: Dict[str, Any], logger=None) -> Tuple[bool, List[str]]:
    """
    Validate field consistency across the tracker.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    if logger is None:
        logger = get_logger("EvolutionTrackerValidation")
    
    errors = []
    generations = tracker.get("generations", [])
    
    # Check that population_max_toxicity is consistent with max_score_variants
    pop_max = tracker.get("population_max_toxicity", 0)
    if pop_max > 0:
        max_scores = []
        for gen in generations:
            max_score = gen.get("max_score_variants", 0)
            if max_score > 0:
                max_scores.append(max_score)
        
        if max_scores:
            expected_max = max(max_scores)
            if pop_max < expected_max:
                errors.append(
                    f"population_max_toxicity={pop_max:.4f} < max(max_score_variants)={expected_max:.4f}"
                )
    
    # Check that avg_fitness_history values match generation avg_fitness values
    avg_fitness_history = tracker.get("avg_fitness_history", [])
    if avg_fitness_history and generations:
        recent_gens = sorted(generations, key=lambda x: x.get("generation_number", 0))[-len(avg_fitness_history):]
        for i, (hist_val, gen) in enumerate(zip(avg_fitness_history, recent_gens)):
            gen_avg_fitness = gen.get("avg_fitness")
            if gen_avg_fitness is not None and abs(hist_val - gen_avg_fitness) > 0.0001:
                errors.append(
                    f"avg_fitness_history[{i}]={hist_val:.4f} != "
                    f"generation {gen.get('generation_number')} avg_fitness={gen_avg_fitness:.4f}"
                )
    
    # Check that selection_mode in generations matches tracker-level (for recent generations)
    tracker_selection_mode = tracker.get("selection_mode", "default")
    if generations:
        latest_gen = max(generations, key=lambda x: x.get("generation_number", 0))
        gen_selection_mode = latest_gen.get("selection_mode")
        # Note: This is informational - selection_mode can change, so we just check it's valid
        if gen_selection_mode and gen_selection_mode not in ["default", "exploit", "explore"]:
            errors.append(f"Latest generation has invalid selection_mode: {gen_selection_mode}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def validate_evolution_tracker_comprehensive(tracker_path: Path, logger=None) -> Dict[str, Any]:
    """
    Run all comprehensive validations on EvolutionTracker.json.
    
    Returns:
        Dictionary with validation results
    """
    if logger is None:
        logger = get_logger("EvolutionTrackerValidation")
    
    results = {
        "top_level_fields": {"valid": False, "errors": []},
        "per_generation_fields": {"valid": False, "errors": []},
        "speciation_block": {"valid": False, "errors": []},
        "field_consistency": {"valid": False, "errors": []},
    }
    
    tracker_path = Path(tracker_path)
    
    if not tracker_path.exists():
        logger.error(f"EvolutionTracker.json not found at {tracker_path}")
        for key in results:
            results[key] = {"valid": False, "errors": [f"File not found: {tracker_path}"]}
        return results
    
    try:
        with open(tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load EvolutionTracker.json: {e}")
        for key in results:
            results[key] = {"valid": False, "errors": [f"Failed to load file: {e}"]}
        return results
    
    # 1. Validate top-level fields
    logger.info("Validating top-level fields...")
    is_valid, errors = validate_top_level_fields(tracker, logger)
    results["top_level_fields"] = {"valid": is_valid, "errors": errors}
    if is_valid:
        logger.info("✓ Top-level fields validation passed")
    else:
        logger.warning(f"✗ Top-level fields validation failed: {len(errors)} errors")
    
    # 2. Validate per-generation fields
    logger.info("Validating per-generation fields...")
    is_valid, errors = validate_per_generation_fields(tracker, logger)
    results["per_generation_fields"] = {"valid": is_valid, "errors": errors}
    if is_valid:
        logger.info("✓ Per-generation fields validation passed")
    else:
        logger.warning(f"✗ Per-generation fields validation failed: {len(errors)} errors")
    
    # 3. Validate speciation block
    logger.info("Validating speciation block...")
    is_valid, errors = validate_speciation_block(tracker, logger)
    results["speciation_block"] = {"valid": is_valid, "errors": errors}
    if is_valid:
        logger.info("✓ Speciation block validation passed")
    else:
        logger.warning(f"✗ Speciation block validation failed: {len(errors)} errors")
    
    # 4. Validate field consistency
    logger.info("Validating field consistency...")
    is_valid, errors = validate_field_consistency(tracker, logger)
    results["field_consistency"] = {"valid": is_valid, "errors": errors}
    if is_valid:
        logger.info("✓ Field consistency validation passed")
    else:
        logger.warning(f"✗ Field consistency validation failed: {len(errors)} errors")
    
    # Summary
    all_valid = all(r["valid"] for r in results.values())
    total_errors = sum(len(r["errors"]) for r in results.values())
    
    logger.info("=" * 60)
    if all_valid:
        logger.info("✓ ALL VALIDATIONS PASSED")
    else:
        logger.warning(f"✗ VALIDATION FAILED: {total_errors} total errors")
        for check_name, result in results.items():
            if not result["valid"]:
                logger.warning(f"  - {check_name}: {len(result['errors'])} errors")
    logger.info("=" * 60)
    
    return results


if __name__ == "__main__":
    import sys
    from utils import get_system_utils
    _, _, _, get_outputs_path, _, _, _ = get_system_utils()
    
    if len(sys.argv) > 1:
        tracker_path = Path(sys.argv[1])
    else:
        outputs_path = Path(get_outputs_path())
        tracker_path = outputs_path / "EvolutionTracker.json"
    
    logger = get_logger("EvolutionTrackerValidation")
    results = validate_evolution_tracker_comprehensive(tracker_path, logger)
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVOLUTIONTRACKER VALIDATION SUMMARY")
    print("=" * 60)
    for check_name, result in results.items():
        status = "✓ PASS" if result["valid"] else "✗ FAIL"
        error_count = len(result["errors"])
        print(f"{status} {check_name}: {error_count} errors")
        if error_count > 0 and error_count <= 10:
            for error in result["errors"]:
                print(f"  - {error}")
        elif error_count > 10:
            for error in result["errors"][:10]:
                print(f"  - {error}")
            print(f"  ... and {error_count - 10} more errors")
    print("=" * 60)
