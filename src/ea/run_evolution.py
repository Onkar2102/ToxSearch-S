"""
Evolution entry point. Reads elites and reserves via population_io; writes variants
to temp.json. Species assignment and file distribution (elites/reserves/archive) are
done by speciation Phase 7, not by this module.
"""

import json
from typing import Dict, Any, List, Optional
def get_EvolutionEngine():
    """Lazy import of EvolutionEngine to avoid torch dependency issues"""
    from ea.evolution_engine import EvolutionEngine
    return EvolutionEngine
from utils import get_population_io, get_custom_logging
from utils.population_io import update_population_index_single_file

from pathlib import Path

get_logger, _, _, PerformanceLogger = get_custom_logging()

from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system, _ = get_system_utils()

project_root = Path(__file__).resolve().parents[2]

def _reset_temp_json(logger):
    """Reset temp.json to empty list at the start of variant generation."""
    try:
        temp_path = get_outputs_path() / "temp.json"
        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        logger.debug("Reset temp.json for new generation")
    except Exception as e:
        logger.error(f"Failed to reset temp.json: {e}")
        raise

def _deduplicate_variants_in_temp(logger, operator_stats=None):
    """
    Deduplicate variants in temp.json by comparing against existing genomes in all files.
    This function only performs deduplication; distribution is handled by speciation.

    Args:
        logger: Logger instance
        operator_stats: Optional OperatorStatistics instance to track duplicates

    Returns:
        int: Number of duplicates removed
    """
    try:
        outputs_path = get_outputs_path()
        temp_path = outputs_path / "temp.json"
        elites_path = outputs_path / "elites.json"
        reserves_path = outputs_path / "reserves.json"  # Renamed from non_elites.json in legacy format

        if not temp_path.exists():
            logger.warning("temp.json not found for deduplication")
            return 0

        with open(temp_path, 'r', encoding='utf-8') as f:
            temp_variants = json.load(f)

        if not temp_variants:
            logger.debug("No variants in temp.json to deduplicate")
            return 0

        existing_prompts = set()
        existing_ids = set()

        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
                for genome in elites:
                    if genome and genome.get("prompt"):
                        existing_prompts.add(genome["prompt"])  # Exact match, no normalization
                        existing_ids.add(genome.get("id"))

        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                cluster0_genomes = json.load(f)
                for genome in cluster0_genomes:
                    if genome and genome.get("prompt"):
                        existing_prompts.add(genome["prompt"])  # Exact match, no normalization
                        existing_ids.add(genome.get("id"))

        unique_variants = []
        duplicates_removed = 0

        for variant in temp_variants:
            if not variant or not variant.get("prompt"):
                duplicates_removed += 1
                continue

            prompt = variant["prompt"]  # Exact match, no normalization
            genome_id = variant.get("id")

            if prompt in existing_prompts or genome_id in existing_ids:
                duplicates_removed += 1
                if operator_stats:
                    operator_name = variant.get("creation_info", {}).get("operator", "unknown")
                    operator_stats.record_duplicate(operator_name)
                logger.debug(f"Removing duplicate genome {genome_id}")
                continue

            unique_variants.append(variant)

        with open(temp_path, 'w', encoding='utf-8') as f:
            json.dump(unique_variants, f, indent=2, ensure_ascii=False)

        logger.debug(f"Deduplication: {len(temp_variants)} → {len(unique_variants)} ({duplicates_removed} duplicates)")

        return duplicates_removed

    except Exception as e:
        logger.error(f"Failed to deduplicate variants in temp.json: {e}")
        raise


# Distribution is handled by the speciation pipeline (`run_speciation`), not this module.


population_path = None
evolution_tracker_path = None
parent_selection_tracker_path = None


def check_threshold_and_update_tracker(population, north_star_metric, log_file=None):
    """Update EvolutionTracker.json with current best score from the population.
    Uses the full population (elites + reserves) and tracker scope 'global'."""
    get_logger, _, _, _ = get_custom_logging()
    logger = get_logger("RunEvolution", log_file)
    try:
        outputs_path = get_outputs_path()
        evolution_tracker_path = outputs_path / "EvolutionTracker.json"

        if evolution_tracker_path.exists():
            try:
                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        evolution_tracker = json.loads(content)
                    else:
                        evolution_tracker = {
                            "scope": "global",
                            "status": "not_complete",
                            "total_generations": 1,
                            "generations": []
                        }
            except (json.JSONDecodeError, FileNotFoundError):
                evolution_tracker = {
                    "scope": "global",
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations": []
                }
        else:
            evolution_tracker = {
                "scope": "global",
                "status": "not_complete",
                "total_generations": 1,
                "generations": []
            }

        completed_genomes = [g for g in population if g.get("status") == "complete"]
        if completed_genomes:
            best_genome = max(completed_genomes, key=lambda g: _extract_north_star_score(g, north_star_metric))
            best_score = _extract_north_star_score(best_genome, north_star_metric)
            evolution_tracker["status"] = "not_complete"
            logger.debug("Best score: %.4f", best_score)
        else:
            evolution_tracker["status"] = "not_complete"
            logger.debug("No completed genomes found")

        if not evolution_tracker.get("generations"):
            gen0_genomes = [g for g in population if g.get("generation") == 0]
            if gen0_genomes:
                best_gen0_genome = max(gen0_genomes, key=lambda g: _extract_north_star_score(g, north_star_metric))
                best_gen0_id = best_gen0_genome["id"]
                best_gen0_score = _extract_north_star_score(best_gen0_genome, north_star_metric)
                selection_mode = evolution_tracker.get("selection_mode", "default")
                evolution_tracker["generations"] = [{
                    "generation_number": 0,
                    "genome_id": best_gen0_id,
                    "max_score_variants": best_gen0_score,
                    "min_score_variants": 0.0001,
                    "avg_fitness": 0.0001,
                    "avg_fitness_variants": 0.0001,
                    "avg_fitness_generation": 0.0001,
                    "avg_fitness_elites": 0.0001,
                    "avg_fitness_reserves": 0.0001,
                    "parents": None,
                    "top_10": None,
                    "variants_created": None,
                    "mutation_variants": None,
                    "crossover_variants": None,
                    "elites_count": 0,
                    "selection_mode": selection_mode,
                }]
                logger.debug("Created gen 0 entry: genome %s, score: %.4f", best_gen0_id, best_gen0_score)

        logger.debug("Best score: %.4f, continuing evolution", best_score if completed_genomes else 0.0)

        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)

        return evolution_tracker
    except Exception as e:
        logger.error("Failed to check threshold and update tracker: %s", e, exc_info=True)
        return {
            "scope": "global",
            "status": "error",
            "total_generations": 1,
            "generations": []
        }

def get_pending_status(evolution_tracker, logger):
    """Get status of global evolution tracker"""
    try:
        status = evolution_tracker.get("status", "not_complete")
        logger.debug("Evolution status: %s", status)
        return status
    except Exception as e:
        logger.error("Failed to get pending status: %s", e, exc_info=True)
        raise

def update_evolution_tracker_with_generation_global(generation_data, evolution_tracker, logger, population=None, north_star_metric=None):
    """Update evolution tracker with generation data for global population"""
    _logger = logger or get_logger("update_evolution_tracker", log_file=None)
    try:
        gen_number = generation_data.get("generation_number")
        if gen_number is None:
            _logger.error("generation_number not provided in generation_data")
            return

        best_genome_id = None
        best_score = 0.0001
        min_score = 0.0001
        avg_score = 0.0001

        if population and north_star_metric:
            generation_genomes = [g for g in population if g.get("generation") == gen_number]

            if generation_genomes:
                genome_scores = []
                for genome in generation_genomes:
                    score = _extract_north_star_score(genome, north_star_metric)
                    if score > 0:
                        genome_scores.append((genome["id"], score))

                if genome_scores:
                    best_genome_id, best_score = max(genome_scores, key=lambda x: x[1])
                    _, min_score = min(genome_scores, key=lambda x: x[1])
                    all_scores = [s for _, s in genome_scores]
                    avg_score = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0.0001
                    _logger.info(f"Generation {gen_number} scores: max={best_score:.4f}, min={min_score:.4f}, avg={avg_score:.4f}")
                else:
                    _logger.warning(f"No valid scores found for generation {gen_number}")
            else:
                _logger.warning(f"No genomes found for generation {gen_number}")
        else:
            if generation_data.get("parents"):
                best_parent = max(generation_data["parents"],
                                key=lambda p: p.get("north_star_score", 0.0))
                best_genome_id = best_parent["id"]
                best_score = best_parent["north_star_score"]
                _logger.warning(f"Using parent score as fallback for generation {gen_number}: {best_score}")

        avg_fitness = 0.0001
        try:
            from utils.population_io import calculate_average_fitness
            outputs_path = get_outputs_path()
            avg_fitness = calculate_average_fitness(str(outputs_path), north_star_metric, logger=_logger)
            _logger.info(f"Calculated avg_fitness for generation {gen_number}: {avg_fitness:.4f}")
        except Exception as e:
            _logger.warning(f"Failed to calculate avg_fitness for generation {gen_number}: {e}")

        existing_gen = None
        for gen in evolution_tracker.get("generations", []):
            if gen["generation_number"] == gen_number:
                existing_gen = gen
                break

        selection_mode = evolution_tracker.get("selection_mode", "default")
        
        # Import helper functions
        from utils.population_io import _get_standard_generation_entry_template, _ensure_generation_entry_has_all_fields
        
        if existing_gen:
            # Ensure existing entry has all fields
            existing_gen = _ensure_generation_entry_has_all_fields(existing_gen, gen_number, selection_mode)
            
            variants_created = generation_data.get("variants_created", 0)
            mutation_variants = generation_data.get("mutation_variants", 0)
            crossover_variants = generation_data.get("crossover_variants", 0)

            _logger.info(f"Updating generation {gen_number} with variant counts: created={variants_created}, mutation={mutation_variants}, crossover={crossover_variants}")

            # Preserve existing speciation data if present
            existing_speciation = existing_gen.get("speciation")
            
            existing_gen.update({
                "genome_id": best_genome_id,
                "max_score_variants": best_score,
                "avg_fitness": round(avg_fitness, 4),
                "variants_created": variants_created,
                "mutation_variants": mutation_variants,
                "crossover_variants": crossover_variants,
                "selection_mode": selection_mode
            })
            
            # Restore speciation data if it was present
            if existing_speciation is not None:
                existing_gen["speciation"] = existing_speciation
            
            _logger.info("Updated existing generation %d globally with max_score_variants %.4f and %d variants", gen_number, best_score, variants_created)
        else:
            _logger.warning("Generation %d not found - creating new entry", gen_number)
            variants_created = generation_data.get("variants_created", 0)
            mutation_variants = generation_data.get("mutation_variants", 0)
            crossover_variants = generation_data.get("crossover_variants", 0)

            # Create new entry with all standard fields
            new_gen = _get_standard_generation_entry_template(gen_number, selection_mode)
            new_gen.update({
                "genome_id": best_genome_id,
                "avg_fitness": round(avg_fitness, 4),
                "max_score_variants": best_score,
                "min_score_variants": min_score,
                "avg_fitness_variants": avg_score,
                "avg_fitness_generation": round(avg_fitness, 4),
                "variants_created": variants_created,
                "mutation_variants": mutation_variants,
                "crossover_variants": crossover_variants,
            })
            evolution_tracker.setdefault("generations", []).append(new_gen)
            _logger.info("Created new generation entry %d with max_score_variants %.4f and %d variants", gen_number, best_score, variants_created)

        evolution_tracker["generations"].sort(key=lambda x: x["generation_number"])


        outputs_path = get_outputs_path()
        evolution_tracker_path = outputs_path / "EvolutionTracker.json"

        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(evolution_tracker, f, indent=4, ensure_ascii=False)

        _logger.info("Updated global evolution tracker with generation %d data: %d variants created, max_score %.4f",
                   gen_number, generation_data.get("variants_created", 0), best_score)

    except Exception as e:
        _logger.error("Failed to update global evolution tracker with generation data: %s", e, exc_info=True)
        raise

def create_final_statistics_with_tracker(evolution_tracker: List[dict], north_star_metric: str,
                                       execution_time: float, generations_completed: int,
                                       *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create comprehensive final statistics using tracker information

    Parameters
    ----------
    evolution_tracker : List[dict]
        The evolution tracker containing all evolution data
    north_star_metric : str
        The north star metric used for optimization
    execution_time : float
        Total execution time in seconds
    generations_completed : int
        Number of generations completed
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created

    Returns
    -------
    Dict[str, Any]
        Comprehensive final statistics
    """
    _logger = logger or get_logger("create_final_statistics", log_file)

    try:
        total_generations = evolution_tracker.get("total_generations", 0)
        status = evolution_tracker.get("status", "not_complete")
        completed = 1 if status == "complete" else 0
        pending = 1 if status == "not_complete" else 0

        all_scores = []
        best_scores = []
        for gen_entry in evolution_tracker.get("generations", []):
            score = gen_entry.get("best_fitness", gen_entry.get("max_score_variants", 0.0001))
            all_scores.append(score)
            if gen_entry.get("generation_number") == total_generations - 1:
                best_scores.append(score)

        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0.0001
        best_avg_score = sum(best_scores) / len(best_scores) if best_scores else 0.0001
        max_score = max(all_scores) if all_scores else 0.0001
        min_score = min(all_scores) if all_scores else 0.0001

        total_variants_created = 0
        total_mutation_variants = 0
        total_crossover_variants = 0

        for gen_entry in evolution_tracker.get("generations", []):
            total_variants_created += gen_entry.get("variants_created") or 0
            total_mutation_variants += gen_entry.get("mutation_variants") or 0
            total_crossover_variants += gen_entry.get("crossover_variants") or 0

        final_stats = {
            "execution_summary": {
                "execution_time_seconds": execution_time,
                "generations_completed": generations_completed,
                "total_prompts": 1,
                "completed_prompts": completed,
                "pending_prompts": pending,
                "completion_rate": (completed * 100)
            },
            "generation_statistics": {
                "total_generations": total_generations,
                "average_generations_per_prompt": total_generations,
                "max_generations_for_any_prompt": total_generations
            },
            "score_statistics": {
                "average_score": avg_score,
                "best_average_score": best_avg_score,
                "max_score_variants": max_score,
                "min_score_variants": min_score,
                "north_star_metric": north_star_metric
            },
            "variant_statistics": {
                "total_variants_created": total_variants_created,
                "total_mutation_variants": total_mutation_variants,
                "total_crossover_variants": total_crossover_variants,
                "average_variants_per_generation": (total_variants_created / total_generations) if total_generations > 0 else 0.0
            },
            "prompt_details": []
        }

        prompt_detail = {
            "scope": "global",
            "status": status,
            "total_generations": total_generations,
            "best_score": 0.0,
            "initial_score": 0.0,
            "score_improvement": 0.0,
            "variants_created": 0
        }

        generations = evolution_tracker.get("generations", [])
        if generations:
            gen_0 = next((gen for gen in generations if gen.get("generation_number") == 0), None)
            if gen_0:
                prompt_detail["initial_score"] = gen_0.get("max_score_variants", 0.0)

            latest_gen = max(generations, key=lambda g: g.get("generation_number", 0))
            prompt_detail["best_score"] = latest_gen.get("max_score_variants", 0.0)
            prompt_detail["score_improvement"] = prompt_detail["best_score"] - prompt_detail["initial_score"]

            prompt_detail["variants_created"] = sum(gen.get("variants_created") or 0 for gen in generations)

        final_stats["prompt_details"].append(prompt_detail)

        _logger.info(f"Created comprehensive final statistics for global population")
        return final_stats

    except Exception as e:
        _logger.error(f"Failed to create final statistics: {e}", exc_info=True)
        return {
            "error": f"Failed to create final statistics: {str(e)}",
            "execution_time_seconds": execution_time,
            "generations_completed": generations_completed
        }

def run_evolution(north_star_metric, log_file=None, current_cycle=None, max_variants=1, max_num_parents=4, operators="all"):
    """Run one evolution generation with comprehensive logging.
    Steady-state support: population is loaded from reserves.json or elites.json each time; each call runs a single
    generation and can be invoked repeatedly (e.g. by an external scheduler) without an in-process loop."""
    outputs_path = get_outputs_path()
    # Check for population files - use reserves.json (cluster 0) or elites.json
    reserves_path = outputs_path / "reserves.json"
    elites_path = outputs_path / "elites.json"
    evolution_tracker_path = outputs_path / "EvolutionTracker.json"

    logger = get_logger("RunEvolution", log_file)
    logger.info("Starting evolution: cycle=%s, metric=%s", current_cycle, north_star_metric)

    # Check if any population file exists
    if not reserves_path.exists() and not elites_path.exists():
        logger.error("No population file found: checked reserves.json and elites.json")
        raise FileNotFoundError(f"No population file found in {outputs_path}")

    try:
        with PerformanceLogger(logger, "Evolution: Load population"):
            _, _, load_population, _, _, _, _, _, _, _, _, _ = get_population_io()
            population = load_population(str(outputs_path), logger=logger)
            logger.debug("Loaded %d genomes", len(population))
    except Exception as e:
        logger.error("Unexpected error loading population: %s", e, exc_info=True)
        raise

    with PerformanceLogger(logger, "Evolution: Check tracker and update"):
        evolution_tracker = check_threshold_and_update_tracker(population, north_star_metric, log_file)

    evolution_status = get_pending_status(evolution_tracker, logger)

    if evolution_status == "complete":
        logger.info("Evolution completed (tracker marked complete)")
        return

    try:
        EvolutionEngine = get_EvolutionEngine()
        engine = EvolutionEngine(north_star_metric, log_file, current_cycle=current_cycle, max_variants=max_variants, adaptive_selection_after=5, max_num_parents=max_num_parents, operators=operators, outputs_path=outputs_path)
        engine.update_next_id()
        logger.debug("EvolutionEngine next_id set to %d", engine.next_id)
    except Exception as e:
        logger.error("Failed to initialize evolution engine: %s", e, exc_info=True)
        raise

    try:
        with PerformanceLogger(logger, "Evolution: Generate variants global"):
            logger.info("Processing global evolution")
            logger.debug("Calling generate_variants_global()")
            _reset_temp_json(logger)
            engine.generate_variants_global(evolution_tracker=evolution_tracker)

            operator_stats_dict = engine.operator_stats.to_dict()
            logger.debug(f"Operator statistics: {operator_stats_dict}")

            temp_path = outputs_path / "temp.json"
            variant_count = 0
            if temp_path.exists():
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_variants = json.load(f)
                    variant_count = len(temp_variants)
            try:
                update_population_index_single_file(str(outputs_path), len(engine.genomes), logger=logger)
                logger.debug("Updated population index after evolution")
            except Exception as e:
                logger.error("Failed to update population index: %s", e, exc_info=True)
    except Exception as e:
        logger.error("Failed to process global evolution: %s", e, exc_info=True)
        raise
    logger.debug("Evolution processing completed")

    # Step 1: Remove duplicates within temp.json itself (intra-temp deduplication)
    # This removes duplicates generated by different operators or multiple calls to the same operator
    try:
        with PerformanceLogger(logger, "Evolution: Intra-temp deduplication"):
            temp_path_before = outputs_path / "temp.json"
            variants_before = 0
            if temp_path_before.exists():
                with open(temp_path_before, 'r', encoding='utf-8') as f:
                    variants_before = len(json.load(f))
            
            intra_temp_duplicates = engine._deduplicate_temp_json()
            
            if intra_temp_duplicates > 0:
                logger.info(f"Step 1 (intra-temp): Removed {intra_temp_duplicates} duplicates within temp.json ({variants_before} → {variants_before - intra_temp_duplicates})")
            else:
                logger.debug(f"Step 1 (intra-temp): No duplicates found within temp.json ({variants_before} variants)")
    except Exception as e:
        logger.warning(f"Failed to deduplicate within temp.json: {e}, continuing with population deduplication")

    # Step 2: Remove duplicates that already exist in the population (elites.json, reserves.json)
    try:
        with PerformanceLogger(logger, "Evolution: Population deduplication"):
            temp_path_before = outputs_path / "temp.json"
            variants_before = 0
            if temp_path_before.exists():
                with open(temp_path_before, 'r', encoding='utf-8') as f:
                    variants_before = len(json.load(f))
            
            duplicates_removed = _deduplicate_variants_in_temp(logger, engine.operator_stats)
            
            if duplicates_removed > 0:
                logger.info(f"Step 2 (population): Removed {duplicates_removed} duplicates against existing population ({variants_before} → {variants_before - duplicates_removed})")
            else:
                logger.debug(f"Step 2 (population): No duplicates found against population ({variants_before} variants)")
    except Exception as e:
        logger.error("Failed to deduplicate variants in temp.json: %s", e, exc_info=True)
        raise

    try:
        with PerformanceLogger(logger, "Evolution: Prepare EvolutionTracker data"):
            current_generation = current_cycle
            if current_generation is None:
                logger.error("current_cycle is None - cannot determine generation number")
                return
            new_generation_data = {
                "generation_number": current_generation,
                "genome_id": None,
                "max_score_variants": 0.0,
                "parents": [],
                "operator_statistics": operator_stats_dict
            }
    except Exception as e:
        logger.error("Failed to prepare EvolutionTracker data: %s", e, exc_info=True)

    return {
        "operator_statistics": engine.operator_stats.to_dict(),
        "total_genomes": len(engine.genomes)
    }



