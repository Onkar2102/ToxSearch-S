"""
main.py

Entry point for the evolutionary pipeline. Orchestrates the generation loop:
load/reset state, run evolution (population_io, EA), run moderation, run speciation,
and persist state. Speciation is invoked after each generation; state is reset or
loaded from outputs (elites, reserves, speciation_state) as configured.
"""

import sys
import time
import json
import os

from typing import Optional
from pathlib import Path
from datetime import datetime

from utils.device_utils import get_optimal_device, get_device_info

DEVICE = get_optimal_device()
DEVICE_INFO = get_device_info()

from utils import get_custom_logging
from utils.population_io import (
    update_population_index_single_file,
    update_adaptive_selection_logic,
    calculate_average_fitness,
    calculate_generation_statistics,
    update_evolution_tracker_with_statistics
)
from speciation.config import SpeciationConfig
from gne import get_run_moderation_on_population
from utils import get_population_io
from ea.run_evolution import run_evolution
from ea import get_create_final_statistics_with_tracker
import yaml

from utils import get_system_utils
get_project_root, get_config_path, get_data_path, get_outputs_path, _extract_north_star_score, initialize_system, set_outputs_path = get_system_utils()


def _is_gguf_path(value: str) -> bool:
    """Return True if the given value looks like a direct GGUF file path."""
    p = Path(value)
    return str(value).lower().endswith(".gguf") and (p.is_absolute() or str(value).startswith("./") or str(value).startswith("models/"))

def update_model_configs(rg_model, pg_model, logger):
    """Update configuration files with selected models.

    Resolves the concrete .gguf file for each alias by scanning models/{alias}.
    
    Model Quantization Preference Order (Performance vs. Size Trade-offs):
    Q4_K_M → Q4_K_S → Q4_0 → Q5_K_M → Q5_K_S → Q4_K → Q3_K_M → Q3_K_L → Q2_K
    
    - Q4_K_M: Best balance of quality and speed (recommended)
    - Q4_K_S: Smaller size, slightly lower quality
    - Q5_K_M: Higher quality, larger size, slower inference
    - Q3_K_M: Lower quality, smaller size, faster inference
    - Q2_K: Lowest quality, smallest size, fastest inference
    """
    try:
        logger.info("Updating config files with models: RG=%s, PG=%s", rg_model, pg_model)

        pref_order = [
            "f32", "Q8_0", "Q8_K", "Q8_K_M", "Q4_K_M", "Q4_K_S", "Q4_0", "Q5_K_M", "Q5_K_S", "Q4_K", "Q3_K_M", "Q3_K_L", "Q2_K"
        ]

        def resolve_model_entry(value: str) -> Optional[str]:
            """
            Resolve the provided model value to a concrete path string.
            - If it's a direct .gguf path that exists, return it as-is
            - If it's a direct .gguf path that does not exist, resolve from parent dir and prefer the quantization in the requested filename (e.g. Q8_0)
            - Otherwise, treat it as an alias directory under models/ and pick a file by preference
            """
            if not value:
                return None
            requested_path = None  # When we fall back to alias from a missing .gguf path, keep hint for preference
            if _is_gguf_path(value):
                p = Path(value)
                if not p.is_absolute():
                    p = get_project_root() / value
                if p.exists():
                    return value
                # File missing: resolve from parent directory; prefer quantization from requested filename
                parent = p.parent
                if parent.exists():
                    requested_path = value  # e.g. .../Llama-3.1-8B-Instruct.Q8_0.gguf
                    alias = str(Path(value).parent).replace("\\", "/")
                    value = alias
                else:
                    logger.warning("Model file not found and parent dir missing: %s", value)
                    return None

            alias = value
            if str(alias).startswith("models/") or Path(alias).is_absolute():
                base_dir = get_project_root() / alias if not Path(alias).is_absolute() else Path(alias)
            else:
                base_dir = get_project_root() / "models" / alias
            if not base_dir.exists():
                logger.warning("Model alias directory not found: %s", base_dir)
                return None
            ggufs = sorted([p for p in base_dir.glob("*.gguf")], key=lambda p: p.name)
            if not ggufs:
                logger.warning("No GGUF files found under: %s", base_dir)
                return None
            # If user requested a specific file (e.g. ...Q8_0.gguf), prefer that quantization when resolving
            order = pref_order
            if requested_path:
                preferred = [q for q in pref_order if q in requested_path]
                if preferred:
                    order = preferred + [p for p in pref_order if p not in preferred]
                    logger.info("Preferring quantization from requested path: %s", preferred[0])
            for pref in order:
                for f in ggufs:
                    if pref in f.name:
                        rel = (Path(alias) / f.name) if str(alias).startswith("models/") else (Path("./models") / alias / f.name)
                        logger.info("Resolved %s -> %s", alias, rel)
                        return str(rel)

        rg_file = resolve_model_entry(rg_model)
        pg_file = resolve_model_entry(pg_model)

        if not rg_file and not pg_file:
            logger.error("No models could be resolved for RG=%s, PG=%s", rg_model, pg_model)
            raise ValueError(f"No models could be resolved for RG={rg_model}, PG={pg_model}")

        rg_config_path = get_config_path() / "RGConfig.yaml"
        if rg_config_path.exists():
            with open(rg_config_path, 'r') as f:
                rg_config = yaml.safe_load(f) or {}

            if rg_file:
                rg_section = rg_config.get("response_generator", {})
                rg_section["name"] = rg_file
                rg_config["response_generator"] = rg_section
                with open(rg_config_path, 'w') as f:
                    yaml.dump(rg_config, f, default_flow_style=False)
                logger.info("Config updated from script (--rg): RGConfig.yaml response_generator.name = %s", rg_file)
            else:
                logger.warning("Skipped RGConfig.yaml update; no file resolved for alias '%s'", rg_model)

        pg_config_path = get_config_path() / "PGConfig.yaml"
        if pg_config_path.exists():
            with open(pg_config_path, 'r') as f:
                pg_config = yaml.safe_load(f) or {}

            if pg_file:
                pg_section = pg_config.get("prompt_generator", {})
                pg_section["name"] = pg_file
                pg_config["prompt_generator"] = pg_section
                with open(pg_config_path, 'w') as f:
                    yaml.dump(pg_config, f, default_flow_style=False)
                logger.info("Config updated from script (--pg): PGConfig.yaml prompt_generator.name = %s", pg_file)
            else:
                logger.warning("Skipped PGConfig.yaml update; no file resolved for alias '%s'", pg_model)

        logger.info("Project configs updated from script parameters: RG=%s, PG=%s", rg_file or "(unchanged)", pg_file or "(unchanged)")

    except Exception as e:
        logger.error("Failed to update model configurations: %s", e)
        raise


def main(max_generations=None, moderation_methods=None, rg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", pg_model="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf", operators="all", max_variants=1, stagnation_limit=5, seed_file="data/prompt.csv",
         max_total_genomes=None, seed=None,
         # Speciation parameters
         theta_sim=0.2, theta_merge=0.1, min_stability_gens=5, species_capacity=100, cluster0_max_capacity=1000,
         cluster0_min_cluster_size=2, min_island_size=2, species_stagnation=20,
         embedding_model="all-MiniLM-L6-v2", embedding_dim=384, embedding_batch_size=64):
    """
    Main entry point for evolutionary text generation with toxicity optimization.

    Runs the evolutionary algorithm to generate and evolve text prompts, optimizing
    for low toxicity scores using content moderation (Google Perspective API by
    default). Evolution runs for the specified number of generations.

    Args:
        max_generations (int, optional): Not used for termination. Kept for API compatibility;
            termination is only by max_total_genomes.
        moderation_methods (List[str], optional): Moderation APIs to use.
            If None, uses ['google'] for evaluation.
        rg_model (str): Response generator model path. Default GGUF path.
        pg_model (str): Prompt generator model path. Default GGUF path.
        operators (str): Operator configuration mode. Options:
            "ie" (InformedEvolution only), "cm" (mutation-crossover only), "all" (default).
        max_variants (int): Maximum number of variants per evolution cycle. Default 1.
        stagnation_limit (int): Unused; kept for API compatibility.
        seed_file (str): Path to initial prompt CSV. Default "data/prompt.csv".
        theta_sim, theta_merge, min_stability_gens, species_capacity, cluster0_*, min_island_size,
        species_stagnation, embedding_*: Speciation and embedding configuration.

    Returns:
        None
    """
    
    get_logger, get_log_filename, log_system_info, PerformanceLogger = get_custom_logging()
    log_file = get_log_filename()
    logger = get_logger("main", log_file)
    
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Device info: {DEVICE_INFO}")
    
    log_system_info(logger)
    
    if moderation_methods is None:
        moderation_methods = ["google"]
    
    if "all" in moderation_methods:
        moderation_methods = ["google"]
    
    start_time = time.time()
    north_star_metric = "toxicity"
    # Primary termination is by total genomes (elites + reserves + archives).
    if max_total_genomes is None:
        raise ValueError("Sequential mode requires --max-total-genomes; primary termination is by total genomes.")
    logger.info("Starting evolutionary run: metric=%s, max_total_genomes=%s (primary termination criteria)",
                north_star_metric, max_total_genomes)

    # Resolve RG/PG model paths (aliases → concrete GGUF files) before initialization
    try:
        with PerformanceLogger(logger, "Update model configs"):
            update_model_configs(rg_model, pg_model, logger)
    except Exception as e:
        logger.error("Config update failed: %s", e, exc_info=True)
        return

    # Boot model instances, outputs directory, and seed data
    try:
        with PerformanceLogger(logger, "Initialize system", seed_file=seed_file):
            response_generator, prompt_generator = initialize_system(logger, log_file, seed_file=seed_file, seed=seed)
    except Exception as e:
        logger.error("System initialization failed: %s", e, exc_info=True)
        return

    # Gen 0: wall-clock start (for generation_duration_seconds in tracker)
    gen0_start = time.time()
    # Generate initial candidates into temp.json
    try:
        with PerformanceLogger(logger, "Gen 0: Generate initial responses"):
            logger.info("Generating responses using response generation model...")
            temp_path = str(get_outputs_path() / "temp.json")
            response_generator.process_population(pop_path=temp_path)
        
        # Validate that temp.json has content
        temp_path_obj = get_outputs_path() / "temp.json"
        if temp_path_obj.exists():
            with open(temp_path_obj, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
                if not temp_genomes or len(temp_genomes) == 0:
                    logger.error("Generation 0 failed: temp.json is empty after response generation.")
                    logger.error("No initial population was generated. Check seed file and model configuration.")
                    return
                logger.info("Generated %d initial genomes in temp.json", len(temp_genomes))
        else:
            logger.error("Generation 0 failed: temp.json was not created.")
            return
    except Exception as e:
        logger.error("Generation failed: %s", e, exc_info=True)
        return

    # Score generated candidates via moderation APIs
    try:
        with PerformanceLogger(logger, "Gen 0: Evaluate (moderation)", methods=", ".join(moderation_methods)):
            run_moderation_on_population = get_run_moderation_on_population()
            logger.info("Evaluating generated responses using moderation (%s)...", " + ".join(moderation_methods))
            temp_path = str(get_outputs_path() / "temp.json")
            run_moderation_on_population(
                pop_path=temp_path,
                log_file=log_file,
                north_star_metric=north_star_metric,
                moderation_methods=moderation_methods
            )
        
        # Validate that temp.json still has content after evaluation
        temp_path_obj = get_outputs_path() / "temp.json"
        if temp_path_obj.exists():
            with open(temp_path_obj, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
                if not temp_genomes or len(temp_genomes) == 0:
                    logger.error("Generation 0 failed: temp.json is empty after evaluation.")
                    logger.error("All genomes were removed during evaluation.")
                    return
    except Exception as e:
        logger.error("Evaluation failed: %s", e, exc_info=True)
        return

    # Apply refusal penalties after evaluation, before speciation (not needed after implementing NSGA-3)
    try:
        with PerformanceLogger(logger, "Gen 0: Apply refusal penalties"):
            from utils.refusal_penalty import apply_refusal_penalties
            refusal_stats = apply_refusal_penalties(
                pop_path=temp_path,
                north_star_metric=north_star_metric,
                logger=logger,
                log_file=log_file
            )
            logger.info("Gen 0: Refusal analysis - %d refusals detected, %d penalties applied",
                       refusal_stats.get("refusals_detected", 0),
                       refusal_stats.get("penalties_applied", 0))
    except Exception as e:
        logger.error("Gen 0: Refusal penalty application failed: %s", e, exc_info=True)

    # avg_fitness: mean(old elites + old reserves + all new variants) before speciation, after evaluation.
    # Gen 0: elites/reserves empty, so effectively mean(temp).
    avg_fitness_before_speciation = 0.0001
    try:
        with PerformanceLogger(logger, "Gen 0: Calculate avg_fitness before speciation"):
            avg_fitness_before_speciation = calculate_average_fitness(
                str(get_outputs_path()), north_star_metric, include_temp=True, logger=logger, log_file=log_file
            )
    except Exception as e:
        logger.warning("Gen 0: Failed to compute avg_fitness before speciation: %s", e)

    # Create speciation config from parameters
    speciation_config = SpeciationConfig(
        theta_sim=theta_sim,
        theta_merge=theta_merge,
        min_stability_gens=min_stability_gens,
        species_capacity=species_capacity,
        cluster0_max_capacity=cluster0_max_capacity,
        cluster0_min_cluster_size=cluster0_min_cluster_size,
        min_island_size=min_island_size,
        species_stagnation=species_stagnation,
        embedding_model=embedding_model,
        embedding_dim=embedding_dim,
        embedding_batch_size=embedding_batch_size
    )

    # Calculate max_score_variants from temp.json BEFORE speciation (temp.json gets cleared during speciation)
    # max_score_variants should reflect the highest fitness from variants created in this generation
    temp_path_obj = get_outputs_path() / "temp.json"
    gen0_max_score = 0.0001
    if temp_path_obj.exists():
        try:
            with open(temp_path_obj, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
            if temp_genomes:
                scores = [_extract_north_star_score(g, north_star_metric) for g in temp_genomes if g]
                scores = [s for s in scores if s > 0]  # Filter out zero scores
                if scores:
                    gen0_max_score = round(max(scores), 4)
        except Exception:
            pass
    
    # Assign species/reserves for generation 0 (this also distributes genomes)
    try:
        with PerformanceLogger(logger, "Gen 0: Run speciation"):
            logger.info("Running speciation on evaluated genomes...")
            from speciation import run_speciation
            
            temp_path = str(get_outputs_path() / "temp.json")
            speciation_result = run_speciation(
                temp_path=temp_path,
                current_generation=0,
                config=speciation_config,
                log_file=log_file,
                north_star_metric=north_star_metric
            )
            
            if speciation_result.get("success"):
                logger.info("Speciation complete: %d species, %d in reserves, %d genomes updated",
                           speciation_result.get("species_count", 0),
                           speciation_result.get("reserves_size", 0),
                           speciation_result.get("genomes_updated", 0))
            else:
                logger.warning("Speciation completed with warnings: %s", speciation_result.get("error", "unknown"))
    except Exception as e:
        logger.error("Speciation failed: %s", e, exc_info=True)
        return
    
    # Phase 4.5: Calculate operator effectiveness metrics for Generation 0 (RQ1)
    try:
        with PerformanceLogger(logger, "Gen 0: Operator effectiveness metrics"):
            from utils.operator_effectiveness import (
                calculate_table4_metrics, 
                save_operator_effectiveness_cumulative,
                generate_operator_effectiveness_visualizations
            )
            
            operator_metrics_df = calculate_table4_metrics(
                outputs_path=str(get_outputs_path()),
                current_generation=0,
                north_star_metric=north_star_metric,
                logger=logger
            )
            
            if operator_metrics_df is not None:
                # For generation 0, DataFrame will be empty (no operator-created variants)
                # But we still save it to maintain consistency in the cumulative file
                save_operator_effectiveness_cumulative(
                    metrics_df=operator_metrics_df,
                    outputs_path=str(get_outputs_path()),
                    current_generation=0,
                    logger=logger
                )
                
                # Only generate visualizations if we have data (not for generation 0)
                if not operator_metrics_df.empty:
                    viz_paths = generate_operator_effectiveness_visualizations(
                        outputs_path=str(get_outputs_path()),
                        current_generation=0,
                        logger=logger
                    )
                    logger.info("Gen 0: Operator effectiveness metrics calculated and saved (%d operators, %d visualizations)", 
                               len(operator_metrics_df), len(viz_paths))
                else:
                    logger.info("Gen 0: No operator-created variants (initial seed population). Metrics file initialized.")
            else:
                logger.warning("Gen 0: Failed to initialize operator effectiveness metrics")
    except Exception as e:
        logger.warning("Gen 0: Failed to calculate operator effectiveness metrics: %s", e)

    # Phase 5: Calculate comprehensive generation 0 statistics
    try:
        with PerformanceLogger(logger, "Gen 0: Calculate generation statistics and update EvolutionTracker"):
            evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
            
            # Calculate comprehensive generation statistics
            gen0_stats = calculate_generation_statistics(
                outputs_path=str(get_outputs_path()),
                north_star_metric=north_star_metric,
                current_generation=0,
                logger=logger,
                log_file=log_file
            )
            
            # Add additional metrics to stats
            # max_score_variants should reflect the highest fitness from temp.json (variants created in this generation)
            # NOTE: population_max_toxicity should be max from elites+reserves (after distribution),
            # NOT from temp.json variants. calculate_generation_statistics() already calculates this correctly.
            # For generation 0, gen0_stats["population_max_toxicity"] is already set correctly by calculate_generation_statistics()
            gen0_stats["best_genome_id"] = None  # Can be calculated if needed
            gen0_stats["avg_fitness"] = avg_fitness_before_speciation  # Before speciation, after evaluation
            gen0_stats["variants_created"] = 0  # No operator-generated variants in generation 0
            gen0_stats["mutation_variants"] = 0
            gen0_stats["crossover_variants"] = 0
            # Genomes added to population in gen 0 = seed-evaluated and placed (elites + reserves + archive)
            gen0_stats["variants_integrated"] = (
                gen0_stats.get("elites_count", 0)
                + gen0_stats.get("reserves_count", 0)
                + gen0_stats.get("archived_count", 0)
            )
            # For generation 0, max_score_variants is the max from temp.json (initial population before speciation)
            gen0_stats["max_score_variants"] = gen0_max_score
            gen0_stats["min_score_variants"] = 0.0001  # Default for generation 0
            gen0_stats["avg_fitness_variants"] = 0.0001  # Default for generation 0
            
            # Add speciation metrics from the speciation result (for EvolutionTracker speciation block)
            gen0_stats["species_count"] = speciation_result.get("species_count", 0)
            gen0_stats["active_species_count"] = speciation_result.get("active_species_count", 0)
            gen0_stats["frozen_species_count"] = speciation_result.get("frozen_species_count", 0)
            gen0_stats["reserves_size"] = speciation_result.get("reserves_size", 0)
            gen0_stats["speciation_events"] = speciation_result.get("speciation_events", 0)
            gen0_stats["merge_events"] = speciation_result.get("merge_events", 0)
            gen0_stats["extinction_events"] = speciation_result.get("extinction_events", 0)
            gen0_stats["archived_count"] = speciation_result.get("archived_count", 0)
            gen0_stats["elites_moved"] = speciation_result.get("elites_moved", 0)
            gen0_stats["reserves_moved"] = speciation_result.get("reserves_moved", 0)
            gen0_stats["genomes_updated"] = speciation_result.get("genomes_updated", 0)
            gen0_stats["generation_duration_seconds"] = time.time() - gen0_start
            if speciation_result.get("speciation_duration_seconds") is not None:
                gen0_stats["speciation_duration_seconds"] = speciation_result["speciation_duration_seconds"]
            
            # Update EvolutionTracker with all statistics (include run params for RQ analysis)
            update_evolution_tracker_with_statistics(
                evolution_tracker_path=str(evolution_tracker_path),
                current_generation=0,
                statistics=gen0_stats,
                operator_statistics=None,  # No operators in generation 0
                logger=logger,
                log_file=log_file,
                run_metadata_update=dict(
                    theta_sim=theta_sim,
                    species_capacity=species_capacity,
                    cluster0_max_capacity=cluster0_max_capacity,
                    **({"max_total_genomes": max_total_genomes} if max_total_genomes is not None else {}),
                ),
            )
            
            # Update adaptive selection logic (AFTER statistics are calculated, consistent with Generation N)
            # Use population_max_toxicity from gen0_stats (max from elites+reserves), not max_toxicity from temp.json
            gen0_population_max = gen0_stats.get("population_max_toxicity", 0.0001)
            try:
                adaptive_results = update_adaptive_selection_logic(
                    outputs_path=str(get_outputs_path()),
                    current_max_toxicity=gen0_population_max,  # Use population_max_toxicity (consistent with Gen N)
                    previous_max_toxicity=0.0,
                    stagnation_limit=stagnation_limit,
                    north_star_metric=north_star_metric,
                    current_gen_avg_fitness=avg_fitness_before_speciation,
                    logger=logger,
                    log_file=log_file
                )
                logger.debug("Adaptive selection updated: mode=%s, generations_since_improvement=%d, avg_fitness=%.4f, slope=%.4f",
                           adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                           adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
            except Exception as e:
                logger.warning("Failed to update adaptive selection logic: %s", e)
            
            logger.info("Gen0 metrics: elites=%d (avg=%.4f), reserves=%d (avg=%.4f), archived=%d, total=%d, avg_gen=%.4f",
                        gen0_stats["elites_count"], gen0_stats["avg_fitness_elites"],
                        gen0_stats["reserves_count"], gen0_stats["avg_fitness_reserves"],
                        gen0_stats.get("archived_count", 0), gen0_stats["total_population"], gen0_stats["avg_fitness_generation"])
    except Exception as e:
        logger.warning("Failed to update generation 0 metrics in EvolutionTracker: %s", e)

    # Validate that Generation 0 populated the population files before proceeding
    elites_path = get_outputs_path() / "elites.json"
    reserves_path = get_outputs_path() / "reserves.json"
    has_population = False
    
    if elites_path.exists():
        try:
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_data = json.load(f)
                if isinstance(elites_data, list) and len(elites_data) > 0:
                    has_population = True
                    logger.info("Generation 0 validation: elites.json has %d genomes", len(elites_data))
        except Exception as e:
            logger.warning("Failed to read elites.json for validation: %s", e)
    
    if not has_population and reserves_path.exists():
        try:
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_data = json.load(f)
                if isinstance(reserves_data, list) and len(reserves_data) > 0:
                    has_population = True
                    logger.info("Generation 0 validation: reserves.json has %d genomes", len(reserves_data))
        except Exception as e:
            logger.warning("Failed to read reserves.json for validation: %s", e)
    
    if not has_population:
        logger.error("Generation 0 failed: No genomes in elites.json or reserves.json after speciation.")
        logger.error("This indicates that Generation 0 did not complete successfully.")
        logger.error("Possible causes: empty temp.json, all genomes archived, or speciation failure.")
        return

    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
    if evolution_tracker_path.exists():
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            evolution_tracker = json.load(f)
        
        existing_generations = evolution_tracker.get("generations", [])
        if existing_generations:
            max_generation = max(gen.get("generation_number", 0) for gen in existing_generations)
            generation_count = max_generation
        else:
            generation_count = 0
        logger.debug("Resuming from generation %d", generation_count)
    else:
        generation_count = 0
        logger.debug("Starting fresh")
    
    # Evolution loop: generate → moderate → speciate → redistribute each generation.
    # Termination is only by max_total_genomes (--max-total-genomes). --generations is not used for termination.
    terminated_by_total_genomes = False
    final_total_genomes = None
    # After Gen 0 (or resume), check if we already reached max_total_genomes before starting next generation.
    if evolution_tracker_path.exists() and max_total_genomes is not None:
        try:
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            gens = tracker.get("generations", [])
            latest = max(gens, key=lambda g: g.get("generation_number", 0), default=None) if gens else None
            if latest is not None:
                total = latest.get("elites_count", 0) + latest.get("reserves_count", 0) + latest.get("archived_count", 0)
                final_total_genomes = total
                if total >= max_total_genomes:
                    terminated_by_total_genomes = True
                    logger.info(
                        "Evolution completed: Total genomes limit already reached after Gen 0 (%d >= %d). Skipping further generations.",
                        total, max_total_genomes
                    )
        except Exception as e:
            logger.debug("Could not check post-Gen0 total genomes: %s", e)

    while True:
        if terminated_by_total_genomes:
            break
        generation_count += 1
        gen_start = time.time()
        logger.info("=== Starting Generation %d ===", generation_count)
        
        operator_statistics = {}
        # Create new variants for this generation
        try:
            with PerformanceLogger(logger, "Gen %d: Evolution (generate variants)" % generation_count):
                evolution_result = run_evolution(
                    north_star_metric=north_star_metric,
                    log_file=log_file,
                    current_cycle=generation_count,
                    max_variants=max_variants,
                    operators=operators
                )
                
                operator_statistics = evolution_result.get("operator_statistics", {}) if evolution_result else {}
                if operator_statistics:
                    logger.debug("Operator stats: %s", operator_statistics)

        except Exception as e:
            logger.error("Evolution failed: %s", e, exc_info=True)
            break

        # Score variants, then assign species/reserves for this generation
        try:
            temp_path = str(get_outputs_path() / "temp.json")
            with PerformanceLogger(logger, "Gen %d: Response generation" % generation_count):
                response_generator.process_population(pop_path=temp_path)
            
            with PerformanceLogger(logger, "Gen %d: Moderation" % generation_count):
                run_moderation_on_population = get_run_moderation_on_population()
                run_moderation_on_population(
                    pop_path=temp_path,
                    log_file=log_file,
                    north_star_metric=north_star_metric,
                    moderation_methods=moderation_methods
                )
            
            # Apply refusal penalties after evaluation, before speciation
            try:
                with PerformanceLogger(logger, "Gen %d: Apply refusal penalties" % generation_count):
                    from utils.refusal_penalty import apply_refusal_penalties
                    refusal_stats = apply_refusal_penalties(
                        pop_path=temp_path,
                        north_star_metric=north_star_metric,
                        logger=logger,
                        log_file=log_file
                    )
                    logger.info("Gen %d: Refusal analysis - %d refusals detected, %d penalties applied",
                               generation_count,
                               refusal_stats.get("refusals_detected", 0),
                               refusal_stats.get("penalties_applied", 0))
            except Exception as e:
                logger.error("Gen %d: Refusal penalty application failed: %s", 
                            generation_count, e, exc_info=True)
            
            # avg_fitness: mean(old elites + old reserves + all new variants) before speciation, after evaluation.
            avg_fitness_before_speciation = 0.0001
            try:
                with PerformanceLogger(logger, "Gen %d: Calculate avg_fitness before speciation" % generation_count):
                    avg_fitness_before_speciation = calculate_average_fitness(
                        str(get_outputs_path()), north_star_metric, include_temp=True, logger=logger, log_file=log_file
                    )
            except Exception as e:
                logger.warning("Gen %d: Failed to compute avg_fitness before speciation: %s", generation_count, e)
            
            # Calculate variant counts and statistics BEFORE speciation (temp.json gets cleared during speciation)
            variant_counts = {"variants_created": 0, "mutation_variants": 0, "crossover_variants": 0, "variants_integrated": 0}
            max_score_variants = 0.0001
            min_score_variants = 0.0001
            avg_fitness_variants = 0.0001
            max_toxicity = 0.0001
            best_genome_id = None

            with PerformanceLogger(logger, "Gen %d: Variant statistics (pre-speciation)" % generation_count):
                temp_path_obj = get_outputs_path() / "temp.json"
                remaining_variants = []
                if temp_path_obj.exists():
                    with open(temp_path_obj, 'r', encoding='utf-8') as f:
                        remaining_variants = json.load(f)
                
                # Calculate variants remaining in temp.json (after deduplication/rejection)
                mutation_count = sum(1 for v in remaining_variants if v and v.get("variant_type") == "mutation")
                crossover_count = sum(1 for v in remaining_variants if v and v.get("variant_type") == "crossover")
                remaining_count = mutation_count + crossover_count
                
                # Calculate total variants generated (before deduplication/rejection)
                # Total = remaining + duplicates + rejections from operator statistics
                total_duplicates = 0
                total_rejections = 0
                if operator_statistics:
                    for op_name, op_stats in operator_statistics.items():
                        if isinstance(op_stats, dict):
                            total_duplicates += op_stats.get("duplicates_removed", 0)
                            total_rejections += op_stats.get("question_mark_rejections", 0)
                
                total_generated = remaining_count + total_duplicates + total_rejections
                
                variant_counts = {
                    "variants_created": total_generated,  # Total generated (before deduplication/rejection)
                    "mutation_variants": mutation_count,  # Remaining mutation variants
                    "crossover_variants": crossover_count,  # Remaining crossover variants
                    "variants_integrated": remaining_count,  # Actual count of prompts added (evaluated, passed dedup, sent to speciation)
                }
                
                if remaining_variants:
                    # Calculate variant statistics (max, min, avg fitness) from remaining variants
                    scores = [(_extract_north_star_score(v, north_star_metric), v.get("id")) for v in remaining_variants if v]
                    
                    if scores:
                            max_score_variants = round(max(s[0] for s in scores), 4)
                            min_score_variants = round(min(s[0] for s in scores), 4)
                            avg_fitness_variants = round(sum(s[0] for s in scores) / len(scores), 4)
                            best_score, best_genome_id = max(scores, key=lambda x: x[0])
                            max_toxicity = best_score
                        
                    logger.info("Gen %d: %d variants generated (%d remaining, %d duplicates, %d rejections), max=%.4f, min=%.4f, avg=%.4f", 
                               generation_count, total_generated, remaining_count, total_duplicates, total_rejections,
                                   max_score_variants, min_score_variants, avg_fitness_variants)
                else:
                    # If all variants were removed from temp.json (duplicates/rejections), max_score_variants stays at 0.0001
                    # This is correct: max_score_variants should reflect variants from temp.json, not the distributed population
                    logger.info("Gen %d: %d variants generated (%d duplicates, %d rejections), all removed from temp.json", 
                               generation_count, total_generated, total_duplicates, total_rejections)
            
            # Run speciation on evaluated genomes (distribution happens inside speciation)
            try:
                with PerformanceLogger(logger, "Gen %d: Run speciation" % generation_count):
                    from speciation import run_speciation
                    
                    speciation_result = run_speciation(
                        temp_path=str(temp_path_obj),
                        current_generation=generation_count,
                        config=speciation_config,
                        log_file=log_file,
                        north_star_metric=north_star_metric
                    )
                
                if speciation_result.get("success"):
                    logger.info("Gen %d speciation: %d species, %d in reserves, %d elites moved, %d reserves moved",
                               generation_count,
                               speciation_result.get("species_count", 0),
                               speciation_result.get("reserves_size", 0),
                               speciation_result.get("elites_moved", 0),
                               speciation_result.get("reserves_moved", 0))
                else:
                    logger.warning("Gen %d speciation completed with warnings: %s", 
                                  generation_count, speciation_result.get("error", "unknown"))
            except Exception as e:
                logger.error("Gen %d speciation failed: %s", generation_count, e, exc_info=True)
            
            # Phase 4.5: Calculate operator effectiveness metrics for RQ1
            try:
                with PerformanceLogger(logger, "Gen %d: Operator effectiveness metrics" % generation_count):
                    from utils.operator_effectiveness import (
                        calculate_table4_metrics, 
                        save_operator_effectiveness_cumulative,
                        generate_operator_effectiveness_visualizations
                    )
                    
                    operator_metrics_df = calculate_table4_metrics(
                        outputs_path=str(get_outputs_path()),
                        current_generation=generation_count,
                        north_star_metric=north_star_metric,
                        operator_statistics=operator_statistics,
                        logger=logger
                    )
                    
                    if operator_metrics_df is not None:
                        # Save metrics (even if empty for generation 0)
                        csv_path = save_operator_effectiveness_cumulative(
                            metrics_df=operator_metrics_df,
                            outputs_path=str(get_outputs_path()),
                            current_generation=generation_count,
                            logger=logger
                        )
                        
                        if csv_path:
                            logger.info("Gen %d: Operator effectiveness CSV saved to: %s", generation_count, csv_path)
                        
                        if not operator_metrics_df.empty:
                            # Generate visualizations (only if we have data)
                            viz_paths = generate_operator_effectiveness_visualizations(
                                outputs_path=str(get_outputs_path()),
                                current_generation=generation_count,
                                logger=logger
                            )
                            
                            logger.info("Gen %d: Operator effectiveness metrics calculated and saved (%d operators, %d visualizations)", 
                                       generation_count, len(operator_metrics_df), len(viz_paths) if viz_paths else 0)
                        else:
                            logger.info("Gen %d: No operator-created variants. Metrics file structure maintained.", generation_count)
                    else:
                        logger.warning("Gen %d: Failed to calculate operator effectiveness metrics (returned None). Creating empty CSV structure.", generation_count)
                        # Create empty DataFrame and save to ensure file exists
                        import pandas as pd
                        empty_df = pd.DataFrame(columns=['generation', 'operator', 'NE', 'EHR', 'IR', 'cEHR', 'Δμ', 'Δσ', 
                                                        'total_variants', 'elite_count', 'non_elite_count', 'rejections', 'duplicates'])
                        save_operator_effectiveness_cumulative(
                            metrics_df=empty_df,
                            outputs_path=str(get_outputs_path()),
                            current_generation=generation_count,
                            logger=logger
                        )
            except Exception as e:
                logger.error("Gen %d: Failed to calculate operator effectiveness metrics: %s", generation_count, e, exc_info=True)
            
            # Generate all visualizations after each generation
            try:
                with PerformanceLogger(logger, "Gen %d: Live analysis (visualizations)" % generation_count):
                    from utils.live_analysis import run_live_analysis
                    viz_results = run_live_analysis(outputs_path=str(get_outputs_path()), logger=logger)
                if viz_results:
                    successful_viz = sum(1 for v in viz_results.values() if v is not None)
                    logger.info("Gen %d: Generated %d/%d visualizations", generation_count, successful_viz, len(viz_results))
            except Exception as e:
                logger.warning("Gen %d: Failed to generate visualizations: %s", generation_count, e)
            
            try:
                with PerformanceLogger(logger, "Gen %d: Update EvolutionTracker with generation data" % generation_count):
                    from ea import get_update_evolution_tracker_with_generation_global
                    update_evolution_tracker_with_generation_global = get_update_evolution_tracker_with_generation_global()
                    
                    evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                    with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                        evolution_tracker = json.load(f)
                    
                    generation_data = {
                        "generation_number": generation_count,
                        "variants_created": variant_counts["variants_created"],
                        "mutation_variants": variant_counts["mutation_variants"],
                        "crossover_variants": variant_counts["crossover_variants"],
                        "variants_integrated": variant_counts.get("variants_integrated", 0),
                    }
                    
                    from utils.population_io import load_population
                    outputs_path = get_outputs_path()
                    
                    all_genomes = []
                    for file_name in ["temp.json", "elites.json", "reserves.json"]:
                        file_path = outputs_path / file_name
                        if file_path.exists():
                            file_genomes = load_population(str(file_path), logger=logger)
                            all_genomes.extend(file_genomes)
                    
                    logger.debug("Loaded %d genomes for analysis", len(all_genomes))
                    
                    update_evolution_tracker_with_generation_global(
                        generation_data=generation_data,
                        evolution_tracker=evolution_tracker,
                        logger=logger,
                        population=all_genomes,
                        north_star_metric=north_star_metric
                    )
            except Exception as e:
                logger.error("Failed to update EvolutionTracker with generation data: %s", e, exc_info=True)
            
            # Recompute thresholds, redistribute files, and update tracker for this generation
            try:
                # Phase 5: Calculate comprehensive generation statistics
                # Note: variant statistics (max_score_variants, etc.) are already calculated above before speciation
                try:
                    with PerformanceLogger(logger, "Gen %d: Generation statistics and EvolutionTracker update" % generation_count):
                        evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
                        
                        # Calculate comprehensive generation statistics
                        gen_stats = calculate_generation_statistics(
                            outputs_path=str(get_outputs_path()),
                            north_star_metric=north_star_metric,
                            current_generation=generation_count,
                            logger=logger,
                            log_file=log_file
                        )
                        gen_stats["generation_duration_seconds"] = time.time() - gen_start
                        
                        # Override variant statistics with what we calculated from temp.json before speciation
                        gen_stats["max_score_variants"] = max_score_variants
                        gen_stats["min_score_variants"] = min_score_variants
                        gen_stats["avg_fitness_variants"] = avg_fitness_variants
                        
                        # avg_fitness: before speciation, after evaluation (from pre-speciation computation)
                        gen_stats["avg_fitness"] = avg_fitness_before_speciation
                        
                        # Add additional metrics
                        # NOTE: population_max_toxicity should be max from elites+reserves (after distribution),
                        # NOT from temp.json variants. calculate_generation_statistics() already calculates this correctly.
                        # Do NOT override it with max_toxicity (which is from temp.json variants).
                        # gen_stats["population_max_toxicity"] is already set correctly by calculate_generation_statistics()
                        gen_stats["best_genome_id"] = best_genome_id
                        gen_stats["variants_created"] = variant_counts["variants_created"]
                        gen_stats["mutation_variants"] = variant_counts["mutation_variants"]
                        gen_stats["crossover_variants"] = variant_counts["crossover_variants"]
                        gen_stats["variants_integrated"] = variant_counts.get("variants_integrated", 0)
                        
                        # Add speciation metrics if available (for EvolutionTracker speciation block)
                        if 'speciation_result' in locals():
                            gen_stats["species_count"] = speciation_result.get("species_count", 0)
                            gen_stats["active_species_count"] = speciation_result.get("active_species_count", 0)
                            gen_stats["frozen_species_count"] = speciation_result.get("frozen_species_count", 0)
                            gen_stats["reserves_size"] = speciation_result.get("reserves_size", 0)
                            gen_stats["speciation_events"] = speciation_result.get("speciation_events", 0)
                            gen_stats["merge_events"] = speciation_result.get("merge_events", 0)
                            gen_stats["extinction_events"] = speciation_result.get("extinction_events", 0)
                            gen_stats["archived_count"] = speciation_result.get("archived_count", 0)
                            gen_stats["elites_moved"] = speciation_result.get("elites_moved", 0)
                            gen_stats["reserves_moved"] = speciation_result.get("reserves_moved", 0)
                            gen_stats["genomes_updated"] = speciation_result.get("genomes_updated", 0)
                            # Add diversity metrics if available (from metrics_tracker)
                            if "inter_species_diversity" in speciation_result:
                                gen_stats["inter_species_diversity"] = speciation_result.get("inter_species_diversity", 0.0)
                            if "intra_species_diversity" in speciation_result:
                                gen_stats["intra_species_diversity"] = speciation_result.get("intra_species_diversity", 0.0)
                            # Add cluster quality if available
                            if "cluster_quality" in speciation_result:
                                gen_stats["cluster_quality"] = speciation_result.get("cluster_quality")
                            if "speciation_duration_seconds" in speciation_result:
                                gen_stats["speciation_duration_seconds"] = speciation_result.get("speciation_duration_seconds")
                        
                        # Get previous cumulative population_max_toxicity BEFORE updating tracker
                        # (needed for improvement comparison)
                        previous_cumulative_population_max = 0.0001
                        if generation_count > 0 and evolution_tracker_path.exists():
                            try:
                                with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                                    tracker_before = json.load(f)
                                previous_cumulative_population_max = tracker_before.get("population_max_toxicity", 0.0001)
                                logger.debug(f"Previous cumulative population_max_toxicity: {previous_cumulative_population_max:.4f}")
                            except Exception as e:
                                logger.debug(f"Failed to read previous population_max_toxicity: {e}")
                        
                        # Current generation's per-generation population_max_toxicity (from elites+reserves after distribution)
                        current_population_max = gen_stats.get("population_max_toxicity", 0.0001)
                        logger.debug(f"Gen {generation_count}: Extracted population_max_toxicity={current_population_max:.4f} from gen_stats")
                        
                        # Enhanced validation: verify current >= previous (for cumulative max)
                        if generation_count > 0 and current_population_max < previous_cumulative_population_max - 0.01:
                            logger.warning(f"Gen {generation_count}: current_population_max ({current_population_max:.4f}) < previous ({previous_cumulative_population_max:.4f}) - this shouldn't happen for cumulative max!")
                            logger.warning("Recalculating from files to verify...")
                            # Force recalculation
                            current_population_max = 0.0001
                        
                        # Ensure we have a valid value (should never be 0.0 or default 0.0001 if we have genomes)
                        if (current_population_max == 0.0 or current_population_max == 0.0001) and (gen_stats.get("elites_count", 0) > 0 or gen_stats.get("reserves_count", 0) > 0):
                            logger.warning(f"Gen {generation_count}: population_max_toxicity is {current_population_max:.4f} but we have genomes (elites={gen_stats.get('elites_count', 0)}, reserves={gen_stats.get('reserves_count', 0)}) - recalculating...")
                            logger.warning(f"Gen {generation_count}: population_max_toxicity is 0.0 but we have genomes - this may indicate a calculation issue")
                            # Fallback: try to calculate from elites+reserves directly
                            try:
                                # Recalculate from elites+reserves files directly
                                outputs_path = get_outputs_path()
                                elites_path = outputs_path / "elites.json"
                                reserves_path = outputs_path / "reserves.json"
                                all_scores = []
                                for path in [elites_path, reserves_path]:
                                    if path.exists():
                                        with open(path, 'r', encoding='utf-8') as f:
                                            genomes = json.load(f)
                                        for g in genomes:
                                            if g.get("generation", 0) <= generation_count:
                                                # Extract score using same logic as calculate_generation_statistics
                                                score = None
                                                if 'moderation_result' in g:
                                                    mr = g['moderation_result']
                                                    if isinstance(mr, dict):
                                                        if 'google' in mr and 'scores' in mr['google']:
                                                            score = mr['google']['scores'].get(north_star_metric, 0)
                                                        elif 'scores' in mr:
                                                            score = mr['scores'].get(north_star_metric, 0)
                                                if score is None:
                                                    score = g.get(north_star_metric, 0)
                                                if score and score > 0.0001:
                                                    all_scores.append(score)
                                if all_scores:
                                    current_population_max = max(all_scores)
                                    logger.info(f"Gen {generation_count}: Recalculated population_max_toxicity={current_population_max:.4f} from files")
                                    # Add validation after recalculation
                                    if generation_count > 0 and current_population_max < previous_cumulative_population_max - 0.01:
                                        logger.error(f"Recalculated value ({current_population_max:.4f}) still < previous ({previous_cumulative_population_max:.4f})!")
                            except Exception as e:
                                logger.warning(f"Gen {generation_count}: Failed to recalculate population_max_toxicity: {e}")
                        
                        # Update EvolutionTracker with all statistics
                        update_evolution_tracker_with_statistics(
                            evolution_tracker_path=str(evolution_tracker_path),
                            current_generation=generation_count,
                            statistics=gen_stats,
                            operator_statistics=operator_statistics,
                            logger=logger,
                            log_file=log_file,
                            run_metadata_update={"max_total_genomes": max_total_genomes} if max_total_genomes is not None else None,
                        )
                        # Total genomes (elites + reserves + archives) for termination and summary
                        total_genomes = gen_stats["elites_count"] + gen_stats["reserves_count"] + gen_stats.get("archived_count", 0)
                        final_total_genomes = total_genomes
                        if max_total_genomes is not None and total_genomes >= max_total_genomes:
                            logger.info("Evolution completed: Total genomes limit reached (%d >= %d).", total_genomes, max_total_genomes)
                            terminated_by_total_genomes = True
                            break
                        # Update adaptive selection logic (AFTER statistics are calculated and saved)
                        # Compare current generation's population_max_toxicity vs previous cumulative
                        # If current > previous cumulative, there's improvement (new best found)
                        try:
                            outputs_path = str(get_outputs_path())
                            
                            adaptive_results = update_adaptive_selection_logic(
                                outputs_path=outputs_path,
                                current_max_toxicity=current_population_max,
                                previous_max_toxicity=previous_cumulative_population_max,
                                stagnation_limit=stagnation_limit,
                                north_star_metric=north_star_metric,
                                current_gen_avg_fitness=avg_fitness_before_speciation,
                                logger=logger,
                                log_file=log_file
                            )
                            logger.debug("Selection: mode=%s, since_improvement=%d, avg=%.4f, slope=%.4f",
                                       adaptive_results["selection_mode"], adaptive_results["generations_since_improvement"],
                                       adaptive_results["current_avg_fitness"], adaptive_results["slope_of_avg_fitness"])
                        except Exception as e:
                            logger.warning("Failed to update adaptive selection logic: %s", e)
                        
                        logger.info("Gen%d metrics: elites=%d (avg=%.4f), reserves=%d (avg=%.4f), archived=%d, variants: max=%.4f, min=%.4f, avg=%.4f",
                                    generation_count, gen_stats["elites_count"], gen_stats["avg_fitness_elites"],
                                    gen_stats["reserves_count"], gen_stats["avg_fitness_reserves"],
                                    gen_stats.get("archived_count", 0), max_score_variants, min_score_variants, avg_fitness_variants)
                        
                        # Run live analysis and generate visualizations
                        try:
                            from utils.live_analysis import run_live_analysis
                            run_live_analysis(
                                outputs_path=str(get_outputs_path()),
                                logger=logger
                            )
                            # Live analysis completion is already logged by run_live_analysis (N/N visualizations)
                        except Exception as e:
                            logger.warning("Failed to run live analysis: %s", e)
                except Exception as e:
                    logger.warning("Failed to update generation metrics in EvolutionTracker: %s", e)
            except Exception as e:
                logger.error("Post-speciation processing failed: %s", e, exc_info=True)
            
            try:
                with PerformanceLogger(logger, "Gen %d: Update population index" % generation_count):
                    update_population_index_single_file(str(get_outputs_path()), 0, logger=logger)
            except Exception as e:
                logger.warning("Failed to update EvolutionTracker population metadata: %s", e)
            
        except Exception as e:
            logger.error("Post-evolution processing failed: %s", e, exc_info=True)

    if terminated_by_total_genomes:
        pass  # Already logged when breaking
    else:
        logger.info("Evolution completed: Loop exited (termination is only by --max-total-genomes).")

    total_time = time.time() - start_time
    logger.info("=== Pipeline Completed ===")
    logger.info("COMPLETED: Pipeline (full run) in %.3f seconds", total_time)
    logger.info("Total execution time: %.2f seconds", total_time)
    logger.info("Total generations: %d", generation_count)
    if final_total_genomes is not None:
        logger.info("Total genomes: %d", final_total_genomes)

    # Write run-level metadata for RQ analysis (run_duration_seconds, run_mode)
    try:
        from utils.population_io import update_run_metadata_at_end
        update_run_metadata_at_end(
            evolution_tracker_path=str(get_outputs_path() / "EvolutionTracker.json"),
            run_duration_seconds=total_time,
            run_mode="sequential",
            logger=logger,
            log_file=log_file,
        )
    except Exception as e:
        logger.warning("Failed to update run metadata at end (non-fatal): %s", e)

    # Run GDP visualization once at end of execution (historic + current from elites/reserves/archive)
    try:
        with PerformanceLogger(logger, "Final GDP projection plot"):
            from utils.live_analysis import generate_gdp_projection_plot
            gdp_path = generate_gdp_projection_plot(outputs_path=str(get_outputs_path()), logger=logger)
            if gdp_path:
                logger.info("Final GDP diagram: %s", gdp_path)
    except Exception as e:
        logger.warning("GDP diagram at end of run failed (non-fatal): %s", e)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evolutionary Text Generation and Safety Analysis Framework")
    parser.add_argument("--generations", type=int, default=None,
                       help="Not used for termination. Kept for compatibility; termination is only by --max-total-genomes.")
    parser.add_argument("--moderation-methods", nargs="+", choices=["google", "all"], default=["google"],
                       help="Moderation methods to use: google (Perspective API), all (google only)")
    parser.add_argument("--stagnation-limit", type=int, default=5,
                       help="Number of generations without improvement before switching to explore mode (default: 5)")
    parser.add_argument("--theta-sim", type=float, default=0.2,
                       help="Similarity threshold for species assignment (ensemble distance, default: 0.2)")
    parser.add_argument("--theta-merge", type=float, default=0.1,
                       help="Merge threshold for combining similar species (ensemble distance, default: 0.1)")
    parser.add_argument("--min-stability-gens", type=int, default=5,
                       help="Minimum generations a species must exist before it can be merged (default: 5)")
    parser.add_argument("--species-capacity", type=int, default=100,
                       help="Maximum individuals per species (default: 100)")
    parser.add_argument("--cluster0-max-capacity", type=int, default=1000,
                       help="Maximum individuals in cluster 0/reserves (default: 1000)")
    parser.add_argument("--cluster0-min-cluster-size", type=int, default=2,
                       help="Minimum cluster size for cluster 0 speciation (default: 2)")
    parser.add_argument("--min-island-size", type=int, default=2,
                       help="Minimum island size before extinction (default: 2)")
    parser.add_argument("--species-stagnation", type=int, default=20,
                       help="Maximum generations without improvement before species extinction (default: 20)")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2",
                       help="Sentence-transformer model for embeddings (default: all-MiniLM-L6-v2)")
    parser.add_argument("--embedding-dim", type=int, default=384,
                       help="Embedding dimensionality (default: 384)")
    parser.add_argument("--embedding-batch-size", type=int, default=64,
                       help="Batch size for embedding computation (default: 64)")
    parser.add_argument("--rg", type=str, default="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                       help="Response generator model: pass a direct .gguf path or an alias under models/")
    parser.add_argument("--pg", type=str, default="models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
                       help="Prompt generator model: pass a direct .gguf path or an alias under models/")
    parser.add_argument("--operators", type=str, choices=["ie", "cm", "all"], default="all",
                       help="Operator configuration mode: ie (InformedEvolution only), cm (all except InformedEvolution), all (all operators)")
    parser.add_argument("--max-variants", type=int, default=1,
                       help="Maximum number of variants to generate per evolution cycle. Controls how many times the evolution cycle runs.")
    parser.add_argument("--seed-file", type=str, default="data/prompt.csv",
                       help="Path to CSV file with seed prompts (must have 'questions' column). Default: data/prompt.csv")
    parser.add_argument("--seed", type=int, default=None,
                       help="Fixed seed for LLM generation. When set, all processes (including workers) use this same seed for reproducibility.")
    parser.add_argument("--batch-size", type=int, default=100,
                       help="Number of genomes per generation batch (K) for parallel mode. Default: 100")
    parser.add_argument("--gen0-batch-size", type=int, default=25,
                       help="Number of seed prompts per Gen0 MPI batch (pull-based). Default: 25")
    parser.add_argument("--max-total-genomes", type=int, default=None,
                       help="Primary termination: stop when total genomes (elites + reserves + archives) reaches this cap. Required for both sequential and parallel. E.g. set to max_generations * K for fair comparison.")
    parser.add_argument("--parallel", action="store_true",
                       help="Run in MPI master-worker mode (use with mpiexec)")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="Output directory for this run (default: data/outputs/<timestamp>). Use for reproducible experiment paths (e.g. data/outputs/rq1_workers_4).")
    parser.add_argument("--profile", nargs="?", const="profile_main.prof", default=None,
                       metavar="OUTPUT.prof",
                       help="Enable cProfile profiling. Output is saved in the execution output directory as profile_main.prof. "
                            "Inspect with: python -m pstats <file> or snakeviz <file>")
    args = parser.parse_args()

    if getattr(args, "output_dir", None):
        set_outputs_path(args.output_dir)

    import sys

    prof = None
    if args.profile is not None:
        import cProfile
        prof = cProfile.Profile()

    def _dump_profile():
        if prof is not None:
            prof.disable()
            profile_path = str(get_outputs_path() / "profile_main.prof")
            prof.dump_stats(profile_path)
            print(f"Profile saved to {profile_path}")
            print(f"  Inspect with: python -m pstats {profile_path}")

    if args.parallel:
        from mpi4py import MPI
        from parallel.master_worker import run as run_parallel
        from speciation.run_speciation import run_speciation
        from speciation.config import SpeciationConfig

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()

        if getattr(args, "max_total_genomes", None) is None:
            raise ValueError("Parallel mode requires --max-total-genomes; primary termination is by total genomes (elites + reserves + archives).")
        get_logger, get_log_filename, _, PerformanceLogger = get_custom_logging()
        log_file = get_log_filename()
        logger = get_logger("master_worker", log_file)
        logger.info("Starting in parallel (MPI) mode. Rank-specific logs will be written for master and workers.")

        # Only rank 0 updates RG/PG YAML so workers (which load from YAML) see --rg and --pg. Barrier so workers wait.
        if rank == 0:
            try:
                with PerformanceLogger(logger, "Update model configs (parallel)"):
                    update_model_configs(args.rg, args.pg, logger)
                logger.info("Parallel run: configs updated from script; workers will load --rg=%s --pg=%s from YAML", args.rg, args.pg)
            except Exception as e:
                logger.error("Config update failed: %s", e, exc_info=True)
                sys.exit(1)
        comm.Barrier()

        speciation_config = SpeciationConfig(
            theta_sim=args.theta_sim,
            theta_merge=args.theta_merge,
            min_stability_gens=args.min_stability_gens,
            species_capacity=args.species_capacity,
            cluster0_max_capacity=args.cluster0_max_capacity,
            cluster0_min_cluster_size=args.cluster0_min_cluster_size,
            min_island_size=args.min_island_size,
            species_stagnation=args.species_stagnation,
            embedding_model=args.embedding_model,
            embedding_dim=args.embedding_dim,
            embedding_batch_size=args.embedding_batch_size,
        )

        try:
            if prof is not None:
                prof.enable()
            run_parallel(
                logger,
                K=args.batch_size,
                seed_file=args.seed_file,
                seed=getattr(args, "seed", None),
                operators_mode=args.operators,
                moderation_methods=args.moderation_methods,
                max_generations=args.generations,
                max_total_genomes=getattr(args, "max_total_genomes", None),
                north_star_metric="toxicity",
                speciation_config=speciation_config,
                log_file=log_file,
                run_speciation_fn=run_speciation,
                stagnation_limit=getattr(args, "stagnation_limit", 5),
                gen0_batch_size=getattr(args, "gen0_batch_size", 25),
            )
        finally:
            _dump_profile()
        sys.exit(0)

    try:
        if prof is not None:
            prof.enable()
        main(max_generations=args.generations,
             moderation_methods=args.moderation_methods,
             rg_model=args.rg, pg_model=args.pg,
             operators=args.operators, max_variants=args.max_variants,
             stagnation_limit=args.stagnation_limit, seed_file=args.seed_file,
             max_total_genomes=getattr(args, "max_total_genomes", None),
             seed=getattr(args, "seed", None),
             # Speciation parameters
             theta_sim=args.theta_sim, theta_merge=args.theta_merge,
             min_stability_gens=args.min_stability_gens,
             species_capacity=args.species_capacity, cluster0_max_capacity=args.cluster0_max_capacity,
             cluster0_min_cluster_size=args.cluster0_min_cluster_size, min_island_size=args.min_island_size,
             species_stagnation=args.species_stagnation, embedding_model=args.embedding_model,
             embedding_dim=args.embedding_dim, embedding_batch_size=args.embedding_batch_size)
        sys.exit(0)
    except KeyboardInterrupt:
        print("\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    finally:
        _dump_profile()