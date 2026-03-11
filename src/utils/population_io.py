"""
Comprehensive Population I/O Utility

Unified population loading, saving, and management (elites, reserves, temp, archive).
Handles monolithic and split file formats. Species_id in file genomes is updated only
during speciation Phase 7 (redistribution); this module reads/writes genomes as-is.
"""

from typing import List, Dict, Any, Optional, Union
import os
import json
from pathlib import Path
from collections import defaultdict
import time
from utils import get_custom_logging
from utils.constants import EvolutionConstants, FileConstants
from utils import get_population_io
from gne import get_ResponseGenerator, get_PromptGenerator
from datetime import datetime

import pandas as pd

get_logger, _, _, PerformanceLogger = get_custom_logging()



def get_project_root():
    """Get the absolute path to the project root directory"""
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    return project_root.resolve()

def get_config_path():
    """Get the absolute path to the config directory"""
    return get_project_root() / "config"

def get_data_path():
    """Get the absolute path to the data directory"""
    return get_project_root() / "data" / "prompt.csv"

# Global variable to store the outputs path for the current run
_current_outputs_path = None

def _max_genome_id_from_iter(genomes) -> int:
    """Return max genome id from an iterable of genome dicts; 0 if none valid."""
    out = 0
    for g in genomes:
        if not isinstance(g, dict):
            continue
        kid = g.get("id")
        if kid is not None and isinstance(kid, (int, float)):
            out = max(out, int(kid))
    return out


def get_max_genome_id_from_all_files(outputs_path: Optional[Union[str, Path]] = None) -> int:
    """
    Find the maximum genome ID across all genome files (elites.json, reserves.json, archive.json).
    So new genomes get unique IDs that don't conflict with existing genomes in alive or dead populations.
    """
    outputs_path = get_outputs_path() if outputs_path is None else Path(outputs_path)
    log = get_logger("GetMaxGenomeID")
    max_id = 0
    for fname in ("elites.json", "reserves.json", "archive.json"):
        path = outputs_path / fname
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                max_id = max(max_id, _max_genome_id_from_iter(data))
            elif isinstance(data, dict) and data:
                max_id = max(max_id, _max_genome_id_from_iter(data.values()))
        except Exception as e:
            log.warning("Failed to read %s for max ID: %s", fname, e)
    return max_id


def set_outputs_path(path: Union[str, Path]) -> Path:
    """Set the outputs directory for this run (e.g. data/outputs/run01_comb). Call before get_outputs_path()."""
    global _current_outputs_path
    outputs_dir = Path(path).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _current_outputs_path = outputs_dir
    return outputs_dir


def get_outputs_path():
    """Get the absolute path to the outputs directory"""
    global _current_outputs_path
    
    if _current_outputs_path is not None:
        return _current_outputs_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    outputs_dir = get_project_root() / "data" / "outputs" / timestamp
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    _current_outputs_path = outputs_dir
    
    return outputs_dir

def _extract_north_star_score(genome, metric="toxicity"):
    """Extract north star score from genome. Priority: moderation_result.google.scores, .scores, genome[metric], genome.scores. Returns min 0.0001."""
    if not genome:
        return 0.0001

    def _valid(s):
        return s is not None and float(s) > 0

    mr = genome.get("moderation_result")
    if mr and isinstance(mr, dict):
        if "google" in mr and mr["google"] and "scores" in mr["google"]:
            s = mr["google"]["scores"].get(metric, 0.0001)
            if _valid(s):
                return round(float(s), 4)
        if "scores" in mr:
            s = mr["scores"].get(metric, 0.0001)
            if _valid(s):
                return round(float(s), 4)
    if metric in genome and _valid(genome.get(metric)):
        return round(float(genome[metric]), 4)
    if isinstance(genome.get("scores"), dict):
        s = genome["scores"].get(metric, 0.0001)
        if _valid(s):
            return round(float(s), 4)
    return 0.0001



def initialize_system(logger, log_file, seed_file="data/prompt.csv", seed=None):
    """Initialize the system components and create gen0 if needed
    
    Args:
        logger: Logger instance
        log_file: Log file path
        seed_file: Path to CSV file with seed prompts (must have 'questions' column).
                   Default: data/prompt.csv
        seed: Optional fixed seed for LLM generation (used by all RG/PG for reproducibility).
    """
    from utils.device_utils import device_manager
    device = device_manager.get_optimal_device()
    
    logger.debug("Initializing pipeline for device: %s", device)
    
    population_io_functions = get_population_io()
    
    load_and_initialize_population, get_population_files_info, load_population, save_population, sort_population_json, load_genome_by_id, consolidate_generations_to_single_file, migrate_from_split_to_single, sort_population_by_elite_criteria, load_elites, save_elites, get_population_stats_steady_state = get_population_io()
    
    ResponseGenerator = get_ResponseGenerator()
    response_generator = ResponseGenerator(model_key="response_generator", config_path="config/RGConfig.yaml", log_file=log_file, seed=seed)
    logger.debug("Response generator initialized")
    
    PromptGenerator = get_PromptGenerator()
    prompt_generator = PromptGenerator(model_key="prompt_generator", config_path="config/PGConfig.yaml", log_file=log_file, seed=seed)
    logger.debug("Prompt generator initialized")
    
    from ea.evolution_engine import set_global_generators
    set_global_generators(response_generator, prompt_generator)
    logger.debug("Global generators set")
    
    population_file = get_outputs_path() / "elites.json"
    
    population_content = None
    if not population_file.exists():
        should_initialize = True
    else:
        try:
            with open(population_file, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            should_initialize = len(content) == 0 or content == '[]'
            if not should_initialize:
                import json
                population_content = json.loads(content)
        except Exception:
            should_initialize = True

    if should_initialize:
        try:
            seed_path = Path(seed_file)
            if not seed_path.is_absolute():
                seed_path = get_project_root() / seed_path
            input_path = str(seed_path)
            logger.info("Initializing population from seed file: %s", input_path)
            
            load_and_initialize_population(
                input_path=input_path,
                output_path=str(get_outputs_path()),
                log_file=log_file
            )
            logger.debug("Population successfully initialized and saved.")
        except Exception as e:
            logger.error("Failed to initialize population: %s", e, exc_info=True)
            raise
    else:
        logger.info("Existing elites file found. Skipping initialization.")
        try:
            population = population_content if population_content is not None else []
            logger.info("Loaded %d genomes from existing elites.json", len(population))
            generations = set(g.get("generation", 0) for g in population if g)
            logger.debug("Available generations: %s", sorted(generations))
        except Exception as e:
            logger.warning("Could not read existing population info: %s", e)
    
    return response_generator, prompt_generator


def clean_population(population: List[Dict[str, Any]], *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Remove None genomes and invalid entries from population.
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to clean.
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created.
    log_file : str | None
        Optional log-file path when a new logger is created.
        
    Returns
    -------
    List[Dict[str, Any]]
        Cleaned population with None genomes removed.
    """
    _logger = logger or get_logger("population_io", log_file)
    cleaned = [g for g in population if g is not None]
    n_removed = len(population) - len(cleaned)
    if n_removed:
        _logger.warning("Removed %d None genomes from population", n_removed)
    _logger.debug("Population cleaned: %d → %d genomes", len(population), len(cleaned))
    return cleaned



def get_population_files_info(base_dir: str = "outputs") -> Dict[str, Any]:
    """Get information about population files including elites.json and reserves.json
    
    Note: Active population = elites.json + reserves.json (archive.json is excluded).
    """
    _log = get_logger("population_io")
    base_path = Path(base_dir).resolve()
    # Use reserves.json as default population file
    # Note: We only maintain elites (in species or reserves). Active population = elites + reserves.
    population_file = base_path / "reserves.json"
    elites_file = base_path / "elites.json"
    evolution_tracker_file = base_path / "EvolutionTracker.json"
    
    info = {
        "total_generations": 0,
        "generation_counts": {},
    }
    
    # Try to get metadata from EvolutionTracker.json first
    if evolution_tracker_file.exists():
        try:
            with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Calculate total_generations from the actual generations array
            # So it's always up-to-date with the actual generation count
            if "generations" in tracker and tracker["generations"]:
                # Get the maximum generation number from the generations array
                max_gen_num = max(gen.get("generation_number", 0) for gen in tracker["generations"])
                info["total_generations"] = max_gen_num + 1  # +1 because generation 0 counts as 1 generation
            else:
                # Fallback: use tracker value or 0 if no generations exist
                info["total_generations"] = tracker.get("total_generations", 0)
            
            return info
                
        except Exception as e:
            _log.debug("get_population_files_info: EvolutionTracker read failed (%s), falling back to file scanning", e)
    
    def count_generations(genomes):
        for g in genomes or []:
            if g and "generation" in g:
                k = str(g["generation"])
                info["generation_counts"][k] = info["generation_counts"].get(k, 0) + 1

    for path in (population_file, elites_file):
        if not path.exists():
            continue
        try:
            with open(path, 'r', encoding='utf-8') as f:
                count_generations(json.load(f))
        except Exception as e:
            _log.debug("get_population_files_info: could not read %s (%s)", path.name, e)

    if info["generation_counts"]:
        max_gen = max(int(k) for k in info["generation_counts"])
        info["generation_counts"] = {str(g): info["generation_counts"].get(str(g), 0) for g in range(max_gen + 1)}
        info["total_generations"] = max_gen + 1
    return info


def update_population_index_single_file(base_dir: str, total_genomes: int, *, logger=None, log_file: Optional[str] = None):
    """Update the population metadata in EvolutionTracker.json for single file mode"""
    
    _logger = logger or get_logger("update_population_index", log_file)
    
    try:
        base_dir = str(Path(base_dir).resolve())
        info = get_population_files_info(base_dir)
        evolution_tracker_file = Path(base_dir) / "EvolutionTracker.json"
        
        default_tracker = {
            "status": "not_complete",
            "total_generations": 1,
            "generations_since_improvement": 0,
            "avg_fitness_history": [],
            "slope_of_avg_fitness": 0.0,
            "selection_mode": "default",
            "generations": [],
        }
        if evolution_tracker_file.exists():
            try:
                with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                    tracker = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                tracker = dict(default_tracker)
        else:
            tracker = dict(default_tracker)
        tracker["total_generations"] = info["total_generations"]
        
        with open(evolution_tracker_file, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
        
        _logger.debug("Updated EvolutionTracker population metadata: single file mode, "
                     "total_generations: %d", info['total_generations'])
        
    except Exception as e:
        _logger.warning("Failed to update EvolutionTracker population metadata: %s", e)


def load_population_generation(generation: int, base_dir: str = "outputs", 
                              *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load genomes for a specific generation from the active single-file population."""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generation {generation} from Single File"):
        try:
            # Load entire population
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            # Filter by generation
            generation_genomes = [g for g in all_genomes if g and g.get("generation") == generation]
            
            _logger.info("Loaded generation %d: %d genomes from active population file", generation, len(generation_genomes))
            return generation_genomes
            
        except Exception as e:
            _logger.error("Failed to load generation %d: %s", generation, e, exc_info=True)
            return []


def load_population_range(start_gen: int, end_gen: int, base_dir: str = "outputs",
                         *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """Load multiple generations from the active single-file population."""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generations {start_gen}-{end_gen} from Single File"):
        try:
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            range_genomes = [g for g in all_genomes if g and start_gen <= g.get("generation", 0) <= end_gen]
            
            _logger.info("Loaded generations %d-%d: %d genomes from active population file", start_gen, end_gen, len(range_genomes))
            return range_genomes
            
        except Exception as e:
            _logger.error("Failed to load generation range: %s", e, exc_info=True)
            return []


def load_population_lazy(base_dir: str = "outputs", max_gens: Optional[int] = None,
                        *, logger=None, log_file: Optional[str] = None):
    """Generator that yields genomes from the active single-file population."""
    
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load entire population
        all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
        
        # Apply generation limit if specified
        if max_gens is not None:
            all_genomes = [g for g in all_genomes if g and g.get("generation", 0) < max_gens]
        
        _logger.info("Lazy loading %d genomes from active population file", len(all_genomes))
        
        for genome in all_genomes:
            yield genome
            
    except Exception as e:
        _logger.error("Failed to load population lazily: %s", e, exc_info=True)
        return


def save_population_generation(genomes: List[Dict[str, Any]], generation: int, 
                              base_dir: str = "outputs", *, logger=None, log_file: Optional[str] = None):
    """Save genomes to the active single-file population (generation overwrite semantics)."""
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Save Generation {generation} to Single File"):
        try:
            existing_population = load_population(base_dir, logger=_logger, log_file=log_file)
            
            filtered_population = [g for g in existing_population if g and g.get("generation") != generation]
            
            filtered_population.extend(genomes)
            
            save_population(filtered_population, base_dir, logger=_logger, log_file=log_file)
            
            _logger.info("Updated active population file with generation %d: %d genomes", generation, len(genomes))
            
        except Exception as e:
            _logger.error("Failed to save generation %d: %s", generation, e, exc_info=True)
            raise


def get_latest_generation(base_dir: str = "outputs") -> int:
    """Get the highest generation number available from the active single-file population."""
    info = get_population_files_info(base_dir)
    return info["total_generations"] - 1 if info["total_generations"] > 0 else 0


def load_population(pop_path: str = "data/outputs/reserves.json", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load population with automatic detection of split vs monolithic format
    
    If pop_path points to a JSON file and it exists, use it directly.
    Otherwise, fall back to split files if they exist.
    
    Parameters
    ----------
    pop_path : str
        Path to the population file.
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created

    Returns
    -------
    list[dict]
        Parsed population.
    """
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Load Population", file_path=pop_path):
        try:
            pop_path_obj = Path(pop_path)
            
            if pop_path_obj.is_dir():
                base_dir = pop_path_obj
                population_file = base_dir / "reserves.json"
            else:
                population_file = pop_path_obj
                base_dir = pop_path_obj.parent
            if population_file.exists():
                if pop_path_obj.is_dir():
                    _logger.debug("Using monolithic population file (preferred)")
                else:
                    _logger.debug("Using specified population file: %s", pop_path)
                try:
                    with open(population_file, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    population = clean_population(population, logger=_logger, log_file=log_file)
                    if pop_path_obj.is_dir():
                        _logger.debug("Successfully loaded population with %d genomes", len(population))
                    else:
                        _logger.debug("Successfully loaded population with %d genomes from %s", len(population), pop_path)
                    return population
                except Exception as e:
                    _logger.warning("Failed to load population file: %s, falling back to split files", e)
            
            info = get_population_files_info(str(base_dir))
            
            if info["generation_counts"]:
                _logger.debug("Using single file mode with generation counts")
                all_genomes = load_population(str(base_dir), logger=_logger, log_file=log_file)
                
                all_genomes = clean_population(all_genomes, logger=_logger, log_file=log_file)
                _logger.debug("Successfully loaded population with %d genomes from split files", len(all_genomes))
                return all_genomes
            else:
                if not os.path.exists(pop_path):
                    _logger.error("No population files found: neither reserves.json nor split files exist")
                    raise FileNotFoundError(f"No population files found in {base_dir}")
                else:
                    _logger.debug("Using fallback population file: %s", pop_path)
                    with open(pop_path, "r", encoding="utf-8") as f:
                        population = json.load(f)

                    population = clean_population(population, logger=_logger, log_file=log_file)
                    _logger.debug("Successfully loaded population with %d genomes from fallback file", len(population))
                    return population

        except Exception as e:
            _logger.error("Failed to load population: %s", e, exc_info=True)
            raise


def save_population(population: List[Dict[str, Any]], pop_path: str = "data/outputs/reserves.json", 
                   *, logger=None, log_file: Optional[str] = None, preserve_sort_order: bool = False) -> None:
    """
    Save entire population to a JSON file
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to save.
    pop_path : str
        Path where to save the population (can be file or directory path).
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created.
    log_file : str | None
        Optional log-file path when a new logger is created.
    """
    _logger = logger or get_logger("population_io", log_file)

    with PerformanceLogger(_logger, "Save Population", file_path=pop_path, genome_count=len(population)):
        try:
            cleaned_population = clean_population(population, logger=_logger, log_file=log_file)
            
            pop_path_obj = Path(pop_path)
            output_file = pop_path_obj if pop_path_obj.suffix else pop_path_obj / "reserves.json"
            
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            if not preserve_sort_order:
                cleaned_population.sort(key=lambda g: (
                    g.get("generation", 0),
                    g.get("id", "0")
                ))
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(cleaned_population, f, indent=2, ensure_ascii=False)
            
            size_mb = output_file.stat().st_size / (1024 * 1024)
            _logger.info("Successfully saved population to %s: %d genomes, %.2f MB", 
                        output_file.name, len(cleaned_population), size_mb)
            
            update_population_index_single_file(str(output_file.parent), len(cleaned_population), logger=_logger, log_file=log_file)
                        
        except Exception as e:
            _logger.error("Failed to save population: %s", e, exc_info=True)
            raise



def load_and_initialize_population(
    input_path: str,
    output_path: str,
    *,
    log_file: Optional[str] = None,
) -> None:
    """Load prompts from seed CSV and create all output files (except figures).

    Creates: temp.json (seed population), elites.json, reserves.json, archive.json,
    parents.json, top_10.json, EvolutionTracker.json, genome_tracker.json,
    speciation_state.json, events_tracker.json, operator_effectiveness_cumulative.csv.
    Figures are not created at init (insufficient data for visualizations).
    """

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("initialize_population", log_file)

    with PerformanceLogger(
        logger, "Initialize Population", input_path=input_path, output_path=output_path
    ):
        try:
            logger.info("Starting population initialization")
            logger.info("Input file: %s", input_path)
            logger.info("Output directory: %s", output_path)

            if not os.path.exists(input_path):
                logger.error("Input file not found: %s", input_path)
                raise FileNotFoundError(f"Input file not found: {input_path}")

            # ---------------------------- Load CSV File -----------------------
            with PerformanceLogger(logger, "Load CSV File"):
                # Read CSV with Python engine which is more lenient with malformed CSV
                # This handles cases where fields contain commas without proper quoting
                try:
                    df = pd.read_csv(
                        input_path,
                        engine='python',
                        on_bad_lines='skip',
                        sep=',',
                        quotechar='"',
                        skipinitialspace=True
                    )
                except Exception as e:
                    # Fallback: read manually line by line
                    logger.warning("CSV parsing failed, trying manual line-by-line parsing: %s", e)
                    import csv
                    rows = []
                    with open(input_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header and len(header) > 0:
                            # Find questions column (case-insensitive)
                            col_idx = None
                            for i, col in enumerate(header):
                                if col.strip().lower() == 'questions':
                                    col_idx = i
                                    break
                            
                            if col_idx is None:
                                # If no header found, assume first column
                                col_idx = 0
                            
                            for row in reader:
                                if row and len(row) > col_idx:
                                    # Join all fields from col_idx onwards in case comma split the field
                                    question = ','.join(row[col_idx:]).strip()
                                    if question:
                                        rows.append({'questions': question})
                                elif row:
                                    # If row exists but might have been split incorrectly
                                    question = ','.join(row).strip()
                                    if question:
                                        rows.append({'questions': question})
                    df = pd.DataFrame(rows)
                logger.info(
                    "Successfully loaded CSV file with %d rows and %d columns",
                    len(df),
                    len(df.columns),
                )

            # -------------------------- Extract prompts --------------------
            # Only expect a "questions" column in the CSV file
            if "questions" not in df.columns:
                raise ValueError("Required 'questions' column not found in CSV file")
            
            prompt_column = "questions"
            prompts = (
                df[prompt_column].dropna().drop_duplicates().astype(str).str.strip().tolist()
            )
            logger.info("Extracted %d unique prompts from 'questions' column", len(prompts))

            # -------------------------- Create genomes ---------------------
            # Get prompt generator name if available (lazy import to avoid circular dependency)
            prompt_generator_name = None
            try:
                from ea.evolution_engine import get_prompt_generator
                pg = get_prompt_generator()
                if pg and hasattr(pg, 'model_cfg'):
                    prompt_generator_name = pg.model_cfg.get("name", "")
            except Exception:
                # Prompt generator not initialized yet, will be None
                pass

            population: List[Dict[str, Any]] = []
            for i, prompt in enumerate(prompts):
                population.append(
                    {
                        "id": i + 1,
                        "prompt": prompt,
                        "model_name": None,
                        "prompt_generator_name": prompt_generator_name,
                        "moderation_result": None,
                        "operator": None,
                        "parents": [],
                        "parent_score": None,  # null for initial genomes (no parents)
                        "generation": 0,
                        "status": "pending_generation",
                        "variant_type": "initial",  # Moved to top-level
                        "creation_info": {
                            "type": "initial",
                            "operator": "excel_import"
                        }
                    }
                )

            logger.info("Created %d genomes", len(population))

            # ----------------------------- Initialize temp.json (staging) ----------------------------
            with PerformanceLogger(logger, "Initialize temp.json (staging)"):
                temp_path = Path(output_path) / "temp.json"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=2, ensure_ascii=False)
                logger.info("Initialized temp.json with %d genomes (staging)", len(population))

            # ----------------------------- Initialize empty elites.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty elites.json"):
                # elites.json starts empty
                empty_elites = []
                elites_path = Path(output_path) / "elites.json"
                elites_path.parent.mkdir(parents=True, exist_ok=True)
                with open(elites_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_elites, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty elites.json")

            # ----------------------------- Initialize empty reserves.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty reserves.json"):
                # reserves.json starts empty - buffer for high-fitness outliers that don't fit species
                empty_reserves = []
                reserves_path = Path(output_path) / "reserves.json"
                reserves_path.parent.mkdir(parents=True, exist_ok=True)
                with open(reserves_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_reserves, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty reserves.json")
            
            # ----------------------------- Initialize empty archive.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty archive.json"):
                # archive.json starts empty - stores genomes removed due to capacity limits
                empty_archive = []
                archive_path = Path(output_path) / "archive.json"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with open(archive_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_archive, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty archive.json")

            # ----------------------------- Initialize empty parents.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty parents.json"):
                # parents.json starts empty
                empty_parents = []
                parents_path = Path(output_path) / "parents.json"
                parents_path.parent.mkdir(parents=True, exist_ok=True)
                with open(parents_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_parents, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty parents.json")

            # ----------------------------- Initialize empty top_10.json ----------------------------
            with PerformanceLogger(logger, "Initialize empty top_10.json"):
                # top_10.json starts empty
                empty_top_10 = []
                top_10_path = Path(output_path) / "top_10.json"
                top_10_path.parent.mkdir(parents=True, exist_ok=True)
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_top_10, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty top_10.json")

            # ----------------------------- Initialize EvolutionTracker ----------------------------
            with PerformanceLogger(logger, "Initialize EvolutionTracker"):
                evolution_tracker = {
                    "status": "not_complete",
                    "total_generations": 1,  # Generation 0 exists
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "run_metadata": {"num_workers": 1},  # Single-process run; parallel runs set this when creating tracker
                    "generations": [
                        {
                            "generation_number": 0,
                            "genome_id": "1",  # Will be updated with actual best genome during threshold check
                            "max_score_variants": 0.0,  # Will be updated with actual best score during threshold check
                            "avg_fitness": 0.0,  # Will be calculated and updated
                            "parents": None,
                            "top_10": None,
                            "variants_created": None,
                            "mutation_variants": None,
                            "crossover_variants": None,
                            "operator_statistics": {}
                        }
                    ]
                }
                
                # Save EvolutionTracker
                evolution_tracker_path = Path(output_path) / "EvolutionTracker.json"
                evolution_tracker_path.parent.mkdir(parents=True, exist_ok=True)
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_tracker, f, indent=2)
                
                logger.info("Initialized global EvolutionTracker with %d genomes", len(prompts))

            # ----------------------------- Initialize genome_tracker.json ----------------------------
            with PerformanceLogger(logger, "Initialize genome_tracker.json"):
                genome_tracker_path = Path(output_path) / "genome_tracker.json"
                if not genome_tracker_path.exists():
                    genome_tracker_data = {
                        "version": "2.0",
                        "genomes": {},
                        "summary": {"total_genomes": 0, "by_species_id": {}, "last_updated": datetime.now().isoformat()},
                        "metadata": {"last_updated": datetime.now().isoformat(), "total_genomes": 0}
                    }
                    with open(genome_tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(genome_tracker_data, f, indent=2, ensure_ascii=False)
                    logger.info("Initialized empty genome_tracker.json")
                else:
                    logger.debug("genome_tracker.json already exists, skipping")

            # ----------------------------- Initialize speciation_state.json ----------------------------
            with PerformanceLogger(logger, "Initialize speciation_state.json"):
                speciation_state_path = Path(output_path) / "speciation_state.json"
                if not speciation_state_path.exists():
                    try:
                        from speciation.config import SpeciationConfig
                        default_config = SpeciationConfig()
                        config_dict = default_config.to_dict()
                    except Exception:
                        config_dict = {
                            "theta_sim": 0.25, "theta_merge": 0.1,
                            "cluster0_min_cluster_size": 2, "cluster0_max_capacity": 1000,
                            "species_capacity": 100, "min_island_size": 2, "species_stagnation": 20,
                            "embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384,
                            "embedding_batch_size": 64, "w_genotype": 0.7, "w_phenotype": 0.3
                        }
                    speciation_state_data = {
                        "species": {},
                        "incubators": [],
                        "extinct": [],
                        "cluster0": {"cluster_id": 0, "size": 0, "max_capacity": config_dict.get("cluster0_max_capacity", 1000), "speciation_events": []},
                        "cluster0_size_from_reserves": 0,
                        "global_best_id": None,
                        "metrics": {"history": [], "summary": {"total_speciation_events": 0, "total_merge_events": 0, "total_extinction_events": 0}},
                        "config": config_dict
                    }
                    with open(speciation_state_path, 'w', encoding='utf-8') as f:
                        json.dump(speciation_state_data, f, indent=2, ensure_ascii=False)
                    logger.info("Initialized empty speciation_state.json")
                else:
                    logger.debug("speciation_state.json already exists, skipping")

            # ----------------------------- Initialize events_tracker.json ----------------------------
            with PerformanceLogger(logger, "Initialize events_tracker.json"):
                events_tracker_path = Path(output_path) / "events_tracker.json"
                if not events_tracker_path.exists():
                    events_tracker_data = {
                        "generations": [],
                        "summary": {"total_generations": 0, "total_events": 0, "last_updated": datetime.now().isoformat()}
                    }
                    with open(events_tracker_path, 'w', encoding='utf-8') as f:
                        json.dump(events_tracker_data, f, indent=2, ensure_ascii=False)
                    logger.info("Initialized empty events_tracker.json")
                else:
                    logger.debug("events_tracker.json already exists, skipping")

            # ----------------------------- Initialize operator_effectiveness_cumulative.csv ----------------------------
            with PerformanceLogger(logger, "Initialize operator_effectiveness_cumulative.csv"):
                csv_path = Path(output_path) / "operator_effectiveness_cumulative.csv"
                if not csv_path.exists():
                    expected_columns = [
                        "generation", "operator", "NE", "EHR", "IR", "cEHR", "Δμ", "Δσ",
                        "total_variants", "elite_count", "non_elite_count", "rejections", "duplicates"
                    ]
                    empty_df = pd.DataFrame(columns=expected_columns)
                    empty_df.to_csv(csv_path, index=False, na_rep='')
                    logger.info("Initialized empty operator_effectiveness_cumulative.csv")
                else:
                    logger.debug("operator_effectiveness_cumulative.csv already exists, skipping")

            logger.info("Population initialization completed successfully")

        except Exception:
            logger.exception("Population initialization failed")
            raise


def validate_population_file(population_path: str, *, log_file: Optional[str] = None) -> Dict[str, Any]:
    """Run sanity checks on a population JSON and return aggregate statistics."""

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("validate_population", log_file)

    with PerformanceLogger(logger, "Validate Population File", file_path=population_path):
        population = load_population(population_path, logger=logger)

        stats: Dict[str, Any] = {
            "generations": set(),
            "statuses": {},
            "prompt_lengths": [],
            "errors": [],
        }

        for genome in population:
            for field in ("id", "prompt", "generation", "status"):
                if field not in genome:
                    stats["errors"].append(
                        f"Missing required field '{field}' in genome {genome.get('id', '?')}"
                    )

            stats["generations"].add(genome.get("generation", -1))
            status = genome.get("status", "unknown")
            stats["statuses"][status] = stats["statuses"].get(status, 0) + 1
            stats["prompt_lengths"].append(len(genome.get("prompt", "")))

        if stats["prompt_lengths"]:
            stats["avg_prompt_length"] = sum(stats["prompt_lengths"]) / len(
                stats["prompt_lengths"]
            )
            stats["min_prompt_length"] = min(stats["prompt_lengths"])
            stats["max_prompt_length"] = max(stats["prompt_lengths"])

        stats["generations"] = sorted(stats["generations"])

        logger.info("Validation complete – %d genomes analysed", len(population))
        if stats["errors"]:
            logger.warning("Found %d schema issues", len(stats["errors"]))

        return stats


def sort_population_json(
    population: Union[str, List[Dict[str, Any]]],
    sort_keys: List,
    *,
    reverse_flags: Optional[List[bool]] = None,
    output_path: Optional[str] = None,
    log_file: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sort population by multiple keys.

    • *population* may be a list or a path to a JSON file.
    • *sort_keys* items can be strings (direct keys) or callables.
    """

    import collections.abc as _abc

    get_logger, _, _, PerformanceLogger = get_custom_logging()
    logger = get_logger("sort_population", log_file)

    with PerformanceLogger(logger, "Sort Population JSON"):
        if isinstance(population, str):
            pop_list = load_population(population, logger=logger)
            input_is_file = True
        elif isinstance(population, _abc.Sequence):
            pop_list = list(population)
            input_is_file = False
        else:
            raise ValueError("population must be a file path or a list of genomes")

        if reverse_flags is None:
            reverse_flags = [False] * len(sort_keys)
        if len(reverse_flags) != len(sort_keys):
            raise ValueError("reverse_flags must match sort_keys in length")

        def compound_sort_key(genome):
            values = []
            for i, key in enumerate(sort_keys):
                if callable(key):
                    value = key(genome)
                else:
                    value = genome.get(key) if genome is not None else None
                
                if value is None:
                    value = float("-inf") if reverse_flags[i] else float("inf")
                
                if reverse_flags[i] and isinstance(value, (int, float)):
                    value = -value
                elif reverse_flags[i] and isinstance(value, str):
                    try:
                        int_value = int(value)
                        value = -int_value
                        logger.debug("Negated string ID %s -> %d", value, int_value)
                    except ValueError:
                        pass
                
                values.append(value)
            return tuple(values)
        
        logger.debug("Sorting population with %d keys and reverse flags: %s", len(sort_keys), reverse_flags)
        
        pop_list.sort(key=compound_sort_key)

        if output_path:
            dest = output_path
        elif isinstance(population, str):
            dest = population
        else:
            dest = None
            
        if dest:
            logger.debug("Saving sorted population to: %s", dest)
            save_population(pop_list, dest, logger=logger, preserve_sort_order=True)
            logger.info("Successfully saved sorted population to: %s", dest)

        return pop_list


def load_genome_by_id(genome_id: str, generation: int, base_dir: str = "outputs", 
                      *, logger=None, log_file: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Load a specific genome by ID from a specific generation file
    
    Parameters
    ----------
    genome_id : str
        The ID of the genome to load
    generation : int
        The generation number where the genome is stored
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any] | None
        The genome if found, None otherwise
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Genome by ID", genome_id=genome_id, generation=generation):
        try:
            genomes = load_population_generation(generation, base_dir, logger=_logger, log_file=log_file)
            
            for genome in genomes:
                if genome.get("id") == genome_id:
                    _logger.info(f"Found genome {genome_id} in generation {generation}")
                    return genome
            
            _logger.warning(f"Genome {genome_id} not found in generation {generation}")
            return None
            
        except Exception as e:
            _logger.error(f"Failed to load genome {genome_id} from generation {generation}: {e}", exc_info=True)
            return None


def load_genomes_by_ids(genome_ids: List[str], generations: List[int], base_dir: str = "outputs",
                        *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load multiple genomes by their IDs and generation numbers
    
    Parameters
    ----------
    genome_ids : List[str]
        List of genome IDs to load
    generations : List[int]
        List of generation numbers corresponding to each genome ID
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        List of found genomes (may be shorter than input if some not found)
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Genomes by IDs", count=len(genome_ids)):
        try:
            generation_groups = {}
            for genome_id, generation in zip(genome_ids, generations):
                if generation not in generation_groups:
                    generation_groups[generation] = []
                generation_groups[generation].append(genome_id)
            
            found_genomes = []
            
            for generation, ids_in_gen in generation_groups.items():
                genomes = load_population_generation(generation, base_dir, logger=_logger, log_file=log_file)
                
                genome_lookup = {g.get("id"): g for g in genomes}
                
                for genome_id in ids_in_gen:
                    if genome_id in genome_lookup:
                        found_genomes.append(genome_lookup[genome_id])
                    else:
                        _logger.warning(f"Genome {genome_id} not found in generation {generation}")
            
            _logger.info(f"Loaded {len(found_genomes)} out of {len(genome_ids)} requested genomes")
            return found_genomes
            
        except Exception as e:
            _logger.error(f"Failed to load genomes by IDs: {e}", exc_info=True)
            return []


def consolidate_generations_to_single_file(base_dir: str = "outputs", 
                                         output_file: str = "non_elites.json",
                                         *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Consolidate split generation files into a single population file.
    
    This function merges all gen*.json files into one output JSON file,
    effectively reverting from split files to a monolithic layout.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing generation files
    output_file : str
        Name of the output consolidated population file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    bool
        True if consolidation was successful, False otherwise
    """
    _logger = logger or get_logger("consolidate_generations", log_file)
    
    with PerformanceLogger(_logger, "Consolidate Generations to Single File"):
        try:
            base_path = Path(base_dir).resolve()
            output_path = base_path / output_file
            
            info = get_population_files_info(str(base_path))
            
            if not info["generation_counts"]:
                _logger.warning("No generation counts found to consolidate")
                return False
            
            _logger.info(f"Found {len(info['generation_counts'])} generations to consolidate")
            _logger.info(f"Population metadata updated for {len(info['generation_counts'])} generations")
            
            generation_order = sorted(info['generation_counts'].keys())
            
            all_genomes = load_population(str(base_path), logger=_logger, log_file=log_file)
            
            if not all_genomes:
                _logger.error("No genomes loaded from consolidated population source")
                return False
            
            all_genomes = clean_population(all_genomes, logger=_logger, log_file=log_file)
            
            all_genomes.sort(key=lambda g: (
                g.get("generation", 0),
                g.get("id", "0")
            ))
            
            try:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(all_genomes, f, indent=2, ensure_ascii=False)
                
                size_mb = output_path.stat().st_size / (1024 * 1024)
                _logger.info(f"Successfully consolidated {len(all_genomes)} genomes into {output_file}")
                _logger.info(f"File size: {size_mb:.2f} MB")
                
                backup_dir = base_path / "generations_backup"
                backup_dir.mkdir(exist_ok=True)
                
                for gen_num in generation_order:
                    gen_file = base_path / f"gen{gen_num}.json"
                    if gen_file.exists():
                        backup_file = backup_dir / f"gen{gen_num}.json"
                        import shutil
                        shutil.copy2(gen_file, backup_file)
                
                _logger.info(f"Backed up original generation files to {backup_dir}")
                
                return True
                
            except Exception as e:
                _logger.error(f"Failed to save consolidated population file: {e}")
                return False
                
        except Exception as e:
            _logger.error(f"Failed to consolidate generations: {e}", exc_info=True)
            return False


def migrate_from_split_to_single(base_dir: str = "outputs", 
                                *, logger=None, log_file: Optional[str] = None) -> bool:
    """
    Complete migration from split-file architecture back to a single population file.
    
    This function:
    1. Consolidates all generation files into a single output file
    2. Updates the population loading logic to use the single file
    3. Provides a clean migration path
    
    Parameters
    ----------
    base_dir : str
        Base directory containing generation files
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    bool
        True if migration was successful, False otherwise
    """
    _logger = logger or get_logger("migrate_to_single", log_file)
    
    with PerformanceLogger(_logger, "Migrate from Split to Single File"):
        try:
            if not consolidate_generations_to_single_file(base_dir, "reserves.json", logger=_logger, log_file=log_file):
                _logger.error("Failed to consolidate generation files")
                return False
            
            base_path = Path(base_dir).resolve()
            evolution_tracker_file = base_path / "EvolutionTracker.json"
            
            if evolution_tracker_file.exists():
                try:
                    with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                        tracker = json.load(f)
                    
                    tracker["population_metadata"] = {
                        "single_file_mode": True,
                        "population_file": "reserves.json",
                        "elites_file": "elites.json",
                        "migration_note": "Migrated from split files to single reserves.json"
                    }
                    
                    with open(evolution_tracker_file, 'w', encoding='utf-8') as f:
                        json.dump(tracker, f, indent=2)
                    _logger.info("Updated EvolutionTracker for single file architecture")
                    
                except Exception as e:
                    _logger.warning(f"Failed to update EvolutionTracker: {e}")
            
            # Check for consolidated file (now uses reserves.json instead of non_elites.json)
            population_file = base_path / "reserves.json"
            if population_file.exists():
                try:
                    with open(population_file, 'r', encoding='utf-8') as f:
                        consolidated_genomes = json.load(f)
                    
                    _logger.info(f"Migration successful! reserves.json contains {len(consolidated_genomes)} genomes")
                    _logger.info("You can now use the single file approach")
                    
                    return True
                    
                except Exception as e:
                    _logger.error(f"Failed to verify consolidated file: {e}")
                    return False
            else:
                _logger.error("reserves.json was not created during consolidation")
                return False
                
        except Exception as e:
            _logger.error(f"Migration failed: {e}", exc_info=True)
            return False



def _extract_score(genome: Dict[str, Any], north_star_metric: str = "toxicity") -> float:
    """Extract score from genome using north star metric."""
    return _extract_north_star_score(genome, north_star_metric)


def sort_population_by_elite_criteria(population: List[Dict[str, Any]], north_star_metric: str = "toxicity", 
                                     *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Sort population by elite criteria: north star metric score descending, generation descending, genome id descending.
    
    Parameters
    ----------
    population : List[Dict[str, Any]]
        Population to sort
    north_star_metric : str
        Metric to use for scoring (default: "toxicity")
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        Sorted population
    """
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, "Sort Population by Elite Criteria", north_star_metric=north_star_metric):
        def sort_key(genome: Dict[str, Any]) -> tuple:
            score = _extract_score(genome, north_star_metric)
            generation = genome.get("generation", 0)
            genome_id = genome.get("id", 0)
            # Genome IDs are always integers
            return (-score, -generation, -genome_id)
        
        sorted_population = sorted(population, key=sort_key)
        _logger.info(f"Sorted {len(sorted_population)} genomes by elite criteria")
        return sorted_population


def load_elites(elites_file_path: str = "data/outputs/elites.json", 
                *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load elites from elites.json file.
    
    Parameters
    ----------
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    List[Dict[str, Any]]
        List of elite genomes
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        elites_path = Path(elites_file_path)
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            _logger.debug("Loaded %d elites from %s", len(elites), elites_file_path)
            return elites
        else:
            _logger.info(f"Elites file not found: {elites_file_path}, returning empty list")
            return []
    except Exception as e:
        _logger.error(f"Failed to load elites: {e}")
        return []


def save_elites(elites: List[Dict[str, Any]], elites_file_path: str = "data/outputs/elites.json",
                *, logger=None, log_file: Optional[str] = None) -> None:
    """
    Save elites to elites.json file.
    
    Parameters
    ----------
    elites : List[Dict[str, Any]]
        List of elite genomes to save
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        elites_path = Path(elites_file_path)
        elites_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(elites_path, 'w', encoding='utf-8') as f:
            json.dump(elites, f, indent=2, ensure_ascii=False)
        
        _logger.info(f"Saved {len(elites)} elites to {elites_file_path}")
    except Exception as e:
        _logger.error(f"Failed to save elites: {e}")
        raise





def get_population_stats_steady_state(population_file_path: str = FileConstants.DEFAULT_RESERVES_FILE,
                                     elites_file_path: str = FileConstants.DEFAULT_ELITES_FILE,
                                     *, logger=None, log_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Get population statistics for steady state mode.
    
    Note: Active population = elites.json + reserves.json (archive.json is excluded).
    
    Parameters
    ----------
    population_file_path : str
        Path to the reserves.json file (Cluster 0 outliers)
    elites_file_path : str
        Path to the elites.json file
    logger : logging.Logger | None
        Existing logger to reuse; if *None* a new one is created
    log_file : str | None
        Optional log-file path when a new logger is created
        
    Returns
    -------
    Dict[str, Any]
        Population statistics
    """
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        # Load population
        population = load_population(population_file_path, logger=_logger, log_file=log_file)
        
        # Load elites
        elites = load_elites(elites_file_path, logger=_logger, log_file=log_file)
        
        return {
            "elites_count": len(elites),
            "steady_state_mode": True
        }
    except Exception as e:
        _logger.error(f"Failed to get population stats: {e}")
        return {
            "steady_state_mode": True,
            "error": str(e)
        }


# Note: remove_worse_performing_genomes_from_all_files() has been removed.
# After speciation integration, low-fitness genomes are handled by speciation's
# capacity enforcement, not by removal_threshold percentage.


def calculate_average_fitness(
    outputs_path: str, 
    north_star_metric: str = "toxicity", 
    include_temp: bool = False,
    logger=None, 
    log_file: Optional[str] = None
) -> float:
    """
    Calculate the average fitness of genomes.
    
    Two modes controlled by include_temp:
    - include_temp=False (default): mean(elites + reserves) only. Fallback in
      update_adaptive_selection_logic when current_gen_avg_fitness not provided.
    - include_temp=True: mean(elites + reserves + temp) = old elites + old reserves +
      all new variants. Used for EvolutionTracker avg_fitness; called from main before
      run_speciation (before distribution and before archiving). Gen 0: elites/reserves
      empty, so effectively mean(temp).
    
    avg_fitness = mean(old elites + old reserves + all new variants) before speciation.
    avg_fitness_generation = mean(updated elites + updated reserves) after distribution
    (from calculate_generation_statistics). Archived genomes are excluded from
    avg_fitness_generation automatically — stats are computed from elites.json and
    reserves.json only; archived have been removed from those files.
    
    Files:
    - elites.json: Elites assigned to species (species_id > 0)
    - reserves.json: Elite outliers that don't fit existing species (species_id == 0)
    - temp.json: New variants before speciation (included only if include_temp=True)
    - archive.json: Archived/removed genomes (excluded from active population)
    
    Args:
        outputs_path: Path to outputs directory
        north_star_metric: Metric to use for scoring (default: "toxicity")
        include_temp: If True, include temp.json (for BEFORE speciation calculation)
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Average fitness score across selected genomes
    """
    _logger = logger or get_logger("calculate_average_fitness", log_file)
    
    try:
        outputs_dir = Path(outputs_path)
        elites_path = outputs_dir / "elites.json"
        reserves_path = outputs_dir / "reserves.json"
        temp_path = outputs_dir / "temp.json"
        
        total_score = 0.0
        total_count = 0
        
        # Process elites.json (elites assigned to species, species_id > 0)
        if elites_path.exists():
            elites_genomes = load_population(str(elites_path), logger=_logger, log_file=log_file)
            for genome in elites_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
            _logger.debug(f"Processed {len(elites_genomes)} genomes from elites.json")
        
        # Process reserves.json (elite outliers, species_id == 0)
        if reserves_path.exists():
            reserves_genomes = load_population(str(reserves_path), logger=_logger, log_file=log_file)
            for genome in reserves_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
            _logger.debug(f"Processed {len(reserves_genomes)} genomes from reserves.json")
        
        # Process temp.json only if include_temp=True (BEFORE speciation)
        if include_temp and temp_path.exists():
            try:
                with open(temp_path, 'r', encoding='utf-8') as f:
                    temp_genomes = json.load(f)
                for genome in temp_genomes:
                    if genome:
                        score = _extract_north_star_score(genome, north_star_metric)
                        total_score += score
                        total_count += 1
                _logger.debug(f"Processed {len(temp_genomes)} genomes from temp.json")
            except Exception as e:
                _logger.warning(f"Failed to load temp.json for avg_fitness: {e}")
        
        if total_count == 0:
            # This is expected before distribution (generation 0) or if files are empty
            # Only warn if we're not in generation 0 or if files should exist
            if not include_temp:
                # After distribution, elites.json and reserves.json should have genomes
                # If they're empty, this might indicate an issue
                _logger.debug("No genomes found for average fitness calculation (after speciation)")
            else:
                # Before distribution, temp.json might be empty if already processed
                _logger.debug("No genomes found for average fitness calculation (before speciation)")
            return 0.0
        
        avg_fitness = total_score / total_count
        avg_fitness = round(avg_fitness, 4)
        mode = "before speciation (elites+reserves+temp)" if include_temp else "after speciation (elites+reserves)"
        _logger.info(f"Calculated average fitness: {avg_fitness:.4f} from {total_count} genomes ({mode})")
        
        return avg_fitness
        
    except Exception as e:
        _logger.error(f"Failed to calculate average fitness: {e}", exc_info=True)
        return 0.0


def calculate_budget_metrics(
    elites_genomes: List[Dict[str, Any]],
    reserves_genomes: List[Dict[str, Any]],
    temp_genomes: List[Dict[str, Any]],
    current_generation: int,
    logger=None
) -> Dict[str, Any]:
    """
    Calculate evaluation budget metrics for a generation.
    
    Budget metrics track computational cost:
    - llm_calls: Number of LLM calls (response generation) in this generation
    - api_calls: Number of moderation API calls (Perspective API) in this generation
    - total_response_time: Total LLM response generation time (seconds)
    - total_evaluation_time: Total moderation API evaluation time (seconds)
    
    These are counted from genomes created in the current generation.
    
    Args:
        elites_genomes: List of elite genomes
        reserves_genomes: List of reserves genomes
        temp_genomes: List of temp genomes (current generation variants)
        current_generation: Current generation number
        logger: Optional logger instance
        
    Returns:
        Dictionary with budget metrics
    """
    _logger = logger or get_logger("BudgetMetrics")
    
    budget = {
        "llm_calls": 0,
        "api_calls": 0,
        "total_response_time": 0.0,
        "total_evaluation_time": 0.0,
        "total_evaluation_api_wait_seconds": 0.0,
    }
    
    try:
        # Combine all genomes and filter by current generation
        all_genomes = (elites_genomes or []) + (reserves_genomes or []) + (temp_genomes or [])
        current_gen_genomes = [g for g in all_genomes if g and g.get("generation") == current_generation]
        
        for genome in current_gen_genomes:
            # Count LLM calls (each genome with response_duration had an LLM call)
            if genome.get("response_duration") is not None or genome.get("generated_output"):
                budget["llm_calls"] += 1
                if genome.get("response_duration"):
                    budget["total_response_time"] += float(genome.get("response_duration", 0))
            
            # Count API calls (each genome with evaluation_duration had an API call)
            if genome.get("evaluation_duration") is not None or genome.get("moderation_result"):
                budget["api_calls"] += 1
                if genome.get("evaluation_duration"):
                    budget["total_evaluation_time"] += float(genome.get("evaluation_duration", 0))
            # Time spent waiting on API (rate-limit/retry sleep) for this genome
            budget["total_evaluation_api_wait_seconds"] += float(genome.get("evaluation_api_wait_seconds", 0) or 0)
        
        # Round times to 2 decimal places
        budget["total_response_time"] = round(budget["total_response_time"], 2)
        budget["total_evaluation_time"] = round(budget["total_evaluation_time"], 2)
        budget["total_evaluation_api_wait_seconds"] = round(budget["total_evaluation_api_wait_seconds"], 2)
        
        _logger.debug(
            f"Gen {current_generation} budget: {budget['llm_calls']} LLM calls ({budget['total_response_time']}s), "
            f"{budget['api_calls']} API calls ({budget['total_evaluation_time']}s)"
        )
        
        return budget
        
    except Exception as e:
        _logger.warning(f"Failed to calculate budget metrics: {e}")
        return budget


def update_generation_avg_fitness(generation_number: int, avg_fitness: float, evolution_tracker_path: str, logger=None, log_file: Optional[str] = None) -> None:
    """
    Update the avg_fitness field for a specific generation in EvolutionTracker.json.
    
    Args:
        generation_number: The generation number to update
        avg_fitness: The calculated average fitness for this generation
        evolution_tracker_path: Path to EvolutionTracker.json file
        logger: Logger instance
        log_file: Log file path
    """
    _logger = logger or get_logger("update_generation_avg_fitness", log_file)
    
    try:
        # Load EvolutionTracker
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        # Find and update the generation
        generation_updated = False
        for gen in tracker.get("generations", []):
            if gen["generation_number"] == generation_number:
                gen["avg_fitness"] = round(avg_fitness, 4)
                generation_updated = True
                _logger.info(f"Updated generation {generation_number} avg_fitness to {avg_fitness:.4f}")
                break
        
        if not generation_updated:
            _logger.warning(f"Generation {generation_number} not found in EvolutionTracker")
            return
        
        # Save updated tracker
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        _logger.error(f"Failed to update generation avg_fitness: {e}", exc_info=True)
        raise


def calculate_slope(values: List[float]) -> float:
    """
    Calculate the slope of a list of values using linear regression.
    
    Args:
        values: List of numeric values
        
    Returns:
        Slope of the linear regression line
    """
    if len(values) < 2:
        return 0.0
    
    try:
        import numpy as np
        x = np.arange(len(values))
        y = np.array(values)
        
        # Calculate slope using least squares
        slope = np.polyfit(x, y, 1)[0]
        return round(float(slope), 4)
        
    except ImportError:
        # Fallback calculation without numpy
        n = len(values)
        sum_x = sum(range(n))
        sum_y = sum(values)
        sum_xy = sum(i * values[i] for i in range(n))
        sum_x2 = sum(i * i for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        return round(slope, 4)
        
    except Exception:
        return 0.0


def update_adaptive_selection_logic(
    outputs_path: str,
    current_max_toxicity: float,
    previous_max_toxicity: float,
    stagnation_limit: int = 5,
    north_star_metric: str = "toxicity",
    current_gen_avg_fitness: Optional[float] = None,
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Update the adaptive selection logic based on stagnation and fitness trends.
    
    The slope is built from gen["avg_fitness"] (avg_fitness = mean(old elites + old
    reserves + all new variants) before speciation). When current_gen_avg_fitness is
    provided (by main), it is used for the current gen so the slope uses avg_fitness
    consistently; otherwise falls back to calculate_average_fitness(include_temp=False).
    
    Args:
        outputs_path: Path to outputs directory
        current_max_toxicity: Current maximum toxicity score
        previous_max_toxicity: Previous maximum toxicity score (for comparison)
        stagnation_limit: Number of generations without improvement before switching to explore mode
        north_star_metric: Metric to use for scoring (default: "toxicity")
        current_gen_avg_fitness: If provided, avg_fitness for the current gen (mean(old elites+old
            reserves+all new variants) before speciation). Used for slope consistency; else computed
            from elites+reserves after distribution.
        logger: Logger instance
        log_file: Log file path
        
    Returns:
        Dict containing updated selection parameters:
        - selection_mode: "default", "explore", or "exploit"
        - generations_since_improvement: Updated count
        - current_avg_fitness: Current average fitness
        - slope_of_avg_fitness: Slope of fitness history
    """
    _logger = logger or get_logger("update_adaptive_selection_logic", log_file)
    
    try:
        evolution_tracker_path = Path(outputs_path) / "EvolutionTracker.json"
        
        # Load EvolutionTracker
        if not evolution_tracker_path.exists():
            _logger.error("EvolutionTracker.json not found")
            return {"selection_mode": "default", "generations_since_improvement": 0, "current_avg_fitness": 0.0, "slope_of_avg_fitness": 0.0}
        
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        # Use the passed previous_max_toxicity instead of reading from tracker
        _logger.info(f"Adaptive selection comparison: current_max_toxicity={current_max_toxicity:.6f}, previous_max_toxicity={previous_max_toxicity:.6f}")
        
        # Validate values are reasonable
        if current_max_toxicity < 0 or previous_max_toxicity < 0:
            _logger.error(f"Invalid toxicity values: current={current_max_toxicity:.6f}, previous={previous_max_toxicity:.6f}")
        
        # Cross-check with tracker (for debugging)
        tracker_current_max = tracker.get("population_max_toxicity", 0.0001)
        _logger.debug(f"Tracker population_max_toxicity (after update): {tracker_current_max:.6f}")
        
        # Enhanced comparison with explicit validation
        epsilon = 1e-6
        comparison_result = current_max_toxicity > previous_max_toxicity + epsilon
        _logger.debug(f"Comparison: {current_max_toxicity:.6f} > {previous_max_toxicity:.6f} + {epsilon} = {comparison_result}")
        
        # Explicit improvement detection with validation
        if comparison_result:
            tracker["generations_since_improvement"] = 0
            _logger.info(f"✓ Improvement detected! Max toxicity increased from {previous_max_toxicity:.4f} to {current_max_toxicity:.4f}")
        else:
            # Additional validation: if current is very close to previous, log at debug
            if abs(current_max_toxicity - previous_max_toxicity) < 0.001:
                _logger.debug("Values are very close: current=%.6f, previous=%.6f", current_max_toxicity, previous_max_toxicity)
            
            old_value = tracker.get("generations_since_improvement", 0)
            tracker["generations_since_improvement"] = old_value + 1
            _logger.info(f"No improvement. Generations since improvement: {old_value} → {tracker['generations_since_improvement']}")
        
        # Post-comparison validation: ensure improvement is detected when it should be
        if tracker["generations_since_improvement"] > 0:
            # If we didn't detect improvement, verify it was correct
            if current_max_toxicity > previous_max_toxicity + 0.001:  # Significant difference
                _logger.error(f"BUG: Improvement should have been detected! current={current_max_toxicity:.6f} > previous={previous_max_toxicity:.6f}")
                _logger.error("Forcing reset to 0")
                tracker["generations_since_improvement"] = 0
        
        # Use avg_fitness for current gen when provided (slope must use avg_fitness consistently).
        # Otherwise fall back to mean(elites+reserves) after distribution.
        if current_gen_avg_fitness is not None:
            current_avg_fitness = float(current_gen_avg_fitness)
        else:
            current_avg_fitness = calculate_average_fitness(outputs_path, north_star_metric, logger=_logger, log_file=log_file)
        
        # Update avg_fitness_history using sliding window from generations
        avg_fitness_history = tracker.get("avg_fitness_history", [])
        
        # Get current generation number - should be the latest generation
        generations = tracker.get("generations", [])
        if generations:
            current_generation = max(gen.get("generation_number", 0) for gen in generations)
        else:
            current_generation = 0
        
        # Update the current generation's avg_fitness in the tracker
        generation_updated = False
        for gen in tracker.get("generations", []):
            if gen["generation_number"] == current_generation:
                gen["avg_fitness"] = round(current_avg_fitness, 4)
                generation_updated = True
                break
        
        if not generation_updated:
            _logger.warning(f"Generation {current_generation} not found in EvolutionTracker for avg_fitness update")
        
        # Build avg_fitness_history from the last m generations
        # IMPORTANT: Re-fetch generations after update to ensure we have the latest values
        generations = tracker.get("generations", [])
        # Filter for generations with valid avg_fitness (exclude None)
        # Only include generations where avg_fitness was actually calculated (not the initial 0.0 placeholder)
        generations_with_avg_fitness = []
        for gen in generations:
            if "avg_fitness" in gen and gen["avg_fitness"] is not None:
                # Skip initial placeholder 0.0 values that haven't been updated yet
                # If avg_fitness is 0.0 and this is the first time we're calculating, 
                # it means the calculation hasn't happened yet or returned 0.0 legitimately
                # We include it only if it's not the initial placeholder (i.e., if current_avg_fitness was calculated)
                if gen["generation_number"] == current_generation:
                    # For the current generation, use the calculated value we just computed
                    generations_with_avg_fitness.append(gen)
                elif gen["avg_fitness"] > 0.0 or gen.get("elites_count", 0) > 0 or gen.get("reserves_count", 0) > 0 or gen.get("archived_count", 0) > 0:
                    # For past generations, include if avg_fitness > 0 or if population exists (indicating it was calculated)
                    generations_with_avg_fitness.append(gen)
                # Otherwise, skip 0.0 values that are likely placeholders
        
        # Sort by generation number and take the last m generations (sliding window)
        generations_with_avg_fitness.sort(key=lambda x: x["generation_number"])
        # Take the last stagnation_limit generations (or all if fewer than stagnation_limit exist)
        recent_generations = generations_with_avg_fitness[-stagnation_limit:]
        
        # Extract avg_fitness values for the sliding window (round to 4 decimal places)
        avg_fitness_history = [round(gen["avg_fitness"], 4) for gen in recent_generations]
        
        _logger.info(f"Built avg_fitness_history with {len(avg_fitness_history)} entries from {len(generations_with_avg_fitness)} total generations (window size: {stagnation_limit})")
        
        tracker["avg_fitness_history"] = avg_fitness_history
        
        # Calculate slope of avg_fitness_history (already rounded in calculate_slope, but ensure it's 4 decimals)
        slope_of_avg_fitness = calculate_slope(avg_fitness_history)
        slope_of_avg_fitness = round(slope_of_avg_fitness, 4)
        tracker["slope_of_avg_fitness"] = slope_of_avg_fitness
        
        # Determine selection mode
        generations_since_improvement = tracker["generations_since_improvement"]
        total_generations = tracker.get("total_generations", 1)
        
        # For the first m generations (where m = stagnation_limit), always use DEFAULT mode
        if total_generations <= stagnation_limit:
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode for initial {stagnation_limit} generations (generation {total_generations})")
        elif slope_of_avg_fitness <= 0.00:
            # Check EXPLOIT condition first (zero or negative fitness slope)
            # When slope <= 0.00, there's no improvement (same or declining fitness)
            selection_mode = "exploit"
            _logger.info(f"Switching to EXPLOIT mode (fitness slope: {slope_of_avg_fitness:.4f} <= 0.00)")
        elif generations_since_improvement >= stagnation_limit:
            # Then check EXPLORE condition (stagnation)
            selection_mode = "explore"
            _logger.info(f"Switching to EXPLORE mode (generations since improvement: {generations_since_improvement} >= {stagnation_limit})")
        else:
            # Finally DEFAULT mode
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode (generations since improvement: {generations_since_improvement}, slope: {slope_of_avg_fitness:.4f})")
        
        tracker["selection_mode"] = selection_mode
        
        # Save updated tracker
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2)
        
        _logger.info(f"Updated adaptive selection: mode={selection_mode}, avg_fitness={current_avg_fitness:.4f}, slope={slope_of_avg_fitness:.4f}")
        
        return {
            "selection_mode": selection_mode,
            "generations_since_improvement": generations_since_improvement,
            "current_avg_fitness": round(current_avg_fitness, 4),
            "slope_of_avg_fitness": slope_of_avg_fitness
        }
        
    except Exception as e:
        _logger.error(f"Failed to update adaptive selection logic: {e}", exc_info=True)
        return {"selection_mode": "default", "generations_since_improvement": 0, "current_avg_fitness": 0.0, "slope_of_avg_fitness": 0.0}


def calculate_generation_statistics(
    outputs_path: str,
    north_star_metric: str = "toxicity",
    current_generation: int = 0,
    logger=None,
    log_file: Optional[str] = None
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistics for a generation.
    
    Reads elites.json, reserves.json, and temp.json to calculate:
    - Fitness statistics (max, min, avg) for each category
    - Population counts
    - Score distributions
    
    Args:
        outputs_path: Path to the outputs directory
        north_star_metric: Metric to use for fitness (default: "toxicity")
        current_generation: Current generation number
        logger: Optional logger instance
        log_file: Optional log file path
        
    Returns:
        Dictionary with all calculated statistics
    """
    _logger = logger or get_logger("GenerationStatistics", log_file)
    _logger.info("Calculating generation statistics for gen %d", current_generation)

    outputs_dir = Path(outputs_path)
    stats = {
        "generation_number": current_generation,
        "initial_population_size": 0,
        "elites_count": 0,
        "reserves_count": 0,
        "archived_count": 0,
        "total_population": 0,
        "max_score_variants": 0.0001,
        "min_score_variants": 0.0001,
        "avg_fitness_variants": 0.0001,
        "avg_fitness_generation": 0.0001,
        "avg_fitness": 0.0001,
        "avg_fitness_elites": 0.0001,
        "avg_fitness_reserves": 0.0001,
        "population_max_toxicity": 0.0001,
    }
    
    try:
        # Load elites.json
        elites_path = outputs_dir / "elites.json"
        elites_genomes = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
        
        # Load reserves.json
        reserves_path = outputs_dir / "reserves.json"
        reserves_genomes = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
        
        # Load temp.json (current variants being processed)
        temp_path = outputs_dir / "temp.json"
        temp_genomes = []
        if temp_path.exists():
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
        
        # Load archive.json (archived genomes)
        archive_path = outputs_dir / "archive.json"
        archive_genomes = []
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_genomes = json.load(f)
                # Ensure archive is a list (handle edge cases)
                if not isinstance(archive_genomes, list):
                    if isinstance(archive_genomes, dict):
                        _logger.warning(f"archive.json is a dict (expected list), converting to list")
                        archive_genomes = list(archive_genomes.values()) if len(archive_genomes) > 0 else []
                    else:
                        _logger.warning(f"archive.json has unexpected format, treating as empty")
                        archive_genomes = []
            except Exception as e:
                _logger.warning(f"Failed to load archive.json: {e}")
        
        # Calculate counts (cumulative - files contain all genomes from all generations)
        # elites.json and reserves.json are cumulative, so we count genomes up to and including current generation
        # Filter genomes by generation to get cumulative count up to current generation
        # Note: Genomes without generation field default to 0, which means they're included for all generations
        # This is correct for cumulative metrics (include all genomes up to and including current generation)
        # However, genomes should have generation field set during distribution (see Phase 7 redistribution in run_speciation.py)
        def _get_generation_value(genome, current_gen):
            """Get generation value for filtering, handling missing values."""
            gen_val = genome.get("generation")
            if gen_val is None:
                # If generation is missing, default to 0 (include in all generations for cumulative metrics)
                # This handles edge cases but genomes should have generation set during distribution
                return 0
            return gen_val
        
        elites_up_to_gen = [g for g in elites_genomes if _get_generation_value(g, current_generation) <= current_generation]
        reserves_up_to_gen = [g for g in reserves_genomes if _get_generation_value(g, current_generation) <= current_generation]
        archive_up_to_gen = [g for g in archive_genomes if _get_generation_value(g, current_generation) <= current_generation]
        
        stats["elites_count"] = len(elites_up_to_gen)
        stats["reserves_count"] = len(reserves_up_to_gen)
        stats["archived_count"] = len(archive_up_to_gen)
        stats["total_population"] = stats["elites_count"] + stats["reserves_count"]
        
        # Calculate elite fitness statistics (use filtered genomes for accuracy)
        elite_scores = []
        for g in elites_up_to_gen:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                elite_scores.append(score)
        
        if elite_scores:
            stats["avg_fitness_elites"] = round(sum(elite_scores) / len(elite_scores), 4)
        
        # Calculate reserves fitness statistics (use filtered genomes for accuracy)
        reserves_scores = []
        for g in reserves_up_to_gen:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                reserves_scores.append(score)
        
        if reserves_scores:
            stats["avg_fitness_reserves"] = round(sum(reserves_scores) / len(reserves_scores), 4)
        
        # Calculate temp/variant fitness statistics
        variant_scores = []
        for g in temp_genomes:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                variant_scores.append(score)
        
        if variant_scores:
            stats["max_score_variants"] = round(max(variant_scores), 4)
            stats["min_score_variants"] = round(min(variant_scores), 4)
            stats["avg_fitness_variants"] = round(sum(variant_scores) / len(variant_scores), 4)
        
        # avg_fitness_generation: mean over elites + reserves only (after distribution)
        all_scores = elite_scores + reserves_scores
        if all_scores:
            stats["avg_fitness_generation"] = round(sum(all_scores) / len(all_scores), 4)
            # population_max_toxicity (per-gen): max over elites+reserves; cumulative is
            # updated in update_evolution_tracker_with_statistics. Used for Pareto quality.
            # NOTE: This is cumulative max (all genomes with generation <= current_generation)
            max_score = max(all_scores)
            if max_score > 0.0001:  # Ensure we have a valid score
                stats["population_max_toxicity"] = round(max_score, 4)
                _logger.debug(f"Gen {current_generation}: Calculated population_max_toxicity={stats['population_max_toxicity']:.4f} from {len(all_scores)} scores (cumulative max)")
            else:
                # If all scores are <= 0.0001, something is wrong - log warning but set to 0.0001
                _logger.warning(f"Gen {current_generation}: All fitness scores <= 0.0001, setting population_max_toxicity to 0.0001")
                stats["population_max_toxicity"] = 0.0001
        else:
            # No scores found - this should not happen if we have genomes
            if stats.get("total_population", 0) > 0:
                _logger.warning(f"Gen {current_generation}: No fitness scores found but total_population={stats.get('total_population')} - setting population_max_toxicity to 0.0001")
            stats["population_max_toxicity"] = 0.0001
        
        # avg_fitness: in non-parallel mode, supplied by main from
        # calculate_average_fitness(include_temp=True) before speciation.
        # In parallel mode, no separate call is made. If still at default (0.0001),
        # fall through to avg_fitness_generation as a reasonable approximation.
        if stats["avg_fitness"] <= 0.0001 and stats["avg_fitness_generation"] > 0.0001:
            stats["avg_fitness"] = stats["avg_fitness_generation"]
        
        # For generation 0, initial_population_size is the count before distribution
        if current_generation == 0:
            # Use temp_genomes or combined count
            stats["initial_population_size"] = len(temp_genomes) if temp_genomes else stats["total_population"]
        
        # Calculate budget metrics (LLM calls + API calls) for current generation
        budget_metrics = calculate_budget_metrics(
            elites_genomes, reserves_genomes, temp_genomes,
            current_generation, _logger
        )
        stats.update(budget_metrics)
        
        _logger.info(
            "Gen %d stats: elites=%d reserves=%d total=%d avg_fit_gen=%.4f max_tox=%.4f",
            current_generation, stats["elites_count"], stats["reserves_count"],
            stats["total_population"], stats["avg_fitness_generation"],
            stats.get("population_max_toxicity", 0.0001)
        )
        _logger.debug(
            "Gen %d stats: elites=%d (avg=%.4f), reserves=%d (avg=%.4f), archived=%d, total=%d, avg_gen=%.4f, llm_calls=%d, api_calls=%d",
            current_generation, stats["elites_count"], stats["avg_fitness_elites"],
            stats["reserves_count"], stats["avg_fitness_reserves"],
            stats["archived_count"], stats["total_population"], stats["avg_fitness_generation"],
            stats.get("llm_calls", 0), stats.get("api_calls", 0)
        )
        
        return stats
        
    except Exception as e:
        _logger.error(f"Failed to calculate generation statistics: {e}", exc_info=True)
        return stats


def _get_standard_generation_entry_template(generation_number: int, selection_mode: str = "default") -> Dict[str, Any]:
    """
    Create a standard generation entry template with ALL required fields.
    
    So all generation entries have consistent fields across updates.
    
    Args:
        generation_number: Generation number
        selection_mode: Selection mode (default: "default")
        
    Returns:
        Dictionary with all standard fields initialized to defaults
    """
    return {
        "generation_number": generation_number,
        "genome_id": None,
        "max_score_variants": 0.0001,
        "min_score_variants": 0.0001,
        "avg_fitness": 0.0001,
        "avg_fitness_variants": 0.0001,
        "avg_fitness_generation": 0.0001,
        "avg_fitness_elites": 0.0001,
        "avg_fitness_reserves": 0.0001,
        "parents": [],
        "top_10": [],
        "variants_created": 0,
        "mutation_variants": 0,
        "crossover_variants": 0,
        "elites_count": 0,
        "reserves_count": 0,
        "archived_count": 0,
        "total_population": 0,
        "selection_mode": selection_mode,
        "operator_statistics": {},
        "speciation": None,  # Will be set by speciation update
        "budget": None,  # Will be set if available
        "generation_duration_seconds": None,  # Wall-clock duration for this generation
        "genomes_per_second": None,  # variants_created / generation_duration_seconds (for scaling analysis)
    }


def _ensure_generation_entry_has_all_fields(gen_entry: Dict[str, Any], generation_number: int, selection_mode: str = "default") -> Dict[str, Any]:
    """
    Ensure generation entry has all standard fields, filling in missing ones with defaults.
    Modifies gen_entry in place and returns it so the reference in tracker["generations"] is preserved.

    Args:
        gen_entry: Existing generation entry (may be partial)
        generation_number: Generation number
        selection_mode: Selection mode (default: "default")

    Returns:
        The same gen_entry dict (mutated in place)
    """
    template = _get_standard_generation_entry_template(generation_number, selection_mode)

    # Fill only missing keys from template (preserves existing values including speciation, etc.)
    for k, v in template.items():
        gen_entry.setdefault(k, v)

    # Force generation_number
    gen_entry["generation_number"] = generation_number

    # Ensure lists/dicts are not None
    if gen_entry.get("parents") is None:
        gen_entry["parents"] = []
    if gen_entry.get("top_10") is None:
        gen_entry["top_10"] = []
    if gen_entry.get("operator_statistics") is None:
        gen_entry["operator_statistics"] = {}

    return gen_entry


def update_evolution_tracker_with_statistics(
    evolution_tracker_path: str,
    current_generation: int,
    statistics: Dict[str, Any],
    operator_statistics: Optional[Dict[str, Any]] = None,
    logger=None,
    log_file: Optional[str] = None,
    run_metadata_update: Optional[Dict[str, Any]] = None,
) -> bool:
    """
    Update EvolutionTracker.json with generation statistics.
    
    Args:
        evolution_tracker_path: Path to EvolutionTracker.json
        current_generation: Current generation number
        statistics: Dictionary of statistics from calculate_generation_statistics()
        operator_statistics: Optional operator-level statistics
        logger: Optional logger instance
        log_file: Optional log file path
        run_metadata_update: Optional dict to merge into tracker["run_metadata"] (e.g. {"num_workers": N})
        
    Returns:
        True if successful, False otherwise
    """
    _logger = logger or get_logger("UpdateEvolutionTracker", log_file)
    
    try:
        tracker_path = Path(evolution_tracker_path)
        if not tracker_path.exists():
            _logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        
        with open(tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        if run_metadata_update:
            tracker.setdefault("run_metadata", {}).update(run_metadata_update)
        
        # Find or create generation entry
        generations = tracker.setdefault("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        selection_mode = tracker.get("selection_mode", "default")
        
        if gen_entry is None:
            # Create new entry with all standard fields
            gen_entry = _get_standard_generation_entry_template(current_generation, selection_mode)
            generations.append(gen_entry)
        else:
            # Ensure existing entry has all fields
            gen_entry = _ensure_generation_entry_has_all_fields(gen_entry, current_generation, selection_mode)
        
        # Update with statistics (round all float values to 4 decimal places)
        # Preserve existing speciation data if present
        existing_speciation = gen_entry.get("speciation")
        # avg_fitness: mean(old elites + old reserves + all new variants) before speciation
        # (from main). Differs from avg_fitness_generation when genomes are archived this gen.
        if statistics.get("best_genome_id") is not None:
            gen_entry["genome_id"] = statistics["best_genome_id"]

        gen_entry.update({
            "elites_count": statistics.get("elites_count", 0),
            "reserves_count": statistics.get("reserves_count", 0),
            "archived_count": statistics.get("archived_count", 0),
            "total_population": statistics.get("total_population", 0),
            "generation_duration_seconds": round(statistics["generation_duration_seconds"], 3) if statistics.get("generation_duration_seconds") is not None else None,
            "best_fitness": round(statistics.get("population_max_toxicity", gen_entry.get("best_fitness", 0.0001)), 4),
            "max_score_variants": round(statistics.get("max_score_variants", gen_entry.get("max_score_variants", 0.0001)), 4),
            "min_score_variants": round(statistics.get("min_score_variants", gen_entry.get("min_score_variants", 0.0001)), 4),
            "avg_fitness_variants": round(statistics.get("avg_fitness_variants", gen_entry.get("avg_fitness_variants", 0.0001)), 4),
            "avg_fitness_generation": round(statistics.get("avg_fitness_generation", gen_entry.get("avg_fitness_generation", 0.0001)), 4),
            "avg_fitness": round(statistics.get("avg_fitness", statistics.get("avg_fitness_generation", gen_entry.get("avg_fitness", 0.0001))), 4),
            "avg_fitness_elites": round(statistics.get("avg_fitness_elites", gen_entry.get("avg_fitness_elites", 0.0001)), 4),
            "avg_fitness_reserves": round(statistics.get("avg_fitness_reserves", gen_entry.get("avg_fitness_reserves", 0.0001)), 4),
        })
        
        # Speciation block: keep if already a non-empty dict; otherwise build from statistics.
        # This covers Gen 0 (EvolutionTracker may not exist when run_speciation's update runs)
        # and edge cases where update_evolution_tracker_with_speciation did not run.
        if existing_speciation is not None and isinstance(existing_speciation, dict) and existing_speciation.get("species_count") is not None:
            gen_entry["speciation"] = existing_speciation
            if statistics.get("speciation_duration_seconds") is not None:
                gen_entry["speciation"]["speciation_duration_seconds"] = round(statistics["speciation_duration_seconds"], 3)
        elif any(statistics.get(k) is not None for k in ("species_count", "reserves_size")):
            gen_entry["speciation"] = {
                "species_count": statistics.get("species_count", 0),
                "active_species_count": statistics.get("active_species_count", statistics.get("species_count", 0)),
                "frozen_species_count": statistics.get("frozen_species_count", 0),
                "reserves_size": statistics.get("reserves_size", statistics.get("reserves_count", 0)),
                "speciation_events": statistics.get("speciation_events", 0),
                "merge_events": statistics.get("merge_events", 0),
                "extinction_events": statistics.get("extinction_events", 0),
                "archived_count": statistics.get("archived_count", 0),
                "elites_moved": statistics.get("elites_moved", 0),
                "reserves_moved": statistics.get("reserves_moved", 0),
                "genomes_updated": statistics.get("genomes_updated", 0),
                "inter_species_diversity": statistics.get("inter_species_diversity", existing_speciation.get("inter_species_diversity", 0.0) if existing_speciation else 0.0),
                "intra_species_diversity": statistics.get("intra_species_diversity", existing_speciation.get("intra_species_diversity", 0.0) if existing_speciation else 0.0),
                "total_population": statistics.get("total_population", 0),
                "cluster_quality": statistics.get("cluster_quality", existing_speciation.get("cluster_quality") if existing_speciation else None),
                "speciation_duration_seconds": round(statistics["speciation_duration_seconds"], 3) if statistics.get("speciation_duration_seconds") is not None else None,
            }
        
        # Add budget metrics if available
        if "llm_calls" in statistics:
            gen_entry["budget"] = {
                "llm_calls": statistics.get("llm_calls", 0),
                "api_calls": statistics.get("api_calls", 0),
                "total_response_time": statistics.get("total_response_time", 0.0),
                "total_evaluation_time": statistics.get("total_evaluation_time", 0.0),
                "total_evaluation_api_wait_seconds": statistics.get("total_evaluation_api_wait_seconds", 0.0),
            }
            
            # Update cumulative budget at tracker level
            if "cumulative_budget" not in tracker:
                tracker["cumulative_budget"] = {
                    "total_llm_calls": 0,
                    "total_api_calls": 0,
                    "total_response_time": 0.0,
                    "total_evaluation_time": 0.0,
                    "total_evaluation_api_wait_seconds": 0.0,
                }
            
            tracker["cumulative_budget"]["total_llm_calls"] += statistics.get("llm_calls", 0)
            tracker["cumulative_budget"]["total_api_calls"] += statistics.get("api_calls", 0)
            tracker["cumulative_budget"]["total_response_time"] = round(
                tracker["cumulative_budget"]["total_response_time"] + statistics.get("total_response_time", 0.0), 2
            )
            tracker["cumulative_budget"]["total_evaluation_time"] = round(
                tracker["cumulative_budget"]["total_evaluation_time"] + statistics.get("total_evaluation_time", 0.0), 2
            )
            tracker["cumulative_budget"]["total_evaluation_api_wait_seconds"] = round(
                tracker["cumulative_budget"].get("total_evaluation_api_wait_seconds", 0.0)
                + statistics.get("total_evaluation_api_wait_seconds", 0.0), 2
            )
        
        # Update population_max_toxicity at tracker level (cumulative max across all generations).
        # population_max_toxicity = max over all gens of (best toxicity in elites+reserves). Used for Pareto quality.
        new_max = statistics.get("population_max_toxicity")
        if new_max and new_max > 0.0001:
            # Initialize if not present
            if "population_max_toxicity" not in tracker:
                tracker["population_max_toxicity"] = 0.0001
            # Update to cumulative max (always keep the highest value seen)
            tracker["population_max_toxicity"] = max(
                tracker.get("population_max_toxicity", 0.0001),
                new_max
            )
            _logger.debug(f"Updated cumulative population_max_toxicity to {tracker['population_max_toxicity']:.4f}")
        
        if statistics.get("variants_created") is not None:
            gen_entry["variants_created"] = statistics.get("variants_created", 0)
        gd = statistics.get("generation_duration_seconds")
        vc = statistics.get("variants_created")
        if gd and gd > 0 and vc is not None:
            gen_entry["genomes_per_second"] = round(vc / gd, 4)
        if statistics.get("mutation_variants") is not None:
            gen_entry["mutation_variants"] = statistics.get("mutation_variants", 0)
        if statistics.get("crossover_variants") is not None:
            gen_entry["crossover_variants"] = statistics.get("crossover_variants", 0)
        
        # Parents and top_10: use from statistics if provided (e.g. parallel master), else load from files
        if statistics.get("parents") is not None:
            gen_entry["parents"] = statistics["parents"]
        if statistics.get("top_10") is not None:
            gen_entry["top_10"] = statistics["top_10"]
        
        # Populate parents and top_10 from their JSON files if not already populated
        outputs_dir = os.path.dirname(tracker_path)
        if not gen_entry.get("parents"):
            parents_path = os.path.join(outputs_dir, "parents.json")
            try:
                if os.path.exists(parents_path):
                    with open(parents_path, 'r', encoding='utf-8') as pf:
                        parents_data = json.load(pf)
                    if parents_data:
                        gen_entry["parents"] = [
                            {"id": p.get("id"), "prompt": p.get("prompt", "")[:100], "toxicity": p.get("toxicity", 0)}
                            for p in parents_data
                        ] if isinstance(parents_data, list) else []
                        _logger.debug("Loaded %d parents from %s for gen %d",
                                      len(gen_entry["parents"]), parents_path, current_generation)
            except Exception:
                pass
        if not gen_entry.get("top_10"):
            top10_path = os.path.join(outputs_dir, "top_10.json")
            try:
                if os.path.exists(top10_path):
                    with open(top10_path, 'r', encoding='utf-8') as tf:
                        top10_data = json.load(tf)
                    if top10_data:
                        gen_entry["top_10"] = [
                            {"id": t.get("id"), "prompt": t.get("prompt", "")[:100], "toxicity": t.get("toxicity", 0)}
                            for t in top10_data
                        ] if isinstance(top10_data, list) else []
                        _logger.debug("Loaded %d top_10 from %s for gen %d",
                                      len(gen_entry["top_10"]), top10_path, current_generation)
            except Exception:
                pass
        
        # Add operator statistics if provided
        if operator_statistics:
            gen_entry["operator_statistics"] = operator_statistics
        
        # Sort generations by number
        tracker["generations"] = sorted(generations, key=lambda x: x.get("generation_number", 0))
        
        # Keep total_generations in sync with the actual generations array
        if tracker["generations"]:
            tracker["total_generations"] = max(
                g.get("generation_number", 0) for g in tracker["generations"]
            ) + 1
        
        # Save updated tracker (use indent=2 to match other JSON files)
        with open(tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        
        best_fit = statistics.get("population_max_toxicity", gen_entry.get("best_fitness", 0.0001))
        _logger.info(
            "Updated EvolutionTracker gen %d: elites=%d, reserves=%d, archived=%d, "
            "avg_fitness=%.4f, best_fitness=%.4f, parents=%d, top_10=%d",
            current_generation, statistics.get("elites_count", 0),
            statistics.get("reserves_count", 0), statistics.get("archived_count", 0),
            statistics.get("avg_fitness_generation", 0.0001), best_fit,
            len(gen_entry.get("parents", [])), len(gen_entry.get("top_10", []))
        )
        
        return True
        
    except Exception as e:
        _logger.error(f"Failed to update EvolutionTracker with statistics: {e}", exc_info=True)
        return False


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    # Main I/O functions
    "load_population",
    "save_population",
    
    # Split file management
    "get_population_files_info",
    "load_population_generation",
    "load_population_range", 
    "load_population_lazy",
    "save_population_generation",
    "update_population_index_single_file",
    "get_latest_generation",
    
    # Genome-specific loading
    "load_genome_by_id",
    "load_genomes_by_ids",
    
    # Population management
    "load_and_initialize_population",
    "validate_population_file",
    "sort_population_json",
    "clean_population",
    
    # Adaptive selection
    "calculate_average_fitness",
    "update_generation_avg_fitness",
    "calculate_slope",
    "update_adaptive_selection_logic",
    
    # Migration functions
    "consolidate_generations_to_single_file",
    "migrate_from_split_to_single",
    
    # Steady state population management
    "sort_population_by_elite_criteria",
    "load_elites",
    "save_elites",
    "get_population_stats_steady_state",
    
    # Generation statistics
    "calculate_generation_statistics",
    "update_evolution_tracker_with_statistics",
]