

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
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent
    return project_root.resolve()

def get_config_path():
    
    return get_project_root() / "config"

def get_data_path():
    
    return get_project_root() / "data" / "prompt.csv"
_current_outputs_path = None

def _max_genome_id_from_iter(genomes) -> int:
    
    out = 0
    for g in genomes:
        if not isinstance(g, dict):
            continue
        kid = g.get("id")
        if kid is None or isinstance(kid, bool):
            continue
        try:
            if isinstance(kid, str):
                s = kid.strip()
                if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
                    out = max(out, int(s))
            elif isinstance(kid, (int, float)):
                out = max(out, int(kid))
        except (TypeError, ValueError):
            continue
    return out


def get_max_genome_id_from_all_files(outputs_path: Optional[Union[str, Path]] = None) -> int:
    
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
    
    global _current_outputs_path
    outputs_dir = Path(path).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    _current_outputs_path = outputs_dir
    return outputs_dir


def get_outputs_path():
    
    global _current_outputs_path
    
    if _current_outputs_path is not None:
        return _current_outputs_path
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    outputs_dir = get_project_root() / "data" / "outputs" / timestamp
    
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    _current_outputs_path = outputs_dir
    
    return outputs_dir

def get_moderation_scores(genome) -> Optional[Dict[str, float]]:
    if not genome:
        return None
    mr = genome.get("moderation_result")
    if not mr or not isinstance(mr, dict):
        return None
    for backend in ("google", "openai"):
        block = mr.get(backend)
        if isinstance(block, dict) and isinstance(block.get("scores"), dict):
            return block["scores"]
    if isinstance(mr.get("scores"), dict):
        return mr["scores"]
    return None


def _resolve_score_metric(metric: str, scores: Dict[str, float], genome=None) -> str:
    if metric in scores:
        return metric
    try:
        from utils.evaluator_profiles import get_active_evaluator, resolve_evaluator
        evaluator_name = (genome or {}).get("evaluator") if genome else None
        profile = resolve_evaluator(evaluator_name) if evaluator_name else get_active_evaluator()
        resolved = profile.resolve_metric_alias(metric)
        if resolved in scores:
            return resolved
    except Exception:
        pass
    metric_variations = {
        "violence": ["violence", "threat", "violence/graphic"],
        "toxicity": ["toxicity", "severe_toxicity", "harassment"],
        "hate": ["hate", "identity_attack", "hate/threatening"],
    }
    if metric in metric_variations:
        for variation in metric_variations[metric]:
            if variation in scores:
                return variation
    return metric


def _extract_north_star_score(genome, metric="toxicity"):
    
    if not genome:
        return 0.0001

    def _valid(s):
        return s is not None and float(s) > 0

    scores = get_moderation_scores(genome)
    if scores:
        key = _resolve_score_metric(metric, scores, genome=genome)
        s = scores.get(key, 0.0001)
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
    
    from utils.device_utils import device_manager
    device = device_manager.get_optimal_device()
    
    logger.debug("Initializing pipeline for device: %s", device)
    
    load_and_initialize_population = get_population_io()[0]
    
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
    
    _logger = logger or get_logger("population_io", log_file)
    cleaned = [g for g in population if g is not None]
    n_removed = len(population) - len(cleaned)
    if n_removed:
        _logger.warning("Removed %d None genomes from population", n_removed)
    _logger.debug("Population cleaned: %d → %d genomes", len(population), len(cleaned))
    return cleaned



def get_population_files_info(base_dir: str = "outputs") -> Dict[str, Any]:
    
    _log = get_logger("population_io")
    base_path = Path(base_dir).resolve()
    population_file = base_path / "reserves.json"
    elites_file = base_path / "elites.json"
    evolution_tracker_file = base_path / "EvolutionTracker.json"
    
    info = {
        "total_generations": 0,
        "generation_counts": {},
    }
    
    if evolution_tracker_file.exists():
        try:
            with open(evolution_tracker_file, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            if "generations" in tracker and tracker["generations"]:
                max_gen_num = max(gen.get("generation_number", 0) for gen in tracker["generations"])
                info["total_generations"] = max_gen_num + 1
            else:
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
    
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, f"Load Generation {generation} from Single File"):
        try:
            all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
            
            generation_genomes = [g for g in all_genomes if g and g.get("generation") == generation]
            
            _logger.info("Loaded generation %d: %d genomes from active population file", generation, len(generation_genomes))
            return generation_genomes
            
        except Exception as e:
            _logger.error("Failed to load generation %d: %s", generation, e, exc_info=True)
            return []


def load_population_range(start_gen: int, end_gen: int, base_dir: str = "outputs",
                         *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    
    
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
    
    
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        all_genomes = load_population(base_dir, logger=_logger, log_file=log_file)
        
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
    
    info = get_population_files_info(base_dir)
    return info["total_generations"] - 1 if info["total_generations"] > 0 else 0


def load_population(pop_path: str = "data/outputs/reserves.json", *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    
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

            with PerformanceLogger(logger, "Load CSV File"):
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
                    logger.warning("CSV parsing failed, trying manual line-by-line parsing: %s", e)
                    import csv
                    rows = []
                    with open(input_path, 'r', encoding='utf-8') as f:
                        reader = csv.reader(f)
                        header = next(reader, None)
                        if header and len(header) > 0:
                            col_idx = None
                            for i, col in enumerate(header):
                                if col.strip().lower() == 'questions':
                                    col_idx = i
                                    break
                            
                            if col_idx is None:
                                col_idx = 0
                            
                            for row in reader:
                                if row and len(row) > col_idx:
                                    question = ','.join(row[col_idx:]).strip()
                                    if question:
                                        rows.append({'questions': question})
                                elif row:
                                    question = ','.join(row).strip()
                                    if question:
                                        rows.append({'questions': question})
                    df = pd.DataFrame(rows)
                logger.info(
                    "Successfully loaded CSV file with %d rows and %d columns",
                    len(df),
                    len(df.columns),
                )

            if "questions" not in df.columns:
                raise ValueError("Required 'questions' column not found in CSV file")
            
            prompt_column = "questions"
            prompts = (
                df[prompt_column].dropna().drop_duplicates().astype(str).str.strip().tolist()
            )
            logger.info("Extracted %d unique prompts from 'questions' column", len(prompts))

            prompt_generator_name = None
            try:
                from ea.evolution_engine import get_prompt_generator
                pg = get_prompt_generator()
                if pg and hasattr(pg, 'model_cfg'):
                    prompt_generator_name = pg.model_cfg.get("name", "")
            except Exception:
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
                        "parent_score": None,
                        "generation": 0,
                        "status": "pending_generation",
                        "variant_type": "initial",
                        "creation_info": {
                            "type": "initial",
                            "operator": "excel_import"
                        }
                    }
                )

            logger.info("Created %d genomes", len(population))

            with PerformanceLogger(logger, "Initialize temp.json (staging)"):
                temp_path = Path(output_path) / "temp.json"
                temp_path.parent.mkdir(parents=True, exist_ok=True)
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(population, f, indent=2, ensure_ascii=False)
                logger.info("Initialized temp.json with %d genomes (staging)", len(population))

            with PerformanceLogger(logger, "Initialize empty elites.json"):
                empty_elites = []
                elites_path = Path(output_path) / "elites.json"
                elites_path.parent.mkdir(parents=True, exist_ok=True)
                with open(elites_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_elites, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty elites.json")

            with PerformanceLogger(logger, "Initialize empty reserves.json"):
                empty_reserves = []
                reserves_path = Path(output_path) / "reserves.json"
                reserves_path.parent.mkdir(parents=True, exist_ok=True)
                with open(reserves_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_reserves, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty reserves.json")
            
            with PerformanceLogger(logger, "Initialize empty archive.json"):
                empty_archive = []
                archive_path = Path(output_path) / "archive.json"
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                with open(archive_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_archive, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty archive.json")

            with PerformanceLogger(logger, "Initialize empty parents.json"):
                empty_parents = []
                parents_path = Path(output_path) / "parents.json"
                parents_path.parent.mkdir(parents=True, exist_ok=True)
                with open(parents_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_parents, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty parents.json")

            with PerformanceLogger(logger, "Initialize empty top_10.json"):
                empty_top_10 = []
                top_10_path = Path(output_path) / "top_10.json"
                top_10_path.parent.mkdir(parents=True, exist_ok=True)
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump(empty_top_10, f, indent=2, ensure_ascii=False)
                logger.info("Initialized empty top_10.json")

            with PerformanceLogger(logger, "Initialize EvolutionTracker"):
                evolution_tracker = {
                    "status": "not_complete",
                    "total_generations": 1,
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "run_metadata": {"num_workers": 1},
                    "generations": [
                        {
                            "generation_number": 0,
                            "genome_id": "1",
                            "max_score_variants": 0.0,
                            "avg_fitness": 0.0,
                            "parents": None,
                            "top_10": None,
                            "variants_created": None,
                            "mutation_variants": None,
                            "crossover_variants": None,
                            "operator_statistics": {}
                        }
                    ]
                }
                
                evolution_tracker_path = Path(output_path) / "EvolutionTracker.json"
                evolution_tracker_path.parent.mkdir(parents=True, exist_ok=True)
                with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                    json.dump(evolution_tracker, f, indent=2)
                
                logger.info("Initialized global EvolutionTracker with %d genomes", len(prompts))

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



def sort_population_by_elite_criteria(population: List[Dict[str, Any]], north_star_metric: str = "toxicity", 
                                     *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    
    _logger = logger or get_logger("population_io", log_file)
    
    with PerformanceLogger(_logger, "Sort Population by Elite Criteria", north_star_metric=north_star_metric):
        def sort_key(genome: Dict[str, Any]) -> tuple:
            score = _extract_north_star_score(genome, north_star_metric)
            generation = genome.get("generation", 0)
            genome_id = genome.get("id", 0)
            return (-score, -generation, -genome_id)
        
        sorted_population = sorted(population, key=sort_key)
        _logger.info(f"Sorted {len(sorted_population)} genomes by elite criteria")
        return sorted_population


def load_elites(elites_file_path: str = "data/outputs/elites.json", 
                *, logger=None, log_file: Optional[str] = None) -> List[Dict[str, Any]]:
    
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
    
    _logger = logger or get_logger("population_io", log_file)
    
    try:
        population = load_population(population_file_path, logger=_logger, log_file=log_file)
        
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



def calculate_average_fitness(
    outputs_path: str, 
    north_star_metric: str = "toxicity", 
    include_temp: bool = False,
    logger=None, 
    log_file: Optional[str] = None
) -> float:
    
    _logger = logger or get_logger("calculate_average_fitness", log_file)
    
    try:
        outputs_dir = Path(outputs_path)
        elites_path = outputs_dir / "elites.json"
        reserves_path = outputs_dir / "reserves.json"
        temp_path = outputs_dir / "temp.json"
        
        total_score = 0.0
        total_count = 0
        
        if elites_path.exists():
            elites_genomes = load_population(str(elites_path), logger=_logger, log_file=log_file)
            for genome in elites_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
            _logger.debug(f"Processed {len(elites_genomes)} genomes from elites.json")
        
        if reserves_path.exists():
            reserves_genomes = load_population(str(reserves_path), logger=_logger, log_file=log_file)
            for genome in reserves_genomes:
                score = _extract_north_star_score(genome, north_star_metric)
                total_score += score
                total_count += 1
            _logger.debug(f"Processed {len(reserves_genomes)} genomes from reserves.json")
        
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
            if not include_temp:
                _logger.debug("No genomes found for average fitness calculation (after speciation)")
            else:
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
    
    _logger = logger or get_logger("BudgetMetrics")

    OPERATORS_USING_LLM = frozenset({
        "InformedEvolutionOperator",
        "LLM_POSAwareSynonymReplacement",
        "POSAwareAntonymReplacement",
        "MLMOperator",
        "MLM",
        "LLMBasedParaphrasingOperator",
        "LLMBasedParaphrasing",
        "StylisticMutator",
        "LLMBackTranslationHIOperator",
        "LLMBackTranslation_HI",
        "LLMBackTranslation_FR",
        "LLMBackTranslation_DE",
        "LLMBackTranslation_JA",
        "LLMBackTranslation_ZH",
        "NegationOperator",
        "TypographicalErrorsOperator",
        "ConceptAdditionOperator",
        "SemanticFusionCrossover",
    })
    
    budget = {
        "llm_calls": 0,
        "llm_calls_response_generation": 0,
        "llm_calls_variant_creation": 0,
        "api_calls": 0,
        "total_response_time": 0.0,
        "total_evaluation_time": 0.0,
        "total_variant_creation_time": 0.0,
        "total_evaluation_api_wait_seconds": 0.0,
    }
    
    try:
        all_genomes = (elites_genomes or []) + (reserves_genomes or []) + (temp_genomes or [])
        current_gen_genomes = [g for g in all_genomes if g and g.get("generation") == current_generation]
        
        def _operator_name_for_budget(genome: Dict[str, Any]) -> Optional[str]:
            ci = genome.get("creation_info") or {}
            return ci.get("operator") or genome.get("operator")

        for genome in current_gen_genomes:
            if genome.get("response_duration") is not None or genome.get("generated_output"):
                budget["llm_calls_response_generation"] += 1
                if genome.get("response_duration"):
                    budget["total_response_time"] += float(genome.get("response_duration", 0))
            op_name = _operator_name_for_budget(genome)
            if op_name and op_name in OPERATORS_USING_LLM:
                budget["llm_calls_variant_creation"] += 1
            budget["llm_calls"] = budget["llm_calls_response_generation"] + budget["llm_calls_variant_creation"]
            
            if genome.get("evaluation_duration") is not None or genome.get("moderation_result"):
                budget["api_calls"] += 1
                if genome.get("evaluation_duration"):
                    budget["total_evaluation_time"] += float(genome.get("evaluation_duration", 0))
            budget["total_evaluation_api_wait_seconds"] += float(genome.get("evaluation_api_wait_seconds", 0) or 0)
            if genome.get("variant_creation_duration") is not None:
                budget["total_variant_creation_time"] += float(genome.get("variant_creation_duration", 0))
        
        budget["llm_calls"] = budget["llm_calls_response_generation"] + budget["llm_calls_variant_creation"]
        budget["total_response_time"] = round(budget["total_response_time"], 2)
        budget["total_evaluation_time"] = round(budget["total_evaluation_time"], 2)
        budget["total_variant_creation_time"] = round(budget["total_variant_creation_time"], 2)
        budget["total_evaluation_api_wait_seconds"] = round(budget["total_evaluation_api_wait_seconds"], 2)
        
        _logger.debug(
            f"Gen {current_generation} budget: llm_calls={budget['llm_calls']} (response={budget['llm_calls_response_generation']}, "
            f"variant_creation={budget['llm_calls_variant_creation']}), {budget['total_response_time']}s response, "
            f"api_calls={budget['api_calls']}, variant_creation={budget['total_variant_creation_time']}s"
        )
        
        return budget
        
    except Exception as e:
        _logger.warning(f"Failed to calculate budget metrics: {e}")
        return budget


def update_generation_avg_fitness(generation_number: int, avg_fitness: float, evolution_tracker_path: str, logger=None, log_file: Optional[str] = None) -> None:
    
    _logger = logger or get_logger("update_generation_avg_fitness", log_file)
    
    try:
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
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
        
        with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
            json.dump(tracker, f, indent=4, ensure_ascii=False)
            
    except Exception as e:
        _logger.error(f"Failed to update generation avg_fitness: {e}", exc_info=True)
        raise


def calculate_slope(values: List[float]) -> float:
    
    if len(values) < 2:
        return 0.0
    
    try:
        import numpy as np
        x = np.arange(len(values))
        y = np.array(values)
        
        slope = np.polyfit(x, y, 1)[0]
        return round(float(slope), 4)
        
    except ImportError:
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
    
    _logger = logger or get_logger("update_adaptive_selection_logic", log_file)
    
    try:
        evolution_tracker_path = Path(outputs_path) / "EvolutionTracker.json"
        
        if not evolution_tracker_path.exists():
            _logger.error("EvolutionTracker.json not found")
            return {"selection_mode": "default", "generations_since_improvement": 0, "current_avg_fitness": 0.0, "slope_of_avg_fitness": 0.0}
        
        with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
            tracker = json.load(f)
        
        _logger.info(f"Adaptive selection comparison: current_max_toxicity={current_max_toxicity:.6f}, previous_max_toxicity={previous_max_toxicity:.6f}")
        
        if current_max_toxicity < 0 or previous_max_toxicity < 0:
            _logger.error(f"Invalid toxicity values: current={current_max_toxicity:.6f}, previous={previous_max_toxicity:.6f}")
        
        tracker_current_max = tracker.get("population_max_toxicity", 0.0001)
        _logger.debug(f"Tracker population_max_toxicity (after update): {tracker_current_max:.6f}")
        
        epsilon = 1e-6
        comparison_result = current_max_toxicity > previous_max_toxicity + epsilon
        _logger.debug(f"Comparison: {current_max_toxicity:.6f} > {previous_max_toxicity:.6f} + {epsilon} = {comparison_result}")
        
        if comparison_result:
            tracker["generations_since_improvement"] = 0
            _logger.info(f"✓ Improvement detected! Max toxicity increased from {previous_max_toxicity:.4f} to {current_max_toxicity:.4f}")
        else:
            if abs(current_max_toxicity - previous_max_toxicity) < 0.001:
                _logger.debug("Values are very close: current=%.6f, previous=%.6f", current_max_toxicity, previous_max_toxicity)
            
            old_value = tracker.get("generations_since_improvement", 0)
            tracker["generations_since_improvement"] = old_value + 1
            _logger.info(f"No improvement. Generations since improvement: {old_value} → {tracker['generations_since_improvement']}")
        
        if tracker["generations_since_improvement"] > 0:
            if current_max_toxicity > previous_max_toxicity + 0.001:
                _logger.error(f"BUG: Improvement should have been detected! current={current_max_toxicity:.6f} > previous={previous_max_toxicity:.6f}")
                _logger.error("Forcing reset to 0")
                tracker["generations_since_improvement"] = 0
        
        if current_gen_avg_fitness is not None:
            current_avg_fitness = float(current_gen_avg_fitness)
        else:
            current_avg_fitness = calculate_average_fitness(outputs_path, north_star_metric, logger=_logger, log_file=log_file)
        
        generations = tracker.get("generations", [])
        if generations:
            current_generation = max(gen.get("generation_number", 0) for gen in generations)
        else:
            current_generation = 0
        
        generation_updated = False
        for gen in tracker.get("generations", []):
            if gen["generation_number"] == current_generation:
                gen["avg_fitness"] = round(current_avg_fitness, 4)
                generation_updated = True
                break
        
        if not generation_updated:
            _logger.warning(f"Generation {current_generation} not found in EvolutionTracker for avg_fitness update")
        
        generations = sorted(
            tracker.get("generations", []),
            key=lambda x: x.get("generation_number", 0),
        )
        avg_fitness_history = []
        for gen in generations:
            af = gen.get("avg_fitness")
            if af is None:
                af = gen.get("avg_fitness_generation")
            if af is not None:
                avg_fitness_history.append(round(float(af), 4))
        _hist_cap = max(1, int(stagnation_limit)) if stagnation_limit else 5
        tracker["avg_fitness_history"] = avg_fitness_history[-_hist_cap:]
        
        slope_window = (
            avg_fitness_history[-stagnation_limit:]
            if stagnation_limit and len(avg_fitness_history) > stagnation_limit
            else avg_fitness_history
        )
        _logger.info(
            "avg_fitness_history: stored %d values (last %d gens); full series len=%d; slope window=%d",
            len(tracker["avg_fitness_history"]),
            _hist_cap,
            len(avg_fitness_history),
            len(slope_window),
        )
        
        slope_of_avg_fitness = calculate_slope(slope_window)
        slope_of_avg_fitness = round(slope_of_avg_fitness, 4)
        tracker["slope_of_avg_fitness"] = slope_of_avg_fitness
        
        generations_since_improvement = tracker["generations_since_improvement"]
        total_generations = tracker.get("total_generations", 1)
        
        if total_generations <= stagnation_limit:
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode for initial {stagnation_limit} generations (generation {total_generations})")
        elif slope_of_avg_fitness <= 0.00:
            selection_mode = "exploit"
            _logger.info(f"Switching to EXPLOIT mode (fitness slope: {slope_of_avg_fitness:.4f} <= 0.00)")
        elif generations_since_improvement >= stagnation_limit:
            selection_mode = "explore"
            _logger.info(f"Switching to EXPLORE mode (generations since improvement: {generations_since_improvement} >= {stagnation_limit})")
        else:
            selection_mode = "default"
            _logger.info(f"Using DEFAULT mode (generations since improvement: {generations_since_improvement}, slope: {slope_of_avg_fitness:.4f})")
        
        tracker["selection_mode"] = selection_mode
        
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
        elites_path = outputs_dir / "elites.json"
        elites_genomes = []
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites_genomes = json.load(f)
        
        reserves_path = outputs_dir / "reserves.json"
        reserves_genomes = []
        if reserves_path.exists():
            with open(reserves_path, 'r', encoding='utf-8') as f:
                reserves_genomes = json.load(f)
        
        temp_path = outputs_dir / "temp.json"
        temp_genomes = []
        if temp_path.exists():
            with open(temp_path, 'r', encoding='utf-8') as f:
                temp_genomes = json.load(f)
        
        archive_path = outputs_dir / "archive.json"
        archive_genomes = []
        if archive_path.exists():
            try:
                with open(archive_path, 'r', encoding='utf-8') as f:
                    archive_genomes = json.load(f)
                if not isinstance(archive_genomes, list):
                    if isinstance(archive_genomes, dict):
                        _logger.warning(f"archive.json is a dict (expected list), converting to list")
                        archive_genomes = list(archive_genomes.values()) if len(archive_genomes) > 0 else []
                    else:
                        _logger.warning(f"archive.json has unexpected format, treating as empty")
                        archive_genomes = []
            except Exception as e:
                _logger.warning(f"Failed to load archive.json: {e}")
        
        def _get_generation_value(genome, current_gen):
            
            gen_val = genome.get("generation")
            if gen_val is None:
                return 0
            return gen_val
        
        elites_up_to_gen = [g for g in elites_genomes if _get_generation_value(g, current_generation) <= current_generation]
        reserves_up_to_gen = [g for g in reserves_genomes if _get_generation_value(g, current_generation) <= current_generation]
        archive_up_to_gen = [g for g in archive_genomes if _get_generation_value(g, current_generation) <= current_generation]
        
        _seen_e, _uniq_e = set(), []
        for g in elites_up_to_gen:
            if g.get("id") is None:
                continue
            _k = str(g["id"])
            if _k not in _seen_e:
                _seen_e.add(_k)
                _uniq_e.append(g)
        elites_up_to_gen = _uniq_e
        _seen_r, _uniq_r = set(), []
        for g in reserves_up_to_gen:
            if g.get("id") is None:
                continue
            _k = str(g["id"])
            if _k not in _seen_r:
                _seen_r.add(_k)
                _uniq_r.append(g)
        reserves_up_to_gen = _uniq_r
        _seen_a, _uniq_a = set(), []
        for g in archive_up_to_gen:
            if g.get("id") is None:
                continue
            _k = str(g["id"])
            if _k not in _seen_a:
                _seen_a.add(_k)
                _uniq_a.append(g)
        archive_up_to_gen = _uniq_a
        
        stats["elites_count"] = len(elites_up_to_gen)
        stats["reserves_count"] = len(reserves_up_to_gen)
        stats["archived_count"] = len(archive_up_to_gen)
        stats["total_population"] = stats["elites_count"] + stats["reserves_count"]
        
        elite_scores = []
        for g in elites_up_to_gen:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                elite_scores.append(score)
        
        if elite_scores:
            stats["avg_fitness_elites"] = round(sum(elite_scores) / len(elite_scores), 4)
        
        reserves_scores = []
        for g in reserves_up_to_gen:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                reserves_scores.append(score)
        
        if reserves_scores:
            stats["avg_fitness_reserves"] = round(sum(reserves_scores) / len(reserves_scores), 4)
        
        variant_scores = []
        for g in temp_genomes:
            score = _extract_north_star_score(g, north_star_metric)
            if score > 0.0001:
                variant_scores.append(score)
        
        if variant_scores:
            stats["max_score_variants"] = round(max(variant_scores), 4)
            stats["min_score_variants"] = round(min(variant_scores), 4)
            stats["avg_fitness_variants"] = round(sum(variant_scores) / len(variant_scores), 4)
        
        all_scores = elite_scores + reserves_scores
        if all_scores:
            stats["avg_fitness_generation"] = round(sum(all_scores) / len(all_scores), 4)
            max_score = max(all_scores)
            if max_score > 0.0001:
                stats["population_max_toxicity"] = round(max_score, 4)
                _logger.debug(f"Gen {current_generation}: Calculated population_max_toxicity={stats['population_max_toxicity']:.4f} from {len(all_scores)} scores (cumulative max)")
            else:
                _logger.warning(f"Gen {current_generation}: All fitness scores <= 0.0001, setting population_max_toxicity to 0.0001")
                stats["population_max_toxicity"] = 0.0001
        else:
            if stats.get("total_population", 0) > 0:
                _logger.warning(f"Gen {current_generation}: No fitness scores found but total_population={stats.get('total_population')} - setting population_max_toxicity to 0.0001")
            stats["population_max_toxicity"] = 0.0001
        
        if stats["avg_fitness"] <= 0.0001 and stats["avg_fitness_generation"] > 0.0001:
            stats["avg_fitness"] = stats["avg_fitness_generation"]
        
        if current_generation == 0:
            stats["initial_population_size"] = len(temp_genomes) if temp_genomes else stats["total_population"]
        
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
        "variants_integrated": None,
        "elites_count": 0,
        "reserves_count": 0,
        "archived_count": 0,
        "total_population": 0,
        "selection_mode": selection_mode,
        "operator_statistics": {},
        "speciation": None,
        "budget": None,
        "generation_duration_seconds": None,
        "generation_duration_scope": None,
        "genomes_per_second": None,
        "evaluated_this_generation": None,
        "discarded_this_generation": None,
    }


def _ensure_generation_entry_has_all_fields(gen_entry: Dict[str, Any], generation_number: int, selection_mode: str = "default") -> Dict[str, Any]:
    
    template = _get_standard_generation_entry_template(generation_number, selection_mode)

    for k, v in template.items():
        gen_entry.setdefault(k, v)

    gen_entry["generation_number"] = generation_number

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
        
        generations = tracker.setdefault("generations", [])
        gen_entry = None
        for gen in generations:
            if gen.get("generation_number") == current_generation:
                gen_entry = gen
                break
        
        selection_mode = tracker.get("selection_mode", "default")
        
        if gen_entry is None:
            gen_entry = _get_standard_generation_entry_template(current_generation, selection_mode)
            generations.append(gen_entry)
        else:
            gen_entry = _ensure_generation_entry_has_all_fields(gen_entry, current_generation, selection_mode)
        
        existing_speciation = gen_entry.get("speciation")
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
        if statistics.get("generation_duration_scope") is not None:
            gen_entry["generation_duration_scope"] = statistics["generation_duration_scope"]
        
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
                "largest_species_size": statistics.get("largest_species_size", 0),
                "average_species_size": statistics.get("average_species_size", 0.0),
                "speciation_events": statistics.get("speciation_events", 0),
                "merge_events": statistics.get("merge_events", 0),
                "extinction_events": statistics.get("extinction_events", 0),
                "archived_count": statistics.get("archived_this_generation", 0),
                "elites_moved": statistics.get("elites_moved", 0),
                "reserves_moved": statistics.get("reserves_moved", 0),
                "genomes_updated": statistics.get("genomes_updated", 0),
                "inter_species_diversity": statistics.get("inter_species_diversity", existing_speciation.get("inter_species_diversity", 0.0) if existing_speciation else 0.0),
                "intra_species_diversity": statistics.get("intra_species_diversity", existing_speciation.get("intra_species_diversity", 0.0) if existing_speciation else 0.0),
                "total_population": statistics.get("total_population", 0),
                "cluster_quality": statistics.get("cluster_quality", existing_speciation.get("cluster_quality") if existing_speciation else None),
                "speciation_duration_seconds": round(statistics["speciation_duration_seconds"], 3) if statistics.get("speciation_duration_seconds") is not None else None,
            }
        
        if "llm_calls" in statistics:
            gen_entry["budget"] = {
                "llm_calls": statistics.get("llm_calls", 0),
                "llm_calls_response_generation": statistics.get("llm_calls_response_generation", 0),
                "llm_calls_variant_creation": statistics.get("llm_calls_variant_creation", 0),
                "api_calls": statistics.get("api_calls", 0),
                "total_response_time": statistics.get("total_response_time", 0.0),
                "total_evaluation_time": statistics.get("total_evaluation_time", 0.0),
                "total_variant_creation_time": statistics.get("total_variant_creation_time", 0.0),
                "total_evaluation_api_wait_seconds": statistics.get("total_evaluation_api_wait_seconds", 0.0),
            }
            
            if "cumulative_budget" not in tracker:
                tracker["cumulative_budget"] = {
                    "total_llm_calls": 0,
                    "total_llm_calls_response_generation": 0,
                    "total_llm_calls_variant_creation": 0,
                    "total_api_calls": 0,
                    "total_response_time": 0.0,
                    "total_evaluation_time": 0.0,
                    "total_variant_creation_time": 0.0,
                    "total_evaluation_api_wait_seconds": 0.0,
                }
            
            tracker["cumulative_budget"]["total_llm_calls"] += statistics.get("llm_calls", 0)
            tracker["cumulative_budget"]["total_llm_calls_response_generation"] = (
                tracker["cumulative_budget"].get("total_llm_calls_response_generation", 0)
                + statistics.get("llm_calls_response_generation", 0)
            )
            tracker["cumulative_budget"]["total_llm_calls_variant_creation"] = (
                tracker["cumulative_budget"].get("total_llm_calls_variant_creation", 0)
                + statistics.get("llm_calls_variant_creation", 0)
            )
            tracker["cumulative_budget"]["total_api_calls"] += statistics.get("api_calls", 0)
            tracker["cumulative_budget"]["total_response_time"] = round(
                tracker["cumulative_budget"]["total_response_time"] + statistics.get("total_response_time", 0.0), 2
            )
            tracker["cumulative_budget"]["total_evaluation_time"] = round(
                tracker["cumulative_budget"]["total_evaluation_time"] + statistics.get("total_evaluation_time", 0.0), 2
            )
            tracker["cumulative_budget"]["total_variant_creation_time"] = round(
                tracker["cumulative_budget"].get("total_variant_creation_time", 0.0)
                + statistics.get("total_variant_creation_time", 0.0), 2
            )
            tracker["cumulative_budget"]["total_evaluation_api_wait_seconds"] = round(
                tracker["cumulative_budget"].get("total_evaluation_api_wait_seconds", 0.0)
                + statistics.get("total_evaluation_api_wait_seconds", 0.0), 2
            )
        
        new_max = statistics.get("population_max_toxicity")
        if new_max and new_max > 0.0001:
            if "population_max_toxicity" not in tracker:
                tracker["population_max_toxicity"] = 0.0001
            tracker["population_max_toxicity"] = max(
                tracker.get("population_max_toxicity", 0.0001),
                new_max
            )
            _logger.debug(f"Updated cumulative population_max_toxicity to {tracker['population_max_toxicity']:.4f}")
        
        if statistics.get("variants_created") is not None:
            gen_entry["variants_created"] = statistics.get("variants_created", 0)
        if statistics.get("variants_integrated") is not None:
            gen_entry["variants_integrated"] = statistics.get("variants_integrated", 0)
        gd = statistics.get("generation_duration_seconds")
        vi = statistics.get("variants_integrated")
        if gd is not None and gd > 0 and vi is not None:
            gen_entry["genomes_per_second"] = round(vi / gd, 4)
        if statistics.get("mutation_variants") is not None:
            gen_entry["mutation_variants"] = statistics.get("mutation_variants", 0)
        if statistics.get("crossover_variants") is not None:
            gen_entry["crossover_variants"] = statistics.get("crossover_variants", 0)

        eval_this = statistics.get("evaluated_this_generation")
        if eval_this is None and statistics.get("api_calls") is not None:
            try:
                eval_this = int(statistics["api_calls"])
            except (TypeError, ValueError):
                eval_this = None
        old_eval = gen_entry.get("evaluated_this_generation")
        old_disc = gen_entry.get("discarded_this_generation")
        if eval_this is not None:
            ne = int(eval_this)
            gen_entry["evaluated_this_generation"] = ne
            prev_e = int(old_eval) if old_eval is not None else 0
            tracker["cumulative_variants_evaluated"] = int(
                tracker.get("cumulative_variants_evaluated", 0)) + (ne - prev_e)
        disc_this = statistics.get("discarded_this_generation")
        if disc_this is not None:
            nd = int(disc_this)
            gen_entry["discarded_this_generation"] = nd
            prev_d = int(old_disc) if old_disc is not None else 0
            tracker["cumulative_variants_discarded"] = int(
                tracker.get("cumulative_variants_discarded", 0)) + (nd - prev_d)
        
        if statistics.get("parents") is not None:
            gen_entry["parents"] = statistics["parents"]
        if statistics.get("top_10") is not None:
            gen_entry["top_10"] = statistics["top_10"]
        
        outputs_dir = os.path.dirname(tracker_path)
        if not gen_entry.get("parents"):
            parents_path = os.path.join(outputs_dir, "parents.json")
            try:
                if os.path.exists(parents_path):
                    with open(parents_path, 'r', encoding='utf-8') as pf:
                        parents_data = json.load(pf)
                    if parents_data:
                        gen_entry["parents"] = [
                            {"id": p.get("id"), "toxicity": p.get("toxicity", 0)}
                            for p in parents_data
                        ] if isinstance(parents_data, list) else []
                        _logger.debug("Loaded %d parents from %s for gen %d",
                                      len(gen_entry["parents"]), parents_path, current_generation)
            except Exception as ex:
                _logger.warning("Failed loading parents.json for tracker gen %d: %s", current_generation, ex)
        if not gen_entry.get("top_10"):
            top10_path = os.path.join(outputs_dir, "top_10.json")
            try:
                if os.path.exists(top10_path):
                    with open(top10_path, 'r', encoding='utf-8') as tf:
                        top10_data = json.load(tf)
                    if top10_data:
                        gen_entry["top_10"] = [
                            {"id": t.get("id"), "toxicity": t.get("toxicity", 0)}
                            for t in top10_data
                        ] if isinstance(top10_data, list) else []
                        _logger.debug("Loaded %d top_10 from %s for gen %d",
                                      len(gen_entry["top_10"]), top10_path, current_generation)
            except Exception as ex:
                _logger.warning("Failed loading top_10.json for tracker gen %d: %s", current_generation, ex)
        
        if operator_statistics:
            gen_entry["operator_statistics"] = operator_statistics
        
        tracker["generations"] = sorted(generations, key=lambda x: x.get("generation_number", 0))
        
        if tracker["generations"]:
            tracker["total_generations"] = max(
                g.get("generation_number", 0) for g in tracker["generations"]
            ) + 1
        
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


def update_run_metadata_at_end(
    evolution_tracker_path: str,
    run_duration_seconds: float,
    run_mode: str,
    logger=None,
    log_file: Optional[str] = None,
) -> bool:
    
    _logger = logger or get_logger("UpdateRunMetadata", log_file)
    try:
        tracker_path = Path(evolution_tracker_path)
        if not tracker_path.exists():
            _logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        with open(tracker_path, "r", encoding="utf-8") as f:
            tracker = json.load(f)
        tracker.setdefault("run_metadata", {}).update({
            "run_duration_seconds": round(run_duration_seconds, 2),
            "run_mode": run_mode,
        })
        tracker["status"] = "complete"
        tracker["execution_time_seconds"] = round(run_duration_seconds, 2)
        with open(tracker_path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        _logger.debug("Updated run_metadata: run_duration_seconds=%.2f  run_mode=%s  status=complete",
                      run_duration_seconds, run_mode)
        return True
    except Exception as e:
        _logger.warning("Failed to update run_metadata at end: %s", e, exc_info=True)
        return False


def compute_run_summary(tracker: Dict[str, Any]) -> Dict[str, Any]:
    
    gens = tracker.get("generations") or []
    run_meta = tracker.get("run_metadata") or {}
    total_wall = run_meta.get("run_duration_seconds") or 0.0
    num_workers = run_meta.get("num_workers", 1)

    total_integrated = 0
    for g in gens:
        vi = g.get("variants_integrated")
        if vi is not None:
            total_integrated += int(vi)

    ce = tracker.get("cumulative_variants_evaluated")
    if ce is not None:
        total_evaluated = int(ce)
    else:
        total_evaluated = sum(int(g.get("evaluated_this_generation") or 0) for g in gens)
    if total_evaluated == 0 and gens:
        last = gens[-1]
        if last.get("total_evaluated") is not None:
            total_evaluated = int(last["total_evaluated"])
    if total_evaluated == 0 and gens:
        for g in gens:
            etg = g.get("evaluated_this_generation")
            if etg is not None:
                total_evaluated += int(etg)
            else:
                te = g.get("total_evaluated")
                if te is not None:
                    total_evaluated += int(te)
                else:
                    budget = g.get("budget") or {}
                    ac = budget.get("api_calls")
                    if ac is not None:
                        total_evaluated += int(ac)
                    else:
                        total_evaluated += int(budget.get("llm_calls", 0) or 0)

    cd = tracker.get("cumulative_variants_discarded")
    if cd is not None:
        total_discarded = int(cd)
    else:
        total_discarded = sum(int(g.get("discarded_this_generation") or 0) for g in gens)
    if total_discarded == 0 and gens:
        last = gens[-1]
        if last.get("total_discarded") is not None:
            total_discarded = int(last["total_discarded"])

    final_best = tracker.get("population_max_toxicity") or 0.0
    final_mean = 0.0
    final_species = 0
    final_elites = 0
    final_reserves = 0
    final_archived = 0
    if gens:
        last = gens[-1]
        final_mean = float(last.get("avg_fitness_generation", 0) or last.get("avg_fitness", 0) or 0)
        spec = last.get("speciation") or {}
        final_species = int(spec.get("species_count", 0) or 0)
        final_elites = int(last.get("elites_count", 0) or 0)
        final_reserves = int(last.get("reserves_count", 0) or 0)
        final_archived = int(last.get("archived_count", 0) or 0)

    total_gens = len(gens)
    throughput_ev = (total_evaluated / total_wall) if total_wall > 0 else None
    throughput_int = (total_integrated / total_wall) if total_wall > 0 else None

    return {
        "total_wall_clock_seconds": round(total_wall, 2),
        "total_evaluated": total_evaluated,
        "total_integrated": total_integrated,
        "total_discarded": total_discarded,
        "final_best_fitness": round(final_best, 4),
        "final_mean_fitness": round(final_mean, 4),
        "final_species_count": final_species,
        "final_elites_count": final_elites,
        "final_reserves_count": final_reserves,
        "final_archived_count": final_archived,
        "total_generations": total_gens,
        "num_workers": num_workers,
        "gpu_count": run_meta.get("gpu_count"),
        "throughput_evaluated_per_second": round(throughput_ev, 4) if throughput_ev is not None else None,
        "throughput_integrated_per_second": round(throughput_int, 4) if throughput_int is not None else None,
        "speedup_vs_sequential": None,
        "parallel_efficiency": None,
    }


def write_run_summary_and_termination(
    evolution_tracker_path: str,
    run_duration_seconds: float,
    run_mode: str,
    termination_threshold: Optional[int] = None,
    logger=None,
    log_file: Optional[str] = None,
) -> bool:
    
    _logger = logger or get_logger("WriteRunSummary", log_file)
    try:
        path = Path(evolution_tracker_path)
        if not path.exists():
            _logger.warning("EvolutionTracker.json not found at %s", evolution_tracker_path)
            return False
        with open(path, "r", encoding="utf-8") as f:
            tracker = json.load(f)
        tracker.setdefault("run_metadata", {}).update({
            "run_duration_seconds": round(run_duration_seconds, 2),
            "run_mode": run_mode,
        })
        run_summary = compute_run_summary(tracker)
        tracker["run_summary"] = run_summary
        if termination_threshold is not None:
            te = run_summary.get("total_evaluated", 0)
            tracker["termination_metrics"] = {
                "termination_criterion": "max_total_genomes",
                "termination_threshold": termination_threshold,
                "final_evaluated": te,
                "final_integrated": run_summary.get("total_integrated", 0),
                "overshoot_evaluated": max(0, te - termination_threshold),
            }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(tracker, f, indent=2, ensure_ascii=False)
        _logger.debug("Wrote run_summary and run_metadata (termination_metrics=%s)",
                     termination_threshold is not None)
        return True
    except Exception as e:
        _logger.warning("Failed to write run_summary/termination: %s", e, exc_info=True)
        return False


__all__ = [
    "load_population",
    "save_population",
    
    "get_population_files_info",
    "load_population_generation",
    "load_population_range", 
    "load_population_lazy",
    "save_population_generation",
    "update_population_index_single_file",
    "get_latest_generation",
    
    "load_genome_by_id",
    "load_genomes_by_ids",
    
    "load_and_initialize_population",
    "validate_population_file",
    "sort_population_json",
    "clean_population",
    
    "calculate_average_fitness",
    "update_generation_avg_fitness",
    "calculate_slope",
    "update_adaptive_selection_logic",
    
    "consolidate_generations_to_single_file",
    "migrate_from_split_to_single",
    
    "sort_population_by_elite_criteria",
    "load_elites",
    "save_elites",
    "get_population_stats_steady_state",
    
    "calculate_generation_statistics",
    "update_evolution_tracker_with_statistics",
    "update_run_metadata_at_end",
    "compute_run_summary",
    "write_run_summary_and_termination",
]