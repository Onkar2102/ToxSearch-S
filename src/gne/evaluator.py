

import os
import re
import json
import time
import hashlib
import threading
import sys
from itertools import islice
from typing import List, Dict, Optional, Any
from dotenv import load_dotenv
from utils import get_custom_logging, get_population_io
from utils.evaluator_profiles import (
    resolve_evaluator,
    set_active_evaluator,
    moderation_methods_to_backend_list,
)
from utils.population_io import _extract_north_star_score
from concurrent.futures import ThreadPoolExecutor

load_dotenv(override=True)

get_logger, get_log_filename, _, _ = get_custom_logging()
logger = get_logger("evaluator", get_log_filename())

_moderation_cache = {}
_MAX_CACHE_SIZE = 5000
_cache_lock = threading.Lock()

_thread_pool = None


def _redact_url_api_key_query_param(msg: str) -> str:
    
    if "key=" not in msg:
        return msg
    return re.sub(
        r"key=([^&\s\"']+)",
        lambda m: (
            f"key=***...{m.group(1)[-4:]}" if len(m.group(1)) > 4 else "key=***"
        ),
        msg,
    )


def _get_thread_pool():
    
    global _thread_pool
    if _thread_pool is None:
        _thread_pool = ThreadPoolExecutor(max_workers=8)
    return _thread_pool

def _get_text_hash(text: str, api_name: str = "") -> str:
    
    cache_key = f"{api_name}:{text}"
    return hashlib.md5(cache_key.encode('utf-8')).hexdigest()

def _get_cached_result(text: str, api_name: str = "") -> Optional[Dict]:
    
    text_hash = _get_text_hash(text, api_name)
    with _cache_lock:
        return _moderation_cache.get(text_hash)

def _cache_result(text: str, result: Dict, api_name: str = ""):
    
    text_hash = _get_text_hash(text, api_name)
    with _cache_lock:
        _moderation_cache[text_hash] = result
        if len(_moderation_cache) > _MAX_CACHE_SIZE:
            _cleanup_cache_if_needed()

def _cleanup_cache_if_needed():
    
    global _moderation_cache
    n = len(_moderation_cache)
    if n <= _MAX_CACHE_SIZE:
        return
    to_remove = list(islice(_moderation_cache.keys(), n // 5))
    for k in to_remove:
        del _moderation_cache[k]
    logger.info("Cleaned moderation cache: removed %d entries, cache size now: %d", len(to_remove), len(_moderation_cache))

class HybridModerationEvaluator:
    """Content moderation evaluator (Google Perspective or OpenAI omni-moderation)."""
    
    def __init__(self, log_file: Optional[str] = None, config_path: str = None,
                 api_keys: Optional[List[str]] = None,
                 evaluator: Optional[str] = None,
                 openai_model: str = "omni-moderation-latest"):
        
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("HybridModerationEvaluator", log_file)
        self.profile = set_active_evaluator(evaluator or "google")
        self.openai_model = openai_model
        self.logger.info(
            "Initializing moderation evaluator: backend=%s, north_star_default=%s",
            self.profile.name,
            self.profile.default_north_star,
        )

        import yaml
        from pathlib import Path
        if config_path is None:
            project_root = Path(__file__).resolve().parents[2]
            config_path = project_root / "config" / "RGConfig.yaml"
        else:
            config_path = Path(config_path)
            if not config_path.is_absolute():
                project_root = Path(__file__).resolve().parents[2]
                config_path = project_root / config_path
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        model_key = list(config.keys())[0] if config else "response_generator"
        self.model_cfg = config.get(model_key, {})
        self.logger.info("Model config loaded")

        self.google_available = False
        self.openai_available = False
        self._api_keys: List[str] = []
        self._active_key_index = 0
        self.google_client = None
        self.openai_client = None

        if self.profile.backend_key == "google":
            self._api_keys = self._resolve_api_keys(api_keys)
            self.google_available = len(self._api_keys) > 0
            if not self.google_available:
                error_msg = (
                    "No Perspective API key found.\n"
                    "Provide api_keys, set PERSPECTIVE_API_KEYS (comma-separated),\n"
                    "or set PERSPECTIVE_API_KEY as an environment variable."
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.logger.info("API Availability - Google: OK  (%d key(s))", len(self._api_keys))
        elif self.profile.backend_key == "openai":
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
            self.openai_available = bool(api_key)
            if not self.openai_available:
                error_msg = "OPENAI_API_KEY is required when --evaluator openai is selected."
                self.logger.error(error_msg)
                raise ValueError(error_msg)
            self.logger.info("API Availability - OpenAI: OK  (model=%s)", self.openai_model)

        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        
        self._initialize_clients()
        
        self.logger.debug("Moderation evaluator initialized successfully")

    @staticmethod
    def _resolve_api_keys(api_keys: Optional[List[str]] = None) -> List[str]:
        
        if api_keys:
            return [k.strip() for k in api_keys if k and k.strip()]

        multi = os.getenv("PERSPECTIVE_API_KEYS", "").strip()
        if multi:
            return [k.strip() for k in multi.split(",") if k.strip()]

        idx = 0
        indexed: List[str] = []
        while True:
            val = os.getenv(f"PERSPECTIVE_API_KEY_{idx}", "").strip()
            if not val:
                break
            indexed.append(val)
            idx += 1
        if indexed:
            return indexed

        single = os.getenv("PERSPECTIVE_API_KEY", "").strip()
        if single:
            if "," in single:
                return [k.strip() for k in single.split(",") if k.strip()]
            return [single]

        return []

    def select_key(self, index: int) -> None:
        
        if not self._api_keys:
            return
        clamped = index % len(self._api_keys)
        if clamped == self._active_key_index:
            return
        self._active_key_index = clamped
        self._initialize_clients()
        self.logger.debug("Switched to API key index %d", clamped)
    
    def _initialize_clients(self):
        
        try:
            if self.profile.backend_key == "google" and self.google_available:
                from googleapiclient import discovery
                api_key = self._api_keys[self._active_key_index]
                self.google_client = discovery.build(
                    "commentanalyzer",
                    "v1alpha1",
                    developerKey=api_key,
                    discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1"
                )
                self.logger.info("Google Perspective API client initialized (key index %d)",
                                 self._active_key_index)
            elif self.profile.backend_key == "openai" and self.openai_available:
                from openai import OpenAI
                self.openai_client = OpenAI(
                    api_key=os.environ["OPENAI_API_KEY"],
                    organization=os.getenv("OPENAI_ORG_ID") or None,
                    project=os.getenv("OPENAI_PROJECT_ID") or None,
                )
                self.logger.info("OpenAI moderation client initialized (model=%s)", self.openai_model)
                
        except Exception as e:
            self.logger.error(
                "Failed to initialize API clients: %s",
                _redact_url_api_key_query_param(str(e)),
            )
            raise
    
    def _evaluate_with_google(self, text: str, genome_id: str, max_retries: int = 10) -> Optional[Dict[str, Any]]:
        
        import time
        
        
        MAX_TEXT_BYTES = 20480
        text_bytes = text.encode('utf-8')
        
        if len(text_bytes) > MAX_TEXT_BYTES:
            self.logger.warning("Text for genome %s exceeds Perspective API limit (%d bytes). Truncating to %d bytes.", 
                              genome_id, len(text_bytes), MAX_TEXT_BYTES)
            text = text_bytes[:MAX_TEXT_BYTES].decode('utf-8', errors='ignore')
            
        attempt_durations = []
        api_wait_seconds = 0.0
        for attempt in range(max_retries + 1):
            attempt_start = time.time()
            try:
                analyze_request = {
                    'comment': {
                        'text': text
                    },
                    'languages': ['en'],
                    'requestedAttributes': {
                        'TOXICITY': {},
                        'SEVERE_TOXICITY': {},
                        'IDENTITY_ATTACK': {},
                        'INSULT': {},
                        'PROFANITY': {},
                        'THREAT': {},
                        'SEXUALLY_EXPLICIT': {},
                        'FLIRTATION': {}
                    },
                    'doNotStore': True
                }
                
                response = self.google_client.comments().analyze(body=analyze_request).execute()
            
                attribute_scores = response.get("attributeScores", {})
                raw_categories = {}
                for category, data in attribute_scores.items():
                    if "summaryScore" in data:
                        raw_categories[category.lower()] = data["summaryScore"]["value"]
                
                normalized_scores = self._normalize_scores(raw_categories)
                
                result = {
                    "scores": normalized_scores,
                    "api_version": "google",
                    "model": "perspective-api"
                }
                
                _cache_result(text, result, "google")
                attempt_durations.append(round(time.time() - attempt_start, 4))
                return result, {"retries": attempt, "attempt_durations": attempt_durations, "api_wait_seconds": round(api_wait_seconds, 4)}
                
            except Exception as e:
                attempt_durations.append(round(time.time() - attempt_start, 4))
                is_retriable = self._is_retriable_error(e)
                safe_err = _redact_url_api_key_query_param(str(e))

                if attempt < max_retries and is_retriable:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        "Google API evaluation failed for genome %s (attempt %d/%d): %s. Retrying in %ds...",
                        genome_id,
                        attempt + 1,
                        max_retries + 1,
                        safe_err,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    api_wait_seconds += wait_time
                    continue
                else:
                    self.logger.error(
                        "Google API evaluation failed for genome %s after %d attempts: %s",
                        genome_id,
                        attempt + 1,
                        safe_err,
                    )
                    return None, {"retries": attempt + 1, "attempt_durations": attempt_durations, "api_wait_seconds": round(api_wait_seconds, 4)}
        
        return None, {"retries": max_retries + 1, "attempt_durations": attempt_durations, "api_wait_seconds": round(api_wait_seconds, 4)}

    def _evaluate_with_openai(self, text: str, genome_id: str, max_retries: int = 10):
        
        import time

        cached = _get_cached_result(text, "openai")
        if cached is not None:
            return cached, {"retries": 0, "attempt_durations": [], "api_wait_seconds": 0.0}

        attempt_durations = []
        api_wait_seconds = 0.0
        for attempt in range(max_retries + 1):
            attempt_start = time.time()
            try:
                response = self.openai_client.moderations.create(
                    model=self.openai_model,
                    input=text,
                )
                result_block = response.results[0] if response.results else None
                category_scores = {}
                if result_block is not None:
                    scores_obj = getattr(result_block, "category_scores", None)
                    if scores_obj is not None:
                        if hasattr(scores_obj, "model_dump"):
                            raw = scores_obj.model_dump()
                        elif isinstance(scores_obj, dict):
                            raw = scores_obj
                        else:
                            raw = dict(scores_obj)
                        category_scores = {str(k).lower(): float(v) for k, v in raw.items()}

                normalized_scores = self._normalize_scores(category_scores)
                result = {
                    "scores": normalized_scores,
                    "api_version": "openai",
                    "model": self.openai_model,
                }
                _cache_result(text, result, "openai")
                attempt_durations.append(round(time.time() - attempt_start, 4))
                return result, {
                    "retries": attempt,
                    "attempt_durations": attempt_durations,
                    "api_wait_seconds": round(api_wait_seconds, 4),
                }
            except Exception as e:
                attempt_durations.append(round(time.time() - attempt_start, 4))
                is_retriable = self._is_retriable_error(e)
                safe_err = _redact_url_api_key_query_param(str(e))
                if attempt < max_retries and is_retriable:
                    wait_time = 2 ** attempt
                    self.logger.warning(
                        "OpenAI moderation failed for genome %s (attempt %d/%d): %s. Retrying in %ds...",
                        genome_id,
                        attempt + 1,
                        max_retries + 1,
                        safe_err,
                        wait_time,
                    )
                    time.sleep(wait_time)
                    api_wait_seconds += wait_time
                    continue
                self.logger.error(
                    "OpenAI moderation failed for genome %s after %d attempts: %s",
                    genome_id,
                    attempt + 1,
                    safe_err,
                )
                return None, {
                    "retries": attempt + 1,
                    "attempt_durations": attempt_durations,
                    "api_wait_seconds": round(api_wait_seconds, 4),
                }
        return None, {
            "retries": max_retries + 1,
            "attempt_durations": attempt_durations,
            "api_wait_seconds": round(api_wait_seconds, 4),
        }
    
    def _is_retriable_error(self, error: Exception) -> bool:
        
        error_str = str(error).lower()
        
        retriable_codes = ['429', '500', '502', '503', '504']
        
        if any(code in error_str for code in retriable_codes):
            return True
        
        if 'quota' in error_str or 'rate limit' in error_str or 'too many requests' in error_str:
            return True
        
        if 'timeout' in error_str or 'connection' in error_str or 'network' in error_str:
            return True
        
        if 'internal server error' in error_str or 'service unavailable' in error_str:
            return True
        
        if '400' in error_str or '401' in error_str or '403' in error_str or '404' in error_str:
            return False
        
        return False
    
    def _normalize_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        
        normalized_scores = {}
        
        for category, score in scores.items():
            score = float(score)
            
            if score < 0.0001:
                score = 0.0001
            
            if score > 1.0000:
                score = 1.0000
            
            normalized_score = round(score, 4)
            
            if normalized_score == 0.0:
                normalized_score = 0.0001
                
            normalized_scores[category] = normalized_score
        
        return normalized_scores
    
    def _evaluate_text_hybrid(self, text: str, genome_id: str, moderation_methods: List[str] = None) -> Dict[str, Any]:
        
        import time
        start_time = time.time()
        backend = self.profile.backend_key
        
        try:
            self.logger.debug("Evaluating text for genome %s: %d characters", genome_id, len(text))
            
            if moderation_methods is None:
                moderation_methods = moderation_methods_to_backend_list(self.profile)
            
            self.logger.debug("Using moderation backend %s for genome %s", backend, genome_id)
            
            results = {}
            
            if backend == "google" and "google" in moderation_methods and self.google_available:
                google_result, retry_info = self._evaluate_with_google(text, genome_id)
                if not hasattr(self, '_last_evaluation_time'):
                    self._last_evaluation_time = {}
                self._last_evaluation_time['retries'] = retry_info.get("retries", 0)
                self._last_evaluation_time['attempt_durations'] = retry_info.get("attempt_durations", [])
                self._last_evaluation_time['api_wait_seconds'] = retry_info.get("api_wait_seconds", 0.0)
                if google_result is not None:
                    results["google"] = google_result
                    self.logger.debug("Google evaluation completed for genome %s", genome_id)
            elif backend == "openai" and "openai" in moderation_methods and self.openai_available:
                openai_result, retry_info = self._evaluate_with_openai(text, genome_id)
                if not hasattr(self, '_last_evaluation_time'):
                    self._last_evaluation_time = {}
                self._last_evaluation_time['retries'] = retry_info.get("retries", 0)
                self._last_evaluation_time['attempt_durations'] = retry_info.get("attempt_durations", [])
                self._last_evaluation_time['api_wait_seconds'] = retry_info.get("api_wait_seconds", 0.0)
                if openai_result is not None:
                    results["openai"] = openai_result
                    self.logger.debug("OpenAI evaluation completed for genome %s", genome_id)
            elif backend == "google" and not self.google_available:
                self.logger.warning("Google Perspective API requested but not available for genome %s", genome_id)
            elif backend == "openai" and not self.openai_available:
                self.logger.warning("OpenAI moderation requested but not available for genome %s", genome_id)
            
            if not results:
                return {
                    "error": "All available APIs failed"
                }
            
            unified_result = results
            
            self.logger.info("Genome %s moderation evaluation completed", genome_id)
            
            return unified_result
            
        except Exception as e:
            safe_err = _redact_url_api_key_query_param(str(e))
            self.logger.error(
                "Unexpected error evaluating genome %s: %s",
                genome_id,
                safe_err,
                exc_info=True,
            )
            return {
                "error": safe_err
            }
        finally:
            end_time = time.time()
            evaluation_time = end_time - start_time
            if not hasattr(self, '_last_evaluation_time'):
                self._last_evaluation_time = {}
            self._last_evaluation_time['duration'] = evaluation_time
            if 'retries' not in self._last_evaluation_time:
                self._last_evaluation_time['retries'] = 0
            if 'attempt_durations' not in self._last_evaluation_time:
                self._last_evaluation_time['attempt_durations'] = []
            if 'api_wait_seconds' not in self._last_evaluation_time:
                self._last_evaluation_time['api_wait_seconds'] = 0.0
    
    def _evaluate_population_sync(self, population: List[Dict[str, Any]], 
                                 north_star_metric: str = "toxicity", 
                                 pop_path: str = "", moderation_methods: List[str] = None) -> List[Dict[str, Any]]:
        
        backend = self.profile.backend_key
        progress_label = north_star_metric
        
        try:
            self.logger.info("Starting population evaluation (backend=%s)", backend)
            self.logger.info("North star metric: %s", north_star_metric)
            
            pending_genomes = [g for g in population if g.get('status') == 'pending_evaluation']
            self.logger.info("Found %d genomes pending evaluation out of %d total", 
                           len(pending_genomes), len(population))
            
            if not pending_genomes:
                self.logger.info("No genomes pending evaluation. Skipping processing.")
                return population
            
            total_processed = 0
            total_errors = 0
            total_genomes = len(pending_genomes)
            start_time = time.time()
            
            print(f"\nEvaluating {progress_label}: 0/{total_genomes} (0%)", end='', flush=True)
            
            for i, genome in enumerate(pending_genomes, 1):
                try:
                    genome_id = genome.get('id', 'unknown')
                    generated_text = genome.get('generated_output', '')
                    
                    if not generated_text:
                        self.logger.warning("No generated output for genome %s", genome_id)
                        genome['status'] = 'error'
                        genome['error'] = 'No generated output'
                        total_errors += 1
                        self._save_single_genome(genome, pop_path)
                        
                        elapsed = time.time() - start_time
                        rate = i / elapsed if elapsed > 0 else 0
                        percentage = (i / total_genomes) * 100
                        remaining = (total_genomes - i) / rate if rate > 0 else 0
                        print(f"\rEvaluating {progress_label}: {i}/{total_genomes} ({percentage:.1f}%) | "
                              f"Processed: {total_processed} | Errors: {total_errors} | "
                              f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s", end='', flush=True)
                        continue
                    
                    evaluation_result = self._evaluate_text_hybrid(generated_text, genome_id, moderation_methods=moderation_methods)
                    
                    if backend in evaluation_result:
                        genome['moderation_result'] = evaluation_result
                        genome['evaluator'] = self.profile.name
                        
                        if hasattr(self, '_last_evaluation_time'):
                            genome['evaluation_duration'] = round(self._last_evaluation_time['duration'], 4)
                            genome['evaluation_retries'] = self._last_evaluation_time.get('retries', 0)
                            genome['evaluation_attempt_durations'] = self._last_evaluation_time.get('attempt_durations', [])
                            genome['evaluation_api_wait_seconds'] = round(self._last_evaluation_time.get('api_wait_seconds', 0.0), 4)
                        
                        north_star_score = _extract_north_star_score(genome, north_star_metric)
                        
                        genome['status'] = 'complete'
                        total_processed += 1
                        self.logger.debug("Genome %s %s score: %.4f", genome.get('id'), north_star_metric, north_star_score)
                    else:
                        genome["status"] = "error"
                        genome["error"] = evaluation_result.get("error", "Unknown error")
                        total_errors += 1
                    
                    self._save_single_genome(genome, pop_path)
                    self.logger.debug("Saved genome %s immediately after evaluation", genome_id)
                    
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    percentage = (i / total_genomes) * 100
                    remaining = (total_genomes - i) / rate if rate > 0 else 0
                    avg_score = ""
                    if total_processed > 0 and backend in evaluation_result:
                        avg_score = f" | Score: {north_star_score:.3f}"
                    print(f"\rEvaluating {progress_label}: {i}/{total_genomes} ({percentage:.1f}%) | "
                          f"Processed: {total_processed} | Errors: {total_errors}{avg_score} | "
                          f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s", end='', flush=True)
                    
                    time.sleep(0.75)
                        
                except Exception as e:
                    safe_err = _redact_url_api_key_query_param(str(e))
                    self.logger.error(
                        "Failed to process evaluation for genome %s: %s",
                        genome.get('id'),
                        safe_err,
                        exc_info=True,
                    )
                    genome['status'] = 'error'
                    genome['error'] = safe_err
                    total_errors += 1
                    self._save_single_genome(genome, pop_path)
                    
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    percentage = (i / total_genomes) * 100
                    remaining = (total_genomes - i) / rate if rate > 0 else 0
                    print(f"\rEvaluating {progress_label}: {i}/{total_genomes} ({percentage:.1f}%) | "
                          f"Processed: {total_processed} | Errors: {total_errors} | "
                          f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s", end='', flush=True)
            
            elapsed = time.time() - start_time
            print(f"\rEvaluating {progress_label}: {total_genomes}/{total_genomes} (100.0%) | "
                  f"Processed: {total_processed} | Errors: {total_errors} | "
                  f"Completed in {elapsed:.1f}s{'':<20}", flush=True)
            
            self.logger.info("Population evaluation completed:")
            self.logger.info("  - Total genomes: %d", len(population))
            self.logger.info("  - Processed: %d", total_processed)
            self.logger.info("  - Errors: %d", total_errors)
            
            return population
            
        except Exception as e:
            self.logger.error(
                "Population evaluation failed: %s",
                _redact_url_api_key_query_param(str(e)),
                exc_info=True,
            )
            raise
    
    def _save_single_genome(self, genome: Dict[str, Any], pop_path: str) -> None:
        
        try:
            from pathlib import Path
            
            pop_path_obj = Path(pop_path)
            if not pop_path_obj.exists():
                self.logger.debug(f"Population file {pop_path} does not exist for incremental save, skipping (final batch save will persist changes)")
                return
            
            with open(pop_path_obj, 'r', encoding='utf-8') as f:
                population = json.load(f)
            
            genome_id = genome.get('id')
            updated = False
            for i, existing_genome in enumerate(population):
                if existing_genome.get('id') == genome_id:
                    population[i] = genome
                    updated = True
                    break
            
            if not updated:
                self.logger.debug(f"Genome {genome_id} not found in file for incremental update (may be in memory only)")
                return
            
            with open(pop_path_obj, 'w', encoding='utf-8') as f:
                json.dump(population, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Incremental save completed for genome {genome_id}")
            
        except Exception as e:
            self.logger.debug(f"Incremental save failed for genome {genome.get('id', 'unknown')}: {e} (final batch save will persist changes)")

    def evaluate_population_sync(self, pop_path: str, north_star_metric: str = "toxicity", moderation_methods: List[str] = None) -> None:
        
        try:
            self.logger.info("Starting population evaluation pipeline")
            
            _pio = get_population_io()
            load_population, save_population = _pio[2], _pio[3]
            population = load_population(pop_path, logger=self.logger)
            
            if moderation_methods is None:
                moderation_methods = moderation_methods_to_backend_list(self.profile)
            
            self.logger.info("Using moderation backend: %s", self.profile.backend_key)
            
            updated_population = self._evaluate_population_sync(population, north_star_metric, pop_path=pop_path, moderation_methods=moderation_methods)
            
            save_population(updated_population, pop_path, logger=self.logger)
            
            self.logger.info("Population evaluation completed successfully")
            
        except Exception as e:
            self.logger.error(
                "Population evaluation pipeline failed: %s",
                _redact_url_api_key_query_param(str(e)),
                exc_info=True,
            )
            raise

def run_moderation_on_population(pop_path: str, log_file: Optional[str] = None, 
                               north_star_metric: str = "toxicity", moderation_methods: List[str] = None,
                               outputs_path: str = None, evaluator: Optional[str] = None,
                               openai_model: str = "omni-moderation-latest") -> None:
    
    get_logger, _, _, _ = get_custom_logging()
    logger = get_logger("run_moderation", log_file)
    
    try:
        logger.info("Starting moderation evaluation for population")
        
        profile = set_active_evaluator(evaluator or "google")
        if moderation_methods is None:
            moderation_methods = moderation_methods_to_backend_list(profile)
        
        logger.info("Evaluator=%s, moderation_methods=%s", profile.name, moderation_methods)
        
        from pathlib import Path
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "RGConfig.yaml"
        mod_evaluator = HybridModerationEvaluator(
            config_path=str(config_path),
            log_file=log_file,
            evaluator=profile.name,
            openai_model=openai_model,
        )
        
        mod_evaluator.evaluate_population_sync(pop_path, north_star_metric, moderation_methods=moderation_methods)
        
        logger.info("Moderation evaluation completed successfully")
        
    except Exception as e:
        logger.error(
            "Moderation evaluation failed: %s",
            _redact_url_api_key_query_param(str(e)),
            exc_info=True,
        )


def evaluate_single_genome(evaluator, genome, moderation_methods=None):
    
    generated_text = genome.get("generated_output", "")
    genome_id = genome.get("local_variant_id", genome.get("id", "unknown"))
    backend = evaluator.profile.backend_key

    if not generated_text:
        genome["status"] = "error"
        genome["error"] = "No generated output"
        return genome

    if moderation_methods is None:
        moderation_methods = moderation_methods_to_backend_list(evaluator.profile)

    evaluation_result = evaluator._evaluate_text_hybrid(
        generated_text, genome_id, moderation_methods=moderation_methods)

    if backend in evaluation_result:
        genome["moderation_result"] = evaluation_result
        genome["evaluator"] = evaluator.profile.name
        if hasattr(evaluator, "_last_evaluation_time"):
            genome["evaluation_duration"] = round(
                evaluator._last_evaluation_time.get("duration", 0.0), 4)
            genome["evaluation_retries"] = evaluator._last_evaluation_time.get("retries", 0)
            genome["evaluation_attempt_durations"] = evaluator._last_evaluation_time.get("attempt_durations", [])
            genome["evaluation_api_wait_seconds"] = round(evaluator._last_evaluation_time.get("api_wait_seconds", 0.0), 4)
        genome["status"] = "complete"
    else:
        genome["status"] = "error"
        genome["error"] = evaluation_result.get("error", "Unknown error")

    return genome
