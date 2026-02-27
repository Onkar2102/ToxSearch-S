"""
ResponseGenerator: Text generator for response generation using prompt templates.
"""

import os
import json
import yaml
import time
import psutil
import sys
from typing import List, Dict, Any, Optional
from llama_cpp import Llama
from utils import get_custom_logging
from .model_interface import LlamaCppChatInterface

get_logger, _, _, _ = get_custom_logging()

class ResponseGenerator:
    """
    Response generator using v1/chat/completions interface for efficient inference.
    """
    
    def __init__(self, model_key="response_generator", config_path="config/RGConfig.yaml", log_file: Optional[str] = None):
        self.log_file = log_file
        self.logger = get_logger("ResponseGenerator", self.log_file)
        self.logger.debug(f"Logger correctly initialized with log_file: {self.log_file}")

        try:
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
            if not config:
                raise ValueError(f"Configuration file is empty: {config_path}")
            if model_key not in config:
                raise ValueError(f"Model '{model_key}' not found in configuration. Available keys: {list(config.keys())}")
            self.model_cfg = config[model_key]
            
            if not self.model_cfg.get("name"):
                raise ValueError(f"Model configuration missing 'name' field for {model_key}")
                
        except FileNotFoundError:
            self.logger.error(f"Configuration file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Failed to parse YAML configuration: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Failed to load model configuration: {e}")
            raise

        try:
            self.model_interface = LlamaCppChatInterface(self.model_cfg, log_file)
        except Exception as e:
            self.logger.error(f"Failed to initialize model interface: {e}")
            raise
        
        tmpl = self.model_cfg.get("prompt_template", {})
        self.prompt_messages = tmpl.get("messages", [])
        
        self.generation_count = 0
        self.total_tokens_generated = 0
        self.total_generation_time = 0.0

    def _build_messages(self, raw_prompt: str) -> List[Dict[str, str]]:
        """Build messages array from prompt template and user input."""
        messages = []
        
        for msg_template in self.prompt_messages:
            role = msg_template.get("role", "user")
            content = msg_template.get("content", "")
            
            if "{{prompt}}" in content:
                content = content.replace("{{prompt}}", raw_prompt)
            
            if content.strip():
                messages.append({"role": role, "content": content})
        
        if not messages:
            messages.append({"role": "user", "content": raw_prompt})
        
        return messages

    def generate_response(self, prompt: str, **kwargs) -> tuple[str, float]:
        """Generate a response to a prompt using chat completions interface.

        Args:
            prompt: Raw prompt text to send to the model.
            **kwargs: Optional arguments passed through to model_interface.chat_completion (e.g. max_tokens, temperature).

        Returns:
            tuple: (response_text, duration_in_seconds). response_text is empty string on failure.
        """
        start_time = time.time()
        
        try:
            messages = self._build_messages(prompt)
            
            self.logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            
            generated_text = self.model_interface.chat_completion(messages, **kwargs)
            
            self.generation_count += 1
            self.total_tokens_generated += len(generated_text.split())
            
            self.logger.debug(f"Generated response: {generated_text[:100]}...")
            return generated_text, time.time() - start_time
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            return "", time.time() - start_time

    def process_population(self, pop_path: str = "data/outputs/temp.json") -> None:
        """Process entire population for text generation one genome at a time."""
        try:
            self.logger.info("Starting population processing for text generation with chat completions")
            
            from utils.population_io import load_population
            
            population = load_population(pop_path, logger=self.logger)
            
            pending_genomes = [g for g in population if g.get('status') == 'pending_generation']
            self.logger.info("Found %d genomes pending generation out of %d total", len(pending_genomes), len(population))
            
            if not pending_genomes:
                self.logger.info("No genomes pending generation. Skipping processing.")
                return
            
            total_processed = 0
            total_errors = 0
            total_genomes = len(pending_genomes)
            start_time = time.time()
            
            # Simple progress indicator
            print(f"\nGenerating responses: 0/{total_genomes} (0%)", end='', flush=True)
            
            for i, genome in enumerate(pending_genomes, 1):
                genome_id = genome.get('id', 'unknown')
                try:
                    response, response_duration = self.generate_response(genome['prompt'])
                    
                    if response:
                        genome['model_name'] = self.model_cfg.get("name", "")
                        genome['generated_output'] = response
                        genome['response_duration'] = round(response_duration, 4)
                        genome['status'] = 'pending_evaluation'
                        
                        total_processed += 1
                        self.logger.debug("Generated response for genome %s", genome_id)
                    else:
                        genome['status'] = 'error'
                        genome['error'] = 'Failed to generate response'
                        total_errors += 1
                        self.logger.warning("Failed to generate response for genome %s", genome_id)
                    
                    self._save_single_genome(genome, pop_path)
                    self.logger.debug("Saved genome %s immediately after generation", genome_id)
                    
                    # Update progress indicator
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    percentage = (i / total_genomes) * 100
                    remaining = (total_genomes - i) / rate if rate > 0 else 0
                    print(f"\rGenerating responses: {i}/{total_genomes} ({percentage:.1f}%) | "
                          f"Processed: {total_processed} | Errors: {total_errors} | "
                          f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s", end='', flush=True)
                        
                except Exception as e:
                    genome['status'] = 'error'
                    genome['error'] = str(e)
                    total_errors += 1
                    self.logger.error("Error processing genome %s: %s", genome_id, e)
                    self._save_single_genome(genome, pop_path)
                    
                    # Update progress even on error
                    elapsed = time.time() - start_time
                    rate = i / elapsed if elapsed > 0 else 0
                    percentage = (i / total_genomes) * 100
                    remaining = (total_genomes - i) / rate if rate > 0 else 0
                    print(f"\rGenerating responses: {i}/{total_genomes} ({percentage:.1f}%) | "
                          f"Processed: {total_processed} | Errors: {total_errors} | "
                          f"Rate: {rate:.1f}/s | ETA: {remaining:.0f}s", end='', flush=True)
            
            # Final update and newline
            elapsed = time.time() - start_time
            print(f"\rGenerating responses: {total_genomes}/{total_genomes} (100.0%) | "
                  f"Processed: {total_processed} | Errors: {total_errors} | "
                  f"Completed in {elapsed:.1f}s{'':<20}", flush=True)
            
            try:
                from utils.population_io import save_population
                save_population(population, pop_path, logger=self.logger)
                self.logger.info("Final batch save completed successfully")
            except Exception as e:
                self.logger.error(f"Failed to perform final batch save: {e}")
            
            self.logger.info("Population processing completed:")
            self.logger.info("  - Total genomes: %d", len(population))
            self.logger.info("  - Processed: %d", total_processed)
            self.logger.info("  - Errors: %d", total_errors)
            
        except Exception as e:
            self.logger.error("Population processing failed: %s", e, exc_info=True)
            raise

    def _save_single_genome(self, genome: Dict[str, Any], pop_path: str) -> None:
        """
        Save a single genome immediately by updating the existing population file.
        This is a best-effort incremental save for crash recovery.
        A final batch save is always performed at the end of processing.
        """
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


def process_single_genome(response_generator, genome):
    """Generate an LLM response for a single genome dict in-memory.

    Updates *genome* in-place (generated_output, model_name,
    response_duration, status) and returns it.
    """
    prompt = genome.get("prompt", "")
    response_text, duration = response_generator.generate_response(prompt)

    genome["generated_output"] = response_text
    genome["model_name"] = getattr(response_generator, "model_name", None)
    genome["response_duration"] = round(duration, 4)
    genome["status"] = "pending_evaluation"
    return genome
