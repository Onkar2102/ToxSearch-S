"""
EvolutionEngine.py

Core evolutionary algorithm engine for text generation optimization.

This module implements the main evolution logic including parent selection,
operator application, and population management. Uses steady-state evolution
with elites preservation and multi-API moderation scoring.
"""

import json
import random
from typing import List, Dict, Any, Optional
from utils import get_custom_logging
from utils.population_io import _extract_north_star_score
from .parent_selector import ParentSelector
from itertools import combinations
from pathlib import Path
from .synonym_replacement import LLM_POSAwareSynonymReplacement
from .mlm_operator import MLMOperator
from .paraphrasing import LLMBasedParaphrasingOperator
from .antonym_replacement import POSAwareAntonymReplacement
from .stylistic_mutator import StylisticMutator
from .back_translation import (
    LLMBackTranslationHIOperator
)
from .semantic_similarity_crossover import SemanticSimilarityCrossover
from .fusion_crossover import SemanticFusionCrossover
from .negation_operator import NegationOperator
from .typographical_errors import TypographicalErrorsOperator
from .concept_addition import ConceptAdditionOperator
from .informed_evolution import InformedEvolutionOperator
from .operator_statistics import OperatorStatistics

_global_response_generator = None
_global_prompt_generator = None

def set_global_generators(response_generator, prompt_generator):
    """Set the global generator instances to be used by the system."""
    global _global_response_generator, _global_prompt_generator
    _global_response_generator = response_generator
    _global_prompt_generator = prompt_generator

def get_response_generator():
    """Get the shared response generator instance."""
    global _global_response_generator
    if _global_response_generator is None:
        raise RuntimeError("No global response generator set. Call set_global_generators() first from main.py")
    return _global_response_generator

def get_prompt_generator():
    """Get the shared prompt generator instance."""
    global _global_prompt_generator
    if _global_prompt_generator is None:
        raise RuntimeError("No global prompt generator set. Call set_global_generators() first from main.py")
    return _global_prompt_generator


class EvolutionEngine:

    def __init__(self, north_star_metric, log_file, current_cycle=None, max_variants=3, adaptive_selection_after=5, max_num_parents=4, operators="all", outputs_path=None):
        self._genomes_loaded = False
        self._genomes_cache = []
        self.next_id = 0
        self.north_star_metric = north_star_metric
        self.log_file = log_file
        self.current_cycle = current_cycle
        self.use_steady_state = True
        self.max_variants = max_variants
        self.operators = operators
        self.outputs_path = outputs_path
        get_logger, _, _, _ = get_custom_logging()
        self.logger = get_logger("EvolutionEngine", log_file)
        self.parent_selector = ParentSelector(north_star_metric, log_file)
        # Initialize the shared generator instances
        self.prompt_generator = get_prompt_generator()
        self.response_generator = get_response_generator()

        self.operator_stats = OperatorStatistics()

        self.logger.debug(f"EvolutionEngine initialized with next_id={self.next_id}, north_star_metric={north_star_metric}, current_cycle={current_cycle}, max_variants={max_variants}, adaptive_selection_after={adaptive_selection_after}, max_num_parents={max_num_parents}, operators={operators}, use_steady_state=True")

    @property
    def genomes(self):
        """Lazy load genomes only when needed"""
        if not self._genomes_loaded:
            from utils.population_io import load_population
            self._genomes_cache = load_population(str(self.outputs_path), logger=self.logger)
            self._genomes_loaded = True
            self.logger.debug(f"Lazy loaded {len(self._genomes_cache)} genomes")
        return self._genomes_cache

    @genomes.setter
    def genomes(self, value):
        """Allow setting genomes directly"""
        self._genomes_cache = value
        self._genomes_loaded = True

    def update_next_id(self):
        """
        Update next_id based on ALL genome files (elites.json, reserves.json, archive.json).
        
        So next_id stays higher than any existing genome ID and we avoid duplicate IDs.
        """
        from utils.population_io import get_max_genome_id_from_all_files
        
        # Get max ID from all files (elites, reserves, archive)
        # get_max_genome_id_from_all_files handles None outputs_path by using get_outputs_path()
        max_id = get_max_genome_id_from_all_files(self.outputs_path)
        
        # Also check in-memory genomes as fallback/validation
        if self.genomes:
            in_memory_max = max((g["id"] for g in self.genomes if g.get("id") is not None), default=0)
            max_id = max(max_id, in_memory_max)
        
        # Set next_id to max_id + 1 (or 1 if no genomes exist)
        self.next_id = max_id + 1 if max_id > 0 else 1
        self.logger.debug(f"Updated next_id to {self.next_id} (max_id found: {max_id})")

    def _calculate_parent_score(self, parents: List[Dict], variant_type: str, operator: Any = None) -> float:
        """
        Calculate parent score based on variant type.

        Args:
            parents: List of parent genomes (simplified structure with 'toxicity' field)
            variant_type: Type of variant ("mutation" or "crossover")
            operator: Operator instance (for InformedEvolutionOperator special handling)

        Returns:
            float: Parent score (minimum 0.0001 for consistency)
        """
        if operator and hasattr(operator, 'top_10_avg_score'):
            self.logger.debug(f"Using top_10 average score: {operator.top_10_avg_score:.4f}")
            return operator.top_10_avg_score

        if variant_type == "mutation":
            if not parents:
                return 0.0001
            parent_score = _extract_north_star_score(parents[0], "toxicity")
            return max(round(parent_score, 4), 0.0001)
        elif variant_type == "crossover":
            if not parents:
                return 0.0001
            scores = [max(_extract_north_star_score(p, "toxicity"), 0.0001) for p in parents]
            avg_score = sum(scores) / len(scores)
            return round(avg_score, 4)

        return 0.0001

    def _create_child_genome(self, prompt: str, operator: Any, parents: List[Dict], variant_type: str) -> Dict:
        """Create a child genome from a prompt and operator."""
        parent_score = self._calculate_parent_score(parents, variant_type, operator)

        parents_info = []
        for p in parents:
            parent_id = p.get("id")
            parent_toxicity = _extract_north_star_score(p, "toxicity")
            parents_info.append({
                "id": parent_id,
                "score": round(parent_toxicity, 4)
            })

        # Get prompt generator name if available
        prompt_generator_name = None
        if self.prompt_generator and hasattr(self.prompt_generator, 'model_cfg'):
            prompt_generator_name = self.prompt_generator.model_cfg.get("name", "")

        # Ensure next_id is updated before creating child (defensive check)
        if self.next_id == 0:
            self.update_next_id()
        
        child = {
            "id": self.next_id,
            "prompt": prompt,
            "model_name": None,
            "prompt_generator_name": prompt_generator_name,
            "moderation_result": None,
            "operator": operator.name,
            "parents": parents_info,
            "generation": self.current_cycle,
            "status": "pending_generation",
            "parent_score": parent_score,
            "variant_type": variant_type,
            "creation_info": {
                "type": variant_type,
                "operator": operator.name,
                "parent_score": parent_score
            }
        }

        if hasattr(operator, '_last_operation_time'):
            child['variant_creation_duration'] = round(operator._last_operation_time.get('duration', 0.0), 4)

        self.next_id += 1
        return child

    def generate_variants_global(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """
        Generate variants globally for evolution cycle.
        Updates temp.json with unique variants created.

        Args:
            evolution_tracker (Dict[str, Any]): Evolution tracker data for determining parent counts
        """
        self.logger.debug(f"Generating variants globally for evolution cycle {self.current_cycle}")

        # Step 1: Synchronize next_id with current genomes
        # Prevents ID reuse if engine persists across cycles.
        self.update_next_id()

        if self.operators == "ie":
            # Mode "ie": Only use top_10.json, skip parent selection
            self._generate_variants_ie_mode(evolution_tracker)
        elif self.operators == "cm":
            self._generate_variants_cm_mode(evolution_tracker)
        elif self.operators == "all":
            # Mode "all": Use both files (default behavior)
            self._generate_variants_all_mode(evolution_tracker)
        else:
            self.logger.warning(f"Unknown operator mode '{self.operators}', defaulting to 'all'")
            self._generate_variants_all_mode(evolution_tracker)
        
        # Step 3: EvolutionTracker updates are handled by parent_selector.py only
        # No additional updates needed here to avoid conflicts

    def _generate_variants_ie_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using only InformedEvolution operator with top_10.json"""

        try:
            elites_path = str(Path(self.outputs_path) / "elites.json")
            top_10_path = str(Path(self.outputs_path) / "top_10.json")
            self.parent_selector._save_top_10_by_toxicity(elites_path, top_10_path)


        except Exception as e:
            self.logger.error(f"Failed to populate top_10.json: {e}")
            return

        ie_operators = self._get_single_parent_operators()

        if not ie_operators:
            self.logger.error("No InformedEvolution operators found in IE mode")
            return

        top_10_path = Path(self.outputs_path) / "top_10.json"
        parent_example = None

        if top_10_path.exists():
            with open(top_10_path, 'r', encoding='utf-8') as f:
                top_10_examples = json.load(f)
            if top_10_examples:
                parent_example = top_10_examples[0]
                self.logger.debug(f"Using parent example from top_10.json: {parent_example['id']}")
            else:
                self.logger.error("No examples found in top_10.json")
                return
        else:
            self.logger.error("top_10.json not found")
            return

        if evolution_tracker is None:
            selection_mode = "default"
        else:
            selection_mode = evolution_tracker.get("selection_mode", "default")

        if selection_mode == "explore" or selection_mode == "exploit":
            num_calls = 3
        else:
            num_calls = 2

        self.logger.info(f"IE mode: Selection mode={selection_mode}, calling operator {num_calls} times")

        for operator in ie_operators:
            try:
                self.logger.debug(f"Running operator: {operator.__class__.__name__} {num_calls} times")

                variants_to_save = []
                for variant_iteration in range(num_calls):
                    operator_input = {
                        "parent_data": parent_example
                    }

                    variants = operator.apply(operator_input)

                    if variants:
                        variants_to_save.extend([self._create_child_genome(vp, operator, [parent_example], "mutation") for vp in variants])
                    else:
                        self.operator_stats.record_question_mark_rejection(operator.name)
                        self.logger.debug(f"{operator.name} call {variant_iteration + 1}/{num_calls} returned empty variants (tracked as rejection)")

                if variants_to_save:
                    self._append_variants_to_temp(variants_to_save)
                    self.logger.debug(f"Generated {len(variants_to_save)} variants using {operator.__class__.__name__}")
                else:
                    self.logger.warning(f"No variants generated by {operator.__class__.__name__} after {num_calls} calls")

            except Exception as e:
                self.logger.error(f"Error running operator {operator.__class__.__name__}: {e}", exc_info=True)
                self.operator_stats.record_question_mark_rejection(operator.name)

        parents_path = Path(self.outputs_path) / "parents.json"
        self._update_evolution_tracker_from_files(parents_path, top_10_path)
        try:
            if top_10_path.exists():
                with open(top_10_path, 'w', encoding='utf-8') as f:
                    json.dump([], f, indent=2, ensure_ascii=False)
                self.logger.debug(f"Emptied file: {top_10_path}")
        except Exception as e:
            self.logger.error(f"Failed to empty top_10 file: {e}")

    def _generate_variants_cm_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using all operators except InformedEvolution, using parents.json"""

        elites_path = Path(self.outputs_path) / "elites.json"
        if elites_path.exists():
            with open(elites_path, 'r', encoding='utf-8') as f:
                elites = json.load(f)
            if not elites:
                self.logger.error("CRITICAL ERROR: elites.json exists but is empty - this indicates a fundamental problem")
                self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
                raise RuntimeError("Empty elites.json - evolution cannot continue. This indicates a critical system failure.")
        else:
            self.logger.error("CRITICAL ERROR: elites.json does not exist - this indicates a fundamental problem")
            self.logger.error("Evolution cannot continue without elites. Stopping immediately.")
            raise RuntimeError("Missing elites.json - evolution cannot continue. This indicates a critical system failure.")

        self.parent_selector.adaptive_tournament_selection(evolution_tracker, outputs_path=str(self.outputs_path))

        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("No parents selected or failed to load parents from file")
            return

        single_parent_operators = self._get_single_parent_operators()
        multi_parent_operators = self._get_multi_parent_operators()

        if len(parents) >= 2:
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(multi_parent_operators)} operators.")
            self._run_crossover_operators(parents, multi_parent_operators)

        if len(parents) >= 1:
            self.logger.debug(f"Running mutation globally with {len(parents)} parents and {len(single_parent_operators)} operators.")
            self._run_mutation_operators(parents, single_parent_operators)

    def _generate_variants_all_mode(self, evolution_tracker: Dict[str, Any] = None) -> None:
        """Generate variants using all operators with both parents.json and top_10.json"""

        # Check for population files - elites.json is preferred, but reserves.json can be used as fallback
        elites_path = Path(self.outputs_path) / "elites.json"
        reserves_path = Path(self.outputs_path) / "reserves.json"
        
        # Check if files exist and have content (non-empty lists)
        has_elites = False
        has_reserves = False
        
        if elites_path.exists():
            try:
                elites_data = json.loads(elites_path.read_text())
                has_elites = isinstance(elites_data, list) and len(elites_data) > 0
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Failed to read elites.json: {e}")
                has_elites = False
        
        if reserves_path.exists():
            try:
                reserves_data = json.loads(reserves_path.read_text())
                has_reserves = isinstance(reserves_data, list) and len(reserves_data) > 0
            except (json.JSONDecodeError, Exception) as e:
                self.logger.warning(f"Failed to read reserves.json: {e}")
                has_reserves = False
        
        # Only raise error if both files are missing or both are empty
        if not has_elites and not has_reserves:
            self.logger.error("CRITICAL ERROR: No population files found with content (elites.json or reserves.json)")
            self.logger.error("Evolution cannot continue without any genomes. Stopping immediately.")
            raise RuntimeError("No population files found - evolution cannot continue. This indicates a critical system failure.")
        
        if not has_elites:
            # Fallback: use reserves for parent selection
            self.logger.warning("elites.json is empty or missing, using reserves.json for parent selection")

        self.parent_selector.adaptive_tournament_selection(evolution_tracker, outputs_path=str(self.outputs_path))

        parents = self._load_parents_from_file()
        if not parents:
            self.logger.error("No parents selected or failed to load parents from file")
            return

        single_parent_operators = self._get_single_parent_operators()
        multi_parent_operators = self._get_multi_parent_operators()

        # Note: expected_variant_count removed - not used for metric calculations, only validation
        # Metrics use calculated_total from operator statistics instead

        if len(parents) >= 2:
            self.logger.debug(f"Running crossover globally with {len(parents)} parents and {len(multi_parent_operators)} operators.")
            self._run_crossover_operators(parents, multi_parent_operators)

        if len(parents) >= 1:
            self.logger.debug(f"Running mutation globally with {len(parents)} parents and {len(single_parent_operators)} operators.")
            self._run_mutation_operators(parents, single_parent_operators)

    def _run_crossover_operators(self, parents: List[Dict], crossover_operators: List) -> None:
        """Run crossover operators on parent pairs"""
        for op in crossover_operators:
            if op.operator_type != "crossover":
                continue

            for parent_pair in combinations(parents, 2):  # All pairs of parents
                try:
                    # Call crossover operator max_variant times since it outputs one variant
                    variants_to_save = []
                    for _ in range(self.max_variants):
                        operator_input = {
                            "parent_data": list(parent_pair)
                        }
                        variants = op.apply(operator_input)
                        
                        if variants:
                            variants_to_save.extend([self._create_child_genome(vp, op, list(parent_pair), "crossover") for vp in variants])
                        else:
                            # Track question mark rejections (empty variants = rejections)
                            self.operator_stats.record_question_mark_rejection(op.name)
                            self.logger.warning(f"{op.name} failed to generate variants for crossover")
                    
                    # Save variants immediately to temp.json
                    if variants_to_save:
                        self._append_variants_to_temp(variants_to_save)
                        self.logger.debug(f"Saved {len(variants_to_save)} crossover variants from {op.name}")
                        
                except Exception as e:
                    self.logger.error(f"[Crossover Error] {op.name} with parents {[p['id'] for p in parent_pair]}: {e}")
                    # Track question mark rejections (exceptions = rejections)
                    self.operator_stats.record_question_mark_rejection(op.name)

    def _run_mutation_operators(self, parents: List[Dict], mutation_operators: List) -> None:
        """Run mutation operators on parents"""
        for op in mutation_operators:
            if op.operator_type != "mutation":
                continue

            for parent in parents:
                try:
                    variants_to_save = []
                    for variant_iteration in range(self.max_variants):
                        operator_input = {
                            "parent_data": parent
                        }
                        variants = op.apply(operator_input)
                        
                        if variants:
                            variants_to_save.extend([self._create_child_genome(vp, op, [parent], "mutation") for vp in variants])
                        else:
                            self.operator_stats.record_question_mark_rejection(op.name)
                            self.logger.warning(f"{op.name} failed to generate variants for mutation")
                    
                    if variants_to_save:
                        self._append_variants_to_temp(variants_to_save)
                        self.logger.debug(f"Saved {len(variants_to_save)} mutation variants from {op.name} for parent {parent['id']} ({self.max_variants} calls)")
                        
                except Exception as e:
                    self.logger.error(f"[Mutation Error] {op.name} with parent {parent['id']}: {e}")
                    self.operator_stats.record_question_mark_rejection(op.name)

        self.clean_parents_file()
    
    def _load_parents_from_file(self) -> List[Dict]:
        """
        Load parents from parents.json file.
        
        Returns:
            List[Dict]: List of parent genomes
        """
        try:
            parents_path = Path(self.outputs_path) / "parents.json"
            if not parents_path.exists():
                self.logger.warning("Parents file not found: %s", parents_path)
                return []
            
            with open(parents_path, 'r', encoding='utf-8') as f:
                parents_data = json.load(f)
            
            if isinstance(parents_data, list):
                parents = parents_data
            else:
                self.logger.warning("Unexpected parents.json structure")
                return []
            
            self.logger.debug(f"Loaded {len(parents)} parents from file: {[p['id'] for p in parents]}")
            
            return parents
            
        except Exception as e:
            self.logger.error(f"Failed to load parents from file: {e}")
            return []
    

    def _append_variants_to_temp(self, variants: List[Dict]) -> None:
        """
        Append variants to temp.json file.
        
        Args:
            variants: List of variant genomes to append
        """
        try:
            temp_path = Path(self.outputs_path) / "temp.json"
            
            if temp_path.exists():
                with open(temp_path, 'r', encoding='utf-8') as f:
                    existing_variants = json.load(f)
            else:
                existing_variants = []
            
            existing_variants.extend(variants)
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                json.dump(existing_variants, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Appended {len(variants)} variants to temp.json (total: {len(existing_variants)})")
            
        except Exception as e:
            self.logger.error(f"Failed to append variants to temp.json: {e}")
            raise
    
    def clean_parents_file(self) -> None:
        """Read parents.json and top_10.json, update EvolutionTracker, then empty top_10.json.
        Do NOT clear parents.json: run_speciation (Phase 4) needs it to determine which species
        were selected as parents for stagnation logic. parents.json is overwritten by the next
        generation's parent selection."""
        try: 
            parents_path = Path(self.outputs_path) / "parents.json"
            top10_path = Path(self.outputs_path) / "top_10.json"
            
            # Read files before cleaning and update EvolutionTracker
            self._update_evolution_tracker_from_files(parents_path, top10_path)
            
            # Only clear top_10.json. Keep parents.json for run_speciation to compute
            # selected_species_ids and was_selected_as_parent (stagnation).
            top10_path.parent.mkdir(parents=True, exist_ok=True)
            with open(top10_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            self.logger.debug("Emptied top_10.json (parents.json kept for speciation stagnation logic)")
        except Exception as e:
            self.logger.error(f"Failed to empty parents/top_10 file: {e}")

    def _update_evolution_tracker_from_files(self, parents_path: Path, top10_path: Path) -> None:
        """
        Read parents.json and top_10.json files and update EvolutionTracker with just the genome IDs.
        
        Args:
            parents_path: Path to parents.json file
            top10_path: Path to top_10.json file
        """
        try:
            import json
            from pathlib import Path
            from utils.population_io import get_outputs_path
            
            # Load EvolutionTracker
            evolution_tracker_path = get_outputs_path() / "EvolutionTracker.json"
            if not evolution_tracker_path.exists():
                self.logger.warning("EvolutionTracker.json not found for update")
                return
            
            with open(evolution_tracker_path, 'r', encoding='utf-8') as f:
                tracker = json.load(f)
            
            # Get current generation number - use current_cycle directly
            current_generation = self.current_cycle
            if current_generation is None:
                self.logger.error("current_cycle is None - cannot determine generation number")
                return
            
            # Find or create the current generation entry
            current_gen = None
            for gen in tracker.get("generations", []):
                if gen.get("generation_number") == current_generation:
                    current_gen = gen
                    break
            
            if current_gen is None:
                # Get selection_mode from EvolutionTracker root level for this generation
                selection_mode = tracker.get("selection_mode", "default")
                # Create new generation entry with all standard fields
                from utils.population_io import _get_standard_generation_entry_template
                current_gen = _get_standard_generation_entry_template(current_generation, selection_mode)
                tracker.setdefault("generations", []).append(current_gen)
                self.logger.info(f"Created new generation entry: {current_generation}")
            else:
                # Ensure existing entry has all fields
                from utils.population_io import _ensure_generation_entry_has_all_fields
                selection_mode = tracker.get("selection_mode", "default")
                current_gen = _ensure_generation_entry_has_all_fields(current_gen, current_generation, selection_mode)
            
            # Read parent IDs from parents.json
            parent_ids = []
            if parents_path.exists():
                with open(parents_path, 'r', encoding='utf-8') as f:
                    parents_data = json.load(f)
                if isinstance(parents_data, list) and parents_data:
                    parent_ids = [str(p.get("id")) for p in parents_data if p.get("id")]
            
            # Read top_10 IDs from top_10.json
            top_10_ids = []
            if top10_path.exists():
                with open(top10_path, 'r', encoding='utf-8') as f:
                    top_10_data = json.load(f)
                if isinstance(top_10_data, list) and top_10_data:
                    top_10_ids = [str(genome.get("id")) for genome in top_10_data if genome and genome.get("id")]
            
            # Update the generation entry with just the IDs
            current_gen["parents"] = parent_ids
            current_gen["top_10"] = top_10_ids
            
            # Save updated EvolutionTracker
            with open(evolution_tracker_path, 'w', encoding='utf-8') as f:
                json.dump(tracker, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Updated EvolutionTracker with {len(parent_ids)} parent IDs and {len(top_10_ids)} top_10 IDs for generation {current_generation}")
            
        except Exception as e:
            self.logger.error(f"Failed to update EvolutionTracker from files: {e}")

    def _get_single_parent_operators(self):
        """Return list of mutation operators that require only a single parent."""
        
        if self.operators == "ie":
            filtered_operators = [
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json"))
            ]
            self.logger.debug("IE mode: %d operators", len(filtered_operators))
        elif self.operators == "cm":
            filtered_operators = [
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, num_error_types=3, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, num_concept_types=1, generator=self.prompt_generator),
            ]
            self.logger.debug("CM mode: %d operators", len(filtered_operators))
        elif self.operators == "all":
            filtered_operators = [
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, num_error_types=3, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, num_concept_types=1, generator=self.prompt_generator),
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json")),
            ]
            self.logger.debug("ALL mode: %d operators", len(filtered_operators))
        else:
            filtered_operators = [
                LLM_POSAwareSynonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                POSAwareAntonymReplacement(self.north_star_metric, log_file=self.log_file, num_POS_tags=1, generator=self.prompt_generator),
                
                MLMOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                LLMBasedParaphrasingOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                StylisticMutator(log_file=self.log_file, generator=self.prompt_generator),
                
                LLMBackTranslationHIOperator(log_file=self.log_file, generator=self.prompt_generator),
                
                NegationOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator),
                TypographicalErrorsOperator(self.north_star_metric, log_file=self.log_file, num_error_types=3, generator=self.prompt_generator),
                ConceptAdditionOperator(self.north_star_metric, log_file=self.log_file, num_concept_types=1, generator=self.prompt_generator),
                InformedEvolutionOperator(self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator, top_10_path=str(self.outputs_path / "top_10.json")),
            ]
            self.logger.warning("Invalid operator mode '%s', defaulting to 'all' (%d operators)", self.operators, len(filtered_operators))
        
        return filtered_operators

    def _get_multi_parent_operators(self):
        """Return list of crossover operators that require multiple parents."""
        
        # Initialize operators based on configuration to avoid unnecessary initialization
        if self.operators == "ie":
            # Only InformedEvolution operator - no crossover operators
            filtered_operators = []
            self.logger.debug("IE mode: No crossover operators")
        elif self.operators == "cm":
            # All crossover operators (no InformedEvolution in crossover)
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.debug("CM mode: %d crossover operators", len(filtered_operators))
        elif self.operators == "all":
            # All crossover operators
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.debug("ALL mode: %d crossover operators", len(filtered_operators))
        else:
            # Default to all operators if invalid mode
            filtered_operators = [
                SemanticSimilarityCrossover(log_file=self.log_file),
                SemanticFusionCrossover(north_star_metric=self.north_star_metric, log_file=self.log_file, generator=self.prompt_generator)
            ]
            self.logger.warning("Invalid operator mode '%s', defaulting to 'all' for crossover (%d operators)", self.operators, len(filtered_operators))
        
        return filtered_operators

    def _deduplicate_temp_json(self) -> int:
        """
        Remove duplicate variants within temp.json based on normalized prompt.
        Keeps the first occurrence and discards subsequent duplicates.
        
        Returns:
            int: Number of duplicates removed
        """
        try:
            temp_path = Path(self.outputs_path) / "temp.json"
            if not temp_path.exists():
                return 0

            with open(temp_path, 'r', encoding='utf-8') as f:
                variants = json.load(f)

            if not isinstance(variants, list) or not variants:
                return 0

            seen_prompts = set()
            seen_ids = set()
            unique_variants = []
            duplicates_removed = 0

            for v in variants:
                if not isinstance(v, dict):
                    duplicates_removed += 1
                    continue

                prompt = v.get("prompt")
                vid = v.get("id")

                # Exact match, no normalization
                if (prompt is not None and prompt in seen_prompts) or (vid is not None and vid in seen_ids):
                    duplicates_removed += 1
                    continue

                if prompt is not None:
                    seen_prompts.add(prompt)
                if vid is not None:
                    seen_ids.add(vid)
                unique_variants.append(v)

            if duplicates_removed > 0:
                with open(temp_path, 'w', encoding='utf-8') as f:
                    json.dump(unique_variants, f, indent=2, ensure_ascii=False)
                self.logger.debug(f"Intra-temp deduplication: {len(variants)} → {len(unique_variants)} ({duplicates_removed} duplicates removed)")
            else:
                self.logger.debug(f"Intra-temp deduplication: {len(variants)} variants, no duplicates found")

            return duplicates_removed
        except Exception as e:
            self.logger.error(f"Failed to deduplicate temp.json: {e}")
            return 0

