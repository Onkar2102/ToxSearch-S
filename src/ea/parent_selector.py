

import random
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from utils import get_custom_logging
from utils.population_io import load_elites, _extract_north_star_score
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()
CLUSTER_0_ID = 0


class ParentSelector:
    """Parent selection based on species with Category 1 (active + species 0) and Category 2 (frozen). Use Category 2 only when Category 1 has no genomes. Sorting uses actual max fitness over current genomes only (no merge with stored values). If no genomes in active, reserves, or frozen: raises an error to end the evolution run. Selection modes (applied to the chosen category): - DEFAULT: Pick any species (random from sorted), 2 parents; if chosen has <2, fill from category. - EXPLOIT: Pick species with highest max fitness, 3 parents; if <3, fill from category. - EXPLORE: Pick top + 2 random species, 1 parent (best) from each; if <3 species, reuse/fill from category."""

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        
        self.north_star_metric = north_star_metric
        self.logger = get_logger("ParentSelector", log_file)
        self.logger.debug(f"ParentSelector initialized with north_star_metric={north_star_metric}")

    def _load_speciation_state(self, outputs_path: str) -> Dict[str, Any]:
        
        speciation_state_path = Path(outputs_path) / "speciation_state.json"
        if not speciation_state_path.exists():
            self.logger.warning(f"Speciation state file not found: {speciation_state_path}")
            return {}
        
        try:
            with open(speciation_state_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load speciation state: {e}")
            return {}

    def _get_active_species_ids(self, speciation_state: Dict[str, Any], all_species_in_genomes: set = None) -> Tuple[set, set]:
        
        frozen_ids = set()
        species_dict = speciation_state.get("species", {})
        for sid_str, sp_data in species_dict.items():
            sid = int(sid_str)
            species_state = sp_data.get("species_state", "active")
            if species_state == "frozen":
                frozen_ids.add(sid)
        
        if all_species_in_genomes is not None:
            all_in_genomes = set(all_species_in_genomes)
        else:
            all_in_genomes = set(int(s) for s in species_dict.keys()) - frozen_ids
        
        category1_ids = (all_in_genomes - frozen_ids) | {CLUSTER_0_ID}
        
        self.logger.debug(f"Category1 (active+0): {sorted(category1_ids)}, Frozen: {sorted(frozen_ids)}")
        
        return (category1_ids, frozen_ids)

    def _group_by_species(self, genomes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        
        species_groups = defaultdict(list)
        for genome in genomes:
            species_id = genome.get("species_id")
            if species_id is not None:
                species_groups[species_id].append(genome)
        return dict(species_groups)

    def _calculate_species_best_fitness(self, species_groups: Dict[int, List[Dict[str, Any]]]) -> Dict[int, float]:
        
        best_fitness = {}
        for species_id, genomes in species_groups.items():
            if genomes:
                max_fitness = max(_extract_north_star_score(g, self.north_star_metric) for g in genomes)
                best_fitness[species_id] = max_fitness
            else:
                best_fitness[species_id] = 0.0
        return best_fitness

    def _get_sorted_active_species(
        self, 
        species_groups: Dict[int, List[Dict[str, Any]]], 
        active_species_ids: set
    ) -> List[Tuple[int, List[Dict[str, Any]], float]]:
        
        species_fitness = self._calculate_species_best_fitness(species_groups)
        
        active_species = []
        for species_id, genomes in species_groups.items():
            if species_id in active_species_ids and genomes:
                best_fit = species_fitness.get(species_id, 0.0)
                active_species.append((species_id, genomes, best_fit))
        
        active_species.sort(key=lambda x: x[2], reverse=True)
        
        return active_species

    def _get_genome_with_highest_fitness(self, genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        if not genomes:
            return {}
        return max(genomes, key=lambda g: _extract_north_star_score(g, self.north_star_metric))

    def _select_parents_default(
        self,
        elites: List[Dict[str, Any]],
        reserves: List[Dict[str, Any]],
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        
        all_genomes = elites + reserves
        species_groups = self._group_by_species(all_genomes)
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)

        if not sorted_species:
            raise RuntimeError(
                "No genomes in this category (sorted_species empty); should not happen after adaptive_tournament_selection."
            )

        selected_species = random.choice(sorted_species)
        sid, genomes, _ = selected_species

        if len(genomes) >= 2:
            return random.sample(genomes, 2)

        all_cat = [g for sid in active_species_ids for g in species_groups.get(sid, [])]
        if len(all_cat) >= 2:
            return random.sample(all_cat, 2)
        if len(all_cat) == 1:
            return random.choices(all_cat, k=2)
        raise RuntimeError("No genomes in this category (all_cat empty); cannot supply 2 parents.")

    def _select_parents_exploitation(
        self,
        elites: List[Dict[str, Any]],
        reserves: List[Dict[str, Any]],
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        
        all_genomes = elites + reserves
        species_groups = self._group_by_species(all_genomes)
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)

        if not sorted_species:
            raise RuntimeError(
                "No genomes in this category (sorted_species empty); should not happen after adaptive_tournament_selection."
            )

        top_species_id, top_genomes, top_fitness = sorted_species[0]

        if len(top_genomes) >= 3:
            return random.sample(top_genomes, 3)

        all_cat = [g for sid in active_species_ids for g in species_groups.get(sid, [])]
        if not all_cat:
            raise RuntimeError("No genomes in this category (all_cat empty); cannot supply 3 parents.")
        selected = list(top_genomes)
        needed = 3 - len(selected)
        selected.extend(random.choices(all_cat, k=needed))
        return selected[:3]

    def _select_parents_exploration(
        self,
        elites: List[Dict[str, Any]],
        reserves: List[Dict[str, Any]],
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        
        all_genomes = elites + reserves
        species_groups = self._group_by_species(all_genomes)
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)

        if not sorted_species:
            raise RuntimeError(
                "No genomes in this category (sorted_species empty); should not happen after adaptive_tournament_selection."
            )

        all_cat = [g for sid in active_species_ids for g in species_groups.get(sid, [])]

        first_id, first_genomes, first_fit = sorted_species[0]
        parent1 = self._get_genome_with_highest_fitness(first_genomes)

        if len(sorted_species) >= 3:
            other = random.choice(sorted_species[1:])
            parent2 = self._get_genome_with_highest_fitness(other[1])
            exclude = {first_id, other[0]}
            candidates = [sp for sp in sorted_species if sp[0] not in exclude]
            if candidates:
                parent3 = self._get_genome_with_highest_fitness(random.choice(candidates)[1])
            else:
                parent3 = parent1
            return [parent1, parent2, parent3]

        if len(sorted_species) == 2:
            second_id, second_genomes, _ = sorted_species[1]
            parent2 = self._get_genome_with_highest_fitness(second_genomes)
            parent3 = parent1
            return [parent1, parent2, parent3]

        if len(first_genomes) >= 3:
            return random.sample(first_genomes, 3)
        if not all_cat:
            raise RuntimeError("No genomes in this category (all_cat empty); cannot supply 3 parents.")
        selected = list(first_genomes)
        selected.extend(random.choices(all_cat, k=3 - len(selected)))
        return selected[:3]

    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None, outputs_path: str = None, current_generation: int = None) -> None:
        
        try:
            if outputs_path is None:
                outputs_path = get_outputs_path()

            elites_path = str(Path(outputs_path) / "elites.json")
            reserves_path = str(Path(outputs_path) / "reserves.json")

            elites = load_elites(elites_path, log_file=None)
            
            reserves = []
            reserves_file = Path(reserves_path)
            if reserves_file.exists():
                with open(reserves_file, 'r', encoding='utf-8') as f:
                    reserves = json.load(f)
            else:
                self.logger.warning(f"Reserves file not found: {reserves_path}")

            if not elites and not reserves:
                self.logger.critical("No genomes in elites.json or reserves.json - evolution cannot continue")
                raise RuntimeError("No genomes available - evolution cannot continue.")

            all_genomes = elites + reserves
            species_groups = self._group_by_species(all_genomes)
            all_species_in_genomes = set(species_groups.keys())

            speciation_state = self._load_speciation_state(outputs_path)
            category1_ids, frozen_ids = self._get_active_species_ids(speciation_state, all_species_in_genomes)

            sorted_cat1 = self._get_sorted_active_species(species_groups, category1_ids)
            if sorted_cat1:
                ids_to_use = category1_ids
            else:
                sorted_frozen = self._get_sorted_active_species(species_groups, frozen_ids)
                if sorted_frozen:
                    ids_to_use = frozen_ids
                else:
                    raise RuntimeError(
                        "No genomes in active, reserves, or frozen - evolution cannot continue."
                    )

            self.logger.debug(f"Using category with IDs: {sorted(ids_to_use)}")

            selection_mode = "default"
            if evolution_tracker:
                selection_mode = evolution_tracker.get("selection_mode", "default").lower()

            self.logger.debug(f"Selection mode: {selection_mode}")

            if selection_mode == "exploit" or selection_mode == "exploitation":
                selected_parents = self._select_parents_exploitation(elites, reserves, ids_to_use, outputs_path)
            elif selection_mode == "explore" or selection_mode == "exploration":
                selected_parents = self._select_parents_exploration(elites, reserves, ids_to_use, outputs_path)
            else:
                selected_parents = self._select_parents_default(elites, reserves, ids_to_use, outputs_path)

            expected_count = 3 if selection_mode in ["exploit", "exploitation", "explore", "exploration"] else 2
            if len(selected_parents) < expected_count:
                self.logger.warning(f"Only {len(selected_parents)} parents selected, expected {expected_count}")

            self.logger.debug(f"Selected {len(selected_parents)} parents: {[p.get('id') for p in selected_parents]}")

            self._save_parents_to_file(selected_parents, outputs_path)

            self._save_top_10_by_toxicity(elites_path, reserves_path, str(Path(outputs_path) / "top_10.json"))

        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
            raise

    def _save_parents_to_file(self, parents: List[Dict], outputs_path: str = None) -> None:
        
        try:
            slim_parents = []
            for parent in parents:
                toxicity_score = round(_extract_north_star_score(parent, self.north_star_metric), 4)
                slim_parent = {
                    "id": parent.get("id"),
                    "prompt": parent.get("prompt", ""),
                    "toxicity": toxicity_score,
                    "species_id": parent.get("species_id")
                }
                slim_parents.append(slim_parent)

            parents_path = Path(outputs_path) / "parents.json"
            parents_path.parent.mkdir(exist_ok=True)

            with open(parents_path, 'w', encoding='utf-8') as f:
                json.dump(slim_parents, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved {len(slim_parents)} slimmed parents to {parents_path}")

        except Exception as e:
            self.logger.error(f"Failed to save parents to file: {e}")
            raise

    def _save_top_10_by_toxicity(self, elites_path: str = None, reserves_path: str = None, output_path: str = None) -> None:
        
        try:
            if elites_path is None:
                outputs_path = get_outputs_path()
                elites_path = str(outputs_path / "elites.json")
            if reserves_path is None:
                outputs_path = get_outputs_path()
                reserves_path = str(outputs_path / "reserves.json")
            if output_path is None:
                outputs_path = get_outputs_path()
                output_path = str(outputs_path / "top_10.json")

            elites_file = Path(elites_path)
            elites = []
            if elites_file.exists():
                with open(elites_file, 'r', encoding='utf-8') as f:
                    elites = json.load(f)
            else:
                self.logger.warning(f"Elites file not found: {elites_path}")

            reserves_file = Path(reserves_path)
            reserves = []
            if reserves_file.exists():
                with open(reserves_file, 'r', encoding='utf-8') as f:
                    reserves = json.load(f)
            else:
                self.logger.warning(f"Reserves file not found: {reserves_path}")

            all_genomes = elites + reserves

            if not all_genomes:
                self.logger.error("No genomes found in elites or reserves")
                return

            sorted_genomes = sorted(all_genomes, key=lambda g: _extract_north_star_score(g, self.north_star_metric), reverse=True)
            top_10_full = sorted_genomes[:10]

            top_10_slim = []
            for genome in top_10_full:
                original_score = round(_extract_north_star_score(genome, self.north_star_metric), 4)
                slim_genome = {
                    "id": genome.get("id"),
                    "prompt": genome.get("prompt", ""),
                    "toxicity": original_score
                }
                top_10_slim.append(slim_genome)

            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(top_10_slim, f, indent=2, ensure_ascii=False)
            self.logger.debug(f"Saved top 10 slimmed genomes to {output_path}")
        except Exception as e:
            self.logger.error(f"Failed to save top 10 genomes: {e}")
