

import random
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import defaultdict
from utils import get_custom_logging
from utils.population_io import load_elites, _extract_north_star_score, slim_selection_genome
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


class ParentSelector:
    """Parent selection from elites.json only (species_id > 0). Category 1 = active species; Category 2 = frozen when Category 1 is empty."""

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
            all_in_genomes = {sid for sid in all_species_in_genomes if sid is not None and sid > 0}
        else:
            all_in_genomes = {int(s) for s in species_dict.keys() if int(s) > 0} - frozen_ids
        
        category1_ids = all_in_genomes - frozen_ids

        self.logger.debug(f"Category1 (active elites): {sorted(category1_ids)}, Frozen: {sorted(frozen_ids)}")
        
        return (category1_ids, frozen_ids)

    def _group_by_species(self, genomes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        
        species_groups = defaultdict(list)
        for genome in genomes:
            species_id = genome.get("species_id")
            if species_id is not None and species_id > 0:
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
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        
        species_groups = self._group_by_species(elites)
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

    @staticmethod
    def _sample_random_from(
        genomes: List[Dict[str, Any]],
        k: int,
        exclude_ids: Optional[set] = None,
    ) -> List[Dict[str, Any]]:
        pool = [g for g in genomes if exclude_ids is None or g.get("id") not in exclude_ids]
        if not pool:
            pool = list(genomes)
        if not pool:
            return []
        if len(pool) >= k:
            return random.sample(pool, k)
        return [random.choice(pool) for _ in range(k)]

    def _select_parents_exploitation(
        self,
        elites: List[Dict[str, Any]],
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        """Top genome of best species + 2 random genomes from the same species."""
        species_groups = self._group_by_species(elites)
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)

        if not sorted_species:
            raise RuntimeError(
                "No genomes in this category (sorted_species empty); should not happen after adaptive_tournament_selection."
            )

        _top_species_id, top_genomes, _top_fitness = sorted_species[0]
        parent1 = self._get_genome_with_highest_fitness(top_genomes)
        rest = self._sample_random_from(top_genomes, 2, exclude_ids={parent1.get("id")})
        if len(rest) < 2:
            rest = self._sample_random_from(top_genomes, 2)
        return [parent1, rest[0], rest[1]]

    def _select_parents_exploration(
        self,
        elites: List[Dict[str, Any]],
        active_species_ids: set,
        outputs_path: str = None,
    ) -> List[Dict[str, Any]]:
        """Best of top species; then one genome from each of two randomly chosen other species."""
        species_groups = self._group_by_species(elites)
        sorted_species = self._get_sorted_active_species(species_groups, active_species_ids)

        if not sorted_species:
            raise RuntimeError(
                "No genomes in this category (sorted_species empty); should not happen after adaptive_tournament_selection."
            )

        first_id, first_genomes, _first_fit = sorted_species[0]
        parent1 = self._get_genome_with_highest_fitness(first_genomes)

        others = [sp for sp in sorted_species if sp[0] != first_id]
        if len(others) >= 2:
            chosen = random.sample(others, 2)
            return [parent1, random.choice(chosen[0][1]), random.choice(chosen[1][1])]

        if len(others) == 1:
            # Only one other species: sample twice from it (may repeat)
            other_gs = others[0][1]
            return [parent1, random.choice(other_gs), random.choice(other_gs)]

        # Single species: pad with random genomes from it
        pad = self._sample_random_from(first_genomes, 2, exclude_ids={parent1.get("id")})
        if len(pad) < 2:
            pad = self._sample_random_from(first_genomes, 2)
        return [parent1, pad[0], pad[1]]

    def adaptive_tournament_selection(self, evolution_tracker: Dict[str, Any] = None, outputs_path: str = None, current_generation: int = None) -> None:
        
        try:
            if outputs_path is None:
                outputs_path = get_outputs_path()

            elites_path = str(Path(outputs_path) / "elites.json")

            elites = load_elites(elites_path, log_file=None)
            elites = [g for g in elites if g.get("species_id") is not None and g.get("species_id", 0) > 0]

            if not elites:
                self.logger.critical("No genomes in elites.json with species_id > 0 - evolution cannot continue")
                raise RuntimeError("No elite genomes available - evolution cannot continue.")

            species_groups = self._group_by_species(elites)
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
                        "No genomes in active or frozen elites - evolution cannot continue."
                    )

            self.logger.debug(f"Using category with IDs: {sorted(ids_to_use)}")

            selection_mode = "default"
            if evolution_tracker:
                selection_mode = evolution_tracker.get("selection_mode", "default").lower()

            self.logger.debug(f"Selection mode: {selection_mode}")

            if selection_mode == "exploit" or selection_mode == "exploitation":
                selected_parents = self._select_parents_exploitation(elites, ids_to_use, outputs_path)
            elif selection_mode == "explore" or selection_mode == "exploration":
                selected_parents = self._select_parents_exploration(elites, ids_to_use, outputs_path)
            else:
                selected_parents = self._select_parents_default(elites, ids_to_use, outputs_path)

            expected_count = 3 if selection_mode in ["exploit", "exploitation", "explore", "exploration"] else 2
            if len(selected_parents) < expected_count:
                self.logger.warning(f"Only {len(selected_parents)} parents selected, expected {expected_count}")

            self.logger.debug(f"Selected {len(selected_parents)} parents: {[p.get('id') for p in selected_parents]}")

            self._save_parents_to_file(selected_parents, outputs_path)

            self._save_top_10_by_toxicity(elites_path, str(Path(outputs_path) / "top_10.json"))

        except Exception as e:
            self.logger.error(f"Error in adaptive tournament selection: {e}")
            raise

    def _save_parents_to_file(self, parents: List[Dict], outputs_path: str = None) -> None:
        
        try:
            slim_parents = [
                slim_selection_genome(
                    parent,
                    self.north_star_metric,
                    include_prompt=True,
                    include_species_id=True,
                )
                for parent in parents
            ]

            parents_path = Path(outputs_path) / "parents.json"
            parents_path.parent.mkdir(exist_ok=True)

            with open(parents_path, 'w', encoding='utf-8') as f:
                json.dump(slim_parents, f, indent=2, ensure_ascii=False)

            self.logger.debug(
                "Saved %d slimmed parents to %s (score key=%s)",
                len(slim_parents), parents_path, self.north_star_metric,
            )

        except Exception as e:
            self.logger.error(f"Failed to save parents to file: {e}")
            raise

    def _save_top_10_by_toxicity(self, elites_path: str = None, output_path: str = None) -> None:
        """Write top_10.json ranked by the active north-star metric (score key = metric name)."""
        try:
            if elites_path is None:
                outputs_path = get_outputs_path()
                elites_path = str(outputs_path / "elites.json")
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

            elites = [g for g in elites if g.get("species_id") is not None and g.get("species_id", 0) > 0]

            if not elites:
                self.logger.error("No elite genomes found for top_10")
                return

            sorted_genomes = sorted(elites, key=lambda g: _extract_north_star_score(g, self.north_star_metric), reverse=True)
            top_10_slim = [
                slim_selection_genome(genome, self.north_star_metric, include_prompt=True)
                for genome in sorted_genomes[:10]
            ]

            output_file = Path(output_path)
            output_file.parent.mkdir(exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(top_10_slim, f, indent=2, ensure_ascii=False)
            self.logger.debug(
                "Saved top 10 slimmed genomes to %s (score key=%s)",
                output_path, self.north_star_metric,
            )
        except Exception as e:
            self.logger.error(f"Failed to save top 10 genomes: {e}")
