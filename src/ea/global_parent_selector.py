"""Global parent selection (``clustering_method=none``): no mating restriction."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import get_custom_logging
from utils import get_system_utils
from utils.population_io import load_elites, _extract_north_star_score, slim_selection_genome

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


class GlobalParentSelector:
    """Select parents from the whole elite pool without species constraints.

    Modes (from EvolutionTracker.selection_mode):
      - default: 2 random elites
      - exploit: global best + 2 random elites
      - explore: 3 random elites
    """

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        self.north_star_metric = north_star_metric
        self.logger = get_logger("GlobalParentSelector", log_file)
        self.logger.debug(
            "GlobalParentSelector initialized with north_star_metric=%s",
            north_star_metric,
        )

    @staticmethod
    def _sample(genomes: List[Dict[str, Any]], k: int, exclude_ids: Optional[set] = None) -> List[Dict[str, Any]]:
        pool = [g for g in genomes if exclude_ids is None or g.get("id") not in exclude_ids]
        if not pool:
            pool = list(genomes)
        if not pool:
            return []
        if len(pool) >= k:
            return random.sample(pool, k)
        return [random.choice(pool) for _ in range(k)]

    def _best(self, genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(genomes, key=lambda g: _extract_north_star_score(g, self.north_star_metric))

    def _select_default(self, elites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._sample(elites, 2)

    def _select_exploit(self, elites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        best = self._best(elites)
        rest = self._sample(elites, 2, exclude_ids={best.get("id")})
        if len(rest) < 2:
            rest = self._sample(elites, 2)
        return [best, rest[0], rest[1]]

    def _select_explore(self, elites: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return self._sample(elites, 3)

    def adaptive_tournament_selection(
        self,
        evolution_tracker: Dict[str, Any] = None,
        outputs_path: str = None,
        current_generation: int = None,
    ) -> None:
        del current_generation
        try:
            if outputs_path is None:
                outputs_path = get_outputs_path()

            elites_path = str(Path(outputs_path) / "elites.json")
            elites = load_elites(elites_path, log_file=None)
            # Include all non-archive elites (Global-EA uses species_id=1).
            elites = [
                g
                for g in elites
                if g.get("species_id") is not None and int(g.get("species_id", -1)) >= 0
            ]
            if not elites:
                self.logger.critical("No genomes in elites.json — evolution cannot continue")
                raise RuntimeError("No elite genomes available - evolution cannot continue.")

            selection_mode = "default"
            if evolution_tracker:
                selection_mode = str(evolution_tracker.get("selection_mode", "default")).lower()

            self.logger.debug("Global selection mode: %s (n_elites=%d)", selection_mode, len(elites))

            if selection_mode in ("exploit", "exploitation"):
                selected = self._select_exploit(elites)
            elif selection_mode in ("explore", "exploration"):
                selected = self._select_explore(elites)
            else:
                selected = self._select_default(elites)

            expected = 3 if selection_mode in ("exploit", "exploitation", "explore", "exploration") else 2
            if len(selected) < expected:
                self.logger.warning(
                    "Only %d parents selected, expected %d", len(selected), expected
                )

            self._save_parents_to_file(selected, outputs_path)
            self._save_top_10(elites_path, str(Path(outputs_path) / "top_10.json"))
        except Exception as e:
            self.logger.error("Error in global adaptive tournament selection: %s", e)
            raise

    def _save_parents_to_file(self, parents: List[Dict], outputs_path: str = None) -> None:
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
        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(slim_parents, f, indent=2, ensure_ascii=False)
        self.logger.debug("Saved %d global parents to %s", len(slim_parents), parents_path)

    def _save_top_10(self, elites_path: str, output_path: str) -> None:
        elites_file = Path(elites_path)
        elites = []
        if elites_file.exists():
            with open(elites_file, "r", encoding="utf-8") as f:
                elites = json.load(f)
        elites = [
            g
            for g in elites
            if g.get("species_id") is not None and int(g.get("species_id", -1)) >= 0
        ]
        if not elites:
            self.logger.error("No elite genomes found for top_10")
            return
        sorted_genomes = sorted(
            elites,
            key=lambda g: _extract_north_star_score(g, self.north_star_metric),
            reverse=True,
        )
        top_10_slim = [
            slim_selection_genome(genome, self.north_star_metric, include_prompt=True)
            for genome in sorted_genomes[:10]
        ]
        out = Path(output_path)
        out.parent.mkdir(exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(top_10_slim, f, indent=2, ensure_ascii=False)
