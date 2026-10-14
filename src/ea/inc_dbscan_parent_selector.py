"""Parent selection from IncDBSCAN density memory (temp.json = D_t).

Breedable set (before mode), same strategy as Leader–Follower:
  Category 1 = active mature species (label > 0, non-frozen) + noise/singleton pool (0).
  If Category 1 is empty, fall back to frozen mature species.

Does not read elites.json / archive.json.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from utils import get_custom_logging, get_system_utils
from utils.population_io import _extract_north_star_score, slim_selection_genome
from speciation.density_memory import NOISE_LABEL, load_density_memory
from speciation.clustering._io import SINGLETON_SPECIES_ID

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()

Pool = Tuple[int, List[Dict[str, Any]], float]


class IncDBSCANParentSelector:
    """Species-aware tournament over density memory (mature clusters + singleton pool)."""

    def __init__(self, north_star_metric: str, log_file: Optional[str] = None):
        self.north_star_metric = north_star_metric
        self.logger = get_logger("IncDBSCANParentSelector", log_file)

    def _score(self, g: Dict[str, Any]) -> float:
        return float(_extract_north_star_score(g, self.north_star_metric))

    def _label(self, g: Dict[str, Any]) -> int:
        if g.get("density_label") is not None:
            return int(g["density_label"])
        sid = g.get("species_id")
        if sid is None:
            return NOISE_LABEL
        return int(sid)

    def _group(self, genomes: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
        groups: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        for g in genomes:
            groups[self._label(g)].append(g)
        return groups

    def _load_frozen_ids(self, outputs_path: str) -> Set[int]:
        path = Path(outputs_path) / "speciation_state.json"
        if not path.exists():
            return set()
        try:
            with open(path, "r", encoding="utf-8") as f:
                state = json.load(f)
        except Exception as e:
            self.logger.warning("Failed to load speciation_state for frozen species: %s", e)
            return set()
        frozen: Set[int] = set()
        for sid_str, sp in (state.get("species") or {}).items():
            if (sp or {}).get("species_state") == "frozen":
                try:
                    frozen.add(int(sid_str))
                except (TypeError, ValueError):
                    continue
        return frozen

    def _best_fitness(self, groups: Dict[int, List[Dict[str, Any]]]) -> Dict[int, float]:
        return {sid: max((self._score(g) for g in gs), default=0.0) for sid, gs in groups.items()}

    def _sorted_pools(
        self, groups: Dict[int, List[Dict[str, Any]]]
    ) -> List[Pool]:
        fits = self._best_fitness(groups)
        pools = [(sid, gs, fits[sid]) for sid, gs in groups.items() if gs]
        pools.sort(key=lambda x: x[2], reverse=True)
        return pools

    def _resolve_breedable_pools(
        self,
        groups: Dict[int, List[Dict[str, Any]]],
        frozen_ids: Set[int],
    ) -> List[Pool]:
        """Category 1 (active mature + noise) else frozen mature — same as L–F.

        Args:
            groups: Label → genomes from density memory.
            frozen_ids: Mature species ids marked frozen in speciation_state.

        Returns:
            Fitness-sorted pools used by default / exploit / explore.
        """
        category1: Dict[int, List[Dict[str, Any]]] = {}
        for sid, gs in groups.items():
            if not gs:
                continue
            # Active mature (non-frozen) + singleton/noise pool (0).
            if sid > 0 and sid not in frozen_ids:
                category1[sid] = gs
            elif sid == SINGLETON_SPECIES_ID:
                category1[sid] = gs

        pools = self._sorted_pools(category1)
        if pools:
            self.logger.debug(
                "IncDBSCAN Category 1 (active mature + noise): %s",
                sorted(category1.keys()),
            )
            return pools

        frozen = {
            sid: gs for sid, gs in groups.items()
            if sid > 0 and sid in frozen_ids and gs
        }
        pools = self._sorted_pools(frozen)
        if pools:
            self.logger.warning(
                "IncDBSCAN: Category 1 empty — falling back to frozen mature species %s",
                sorted(frozen.keys()),
            )
            return pools

        raise RuntimeError(
            "IncDBSCAN: no genomes in active mature+noise or frozen species — "
            "cannot select parents."
        )

    def _highest(self, genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(genomes, key=self._score)

    @staticmethod
    def _sample_random(
        genomes: List[Dict[str, Any]],
        k: int,
        exclude_ids: Optional[Set[Any]] = None,
    ) -> List[Dict[str, Any]]:
        pool = [g for g in genomes if exclude_ids is None or g.get("id") not in exclude_ids]
        if not pool:
            pool = list(genomes)
        if not pool:
            return []
        if len(pool) >= k:
            return random.sample(pool, k)
        return [random.choice(pool) for _ in range(k)]

    def _select_default(self, pools: List[Pool]) -> List[Dict[str, Any]]:
        """Pick a random breedable species; sample 2 genomes (L–F default parity)."""
        if not pools:
            raise RuntimeError("IncDBSCAN: empty breedable set for default selection")

        sid, genomes, _ = random.choice(pools)
        if len(genomes) >= 2:
            return random.sample(genomes, 2)

        all_cat = [g for _, gs, _ in pools for g in gs]
        if len(all_cat) >= 2:
            return random.sample(all_cat, 2)
        if len(all_cat) == 1:
            return [all_cat[0], all_cat[0]]
        raise RuntimeError("IncDBSCAN: no genomes in breedable set for default selection")

    def _select_exploit(self, pools: List[Pool]) -> List[Dict[str, Any]]:
        """Top genome of best breedable species + 2 random from the same species."""
        if not pools:
            raise RuntimeError("IncDBSCAN: no genomes for exploitation selection")
        _, top_gs, _ = pools[0]
        p1 = self._highest(top_gs)
        rest = self._sample_random(top_gs, 2, exclude_ids={p1.get("id")})
        if len(rest) < 2:
            rest = self._sample_random(top_gs, 2)
        return [p1, rest[0], rest[1]]

    def _select_explore(self, pools: List[Pool]) -> List[Dict[str, Any]]:
        """One random genome from each of three randomly chosen breedable species."""
        if not pools:
            raise RuntimeError("IncDBSCAN: no genomes for exploration selection")

        n = len(pools)
        if n >= 3:
            chosen = random.sample(pools, 3)
        else:
            chosen = [pools[i] for i in (random.choices(range(n), k=3))]

        return [random.choice(gs) for _, gs, _ in chosen]

    def adaptive_tournament_selection(
        self,
        evolution_tracker: Dict[str, Any] = None,
        outputs_path: str = None,
        current_generation: int = None,
    ) -> None:
        del current_generation
        if outputs_path is None:
            outputs_path = str(get_outputs_path())
        genomes = load_density_memory(str(Path(outputs_path) / "temp.json"), logger=self.logger)
        eligible = [
            g for g in genomes
            if g.get("density_label") is not None or g.get("species_id") is not None
        ]
        # Keep only mature + singleton pool labels for selection (drop residual -1).
        eligible = [g for g in eligible if self._label(g) >= 0]
        if not eligible:
            self.logger.critical("IncDBSCAN: no labeled genomes in temp.json — cannot select parents")
            raise RuntimeError("No density-memory genomes available for parent selection.")

        groups = self._group(eligible)
        frozen_ids = self._load_frozen_ids(outputs_path)
        pools = self._resolve_breedable_pools(groups, frozen_ids)

        mode = "default"
        if evolution_tracker:
            mode = str(evolution_tracker.get("selection_mode", "default")).lower()

        if mode in ("exploit", "exploitation"):
            parents = self._select_exploit(pools)
        elif mode in ("explore", "exploration"):
            parents = self._select_explore(pools)
        else:
            parents = self._select_default(pools)

        slim = []
        for p in parents:
            if p.get("species_id") is None and p.get("density_label") is not None:
                p = {**p, "species_id": int(p["density_label"])}
            elif p.get("density_label") is not None and p.get("species_id") != p.get("density_label"):
                p = {**p, "species_id": int(p["density_label"])}
            slim.append(
                slim_selection_genome(p, self.north_star_metric, include_prompt=True, include_species_id=True)
            )
        parents_path = Path(outputs_path) / "parents.json"
        parents_path.parent.mkdir(parents=True, exist_ok=True)
        with open(parents_path, "w", encoding="utf-8") as f:
            json.dump(slim, f, indent=2, ensure_ascii=False)
        self.logger.debug(
            "IncDBSCAN selected %d parents (mode=%s): %s",
            len(slim), mode, [p.get("id") for p in slim],
        )

        self._save_top_10_by_toxicity(
            str(Path(outputs_path) / "temp.json"),
            str(Path(outputs_path) / "top_10.json"),
        )

    def _save_top_10_by_toxicity(self, density_path: str = None, output_path: str = None) -> None:
        """Write top_10.json from density memory (temp.json); sid/density_label >= 0."""
        try:
            if density_path is None:
                density_path = str(get_outputs_path() / "temp.json")
            if output_path is None:
                output_path = str(get_outputs_path() / "top_10.json")
            path = Path(density_path)
            genomes: List[Dict[str, Any]] = []
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                genomes = raw if isinstance(raw, list) else []
            else:
                self.logger.warning("Density file not found: %s", density_path)
            eligible = [g for g in genomes if g and self._label(g) >= 0]
            if not eligible:
                self.logger.error("No density genomes found for top_10")
                return
            ranked = sorted(eligible, key=self._score, reverse=True)[:10]
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(
                    [
                        slim_selection_genome(
                            g, self.north_star_metric, include_prompt=True, include_species_id=True
                        )
                        for g in ranked
                    ],
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
            self.logger.debug("Saved top 10 from density memory to %s", output_path)
        except Exception as e:
            self.logger.error("Failed to save top 10 from density memory: %s", e)
