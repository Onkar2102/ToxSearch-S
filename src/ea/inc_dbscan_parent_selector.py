"""Parent selection from IncDBSCAN density memory (temp.json = D_t).

All genomes are eligible for default mode, including noise (species_id / density_label = -1).
Does not read elites.json / archive.json.
Exploit / explore prefer active (non-frozen) clusters with label > 0.
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

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()


class IncDBSCANParentSelector:
    """Species-aware tournament over full density memory (clusters + noise pool)."""

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
    ) -> List[Tuple[int, List[Dict[str, Any]], float]]:
        fits = self._best_fitness(groups)
        pools = [(sid, gs, fits[sid]) for sid, gs in groups.items() if gs]
        pools.sort(key=lambda x: x[2], reverse=True)
        return pools

    def _highest(self, genomes: List[Dict[str, Any]]) -> Dict[str, Any]:
        return max(genomes, key=self._score)

    @staticmethod
    def _sample_random(genomes: List[Dict[str, Any]], k: int, exclude_ids: Optional[Set[Any]] = None) -> List[Dict[str, Any]]:
        pool = [g for g in genomes if exclude_ids is None or g.get("id") not in exclude_ids]
        if not pool:
            pool = list(genomes)
        if not pool:
            return []
        if len(pool) >= k:
            return random.sample(pool, k)
        return [random.choice(pool) for _ in range(k)]

    def _species_pools_for_modes(
        self,
        groups: Dict[int, List[Dict[str, Any]]],
        frozen_ids: Set[int],
    ) -> List[Tuple[int, List[Dict[str, Any]], float]]:
        """Active clusters (label > 0, not frozen); fallback to frozen; then any non-empty pool."""
        active = {
            sid: gs for sid, gs in groups.items()
            if sid > 0 and sid not in frozen_ids and gs
        }
        pools = self._sorted_pools(active)
        if pools:
            return pools
        frozen = {
            sid: gs for sid, gs in groups.items()
            if sid > 0 and sid in frozen_ids and gs
        }
        pools = self._sorted_pools(frozen)
        if pools:
            self.logger.warning("IncDBSCAN: no active species — falling back to frozen clusters")
            return pools
        # Last resort: include noise / anything labeled
        return self._sorted_pools(groups)

    def _select_default(self, groups: Dict[int, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """Default: two uniformly random genomes from the entire density memory."""
        all_g = [g for gs in groups.values() for g in gs]
        if not all_g:
            raise RuntimeError("IncDBSCAN: no genomes in density memory for parent selection")
        if len(all_g) >= 2:
            return random.sample(all_g, 2)
        return [all_g[0], all_g[0]]

    def _select_exploit(
        self,
        groups: Dict[int, List[Dict[str, Any]]],
        frozen_ids: Set[int],
    ) -> List[Dict[str, Any]]:
        """Top genome of best species + 2 random genomes from the same species."""
        pools = self._species_pools_for_modes(groups, frozen_ids)
        if not pools:
            raise RuntimeError("IncDBSCAN: no genomes for exploitation selection")
        _, top_gs, _ = pools[0]
        p1 = self._highest(top_gs)
        rest = self._sample_random(top_gs, 2, exclude_ids={p1.get("id")})
        if len(rest) < 2:
            rest = self._sample_random(top_gs, 2)
        return [p1, rest[0], rest[1]]

    def _select_explore(
        self,
        groups: Dict[int, List[Dict[str, Any]]],
        frozen_ids: Set[int],
    ) -> List[Dict[str, Any]]:
        """Best of top species; then one genome from each of two randomly chosen other species."""
        pools = self._species_pools_for_modes(groups, frozen_ids)
        if not pools:
            raise RuntimeError("IncDBSCAN: no genomes for exploration selection")
        top_id, top_gs, _ = pools[0]
        p1 = self._highest(top_gs)

        others = [p for p in pools if p[0] != top_id]
        if len(others) >= 2:
            chosen = random.sample(others, 2)
        elif len(others) == 1:
            chosen = [others[0], others[0]]
        else:
            # Single species: pad with random genomes from it
            pad = self._sample_random(top_gs, 2, exclude_ids={p1.get("id")})
            if len(pad) < 2:
                pad = self._sample_random(top_gs, 2)
            return [p1, pad[0], pad[1]]

        parents = [p1]
        for _, gs, _ in chosen:
            # Random genome from the randomly selected species
            parents.append(random.choice(gs))
        return parents

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
        # Eligible: any genome with a density_label or species_id (including -1 noise)
        eligible = [
            g for g in genomes
            if g.get("density_label") is not None or g.get("species_id") is not None
        ]
        if not eligible:
            self.logger.critical("IncDBSCAN: no labeled genomes in temp.json — cannot select parents")
            raise RuntimeError("No density-memory genomes available for parent selection.")

        groups = self._group(eligible)
        frozen_ids = self._load_frozen_ids(outputs_path)
        mode = "default"
        if evolution_tracker:
            mode = str(evolution_tracker.get("selection_mode", "default")).lower()

        if mode in ("exploit", "exploitation"):
            parents = self._select_exploit(groups, frozen_ids)
        elif mode in ("explore", "exploration"):
            parents = self._select_explore(groups, frozen_ids)
        else:
            parents = self._select_default(groups)

        slim = []
        for p in parents:
            # Ensure species_id mirrors density_label for freeze bookkeeping
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

        # top_10 from full density memory
        ranked = sorted(eligible, key=self._score, reverse=True)[:10]
        top_path = Path(outputs_path) / "top_10.json"
        with open(top_path, "w", encoding="utf-8") as f:
            json.dump(
                [
                    slim_selection_genome(g, self.north_star_metric, include_prompt=True, include_species_id=True)
                    for g in ranked
                ],
                f,
                indent=2,
                ensure_ascii=False,
            )
