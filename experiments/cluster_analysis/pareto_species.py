"""
Incremental Pareto-dominance bookkeeping for capped species sets (offline analysis).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Hashable, Iterable, List, Optional, Set, Tuple

import numpy as np

from pareto_mo_utils import crowding_distance, dominates, fast_non_dominated_sort


@dataclass
class MoIndividual:
    id: Hashable
    objectives: np.ndarray
    dominated_individuals: Set["MoIndividual"] = field(default_factory=set)
    dominated_by_count: int = 0

    def __hash__(self) -> int:
        return hash((type(self), self.id))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MoIndividual) and self.id == other.id


class ParetoSpecies:
    def __init__(
        self,
        max_size: Optional[int],
        *,
        maximize: bool = True,
        members: Optional[Iterable[MoIndividual]] = None,
    ) -> None:
        self.max_size = max_size
        self._maximize = maximize
        self.members: Dict[Hashable, MoIndividual] = {}
        if members:
            for m in members:
                self.add(m)

    def __len__(self) -> int:
        return len(self.members)

    def add(self, p: MoIndividual) -> None:
        if p.id in self.members:
            raise ValueError(f"individual id already in species: {p.id!r}")
        for s in list(self.members.values()):
            if dominates(p.objectives, s.objectives, maximize=self._maximize):
                p.dominated_individuals.add(s)
                s.dominated_by_count += 1
            elif dominates(s.objectives, p.objectives, maximize=self._maximize):
                s.dominated_individuals.add(p)
                p.dominated_by_count += 1
        self.members[p.id] = p
        self._trim_if_needed()

    def _trim_if_needed(self) -> None:
        if self.max_size is None:
            return
        while len(self.members) > self.max_size and any(
            ind.dominated_by_count > 0 for ind in self.members.values()
        ):
            s_removed = self._pick_most_dominated()
            self._remove_and_repair(s_removed)
        if len(self.members) > self.max_size:
            self._crowding_prune_f0()

    def _pick_most_dominated(self) -> MoIndividual:
        best: Optional[MoIndividual] = None
        best_key: Tuple[int, str] = (-1, "")
        for s in self.members.values():
            key = (s.dominated_by_count, str(s.id))
            if best is None or key > best_key:
                best = s
                best_key = key
        assert best is not None
        return best

    def _remove_and_repair(self, s_removed: MoIndividual) -> None:
        if s_removed.id not in self.members:
            return
        del self.members[s_removed.id]
        for s in list(self.members.values()):
            if s in s_removed.dominated_individuals:
                s.dominated_by_count -= 1
            if s_removed in s.dominated_individuals:
                s.dominated_individuals.discard(s_removed)
        s_removed.dominated_individuals.clear()
        s_removed.dominated_by_count = 0

    def _crowding_prune_f0(self) -> None:
        assert self.max_size is not None
        ids = list(self.members.keys())
        F = np.stack([self.members[i].objectives for i in ids], axis=0)
        fronts = fast_non_dominated_sort(F)
        if not fronts or not fronts[0]:
            return
        f0_local = list(range(len(ids)))
        cd = crowding_distance(F, f0_local)
        order = sorted(
            range(len(ids)),
            key=lambda i: (cd[i], str(ids[i])),
        )
        while len(self.members) > self.max_size:
            drop_id = ids[order.pop(0)]
            self._remove_and_repair(self.members[drop_id])

    def pareto_front_ids(self) -> List[Hashable]:
        return [ind.id for ind in self.members.values() if ind.dominated_by_count == 0]


__all__ = ["MoIndividual", "ParetoSpecies"]
