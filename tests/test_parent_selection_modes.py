"""Parent selection rules for L–F and IncDBSCAN (default / exploit / explore)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from ea.parent_selector import ParentSelector
from ea.inc_dbscan_parent_selector import IncDBSCANParentSelector
from speciation.run_inc_dbscan import apply_inc_dbscan_stagnation_freeze
from speciation.species import Individual, Species


def _g(gid, sid, score, prompt=None):
    return {
        "id": gid,
        "species_id": sid,
        "density_label": sid,
        "prompt": prompt or f"q{gid}?",
        "moderation_result": {"google": {"scores": {"toxicity": score}}},
    }


class TestLeaderFollowerSelection:
    def setup_method(self):
        self.sel = ParentSelector("toxicity")
        self.elites = [
            _g(1, 10, 0.9),
            _g(2, 10, 0.5),
            _g(3, 10, 0.4),
            _g(4, 10, 0.3),
            _g(5, 20, 0.7),
            _g(6, 20, 0.2),
            _g(7, 30, 0.6),
            _g(8, 30, 0.1),
        ]
        self.active = {10, 20, 30}

    def test_exploit_top_plus_two_random_same_species(self):
        parents = self.sel._select_parents_exploitation(self.elites, self.active)
        assert len(parents) == 3
        assert parents[0]["id"] == 1  # top of species 10
        assert all(p["species_id"] == 10 for p in parents)

    def test_explore_three_random_species(self):
        parents = self.sel._select_parents_exploration(self.elites, self.active)
        assert len(parents) == 3
        sids = [p["species_id"] for p in parents]
        assert all(sid in self.active for sid in sids)
        # With 3+ breedable species, parents come from 3 distinct niches
        assert len(set(sids)) == 3


class TestIncDBSCANSelection:
    def setup_method(self):
        self.sel = IncDBSCANParentSelector("toxicity")
        self.groups = {
            10: [_g(1, 10, 0.9), _g(2, 10, 0.5), _g(3, 10, 0.4)],
            20: [_g(5, 20, 0.7), _g(6, 20, 0.2)],
            30: [_g(7, 30, 0.6), _g(8, 30, 0.1)],
            0: [_g(50, 0, 0.15), _g(51, 0, 0.12)],
            -1: [_g(99, -1, 0.05)],
        }

    def _pools(self, frozen_ids=None):
        return self.sel._resolve_breedable_pools(self.groups, frozen_ids or set())

    def test_category1_includes_active_mature_and_noise(self):
        pools = self._pools()
        ids = {sid for sid, _, _ in pools}
        assert ids == {10, 20, 30, 0}
        assert -1 not in ids

    def test_category1_empty_falls_back_to_frozen(self):
        # Only frozen mature + no noise
        groups = {
            10: [_g(1, 10, 0.9), _g(2, 10, 0.5)],
            20: [_g(5, 20, 0.7)],
        }
        pools = self.sel._resolve_breedable_pools(groups, frozen_ids={10, 20})
        assert {sid for sid, _, _ in pools} == {10, 20}

    def test_default_two_from_breedable_set(self):
        pools = self._pools()
        parents = self.sel._select_default(pools)
        assert len(parents) == 2
        breedable_ids = {g["id"] for _, gs, _ in pools for g in gs}
        assert parents[0]["id"] in breedable_ids
        assert parents[1]["id"] in breedable_ids
        assert 99 not in breedable_ids  # residual -1 excluded

    def test_exploit_top_plus_two_from_same_species(self):
        parents = self.sel._select_exploit(self._pools())
        assert len(parents) == 3
        assert parents[0]["id"] == 1
        assert all(p["species_id"] == 10 for p in parents)

    def test_exploit_skips_frozen_top_species(self):
        parents = self.sel._select_exploit(self._pools(frozen_ids={10}))
        assert parents[0]["species_id"] == 20  # next-best in Category 1
        assert all(p["species_id"] == 20 for p in parents)

    def test_exploit_can_use_noise_when_only_noise_in_category1(self):
        groups = {0: [_g(50, 0, 0.4), _g(51, 0, 0.3), _g(52, 0, 0.2)]}
        pools = self.sel._resolve_breedable_pools(groups, frozen_ids=set())
        parents = self.sel._select_exploit(pools)
        assert len(parents) == 3
        assert all(p["species_id"] == 0 for p in parents)
        assert parents[0]["id"] == 50

    def test_explore_three_random_species(self):
        pools = self._pools()
        breedable = {sid for sid, _, _ in pools}
        parents = self.sel._select_explore(pools)
        assert len(parents) == 3
        sids = [p["species_id"] for p in parents]
        assert all(sid in breedable for sid in sids)
        # With 3+ breedable species/pools, parents come from 3 distinct niches
        assert len(set(sids)) == 3


class TestIncDBSCANFreeze:
    def test_freeze_when_parent_species_does_not_improve(self, tmp_path: Path):
        outputs = tmp_path
        state_path = outputs / "speciation_state.json"
        # Previous max for species 10 was 0.9
        state_path.write_text(json.dumps({
            "species": {
                "10": {"max_fitness": 0.9, "species_state": "active", "stagnation": 19},
            },
            "prev_max_fitness": {"10": 0.9},
        }))
        (outputs / "parents.json").write_text(json.dumps([
            {"id": 1, "species_id": 10, "toxicity": 0.9},
        ]))

        leader = Individual(id=1, prompt="q?", fitness=0.9)
        sp = Species(
            id=10, leader=leader, members=[leader], radius=0.25,
            stagnation=19, max_fitness=0.9, species_state="active",
        )
        genomes = [_g(1, 10, 0.9)]  # no improvement
        logger = MagicMock()

        frozen = apply_inc_dbscan_stagnation_freeze(
            {10: sp},
            genomes,
            current_generation=5,
            species_stagnation=20,
            outputs_path=outputs,
            state_path=state_path,
            north_star_metric="toxicity",
            logger=logger,
        )
        assert frozen == 1
        assert sp.species_state == "frozen"
        assert sp.stagnation >= 20

    def test_no_freeze_on_improvement(self, tmp_path: Path):
        outputs = tmp_path
        state_path = outputs / "speciation_state.json"
        state_path.write_text(json.dumps({
            "species": {"10": {"max_fitness": 0.5, "species_state": "active", "stagnation": 5}},
            "prev_max_fitness": {"10": 0.5},
        }))
        (outputs / "parents.json").write_text(json.dumps([
            {"id": 1, "species_id": 10, "toxicity": 0.5},
        ]))
        leader = Individual(id=1, prompt="q?", fitness=0.5)
        sp = Species(
            id=10, leader=leader, members=[leader], radius=0.25,
            stagnation=5, max_fitness=0.5, species_state="active",
        )
        genomes = [_g(1, 10, 0.8)]
        frozen = apply_inc_dbscan_stagnation_freeze(
            {10: sp},
            genomes,
            current_generation=3,
            species_stagnation=20,
            outputs_path=outputs,
            state_path=state_path,
            north_star_metric="toxicity",
            logger=MagicMock(),
        )
        assert frozen == 0
        assert sp.species_state == "active"
        assert sp.stagnation == 0
        assert sp.max_fitness == pytest.approx(0.8)
