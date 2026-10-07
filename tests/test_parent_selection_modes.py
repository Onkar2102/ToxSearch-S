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

    def test_explore_top_then_two_other_species(self):
        parents = self.sel._select_parents_exploration(self.elites, self.active)
        assert len(parents) == 3
        assert parents[0]["id"] == 1
        assert parents[1]["species_id"] in {20, 30}
        assert parents[2]["species_id"] in {20, 30}


class TestIncDBSCANSelection:
    def setup_method(self):
        self.sel = IncDBSCANParentSelector("toxicity")
        self.groups = {
            10: [_g(1, 10, 0.9), _g(2, 10, 0.5), _g(3, 10, 0.4)],
            20: [_g(5, 20, 0.7), _g(6, 20, 0.2)],
            30: [_g(7, 30, 0.6), _g(8, 30, 0.1)],
            -1: [_g(99, -1, 0.05)],
        }

    def test_default_two_random_from_entire_population(self):
        parents = self.sel._select_default(self.groups)
        assert len(parents) == 2
        all_ids = {g["id"] for gs in self.groups.values() for g in gs}
        assert parents[0]["id"] in all_ids
        assert parents[1]["id"] in all_ids

    def test_exploit_top_plus_two_from_same_species(self):
        parents = self.sel._select_exploit(self.groups, frozen_ids=set())
        assert len(parents) == 3
        assert parents[0]["id"] == 1
        assert all(p["species_id"] == 10 for p in parents)

    def test_exploit_skips_frozen_top_species(self):
        parents = self.sel._select_exploit(self.groups, frozen_ids={10})
        assert parents[0]["species_id"] == 20  # next-best active
        assert all(p["species_id"] == 20 for p in parents)

    def test_explore_top_then_other_species(self):
        parents = self.sel._select_explore(self.groups, frozen_ids=set())
        assert len(parents) == 3
        assert parents[0]["id"] == 1
        assert parents[1]["species_id"] in {20, 30}
        assert parents[2]["species_id"] in {20, 30}


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
