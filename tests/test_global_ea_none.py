"""Tests for clustering_method=none (Global-EA)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from ea.global_parent_selector import GlobalParentSelector
from speciation.clustering_mode import is_global_mode, is_inc_dbscan_mode
from speciation.config import SpeciationConfig
from speciation.run_global import GLOBAL_SPECIES_ID, run_global_speciation
from speciation.run_speciation import run_speciation


def _g(gid, sid, score):
    return {
        "id": gid,
        "species_id": sid,
        "prompt": f"q{gid}?",
        "moderation_result": {"google": {"scores": {"toxicity": score}}},
    }


class TestConfigAndMode:
    def test_config_accepts_none(self):
        cfg = SpeciationConfig(clustering_method="none", theta_sim=0.1, theta_merge=0.1)
        assert cfg.clustering_method == "none"

    def test_config_normalizes_global_alias(self):
        cfg = SpeciationConfig(clustering_method="global", theta_sim=0.1, theta_merge=0.1)
        assert cfg.clustering_method == "none"

    def test_is_global_mode(self):
        assert is_global_mode(clustering_method="none")
        assert is_global_mode(clustering_method="global")
        assert not is_global_mode(clustering_method="leader_follower")
        assert not is_inc_dbscan_mode(clustering_method="none")


class TestGlobalParentSelector:
    def setup_method(self):
        self.sel = GlobalParentSelector("toxicity")
        self.elites = [
            _g(1, 1, 0.9),
            _g(2, 1, 0.5),
            _g(3, 1, 0.4),
            _g(4, 1, 0.3),
            _g(5, 1, 0.7),
            _g(6, 1, 0.2),
        ]

    def test_default_two_parents(self):
        parents = self.sel._select_default(self.elites)
        assert len(parents) == 2
        assert all(p["id"] in {1, 2, 3, 4, 5, 6} for p in parents)

    def test_exploit_starts_with_global_best(self):
        parents = self.sel._select_exploit(self.elites)
        assert len(parents) == 3
        assert parents[0]["id"] == 1

    def test_explore_three_parents(self):
        parents = self.sel._select_explore(self.elites)
        assert len(parents) == 3


class TestRunGlobalSpeciation:
    def test_merges_temp_into_elites_and_assigns_species_id(self, tmp_path: Path):
        elites = [_g(1, 0, 0.5)]
        temp = [_g(2, None, 0.8), _g(3, None, 0.1)]
        (tmp_path / "elites.json").write_text(json.dumps(elites), encoding="utf-8")
        (tmp_path / "temp.json").write_text(json.dumps(temp), encoding="utf-8")
        (tmp_path / "archive.json").write_text("[]", encoding="utf-8")

        with patch("speciation.run_global.get_outputs_path", return_value=tmp_path):
            result = run_global_speciation(
                temp_path=str(tmp_path / "temp.json"),
                current_generation=1,
                north_star_metric="toxicity",
            )

        assert result["success"] is True
        assert result["population_mode"] == "global"
        assert result["global"] is True
        assert result["active_species_count"] == 1

        saved = json.loads((tmp_path / "elites.json").read_text(encoding="utf-8"))
        assert len(saved) == 3
        assert all(int(g["species_id"]) == GLOBAL_SPECIES_ID for g in saved)
        assert json.loads((tmp_path / "temp.json").read_text(encoding="utf-8")) == []

        state = json.loads((tmp_path / "speciation_state.json").read_text(encoding="utf-8"))
        assert state["clustering_method"] == "none"
        assert state["population_mode"] == "global"

    def test_run_speciation_dispatches_to_global(self, tmp_path: Path):
        (tmp_path / "elites.json").write_text("[]", encoding="utf-8")
        (tmp_path / "temp.json").write_text(
            json.dumps([_g(10, None, 0.4)]), encoding="utf-8"
        )
        (tmp_path / "archive.json").write_text("[]", encoding="utf-8")
        cfg = SpeciationConfig(clustering_method="none", theta_sim=0.1, theta_merge=0.1)

        with patch("speciation.run_global.get_outputs_path", return_value=tmp_path):
            # run_speciation also calls get_outputs_path after dispatch; patch both modules.
            with patch("speciation.run_speciation.get_outputs_path", return_value=tmp_path):
                result = run_speciation(
                    temp_path=str(tmp_path / "temp.json"),
                    current_generation=0,
                    config=cfg,
                    north_star_metric="toxicity",
                )

        assert result.get("population_mode") == "global"
        assert result.get("clustering_method") == "none"
        saved = json.loads((tmp_path / "elites.json").read_text(encoding="utf-8"))
        assert len(saved) == 1
        assert saved[0]["species_id"] == GLOBAL_SPECIES_ID
