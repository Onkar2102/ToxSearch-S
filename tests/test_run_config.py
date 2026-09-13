"""Tests for run configuration snapshot."""

from __future__ import annotations

import json
from pathlib import Path

from speciation.config import SpeciationConfig
from utils.run_config import build_run_config, write_run_config


def test_write_run_config(tmp_path: Path) -> None:
    spec = SpeciationConfig()
    cfg = build_run_config(
        evaluator="google",
        north_star_metric="toxicity",
        max_total_genomes=150,
        seed_file="data/prompt.csv",
        seed=42,
        operators="all",
        max_variants=1,
        stagnation_limit=5,
        rg_model="models/x.gguf",
        pg_model="models/x.gguf",
        openai_model="omni-moderation-latest",
        parallel=False,
        output_dir=str(tmp_path),
        speciation_config=spec,
    )
    out = write_run_config(tmp_path, cfg)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["max_total_genomes"] == 150
    assert data["seed"] == 42
    assert data["speciation"]["theta_sim"] == spec.theta_sim
