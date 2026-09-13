"""Serialize full run configuration for reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from speciation.config import SpeciationConfig


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


@dataclass
class RunConfig:
    """Snapshot of CLI and speciation settings for one evolution run."""

    evaluator: str
    north_star_metric: str
    max_total_genomes: int
    seed_file: str
    seed: int | None
    operators: str
    max_variants: int
    stagnation_limit: int
    rg_model: str
    pg_model: str
    openai_model: str
    parallel: bool
    output_dir: str | None
    speciation: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    git_commit: str | None = field(default_factory=_git_commit)
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_run_config(
    *,
    evaluator: str,
    north_star_metric: str,
    max_total_genomes: int,
    seed_file: str,
    seed: int | None,
    operators: str,
    max_variants: int,
    stagnation_limit: int,
    rg_model: str,
    pg_model: str,
    openai_model: str,
    parallel: bool,
    output_dir: str | None,
    speciation_config: SpeciationConfig,
) -> RunConfig:
    return RunConfig(
        evaluator=evaluator,
        north_star_metric=north_star_metric,
        max_total_genomes=max_total_genomes,
        seed_file=seed_file,
        seed=seed,
        operators=operators,
        max_variants=max_variants,
        stagnation_limit=stagnation_limit,
        rg_model=rg_model,
        pg_model=pg_model,
        openai_model=openai_model,
        parallel=parallel,
        output_dir=output_dir,
        speciation=speciation_config.to_dict(),
    )


def write_run_config(outputs_path: Path, config: RunConfig) -> Path:
    """Write ``run_config.json`` under the run output directory."""
    outputs_path.mkdir(parents=True, exist_ok=True)
    out = outputs_path / "run_config.json"
    out.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return out
