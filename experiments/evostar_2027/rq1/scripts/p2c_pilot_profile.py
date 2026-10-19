#!/usr/bin/env python3
"""Part 2c: pilot call-count / wall summary → memos/pilot_profile.json.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p2c_pilot_profile.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PILOTS = Path("experiments/evostar_2027/rq1/pilots")
DEFAULT_OUT = Path("experiments/evostar_2027/rq1/memos/pilot_profile.json")


def _load_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size < 3:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _profile_cell(cell_dir: Path) -> Dict[str, Any]:
    cell = _load_json(cell_dir / "cell.json") or {}
    run_dir = cell_dir / "run"
    tracker = _load_json(run_dir / "EvolutionTracker.json") if run_dir.is_dir() else None
    snap = None
    if run_dir.is_dir():
        # Optional speciation snapshot with NLI stats
        for name in ("speciation_metrics.json", "nli_stats.json"):
            snap = _load_json(run_dir / name)
            if snap:
                break

    wall = None
    genomes = None
    nli = {}
    if isinstance(tracker, dict):
        wall = tracker.get("wall_seconds") or tracker.get("elapsed_seconds")
        genomes = tracker.get("total_genomes") or tracker.get("n_genomes")
        meta = tracker.get("run_metadata") or {}
        if isinstance(meta, dict) and isinstance(meta.get("nli"), dict):
            nli = meta["nli"]
    if isinstance(snap, dict) and not nli:
        nli = snap.get("nli") or snap

    return {
        "cell_id": cell.get("cell_id", cell_dir.name),
        "distance_method": cell.get("distance_method"),
        "clustering_method": cell.get("clustering_method"),
        "seed": cell.get("seed"),
        "mode": cell.get("mode"),
        "max_total_genomes": cell.get("max_total_genomes"),
        "wall_seconds": wall,
        "total_genomes": genomes,
        "nli_model_passes": nli.get("model_passes") if isinstance(nli, dict) else None,
        "nli_cache_hits": nli.get("cache_hits") if isinstance(nli, dict) else None,
        "nli_distance_calls": nli.get("distance_calls") if isinstance(nli, dict) else None,
        "has_run_dir": run_dir.is_dir(),
        "dry_run_only": cell.get("mode") == "dry-run" and not (run_dir / "EvolutionTracker.json").is_file(),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p2c: pilot wall / NLI profile")
    p.add_argument("--pilots-dir", type=Path, default=DEFAULT_PILOTS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()

    cells: List[Dict[str, Any]] = []
    if args.pilots_dir.is_dir():
        for cell_dir in sorted(args.pilots_dir.iterdir()):
            if cell_dir.is_dir():
                cells.append(_profile_cell(cell_dir))

    profile = {
        "n_cells": len(cells),
        "n_with_run_artifacts": sum(1 for c in cells if c.get("has_run_dir") and not c.get("dry_run_only")),
        "cells": cells,
        "note": (
            "Dry-run cells have command wiring only. Re-run p2a with --run to populate "
            "wall / NLI counters from EvolutionTracker."
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} ({len(cells)} cells)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
