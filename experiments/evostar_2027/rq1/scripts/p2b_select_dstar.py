#!/usr/bin/env python3
"""Part 2b: aggregate finished pilot cells → selection_memo.json.

Lexicographic rule (locked):
  (1) higher best fitness
  (2) higher top-10 mean fitness
  (3) higher mature species count (non-degenerate)
  (4) lower NLI/wall cost if tied

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p2b_select_dstar.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PILOTS = Path("experiments/evostar_2027/rq1/pilots")
DEFAULT_MEMO = Path("experiments/evostar_2027/rq1/memos/selection_memo.json")


def _load_json(path: Path) -> Any:
    if not path.is_file() or path.stat().st_size < 3:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fitness_from_genome(g: Dict[str, Any]) -> float:
    for k in ("north_star_score", "fitness"):
        if g.get(k) is not None:
            return float(g[k])
    return float("-inf")


def _summarize_run(run_dir: Path) -> Dict[str, Any]:
    elites = _load_json(run_dir / "elites.json") or []
    if isinstance(elites, dict):
        elites = list(elites.values())
    fits = sorted((_fitness_from_genome(g) for g in elites if isinstance(g, dict)), reverse=True)
    best = float(fits[0]) if fits else float("nan")
    top10 = float(sum(fits[:10]) / max(len(fits[:10]), 1)) if fits else float("nan")

    state = _load_json(run_dir / "speciation_state.json") or {}
    species = state.get("species") or state.get("active_species") or {}
    if isinstance(species, list):
        n_species = len(species)
    elif isinstance(species, dict):
        n_species = len(species)
    else:
        n_species = 0

    tracker = _load_json(run_dir / "EvolutionTracker.json") or {}
    wall = None
    nli_calls = None
    if isinstance(tracker, dict):
        wall = tracker.get("wall_seconds") or tracker.get("elapsed_seconds")
        meta = tracker.get("run_metadata") or {}
        nli_calls = (meta.get("nli") or {}).get("model_passes") if isinstance(meta, dict) else None

    return {
        "best_fitness": best,
        "top10_mean_fitness": top10,
        "n_elites": len(fits),
        "mature_species_count": int(n_species),
        "wall_seconds": wall,
        "nli_model_passes": nli_calls,
        "finished": bool(fits) or (run_dir / "EvolutionTracker.json").is_file(),
    }


def _rank_key(row: Dict[str, Any]) -> Tuple:
    # Higher better for first three; lower cost better for last
    wall = row.get("wall_seconds")
    nli = row.get("nli_model_passes")
    cost = float(nli if nli is not None else (wall if wall is not None else 0.0))
    return (
        float(row.get("best_fitness") or float("-inf")),
        float(row.get("top10_mean_fitness") or float("-inf")),
        int(row.get("mature_species_count") or 0),
        -cost,
    )


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p2b: select d* + clustering from pilots")
    p.add_argument("--pilots-dir", type=Path, default=DEFAULT_PILOTS)
    p.add_argument("--out", type=Path, default=DEFAULT_MEMO)
    args = p.parse_args()

    rows: List[Dict[str, Any]] = []
    if args.pilots_dir.is_dir():
        for cell_dir in sorted(args.pilots_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            cell = _load_json(cell_dir / "cell.json") or {}
            run_dir = cell_dir / "run"
            summary = _summarize_run(run_dir) if run_dir.is_dir() else {
                "best_fitness": float("nan"),
                "top10_mean_fitness": float("nan"),
                "n_elites": 0,
                "mature_species_count": 0,
                "wall_seconds": None,
                "nli_model_passes": None,
                "finished": False,
            }
            rows.append({
                "cell_id": cell.get("cell_id", cell_dir.name),
                "distance_method": cell.get("distance_method"),
                "clustering_method": cell.get("clustering_method"),
                "seed": cell.get("seed"),
                "mode": cell.get("mode"),
                **summary,
            })

    finished = [r for r in rows if r.get("finished")]
    winner: Optional[Dict[str, Any]] = None
    if finished:
        winner = max(finished, key=_rank_key)

    memo = {
        "rule": [
            "higher best_fitness",
            "higher top10_mean_fitness",
            "higher mature_species_count (non-degenerate)",
            "lower NLI/wall cost on ties",
        ],
        "n_cells_seen": len(rows),
        "n_finished": len(finished),
        "cells": rows,
        "selected": None if winner is None else {
            "d_star": winner.get("distance_method"),
            "clustering_star": winner.get("clustering_method"),
            "cell_id": winner.get("cell_id"),
            "metrics": {
                "best_fitness": winner.get("best_fitness"),
                "top10_mean_fitness": winner.get("top10_mean_fitness"),
                "mature_species_count": winner.get("mature_species_count"),
                "wall_seconds": winner.get("wall_seconds"),
                "nli_model_passes": winner.get("nli_model_passes"),
            },
        },
        "status": (
            "selected"
            if winner is not None
            else ("no_finished_cells" if rows else "no_pilot_cells")
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(memo, indent=2), encoding="utf-8")
    print(f"Wrote {args.out} status={memo['status']}")
    if winner:
        print(f"  d*={winner.get('distance_method')} clustering*={winner.get('clustering_method')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
