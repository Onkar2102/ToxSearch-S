#!/usr/bin/env python3
"""
Post-run script: read worker_<rank>_stats.json and optional master_metrics.json
from a run directory, then write worker_metrics.json (and worker imbalance)
in the same run dir. Merges accepted_per_worker from master and optional
termination fields (in_flight_at_stop, discarded_buffered_at_stop).

Used for RQ1 (throughput, bottlenecks) and worker-load imbalance analysis.
Run after a parallel run; no MPI or main code required.

Usage (from project root):

  PYTHONPATH=src python scripts/aggregate_worker_metrics.py [run_dir]

  run_dir: path to a run output directory (e.g. data/outputs/20260311_1742).
           If omitted, uses the latest directory under data/outputs/.

Outputs (under run_dir):

  - worker_metrics.json: workers list (from worker_*_stats), accepted_per_worker
    (from master_metrics if present), worker_imbalance (max/min tasks and
    genomes_evaluated, straggler_gap), and optional termination (in_flight_at_stop,
    total_discarded_buffered_at_stop).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ = Path(__file__).resolve().parents[1]


def find_latest_run_dir() -> Optional[Path]:
    base = PROJ / "data" / "outputs"
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def load_worker_stats(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all worker_<rank>_stats.json files written by workers on exit."""
    workers = []
    for f in run_dir.glob("worker_*_stats.json"):
        try:
            with open(f, "r", encoding="utf-8") as fp:
                workers.append(json.load(fp))
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}", file=sys.stderr)
    return sorted(workers, key=lambda w: w.get("worker_id", -1))


def load_master_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load master_metrics.json (accepted_per_worker, in_flight_at_stop) if present."""
    path = run_dir / "master_metrics.json"
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: failed to read {path}: {e}", file=sys.stderr)
        return None


def compute_worker_imbalance(workers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute max/min tasks and genomes_evaluated and straggler_gap for load-balance analysis."""
    if not workers:
        return {"max_tasks": 0, "min_tasks": 0, "max_genomes_evaluated": 0, "min_genomes_evaluated": 0, "straggler_gap": 0}
    tasks = [w.get("tasks_received", 0) for w in workers]
    genomes = [w.get("genomes_evaluated", 0) for w in workers]
    return {
        "max_tasks": max(tasks),
        "min_tasks": min(tasks),
        "max_genomes_evaluated": max(genomes),
        "min_genomes_evaluated": min(genomes),
        "straggler_gap": max(genomes) - min(genomes) if genomes else 0,
    }


def main() -> int:
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = find_latest_run_dir()
    if not run_dir or not run_dir.exists():
        print("Usage: PYTHONPATH=src python scripts/aggregate_worker_metrics.py [run_dir]", file=sys.stderr)
        print("run_dir: e.g. data/outputs/20260311_1742", file=sys.stderr)
        return 1

    workers = load_worker_stats(run_dir)
    master = load_master_metrics(run_dir)

    out: Dict[str, Any] = {
        "workers": workers,
        "worker_imbalance": compute_worker_imbalance(workers),
    }
    if master:
        out["accepted_per_worker"] = master.get("accepted_per_worker", {})
        if master.get("in_flight_at_stop") is not None:
            out.setdefault("termination", {})["in_flight_at_stop"] = master["in_flight_at_stop"]
    total_discarded = sum(w.get("discarded_buffered_at_stop", 0) for w in workers)
    if total_discarded > 0 or (master and master.get("in_flight_at_stop") is not None):
        out.setdefault("termination", {})["total_discarded_buffered_at_stop"] = total_discarded

    out_path = run_dir / "worker_metrics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
