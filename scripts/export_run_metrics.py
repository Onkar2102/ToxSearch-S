#!/usr/bin/env python3
"""
Export run-level and generation-derived metrics into the run output directory.

Reads EvolutionTracker.json from the given run dir, computes run_summary,
termination_metrics, config_snapshot, and generation_metrics_derived, and
writes them as separate JSON files under the same directory. Does not modify
EvolutionTracker.json or any other existing files.

Usage (from project root):

  PYTHONPATH=src python scripts/export_run_metrics.py [run_dir]

  run_dir: path to a run output directory (e.g. data/outputs/20260311_1742).
           If omitted, uses the latest directory under data/outputs/.

Outputs (all under run_dir):

  - run_summary.json
  - termination_metrics.json (only if run_metadata has max_total_genomes)
  - config_snapshot.json
  - generation_metrics_derived.json
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

PROJ = Path(__file__).resolve().parents[1]
SRC = PROJ / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
elif str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

# Import after path setup so script can run with PYTHONPATH=src
try:
    from utils.population_io import compute_run_summary
except ImportError:
    compute_run_summary = None


def load_tracker(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "EvolutionTracker.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_latest_run_dir() -> Optional[Path]:
    base = PROJ / "data" / "outputs"
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def compute_termination_metrics(tracker: Dict[str, Any], run_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    run_meta = tracker.get("run_metadata") or {}
    threshold = run_meta.get("max_total_genomes")
    if threshold is None:
        return None
    te = run_summary.get("total_evaluated", 0)
    return {
        "termination_criterion": "max_total_genomes",
        "termination_threshold": threshold,
        "final_evaluated": te,
        "final_integrated": run_summary.get("total_integrated", 0),
        "overshoot_evaluated": max(0, te - threshold),
    }


def compute_config_snapshot(tracker: Dict[str, Any]) -> Dict[str, Any]:
    run_meta = tracker.get("run_metadata") or {}
    return dict(run_meta)


def compute_generation_metrics_derived(tracker: Dict[str, Any]) -> List[Dict[str, Any]]:
    gens = tracker.get("generations") or []
    derived = []
    prev_total_evaluated = 0
    for g in sorted(gens, key=lambda x: x.get("generation_number", 0)):
        gen_num = g.get("generation_number")
        duration = g.get("generation_duration_seconds")
        vi = g.get("variants_integrated")
        te_cum = g.get("total_evaluated")
        if te_cum is not None:
            te_this_gen = int(te_cum) - prev_total_evaluated
            prev_total_evaluated = int(te_cum)
        else:
            budget = g.get("budget") or {}
            te_this_gen = int(budget.get("llm_calls", 0) or 0)
        entry = {"generation_number": gen_num}
        if duration is not None and duration > 0 and te_this_gen is not None:
            entry["evaluated_per_second"] = round(te_this_gen / duration, 4)
        if te_this_gen is not None and te_this_gen > 0 and vi is not None:
            entry["accepted_ratio"] = round(vi / te_this_gen, 4)
        if duration is not None and vi is not None:
            entry["generation_duration_seconds"] = round(duration, 3)
            entry["variants_integrated"] = vi
        derived.append(entry)
    return derived


def main() -> int:
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = find_latest_run_dir()
    if not run_dir or not run_dir.exists():
        print("Usage: PYTHONPATH=src python scripts/export_run_metrics.py [run_dir]", file=sys.stderr)
        print("run_dir: e.g. data/outputs/20260311_1742", file=sys.stderr)
        return 1

    tracker = load_tracker(run_dir)
    if not tracker:
        print(f"EvolutionTracker.json not found in {run_dir}", file=sys.stderr)
        return 1

    if compute_run_summary is None:
        print("Could not import compute_run_summary from utils.population_io; run with PYTHONPATH=src", file=sys.stderr)
        return 1

    run_summary = compute_run_summary(tracker)

    with open(run_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=False)
    print(f"Wrote {run_dir / 'run_summary.json'}")

    term = compute_termination_metrics(tracker, run_summary)
    if term is not None:
        with open(run_dir / "termination_metrics.json", "w", encoding="utf-8") as f:
            json.dump(term, f, indent=2, ensure_ascii=False)
        print(f"Wrote {run_dir / 'termination_metrics.json'}")

    config = compute_config_snapshot(tracker)
    with open(run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Wrote {run_dir / 'config_snapshot.json'}")

    gen_derived = compute_generation_metrics_derived(tracker)
    with open(run_dir / "generation_metrics_derived.json", "w", encoding="utf-8") as f:
        json.dump(gen_derived, f, indent=2, ensure_ascii=False)
    print(f"Wrote {run_dir / 'generation_metrics_derived.json'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
