#!/usr/bin/env python3
"""
Compute experiment metrics from EvolutionTracker.json for reporting:

  1. Throughput: evaluated genomes per second (per generation and overall).
  2. Search performance: best toxicity over time; time to reach toxicity thresholds.

Usage (from project root):

  PYTHONPATH=src python scripts/experiment_metrics.py [run_dir]

  run_dir: path to a run directory (e.g. data/outputs/20260311_1742). Default: latest
           data/outputs/<timestamp>.

Outputs:
  - Printed summary (throughput, search performance, time-to-threshold).
  - run_dir/experiment_metrics.json (numeric metrics).
  - run_dir/figures/toxicity_vs_time.png (best fitness vs cumulative wall-clock time).
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Project root
PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

try:
    import matplotlib
    matplotlib.use("Agg")
    if str(PROJ / "src") not in sys.path:
        sys.path.insert(0, str(PROJ / "src"))
    from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

    configure_matplotlib_embedded_fonts()
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_tracker(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = run_dir / "EvolutionTracker.json"
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _evaluated_in_generation(gen: Dict[str, Any]) -> Optional[int]:
    """Number of genomes evaluated in this generation (moderation / pipeline completions)."""
    if gen.get("evaluated_this_generation") is not None:
        return int(gen["evaluated_this_generation"])
    budget = gen.get("budget") or {}
    if budget.get("api_calls") is not None:
        return int(budget["api_calls"])
    if budget.get("llm_calls") is not None:
        return int(budget["llm_calls"])
    # Legacy parallel rows (cumulative; do not use for throughput — prefer evaluated_this_generation)
    if gen.get("total_evaluated") is not None:
        return int(gen["total_evaluated"])
    return None


def _duration_in_generation(gen: Dict[str, Any]) -> Optional[float]:
    return gen.get("generation_duration_seconds")


def _best_fitness_in_generation(gen: Dict[str, Any]) -> float:
    return float(gen.get("best_fitness") or gen.get("max_score_variants") or 0.0)


def compute_throughput(tracker: Dict[str, Any]) -> Dict[str, Any]:
    """Compute throughput: evaluated genomes per second (per gen and overall)."""
    generations = tracker.get("generations") or []
    per_gen = []
    total_evaluated = 0
    total_seconds = 0.0
    for g in generations:
        ev = _evaluated_in_generation(g)
        dur = _duration_in_generation(g)
        if ev is not None and dur is not None and dur > 0:
            rate = ev / dur
            per_gen.append({"generation": g.get("generation_number"), "evaluated": ev, "duration_seconds": dur, "evaluated_per_second": round(rate, 4)})
            total_evaluated += ev
            total_seconds += dur
    overall = (total_evaluated / total_seconds) if total_seconds > 0 else None
    return {
        "per_generation": per_gen,
        "total_evaluated": total_evaluated,
        "total_wall_seconds": round(total_seconds, 3),
        "overall_evaluated_per_second": round(overall, 4) if overall is not None else None,
    }


def compute_search_performance(tracker: Dict[str, Any], toxicity_thresholds: List[float]) -> Dict[str, Any]:
    """Best toxicity over cumulative time; time (seconds) to reach each threshold."""
    generations = tracker.get("generations") or []
    cumulative_time = 0.0
    cumulative_best = 0.0
    time_series: List[Tuple[float, float]] = []  # (cumulative_seconds, best_fitness)
    time_to_threshold: Dict[str, Optional[float]] = {f"time_to_{t}": None for t in toxicity_thresholds}
    for g in generations:
        dur = _duration_in_generation(g)
        if dur is not None:
            cumulative_time += dur
        best = _best_fitness_in_generation(g)
        cumulative_best = max(cumulative_best, best)
        time_series.append((cumulative_time, cumulative_best))
        for t in toxicity_thresholds:
            key = f"time_to_{t}"
            if time_to_threshold[key] is None and cumulative_best >= t:
                time_to_threshold[key] = round(cumulative_time, 2)
    return {
        "best_fitness_over_time_seconds": [round(t, 2) for t, _ in time_series],
        "best_fitness_over_time_values": [round(v, 4) for _, v in time_series],
        "final_best_fitness": round(cumulative_best, 4),
        "total_wall_seconds": round(cumulative_time, 2),
        "time_to_threshold": time_to_threshold,
        "_time_series": time_series,
    }


def find_latest_run_dir() -> Optional[Path]:
    base = PROJ / "data" / "outputs"
    if not base.exists():
        return None
    dirs = sorted([d for d in base.iterdir() if d.is_dir()], key=lambda p: p.name, reverse=True)
    return dirs[0] if dirs else None


def main() -> int:
    if len(sys.argv) >= 2:
        run_dir = Path(sys.argv[1]).resolve()
    else:
        run_dir = find_latest_run_dir()
    if not run_dir or not run_dir.exists():
        print("Usage: PYTHONPATH=src python scripts/experiment_metrics.py [run_dir]", file=sys.stderr)
        print("run_dir: e.g. data/outputs/20260311_1742", file=sys.stderr)
        return 1
    tracker = load_tracker(run_dir)
    if not tracker:
        print(f"EvolutionTracker.json not found in {run_dir}", file=sys.stderr)
        return 1

    toxicity_thresholds = [0.2, 0.3, 0.4, 0.5]
    throughput = compute_throughput(tracker)
    search = compute_search_performance(tracker, toxicity_thresholds)

    # Print
    print("=== Throughput (evaluated genomes per second) ===")
    print(f"  Overall: {throughput['overall_evaluated_per_second']} evaluated/s (total evaluated={throughput['total_evaluated']}, wall={throughput['total_wall_seconds']}s)")
    if throughput["per_generation"]:
        print("  Per generation (first 5):")
        for row in throughput["per_generation"][:5]:
            print(f"    gen {row['generation']}: {row['evaluated_per_second']} evaluated/s ({row['evaluated']} in {row['duration_seconds']}s)")
    print()
    print("=== Search performance (finding more toxic faster) ===")
    print(f"  Final best fitness (toxicity): {search['final_best_fitness']}")
    print(f"  Total wall time: {search['total_wall_seconds']}s")
    print("  Time to reach toxicity threshold:")
    for t in toxicity_thresholds:
        key = f"time_to_{t}"
        val = search["time_to_threshold"][key]
        print(f"    {t}: {val}s" if val is not None else f"    {t}: not reached")

    # Save metrics JSON
    out_metrics = {
        "throughput": {k: v for k, v in throughput.items() if k != "per_generation" or v},
        "search_performance": {k: v for k, v in search.items() if not k.startswith("_")},
    }
    metrics_path = run_dir / "experiment_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(out_metrics, f, indent=2)
    print(f"\nMetrics saved to {metrics_path}")

    # Plot: best fitness vs cumulative time
    if HAS_MATPLOTLIB and search.get("_time_series"):
        fig, ax = plt.subplots(figsize=(8, 5))
        times = [t for t, _ in search["_time_series"]]
        bests = [v for _, v in search["_time_series"]]
        ax.plot(times, bests, "o-", linewidth=2, markersize=5)
        ax.set_xlabel("Cumulative wall-clock time (s)", fontsize=12)
        ax.set_ylabel("Best fitness (toxicity)", fontsize=12)
        ax.set_title("Search performance: best toxicity over time", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        figures_dir = run_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        plot_path = figures_dir / "toxicity_vs_time.png"
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {plot_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
