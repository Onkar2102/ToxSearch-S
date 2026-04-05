#!/usr/bin/env python3
"""
Compare sequential vs parallel runs using **population-integrated genomes** and a **single** figure.

**Counts:** Prefer ``variants_integrated`` per generation (MPI). If missing (older sequential logs),
fall back to ``budget.llm_calls`` (evaluations that feed the population update).

**Figure** ``performance_overview.png`` (one file):
  - Stacked bar: wall-clock time split into LLM (response + variant-creation), moderation/API eval,
    speciation, and overhead (sync / idle / other).
  - Bar chart: population-integrated genomes per second (wall time and active-work time).

Also writes ``comparison_summary.json`` (time breakdown, integrated totals, throughput, optional toxicity times).

Usage (from repository root):

  PYTHONPATH=src python experiments/compare_sequential_vs_parallel.py

Requires: matplotlib, numpy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJ = Path(__file__).resolve().parents[1]
if str(PROJ / "src") not in sys.path:
    sys.path.insert(0, str(PROJ / "src"))

try:
    import matplotlib

    matplotlib.use("Agg")
    from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

    configure_matplotlib_embedded_fonts()
    import matplotlib.pyplot as plt
    import numpy as np

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    np = None  # type: ignore


def load_tracker(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "EvolutionTracker.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def population_integrated_count(gen: Dict[str, Any]) -> int:
    """Genomes integrated into the population this generation (tracker field when available)."""
    vi = gen.get("variants_integrated")
    if vi is not None:
        return int(vi)
    if gen.get("total_evaluated") is not None:
        return int(gen["total_evaluated"])
    b = gen.get("budget") or {}
    if b.get("llm_calls") is not None:
        return int(b["llm_calls"])
    return 0


def budget_generation_duration_seconds(gen: Dict[str, Any]) -> Optional[float]:
    """Comparable 'work' time: LLM + moderation + speciation step (no idle / sync)."""
    b = gen.get("budget") or {}
    sp = gen.get("speciation") or {}
    parts = float(b.get("total_response_time") or 0) + float(b.get("total_evaluation_time") or 0)
    parts += float(sp.get("speciation_duration_seconds") or 0)
    return parts if parts > 0 else None


def wall_generation_duration_seconds(gen: Dict[str, Any]) -> Optional[float]:
    d = gen.get("generation_duration_seconds")
    if d is not None and float(d) > 0:
        return float(d)
    return None


def effective_generation_duration_seconds(gen: Dict[str, Any]) -> Optional[float]:
    """Prefer wall clock when present; else fall back to budget time (sequential logs)."""
    w = wall_generation_duration_seconds(gen)
    if w is not None:
        return w
    return budget_generation_duration_seconds(gen)


def best_fitness(gen: Dict[str, Any]) -> float:
    return float(gen.get("best_fitness") or gen.get("max_score_variants") or 0.0)


def extract_series(
    tracker: Dict[str, Any], label: str
) -> Tuple[Dict[str, List[Any]], Dict[str, Any]]:
    gens = sorted(tracker.get("generations") or [], key=lambda x: int(x.get("generation_number", 0)))
    meta = tracker.get("run_metadata") or {}
    out: Dict[str, List[Any]] = {
        "generation": [],
        "duration_s": [],
        "duration_budget_s": [],
        "integrated": [],
        "throughput": [],
        "throughput_budget": [],
        "best_fitness": [],
        "species_count": [],
        "cumulative_time": [],
        "cumulative_integrated": [],
        "cumulative_best": [],
        "cumulative_budget_time": [],
    }
    cum_t = 0.0
    cum_bt = 0.0
    cum_int = 0
    cum_best = 0.0
    for g in gens:
        gn = g.get("generation_number")
        if gn is None:
            continue
        out["generation"].append(int(gn))
        dur = effective_generation_duration_seconds(g)
        bdur = budget_generation_duration_seconds(g)
        inte = population_integrated_count(g)
        out["duration_s"].append(dur)
        out["duration_budget_s"].append(bdur)
        out["integrated"].append(inte)
        if dur is not None and dur > 0:
            out["throughput"].append(inte / dur)
        else:
            out["throughput"].append(None)
        if bdur is not None and bdur > 0:
            out["throughput_budget"].append(inte / bdur)
        else:
            out["throughput_budget"].append(None)
        bf = best_fitness(g)
        out["best_fitness"].append(bf)
        sp = g.get("speciation") or {}
        out["species_count"].append(sp.get("species_count"))
        if dur is not None:
            cum_t += dur
        if bdur is not None:
            cum_bt += bdur
        cum_int += inte
        cum_best = max(cum_best, bf)
        out["cumulative_time"].append(cum_t)
        out["cumulative_integrated"].append(cum_int)
        out["cumulative_best"].append(cum_best)
        out["cumulative_budget_time"].append(cum_bt)

    summary = {
        "label": label,
        "run_metadata": meta,
        "total_generations": len(out["generation"]),
        "total_population_integrated": cum_int,
        "integrated_count_source": "variants_integrated when present else llm_calls",
        "total_duration_seconds": cum_t,
        "total_budget_duration_seconds": cum_bt,
        "overall_integrated_per_wall_second": (cum_int / cum_t) if cum_t > 0 else None,
        "overall_integrated_per_budget_second": (cum_int / cum_bt) if cum_bt > 0 else None,
        "overall_throughput_eval_per_s": (cum_int / cum_t) if cum_t > 0 else None,
        "overall_throughput_budget_eval_per_s": (cum_int / cum_bt) if cum_bt > 0 else None,
        "final_best_fitness": cum_best,
        "duration_is_wall_clock": bool(
            gens and gens[0].get("generation_duration_seconds") is not None
        ),
    }
    return out, summary


def time_to_toxicity_thresholds(
    cumulative_times: List[float], cumulative_bests: List[float], thresholds: List[float]
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for t in thresholds:
        key = f"time_to_best_ge_{t}"
        out[key] = None
        for i in range(len(cumulative_times)):
            if cumulative_bests[i] >= t:
                out[key] = round(float(cumulative_times[i]), 2)
                break
    return out


def time_breakdown_totals(tracker: Dict[str, Any]) -> Dict[str, float]:
    """
    Sum time (seconds) across generations. LLM = response + variant-creation.
    Overhead = wall - (llm + moderation + speciation) when wall is logged.
    """
    llm = mod = spec = wall = 0.0
    for g in tracker.get("generations") or []:
        b = g.get("budget") or {}
        sp = g.get("speciation") or {}
        llm += float(b.get("total_response_time") or 0) + float(b.get("total_variant_creation_time") or 0)
        mod += float(b.get("total_evaluation_time") or 0) + float(b.get("total_evaluation_api_wait_seconds") or 0)
        spec += float(sp.get("speciation_duration_seconds") or 0)
        w = g.get("generation_duration_seconds")
        if w is not None:
            wall += float(w)
    work = llm + mod + spec
    overhead = max(0.0, wall - work) if wall > 0 else 0.0
    return {
        "seconds_llm_response_and_variant": llm,
        "seconds_moderation_and_eval": mod,
        "seconds_speciation": spec,
        "seconds_wall_total": wall,
        "seconds_overhead_sync_idle": overhead,
        "seconds_active_work_sum": work,
    }


def total_population_integrated(tracker: Dict[str, Any]) -> int:
    return sum(population_integrated_count(g) for g in tracker.get("generations") or [])


def optional_metrics_json(run_dir: Path) -> Optional[Dict[str, Any]]:
    p = run_dir / "analysis" / "metrics.json"
    if not p.exists():
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_performance_overview(
    seq_tracker: Dict[str, Any],
    par_tracker: Dict[str, Any],
    seq_sum: Dict[str, Any],
    par_sum: Dict[str, Any],
    seq_label: str,
    par_label: str,
    out_dir: Path,
) -> List[str]:
    """Single figure: time breakdown (stacked) + population-integrated genomes/s."""
    if not HAS_MPL or np is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)

    tb_s = time_breakdown_totals(seq_tracker)
    tb_p = time_breakdown_totals(par_tracker)

    def _short(s: str, n: int = 32) -> str:
        return s if len(s) <= n else s[: n - 1] + "…"

    y_labs = [_short(seq_label), _short(par_label)]
    y = np.arange(2)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.2),
        gridspec_kw={"width_ratios": [1.65, 1.0]},
    )

    segments = [
        ("seconds_llm_response_and_variant", "LLM (response + variant)", "#1f77b4"),
        ("seconds_moderation_and_eval", "Moderation / API eval", "#ff7f0e"),
        ("seconds_speciation", "Speciation", "#2ca02c"),
        ("seconds_overhead_sync_idle", "Overhead (sync, idle, …)", "#7f7f7f"),
    ]
    left = np.zeros(2)
    for key, leg_label, color in segments:
        w = np.array([tb_s[key], tb_p[key]])
        ax1.barh(y, w, left=left, label=leg_label, color=color, height=0.55)
        left = left + w
    ax1.set_yticks(y)
    ax1.set_yticklabels(y_labs)
    ax1.set_xlabel("Seconds (summed over all generations)")
    ax1.set_title("Where time went (wall overhead only when per-gen wall is logged)")
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=8)
    ax1.grid(True, axis="x", alpha=0.25)

    rw_s = seq_sum.get("overall_integrated_per_wall_second") or 0.0
    rw_p = par_sum.get("overall_integrated_per_wall_second") or 0.0
    rb_s = seq_sum.get("overall_integrated_per_budget_second") or 0.0
    rb_p = par_sum.get("overall_integrated_per_budget_second") or 0.0

    xg = np.arange(2)
    bw = 0.35
    ax2.bar(xg - bw / 2, [rw_s, rw_p], bw, label="÷ wall time", color="#4c72b0")
    ax2.bar(xg + bw / 2, [rb_s, rb_p], bw, label="÷ active work time", color="#dd8452")
    ax2.set_xticks(xg)
    ax2.set_xticklabels(["Sequential", "Parallel"])
    ax2.set_ylabel("Population-integrated genomes / s")
    ax2.set_title("Throughput (integrated genomes per second)")
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.text(
        0.02,
        0.98,
        "Integrated = variants_integrated if logged, else llm_calls.",
        transform=ax2.transAxes,
        fontsize=7,
        verticalalignment="top",
        color="0.35",
    )

    fig.suptitle("Sequential vs parallel performance overview", fontsize=13, fontweight="bold", y=1.02)
    out_path = out_dir / "performance_overview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    return [str(out_path)]


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare sequential vs parallel run performance")
    ap.add_argument(
        "--sequential",
        type=Path,
        default=PROJ / "data" / "outputs" / "20260211_2122",
        help="Run directory (sequential single-GPU)",
    )
    ap.add_argument(
        "--parallel",
        type=Path,
        default=PROJ / "data" / "outputs" / "20260323_0306-4",
        help="Run directory (parallel multi-GPU)",
    )
    ap.add_argument(
        "--tag",
        default="20260211_2122__20260323_0306-4",
        help="Output subfolder under experiments/outputs/sequential_vs_parallel/",
    )
    args = ap.parse_args()

    seq_dir = args.sequential.resolve()
    par_dir = args.parallel.resolve()
    seq_t = load_tracker(seq_dir)
    par_t = load_tracker(par_dir)
    if not seq_t:
        print(f"Missing EvolutionTracker.json in {seq_dir}", file=sys.stderr)
        return 1
    if not par_t:
        print(f"Missing EvolutionTracker.json in {par_dir}", file=sys.stderr)
        return 1

    seq_label = f"sequential ({seq_dir.name})"
    par_label = f"parallel ({par_dir.name})"
    seq_series, seq_sum = extract_series(seq_t, seq_label)
    par_series, par_sum = extract_series(par_t, par_label)

    t_seq = seq_sum.get("overall_integrated_per_wall_second")
    t_par = par_sum.get("overall_integrated_per_wall_second")
    speedup_wall = (t_par / t_seq) if (t_seq and t_par) else None
    b_seq = seq_sum.get("overall_integrated_per_budget_second")
    b_par = par_sum.get("overall_integrated_per_budget_second")
    speedup_budget = (b_par / b_seq) if (b_seq and b_par) else None

    tox_thresholds = [0.3, 0.5, 0.7, 0.9]
    seq_t_wall = time_to_toxicity_thresholds(
        seq_series["cumulative_time"], seq_series["cumulative_best"], tox_thresholds
    )
    par_t_wall = time_to_toxicity_thresholds(
        par_series["cumulative_time"], par_series["cumulative_best"], tox_thresholds
    )
    seq_t_budget = time_to_toxicity_thresholds(
        seq_series["cumulative_budget_time"], seq_series["cumulative_best"], tox_thresholds
    )
    par_t_budget = time_to_toxicity_thresholds(
        par_series["cumulative_budget_time"], par_series["cumulative_best"], tox_thresholds
    )

    comparison = {
        "sequential_dir": str(seq_dir),
        "parallel_dir": str(par_dir),
        "sequential_summary": seq_sum,
        "parallel_summary": par_sum,
        "time_to_best_toxicity_cumulative_wall_s": {
            "sequential": seq_t_wall,
            "parallel": par_t_wall,
            "thresholds": tox_thresholds,
        },
        "time_to_best_toxicity_cumulative_budget_s": {
            "sequential": seq_t_budget,
            "parallel": par_t_budget,
            "thresholds": tox_thresholds,
        },
        "parallel_run_metadata": par_t.get("run_metadata"),
        "sequential_run_metadata": seq_t.get("run_metadata"),
        "time_breakdown_sequential": time_breakdown_totals(seq_t),
        "time_breakdown_parallel": time_breakdown_totals(par_t),
        "notes": {
            "population_integrated": "Sum of variants_integrated per generation when present; else llm_calls (sequential legacy).",
            "time_breakdown": "LLM = total_response_time + total_variant_creation_time; moderation = eval + API wait; overhead = wall − (LLM+mod+spec) when wall is logged.",
            "throughput": "integrated / wall_seconds and integrated / active_work_seconds (sum of LLM+mod+spec).",
            "fairness": "Runs differ in generations, batch, and caps; interpret side-by-side cautiously.",
        },
        "throughput_speedup_wall_parallel_over_sequential": speedup_wall,
        "throughput_speedup_budget_parallel_over_sequential": speedup_budget,
        "optional_metrics_sequential": optional_metrics_json(seq_dir),
        "optional_metrics_parallel": optional_metrics_json(par_dir),
    }

    out_root = PROJ / "experiments" / "outputs" / "sequential_vs_parallel" / args.tag
    out_root.mkdir(parents=True, exist_ok=True)
    out_json = out_root / "comparison_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"Wrote {out_json}")
    print(
        f"  Sequential: G={seq_sum['total_generations']} integrated={seq_sum['total_population_integrated']} "
        f"duration_s={seq_sum['total_duration_seconds']:.1f} int/s_wall={seq_sum.get('overall_integrated_per_wall_second')}"
    )
    print(
        f"  Parallel:   G={par_sum['total_generations']} integrated={par_sum['total_population_integrated']} "
        f"duration_s={par_sum['total_duration_seconds']:.1f} int/s_wall={par_sum.get('overall_integrated_per_wall_second')}"
    )
    if speedup_wall is not None:
        print(f"  Throughput speedup wall (parallel / sequential): {speedup_wall:.3f}x")
    if speedup_budget is not None:
        print(f"  Throughput speedup budget-comparable (parallel / sequential): {speedup_budget:.3f}x")

    if HAS_MPL:
        figs = plot_performance_overview(
            seq_t, par_t, seq_sum, par_sum, seq_label, par_label, out_root
        )
        for p in figs:
            print(f"  Figure: {p}")
    else:
        print("matplotlib not available; skipped figures.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
