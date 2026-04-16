#!/usr/bin/env python3
"""
C2: ToxSearch-S sequential vs parallel (2 workers) — PPSN2026 outputs.

Writes:
  experiments/comparison_results/c2_ppsn2026_seq_vs_2w/
    run_manifest.csv, metrics_per_run.csv, execution_throughput_table.csv,
    stats_summary.json, stats_table.csv
    figures/wall_time.pdf, throughput_per_execution.pdf, time_breakdown_mean_fractions.pdf

execution_throughput_table.csv: per run, total_genomes (sum of per-generation integrated
counts in EvolutionTracker) / run_metadata.run_duration_seconds.

Run (from repo root):
  python experiments/comparison_results/c2_ppsn2026_seq_vs_2w/c2_seq_vs_2w_report.py
"""

from __future__ import annotations

import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import mannwhitneyu

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402

configure_matplotlib_embedded_fonts()

DATA = PROJ / "data" / "outputs" / "ppsn2026"
SEQ_ROOT = DATA / "toxsearch_s"
PAR_ROOT = DATA / "toxsearch_s_2w"

OUT = PROJ / "experiments" / "comparison_results" / "c2_ppsn2026_seq_vs_2w"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

RNG_SEED = 12345
BOOTSTRAP_R = 10000


# --- tracker helpers (aligned with compare_sequential_vs_parallel) ---

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def population_integrated_count(gen: Dict[str, Any]) -> int:
    vi = gen.get("variants_integrated")
    if vi is not None:
        return int(vi)
    if gen.get("total_evaluated") is not None:
        return int(gen["total_evaluated"])
    b = gen.get("budget") or {}
    if b.get("llm_calls") is not None:
        return int(b["llm_calls"])
    return 0


def total_integrated(tracker: Dict[str, Any]) -> int:
    return sum(population_integrated_count(g) for g in tracker.get("generations") or [])


def qmax_tracker(tracker: Dict[str, Any]) -> float:
    best = 0.0
    for g in tracker.get("generations") or []:
        m = g.get("max_score_variants")
        if isinstance(m, (int, float)) and not math.isnan(m):
            best = max(best, float(m))
    return float(best)


def time_breakdown_totals(tracker: Dict[str, Any]) -> Dict[str, float]:
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
        "llm": llm,
        "moderation_eval": mod,
        "speciation": spec,
        "wall": wall,
        "overhead": overhead,
    }


def stepwise_at(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    cur = float(ys[0]) if ys else 0.0
    for x, y in zip(xs, ys):
        if float(x) <= xq:
            cur = float(y)
        else:
            break
    return cur


def cumulative_wall_vs_evaluations(tracker: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative wall-clock time (sum of generation_duration_seconds) vs cumulative evaluated genomes.
    Uses evaluated_this_generation per generation, falling back to population_integrated_count.
    Starts at (0, 0).
    """
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return np.asarray([0.0]), np.asarray([0.0])
    xe = [0.0]
    yw = [0.0]
    for g in gens:
        ev = g.get("evaluated_this_generation")
        if ev is None:
            ev = population_integrated_count(g)
        ev = float(ev) if isinstance(ev, (int, float)) else 0.0
        wall = g.get("generation_duration_seconds")
        wall = float(wall) if wall is not None and float(wall) >= 0 else 0.0
        xe.append(xe[-1] + max(0.0, ev))
        yw.append(yw[-1] + wall)
    return np.asarray(xe, dtype=float), np.asarray(yw, dtype=float)


def discover_runs(
    root: Path,
    expected_mode: str,
    expected_workers: int,
) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        return out
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        tpath = p / "EvolutionTracker.json"
        if not tpath.exists():
            continue
        t = load_json(tpath)
        st = (t.get("status") or "").lower()
        if st and st != "complete":
            continue
        rm = t.get("run_metadata") or {}
        mode = str(rm.get("run_mode") or "")
        nw = int(rm.get("num_workers") or 0)
        if mode != expected_mode or nw != expected_workers:
            continue
        out.append((p.name, p))
    return out


def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    x = [float(v) for v in x]
    y = [float(v) for v in y]
    if not x or not y:
        return float("nan")
    dom = 0
    for a in x:
        for b in y:
            if a > b:
                dom += 1
            elif a < b:
                dom -= 1
    return dom / (len(x) * len(y))


def bootstrap_ci_median_diff(
    seq: Sequence[float],
    par: Sequence[float],
    rng: random.Random,
    n: int = BOOTSTRAP_R,
) -> Tuple[float, float]:
    seq = list(seq)
    par = list(par)
    if not seq or not par:
        return float("nan"), float("nan")
    diffs: List[float] = []
    for _ in range(n):
        sx = [seq[rng.randrange(len(seq))] for _ in range(len(seq))]
        py = [par[rng.randrange(len(par))] for _ in range(len(par))]
        diffs.append(float(np.median(py) - np.median(sx)))
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return float(lo), float(hi)


def bootstrap_ci_cliffs(
    seq: Sequence[float],
    par: Sequence[float],
    rng: random.Random,
    n: int = BOOTSTRAP_R,
) -> Tuple[float, float]:
    seq = list(seq)
    par = list(par)
    if not seq or not par:
        return float("nan"), float("nan")
    vals: List[float] = []
    for _ in range(n):
        sx = [seq[rng.randrange(len(seq))] for _ in range(len(seq))]
        py = [par[rng.randrange(len(par))] for _ in range(len(par))]
        vals.append(cliffs_delta(sx, py))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    """
    Holm–Bonferroni adjusted p-values.
    For ordered p_(1) <= ... <= p_(m): adjusted for rank j is
    max_{k<=j} min(1, (m-k+1) * p_(k)) (1-based rank in sorted list).
    """
    m = len(pvals)
    order = sorted(range(m), key=lambda i: float(pvals[i]))
    ps = [float(pvals[i]) for i in order]
    adj_sorted: List[float] = []
    for j in range(m):
        adj_sorted.append(max(min(1.0, (m - k) * ps[k]) for k in range(j + 1)))
    out = [0.0] * m
    for j in range(m):
        out[order[j]] = adj_sorted[j]
    return out


@dataclass
class RunRow:
    run_id: str
    mode: str
    run_dir: Path
    wall_s: float
    total_integrated: int
    throughput: float
    qmax: float
    max_total_genomes: Optional[int]


def collect_metrics(seq_runs: List[Tuple[str, Path]], par_runs: List[Tuple[str, Path]]) -> List[RunRow]:
    rows: List[RunRow] = []
    for label, pairs in [("sequential", seq_runs), ("parallel_2w", par_runs)]:
        for run_id, run_dir in pairs:
            t = load_json(run_dir / "EvolutionTracker.json")
            rm = t.get("run_metadata") or {}
            wall = float(rm.get("run_duration_seconds") or 0)
            mtg = rm.get("max_total_genomes")
            mtg_i = int(mtg) if isinstance(mtg, (int, float)) else None
            n_int = total_integrated(t)
            thr = (n_int / wall) if wall > 0 else 0.0
            qm = qmax_tracker(t)
            rows.append(
                RunRow(
                    run_id=run_id,
                    mode=label,
                    run_dir=run_dir,
                    wall_s=wall,
                    total_integrated=n_int,
                    throughput=thr,
                    qmax=qm,
                    max_total_genomes=mtg_i,
                )
            )
    return rows


def plot_throughput_per_execution(rows: Sequence["RunRow"], out_path: Path) -> None:
    """
    Per-execution dot plot (no pairing implied).
    Shows genomes/s for each execution index, split by mode.
    """
    seq = sorted([r for r in rows if r.mode == "sequential"], key=lambda r: r.run_id)
    par = sorted([r for r in rows if r.mode == "parallel_2w"], key=lambda r: r.run_id)

    n = max(len(seq), len(par))
    y = np.arange(1, n + 1, dtype=float)
    ylabels = [f"#{i}" for i in range(1, n + 1)]

    # Narrow width so the small throughput range is not stretched across a wide figure.
    plt.figure(figsize=(2.3, 4.2), constrained_layout=True)
    ax = plt.gca()

    # Small vertical dodge so the two series don't overlap.
    dodge = 0.12
    c_seq = "#1f77b4"
    c_par = "#ff7f0e"

    if seq:
        x_seq = np.asarray([r.throughput for r in seq], dtype=float)
        ax.scatter(
            x_seq,
            y[: len(x_seq)] - dodge,
            s=48,
            c=c_seq,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    if par:
        x_par = np.asarray([r.throughput for r in par], dtype=float)
        ax.scatter(
            x_par,
            y[: len(x_par)] + dodge,
            s=48,
            c=c_par,
            alpha=0.95,
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(ylabels)
    ax.set_xlabel("Throughput")
    ax.set_ylabel("Execution IDs")
    ax.grid(True, axis="x", alpha=0.35)

    # Short axis: ticks 0.00, 0.02, 0.04 only; upper limit ~max throughput (~0.042) + margin
    ax.set_xlim(0.0, 0.045)
    ax.set_xticks([0.0, 0.02, 0.04])
    ax.set_xticklabels(["0.00", "0.02", "0.04"])
    ax.set_ylim(0.5, n + 0.5)

    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_time_breakdown_aggregate(
    seq_dirs: List[Path],
    par_dirs: List[Path],
    out_path: Path,
) -> None:
    def mean_fractions(dirs: List[Path]) -> Tuple[float, float, float, float]:
        llm_l, mod_l, sp_l, oh_l = [], [], [], []
        for d in dirs:
            t = load_json(d / "EvolutionTracker.json")
            tb = time_breakdown_totals(t)
            tot = tb["llm"] + tb["moderation_eval"] + tb["speciation"] + tb["overhead"]
            if tot <= 0:
                continue
            llm_l.append(tb["llm"] / tot)
            mod_l.append(tb["moderation_eval"] / tot)
            sp_l.append(tb["speciation"] / tot)
            oh_l.append(tb["overhead"] / tot)
        if not llm_l:
            return 0.0, 0.0, 0.0, 0.0
        return float(np.mean(llm_l)), float(np.mean(mod_l)), float(np.mean(sp_l)), float(np.mean(oh_l))

    fs = mean_fractions(seq_dirs)
    fp = mean_fractions(par_dirs)
    labels = ["Sequential", "Parallel\n(2 workers)"]
    categories = ["LLM", "Moderation /\neval", "Speciation", "Overhead"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#7f7f7f"]
    y = np.arange(2)
    plt.figure(figsize=(8.0, 4.2))
    left = np.zeros(2)
    for i, cat in enumerate(categories):
        w = np.array([fs[i], fp[i]])
        plt.barh(y, w, left=left, label=cat, color=colors[i], height=0.55)
        left = left + w
    plt.yticks(y, labels)
    plt.xlabel("Fraction of summed time components")
    plt.xlim(0, 1)
    plt.grid(False)
    plt.legend(frameon=False, bbox_to_anchor=(1.02, 0.5), loc="center left")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_wall_vs_evaluated_genomes_median_iqr(
    seq_dirs: List[Path],
    par_dirs: List[Path],
    out_path: Path,
    milestones: Sequence[int] = tuple(range(0, 1100, 100)),
) -> None:
    """
    Cumulative wall time (s) vs cumulative evaluated genomes: median + IQR per mode.
    Y-axis log-scaled (values at 0 s omitted on log scale).
    """
    def series_for_run(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yw = cumulative_wall_vs_evaluations(t)
        pairs = sorted(zip(xe.tolist(), yw.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        return [stepwise_at(xs2, ys2, float(m)) for m in milestones]

    s_rows = [series_for_run(d) for d in seq_dirs]
    p_rows = [series_for_run(d) for d in par_dirs]
    if not s_rows or not p_rows:
        return

    S = np.asarray(s_rows, dtype=float)
    P = np.asarray(p_rows, dtype=float)
    med_s = np.median(S, axis=0)
    med_p = np.median(P, axis=0)
    q1_s, q3_s = np.quantile(S, 0.25, axis=0), np.quantile(S, 0.75, axis=0)
    q1_p, q3_p = np.quantile(P, 0.25, axis=0), np.quantile(P, 0.75, axis=0)

    x = np.asarray(list(milestones), dtype=float)

    def mask_pos(y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        return np.where(y > 0, y, np.nan)

    plt.figure(figsize=(7.0, 4.2))
    ax = plt.gca()
    plt.plot(x, mask_pos(med_s), linewidth=2.5, label="Sequential", color="#1f77b4")
    plt.fill_between(x, mask_pos(q1_s), mask_pos(q3_s), alpha=0.2, color="#1f77b4")
    plt.plot(x, mask_pos(med_p), linewidth=2.5, label="Parallel (2w)", color="#ff7f0e")
    plt.fill_between(x, mask_pos(q1_p), mask_pos(q3_p), alpha=0.2, color="#ff7f0e")

    ax.set_yscale("log")
    plt.xlabel("Evaluated genomes")
    plt.ylabel("Wall-clock time (s)")
    plt.xlim(0, 1000)
    plt.xticks(list(range(0, 1100, 100)))
    pos = np.concatenate([med_s[med_s > 0], med_p[med_p > 0]])
    if pos.size > 0:
        plt.ylim(bottom=float(np.min(pos)) * 0.7)
    plt.grid(True, which="major", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_wall_vs_evaluated_genomes_from0_median_iqr(
    seq_dirs: List[Path],
    par_dirs: List[Path],
    out_path: Path,
    milestones: Sequence[int] = tuple(range(0, 1100, 100)),
) -> None:
    """
    Same data as wall_time.pdf, but *includes genome 0 visibly* by using a
    zero-friendly y scale (symlog). This lets the curve start at (0, 0).
    """
    def series_for_run(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yw = cumulative_wall_vs_evaluations(t)
        pairs = sorted(zip(xe.tolist(), yw.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        return [stepwise_at(xs2, ys2, float(m)) for m in milestones]

    s_rows = [series_for_run(d) for d in seq_dirs]
    p_rows = [series_for_run(d) for d in par_dirs]
    if not s_rows or not p_rows:
        return

    S = np.asarray(s_rows, dtype=float)
    P = np.asarray(p_rows, dtype=float)
    med_s = np.median(S, axis=0)
    med_p = np.median(P, axis=0)
    q1_s, q3_s = np.quantile(S, 0.25, axis=0), np.quantile(S, 0.75, axis=0)
    q1_p, q3_p = np.quantile(P, 0.25, axis=0), np.quantile(P, 0.75, axis=0)

    x = np.asarray(list(milestones), dtype=float)

    plt.figure(figsize=(7.0, 4.2))
    ax = plt.gca()
    plt.plot(x, med_s, linewidth=2.5, label="Sequential", color="#1f77b4")
    plt.fill_between(x, q1_s, q3_s, alpha=0.2, color="#1f77b4")
    plt.plot(x, med_p, linewidth=2.5, label="Parallel (2w)", color="#ff7f0e")
    plt.fill_between(x, q1_p, q3_p, alpha=0.2, color="#ff7f0e")

    # symlog: supports 0 while still compressing the large tail like log.
    ax.set_yscale("symlog", linthresh=1.0, linscale=1.0, base=10)
    plt.xlabel("Evaluated genomes")
    plt.ylabel("Wall-clock time (s)")
    plt.xlim(0, 1000)
    plt.xticks(list(range(0, 1100, 100)))
    plt.ylim(bottom=0.0)
    plt.grid(True, which="major", alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def main() -> int:
    seq_runs = discover_runs(SEQ_ROOT, "sequential", 1)
    par_runs = discover_runs(PAR_ROOT, "parallel", 2)

    rows = collect_metrics(seq_runs, par_runs)

    with (OUT / "run_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run_id", "run_path", "wall_s", "num_workers_expected"])
        for r in rows:
            w.writerow([r.mode, r.run_id, str(r.run_dir), r.wall_s, 1 if r.mode == "sequential" else 2])

    with (OUT / "metrics_per_run.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "mode",
            "run_id",
            "wall_s",
            "total_integrated",
            "throughput_integrated_per_s",
            "qmax_tracker",
            "max_total_genomes",
        ]
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()
        for r in rows:
            wr.writerow(
                {
                    "mode": r.mode,
                    "run_id": r.run_id,
                    "wall_s": r.wall_s,
                    "total_integrated": r.total_integrated,
                    "throughput_integrated_per_s": r.throughput,
                    "qmax_tracker": r.qmax,
                    "max_total_genomes": r.max_total_genomes,
                }
            )

    with (OUT / "execution_throughput_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "mode",
                "run_id",
                "total_genomes",
                "execution_duration_s",
                "genomes_per_s",
            ]
        )
        for r in sorted(
            rows,
            key=lambda x: (0 if x.mode == "sequential" else 1, x.run_id),
        ):
            g_per_s = (r.total_integrated / r.wall_s) if r.wall_s > 0 else 0.0
            w.writerow([r.mode, r.run_id, r.total_integrated, r.wall_s, g_per_s])

    seq_wall = [r.wall_s for r in rows if r.mode == "sequential"]
    par_wall = [r.wall_s for r in rows if r.mode == "parallel_2w"]
    seq_thr = [r.throughput for r in rows if r.mode == "sequential"]
    par_thr = [r.throughput for r in rows if r.mode == "parallel_2w"]

    rng = random.Random(RNG_SEED)
    metrics_order = ["wall_s", "throughput"]
    pvals: Dict[str, float] = {}
    cliffs: Dict[str, float] = {}
    boot_med: Dict[str, Tuple[float, float]] = {}
    boot_cl: Dict[str, Tuple[float, float]] = {}

    pairs = [
        ("wall_s", seq_wall, par_wall),
        ("throughput", seq_thr, par_thr),
    ]
    for name, a, b in pairs:
        if len(a) >= 1 and len(b) >= 1:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            pvals[name] = float(p)
            cliffs[name] = cliffs_delta(a, b)
            boot_med[name] = bootstrap_ci_median_diff(a, b, rng)
            boot_cl[name] = bootstrap_ci_cliffs(a, b, rng)
        else:
            pvals[name] = float("nan")
            cliffs[name] = float("nan")
            boot_med[name] = (float("nan"), float("nan"))
            boot_cl[name] = (float("nan"), float("nan"))

    valid_m = [m for m in metrics_order if not math.isnan(pvals[m])]
    if valid_m:
        adj = holm_adjust([pvals[m] for m in valid_m])
        holm_dict = {m: float("nan") for m in metrics_order}
        for m, a in zip(valid_m, adj):
            holm_dict[m] = float(a)
    else:
        holm_dict = {m: float("nan") for m in metrics_order}

    stats_out = {
        "n_sequential": len(seq_wall),
        "n_parallel_2w": len(par_wall),
        "mann_whitney_p": pvals,
        "holm_p_adj": holm_dict,
        "cliffs_delta_sequential_vs_parallel": cliffs,
        "bootstrap_median_diff_parallel_minus_sequential": {k: {"lo": v[0], "hi": v[1]} for k, v in boot_med.items()},
        "bootstrap_cliffs_delta": {k: {"lo": v[0], "hi": v[1]} for k, v in boot_cl.items()},
        "note": "Positive Cliff's delta means sequential tends to be larger than parallel for that metric.",
    }
    (OUT / "stats_summary.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    stat_lines = []
    for m in metrics_order:
        stat_lines.append(
            {
                "metric": m,
                "p_mwu": pvals[m],
                "p_holm": holm_dict[m],
                "cliffs_delta": cliffs[m],
                "median_diff_ci_lo": boot_med[m][0],
                "median_diff_ci_hi": boot_med[m][1],
            }
        )
    with (OUT / "stats_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "p_mwu",
                "p_holm",
                "cliffs_delta",
                "median_diff_ci_lo",
                "median_diff_ci_hi",
            ],
        )
        w.writeheader()
        for row in stat_lines:
            w.writerow(row)

    # Figures
    plot_wall_vs_evaluated_genomes_median_iqr(
        [p for _, p in seq_runs], [p for _, p in par_runs], FIG / "wall_time.pdf"
    )
    plot_wall_vs_evaluated_genomes_from0_median_iqr(
        [p for _, p in seq_runs], [p for _, p in par_runs], FIG / "wall_time_from0.pdf"
    )
    plot_throughput_per_execution(rows, FIG / "throughput_per_execution.pdf")
    plot_time_breakdown_aggregate([p for _, p in seq_runs], [p for _, p in par_runs], FIG / "time_breakdown_mean_fractions.pdf")

    print(f"Wrote {OUT}")
    print(f"  sequential runs: {len(seq_runs)}, parallel 2w runs: {len(par_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
