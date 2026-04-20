#!/usr/bin/env python3
"""
C2: ToxSearch-S sequential vs parallel (2w, 4w) — PPSN2026 outputs.

Writes:
  experiments/comparison_results/c2_ppsn2026_seq_vs_2w/
    run_manifest.csv, metrics_per_run.csv, execution_throughput_table.csv,
    stats_summary.json, stats_table.csv
    figures/throughput_wall_vs_evaluated_genomes.pdf (two panels: wall time left, throughput right;
      LNCS/PPSN: width = Springer text width ~122 mm; x-axis 0--1000, major ticks every 200 genomes, no grid)
    figures/best_so_far_vs_wall_time.pdf (median + IQR vs cumulative wall time; no grid; vertical tick
      where each median reaches its final plateau)
    figures/toxicity_diversity_vs_evaluated_genomes.pdf (2x2: top inter/intra diversity; bottom toxicity + species; x ticks every 200)

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

from scipy.stats import kruskal, mannwhitneyu, wilcoxon

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
PAR4_ROOT = DATA / "toxsearch_s_4w"

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


def cumulative_wall_linear_at(xs: Sequence[float], ys: Sequence[float], g: float) -> float:
    """Piecewise-linear cumulative wall time (s) at cumulative evaluated genomes g."""
    if g <= 0:
        return 0.0
    xsa = np.asarray(xs, dtype=float)
    ysa = np.asarray(ys, dtype=float)
    if xsa.size < 2:
        return 0.0
    if g >= float(xsa[-1]):
        return float(ysa[-1])
    j = int(np.searchsorted(xsa, g, side="right"))
    j = max(1, min(j, xsa.size - 1))
    x0, x1 = float(xsa[j - 1]), float(xsa[j])
    y0, y1 = float(ysa[j - 1]), float(ysa[j])
    if x1 <= x0:
        return y1
    return y0 + (g - x0) * (y1 - y0) / (x1 - x0)


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


def speciation_metric_from_generation(gen: Dict[str, Any], key: str) -> Optional[float]:
    """Read a numeric field from generation['speciation'], if present."""
    sp = gen.get("speciation")
    if not isinstance(sp, dict):
        return None
    v = sp.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f):
        return None
    return f


def cumulative_speciation_metric_vs_evaluations(
    tracker: Dict[str, Any],
    spec_key: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative evaluated genomes vs end-of-generation speciation metric (stepwise held).
    Same x-axis construction as cumulative_best_vs_evaluations; y carries the last observed
    value for spec_key after each generation (missing generations repeat the previous value).
    """
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    xe = [0.0]
    yv = [float("nan")]
    last: Optional[float] = None
    for g in gens:
        ev = g.get("evaluated_this_generation")
        if ev is None:
            ev = population_integrated_count(g)
        ev = float(ev) if isinstance(ev, (int, float)) else 0.0
        raw = speciation_metric_from_generation(g, spec_key)
        if raw is not None:
            last = raw
        xe.append(xe[-1] + max(0.0, ev))
        yv.append(last if last is not None else float("nan"))
    return np.asarray(xe, dtype=float), np.asarray(yv, dtype=float)


def cumulative_best_vs_evaluations(tracker: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Cumulative best fitness/toxicity vs cumulative evaluated genomes.
    Starts at (0, 0).
    """
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    xe = [0.0]
    yb = [0.0]
    best = 0.0
    for g in gens:
        ev = g.get("evaluated_this_generation")
        if ev is None:
            ev = population_integrated_count(g)
        ev = float(ev) if isinstance(ev, (int, float)) else 0.0
        b = g.get("best_fitness")
        if b is None:
            b = g.get("max_score_variants")
        if isinstance(b, (int, float)) and not math.isnan(float(b)):
            best = max(best, float(b))
        xe.append(xe[-1] + max(0.0, ev))
        yb.append(best)
    return np.asarray(xe, dtype=float), np.asarray(yb, dtype=float)


def auc_stepwise(xs: Sequence[float], ys: Sequence[float]) -> float:
    """AUC for stepwise/monotone curve using trapezoid over (xs, ys)."""
    if len(xs) < 2:
        return 0.0
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if x[0] != 0.0:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, y[0] if y.size else 0.0)
    # NumPy 2.x: prefer trapezoid; keep compatibility with older installs.
    trapezoid = getattr(np, "trapezoid", None)
    if callable(trapezoid):
        return float(trapezoid(y, x))
    return float(np.trapz(y, x))


def time_to_threshold_seconds(tracker: Dict[str, Any], threshold: float) -> Optional[float]:
    """First cumulative wall time (sum of generation_duration_seconds) where cumulative best >= threshold."""
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    best = 0.0
    cum_t = 0.0
    for g in gens:
        b = g.get("best_fitness")
        if b is None:
            b = g.get("max_score_variants")
        if isinstance(b, (int, float)) and not math.isnan(float(b)):
            best = max(best, float(b))
        dt = g.get("generation_duration_seconds")
        dt = float(dt) if dt is not None and float(dt) >= 0 else 0.0
        cum_t += dt
        if best >= threshold:
            return float(cum_t)
    return None


def final_speciation_metrics(tracker: Dict[str, Any]) -> Dict[str, Optional[float]]:
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return {
            "final_species_count": None,
            "final_active_species_count": None,
            "final_inter_species_diversity": None,
            "final_intra_species_diversity": None,
            "final_silhouette": None,
            "final_davies_bouldin": None,
            "final_calinski_harabasz": None,
        }
    sp = (gens[-1].get("speciation") or {})
    cq = (sp.get("cluster_quality") or {})
    return {
        "final_species_count": float(sp.get("species_count")) if sp.get("species_count") is not None else None,
        "final_active_species_count": float(sp.get("active_species_count")) if sp.get("active_species_count") is not None else None,
        "final_inter_species_diversity": float(sp.get("inter_species_diversity")) if sp.get("inter_species_diversity") is not None else None,
        "final_intra_species_diversity": float(sp.get("intra_species_diversity")) if sp.get("intra_species_diversity") is not None else None,
        "final_silhouette": float(cq.get("silhouette_score")) if cq.get("silhouette_score") is not None else None,
        "final_davies_bouldin": float(cq.get("davies_bouldin_index")) if cq.get("davies_bouldin_index") is not None else None,
        "final_calinski_harabasz": float(cq.get("calinski_harabasz_index")) if cq.get("calinski_harabasz_index") is not None else None,
    }


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
    auc_best_vs_eval: float
    time_to_best_ge_0_10_s: Optional[float]
    time_to_best_ge_0_15_s: Optional[float]
    time_to_best_ge_0_20_s: Optional[float]
    final_species_count: Optional[float]
    final_active_species_count: Optional[float]
    final_inter_species_diversity: Optional[float]
    final_intra_species_diversity: Optional[float]
    final_silhouette: Optional[float]
    final_davies_bouldin: Optional[float]
    final_calinski_harabasz: Optional[float]
    tb_llm_s: float
    tb_moderation_eval_s: float
    tb_speciation_s: float
    tb_overhead_s: float
    tb_wall_s: float
    tb_llm_frac: Optional[float]
    tb_moderation_eval_frac: Optional[float]
    tb_speciation_frac: Optional[float]
    tb_overhead_frac: Optional[float]
    max_total_genomes: Optional[int]


def collect_metrics(runs_by_mode: Dict[str, List[Tuple[str, Path]]]) -> List[RunRow]:
    rows: List[RunRow] = []
    for label, pairs in runs_by_mode.items():
        for run_id, run_dir in pairs:
            t = load_json(run_dir / "EvolutionTracker.json")
            rm = t.get("run_metadata") or {}
            wall = float(rm.get("run_duration_seconds") or 0)
            mtg = rm.get("max_total_genomes")
            mtg_i = int(mtg) if isinstance(mtg, (int, float)) else None
            n_int = total_integrated(t)
            thr = (n_int / wall) if wall > 0 else 0.0
            qm = qmax_tracker(t)
            xe, yb = cumulative_best_vs_evaluations(t)
            auc = auc_stepwise(xe.tolist(), yb.tolist())
            t10 = time_to_threshold_seconds(t, 0.10)
            t15 = time_to_threshold_seconds(t, 0.15)
            t20 = time_to_threshold_seconds(t, 0.20)
            spf = final_speciation_metrics(t)
            tb = time_breakdown_totals(t)
            tot = float(tb["llm"] + tb["moderation_eval"] + tb["speciation"] + tb["overhead"])
            llm_f = float(tb["llm"] / tot) if tot > 0 else None
            mod_f = float(tb["moderation_eval"] / tot) if tot > 0 else None
            sp_f = float(tb["speciation"] / tot) if tot > 0 else None
            oh_f = float(tb["overhead"] / tot) if tot > 0 else None
            rows.append(
                RunRow(
                    run_id=run_id,
                    mode=label,
                    run_dir=run_dir,
                    wall_s=wall,
                    total_integrated=n_int,
                    throughput=thr,
                    qmax=qm,
                    auc_best_vs_eval=auc,
                    time_to_best_ge_0_10_s=t10,
                    time_to_best_ge_0_15_s=t15,
                    time_to_best_ge_0_20_s=t20,
                    final_species_count=spf["final_species_count"],
                    final_active_species_count=spf["final_active_species_count"],
                    final_inter_species_diversity=spf["final_inter_species_diversity"],
                    final_intra_species_diversity=spf["final_intra_species_diversity"],
                    final_silhouette=spf["final_silhouette"],
                    final_davies_bouldin=spf["final_davies_bouldin"],
                    final_calinski_harabasz=spf["final_calinski_harabasz"],
                    tb_llm_s=float(tb["llm"]),
                    tb_moderation_eval_s=float(tb["moderation_eval"]),
                    tb_speciation_s=float(tb["speciation"]),
                    tb_overhead_s=float(tb["overhead"]),
                    tb_wall_s=float(tb["wall"]),
                    tb_llm_frac=llm_f,
                    tb_moderation_eval_frac=mod_f,
                    tb_speciation_frac=sp_f,
                    tb_overhead_frac=oh_f,
                    max_total_genomes=mtg_i,
                )
            )
    return rows


def plot_throughput_wall_vs_evaluated_genomes_panels(
    dirs_by_mode: Dict[str, List[Path]],
    out_path: Path,
    wall_milestones: Sequence[int] = tuple(range(0, 1100, 100)),
    throughput_milestones: Sequence[int] = tuple(range(100, 1025, 25)),
) -> None:
    """
    Single figure, two panels: cumulative wall time (left) and cumulative throughput (right)
    vs evaluated genomes. Both axes span 0--1000 with major ticks every 200 genomes (no grid).
    Throughput curve uses 25-genome milestones from 100--1000; wall time uses 100-genome steps from 0.
    Per mode: mean across runs with min--max band; solid lines only (no markers). Legend on the
    throughput panel (upper right), labels match the toxicity--diversity figure: Sequential,
    Parallel (2w), Parallel (4w).
    Figure size matches Springer LNCS full text width (~122 mm) for PPSN-style papers.
    """
    def wall_series_for_run(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yw = cumulative_wall_vs_evaluations(t)
        pairs = sorted(zip(xe.tolist(), yw.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        return [stepwise_at(xs2, ys2, float(m)) for m in wall_milestones]

    def throughput_series_for_run(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yw = cumulative_wall_vs_evaluations(t)
        pairs = sorted(zip(xe.tolist(), yw.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        out: List[float] = []
        for m in throughput_milestones:
            w = cumulative_wall_linear_at(xs2, ys2, float(m))
            if m > 0 and w > 1e-9:
                out.append(m / w)
            else:
                out.append(float("nan"))
        return out

    s_w = [wall_series_for_run(d) for d in dirs_by_mode.get("sequential", [])]
    p2_w = [wall_series_for_run(d) for d in dirs_by_mode.get("parallel_2w", [])]
    p4_w = [wall_series_for_run(d) for d in dirs_by_mode.get("parallel_4w", [])]
    s_t = [throughput_series_for_run(d) for d in dirs_by_mode.get("sequential", [])]
    p2_t = [throughput_series_for_run(d) for d in dirs_by_mode.get("parallel_2w", [])]
    p4_t = [throughput_series_for_run(d) for d in dirs_by_mode.get("parallel_4w", [])]
    if not s_w or not p2_w or not p4_w or not s_t or not p2_t or not p4_t:
        return

    Sw = np.asarray(s_w, dtype=float)
    P2w = np.asarray(p2_w, dtype=float)
    P4w = np.asarray(p4_w, dtype=float)
    xw = np.asarray(list(wall_milestones), dtype=float)
    mean_sw = np.nanmean(Sw, axis=0)
    mean_p2w = np.nanmean(P2w, axis=0)
    mean_p4w = np.nanmean(P4w, axis=0)
    min_sw, max_sw = np.nanmin(Sw, axis=0), np.nanmax(Sw, axis=0)
    min_p2w, max_p2w = np.nanmin(P2w, axis=0), np.nanmax(P2w, axis=0)
    min_p4w, max_p4w = np.nanmin(P4w, axis=0), np.nanmax(P4w, axis=0)

    St = np.asarray(s_t, dtype=float)
    P2t = np.asarray(p2_t, dtype=float)
    P4t = np.asarray(p4_t, dtype=float)
    xt = np.asarray(list(throughput_milestones), dtype=float)
    mean_st = np.nanmean(St, axis=0)
    mean_p2t = np.nanmean(P2t, axis=0)
    mean_p4t = np.nanmean(P4t, axis=0)
    min_st, max_st = np.nanmin(St, axis=0), np.nanmax(St, axis=0)
    min_p2t, max_p2t = np.nanmin(P2t, axis=0), np.nanmax(P2t, axis=0)
    min_p4t, max_p4t = np.nanmin(P4t, axis=0), np.nanmax(P4t, axis=0)

    series_cfg: List[
        Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]
    ] = [
        ("Sequential", mean_st, min_st, max_st, mean_sw, min_sw, max_sw, "#1f77b4"),
        ("Parallel (2w)", mean_p2t, min_p2t, max_p2t, mean_p2w, min_p2w, max_p2w, "#ff7f0e"),
        ("Parallel (4w)", mean_p4t, min_p4t, max_p4t, mean_p4w, min_p4w, max_p4w, "#2ca02c"),
    ]

    # Springer LNCS (llncs.cls): \textwidth ≈ 122 mm — use that as figure width for \linewidth inclusion.
    _ln_mm = 122.0
    fig_w_in = _ln_mm / 25.4
    fig_h_in = fig_w_in * 0.48
    fig, (ax_wall, ax_thr) = plt.subplots(
        1,
        2,
        figsize=(fig_w_in, fig_h_in),
        sharey=False,
        constrained_layout=False,
    )

    for label, mnt, lot, hit, mnw, low, hiw, color in series_cfg:
        ax_wall.fill_between(xw, low, hiw, color=color, alpha=0.22, linewidth=0.0, zorder=1)
        ax_wall.plot(
            xw,
            mnw,
            color=color,
            linestyle="-",
            linewidth=1.35,
            zorder=3,
        )
        ax_thr.fill_between(xt, lot, hit, color=color, alpha=0.22, linewidth=0.0, zorder=1)
        ax_thr.plot(
            xt,
            mnt,
            color=color,
            linestyle="-",
            linewidth=1.35,
            label=label,
            zorder=3,
        )

    x_ticks_major = list(range(0, 1001, 200))
    x_lim = (0.0, 1000.0)

    ax_wall.set_xlabel("Evaluated genomes", fontsize=7.5)
    ax_wall.set_ylabel("Wall-clock time (s)", fontsize=7.5)
    ax_wall.set_xlim(*x_lim)
    ax_wall.set_xticks(x_ticks_major)
    ymax_w = float(np.nanmax(np.concatenate([max_sw, max_p2w, max_p4w, mean_sw, mean_p2w, mean_p4w])))
    ax_wall.set_ylim(0.0, ymax_w * 1.05 if np.isfinite(ymax_w) and ymax_w > 0 else 1.0)
    ax_wall.tick_params(axis="both", which="major", labelsize=6.75, length=3.6, width=0.8)
    ax_wall.yaxis.get_offset_text().set_fontsize(6.75)
    ax_wall.minorticks_off()

    ax_thr.set_xlabel("Evaluated genomes", fontsize=7.5)
    ax_thr.set_ylabel("Throughput (genomes / s)", fontsize=7.5)
    ax_thr.set_xlim(*x_lim)
    ax_thr.set_xticks(x_ticks_major)
    ymax_t = float(np.nanmax(np.concatenate([max_st, max_p2t, max_p4t, mean_st, mean_p2t, mean_p4t])))
    ax_thr.set_ylim(0.0, ymax_t * 1.06 if np.isfinite(ymax_t) and ymax_t > 0 else 0.1)
    ax_thr.tick_params(axis="both", which="major", labelsize=6.75, length=3.6, width=0.8)
    ax_thr.yaxis.get_offset_text().set_fontsize(6.75)
    ax_thr.minorticks_off()

    leg = ax_thr.legend(
        frameon=False,
        fontsize=6.25,
        loc="upper right",
        bbox_to_anchor=(1.0, 1.0),
        bbox_transform=ax_thr.transAxes,
        borderaxespad=0.08,
        handlelength=2.0,
        labelspacing=0.3,
    )
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(1.0)

    fig.subplots_adjust(left=0.12, right=0.99, top=0.96, bottom=0.16, wspace=0.42)
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)


def plot_toxicity_diversity_vs_evaluated_genomes(
    dirs_by_mode: Dict[str, List[Path]],
    out_path: Path,
    milestones: Sequence[int] = tuple(range(0, 1100, 100)),
    diversity_milestones: Sequence[int] = tuple(range(100, 1100, 100)),
) -> None:
    """
    Four-panel 2x2 grid (LNCS text width): row 1---inter- and intra-species diversity (legend
    on intra, upper right); row 2---best-so-far toxicity and species count. All panels use
    x in [0,1000] with major ticks every 200 genomes; diversity curves still sampled every
    100 genomes from the tracker.
    """
    def toxicity_series(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yb = cumulative_best_vs_evaluations(t)
        pairs = sorted(zip(xe.tolist(), yb.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        return [stepwise_at(xs2, ys2, float(m)) for m in milestones]

    def diversity_series(run_dir: Path, spec_key: str) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        xe, yv = cumulative_speciation_metric_vs_evaluations(t, spec_key)
        pairs = sorted(zip(xe.tolist(), yv.tolist()), key=lambda z: float(z[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys_raw = [float(p[1]) for p in pairs]
        ys2: List[float] = []
        carry = float("nan")
        for y in ys_raw:
            if math.isfinite(y):
                carry = y
            ys2.append(carry)
        return [stepwise_at(xs2, ys2, float(m)) for m in diversity_milestones]

    def collect_matrix(mode_key: str, fn, n_m: int) -> np.ndarray:
        rows = [fn(d) for d in dirs_by_mode.get(mode_key, [])]
        return np.asarray(rows, dtype=float) if rows else np.zeros((0, n_m))

    S = collect_matrix("sequential", toxicity_series, len(milestones))
    P2 = collect_matrix("parallel_2w", toxicity_series, len(milestones))
    P4 = collect_matrix("parallel_4w", toxicity_series, len(milestones))
    if S.size == 0 or P2.size == 0 or P4.size == 0:
        return

    x = np.asarray(list(milestones), dtype=float)
    x_div = np.asarray(list(diversity_milestones), dtype=float)

    def med_iqr(M: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.nanmedian(M, axis=0),
            np.nanquantile(M, 0.25, axis=0),
            np.nanquantile(M, 0.75, axis=0),
        )

    med_s, q1_s, q3_s = med_iqr(S)
    med_p2, q1_p2, q3_p2 = med_iqr(P2)
    med_p4, q1_p4, q3_p4 = med_iqr(P4)

    _ln_mm = 122.0
    fig_w = _ln_mm / 25.4
    fig_h = fig_w * 0.98
    fig, axes = plt.subplots(2, 2, figsize=(fig_w, fig_h), squeeze=False)
    # Top: diversity trajectories; bottom: toxicity + species count.
    ax_inter = axes[0, 0]
    ax_intra = axes[0, 1]
    ax_tox = axes[1, 0]
    ax_sp = axes[1, 1]

    x_ticks_200 = list(range(0, 1001, 200))

    def plot_diversity_ax(
        dax: Any,
        spec_key: str,
        ylab: str,
        *,
        hide_xticklabels: bool,
        set_xlabel: bool,
        add_legend: bool,
    ) -> None:
        n_div = len(diversity_milestones)
        Sd = collect_matrix("sequential", lambda rd, k=spec_key: diversity_series(rd, k), n_div)
        P2d = collect_matrix("parallel_2w", lambda rd, k=spec_key: diversity_series(rd, k), n_div)
        P4d = collect_matrix("parallel_4w", lambda rd, k=spec_key: diversity_series(rd, k), n_div)
        if Sd.size == 0 or P2d.size == 0 or P4d.size == 0:
            return
        m1, a1, b1 = med_iqr(Sd)
        m2, a2, b2 = med_iqr(P2d)
        m3, a3, b3 = med_iqr(P4d)
        kw1: Dict[str, Any] = {"linewidth": 1.35, "color": "#1f77b4"}
        kw2: Dict[str, Any] = {"linewidth": 1.35, "color": "#ff7f0e"}
        kw3: Dict[str, Any] = {"linewidth": 1.35, "color": "#2ca02c"}
        if add_legend:
            kw1["label"] = "Sequential"
            kw2["label"] = "Parallel (2w)"
            kw3["label"] = "Parallel (4w)"
        dax.plot(x_div, m1, **kw1)
        dax.fill_between(x_div, a1, b1, alpha=0.2, color="#1f77b4", linewidth=0)
        dax.plot(x_div, m2, **kw2)
        dax.fill_between(x_div, a2, b2, alpha=0.2, color="#ff7f0e", linewidth=0)
        dax.plot(x_div, m3, **kw3)
        dax.fill_between(x_div, a3, b3, alpha=0.2, color="#2ca02c", linewidth=0)
        dax.set_ylabel(ylab, fontsize=7.5)
        dax.set_xlim(0.0, 1000.0)
        dax.set_xticks(x_ticks_200)
        dax.tick_params(axis="both", which="major", labelsize=6.75)
        dax.minorticks_off()
        if hide_xticklabels:
            plt.setp(dax.get_xticklabels(), visible=False)
        if set_xlabel:
            dax.set_xlabel("Evaluated genomes", fontsize=7.5)
        if add_legend:
            leg = dax.legend(
                frameon=False,
                fontsize=6.25,
                loc="upper right",
                bbox_to_anchor=(1.0, 1.0),
                bbox_transform=dax.transAxes,
                borderaxespad=0.08,
                handlelength=2.0,
                labelspacing=0.3,
            )
            for leg_line in leg.get_lines():
                leg_line.set_linewidth(1.0)

    plot_diversity_ax(
        ax_inter,
        "inter_species_diversity",
        "Inter-species diversity",
        hide_xticklabels=True,
        set_xlabel=False,
        add_legend=False,
    )
    plot_diversity_ax(
        ax_intra,
        "intra_species_diversity",
        "Intra-species diversity",
        hide_xticklabels=True,
        set_xlabel=False,
        add_legend=True,
    )

    ax_tox.plot(x, med_s, linewidth=1.35, color="#1f77b4")
    ax_tox.fill_between(x, q1_s, q3_s, alpha=0.2, color="#1f77b4", linewidth=0)
    ax_tox.plot(x, med_p2, linewidth=1.35, color="#ff7f0e")
    ax_tox.fill_between(x, q1_p2, q3_p2, alpha=0.2, color="#ff7f0e", linewidth=0)
    ax_tox.plot(x, med_p4, linewidth=1.35, color="#2ca02c")
    ax_tox.fill_between(x, q1_p4, q3_p4, alpha=0.2, color="#2ca02c", linewidth=0)
    ax_tox.set_ylabel("Best-so-far toxicity", fontsize=7.5)
    ax_tox.set_xlim(0, 1000)
    ax_tox.set_xticks(x_ticks_200)
    ax_tox.set_ylim(bottom=0.0)
    ax_tox.tick_params(axis="both", which="major", labelsize=6.75)
    ax_tox.minorticks_off()
    ax_tox.set_xlabel("Evaluated genomes", fontsize=7.5)

    plot_diversity_ax(
        ax_sp,
        "species_count",
        "Species count",
        hide_xticklabels=False,
        set_xlabel=True,
        add_legend=False,
    )

    fig.subplots_adjust(left=0.10, right=0.98, top=0.96, bottom=0.14, hspace=0.35, wspace=0.36)
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)


def _plateau_onset_index(y: np.ndarray, *, rtol: float = 1e-9, atol: float = 1e-12) -> Optional[int]:
    """Smallest i > 0 such that y[i:] is flat at the terminal value; None if not applicable."""
    yf = np.asarray(y, dtype=float)
    if yf.size < 2:
        return None
    tail = float(yf[-1])
    for i in range(yf.size):
        if bool(np.all(np.isclose(yf[i:], tail, rtol=rtol, atol=atol))):
            return int(i) if i > 0 else None
    return None


def plot_best_vs_wall_time_median_iqr(
    dirs_by_mode: Dict[str, List[Path]],
    out_path: Path,
    milestones_s: Sequence[int] = (0, 3000, 6000, 9000, 12000, 15000, 18000, 21000, 24000, 27000, 30000, 36000, 42000, 48000),
) -> None:
    """
    Cumulative best-so-far toxicity vs cumulative wall time (s): median + IQR per mode.
    LNCS text width, no grid, typography aligned with other C2 figures; dashed vertical line
    at the first wall-time milestone where the median has reached its final (flat) value.
    """

    def series_for_run(run_dir: Path) -> List[float]:
        t = load_json(run_dir / "EvolutionTracker.json")
        gens = sorted(t.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
        xs = [0.0]
        ys = [0.0]
        best = 0.0
        cum_t = 0.0
        for g in gens:
            b = g.get("best_fitness")
            if b is None:
                b = g.get("max_score_variants")
            if isinstance(b, (int, float)) and not math.isnan(float(b)):
                best = max(best, float(b))
            dt = g.get("generation_duration_seconds")
            dt = float(dt) if dt is not None and float(dt) >= 0 else 0.0
            cum_t += dt
            xs.append(cum_t)
            ys.append(best)
        return [stepwise_at(xs, ys, float(m)) for m in milestones_s]

    rows_s = [series_for_run(d) for d in dirs_by_mode.get("sequential", [])]
    rows_p2 = [series_for_run(d) for d in dirs_by_mode.get("parallel_2w", [])]
    rows_p4 = [series_for_run(d) for d in dirs_by_mode.get("parallel_4w", [])]
    if not rows_s or not rows_p2 or not rows_p4:
        return

    S = np.asarray(rows_s, dtype=float)
    P2 = np.asarray(rows_p2, dtype=float)
    P4 = np.asarray(rows_p4, dtype=float)

    x = np.asarray(list(milestones_s), dtype=float)
    med_s, q1_s, q3_s = np.median(S, axis=0), np.quantile(S, 0.25, axis=0), np.quantile(S, 0.75, axis=0)
    med_p2, q1_p2, q3_p2 = np.median(P2, axis=0), np.quantile(P2, 0.25, axis=0), np.quantile(P2, 0.75, axis=0)
    med_p4, q1_p4, q3_p4 = np.median(P4, axis=0), np.quantile(P4, 0.25, axis=0), np.quantile(P4, 0.75, axis=0)

    _ln_mm = 122.0
    fig_w_in = _ln_mm / 25.4
    fig_h_in = fig_w_in * 0.52
    fig, ax = plt.subplots(1, 1, figsize=(fig_w_in, fig_h_in))

    series_plot: List[Tuple[str, np.ndarray, np.ndarray, np.ndarray, str]] = [
        ("Sequential", med_s, q1_s, q3_s, "#1f77b4"),
        ("Parallel (2w)", med_p2, q1_p2, q3_p2, "#ff7f0e"),
        ("Parallel (4w)", med_p4, q1_p4, q3_p4, "#2ca02c"),
    ]
    for label, med, q1, q3, color in series_plot:
        ax.fill_between(x, q1, q3, alpha=0.22, color=color, linewidth=0.0, zorder=1)
        ax.plot(x, med, linewidth=1.35, color=color, label=label, zorder=3)
        pi = _plateau_onset_index(med)
        if pi is not None:
            ax.axvline(
                float(x[pi]),
                color=color,
                linestyle=(0, (4, 3)),
                linewidth=0.9,
                alpha=0.55,
                zorder=2,
            )

    ax.set_xlabel("Wall-clock time (s, cumulative)", fontsize=7.5)
    ax.set_ylabel("Best-so-far toxicity", fontsize=7.5)
    ax.set_xlim(left=0.0)
    ax.set_ylim(bottom=0.0)
    ax.tick_params(axis="both", which="major", labelsize=6.75, length=3.6, width=0.8)
    ax.yaxis.get_offset_text().set_fontsize(6.75)
    ax.minorticks_off()

    leg = ax.legend(
        frameon=False,
        fontsize=6.25,
        loc="lower right",
        bbox_to_anchor=(1.0, 0.0),
        bbox_transform=ax.transAxes,
        borderaxespad=0.08,
        handlelength=2.0,
        labelspacing=0.3,
    )
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(1.0)

    fig.subplots_adjust(left=0.11, right=0.99, top=0.96, bottom=0.16)
    fig.savefig(out_path, format="pdf", dpi=300)
    plt.close(fig)


def main() -> int:
    seq_runs = discover_runs(SEQ_ROOT, "sequential", 1)
    par_runs = discover_runs(PAR_ROOT, "parallel", 2)
    par4_runs = discover_runs(PAR4_ROOT, "parallel", 4)

    runs_by_mode = {
        "sequential": seq_runs,
        "parallel_2w": par_runs,
        "parallel_4w": par4_runs,
    }
    rows = collect_metrics(runs_by_mode)

    workers_expected = {"sequential": 1, "parallel_2w": 2, "parallel_4w": 4}

    with (OUT / "run_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "run_id", "run_path", "wall_s", "num_workers_expected"])
        for r in rows:
            w.writerow([r.mode, r.run_id, str(r.run_dir), r.wall_s, workers_expected.get(r.mode, "")])

    with (OUT / "metrics_per_run.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "mode",
            "run_id",
            "wall_s",
            "total_integrated",
            "throughput_integrated_per_s",
            "qmax_tracker",
            "auc_best_vs_eval",
            "time_to_best_ge_0_10_s",
            "time_to_best_ge_0_15_s",
            "time_to_best_ge_0_20_s",
            "final_species_count",
            "final_active_species_count",
            "final_inter_species_diversity",
            "final_intra_species_diversity",
            "final_silhouette",
            "final_davies_bouldin",
            "final_calinski_harabasz",
            "tb_llm_s",
            "tb_moderation_eval_s",
            "tb_speciation_s",
            "tb_overhead_s",
            "tb_wall_s_sum_generations",
            "tb_llm_frac",
            "tb_moderation_eval_frac",
            "tb_speciation_frac",
            "tb_overhead_frac",
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
                    "auc_best_vs_eval": r.auc_best_vs_eval,
                    "time_to_best_ge_0_10_s": r.time_to_best_ge_0_10_s if r.time_to_best_ge_0_10_s is not None else "",
                    "time_to_best_ge_0_15_s": r.time_to_best_ge_0_15_s if r.time_to_best_ge_0_15_s is not None else "",
                    "time_to_best_ge_0_20_s": r.time_to_best_ge_0_20_s if r.time_to_best_ge_0_20_s is not None else "",
                    "final_species_count": r.final_species_count if r.final_species_count is not None else "",
                    "final_active_species_count": r.final_active_species_count if r.final_active_species_count is not None else "",
                    "final_inter_species_diversity": r.final_inter_species_diversity if r.final_inter_species_diversity is not None else "",
                    "final_intra_species_diversity": r.final_intra_species_diversity if r.final_intra_species_diversity is not None else "",
                    "final_silhouette": r.final_silhouette if r.final_silhouette is not None else "",
                    "final_davies_bouldin": r.final_davies_bouldin if r.final_davies_bouldin is not None else "",
                    "final_calinski_harabasz": r.final_calinski_harabasz if r.final_calinski_harabasz is not None else "",
                    "tb_llm_s": r.tb_llm_s,
                    "tb_moderation_eval_s": r.tb_moderation_eval_s,
                    "tb_speciation_s": r.tb_speciation_s,
                    "tb_overhead_s": r.tb_overhead_s,
                    "tb_wall_s_sum_generations": r.tb_wall_s,
                    "tb_llm_frac": r.tb_llm_frac if r.tb_llm_frac is not None else "",
                    "tb_moderation_eval_frac": r.tb_moderation_eval_frac if r.tb_moderation_eval_frac is not None else "",
                    "tb_speciation_frac": r.tb_speciation_frac if r.tb_speciation_frac is not None else "",
                    "tb_overhead_frac": r.tb_overhead_frac if r.tb_overhead_frac is not None else "",
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

    rng = random.Random(RNG_SEED)
    # --- stats (3 modes): Kruskal–Wallis + pairwise MWU with Holm adjustment per metric ---
    modes = ["sequential", "parallel_2w", "parallel_4w"]
    metrics_order = [
        "wall_s",
        "throughput",
        "qmax",
        "auc_best_vs_eval",
        "final_inter_species_diversity",
        "final_intra_species_diversity",
        "final_silhouette",
        "final_davies_bouldin",
        "final_calinski_harabasz",
    ]

    def metric_values(metric: str, mode: str) -> List[float]:
        vals: List[float] = []
        for r in rows:
            if r.mode != mode:
                continue
            v = getattr(r, metric)
            if v is None:
                continue
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            vals.append(fv)
        return vals

    kw_p: Dict[str, float] = {}
    pairwise: Dict[str, Dict[str, Any]] = {}
    for m in metrics_order:
        groups = [metric_values(m, mode) for mode in modes]
        if all(len(g) >= 1 for g in groups):
            _, p = kruskal(*groups)
            kw_p[m] = float(p)
        else:
            kw_p[m] = float("nan")

        comps = [("sequential", "parallel_2w"), ("sequential", "parallel_4w"), ("parallel_2w", "parallel_4w")]
        pw_rows: List[Dict[str, Any]] = []
        raw_ps: List[float] = []
        for a_mode, b_mode in comps:
            a = metric_values(m, a_mode)
            b = metric_values(m, b_mode)
            if len(a) >= 1 and len(b) >= 1:
                _, p = mannwhitneyu(a, b, alternative="two-sided")
                p = float(p)
                raw_ps.append(p)
                cd = cliffs_delta(a, b)
                ci_med = bootstrap_ci_median_diff(a, b, rng)
                ci_cd = bootstrap_ci_cliffs(a, b, rng)
                pw_rows.append(
                    {
                        "a": a_mode,
                        "b": b_mode,
                        "p_mwu": p,
                        "cliffs_delta_a_vs_b": cd,
                        "median_diff_ci_lo_b_minus_a": ci_med[0],
                        "median_diff_ci_hi_b_minus_a": ci_med[1],
                        "cliffs_ci_lo": ci_cd[0],
                        "cliffs_ci_hi": ci_cd[1],
                    }
                )
            else:
                raw_ps.append(float("nan"))
                pw_rows.append(
                    {
                        "a": a_mode,
                        "b": b_mode,
                        "p_mwu": float("nan"),
                        "cliffs_delta_a_vs_b": float("nan"),
                        "median_diff_ci_lo_b_minus_a": float("nan"),
                        "median_diff_ci_hi_b_minus_a": float("nan"),
                        "cliffs_ci_lo": float("nan"),
                        "cliffs_ci_hi": float("nan"),
                    }
                )

        valid_idx = [i for i, p in enumerate(raw_ps) if not math.isnan(float(p))]
        adj = [float("nan")] * len(raw_ps)
        if valid_idx:
            adj_vals = holm_adjust([raw_ps[i] for i in valid_idx])
            for i, ap in zip(valid_idx, adj_vals):
                adj[i] = float(ap)
        for i in range(len(pw_rows)):
            pw_rows[i]["p_holm"] = adj[i]

        pairwise[m] = {"comparisons": pw_rows}

    stats_out = {
        "n_by_mode": {m: len([r for r in rows if r.mode == m]) for m in modes},
        "kruskal_p": kw_p,
        "pairwise_mwu": pairwise,
        "paired_by_execution_index": {},
        "note": "Cliff's delta is reported as delta(a,b): positive means a tends to be larger than b for that metric.",
    }

    # Paired sensitivity analysis: pair runs by sorted run_id index within each mode.
    # This is NOT a guarantee of identical random seeds (seed is not stored in tracker metadata),
    # but it provides a deterministic robustness check when cohorts are intended to be paired.
    def paired_series(metric: str, a_mode: str, b_mode: str) -> Tuple[List[float], List[float]]:
        a_rows = sorted([r for r in rows if r.mode == a_mode], key=lambda r: r.run_id)
        b_rows = sorted([r for r in rows if r.mode == b_mode], key=lambda r: r.run_id)
        n = min(len(a_rows), len(b_rows))
        ax: List[float] = []
        bx: List[float] = []
        for i in range(n):
            av = getattr(a_rows[i], metric)
            bv = getattr(b_rows[i], metric)
            if av is None or bv is None:
                continue
            af = float(av)
            bf = float(bv)
            if math.isnan(af) or math.isnan(bf):
                continue
            ax.append(af)
            bx.append(bf)
        return ax, bx

    paired_comps = [("sequential", "parallel_2w"), ("sequential", "parallel_4w"), ("parallel_2w", "parallel_4w")]
    paired_out: Dict[str, Any] = {}
    for m in metrics_order:
        rows_m: List[Dict[str, Any]] = []
        for a_mode, b_mode in paired_comps:
            a, b = paired_series(m, a_mode, b_mode)
            if len(a) >= 2 and len(b) >= 2 and len(a) == len(b):
                try:
                    # Wilcoxon tests median of (b-a) != 0 by default (two-sided).
                    res = wilcoxon(np.asarray(b) - np.asarray(a), alternative="two-sided", zero_method="wilcox")
                    p = float(res.pvalue)
                except Exception:
                    p = float("nan")
                diffs = [float(bi - ai) for ai, bi in zip(a, b)]
                rows_m.append(
                    {
                        "a": a_mode,
                        "b": b_mode,
                        "n_pairs": len(diffs),
                        "p_wilcoxon": p,
                        "median_diff_b_minus_a": float(np.median(diffs)) if diffs else float("nan"),
                    }
                )
            else:
                rows_m.append(
                    {
                        "a": a_mode,
                        "b": b_mode,
                        "n_pairs": min(len(a), len(b)),
                        "p_wilcoxon": float("nan"),
                        "median_diff_b_minus_a": float("nan"),
                    }
                )
        paired_out[m] = {"comparisons": rows_m}
    stats_out["paired_by_execution_index"] = paired_out
    (OUT / "stats_summary.json").write_text(json.dumps(stats_out, indent=2), encoding="utf-8")

    with (OUT / "stats_table.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "metric",
            "p_kruskal",
            "a",
            "b",
            "p_mwu",
            "p_holm",
            "cliffs_delta_a_vs_b",
            "median_diff_ci_lo_b_minus_a",
            "median_diff_ci_hi_b_minus_a",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for m in metrics_order:
            for row in pairwise[m]["comparisons"]:
                w.writerow(
                    {
                        "metric": m,
                        "p_kruskal": kw_p.get(m, float("nan")),
                        "a": row["a"],
                        "b": row["b"],
                        "p_mwu": row["p_mwu"],
                        "p_holm": row["p_holm"],
                        "cliffs_delta_a_vs_b": row["cliffs_delta_a_vs_b"],
                        "median_diff_ci_lo_b_minus_a": row["median_diff_ci_lo_b_minus_a"],
                        "median_diff_ci_hi_b_minus_a": row["median_diff_ci_hi_b_minus_a"],
                    }
                )

    # Figures
    plot_throughput_wall_vs_evaluated_genomes_panels(
        {
            "sequential": [p for _, p in seq_runs],
            "parallel_2w": [p for _, p in par_runs],
            "parallel_4w": [p for _, p in par4_runs],
        },
        FIG / "throughput_wall_vs_evaluated_genomes.pdf",
    )
    plot_toxicity_diversity_vs_evaluated_genomes(
        {
            "sequential": [p for _, p in seq_runs],
            "parallel_2w": [p for _, p in par_runs],
            "parallel_4w": [p for _, p in par4_runs],
        },
        FIG / "toxicity_diversity_vs_evaluated_genomes.pdf",
    )
    plot_best_vs_wall_time_median_iqr(
        {
            "sequential": [p for _, p in seq_runs],
            "parallel_2w": [p for _, p in par_runs],
            "parallel_4w": [p for _, p in par4_runs],
        },
        FIG / "best_so_far_vs_wall_time.pdf",
    )
    print(f"Wrote {OUT}")
    print(
        f"  sequential runs: {len(seq_runs)}, parallel 2w runs: {len(par_runs)}, parallel 4w runs: {len(par4_runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
