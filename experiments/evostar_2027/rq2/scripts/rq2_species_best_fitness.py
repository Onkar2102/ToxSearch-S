#!/usr/bin/env python3
"""RQ2 Options A+B: species-best fitness coverage.

For each mature species, take F_i^max = max illicit among members (stored scores
already include the refusal penalty when applied).

Pool:
  LF: elites.json + reserves.json (if present) + archive.json
  IncDBSCAN: temp.json live members (noise included for unassigned counts)

Figures (per-bin counts, not cumulative):
  fig_rq2_species_coverage_tau: boxplots of species per F_i^max bin (0.0..0.9)
  fig_rq2_species_coverage_runs: run-level species per bin (from 0.5)
  fig_rq2_species_coverage_runs_singletons: split boxplots per bin
    (left half = mature; right half = noise/singletons)
  fig_rq2_species_fitness_bins_appendix: median species counts over all 0.1 bins

Also writes runs/bins/coverage CSVs and rq2_species_best_table.tex.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import PowerNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import NOISE_SIDS, illicit_of, load_json, moderated_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
FIG_DIR = ROOT / "experiments" / "evostar_2027" / "rq2" / "figures"
PAPER_FIG = ROOT / "notes" / "rough_draft" / "figures"
PAPER_TEX = ROOT / "notes" / "rough_draft" / "rq2_species_best_table.tex"
PAPER_FIG_TEX = ROOT / "notes" / "rough_draft" / "rq2_species_coverage_figure.tex"

BUDGET = 2000
ALL_BINS = np.round(np.arange(0.0, 1.0, 0.1), 1)
BIN_EDGES = np.round(np.linspace(0.0, 1.0, 11), 1)
BIN_LABELS = [f"{BIN_EDGES[i]:.1f}–{BIN_EDGES[i+1]:.1f}" for i in range(10)]
# Coverage curve still records cumulative n_ge_tau for the summary table.
PLOT_TAUS = tuple(float(x) for x in np.round(np.arange(0.0, 1.0, 0.1), 1))
DETAIL_TAUS = (0.7, 0.8, 0.9)
TABLE_TAUS = (0.7, 0.8)
# Tau figure: all 0.1 bins. Runs figure: high-fitness bins from 0.6.
ALL_BIN_LOS = tuple(float(x) for x in ALL_BINS)
# Tick labels use the upper edge of each bin: [0.0,0.1)→0.1, …, [0.9,1.0]→1.0
ALL_BIN_LABELS = [f"{lo + 0.1:.1f}" for lo in ALL_BIN_LOS]
HIGH_BIN_LOS = tuple(float(x) for x in np.round(np.arange(0.6, 1.0, 0.1), 1))
HIGH_BIN_LABELS = [f"{lo:.1f}–{lo + 0.1:.1f}" for lo in HIGH_BIN_LOS]

# cell, label, color, linestyle, marker — four distinct hues
CONFIG: List[Tuple[str, str, str, str, str]] = [
    ("lf_embedding", r"LF $d_E$", "#0F4D92", "-", "o"),
    ("inc_embedding", r"Inc $d_E$", "#2A9D8F", "--", "s"),
    ("lf_nli", r"LF $d_{\neg E}$", "#B64342", "-", "^"),
    ("inc_nli", r"Inc $d_{\neg E}$", "#E9C46A", "--", "D"),
]

CELL_META = {
    "lf_embedding": ("leader_follower", "embedding"),
    "inc_embedding": ("dbscan", "embedding"),
    "lf_nli": ("leader_follower", "nli_entail"),
    "inc_nli": ("dbscan", "nli_entail"),
}


def publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8.5,
            "axes.linewidth": 0.9,
            "axes.labelsize": 8.5,
            "axes.labelpad": 3,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.fontsize": 7.0,
            "legend.frameon": True,
            "legend.handlelength": 1.55,
            "legend.handletextpad": 0.40,
            "legend.borderpad": 0.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "savefig.dpi": 400,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.facecolor": "#FFFFFF",
            "figure.facecolor": "#FFFFFF",
        }
    )


def q(vals: Sequence[float]) -> np.ndarray:
    return np.percentile(np.asarray(vals, dtype=float), [25, 50, 75])


def export(fig: plt.Figure, stem: str, outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.pdf")
    fig.savefig(outdir / f"{stem}.png")
    fig.savefig(PAPER_FIG / f"{stem}.pdf")
    plt.close(fig)


def load_species_pool(run_dir: Path, clustering: str) -> List[dict]:
    """LF: elites + reserves (legacy) + archive. IncDBSCAN: live temp.json.

    ``reserves.json`` is loaded when present (older runs). Current RQ2 LF
    dirs typically only ship ``elites.json`` and ``archive.json``.
    """
    names = (
        ("temp.json",)
        if clustering == "dbscan"
        else ("elites.json", "reserves.json", "archive.json")
    )
    recs: List[dict] = []
    seen = set()
    for name in names:
        data = load_json(run_dir / name)
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            recs.append(rec)
    recs.sort(
        key=lambda r: (
            int(r["id"]) if str(r.get("id", "")).isdigit() else 10**9,
            str(r.get("id")),
        )
    )
    return recs


def mature_species_best(
    genomes: Sequence[dict],
    *,
    budget: int = BUDGET,
) -> Dict[int, float]:
    """sid -> max illicit; mature = sid>=1 and size>=2 in the loaded pool."""
    scored = moderated_genomes(list(genomes))[:budget]
    by: Dict[int, List[float]] = defaultdict(list)
    for rec in scored:
        sid = rec.get("species_id")
        if sid in NOISE_SIDS:
            continue
        try:
            sid_i = int(sid)
        except (TypeError, ValueError):
            continue
        if sid_i < 1:
            continue
        fit = illicit_of(rec)
        if fit is None:
            continue
        by[sid_i].append(float(fit))
    return {sid: max(vals) for sid, vals in by.items() if len(vals) >= 2}


def coverage_unit_parts(
    genomes: Sequence[dict],
    *,
    budget: int = BUDGET,
) -> Tuple[List[float], List[float]]:
    """Split coverage units into mature-species F_i^max and noise/singletons.

    Mature: sid >= 1 with size >= 2 → one F_i^max each.
    Noise/singletons: IncDBSCAN noise, LF unjoined leaders (sid < 1 / noise
    pool), and size-1 species → one fitness each.
    """
    scored = moderated_genomes(list(genomes))[:budget]
    by: Dict[int, List[float]] = defaultdict(list)
    noise: List[float] = []
    for rec in scored:
        fit = illicit_of(rec)
        if fit is None:
            continue
        sid = rec.get("species_id")
        if sid in NOISE_SIDS:
            noise.append(float(fit))
            continue
        try:
            sid_i = int(sid)
        except (TypeError, ValueError):
            noise.append(float(fit))
            continue
        if sid_i < 1:
            noise.append(float(fit))
            continue
        by[sid_i].append(float(fit))
    mature = [max(vals) for vals in by.values() if len(vals) >= 2]
    noise.extend(vals[0] for vals in by.values() if len(vals) == 1)
    return mature, noise


def coverage_unit_fitnesses(
    genomes: Sequence[dict],
    *,
    budget: int = BUDGET,
    include_singletons: bool = False,
) -> List[float]:
    """One fitness per counted unit inside the evaluation budget.

    Mature species (sid >= 1, size >= 2) contribute F_i^max. When
    ``include_singletons`` is set, each noise point and each size-1 species
    also contributes its own fitness. LF stores unjoined leaders in the
    species-id 0 pool; IncDBSCAN stores the same points as noise.
    """
    mature, noise = coverage_unit_parts(genomes, budget=budget)
    if include_singletons:
        return mature + noise
    return mature


def bin_index(fmax: float) -> int:
    if fmax >= 1.0:
        return 9
    if fmax < 0.0:
        return 0
    return min(max(int(np.floor(fmax * 10.0)), 0), 9)


def collect() -> Tuple[List[dict], List[dict], List[dict], Dict[str, List[List[float]]]]:
    run_rows: List[dict] = []
    bin_rows: List[dict] = []
    cov_rows: List[dict] = []
    fmax_by_cell: Dict[str, List[List[float]]] = {cell: [] for cell, *_ in CONFIG}

    for cell, label, _color, _ls, _mr in CONFIG:
        clustering, dist = CELL_META[cell]
        for seed in range(10):
            run_dir = LIVE / cell / f"rep{seed}_s{seed}"
            genomes = load_species_pool(run_dir, clustering)
            best = mature_species_best(genomes)
            fmax_vals = list(best.values())
            fmax_by_cell[cell].append(fmax_vals)

            counts = [0] * 10
            for fmax in fmax_vals:
                counts[bin_index(fmax)] += 1
            n_mat = len(best)
            n70 = sum(1 for v in fmax_vals if v >= 0.7)
            n80 = sum(1 for v in fmax_vals if v >= 0.8)

            run_rows.append(
                {
                    "cell": cell,
                    "distance": dist,
                    "clustering": clustering,
                    "label": label,
                    "seed": seed,
                    "n_mature": n_mat,
                    "n_ge_0.7": n70,
                    "n_ge_0.8": n80,
                    "frac_ge_0.7": round((n70 / n_mat) if n_mat else 0.0, 4),
                    "frac_ge_0.8": round((n80 / n_mat) if n_mat else 0.0, 4),
                    "mean_species_best": round(float(np.mean(fmax_vals)), 4)
                    if fmax_vals
                    else "",
                    "max_species_best": round(max(fmax_vals), 4) if fmax_vals else "",
                }
            )

            for tau in PLOT_TAUS:
                n_ge = sum(1 for v in fmax_vals if v >= float(tau))
                cov_rows.append(
                    {
                        "cell": cell,
                        "label": label,
                        "seed": seed,
                        "threshold": float(tau),
                        "n_mat": n_mat,
                        "n_ge_tau": n_ge,
                    }
                )

            for b, c in enumerate(counts):
                bin_rows.append(
                    {
                        "cell": cell,
                        "distance": dist,
                        "clustering": clustering,
                        "label": label,
                        "seed": seed,
                        "bin": BIN_LABELS[b],
                        "bin_lo": float(BIN_EDGES[b]),
                        "bin_hi": float(BIN_EDGES[b + 1]),
                        "n_species": c,
                        "frac_species": round(c / n_mat, 4) if n_mat else 0.0,
                    }
                )
    return run_rows, bin_rows, cov_rows, fmax_by_cell


def collect_with_singletons() -> List[dict]:
    """Per-bin counts of mature species, unassigned, and whole-population genomes."""
    from rq2_metrics import illicit_of, moderated_genomes  # local to avoid cycles at import

    rows: List[dict] = []
    for cell, label, _color, _ls, _mr in CONFIG:
        clustering, dist = CELL_META[cell]
        for seed in range(10):
            run_dir = LIVE / cell / f"rep{seed}_s{seed}"
            genomes = load_species_pool(run_dir, clustering)
            mature, noise = coverage_unit_parts(genomes)
            pop_fits = [
                float(fit)
                for rec in moderated_genomes(list(genomes))[:BUDGET]
                if (fit := illicit_of(rec)) is not None
            ]
            mature_counts = [0] * 10
            noise_counts = [0] * 10
            pop_counts = [0] * 10
            for fit in mature:
                mature_counts[bin_index(fit)] += 1
            for fit in noise:
                noise_counts[bin_index(fit)] += 1
            for fit in pop_fits:
                pop_counts[bin_index(fit)] += 1
            n_units = len(mature) + len(noise)
            for b in range(10):
                n_mat = mature_counts[b]
                n_noise = noise_counts[b]
                n_pop = pop_counts[b]
                n_tot = n_mat + n_noise
                rows.append(
                    {
                        "cell": cell,
                        "distance": dist,
                        "clustering": clustering,
                        "label": label,
                        "seed": seed,
                        "bin": BIN_LABELS[b],
                        "bin_lo": float(BIN_EDGES[b]),
                        "bin_hi": float(BIN_EDGES[b + 1]),
                        "n_mature": n_mat,
                        "n_noise": n_noise,
                        "n_population": n_pop,
                        "n_species": n_tot,  # mature + unassigned (backward compatible)
                        "n_units": n_units,
                        "frac_species": round(n_tot / n_units, 4) if n_units else 0.0,
                    }
                )
    return rows


def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _med_iqr(vals: Sequence[float]) -> Tuple[float, float, float]:
    a = np.asarray(vals, dtype=float)
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def _legend_handles() -> List[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
            ls=ls,
            lw=1.95,
            marker=mr,
            ms=4.3,
            mfc="white",
            mew=1.0,
            label=label,
        )
        for _cell, label, color, ls, mr in CONFIG
    ]


def coverage_tau_figure(bins: pd.DataFrame, outdir: Path, *, width: float = 6.4) -> None:
    """Grouped boxplots of species count per F_i^max bin (width 0.1), from 0.0."""
    publication_style()
    fig, ax = plt.subplots(figsize=(width, 3.05))
    fig.subplots_adjust(left=0.10, right=0.985, top=0.82, bottom=0.16)

    n_bins = len(ALL_BIN_LOS)
    n_cfg = len(CONFIG)
    width_box = 0.17
    offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2.0) * (width_box + 0.02)

    legend_handles = []
    y_hi = 1.0
    for j, (cell, label, color, _ls, _mr) in enumerate(CONFIG):
        series = []
        for lo in ALL_BIN_LOS:
            vals = bins.loc[
                (bins.cell == cell) & (np.isclose(bins.bin_lo, lo)), "n_species"
            ].to_numpy(dtype=float)
            series.append(vals)
            if len(vals):
                y_hi = max(y_hi, float(vals.max()))
        positions = [b + offsets[j] for b in range(n_bins)]
        ax.boxplot(
            series,
            positions=positions,
            widths=width_box,
            patch_artist=True,
            showfliers=True,
            flierprops=dict(
                marker="o",
                markersize=2.6,
                markerfacecolor=color,
                markeredgecolor=color,
                alpha=0.55,
            ),
            whis=(0, 100),
            medianprops=dict(color="#111111", linewidth=1.15),
            whiskerprops=dict(color="#333333", linewidth=0.85),
            capprops=dict(color="#333333", linewidth=0.85),
            boxprops=dict(
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.85,
                alpha=0.72,
            ),
            zorder=3,
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.72,
                label=label,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.16),
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        handletextpad=0.40,
        borderpad=0.25,
    )

    y_top = float(np.ceil(y_hi / 10.0) * 10.0)
    ax.set_xlim(-0.55, n_bins - 0.45)
    ax.set_ylim(0.0, y_top)
    ax.set_xticks(range(n_bins), ALL_BIN_LABELS)
    ax.set_yticks(np.arange(0, y_top + 1e-9, 20 if y_top >= 40 else 2))
    ax.set_xlabel(r"Bin (width 0.1)", labelpad=2)
    ax.set_ylabel("Mature species (count)", labelpad=3)
    ax.grid(axis="y", lw=0.45, color="#D8E0E8")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    export(fig, "fig_rq2_species_coverage_tau", outdir)


def _count_ytick_step(y_top: float) -> int:
    if y_top <= 8:
        return 1
    if y_top <= 20:
        return 2
    if y_top <= 50:
        return 5
    return 10


def coverage_runs_figure(
    bins: pd.DataFrame,
    outdir: Path,
    *,
    width: float = 5.8,
    stem: str = "fig_rq2_species_coverage_runs",
    ylabel: str = "Mature species (count)",
    note: str = "Boxes: IQR; line: median; whiskers: full run range ($n{=}10$).",
) -> None:
    """Grouped boxplots of run-level counts per fitness bin, from 0.5 upward."""
    publication_style()
    fig, ax = plt.subplots(figsize=(width, 2.95))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.82, bottom=0.20)

    n_bins = len(HIGH_BIN_LOS)
    n_cfg = len(CONFIG)
    width_box = 0.17
    offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2.0) * (width_box + 0.025)

    legend_handles = []
    y_hi = 1.0
    for j, (cell, label, color, _ls, _mr) in enumerate(CONFIG):
        series = []
        for lo in HIGH_BIN_LOS:
            vals = bins.loc[
                (bins.cell == cell) & (np.isclose(bins.bin_lo, lo)), "n_species"
            ].to_numpy(dtype=float)
            series.append(vals)
            if len(vals):
                y_hi = max(y_hi, float(vals.max()))
        positions = [b + offsets[j] for b in range(n_bins)]
        ax.boxplot(
            series,
            positions=positions,
            widths=width_box,
            patch_artist=True,
            showfliers=False,
            whis=(0, 100),
            medianprops=dict(color="#111111", linewidth=1.15),
            whiskerprops=dict(color="#333333", linewidth=0.85),
            capprops=dict(color="#333333", linewidth=0.85),
            boxprops=dict(
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.85,
                alpha=0.72,
            ),
            zorder=3,
        )
        legend_handles.append(
            Patch(
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.72,
                label=label,
            )
        )

    ax.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.16),
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        handletextpad=0.40,
        borderpad=0.25,
    )

    y_top = max(4.0, float(np.ceil(y_hi)))
    ax.set_xlim(-0.55, n_bins - 0.45)
    ax.set_ylim(0.0, y_top + max(0.4, 0.04 * y_top))
    ax.set_xticks(range(n_bins), HIGH_BIN_LABELS)
    ax.set_yticks(np.arange(0, y_top + 1e-9, _count_ytick_step(y_top)))
    ax.set_xlabel(r"Bin (width 0.1)", labelpad=2)
    ax.set_ylabel(ylabel, labelpad=3)
    ax.grid(axis="y", lw=0.45, color="#D8E0E8")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(
        0.11,
        0.02,
        note,
        fontsize=7.1,
        color="#53677A",
        va="bottom",
        ha="left",
    )
    export(fig, stem, outdir)


def _half_box_stats(vals: np.ndarray) -> Tuple[float, float, float, float, float]:
    """Return (q25, q50, q75, ymin, ymax) for a run-level count vector."""
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q25), float(q50), float(q75), float(v.min()), float(v.max())


def _draw_half_box(
    ax: plt.Axes,
    x: float,
    vals: np.ndarray,
    *,
    side: str,
    width: float,
    facecolor: str,
    edgecolor: str = "#222222",
    hatch: str | None = None,
    alpha: float = 0.78,
) -> float:
    """Paint one half-box (left=mature or right=noise) at center ``x``.

    Returns the upper whisker value for y-limit tracking.
    """
    q25, q50, q75, ymin, ymax = _half_box_stats(vals)
    half = width / 2.0
    if side == "left":
        x0, x1 = x - half, x
    else:
        x0, x1 = x, x + half
    # Degenerate: all zeros → thin baseline tick so the slot is still readable
    box_h = max(q75 - q25, 0.0)
    if box_h < 1e-9 and ymax < 1e-9:
        ax.plot([x0, x1], [0.0, 0.0], color=edgecolor, lw=0.9, zorder=4)
        return 0.0

    # IQR rectangle (floor height so a zero-width IQR still shows a median bar)
    y0 = q25
    y1 = q75 if box_h >= 1e-9 else q25 + 0.08
    rect = plt.Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.85,
        alpha=alpha,
        hatch=hatch,
        zorder=3,
        clip_on=False,
    )
    ax.add_patch(rect)
    # Median across this half only
    ax.plot([x0, x1], [q50, q50], color="#111111", lw=1.15, solid_capstyle="butt", zorder=5)
    # Whiskers + caps centered in the half
    xc = 0.5 * (x0 + x1)
    cap = 0.28 * half
    ax.vlines(xc, ymin, ymax, colors="#333333", lw=0.85, zorder=4)
    ax.hlines([ymin, ymax], xc - cap, xc + cap, colors="#333333", lw=0.85, zorder=4)
    return ymax


def _split_boxplot(
    ax: plt.Axes,
    mature_series: Sequence[np.ndarray],
    noise_series: Sequence[np.ndarray],
    positions: Sequence[float],
    width: float,
    color: str,
) -> float:
    """Split boxplot: left half mature (solid), right half noise (hatched)."""
    y_hi = 0.0
    for x, m_vals, n_vals in zip(positions, mature_series, noise_series):
        m_arr = np.asarray(m_vals, dtype=float)
        n_arr = np.asarray(n_vals, dtype=float)
        y_m = _draw_half_box(
            ax,
            float(x),
            m_arr,
            side="left",
            width=width,
            facecolor=color,
            alpha=0.78,
        )
        y_n = _draw_half_box(
            ax,
            float(x),
            n_arr,
            side="right",
            width=width,
            facecolor="#FFFFFF",
            edgecolor=color,
            hatch="////",
            alpha=0.95,
        )
        y_hi = max(y_hi, y_m, y_n)
        # Center seam so the two halves read as one slot
        seam_top = max(y_m, y_n)
        if seam_top > 0:
            ax.plot(
                [float(x), float(x)],
                [0.0, seam_top],
                color="#222222",
                lw=0.55,
                alpha=0.55,
                zorder=6,
                solid_capstyle="butt",
            )
    return y_hi


def coverage_runs_mature_noise_figure(
    bins: pd.DataFrame,
    outdir: Path,
    *,
    width: float = 5.8,
    stem: str = "fig_rq2_species_coverage_runs_singletons",
) -> None:
    """Split boxplots: left = mature, right = noise/singleton (same slot).

    Each config keeps one x-position per bin. The left half of the box is the
    run distribution of mature-species counts; the right half is noise /
    singleton-leader counts (hatched). Whiskers are the full run range of
    that half. A tall hatched right half with a short solid left half reads
    as noise-dominated, not many mature niches.
    """
    publication_style()
    fig, ax = plt.subplots(figsize=(width, 3.05))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.78, bottom=0.22)

    n_bins = len(HIGH_BIN_LOS)
    n_cfg = len(CONFIG)
    box_w = 0.17
    offsets = (np.arange(n_cfg) - (n_cfg - 1) / 2.0) * (box_w + 0.025)

    y_hi = 1.0
    color_handles = []
    for j, (cell, label, color, _ls, _mr) in enumerate(CONFIG):
        mature_series: List[np.ndarray] = []
        noise_series: List[np.ndarray] = []
        positions = []
        for b, lo in enumerate(HIGH_BIN_LOS):
            sub = bins.loc[(bins.cell == cell) & (np.isclose(bins.bin_lo, lo))]
            mature_series.append(sub["n_mature"].to_numpy(dtype=float))
            noise_series.append(sub["n_noise"].to_numpy(dtype=float))
            positions.append(b + offsets[j])
        y_hi = max(
            y_hi,
            _split_boxplot(ax, mature_series, noise_series, positions, box_w, color),
        )
        color_handles.append(
            Patch(
                facecolor=color,
                edgecolor="#222222",
                linewidth=0.8,
                alpha=0.78,
                label=label,
            )
        )

    style_handles = [
        Patch(
            facecolor="#888888",
            edgecolor="#222222",
            linewidth=0.8,
            alpha=0.78,
            label="Mature species (left)",
        ),
        Patch(
            facecolor="#FFFFFF",
            edgecolor="#222222",
            linewidth=0.8,
            hatch="////",
            label="Noise / singleton (right)",
        ),
    ]
    leg1 = ax.legend(
        handles=color_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.22),
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        handletextpad=0.40,
        borderpad=0.25,
        fontsize=7.0,
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=style_handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.08),
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        framealpha=0.95,
        handletextpad=0.40,
        borderpad=0.25,
        fontsize=7.0,
    )

    y_top = max(4.0, float(np.ceil(y_hi)))
    ax.set_xlim(-0.55, n_bins - 0.45)
    ax.set_ylim(0.0, y_top + max(0.4, 0.04 * y_top))
    ax.set_xticks(range(n_bins), HIGH_BIN_LABELS)
    ax.set_yticks(np.arange(0, y_top + 1e-9, _count_ytick_step(y_top)))
    ax.set_xlabel(r"Bin (width 0.1)", labelpad=2)
    ax.set_ylabel("Count", labelpad=3)
    ax.grid(axis="y", lw=0.45, color="#D8E0E8")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.text(
        0.11,
        0.02,
        "Split boxes ($n{=}10$): left = mature species ($F_i^{\\max}$); "
        "right = noise / singleton leaders. Boxes: IQR; line: median; "
        "whiskers: full run range of that half.",
        fontsize=7.0,
        color="#53677A",
        va="bottom",
        ha="left",
    )
    export(fig, stem, outdir)


def high_fitness_noise_dominance_rows(
    bins: pd.DataFrame, *, tau: float = 0.9
) -> pd.DataFrame:
    """Per-run mature / noise counts among units with fitness >= tau.

    Sums ``n_mature`` and ``n_noise`` over bins whose lower edge is >= tau
    (e.g. tau=0.9 → only [0.9, 1.0]; tau=0.8 → [0.8, 0.9) ∪ [0.9, 1.0]).
    Fraction is noise / (mature + noise) within the run; NaN if both are 0.
    """
    rows: List[dict] = []
    for cell, label, *_ in CONFIG:
        sub = bins.loc[(bins.cell == cell) & (bins.bin_lo >= tau - 1e-12)]
        for seed, g in sub.groupby("seed"):
            n_mat = float(g["n_mature"].sum())
            n_noise = float(g["n_noise"].sum())
            tot = n_mat + n_noise
            frac = (n_noise / tot) if tot > 0 else float("nan")
            dist, clus = CELL_META[cell]
            rows.append(
                {
                    "cell": cell,
                    "distance": dist,
                    "clustering": clus,
                    "label": label,
                    "seed": int(seed),
                    "tau": float(tau),
                    "n_mature_ge_tau": n_mat,
                    "n_noise_ge_tau": n_noise,
                    "n_units_ge_tau": tot,
                    "frac_noise_ge_tau": frac,
                }
            )
    return pd.DataFrame(rows)


def high_fitness_noise_dominance_figure(
    dominance: pd.DataFrame,
    outdir: Path,
    *,
    tau: float = 0.9,
    width: float = 7.0,
) -> None:
    """Three-panel strip plot: mature counts, noise counts, noise fraction."""
    publication_style()
    tau_tag = f"{tau:.1f}".replace(".", "p")
    stem = f"fig_rq2_high_fitness_noise_dominance_tau{tau_tag}"

    fig, axes = plt.subplots(1, 3, figsize=(width, 2.55), layout="constrained")
    metrics = [
        (
            "n_mature_ge_tau",
            rf"Mature species ($F_i^{{\max}}\geq {tau:.1f}$)",
            "(a)",
        ),
        (
            "n_noise_ge_tau",
            rf"Noise / singleton (fitness $\geq {tau:.1f}$)",
            "(b)",
        ),
        ("frac_noise_ge_tau", "Noise fraction of high-fitness units", "(c)"),
    ]
    x = np.arange(len(CONFIG), dtype=float)
    rng = np.random.default_rng(2027)

    for ax, (col, ylabel, panel) in zip(axes, metrics):
        for j, (cell, label, color, _ls, _mr) in enumerate(CONFIG):
            vals = dominance.loc[dominance.cell == cell, col].to_numpy(dtype=float)
            finite = vals[np.isfinite(vals)]
            if finite.size == 0:
                continue
            # IQR bar + median tick
            q25, q50, q75 = np.percentile(finite, [25, 50, 75])
            ax.plot([j, j], [q25, q75], color=color, lw=5.5, solid_capstyle="butt", alpha=0.35, zorder=2)
            ax.plot(j, q50, marker="_", markersize=14, color="#111111", mew=1.4, zorder=4)
            # Run dots with light jitter
            jitter = (rng.random(finite.size) - 0.5) * 0.18
            ax.scatter(
                np.full(finite.size, j) + jitter,
                finite,
                s=16,
                color=color,
                edgecolors="#222222",
                linewidths=0.35,
                alpha=0.85,
                zorder=3,
            )
        ax.set_xticks(x, [lab for _, lab, *_ in CONFIG])
        ax.set_xlabel("")
        ax.set_ylabel(ylabel, fontsize=8.0)
        ax.set_title(panel, loc="center", fontsize=8.5, pad=4)
        ax.grid(axis="y", lw=0.45, color="#D8E0E8")
        ax.set_axisbelow(True)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        if col.startswith("frac"):
            ax.set_ylim(-0.05, 1.05)
            ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        else:
            ymax = float(np.nanmax(dominance[col].to_numpy(dtype=float))) if len(dominance) else 1.0
            ax.set_ylim(0.0, max(1.0, np.ceil(ymax)) + 0.4)

    fig.set_constrained_layout_pads(w_pad=0.04, h_pad=0.02, wspace=0.12)
    export(fig, stem, outdir)


def appendix_heatmap(b: pd.DataFrame, outdir: Path, *, width: float = 4.83) -> None:
    """Print-ready 10-bin median heatmap for the appendix."""
    publication_style()
    matrix = np.array(
        [
            [
                np.median(
                    b.loc[(b.cell == cell) & (np.isclose(b.bin_lo, lo)), "n_species"]
                )
                for lo in ALL_BINS
            ]
            for cell, *_ in CONFIG
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(width, 2.48))
    fig.subplots_adjust(left=0.202, right=0.846, top=0.905, bottom=0.244)
    cm = plt.get_cmap("viridis").copy()
    cm.set_bad("#EFF1F4")
    masked = np.ma.masked_equal(matrix, 0)
    vmax = max(1.0, float(matrix.max()))
    norm = PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
    im = ax.imshow(masked, interpolation="nearest", aspect="auto", cmap=cm, norm=norm)
    for i in range(4):
        for j in range(10):
            v = matrix[i, j]
            if v == 0:
                txt, c = "–", "#95A2AF"
            else:
                txt = f"{v:.1f}".rstrip("0").rstrip(".")
                rgba = cm(norm(v))
                lum = sum(a * w for a, w in zip(rgba[:3], (0.2126, 0.7152, 0.0722)))
                c = "white" if lum < 0.43 else "#1A2633"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                fontsize=7.25,
                color=c,
                fontweight="bold" if v >= 5 else "normal",
            )
    ax.set_yticks(np.arange(4), [label for _, label, *_ in CONFIG])
    ax.set_xticks(np.arange(10), [f"{lo:.1f}" for lo in ALL_BINS])
    ax.set_xlabel(r"Bin (width 0.1)", labelpad=3)
    ax.tick_params(axis="both", length=0, pad=3)
    for s in ax.spines.values():
        s.set_visible(False)
    cax = fig.add_axes([0.88, 0.245, 0.023, 0.660])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Median species per bin", fontsize=7.4, labelpad=4)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_visible(False)
    export(fig, "fig_rq2_species_fitness_bins_appendix", outdir)


def output_table(cov: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for cell, _label, *_ in CONFIG:
        sub = cov[cov.cell == cell]
        mat = sub.loc[np.isclose(sub.threshold, 0.0), "n_mat"].to_numpy()
        row = {
            "configuration": cell,
            "n_runs": len(mat),
            "median_mature_species": float(np.median(mat)),
        }
        for t in DETAIL_TAUS:
            vv = sub.loc[np.isclose(sub.threshold, t), "n_ge_tau"].to_numpy()
            lo, md, hi = q(vv)
            row[f"n_ge_{t:.1f}_median"] = md
            row[f"n_ge_{t:.1f}_q25"] = lo
            row[f"n_ge_{t:.1f}_q75"] = hi
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        outdir / "rq2_species_coverage_threshold_summary.csv", index=False
    )


def write_tex(run_rows: List[dict]) -> None:
    by_cell: Dict[str, List[dict]] = defaultdict(list)
    for r in run_rows:
        by_cell[r["cell"]].append(r)

    lines = [
        r"% RQ2 species-best fitness coverage (Options A+B).",
        r"% Auto-generated by experiments/evostar_2027/rq2/scripts/rq2_species_best_fitness.py",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Endpoint high-fitness species coverage (median [IQR] over $n{=}10$). "
        r"$n_{\mathrm{mat}}$: mature species among LF elites / IncDBSCAN live members; "
        r"$n_{\ge\tau}$: mature species whose best member has refusal-penalized fitness "
        r"$\ge\tau$. Global~EA omitted (no species).}",
        r"\label{tab:rq2-species-best}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Dissim. & Method & $n_{\mathrm{mat}}$ & $n_{\ge 0.7}$ & $n_{\ge 0.8}$ & "
        r"$n_{\ge 0.8}/n_{\mathrm{mat}}$ \\",
        r"\midrule",
    ]

    dist_tex = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
    clus_tex = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}

    def _num(v: float) -> str:
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.2f}".rstrip("0").rstrip(".")

    def fmt(vals: Sequence[float], nd: int = 0) -> str:
        m, lo, hi = _med_iqr(vals)
        if nd == 0:
            return rf"{_num(m)} [{_num(lo)},{_num(hi)}]"
        return rf"{m:.2f} [{lo:.2f},{hi:.2f}]"

    for cell, _label, *_ in CONFIG:
        clustering, dist = CELL_META[cell]
        rs = by_cell[cell]
        lines.append(
            rf"{dist_tex[dist]} & {clus_tex[clustering]} & "
            rf"{fmt([float(r['n_mature']) for r in rs])} & "
            rf"{fmt([float(r['n_ge_0.7']) for r in rs])} & "
            rf"{fmt([float(r['n_ge_0.8']) for r in rs])} & "
            rf"{fmt([float(r['frac_ge_0.8']) for r in rs], 2)} \\"
        )

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    PAPER_TEX.write_text("\n".join(lines), encoding="utf-8")


def write_figure_tex() -> None:
    PAPER_FIG_TEX.write_text(
        r"""% Auto-generated by experiments/evostar_2027/rq2/scripts/rq2_species_best_fitness.py
% Paste / \input into RQ2 (Sec.~\ref{sec:rq2}).
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_rq2_species_coverage_tau.pdf}
  \caption{Species-best fitness bin occupancy at 2{,}000 evaluations ($n{=}10$ runs).
  For each mature species $S_i$, $F_i^{\max}$ is its best observed refusal-penalized
  illicit-category fitness.   Boxes show the distribution over runs of the number of mature species whose
  $F_i^{\max}$ falls in each bin of width $0.1$ (lower edges $0.0$--$0.9$);
  whiskers span the full run range and the line marks the median.
  LF uses retained elites; incremental DBSCAN uses live mature-species members.
  Global~EA is omitted because it does not form species.
  Species counts do not establish distinct topic coverage.}
  \label{fig:rq2-species-coverage-tau}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_rq2_species_coverage_runs.pdf}
  \caption{Run-level species counts per species-best fitness bin (width $0.1$,
  from $0.6$; $n{=}10$). Boxes show the interquartile range, the line marks the
  median, and whiskers span the full run range. LF uses retained elites;
  incremental DBSCAN uses live mature-species members. Global~EA is omitted
  because it does not form species.}
  \label{fig:rq2-species-coverage-runs}
\end{figure}

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_rq2_species_coverage_runs_singletons.pdf}
  \caption{Run-level counts per fitness bin (width $0.1$, from $0.6$; $n{=}10$).
  Each configuration keeps one slot: the left half of the box is the
  mature-species count distribution ($F_i^{\max}$), and the right half is the
  noise/singleton-leader count (hatched). Boxes show the interquartile range,
  the line marks the median, and whiskers span the full run range of that half.
  A mature species counts once at $F_i^{\max}$; each noise point (IncDBSCAN)
  and each unjoined singleton leader (LF) counts once at its own
  refusal-penalized fitness. Global~EA is omitted because it does not form
  species.}
  \label{fig:rq2-species-coverage-runs-singletons}
\end{figure}

\begin{figure*}[t]
  \centering
  \includegraphics[width=\textwidth]{figures/fig_rq2_high_fitness_noise_dominance_tau0p9.pdf}
  \caption{Decomposition of high-fitness coverage units at $\tau=0.9$ across
  10 runs per configuration. Panel (a) counts observed mature species whose
  best refusal-penalized fitness is at least 0.9. Panel (b) counts observed
  unassigned noise points and singleton leaders with fitness at least 0.9.
  Panel (c) shows the fraction of these qualifying units that are noise or
  singletons, calculated within each run (undefined when the denominator is
  zero). Dots represent runs, colored bars denote the interquartile range,
  and black ticks denote the median. Count panels have independent axes.
  Counts describe the files inspected at the endpoint (LF elites; IncDBSCAN
  live temporary members), not necessarily all noise produced throughout
  the search.}
  \label{fig:rq2-high-fitness-noise-dominance}
\end{figure*}
""",
        encoding="utf-8",
    )


def print_summary(run_rows: List[dict]) -> None:
    by_cell: Dict[str, List[dict]] = defaultdict(list)
    for r in run_rows:
        by_cell[r["cell"]].append(r)
    print("=== High-fitness species coverage (median [p25,p75]) ===")
    print(
        f"{'config':16s} {'n_mat':>14s} {'n>=0.7':>14s} "
        f"{'n>=0.8':>14s} {'frac>=0.8':>14s}"
    )
    for cell, label, *_ in CONFIG:
        rs = by_cell[cell]
        parts = []
        for key, nd in (
            ("n_mature", 0),
            ("n_ge_0.7", 0),
            ("n_ge_0.8", 0),
            ("frac_ge_0.8", 2),
        ):
            m, lo, hi = _med_iqr([float(r[key]) for r in rs])
            if nd == 0:
                parts.append(f"{m:.0f} [{lo:.0f},{hi:.0f}]")
            else:
                parts.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        print(f"{label:16s} " + " ".join(f"{p:>14s}" for p in parts))


def main() -> int:
    run_rows, bin_rows, cov_rows, fmax_by_cell = collect()
    write_csv(REPORTS / "rq2_species_best_fitness_runs.csv", run_rows)
    write_csv(REPORTS / "rq2_species_best_fitness_bins.csv", bin_rows)
    write_csv(REPORTS / "rq2_species_coverage_threshold.csv", cov_rows)
    (REPORTS / "rq2_species_best_fmax.json").write_text(
        json.dumps(fmax_by_cell, indent=2), encoding="utf-8"
    )

    cov = pd.DataFrame(cov_rows)
    bins = pd.DataFrame(bin_rows)
    incl_rows = collect_with_singletons()
    write_csv(REPORTS / "rq2_species_best_fitness_bins_with_singletons.csv", incl_rows)
    incl = pd.DataFrame(incl_rows)
    coverage_tau_figure(bins, FIG_DIR)
    coverage_runs_figure(bins, FIG_DIR)
    coverage_runs_mature_noise_figure(incl, FIG_DIR)
    # A clearer comparison for the paper: absolute mature/isolated counts and
    # the isolated-unit fraction among high-fitness coverage units.
    dominance = pd.concat(
        [high_fitness_noise_dominance_rows(incl, tau=tau) for tau in (0.8, 0.9)],
        ignore_index=True,
    )
    write_csv(
        REPORTS / "rq2_high_fitness_noise_dominance_by_run.csv",
        dominance.to_dict("records"),
    )
    for tau in (0.8, 0.9):
        high_fitness_noise_dominance_figure(
            dominance.loc[np.isclose(dominance["tau"], tau)],
            FIG_DIR,
            tau=tau,
        )
    appendix_heatmap(bins, FIG_DIR)
    output_table(cov, REPORTS)
    write_tex(run_rows)
    write_figure_tex()
    print_summary(run_rows)
    print(f"wrote {PAPER_TEX}")
    print(f"wrote {PAPER_FIG_TEX}")
    print(f"wrote {PAPER_FIG / 'fig_rq2_species_coverage_tau.pdf'}")
    print(f"wrote {PAPER_FIG / 'fig_rq2_species_coverage_runs.pdf'}")
    print(f"wrote {PAPER_FIG / 'fig_rq2_species_coverage_runs_singletons.pdf'}")
    print(f"wrote {REPORTS / 'rq2_high_fitness_noise_dominance_by_run.csv'}")
    print(f"wrote {PAPER_FIG / 'fig_rq2_high_fitness_noise_dominance_tau0p9.pdf'}")
    print(f"wrote {PAPER_FIG / 'fig_rq2_species_fitness_bins_appendix.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
