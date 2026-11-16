#!/usr/bin/env python3
"""RQ2: single mirrored mature-vs-isolated box-and-whisker figure.

Reuses the same endpoint pools and bin definitions as
``rq2_species_best_fitness.py``. Mature-species counts are drawn below
zero (sign-mirrored for display only); unassigned counts are drawn above.
One figure covers the high-fitness bins from 0.6.
"""

from __future__ import annotations

import argparse
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
from matplotlib.patches import Patch, Rectangle
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_species_best_fitness as base  # noqa: E402

CONFIG = base.CONFIG
HIGH_BIN_LOS = base.HIGH_BIN_LOS
HIGH_BIN_LABELS = base.HIGH_BIN_LABELS
FIG_DIR = base.FIG_DIR
PAPER_FIG = base.PAPER_FIG
PAPER_TEX = base.PAPER_TEX
PAPER_FIG_TEX = base.PAPER_FIG_TEX
REPORTS = base.REPORTS

STEM = "fig_rq2_species_mature_noise_mirrored_boxplots"


def _box_stats(vals: Sequence[float]) -> Tuple[float, float, float, float, float]:
    v = np.asarray(vals, dtype=float)
    if v.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    q25, q50, q75 = np.percentile(v, [25, 50, 75])
    return float(q25), float(q50), float(q75), float(v.min()), float(v.max())


def _draw_v_box(
    ax: plt.Axes,
    x: float,
    vals: Sequence[float],
    *,
    width: float,
    facecolor: str,
    edgecolor: str = "#222222",
    hatch: str | None = None,
    alpha: float = 0.78,
    y_map=None,
) -> float:
    """Vertical box at column ``x`` (mirrored: mature below 0, noise above).

    ``y_map`` maps a nonnegative count to a display y. Returns raw count max.
    """
    q25, q50, q75, vmin, vmax = _box_stats(vals)
    assert y_map is not None
    y25, y50, y75, yvmin, yvmax = (float(y_map(v)) for v in (q25, q50, q75, vmin, vmax))
    y_lo, y_hi = sorted((y25, y75))
    w_lo, w_hi = sorted((yvmin, yvmax))

    if vmax < 1e-12 and q75 < 1e-12:
        ax.plot([x - width / 2, x + width / 2], [0.0, 0.0], color=edgecolor, lw=0.9, zorder=4)
        return 0.0

    box_h = max(y_hi - y_lo, 0.045)
    # Mature (below) grows downward from y_hi; noise grows upward from y_lo.
    y_origin = y_lo if w_hi >= 0 else y_hi - box_h
    ax.add_patch(
        Rectangle(
            (x - width / 2, y_origin),
            width,
            box_h,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
            alpha=alpha,
            hatch=hatch,
            zorder=3,
            clip_on=False,
        )
    )
    ax.plot([x - width / 2, x + width / 2], [y50, y50], color="#111111", lw=1.2, zorder=5)
    ax.vlines(x, w_lo, w_hi, colors="#333333", lw=0.85, zorder=4)
    cap = width * 0.28
    ax.hlines([w_lo, w_hi], x - cap, x + cap, colors="#333333", lw=0.85, zorder=4)
    return float(max(vmin, vmax, q50))


def synthetic_example() -> pd.DataFrame:
    """Small synthetic layout check (not experiment data)."""
    rows = []
    rng = np.random.default_rng(0)
    for cell, label, *_ in CONFIG:
        for lo, lab in zip(HIGH_BIN_LOS, HIGH_BIN_LABELS):
            for seed in range(10):
                base_m = 1 + (0 if "nli" in cell and "inc" in cell else 3)
                base_n = 8 if ("nli" in cell and lo >= 0.8) else 1
                rows.append(
                    {
                        "cell": cell,
                        "label": label,
                        "seed": seed,
                        "bin": lab,
                        "bin_lo": float(lo),
                        "n_mature": int(max(0, rng.poisson(base_m))),
                        "n_noise": int(max(0, rng.poisson(base_n))),
                    }
                )
    return pd.DataFrame(rows)


def mirrored_box_whisker_figure(
    bins: pd.DataFrame,
    outdir: Path,
    *,
    stem: str = STEM,
    axis_mode: str = "auto",
    demo: bool = False,
    width: float = 7.2,
) -> None:
    """Mirrored vertical boxplots in a horizontal (landscape) layout.

    Fitness bins run left→right; configs side-by-side within each bin.
    One shared count scale (step 5): mature $0$–$10$ below zero, unassigned
    $0$–$35$ above — every tick step has the same visual spacing.
    """
    del axis_mode  # dual-linear is the only encoding; kept for CLI compat
    base.publication_style()

    n_cfg = len(CONFIG)
    box_w = 0.30
    cfg_gap = 0.50
    bin_pad = 0.95

    fig, ax = plt.subplots(figsize=(width, 3.55))
    fig.subplots_adjust(left=0.09, right=0.995, top=0.86, bottom=0.24)

    # Shared count scale: equal visual spacing for every step of 5.
    mature_lim = 10.0
    unassigned_lim = 35.0
    tick_step = 5.0
    # Map counts onto display coords with one unit length per tick_step.
    # Top panel uses [0, 1] for 0…35; bottom uses proportional length for 0…10.
    top = 1.0
    scale = top / unassigned_lim  # display units per count

    def mature_y(count: float) -> float:
        return -float(count) * scale

    def unassigned_y(count: float) -> float:
        return float(count) * scale

    x_positions: List[float] = []
    bin_centers: List[Tuple[float, str]] = []
    x = 0.0

    for bi, (lo, blab) in enumerate(zip(HIGH_BIN_LOS, HIGH_BIN_LABELS)):
        xs_this = []
        for j, (cell, label, color, _ls, _mr) in enumerate(CONFIG):
            sub = bins.loc[(bins.cell == cell) & (np.isclose(bins.bin_lo, lo))]
            m = sub["n_mature"].to_numpy(dtype=float) if len(sub) else np.array([0.0])
            u = sub["n_noise"].to_numpy(dtype=float) if len(sub) else np.array([0.0])
            _draw_v_box(
                ax, x, m, width=box_w, facecolor=color, alpha=0.80, y_map=mature_y
            )
            _draw_v_box(
                ax,
                x,
                u,
                width=box_w,
                facecolor="#FFFFFF",
                edgecolor=color,
                hatch="////",
                alpha=0.95,
                y_map=unassigned_y,
            )
            x_positions.append(x)
            xs_this.append(x)
            if j < n_cfg - 1:
                x += cfg_gap
        bin_centers.append((float(np.mean(xs_this)), blab))
        if bi < len(HIGH_BIN_LOS) - 1:
            sep = x + bin_pad / 2
            ax.axvline(sep, color="#E2E8F0", lw=0.7, zorder=1)
            x += bin_pad

    ax.axhline(0.0, color="#444444", lw=0.9, zorder=2)
    ax.set_xticks([xc for xc, _ in bin_centers])
    ax.set_xticklabels([lab for _, lab in bin_centers], fontsize=7.5)
    # Exact shared-scale limits; small pad only above 35 for the legend.
    y_bot = mature_y(mature_lim)
    y_top = unassigned_y(unassigned_lim)
    ax.set_ylim(y_bot, y_top * 1.05)
    ax.set_xlim(x_positions[0] - 0.40, x_positions[-1] + 0.40)

    bot_ticks = list(np.arange(tick_step, mature_lim + 0.1, tick_step))  # 5,10
    top_ticks = list(np.arange(0.0, unassigned_lim + 0.1, tick_step))  # 0,5,…,35
    tick_pos = [mature_y(c) for c in reversed(bot_ticks)] + [
        unassigned_y(c) for c in top_ticks
    ]
    tick_lab = [f"{int(c)}" for c in reversed(bot_ticks)] + [
        ("0" if c < 1e-9 else f"{int(c)}") for c in top_ticks
    ]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)
    ax.tick_params(axis="y", pad=1.5)

    # Title midpoints in axes fraction of the visible y-range.
    ax.set_ylabel("")
    y0, y1 = ax.get_ylim()
    span = y1 - y0
    ax.text(
        -0.048,
        (unassigned_y(unassigned_lim / 2) - y0) / span,
        "Unassigned",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.5,
        color="#334155",
        clip_on=False,
    )
    ax.text(
        -0.048,
        (mature_y(mature_lim / 2) - y0) / span,
        "Mature",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=7.5,
        color="#334155",
        clip_on=False,
    )
    ax.set_xlabel("Fitness bin", labelpad=3)
    ax.grid(axis="y", lw=0.45, color="#D8E0E8")
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    color_handles = [
        Patch(facecolor=color, edgecolor="#222222", linewidth=0.8, alpha=0.80, label=label)
        for _c, label, color, *_ in CONFIG
    ]
    ax.legend(
        handles=color_handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.01),
        frameon=True,
        fancybox=False,
        edgecolor="#DDDDDD",
        fontsize=7.0,
        borderpad=0.30,
        handletextpad=0.35,
        columnspacing=0.85,
    )

    note_l1 = (
        "Four speciated configs × $n{=}10$ runs (40 of 50 RQ2 executions). "
        "Solid = mature $F_i^{\\max}$; hatched = unassigned (noise / size-1; "
        "excludes mature-species members)."
    )
    note_l2 = (
        "LF: elites+reserves+archive; Inc: live temp. "
        "Boxes: IQR; line: median; whiskers: full run range. "
        "Shared count scale (tick step 5): mature $0$–$10$, unassigned $0$–$35$."
    )
    if demo:
        note_l1 = "SYNTHETIC DEMO — " + note_l1
    fig.text(0.09, 0.055, note_l1, fontsize=6.0, color="#53677A", va="bottom", ha="left")
    fig.text(0.09, 0.018, note_l2, fontsize=6.0, color="#53677A", va="bottom", ha="left")

    outdir.mkdir(parents=True, exist_ok=True)
    for dest in (outdir, *((PAPER_FIG,) if not demo else ())):
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / f"{stem}.pdf")
        if dest is outdir:
            fig.savefig(dest / f"{stem}.png", dpi=200)
    plt.close(fig)


def write_figure_tex() -> None:
    """Single LaTeX figure reference for the manuscript."""
    PAPER_FIG_TEX.parent.mkdir(parents=True, exist_ok=True)
    PAPER_FIG_TEX.write_text(
        r"""% Auto-generated by rq2_species_best_fitness_mirrored_boxplots.py
% Use \input{rq2_species_coverage_figure.tex} in the manuscript.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_rq2_species_mature_noise_mirrored_boxplots.pdf}
  \caption{Mature-species and unassigned occupancy in four noncumulative
  fitness bins from $0.6$ to $1.0$. Counts come from the four speciated RQ2
  configurations ($n{=}10$ runs each; $2{,}000$ evaluations), i.e.\ $40$ of
  the $50$ RQ2 executions (Global~EA omitted because it does not form
  species). Mirrored vertical boxplots are arranged by fitness bin (left to
  right). Absolute counts of mature species appear below zero and unassigned
  observations (noise and size-$1$ singletons; members of mature species are
  excluded) above zero. Each mature species contributes once at its highest
  observed member fitness $F_i^{\max}$; each unassigned observation
  contributes once at its own fitness. Boxes show the interquartile range,
  the black line shows the median, and whiskers span the full observed run
  range. Below-zero positions are mirrored for visualization only: counts
  are nonnegative. Mature ($0$–$10$) and unassigned ($0$–$35$) share one
  linear count scale with equal tick spacing (step $5$). LF pools are
  \texttt{elites.json}, \texttt{reserves.json} when present, and
  \texttt{archive.json}; IncDBSCAN pools are live \texttt{temp.json} members
  (including noise).}
  \label{fig:rq2-mature-noise-mirrored}
\end{figure}
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
            m, lo, hi = base._med_iqr([float(r[key]) for r in rs])
            if nd == 0:
                parts.append(f"{m:.0f} [{lo:.0f},{hi:.0f}]")
            else:
                parts.append(f"{m:.2f} [{lo:.2f},{hi:.2f}]")
        print(f"{label:16s} " + " ".join(f"{p:>14s}" for p in parts))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RQ2: single mirrored mature-vs-isolated box-and-whisker figure"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Generate a synthetic layout example without experiment data",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path.cwd() / "rq2_demo",
        help="Output folder for --demo (default: ./rq2_demo)",
    )
    parser.add_argument(
        "--axis",
        choices=("auto", "linear", "symlog"),
        default="linear",
        help="Horizontal count scale (default: linear; use symlog only if needed)",
    )
    args = parser.parse_args()

    if args.demo:
        mirrored_box_whisker_figure(
            synthetic_example(),
            args.outdir,
            stem="example_mirrored_box_whiskers_SYNTHETIC",
            axis_mode=args.axis,
            demo=True,
        )
        print(f"Wrote SYNTHETIC preview to {args.outdir.resolve()}")
        return 0

    # Original experiment data extraction and metric definitions are unchanged.
    run_rows, bin_rows, cov_rows, fmax_by_cell = base.collect()
    base.write_csv(REPORTS / "rq2_species_best_fitness_runs.csv", run_rows)
    base.write_csv(REPORTS / "rq2_species_best_fitness_bins.csv", bin_rows)
    base.write_csv(REPORTS / "rq2_species_coverage_threshold.csv", cov_rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "rq2_species_best_fmax.json").write_text(
        json.dumps(fmax_by_cell, indent=2), encoding="utf-8"
    )
    cov = pd.DataFrame(cov_rows)
    incl_rows = base.collect_with_singletons()
    base.write_csv(REPORTS / "rq2_species_best_fitness_bins_with_singletons.csv", incl_rows)

    mirrored_box_whisker_figure(
        pd.DataFrame(incl_rows), FIG_DIR, axis_mode=args.axis
    )
    base.output_table(cov, REPORTS)
    PAPER_TEX.parent.mkdir(parents=True, exist_ok=True)
    base.write_tex(run_rows)
    write_figure_tex()
    print_summary(run_rows)
    print("WARNING: Reported noise/singleton counts reflect only loaded JSON pools.")
    print(
        "Verify IncDBSCAN true-noise storage and LF singleton memberships before inference."
    )
    print(f"Wrote figure: {FIG_DIR / f'{STEM}.pdf'}")
    print(f"Wrote paper copy: {PAPER_FIG / f'{STEM}.pdf'}")
    print(f"Wrote LaTeX figure definition: {PAPER_FIG_TEX}")
    print(f"Wrote summary table: {PAPER_TEX}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
