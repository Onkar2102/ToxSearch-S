#!/usr/bin/env python3
"""RQ2 combined figure: (a) quality, (b) diversity, (c) mature/unassigned.

One shared legend for all panels. Regenerates from anytime NPZs + bin CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import rq2_species_best_fitness as base  # noqa: E402
import rq2_species_best_fitness_mirrored_boxplots as mir  # noqa: E402

EXP = ROOT / "experiments" / "evostar_2027" / "rq2" / "figures"
PAPER = ROOT / "notes" / "rough_draft" / "figures"
TEX = ROOT / "notes" / "rough_draft" / "rq2_combined_qd_mature_figure.tex"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
STEM = "fig_rq2_qd_mature_combined"

FIT_NPZ = REPORTS / "rq2_anytime_top50_curves.npz"
MPNET_NPZ = REPORTS / "rq2_anytime_mpnet_curves.npz"

# cell key in NPZ → (label, color); all curves use solid lines (color = config).
SERIES: List[Tuple[str, str, str]] = [
    ("embedding__leader_follower", r"LF $d_E$", "#0F4D92"),
    ("embedding__dbscan", r"Inc $d_E$", "#2A9D8F"),
    ("nli_entail__leader_follower", r"LF $d_{\neg E}$", "#B64342"),
    ("nli_entail__dbscan", r"Inc $d_{\neg E}$", "#E9C46A"),
    ("none__none", "Global EA", "#4D4D4D"),
]

SHADE_ALPHA = 0.18
LABEL_FS = 8.8  # axis titles and panel titles (10% above 8 pt)
TICK_FS = 7.0


def _load_curves(path: Path) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    if "grid" in z.files:
        grid = np.asarray(z["grid"], dtype=float)
    else:
        grid = np.arange(50, 2001, 25, dtype=float)
    out: Dict[str, np.ndarray] = {}
    for key in z.files:
        if key == "grid":
            continue
        out[key] = np.asarray(z[key], dtype=float)
    return grid, out


def _median_iqr(runs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """runs: (n_runs, n_budget) → median, p25, p75 along runs."""
    med = np.nanmedian(runs, axis=0)
    lo = np.nanpercentile(runs, 25, axis=0)
    hi = np.nanpercentile(runs, 75, axis=0)
    return med, lo, hi


def _panel_caption(ax: plt.Axes, text: str, *, y: float = -0.20) -> None:
    """Place subfigure title centered just below the x-axis label."""
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_FS,
        color="#111111",
        clip_on=False,
    )


def _panel_caption_mm_below_xlabel(ax: plt.Axes, text: str, *, gap_mm: float = 1.0) -> None:
    """Place panel title with exactly ``gap_mm`` between xlabel bottom and title top."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xlabel = ax.xaxis.label
    bb = xlabel.get_window_extent(renderer=renderer)
    # 1 mm in display pixels
    gap_px = (gap_mm / 25.4) * fig.dpi
    y_disp = bb.y0 - gap_px
    x_disp = 0.5 * (bb.x0 + bb.x1)
    x_ax, y_ax = ax.transAxes.inverted().transform((x_disp, y_disp))
    ax.text(
        0.5,
        y_ax,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=LABEL_FS,
        color="#111111",
        clip_on=False,
    )


def _plot_anytime(
    ax: plt.Axes,
    grid: np.ndarray,
    curves: Dict[str, np.ndarray],
    ylabel: str,
    panel: str,
) -> None:
    for key, label, color in SERIES:
        if key not in curves:
            continue
        med, lo, hi = _median_iqr(curves[key])
        ax.plot(grid, med, color=color, ls="-", lw=1.6, label=label, zorder=3)
        ax.fill_between(grid, lo, hi, color=color, alpha=SHADE_ALPHA, lw=0, zorder=2)
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Unique evaluations", fontsize=LABEL_FS, labelpad=1)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    _panel_caption(ax, panel, y=-0.22)


def _draw_mirrored(ax: plt.Axes, bins: pd.DataFrame) -> float:
    """Panel (c): mature/unassigned mirrored boxes; no legend.

    Returns the data-coordinate box width used for mature/unassigned boxes.
    """
    n_cfg = len(base.CONFIG)
    box_w = 0.30
    cfg_gap = 0.50
    bin_pad = 0.95
    mature_lim = 10.0
    unassigned_lim = 35.0
    tick_step = 5.0
    scale = 1.0 / unassigned_lim

    def mature_y(count: float) -> float:
        return -float(count) * scale

    def unassigned_y(count: float) -> float:
        return float(count) * scale

    x_positions: List[float] = []
    bin_centers: List[Tuple[float, str]] = []
    x = 0.0
    for bi, (lo, blab) in enumerate(zip(base.HIGH_BIN_LOS, base.HIGH_BIN_LABELS)):
        xs_this = []
        for j, (cell, _label, color, *_rest) in enumerate(base.CONFIG):
            sub = bins.loc[(bins.cell == cell) & (np.isclose(bins.bin_lo, lo))]
            m = sub["n_mature"].to_numpy(dtype=float) if len(sub) else np.array([0.0])
            u = sub["n_noise"].to_numpy(dtype=float) if len(sub) else np.array([0.0])
            mir._draw_v_box(ax, x, m, width=box_w, facecolor=color, alpha=0.80, y_map=mature_y)
            mir._draw_v_box(
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
        if bi < len(base.HIGH_BIN_LOS) - 1:
            ax.axvline(x + bin_pad / 2, color="#E2E8F0", lw=0.7, zorder=1)
            x += bin_pad

    ax.axhline(0.0, color="#444444", lw=0.9, zorder=2)
    ax.set_xticks([xc for xc, _ in bin_centers])
    ax.set_xticklabels([lab for _, lab in bin_centers], fontsize=TICK_FS)
    y_bot, y_top = mature_y(mature_lim), unassigned_y(unassigned_lim)
    ax.set_ylim(y_bot, y_top)
    ax.set_xlim(x_positions[0] - 0.40, x_positions[-1] + 0.40)

    bot_ticks = list(np.arange(tick_step, mature_lim + 0.1, tick_step))
    top_ticks = list(np.arange(0.0, unassigned_lim + 0.1, tick_step))
    tick_pos = [mature_y(c) for c in reversed(bot_ticks)] + [
        unassigned_y(c) for c in top_ticks
    ]
    tick_lab = [f"{int(c)}" for c in reversed(bot_ticks)] + [
        ("0" if c < 1e-9 else f"{int(c)}") for c in top_ticks
    ]
    ax.set_yticks(tick_pos)
    ax.set_yticklabels(tick_lab)
    ax.tick_params(axis="y", pad=1.5, labelsize=TICK_FS)

    span = y_top - y_bot
    # +1 mm between y-axis titles and tick labels.
    fig = ax.figure
    fig.canvas.draw()
    ax_w_in = ax.get_window_extent().width / float(fig.dpi)
    title_x = -0.038 - (1.0 / 25.4) / ax_w_in
    ax.text(
        title_x,
        (unassigned_y(unassigned_lim / 2) - y_bot) / span,
        "Unassigned",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=LABEL_FS,
        color="#334155",
        clip_on=False,
    )
    ax.text(
        title_x,
        (mature_y(mature_lim / 2) - y_bot) / span,
        "Mature",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="center",
        fontsize=LABEL_FS,
        color="#334155",
        clip_on=False,
    )
    ax.set_xlabel("Fitness", fontsize=LABEL_FS, labelpad=1)
    ax.grid(False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # Exact 1 mm between x-axis title and panel-(c) title.
    _panel_caption_mm_below_xlabel(
        ax, "(c) Niche occupancy by fitness", gap_mm=1.0
    )
    return box_w


def _draw_aligned_legend(
    fig: plt.Figure,
    ax_c: plt.Axes,
    box_w_data: float,
    *,
    center_x: float,
    y_fig: float = 0.955,
) -> None:
    """One horizontal legend line; box width matches panel-(c) boxes exactly."""
    fig.canvas.draw()
    # Panel-(c) box width in figure fraction (same on-page width as data boxes).
    p0 = ax_c.transData.transform((0.0, 0.0))
    p1 = ax_c.transData.transform((box_w_data, 0.0))
    box_w_px = abs(float(p1[0] - p0[0]))
    # Swatch height ~ typical small mature IQR (3 counts) on the shared scale.
    q0 = ax_c.transData.transform((0.0, 0.0))
    q1 = ax_c.transData.transform((0.0, -3.0 / 35.0))
    box_h_px = abs(float(q1[1] - q0[1]))
    fig_w_px, fig_h_px = fig.bbox.width, fig.bbox.height
    box_w = box_w_px / fig_w_px
    box_h = box_h_px / fig_h_px

    legend_fs = 7.0
    label_pad = 0.006  # gap box → text
    item_gap = 0.020  # gap between legend items
    char_w = (legend_fs * 0.47 / 72.0) / fig.get_figwidth()
    labels = [lab for _k, lab, _c in SERIES]
    plain = [
        lab.replace(r"$d_E$", "dE").replace(r"$d_{\neg E}$", "d-E").replace("$", "")
        for lab in labels
    ]
    item_widths = [box_w + label_pad + max(len(s), 1) * char_w for s in plain]
    total_w = sum(item_widths) + item_gap * (len(SERIES) - 1)
    x = center_x - 0.5 * total_w
    y_c = y_fig

    frame_pad_x, frame_pad_y = 0.012, 0.008
    frame = Rectangle(
        (x - frame_pad_x, y_c - 0.5 * box_h - frame_pad_y),
        total_w + 2 * frame_pad_x,
        box_h + 2 * frame_pad_y,
        transform=fig.transFigure,
        facecolor="#FFFFFF",
        edgecolor="#CCCCCC",
        linewidth=0.6,
        clip_on=False,
        zorder=10,
    )
    fig.patches.append(frame)

    for (_key, label, color), iw in zip(SERIES, item_widths):
        rect = Rectangle(
            (x, y_c - 0.5 * box_h),
            box_w,
            box_h,
            transform=fig.transFigure,
            facecolor=color,
            edgecolor="#222222",
            linewidth=0.85,
            alpha=0.80,
            clip_on=False,
            zorder=11,
        )
        fig.patches.append(rect)
        # center_baseline keeps mathtext on the same visual line as the box.
        fig.text(
            x + box_w + label_pad,
            y_c,
            label,
            transform=fig.transFigure,
            ha="left",
            va="center_baseline",
            fontsize=legend_fs,
            color="#111111",
            clip_on=False,
            zorder=11,
        )
        x += iw + item_gap


def compose() -> Path:
    base.publication_style()

    grid_fit, fit = _load_curves(FIT_NPZ)
    grid_div, div = _load_curves(MPNET_NPZ)
    # Align diversity grid if stored separately
    if "grid" not in np.load(MPNET_NPZ, allow_pickle=True).files:
        grid_div = grid_fit

    bins_path = REPORTS / "rq2_species_best_fitness_bins_with_singletons.csv"
    if bins_path.exists():
        bins = pd.read_csv(bins_path)
    else:
        bins = pd.DataFrame(base.collect_with_singletons())
    bins = bins[bins["bin_lo"] >= 0.6 - 1e-9].copy()

    fig_w_in = 7.2
    mm = 1.0 / 25.4
    # Shrink left margin 4 mm total; grow right margin 2 mm.
    left = 0.09 - (4.0 * mm) / fig_w_in
    right = 0.985 - (2.0 * mm) / fig_w_in
    fig_h_in = 5.45
    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    # Move legend up 1 mm (less top margin) → +1 mm gap before (a)/(b).
    legend_y = 0.955 + mm / fig_h_in
    # Tighten (a)/(b) titles → (c) by 1 mm (hspace ≈ gap / mean axes height).
    subplot_area_h = 0.92 - 0.09
    mean_axes_h = subplot_area_h * (1.15 / (2.3 + 0.36))
    hspace = 0.36 - mm / (fig_h_in * mean_axes_h)
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[1.0, 1.30],
        hspace=hspace,
        wspace=0.30,
        left=left,
        right=right,
        top=0.92,
        bottom=0.09,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    _plot_anytime(ax_a, grid_fit, fit, "Top-50 mean fitness", "(a) Quality")
    _plot_anytime(ax_b, grid_div, div, r"MPNet $D_{r,50}$", "(b) Diversity")
    ax_a.set_ylim(0.0, 0.85)
    ax_b.set_ylim(0.10, 0.45)
    box_w_data = _draw_mirrored(ax_c, bins)

    # One straight legend line; rectangle width == panel-(c) box width.
    _draw_aligned_legend(
        fig,
        ax_c,
        box_w_data,
        center_x=(left + right) / 2.0,
        y_fig=legend_y,
    )

    for dest in (EXP, PAPER):
        dest.mkdir(parents=True, exist_ok=True)
        fig.savefig(dest / f"{STEM}.pdf")
        if dest is EXP:
            fig.savefig(dest / f"{STEM}.png", dpi=200)
    plt.close(fig)

    TEX.write_text(
        r"""% Auto-generated by compose_rq2_qd_mature_combined.py
% Use \input{rq2_combined_qd_mature_figure.tex} in the manuscript.
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/fig_rq2_qd_mature_combined.pdf}
  \caption{RQ2 outcomes over $2000$ evaluations ($n{=}10$). (a) Anytime
  top-$50$ mean fitness. (b) MPNet dissimilarity among the top-$50$ prompts.
  Lines and bands show medians and IQRs. (c) Fitness-binned counts of
  mature-species champions and unassigned individuals (noise and size-$1$ singletons); boxes show IQRs and
  whiskers show run ranges.}
  \label{fig:rq2-qd-mature}
\end{figure}
""",
        encoding="utf-8",
    )
    out = PAPER / f"{STEM}.pdf"
    print(f"Wrote {out}")
    print(f"Wrote {TEX}")
    return out


if __name__ == "__main__":
    compose()
