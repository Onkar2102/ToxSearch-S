#!/usr/bin/env python3
"""Main-text Fig. rq2: anytime top-50 mean fitness + end-of-run D_r,50.

Matches the paper caption: (a) median anytime top-50 mean with IQR;
(b) end-of-run semantic dispersion of top-50 individuals. Includes Global EA.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import load_genomes, moderated_genomes, topk_anytime  # noqa: E402
from rq2_topk50_primary import elite_disp  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
    ("none", "none"): "no_speciation",
}
BLUE = "#0F4D92"
RED = "#B64342"
GRAY = "#4D4D4D"
CELLS: List[Tuple[str, str, str, str, str]] = [
    ("embedding", "leader_follower", r"LF $d_E$", BLUE, "-"),
    ("embedding", "dbscan", r"Inc $d_E$", BLUE, "--"),
    ("nli_entail", "leader_follower", r"LF $d_{\neg E}$", RED, "-"),
    ("nli_entail", "dbscan", r"Inc $d_{\neg E}$", RED, "--"),
    ("none", "none", "Global EA", GRAY, "-"),
]
FIG = ROOT / "experiments" / "evostar_2027" / "rq2" / "figures"
PAPER_FIG = ROOT / "notes" / "rough_draft" / "figures"
ELITE_K = 50
BUDGET = 2000
GRID = np.arange(ELITE_K, BUDGET + 1, 25)

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text()).get("status")) == "complete"
    except Exception:
        return False


def interpolate_curve(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if xs.size == 0:
        return np.full(grid.shape, np.nan)
    out = np.full(grid.shape, np.nan)
    j = 0
    last = np.nan
    for i, b in enumerate(grid):
        while j < xs.size and xs[j] <= b:
            last = ys[j]
            j += 1
        out[i] = last
    return out


def collect_topk() -> Dict[Tuple[str, str], np.ndarray]:
    """Return cell -> (n_runs, len(grid)) array of anytime top-50 means."""
    out: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for distance, clustering in LIVE_DIR:
        curves: List[np.ndarray] = []
        for seed in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
            if not complete(run_dir):
                continue
            genomes = load_genomes(run_dir, clustering, attach_embeddings=False)
            xs, ys = topk_anytime(genomes, k=ELITE_K)
            curves.append(interpolate_curve(xs, ys, GRID))
            print(f"anytime {distance} {clustering} seed{seed}", flush=True)
        out[(distance, clustering)] = np.asarray(curves, dtype=float)
    return out


def collect_dispersion() -> Dict[Tuple[str, str], List[float]]:
    """End-of-run D_r,50 per seed for each cell."""
    out: Dict[Tuple[str, str], List[float]] = {}
    for distance, clustering in LIVE_DIR:
        vals: List[float] = []
        for seed in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
            if not complete(run_dir):
                continue
            genomes = load_genomes(run_dir, clustering, attach_embeddings=True)
            first = moderated_genomes(genomes)[:BUDGET]
            d = elite_disp(first, ELITE_K)
            if d == d and np.isfinite(d):
                vals.append(float(d))
            print(f"disp {distance} {clustering} seed{seed} {d}", flush=True)
        out[(distance, clustering)] = vals
    return out


def plot(
    topk: Dict[Tuple[str, str], np.ndarray],
    disp: Dict[Tuple[str, str], List[float]],
) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.6), layout="constrained")
    y_lo, y_hi = 1.0, 0.0
    handles: List[Any] = []

    # Panel (a): anytime top-50
    for distance, clustering, label, color, ls in CELLS:
        arr = topk.get((distance, clustering))
        if arr is None or arr.size == 0:
            continue
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax0.plot(GRID, med, color=color, ls=ls, lw=1.6, label=label)
        ax0.fill_between(GRID, q25, q75, color=color, alpha=0.12, linewidth=0)
        handles.append(Line2D([0], [0], color=color, ls=ls, lw=1.6, label=label))
        finite = np.concatenate([q25, q75, med])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y_lo = min(y_lo, float(finite.min()))
            y_hi = max(y_hi, float(finite.max()))

    # Panel (b): end-of-run D_r,50 points + median bars
    n_cells = len(CELLS)
    xpos = np.arange(n_cells, dtype=float)
    for i, (distance, clustering, label, color, ls) in enumerate(CELLS):
        vals = disp.get((distance, clustering), [])
        if not vals:
            continue
        jitter = (np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else np.zeros(1))
        ax1.scatter(
            np.full(len(vals), xpos[i]) + jitter,
            vals,
            s=18,
            color=color,
            alpha=0.75,
            zorder=3,
            linewidths=0,
        )
        med = float(np.median(vals))
        ax1.hlines(med, xpos[i] - 0.28, xpos[i] + 0.28, colors=color, lw=2.0, linestyles=ls, zorder=4)

    ax0.set_xlabel("Unique evaluated individuals")
    ax0.set_ylabel(rf"Top-{ELITE_K} mean fitness")
    ax0.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax0.set_xlim(0, BUDGET)
    if y_hi > y_lo:
        pad = 0.05 * (y_hi - y_lo)
        ax0.set_ylim(max(0.0, y_lo - pad), min(1.02, y_hi + pad))
    else:
        ax0.set_ylim(0.0, 1.02)
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(handles=handles + [Patch(facecolor=GRAY, alpha=0.18, edgecolor="none", label="IQR")], fontsize=6.5, loc="lower right", frameon=False)

    ax1.set_xlabel("Configuration")
    ax1.set_ylabel(r"$D_{r,50}$")
    ax1.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([c[2] for c in CELLS], rotation=25, ha="right", fontsize=7)
    ax1.set_xlim(-0.6, n_cells - 0.4)
    ax1.grid(True, axis="y", alpha=0.25)

    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (FIG, PAPER_FIG):
        fig.savefig(dest / "fig_rq2.pdf")
        fig.savefig(dest / "fig_rq2.png", dpi=200)
    plt.close(fig)
    print(f"wrote {PAPER_FIG / 'fig_rq2.pdf'}")


def main() -> int:
    topk = collect_topk()
    disp = collect_dispersion()
    plot(topk, disp)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
