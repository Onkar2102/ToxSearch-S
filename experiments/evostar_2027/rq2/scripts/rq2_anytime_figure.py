#!/usr/bin/env python3
"""Main-text Fig. rq2: anytime top-k fitness + mature-species count."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import load_genomes, topk_anytime  # noqa: E402
from rq2_species_dynamics import BUDGETS, CELLS, _cell_median, collect_all  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}
FIG = ROOT / "experiments" / "evostar_2027" / "rq2" / "figures"
PAPER_FIG = ROOT / "notes" / "rough_draft" / "figures"
ELITE_K = 100
GRID = np.arange(ELITE_K, 2001, 25)


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
    # step-hold: last observed mean at each budget
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
    """Return cell -> (n_runs, len(grid)) array of anytime top-k means."""
    out: Dict[Tuple[str, str], List[np.ndarray]] = {}
    for distance, clustering in LIVE_DIR:
        curves: List[np.ndarray] = []
        for seed in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
            if not complete(run_dir):
                continue
            genomes = load_genomes(run_dir, clustering)
            xs, ys = topk_anytime(genomes, k=ELITE_K)
            curves.append(interpolate_curve(xs, ys, GRID))
            print(f"anytime {distance} {clustering} seed{seed}", flush=True)
        out[(distance, clustering)] = np.asarray(curves, dtype=float)
    return out


def plot(topk: Dict[Tuple[str, str], np.ndarray], dyn_rows: List[Dict[str, Any]]) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.6), layout="constrained")
    y_lo, y_hi = 1.0, 0.0
    for distance, clustering, label, color, ls in CELLS:
        arr = topk.get((distance, clustering))
        if arr is None or arr.size == 0:
            continue
        med = np.nanmedian(arr, axis=0)
        q25 = np.nanpercentile(arr, 25, axis=0)
        q75 = np.nanpercentile(arr, 75, axis=0)
        ax0.plot(GRID, med, color=color, ls=ls, lw=1.6, label=label)
        ax0.fill_between(GRID, q25, q75, color=color, alpha=0.12, linewidth=0)
        finite = np.concatenate([q25, q75, med])
        finite = finite[np.isfinite(finite)]
        if finite.size:
            y_lo = min(y_lo, float(finite.min()))
            y_hi = max(y_hi, float(finite.max()))

        ys, lo, hi = [], [], []
        for B in BUDGETS:
            sub = [
                float(r["n_mature"])
                for r in dyn_rows
                if r["distance"] == distance
                and r["clustering"] == clustering
                and int(r["budget"]) == B
                and r.get("n_mature") == r.get("n_mature")
            ]
            if not sub:
                med_b, _ = _cell_median(dyn_rows, distance, clustering, B, "n_mature")
                ys.append(med_b)
                lo.append(med_b)
                hi.append(med_b)
            else:
                ys.append(float(np.median(sub)))
                lo.append(float(np.percentile(sub, 25)))
                hi.append(float(np.percentile(sub, 75)))
        ax1.plot(BUDGETS, ys, color=color, ls=ls, lw=1.6, marker="o", ms=3.2, label=label)
        ax1.fill_between(BUDGETS, lo, hi, color=color, alpha=0.12, linewidth=0)

    ax0.set_xlabel("Unique evaluated individuals")
    ax1.set_xlabel("Unique evaluated individuals")
    ax0.set_ylabel(rf"Top-{ELITE_K} mean fitness")
    ax1.set_ylabel(r"Mature species $n_{\mathrm{mat}}$")
    ax0.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax1.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax0.set_xlim(0, 2000)
    ax1.set_xlim(0, 2000)
    # Top-100 means sit well below the old top-10 band (~0.9); autoscale with padding.
    if y_hi > y_lo:
        pad = 0.05 * (y_hi - y_lo)
        ax0.set_ylim(max(0.0, y_lo - pad), min(1.02, y_hi + pad))
    else:
        ax0.set_ylim(0.0, 1.02)
    ax0.grid(True, axis="y", alpha=0.25)
    ax1.grid(True, axis="y", alpha=0.25)
    ax0.legend(fontsize=6.5, loc="lower right", frameon=False)

    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (FIG, PAPER_FIG):
        fig.savefig(dest / "fig_rq2.pdf")
        fig.savefig(dest / "fig_rq2.png", dpi=200)
    plt.close(fig)
    print(f"wrote {PAPER_FIG / 'fig_rq2.pdf'}")


def main() -> int:
    topk = collect_topk()
    dyn = collect_all()
    plot(topk, dyn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
