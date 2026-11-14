#!/usr/bin/env python3
"""RQ2/RQ3 mechanism exhibits from existing run artifacts (no new EA).

RQ2:
  - effect-size trajectory (HL Inc−LF) vs budget with bootstrap bands
  - anytime mature-species count curves
  - stratified / pooled LF−Inc contrast (declared reanalysis)

RQ3:
  - within-configuration Spearman of n_mat vs top-50 fitness and vs MPNet D
  - optional distinct P0 founder count among top-50

Example::

    PYTHONPATH=src:. ./venv/bin/python \\
      experiments/evostar_2027/validation/rq2_rq3_mechanism_exhibits.py
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/evostar_2027/rq2/scripts"))

from rq2_metrics import illicit_of, load_genomes, moderated_genomes, topk_anytime  # noqa: E402

LIVE = ROOT / "data/outputs/evostar2027"
OUT = ROOT / "experiments/evostar_2027/validation/reports"
FIG = ROOT / "notes/rough_draft/figures"
TOP50_NPZ = ROOT / "experiments/evostar_2027/rq2/reports/rq2_anytime_top50_curves.npz"
MPNET_NPZ = ROOT / "experiments/evostar_2027/rq2/reports/rq2_anytime_mpnet_curves.npz"
MINILM_NPZ = ROOT / "experiments/evostar_2027/rq2/reports/rq2_anytime_minilm_curves.npz"
TOPK_CSV = ROOT / "experiments/evostar_2027/rq2/reports/rq2_topk_b2000_runs.csv"
MPNET_CSV = ROOT / "experiments/evostar_2027/rq2/reports/rq2_mpnet_disp_runs.csv"
DYN_CSV = ROOT / "experiments/evostar_2027/rq2/reports/rq2_species_dynamics.csv"
SPECIES_NPZ = ROOT / "experiments/evostar_2027/rq2/reports/rq2_anytime_species_curves.npz"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("nli_entail", "dbscan"): "inc_nli",
    ("none", "none"): "no_speciation",
}
BLUE, RED, GRAY = "#0F4D92", "#B64342", "#4D4D4D"
GRID = np.arange(50, 2001, 25)
RNG = np.random.default_rng(2027)

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)


def _read(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def hodges_lehmann(a: Sequence[float], b: Sequence[float]) -> float:
    """HL shift of a−b (pairwise median)."""
    diffs = [float(x) - float(y) for x in a for y in b]
    return float(np.median(diffs)) if diffs else float("nan")


def bootstrap_hl(a: Sequence[float], b: Sequence[float], n_boot: int = 2000) -> Tuple[float, float, float]:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    point = hodges_lehmann(aa, bb)
    boots = []
    for _ in range(n_boot):
        sa = aa[RNG.integers(0, len(aa), len(aa))]
        sb = bb[RNG.integers(0, len(bb), len(bb))]
        boots.append(hodges_lehmann(sa, sb))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def load_top50_curves() -> Dict[Tuple[str, str], np.ndarray]:
    z = np.load(TOP50_NPZ, allow_pickle=True)
    out = {}
    for key in z.files:
        d, c = key.split("__", 1)
        out[(d, c)] = np.asarray(z[key], dtype=float)
    return out


def load_species_curves() -> Dict[Tuple[str, str], np.ndarray]:
    z = np.load(SPECIES_NPZ, allow_pickle=True)
    out = {}
    for key in z.files:
        d, c = key.split("__", 1)
        out[(d, c)] = np.asarray(z[key], dtype=float)
    return out


def _load_disp_curves(path: Path) -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    z = np.load(path, allow_pickle=True)
    grid = np.asarray(z["grid"], dtype=float) if "grid" in z.files else GRID.astype(float)
    out = {}
    for key in z.files:
        if key == "grid":
            continue
        d, c = key.split("__", 1)
        out[(d, c)] = np.asarray(z[key], dtype=float)
    return grid, out


def load_mpnet_curves() -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    return _load_disp_curves(MPNET_NPZ)


def load_minilm_curves() -> Tuple[np.ndarray, Dict[Tuple[str, str], np.ndarray]]:
    return _load_disp_curves(MINILM_NPZ)


def interaction_Iy(
    y_mini_inc: Sequence[float],
    y_mini_lf: Sequence[float],
    y_nli_inc: Sequence[float],
    y_nli_lf: Sequence[float],
    n_boot: int = 10000,
) -> Dict[str, float]:
    """Run-level bootstrap CI for representation × method interaction.

    I_Y = (mean_NLI_Inc - mean_NLI_LF) - (mean_Mini_Inc - mean_Mini_LF)
    """
    a = np.asarray(y_nli_inc, dtype=float)
    b = np.asarray(y_nli_lf, dtype=float)
    c = np.asarray(y_mini_inc, dtype=float)
    d = np.asarray(y_mini_lf, dtype=float)

    def point(aa, bb, cc, dd) -> float:
        return (float(np.mean(aa)) - float(np.mean(bb))) - (float(np.mean(cc)) - float(np.mean(dd)))

    est = point(a, b, c, d)
    boots = []
    for _ in range(n_boot):
        boots.append(
            point(
                a[RNG.integers(0, len(a), len(a))],
                b[RNG.integers(0, len(b), len(b))],
                c[RNG.integers(0, len(c), len(c))],
                d[RNG.integers(0, len(d), len(d))],
            )
        )
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"Iy": float(est), "lo": float(lo), "hi": float(hi)}


def effect_size_trajectory(curves: Dict[Tuple[str, str], np.ndarray]) -> List[Dict[str, Any]]:
    rows = []
    # Inc − LF within each dissimilarity; also pooled by stacking both dissimilarities
    for dist in ("embedding", "nli_entail"):
        A = curves.get((dist, "dbscan"))
        B = curves.get((dist, "leader_follower"))
        if A is None or B is None:
            continue
        for j, b in enumerate(GRID):
            a_vals = A[:, j]
            b_vals = B[:, j]
            a_vals = a_vals[np.isfinite(a_vals)]
            b_vals = b_vals[np.isfinite(b_vals)]
            if len(a_vals) < 5 or len(b_vals) < 5:
                continue
            hl, lo, hi = bootstrap_hl(a_vals, b_vals, n_boot=800)
            rows.append(
                {
                    "contrast": f"Inc-LF|{dist}",
                    "distance": dist,
                    "budget": int(b),
                    "hl": hl,
                    "lo": lo,
                    "hi": hi,
                }
            )
    # pooled: treat 20 Inc vs 20 LF (both dissimilarities)
    for j, b in enumerate(GRID):
        a_vals = []
        b_vals = []
        for dist in ("embedding", "nli_entail"):
            A = curves.get((dist, "dbscan"))
            B = curves.get((dist, "leader_follower"))
            if A is None or B is None:
                continue
            a_vals.extend(A[:, j][np.isfinite(A[:, j])].tolist())
            b_vals.extend(B[:, j][np.isfinite(B[:, j])].tolist())
        if len(a_vals) < 10 or len(b_vals) < 10:
            continue
        hl, lo, hi = bootstrap_hl(a_vals, b_vals, n_boot=800)
        rows.append(
            {
                "contrast": "Inc-LF|pooled",
                "distance": "pooled",
                "budget": int(b),
                "hl": hl,
                "lo": lo,
                "hi": hi,
            }
        )
    return rows


def cliff_delta(a: Sequence[float], b: Sequence[float]) -> float:
    aa, bb = list(a), list(b)
    n = 0
    gt = 0
    for x in aa:
        for y in bb:
            n += 1
            if x > y:
                gt += 1
            elif x < y:
                gt -= 1
    return float(gt / n) if n else float("nan")


def endpoint_contrasts(topk: List[Dict[str, str]], mpnet: Dict[Tuple[str, str, int], float]) -> List[Dict[str, Any]]:
    rows = []
    # per-stratum and pooled LF vs Inc on fitness and MPNet
    for label, dists in [
        ("embedding", ["embedding"]),
        ("nli_entail", ["nli_entail"]),
        ("pooled", ["embedding", "nli_entail"]),
    ]:
        lf, inc = [], []
        lf_d, inc_d = [], []
        for dist in dists:
            for r in topk:
                if r["distance"] != dist:
                    continue
                fit = _f(r["top50_mean_b2000"])
                seed = int(r["seed"])
                disp = mpnet.get((dist, r["clustering"], seed), float("nan"))
                if r["clustering"] == "leader_follower":
                    lf.append(fit)
                    if disp == disp:
                        lf_d.append(disp)
                elif r["clustering"] == "dbscan":
                    inc.append(fit)
                    if disp == disp:
                        inc_d.append(disp)
        if len(lf) >= 5 and len(inc) >= 5:
            hl, lo, hi = bootstrap_hl(inc, lf)  # Inc − LF
            rows.append(
                {
                    "family": "fitness",
                    "contrast": f"Inc-LF|{label}",
                    "n_lf": len(lf),
                    "n_inc": len(inc),
                    "hl": hl,
                    "lo": lo,
                    "hi": hi,
                    "delta": cliff_delta(inc, lf),
                    "note": "declared reanalysis; dissimilarity as block for pooled",
                }
            )
        if len(lf_d) >= 5 and len(inc_d) >= 5:
            hl, lo, hi = bootstrap_hl(inc_d, lf_d)
            rows.append(
                {
                    "family": "mpnet_disp",
                    "contrast": f"Inc-LF|{label}",
                    "n_lf": len(lf_d),
                    "n_inc": len(inc_d),
                    "hl": hl,
                    "lo": lo,
                    "hi": hi,
                    "delta": cliff_delta(inc_d, lf_d),
                    "note": "declared reanalysis",
                }
            )
    return rows


def p0_founders(run_dir: Path, clustering: str, k: int = 50) -> int:
    genomes = load_genomes(run_dir, clustering, attach_embeddings=False)
    by_id = {str(g.get("id")): g for g in genomes if isinstance(g, dict)}
    mod = moderated_genomes(genomes)[:2000]
    scored = [(illicit_of(g), g) for g in mod]
    scored = [(s, g) for s, g in scored if s is not None]
    scored.sort(key=lambda t: (-t[0], str(t[1].get("id"))))
    top = [g for _, g in scored[:k]]

    def founders(g: Dict[str, Any], seen: Optional[set] = None) -> set:
        if seen is None:
            seen = set()
        gid = str(g.get("id"))
        if gid in seen:
            return set()
        seen.add(gid)
        parents = g.get("parents") or []
        if not parents:
            return {gid}
        out = set()
        for p in parents:
            pid = str(p.get("id") if isinstance(p, dict) else p)
            pg = by_id.get(pid)
            if pg is None:
                out.add(pid)  # missing → treat as root
            else:
                out |= founders(pg, seen)
        return out

    all_f = set()
    for g in top:
        all_f |= founders(g)
    return len(all_f)


def spearman_within_config(
    pairs: List[Tuple[float, float]],
) -> Tuple[float, float, float]:
    """Spearman ρ with bootstrap CI on paired observations."""
    if len(pairs) < 5:
        return float("nan"), float("nan"), float("nan")
    x = np.asarray([p[0] for p in pairs], dtype=float)
    y = np.asarray([p[1] for p in pairs], dtype=float)

    def rho(a, b):
        ra = a.argsort().argsort().astype(float)
        rb = b.argsort().argsort().astype(float)
        if ra.std() < 1e-12 or rb.std() < 1e-12:
            return float("nan")
        return float(np.corrcoef(ra, rb)[0, 1])

    point = rho(x, y)
    boots = []
    for _ in range(2000):
        idx = RNG.integers(0, len(x), len(x))
        boots.append(rho(x[idx], y[idx]))
    boots = np.asarray(boots, dtype=float)
    boots = boots[np.isfinite(boots)]
    if boots.size == 0:
        return point, float("nan"), float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def _plot_median_iqr(
    ax,
    curves: Dict[Tuple[str, str], np.ndarray],
    grid: np.ndarray,
    series: Sequence[Tuple[str, str, str, str, str]],
    ylabel: str,
    title: str,
) -> None:
    for d, c, lab, color, ls in series:
        M = curves.get((d, c))
        if M is None or M.size == 0:
            continue
        lo, med, hi = np.nanpercentile(M, [25, 50, 75], axis=0)
        ax.plot(grid, med, color=color, ls=ls, lw=1.3, label=lab)
        ax.fill_between(grid, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlim(50, 2000)
    ax.set_xlabel("Unique evaluations")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=9)


def plot_rq2_mechanism(traj: List[Dict[str, Any]]) -> None:
    """Main-text Fig. RQ2: 2×2 anytime quality / MPNet diversity by representation."""
    fit = load_top50_curves()
    grid_mp, disp_mp = load_mpnet_curves()
    grid_fit = GRID.astype(float)
    if grid_mp.shape != grid_fit.shape or not np.allclose(grid_mp, grid_fit):
        raise RuntimeError("MPNet anytime grid does not match fitness grid")

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 5.0), layout="constrained")
    mini_series = [
        ("embedding", "leader_follower", "LF", BLUE, "-"),
        ("embedding", "dbscan", "IncDBSCAN", BLUE, "--"),
        ("none", "none", "Global EA", GRAY, "-"),
    ]
    nli_series = [
        ("nli_entail", "leader_follower", "LF", RED, "-"),
        ("nli_entail", "dbscan", "IncDBSCAN", RED, "--"),
        ("none", "none", "Global EA", GRAY, "-"),
    ]

    _plot_median_iqr(
        axes[0, 0], fit, grid_fit, mini_series, "Top-50 mean fitness", r"(a) MiniLM — quality"
    )
    _plot_median_iqr(
        axes[0, 1], fit, grid_fit, nli_series, "Top-50 mean fitness", r"(b) NLI — quality"
    )
    _plot_median_iqr(
        axes[1, 0],
        disp_mp,
        grid_mp,
        mini_series,
        r"MPNet $D_{r,50}$",
        r"(c) MiniLM — diversity",
    )
    _plot_median_iqr(
        axes[1, 1],
        disp_mp,
        grid_mp,
        nli_series,
        r"MPNet $D_{r,50}$",
        r"(d) NLI — diversity",
    )

    handles = [
        Line2D([0], [0], color=BLUE, ls="-", lw=1.3, label=r"LF ($d_E$)"),
        Line2D([0], [0], color=BLUE, ls="--", lw=1.3, label=r"Inc ($d_E$)"),
        Line2D([0], [0], color=RED, ls="-", lw=1.3, label=r"LF ($d_{\neg E}$)"),
        Line2D([0], [0], color=RED, ls="--", lw=1.3, label=r"Inc ($d_{\neg E}$)"),
        Line2D([0], [0], color=GRAY, ls="-", lw=1.3, label="Global EA"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.04, hspace=0.08, wspace=0.05)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rq2_mechanism.pdf")
    fig.savefig(FIG / "fig_rq2_mechanism.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_rq2_mechanism.pdf'}")

    # Appendix: HL Inc−LF trajectory (formerly main-text panel).
    plot_rq2_hl_appendix(traj)


def plot_rq2_hl_appendix(traj: List[Dict[str, Any]]) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 2.6), layout="constrained")
    styles = {
        "embedding": (BLUE, "-", r"$d_E$"),
        "nli_entail": (RED, "-", r"$d_{\neg E}$"),
        "pooled": (GRAY, "--", "pooled"),
    }
    for dist, (color, ls, lab) in styles.items():
        sub = [r for r in traj if r["distance"] == dist]
        if not sub:
            continue
        xs = [r["budget"] for r in sub]
        ys = [r["hl"] for r in sub]
        lo = [r["lo"] for r in sub]
        hi = [r["hi"] for r in sub]
        ax.plot(xs, ys, color=color, ls=ls, lw=1.3, label=lab)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0.0, color="#888", lw=0.8)
    ax.axhline(0.05, color="#888", lw=0.7, ls=":")
    ax.axhline(-0.05, color="#888", lw=0.7, ls=":")
    ax.set_xlim(50, 2000)
    ax.set_xlabel("Unique evaluations")
    ax.set_ylabel(r"HL shift (Inc$-$LF)")
    ax.legend(fontsize=7, loc="best")
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rq2_hl_trajectory.pdf")
    fig.savefig(FIG / "fig_rq2_hl_trajectory.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_rq2_hl_trajectory.pdf'}")


def plot_rq3_structure() -> None:
    """RQ2 complement: whole-population niche structure over budget."""
    species = load_species_curves()
    dyn = _read(DYN_CSV)
    budgets = [250, 500, 1000, 1500, 2000]
    cells = [
        ("embedding", "leader_follower", r"LF $d_E$", BLUE, "-"),
        ("embedding", "dbscan", r"Inc $d_E$", BLUE, "--"),
        ("nli_entail", "leader_follower", r"LF $d_{\neg E}$", RED, "-"),
        ("nli_entail", "dbscan", r"Inc $d_{\neg E}$", RED, "--"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), layout="constrained")

    # (a) anytime n_mat
    ax = axes[0]
    for d, c, lab, color, ls in cells:
        M = species.get((d, c))
        if M is None:
            continue
        mean = np.nanmean(M, axis=0)
        lo = np.nanmin(M, axis=0)
        hi = np.nanmax(M, axis=0)
        ax.plot(GRID, mean, color=color, ls=ls, lw=1.3, label=lab)
        ax.fill_between(GRID, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.set_xlim(50, 2000)
    ax.set_xlabel("Unique evaluations")
    ax.set_ylabel(r"Mature species $n_{\mathrm{mat}}$")
    ax.set_title("(a)", loc="left", fontsize=9)

    # (b) assigned dominance from dynamics checkpoints
    ax = axes[1]
    for d, c, lab, color, ls in cells:
        means, los, his = [], [], []
        xs = []
        for b in budgets:
            vals = [
                _f(r["d_assigned"])
                for r in dyn
                if r["distance"] == d
                and r["clustering"] == c
                and abs(_f(r["budget"]) - b) < 1e-9
                and _f(r["d_assigned"]) == _f(r["d_assigned"])
            ]
            if not vals:
                continue
            xs.append(b)
            means.append(float(np.mean(vals)))
            los.append(float(np.min(vals)))
            his.append(float(np.max(vals)))
        if not xs:
            continue
        ax.plot(xs, means, color=color, ls=ls, lw=1.3, label=lab, marker="o", ms=3.5)
        ax.fill_between(xs, los, his, color=color, alpha=0.12, linewidth=0)
    ax.set_xlim(200, 2000)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Unique evaluations")
    ax.set_ylabel(r"$D_{\mathrm{assigned}}$")
    ax.set_title("(b)", loc="left", fontsize=9)

    handles = [Line2D([0], [0], color=color, ls=ls, lw=1.3, label=lab) for _, _, lab, color, ls in cells]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rq2_structure.pdf")
    fig.savefig(FIG / "fig_rq2_structure.png")
    # Keep legacy filename for older notes/build paths.
    fig.savefig(FIG / "fig_rq2_structure.pdf")
    fig.savefig(FIG / "fig_rq2_structure.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_rq2_structure.pdf'}")


def plot_rq3_coupling(
    points: List[Dict[str, Any]],
    global_fit: List[float],
    global_disp: List[float],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5), layout="constrained")
    colors = {
        ("embedding", "leader_follower"): BLUE,
        ("embedding", "dbscan"): BLUE,
        ("nli_entail", "leader_follower"): RED,
        ("nli_entail", "dbscan"): RED,
    }
    markers = {
        "leader_follower": "o",
        "dbscan": "s",
    }
    for ax, ykey, ylab in [
        (axes[0], "fitness", "Top-50 mean fitness"),
        (axes[1], "disp", r"MPNet $D_{r,50}$"),
    ]:
        for r in points:
            if r.get("degenerate"):
                continue
            key = (r["distance"], r["clustering"])
            ax.scatter(
                [r["n_mat"]],
                [r[ykey]],
                s=18,
                color=colors[key],
                marker=markers[r["clustering"]],
                alpha=0.85,
            )
        # Global EA band
        gvals = global_fit if ykey == "fitness" else global_disp
        if gvals:
            med = float(np.median(gvals))
            ax.axhline(med, color=GRAY, lw=1.2, ls="-")
            ax.axhspan(np.percentile(gvals, 25), np.percentile(gvals, 75), color=GRAY, alpha=0.12)
        ax.set_xscale("log")
        ax.set_xlabel(r"End-of-run $n_{\mathrm{mat}}$")
        ax.set_ylabel(ylab)
    axes[0].set_title("(a)", loc="left", fontsize=9)
    axes[1].set_title("(b)", loc="left", fontsize=9)
    fig.savefig(FIG / "fig_rq3_coupling.pdf")
    fig.savefig(FIG / "fig_rq3_coupling.png")
    plt.close(fig)
    print(f"wrote {FIG / 'fig_rq3_coupling.pdf'}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    topk = _read(TOPK_CSV)
    mpnet_rows = _read(MPNET_CSV)
    mpnet = {(r["distance"], r["clustering"], int(r["seed"])): _f(r["disp_mpnet_top50_b2000"]) for r in mpnet_rows}
    dyn = _read(DYN_CSV)

    curves = load_top50_curves()
    print("computing effect-size trajectory...")
    traj = effect_size_trajectory(curves)
    with (OUT / "rq2_effect_trajectory.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(traj[0].keys()))
        w.writeheader()
        w.writerows(traj)

    contrasts = endpoint_contrasts(topk, mpnet)
    with (OUT / "rq2_pooled_contrasts.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(contrasts[0].keys()))
        w.writeheader()
        w.writerows(contrasts)
    print("endpoint contrasts:")
    for r in contrasts:
        print(f"  {r['family']} {r['contrast']}: HL={r['hl']:.3f} [{r['lo']:.3f},{r['hi']:.3f}] δ={r['delta']:.2f}")

    def cell_vals(dist: str, clus: str, kind: str) -> List[float]:
        out = []
        for r in topk:
            if r["distance"] != dist or r["clustering"] != clus:
                continue
            seed = int(r["seed"])
            if kind == "fitness":
                out.append(_f(r["top50_mean_b2000"]))
            else:
                out.append(mpnet.get((dist, clus, seed), float("nan")))
        return [x for x in out if x == x]

    iy_fit = interaction_Iy(
        cell_vals("embedding", "dbscan", "fitness"),
        cell_vals("embedding", "leader_follower", "fitness"),
        cell_vals("nli_entail", "dbscan", "fitness"),
        cell_vals("nli_entail", "leader_follower", "fitness"),
    )
    iy_disp = interaction_Iy(
        cell_vals("embedding", "dbscan", "disp"),
        cell_vals("embedding", "leader_follower", "disp"),
        cell_vals("nli_entail", "dbscan", "disp"),
        cell_vals("nli_entail", "leader_follower", "disp"),
    )
    print(
        f"I_Y fitness={iy_fit['Iy']:.3f} [{iy_fit['lo']:.3f},{iy_fit['hi']:.3f}]; "
        f"disp={iy_disp['Iy']:.3f} [{iy_disp['lo']:.3f},{iy_disp['hi']:.3f}]"
    )
    with (OUT / "rq2_interaction_Iy.json").open("w", encoding="utf-8") as f:
        json.dump({"fitness": iy_fit, "mpnet_disp": iy_disp}, f, indent=2)

    plot_rq2_mechanism(traj)
    plot_rq3_structure()

    # RQ3 coupling points at b=2000 (appendix / legacy coupling view)
    points = []
    for r in topk:
        d, c, seed = r["distance"], r["clustering"], int(r["seed"])
        if c == "none":
            continue
        nmat_vals = [
            _f(x["n_mature"])
            for x in dyn
            if x["distance"] == d and x["clustering"] == c and int(x["seed"]) == seed and abs(_f(x["budget"]) - 2000) < 1e-9
        ]
        n_mat = nmat_vals[0] if nmat_vals else float("nan")
        points.append(
            {
                "distance": d,
                "clustering": c,
                "seed": seed,
                "n_mat": n_mat,
                "fitness": _f(r["top50_mean_b2000"]),
                "disp": mpnet.get((d, c, seed), float("nan")),
                "degenerate": d == "nli_entail" and c == "dbscan",
            }
        )

    # within-cell ranks then pool (exclude degenerate)
    def within_cell_pairs(ykey: str) -> List[Tuple[float, float]]:
        by = defaultdict(list)
        for p in points:
            if p["degenerate"] or not np.isfinite(p["n_mat"]) or not np.isfinite(p[ykey]):
                continue
            by[(p["distance"], p["clustering"])].append(p)
        paired = []
        for cell, rows in by.items():
            xs = np.asarray([r["n_mat"] for r in rows], dtype=float)
            ys = np.asarray([r[ykey] for r in rows], dtype=float)
            # rank within cell
            rx = xs.argsort().argsort().astype(float)
            ry = ys.argsort().argsort().astype(float)
            paired.extend(list(zip(rx.tolist(), ry.tolist())))
        return paired

    rho_fit = spearman_within_config(within_cell_pairs("fitness"))
    rho_disp = spearman_within_config(within_cell_pairs("disp"))

    # P0 founders
    founder_rows = []
    for d, c in LIVE_DIR:
        for seed in range(10):
            run = LIVE / LIVE_DIR[(d, c)] / f"rep{seed}_s{seed}"
            try:
                n_f = p0_founders(run, c, 50)
            except Exception as e:
                n_f = -1
                print(f"founder fail {d} {c} {seed}: {e}")
            founder_rows.append({"distance": d, "clustering": c, "seed": seed, "n_p0_founders": n_f})
            print(f"founders {d} {c} seed{seed}: {n_f}")

    with (OUT / "rq3_p0_founders.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["distance", "clustering", "seed", "n_p0_founders"])
        w.writeheader()
        w.writerows(founder_rows)

    g_fit = [_f(r["top50_mean_b2000"]) for r in topk if r["distance"] == "none"]
    g_disp = [mpnet[("none", "none", int(r["seed"]))] for r in topk if r["distance"] == "none"]
    plot_rq3_coupling(points, g_fit, g_disp)

    summary = {
        "rq2_endpoint_contrasts": contrasts,
        "rq2_interaction_Iy": {"fitness": iy_fit, "mpnet_disp": iy_disp},
        "rq3_spearman_nmat_vs_fitness_within_cell": {
            "rho": rho_fit[0],
            "lo": rho_fit[1],
            "hi": rho_fit[2],
        },
        "rq3_spearman_nmat_vs_mpnet_within_cell": {
            "rho": rho_disp[0],
            "lo": rho_disp[1],
            "hi": rho_disp[2],
        },
        "founder_medians": {},
    }
    for d, c in LIVE_DIR:
        vals = [r["n_p0_founders"] for r in founder_rows if r["distance"] == d and r["clustering"] == c and r["n_p0_founders"] >= 0]
        if vals:
            summary["founder_medians"][f"{d}|{c}"] = float(np.median(vals))

    (OUT / "rq2_rq3_mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
