#!/usr/bin/env python3
"""RQ3: PCA/UMAP projections of unique prompts across 40 RQ2 runs.

2D panels (semantic plane) and 3D panels (PC/UMAP × illicit fitness).

  MPLBACKEND=Agg PYTHONPATH=src:. venv/bin/python \\
    experiments/evostar_2027/rq3/scripts/rq3_projection_figure.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
FIG = ROOT / "experiments" / "evostar_2027" / "rq3" / "figures"
NOTES_FIG = ROOT / "notes" / "rough_draft" / "figures"
CACHE = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports" / "proj_cache"

CONFIGS: List[Tuple[str, str, str, str, str]] = [
    ("embedding", "leader_follower", r"LF $d_E$", "#1E88E5", "#0D47A1"),
    ("embedding", "dbscan", r"Inc $d_E$", "#43A047", "#1B5E20"),
    ("nli_entail", "leader_follower", r"LF $d_{\neg E}$", "#D500F9", "#6A0080"),
    ("nli_entail", "dbscan", r"Inc $d_{\neg E}$", "#FF6D00", "#E65100"),
]

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
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


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def genome_emb(rec: Dict) -> Optional[np.ndarray]:
    emb = rec.get("prompt_embedding")
    if isinstance(emb, list) and len(emb) == 384:
        return l2_normalize(np.asarray(emb, dtype=np.float64))
    return None


def collect_unique(
    tau: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Unique prompts per config.

    Keeps the appearance with max illicit fitness; stores that fitness and the
    corresponding ``generation`` (for the vertical axis).
    """
    embs: List[np.ndarray] = []
    labels: List[int] = []
    fits: List[float] = []
    gens: List[float] = []
    for ci, (distance, clustering, name, _fill, _edge) in enumerate(CONFIGS):
        # prompt -> (emb, max_fitness, generation_at_max_fitness)
        seen: Dict[str, Tuple[np.ndarray, float, float]] = {}
        for seed in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
            if not complete(run_dir):
                continue
            for rec in load_genomes(run_dir, clustering):
                prompt = str(rec.get("prompt") or "").strip()
                if not prompt:
                    continue
                emb = genome_emb(rec)
                if emb is None:
                    continue
                f = illicit_of(rec)
                if f is None:
                    continue
                fv = float(f)
                if tau is not None and fv < tau:
                    continue
                try:
                    gen = float(rec.get("generation", 0) or 0)
                except Exception:
                    gen = 0.0
                prev = seen.get(prompt)
                if prev is None or fv > prev[1]:
                    seen[prompt] = (emb, fv, gen)
        print(f"{name}: {len(seen)} unique prompts", flush=True)
        for emb, fv, gen in seen.values():
            embs.append(emb)
            labels.append(ci)
            fits.append(fv)
            gens.append(gen)
    if not embs:
        raise RuntimeError("no embeddings collected")
    return (
        np.stack(embs, axis=0),
        np.asarray(labels, dtype=int),
        np.asarray(fits, dtype=np.float64),
        np.asarray(gens, dtype=np.float64),
    )


def pca_2d(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, s, vt = np.linalg.svd(Xc, full_matrices=False)
    coords = Xc @ vt[:2].T
    ev = (s**2) / max(float(np.sum(s**2)), 1e-12)
    return coords, ev[:2]


def pca_reduce(X: np.ndarray, n_components: int) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(n_components, vt.shape[0])
    return Xc @ vt[:k].T


def umap_2d(
    X: np.ndarray,
    seed: int,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    pca_dims: int = 50,
) -> np.ndarray:
    import umap

    Xp = pca_reduce(X, pca_dims) if X.shape[1] > pca_dims else X
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        random_state=seed,
        low_memory=True,
        verbose=True,
    )
    return np.asarray(reducer.fit_transform(Xp), dtype=np.float64)


def cov_ellipse(ax, pts: np.ndarray, edge: str, n_std: float = 2.0) -> None:
    if pts.shape[0] < 3:
        return
    mu = pts.mean(axis=0)
    cov = np.cov(pts, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = float(np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(vals, 0.0))
    ax.add_patch(
        Ellipse(
            xy=mu,
            width=width,
            height=height,
            angle=angle,
            facecolor="none",
            edgecolor=edge,
            linewidth=1.6,
            zorder=5,
        )
    )


def _shared_limits(coords: np.ndarray, pad: float = 0.04) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    lo = lo - pad * span
    hi = hi + pad * span
    return (float(lo[0]), float(hi[0])), (float(lo[1]), float(hi[1]))


def _save_fig(fig, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FIG.mkdir(parents=True, exist_ok=True)
    for path in (out, NOTES_FIG / out.name, out.with_suffix(".png"), NOTES_FIG / out.with_suffix(".png").name):
        fig.savefig(path)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def _subsample(idx: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    if idx.size <= k:
        return idx
    return rng.choice(idx, size=k, replace=False)


def plot_projection_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    xlim, ylim = _shared_limits(coords)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 5.6), sharex=True, sharey=True)
    for ci, (_d, _c, name, fill, edge) in enumerate(CONFIGS):
        ax = axes.ravel()[ci]
        other = coords[labels != ci]
        pts = coords[labels == ci]
        if other.shape[0]:
            ax.scatter(other[:, 0], other[:, 1], s=2.5, c="#D0D0D0", alpha=0.15, linewidths=0, rasterized=True, zorder=1)
        ax.scatter(pts[:, 0], pts[:, 1], s=4.5, c=fill, alpha=0.45, linewidths=0, rasterized=True, zorder=2)
        cov_ellipse(ax, pts, edge, n_std=2.0)
        ax.set_title(name, color=edge, fontsize=9, fontweight="bold")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.2, linewidth=0.5)
        if ci % 2 == 0:
            ax.set_ylabel(ylabel)
        if ci >= 2:
            ax.set_xlabel(xlabel)
    fig.suptitle(title, fontsize=10, y=1.01)
    fig.tight_layout()
    _save_fig(fig, out)


def _sizes_from_fitness(fitness: np.ndarray, s_min: float = 2.0, s_max: float = 28.0) -> np.ndarray:
    """Map illicit fitness in [0,1] to marker area (points^2)."""
    f = np.clip(np.asarray(fitness, dtype=np.float64), 0.0, 1.0)
    return s_min + (s_max - s_min) * (f**1.5)


def plot_projection_3d(
    coords: np.ndarray,
    labels: np.ndarray,
    fitness: np.ndarray,
    generations: np.ndarray,
    out: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    seed: int = 0,
    max_fg: int = 5000,
    max_bg: int = 4000,
    elev: float = 22.0,
    azim: float = -55.0,
) -> None:
    """2×2 3D panels: XY = semantic plane, Z = generation; marker size ∝ fitness."""
    rng = np.random.default_rng(seed)
    xlim, ylim = _shared_limits(coords)
    z_hi = float(np.nanmax(generations)) if generations.size else 1.0
    zlim = (0.0, max(z_hi, 1.0))

    fig = plt.figure(figsize=(7.2, 6.4))
    for ci, (_d, _c, name, fill, edge) in enumerate(CONFIGS):
        ax = fig.add_subplot(2, 2, ci + 1, projection="3d")
        other_idx = np.flatnonzero(labels != ci)
        fg_idx = np.flatnonzero(labels == ci)
        other_idx = _subsample(other_idx, max_bg, rng)
        fg_idx = _subsample(fg_idx, max_fg, rng)

        if other_idx.size:
            ax.scatter(
                coords[other_idx, 0],
                coords[other_idx, 1],
                generations[other_idx],
                s=_sizes_from_fitness(fitness[other_idx], s_min=1.5, s_max=10.0),
                c="#C8C8C8",
                alpha=0.08,
                linewidths=0,
                depthshade=False,
                rasterized=True,
            )
        # Draw low-fitness first so high-fitness (larger) markers sit on top.
        order = np.argsort(fitness[fg_idx])
        fg_idx = fg_idx[order]
        ax.scatter(
            coords[fg_idx, 0],
            coords[fg_idx, 1],
            generations[fg_idx],
            s=_sizes_from_fitness(fitness[fg_idx]),
            c=fill,
            alpha=0.55,
            linewidths=0,
            depthshade=False,
            rasterized=True,
        )
        ax.set_title(name, color=edge, fontsize=9, fontweight="bold", pad=2)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_zlim(*zlim)
        ax.set_xlabel(xlabel, labelpad=2, fontsize=7)
        ax.set_ylabel(ylabel, labelpad=2, fontsize=7)
        ax.set_zlabel("generation", labelpad=2, fontsize=7)
        ax.view_init(elev=elev, azim=azim)
        ax.tick_params(labelsize=6)
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

    fig.suptitle(title, fontsize=10, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save_fig(fig, out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-neighbors", type=int, default=30)
    ap.add_argument("--min-dist", type=float, default=0.1)
    ap.add_argument("--pca-dims", type=int, default=50)
    ap.add_argument(
        "--methods",
        type=str,
        default="pca,umap",
        help="comma list: pca and/or umap",
    )
    ap.add_argument(
        "--views",
        type=str,
        default="2d,3d",
        help="comma list: 2d and/or 3d",
    )
    args = ap.parse_args()
    methods = [m.strip().lower() for m in args.methods.split(",") if m.strip()]
    views = [v.strip().lower() for v in args.views.split(",") if v.strip()]

    print("collecting unique prompts (+ fitness, generation)...", flush=True)
    X, y, fitness, generations = collect_unique(tau=args.tau)
    print(
        f"pooled N={X.shape[0]}  D={X.shape[1]}  "
        f"fitness [{fitness.min():.3f},{fitness.max():.3f}]  "
        f"generation [{generations.min():.0f},{generations.max():.0f}]",
        flush=True,
    )

    stem = "all" if args.tau is None else f"tau{args.tau:.2f}"
    tag = "All unique prompts (40 runs)" if args.tau is None else rf"High-fitness prompts ($\tau={args.tau:.2f}$)"

    CACHE.mkdir(parents=True, exist_ok=True)

    if "pca" in methods:
        print("PCA...", flush=True)
        coords, explained = pca_2d(X)
        print(f"explained var PC1={explained[0]:.3f} PC2={explained[1]:.3f}", flush=True)
        xl = f"PC1 ({100.0 * explained[0]:.1f}% var.)"
        yl = f"PC2 ({100.0 * explained[1]:.1f}% var.)"
        if "2d" in views:
            plot_projection_2d(coords, y, FIG / f"fig_rq3_projection_pca_{stem}.pdf", f"{tag} — MiniLM PCA", xl, yl)
            plot_projection_2d(coords, y, FIG / "fig_rq3_projection_all.pdf", f"{tag} — MiniLM PCA", xl, yl)
        if "3d" in views:
            plot_projection_3d(
                coords,
                y,
                fitness,
                generations,
                FIG / f"fig_rq3_projection_pca3d_{stem}.pdf",
                f"{tag} — MiniLM PCA; size$\\propto$fitness",
                xl,
                yl,
                seed=args.seed,
            )

    if "umap" in methods:
        print("UMAP...", flush=True)
        coords_u = umap_2d(
            X,
            seed=args.seed,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            pca_dims=args.pca_dims,
        )
        if "2d" in views:
            plot_projection_2d(
                coords_u,
                y,
                FIG / f"fig_rq3_projection_umap_{stem}.pdf",
                f"{tag} — MiniLM UMAP",
                "UMAP-1",
                "UMAP-2",
            )
        if "3d" in views:
            plot_projection_3d(
                coords_u,
                y,
                fitness,
                generations,
                FIG / f"fig_rq3_projection_umap3d_{stem}.pdf",
                f"{tag} — MiniLM UMAP; size$\\propto$fitness",
                "UMAP-1",
                "UMAP-2",
                seed=args.seed,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
