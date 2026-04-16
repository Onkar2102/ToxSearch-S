#!/usr/bin/env python3
"""
RQ1 / C1 (three-way): ToxSearch vs ToxSearch-S vs RainbowPlus.

This script produces publication-grade, run-level quality + diversity metrics and
figures under a fixed evaluation budget B (default 1000).

Inputs:
  - Manifest CSV with columns: method,run_id,run_path
    - toxsearch / toxsearch_s: run_path is a directory containing EvolutionTracker.json
    - rainbow_plus: run_path is a path to all_genomes.jsonl

Evaluation budget alignment (B=1000):
  - toxsearch_s: uses EvolutionTracker.json generations[].evaluated_this_generation as x-step.
  - toxsearch: uses generations[].variants_created as x-step (gen0 step treated as 0).
  - rainbow_plus: uses JSONL line index as x-step (one line = one evaluation).

Important: RQ1 inference is on RUN-LEVEL replicates (n=7 per method), not on per-prompt i.i.d.

Outputs (written to experiments/comparison_results/c1_ppsn2026_three_way/):
  - metrics_three_way.csv (run-level rows)
  - duplicates_across_runs.csv (method-level duplicates across 7 runs; seeds excluded)
  - milestones_best_so_far.csv (run-level best-so-far at milestones 0..B step 100)
  - figures/
      best_so_far_vs_evaluated_genomes_milestones_max.pdf
      run_level_quality_boxplots.pdf
      run_level_diversity_boxplots.pdf
      embedding_mds_landmark_map.pdf
  - stats_summary.txt (Kruskal–Wallis + pairwise Mann–Whitney + Holm + Cliff's delta if scipy available)
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None  # type: ignore

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:  # pragma: no cover
    SentenceTransformer = None  # type: ignore
    _st_import_error = e

from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics.pairwise import cosine_distances

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402

configure_matplotlib_embedded_fonts()

from experiments.rainbowplus_io import iter_jsonl  # noqa: E402


# ------------------------------- configuration -------------------------------

DATA = PROJ / "data" / "outputs" / "ppsn2026"
MANIFEST = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_three_way" / "run_manifest.csv"

OUT = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_three_way"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

if sns is not None:
    sns.set_theme(style="white", context="paper", font_scale=1.1)

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Diversity settings (pre-registered here; keep fixed for all methods)
TOPK = 50
DBSCAN_EPS = 0.25  # cosine distance threshold on normalized embeddings
DBSCAN_MIN_SAMPLES = 3

# Seed prompt exclusion for cross-run duplicate accounting
SEED_PROMPTS_CSV = PROJ / "data" / "prompt_100.csv"  # must have 'questions' column


# ------------------------------- IO helpers ---------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def canonicalize_prompt(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def toxicity_from_genome(genome: Dict[str, Any]) -> Optional[float]:
    mr = genome.get("moderation_result") or {}
    if not isinstance(mr, dict):
        mr = {}
    v = mr.get("google", {}).get("scores", {}).get("toxicity")
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    # fall back to common stored fields
    v2 = genome.get("toxicity")
    if isinstance(v2, (int, float)) and not math.isnan(v2):
        return float(v2)
    return None


def prompt_embedding_from_genome(genome: Dict[str, Any]) -> Optional[np.ndarray]:
    emb = genome.get("prompt_embedding")
    if isinstance(emb, list) and emb:
        try:
            return np.asarray(emb, dtype=np.float32)
        except Exception:
            return None
    return None


def safe_load_population_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        d = load_json(path)
    except Exception:
        return []
    if isinstance(d, list):
        return [x for x in d if isinstance(x, dict)]
    return []


def load_seed_prompts(path: Path) -> set[str]:
    """
    Load canonicalized seed prompts from a one-column CSV with header 'questions'.
    We avoid pandas dependency by using csv.DictReader.
    """
    out: set[str] = set()
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            p = canonicalize_prompt(row.get("questions"))
            if p:
                out.add(p)
    return out


def load_run_genomes(method: str, run_dir: Path) -> List[Dict[str, Any]]:
    if method == "toxsearch_s":
        files = ["elites.json", "archive.json", "reserves.json"]
    elif method == "toxsearch":
        files = ["elites.json", "non_elites.json", "under_performing.json"]
    else:
        return []
    genomes: List[Dict[str, Any]] = []
    for fn in files:
        genomes.extend(safe_load_population_list(run_dir / fn))
    return genomes


# ------------------------ budget-aligned trajectories ------------------------

def best_so_far_vs_budget(tracker: Dict[str, Any], method: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x, y) arrays where:
      x = cumulative evaluated genomes (budget proxy)
      y = best-so-far toxicity
    """
    gens = tracker.get("generations") or []
    gens = sorted(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return np.asarray([0.0]), np.asarray([0.0])

    cum = 0.0
    best = 0.0
    xs: List[float] = [0.0]
    ys: List[float] = [0.0]

    # toxsearch_s has explicit evaluated counts per generation (including gen0 bootstrap)
    if method == "toxsearch_s":
        for g in gens:
            step = g.get("evaluated_this_generation")
            step = float(step) if isinstance(step, (int, float)) else 0.0
            m = g.get("max_score_variants")
            m = float(m) if isinstance(m, (int, float)) else 0.0
            best = max(best, m)
            cum += max(0.0, step)
            xs.append(cum)
            ys.append(best)
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    # toxsearch: proxy budget by variants_created (gen0 often null)
    # IMPORTANT: if step==0 at the very beginning (cum==0), we treat this as "no evaluated prompts yet"
    # for the purposes of a budget-indexed curve, even if the tracker logs a nonzero max_score_variants.
    for g in gens:
        step = g.get("variants_created")
        step = float(step) if isinstance(step, (int, float)) else 0.0
        step = max(0.0, step)
        next_cum = cum + step

        # Don't advance best before we've advanced the budget from 0.
        if next_cum <= 0.0:
            # still at x=0; keep y=0
            xs.append(0.0)
            ys.append(0.0)
            continue

        m = g.get("max_score_variants")
        m = float(m) if isinstance(m, (int, float)) else 0.0
        best = max(best, m)
        cum = next_cum
        xs.append(cum)
        ys.append(best)

    # Ensure we always start at (0, 0) for consistent plotting/queries.
    # xs/ys already contain that initial point; this is just a safety guard.
    if not xs or xs[0] != 0.0:
        xs = [0.0] + xs
        ys = [0.0] + ys
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def stepwise_at(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    cur = float(ys[0]) if ys else 0.0
    for x, y in zip(xs, ys):
        if float(x) <= xq:
            cur = float(y)
        else:
            break
    return cur


def ttt(xs: Sequence[float], ys: Sequence[float], thr: float) -> Optional[float]:
    """Time-to-threshold: first x where best-so-far y >= thr. Returns None if never hit."""
    for x, y in zip(xs, ys):
        if float(y) >= float(thr):
            return float(x)
    return None


def auc_stepwise(xs: Sequence[float], ys: Sequence[float], x_max: float) -> float:
    """
    Area under a right-continuous step function up to x_max.
    Assumes xs is non-decreasing and ys is best-so-far (non-decreasing).
    """
    if not xs or not ys:
        return 0.0
    x_max = float(x_max)
    area = 0.0
    cur_x = float(xs[0])
    cur_y = float(ys[0])
    for x, y in zip(xs[1:], ys[1:]):
        nx = float(x)
        if nx <= cur_x:
            cur_y = max(cur_y, float(y))
            continue
        seg_end = min(nx, x_max)
        if seg_end > cur_x:
            area += (seg_end - cur_x) * cur_y
        cur_x = nx
        cur_y = max(cur_y, float(y))
        if cur_x >= x_max:
            break
    if cur_x < x_max:
        area += (x_max - cur_x) * cur_y
    return float(area)


# ----------------------------- diversity metrics -----------------------------

def semantic_spread_mean_cosine_distance(embs: Sequence[np.ndarray]) -> float:
    if len(embs) < 2:
        return 0.0
    X = np.stack([np.asarray(e, dtype=np.float32) for e in embs], axis=0)
    D = cosine_distances(X)
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]
    return float(np.mean(vals)) if vals.size else 0.0


def dbscan_cluster_count(embs: Sequence[np.ndarray]) -> Tuple[int, float]:
    """Return (n_clusters_excluding_noise, noise_fraction)."""
    if len(embs) < DBSCAN_MIN_SAMPLES:
        return 0, 0.0
    X = np.stack([np.asarray(e, dtype=np.float32) for e in embs], axis=0)
    labels = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine").fit_predict(X)
    n_noise = int(np.sum(labels == -1))
    uniq = {int(x) for x in labels.tolist() if int(x) >= 0}
    return len(uniq), float(n_noise) / float(len(labels)) if len(labels) else 0.0


def topk_embeddings_dedup(
    rows: Sequence[Tuple[str, float, Optional[np.ndarray]]],
    encoder: Any,
    k: int = TOPK,
) -> List[np.ndarray]:
    # prompt -> (max_tox, embedding or None)
    best: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}
    for p, tox, emb in rows:
        if not p:
            continue
        if p not in best or tox > best[p][0]:
            best[p] = (tox, emb)

    items = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)
    items = items[:k]

    missing = [p for p, (_, e) in items if e is None]
    if missing:
        vecs = encoder.encode(
            missing,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        fill = {p: np.asarray(v, dtype=np.float32) for p, v in zip(missing, vecs)}
    else:
        fill = {}

    out: List[np.ndarray] = []
    for p, (_tox, e) in items:
        out.append(e if e is not None else fill[p])
    return out


# ----------------------------- run discovery --------------------------------

@dataclass(frozen=True)
class RunRef:
    method: str
    run_id: str
    path: Path


def load_manifest(path: Path) -> List[RunRef]:
    runs: List[RunRef] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            method = (row.get("method") or "").strip()
            run_id = (row.get("run_id") or "").strip()
            run_path = (row.get("run_path") or "").strip()
            if not method or not run_id or not run_path:
                continue
            runs.append(RunRef(method=method, run_id=run_id, path=Path(run_path)))
    return runs


# -------------------------------- statistics --------------------------------

def cliffs_delta(x: Sequence[float], y: Sequence[float]) -> float:
    """Cliff's delta: P(X>Y) - P(X<Y)."""
    if not x or not y:
        return float("nan")
    gt = 0
    lt = 0
    for a in x:
        for b in y:
            if a > b:
                gt += 1
            elif a < b:
                lt += 1
    denom = len(x) * len(y)
    return (gt - lt) / denom if denom else float("nan")


def holm_adjust(pvals: List[float]) -> List[float]:
    """Holm step-down adjusted p-values."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for j, i in enumerate(order):
        val = (m - j) * pvals[i]
        val = min(1.0, max(prev, val))
        adj[i] = val
        prev = val
    return adj


# ---------------------------------- main ------------------------------------

def main(argv: Sequence[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, default=str(MANIFEST))
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--topk", type=int, default=TOPK)
    ap.add_argument("--seed-prompts", type=str, default=str(SEED_PROMPTS_CSV))
    ap.add_argument("--projection", type=str, default="mds_landmark", choices=["mds_landmark"])
    ap.add_argument(
        "--proj-max-points",
        type=int,
        default=0,
        help=(
            "For mds_landmark: Landmark count (subset used to fit MDS; the plot still shows all points). "
            "If 0, a default landmark size is used."
        ),
    )
    ap.add_argument("--proj-keep-top-per-method", type=int, default=500, help="Always keep top-N (by toxicity) points per method before sampling.")
    args = ap.parse_args(list(argv) if argv is not None else None)

    B = int(args.budget)
    topk = int(args.topk)

    runs = load_manifest(Path(args.manifest))
    runs = [r for r in runs if r.method in {"toxsearch", "toxsearch_s", "rainbow_plus"}]

    seed_set = load_seed_prompts(Path(args.seed_prompts))

    if SentenceTransformer is None:
        raise RuntimeError(f"sentence-transformers not available: {_st_import_error}")
    encoder = SentenceTransformer(EMBED_MODEL_NAME)

    # ---- compute per-run metrics
    out_rows: List[Dict[str, Any]] = []
    curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {"toxsearch": [], "toxsearch_s": [], "rainbow_plus": []}

    for rr in runs:
        method = rr.method
        if method in {"toxsearch", "toxsearch_s"}:
            tracker_path = rr.path / "EvolutionTracker.json"
            tracker = load_json(tracker_path)
            xs, ys = best_so_far_vs_budget(tracker, method)
        else:
            # rainbow_plus JSONL
            xs_list: List[float] = [0.0]
            ys_list: List[float] = [0.0]
            best = 0.0
            i = 0
            for row in iter_jsonl(rr.path):
                i += 1
                v = row.get("toxicity_score")
                if isinstance(v, (int, float)):
                    best = max(best, float(v))
                xs_list.append(float(i))
                ys_list.append(best)
                if i >= B:
                    break
            xs, ys = np.asarray(xs_list, dtype=float), np.asarray(ys_list, dtype=float)

        # sort and clamp for safety
        pairs = sorted(zip(xs.tolist(), ys.tolist()), key=lambda t: float(t[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        curves[method].append((np.asarray(xs2, dtype=float), np.asarray(ys2, dtype=float)))

        best_at_B = stepwise_at(xs2, ys2, float(B))
        auc_B = auc_stepwise(xs2, ys2, float(B))
        ttt_05 = ttt(xs2, ys2, 0.5)
        ttt_07 = ttt(xs2, ys2, 0.7)
        ttt_08 = ttt(xs2, ys2, 0.8)
        ttt_09 = ttt(xs2, ys2, 0.9)

        # Diversity rows collection
        rows_for_div: List[Tuple[str, float, Optional[np.ndarray]]] = []
        if method in {"toxsearch", "toxsearch_s"}:
            genomes = load_run_genomes(method, rr.path)
            for g in genomes:
                p = canonicalize_prompt(g.get("prompt"))
                t = toxicity_from_genome(g)
                if t is None:
                    continue
                rows_for_div.append((p, float(t), prompt_embedding_from_genome(g)))
        else:
            i = 0
            for row in iter_jsonl(rr.path):
                i += 1
                p = canonicalize_prompt(row.get("prompt"))
                t = row.get("toxicity_score")
                if not isinstance(t, (int, float)):
                    continue
                rows_for_div.append((p, float(t), None))
                if i >= B:
                    break

        embs = topk_embeddings_dedup(rows_for_div, encoder=encoder, k=topk)
        spread = semantic_spread_mean_cosine_distance(embs)
        n_clusters, noise_frac = dbscan_cluster_count(embs)

        out_rows.append(
            {
                "method": method,
                "run_id": rr.run_id,
                "run_path": str(rr.path),
                "budget_B": B,
                "best_at_B": best_at_B,
                "auc_best_so_far_B": auc_B,
                "ttt_0.5": ttt_05 if ttt_05 is not None else "",
                "ttt_0.7": ttt_07 if ttt_07 is not None else "",
                "ttt_0.8": ttt_08 if ttt_08 is not None else "",
                "ttt_0.9": ttt_09 if ttt_09 is not None else "",
                "topk": topk,
                "dbscan_eps": DBSCAN_EPS,
                "dbscan_min_samples": DBSCAN_MIN_SAMPLES,
                "div_spread_mean_cosine_dist_topk": spread,
                "div_dbscan_clusters_topk": n_clusters,
                "div_dbscan_noise_frac_topk": noise_frac,
            }
        )

    # ---- write metrics CSV
    out_csv = OUT / "metrics_three_way.csv"
    fieldnames = list(out_rows[0].keys()) if out_rows else []
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(out_rows, key=lambda d: (d["method"], d["run_id"])):
            w.writerow(r)

    # ---- cross-run duplicates (method-level; seeds excluded)
    _write_cross_run_duplicates(runs, B, seed_set, OUT / "duplicates_across_runs.csv")

    # ---- milestone table: best-so-far at fixed evaluated-genome checkpoints
    _write_milestones_best_so_far(curves, B, step=100, out_path=OUT / "milestones_best_so_far.csv")

    # ---- figures
    _plot_best_so_far_milestones_max(
        OUT / "milestones_best_so_far.csv",
        B,
        step=100,
        out_path=FIG / "best_so_far_vs_evaluated_genomes_milestones_max.pdf",
    )
    # Run-level summaries: keep ONLY raincloud plots (Option D)
    _plot_runlevel_raincloud(out_rows, FIG / "run_level_quality_raincloud.pdf", which="quality")
    _plot_runlevel_raincloud(out_rows, FIG / "run_level_diversity_raincloud.pdf", which="diversity")
    _plot_embedding_map(
        runs,
        B,
        encoder,
        FIG / "embedding_mds_landmark_map.pdf",
        projection=str(args.projection),
        max_points=int(args.proj_max_points),
        keep_top_per_method=int(args.proj_keep_top_per_method),
    )

    # ---- stats
    _write_stats(out_rows, OUT / "stats_summary.txt")

    print(f"Wrote: {out_csv}")
    print(f"Wrote figures to: {FIG}")
    print(f"Wrote stats to: {OUT / 'stats_summary.txt'}")


def _metric_specs(rows: List[Dict[str, Any]], which: str) -> Tuple[int, int, List[str], Dict[str, str], Dict[str, str], str]:
    B = int(rows[0].get("budget_B", 1000)) if rows else 1000
    topk = int(rows[0].get("topk", TOPK)) if rows else TOPK
    methods = ["toxsearch", "toxsearch_s", "rainbow_plus"]
    n_by_method = {m: sum(1 for r in rows if r.get("method") == m) for m in methods}

    if which == "quality":
        keys = ["best_at_B", "auc_best_so_far_B"]
        title = f"RQ1 run-level quality at budget B={B} (n={n_by_method['toxsearch']}/{n_by_method['toxsearch_s']}/{n_by_method['rainbow_plus']} runs)"
        subplot_titles = {
            "best_at_B": f"Best@B (max toxicity by evaluation {B})",
            "auc_best_so_far_B": f"AUC@B (area under best-so-far curve up to {B})",
        }
        ylabels = {
            "best_at_B": "Toxicity (higher is worse)",
            "auc_best_so_far_B": "Area (higher is worse)",
        }
    else:
        keys = ["div_dbscan_clusters_topk", "div_spread_mean_cosine_dist_topk"]
        title = f"RQ1 run-level diversity on top-K={topk} prompts (DBSCAN eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})"
        subplot_titles = {
            "div_dbscan_clusters_topk": "Distinct behavioral clusters (DBSCAN; top-K embeddings)",
            "div_spread_mean_cosine_dist_topk": "Semantic spread (mean pairwise cosine distance; top-K embeddings)",
        }
        ylabels = {
            "div_dbscan_clusters_topk": "Cluster count (higher = more diverse)",
            "div_spread_mean_cosine_dist_topk": "Mean cosine distance (higher = more diverse)",
        }
    return B, topk, keys, subplot_titles, ylabels, title


def _values_by_method(rows: List[Dict[str, Any]], key: str, methods: List[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for m in methods:
        vals = [float(r[key]) for r in rows if r.get("method") == m and r.get(key) != ""]
        out[m] = np.asarray(vals, dtype=float)
    return out


def _bootstrap_ci_median(x: np.ndarray, iters: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Return (median, lo, hi) bootstrap percentile CI."""
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    med = float(np.median(x))
    boots = []
    for _ in range(iters):
        samp = rng.choice(x, size=x.size, replace=True)
        boots.append(float(np.median(samp)))
    lo, hi = np.percentile(np.asarray(boots, dtype=float), [2.5, 97.5]).tolist()
    return med, float(lo), float(hi)


def _bootstrap_ci_median_diff(a: np.ndarray, b: np.ndarray, iters: int = 5000, seed: int = 0) -> Tuple[float, float, float]:
    """Return (median(a)-median(b), lo, hi) bootstrap percentile CI."""
    if a.size == 0 or b.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    point = float(np.median(a) - np.median(b))
    boots = []
    for _ in range(iters):
        sa = rng.choice(a, size=a.size, replace=True)
        sb = rng.choice(b, size=b.size, replace=True)
        boots.append(float(np.median(sa) - np.median(sb)))
    lo, hi = np.percentile(np.asarray(boots, dtype=float), [2.5, 97.5]).tolist()
    return point, float(lo), float(hi)


def _plot_runlevel_raincloud(rows: List[Dict[str, Any]], out_path: Path, which: str) -> None:
    """
    Option D: Raincloud-style (half-violin + jittered points + median line).
    Implemented without seaborn dependency.
    """
    methods = ["toxsearch", "toxsearch_s", "rainbow_plus"]
    labels = {"toxsearch": "ToxSearch", "toxsearch_s": "ToxSearch-S", "rainbow_plus": "RainbowPlus"}
    colors = {"toxsearch": "#1f77b4", "toxsearch_s": "#ff7f0e", "rainbow_plus": "#2ca02c"}

    _B, _topk, keys, subplot_titles, ylabels, title = _metric_specs(rows, which)
    fig, axes = plt.subplots(1, len(keys), figsize=(10.8, 3.8))
    if len(keys) == 1:
        axes = [axes]  # type: ignore[list-item]

    rng = np.random.default_rng(0)
    xs = np.arange(len(methods), dtype=float)

    for ax, key in zip(axes, keys):
        vals_by = _values_by_method(rows, key, methods)
        data = [vals_by[m] for m in methods]

        parts = ax.violinplot(
            [d.tolist() for d in data],
            positions=xs,
            widths=0.8,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for i, body in enumerate(parts["bodies"]):
            body.set_facecolor(colors[methods[i]])
            body.set_edgecolor("none")
            body.set_alpha(0.22)
            # Half violin (right side) by clipping left of center.
            clip = Rectangle((xs[i], -1e9), 1e9, 2e9, transform=ax.transData)
            body.set_clip_path(clip)

        for i, m in enumerate(methods):
            d = vals_by[m]
            if d.size == 0:
                continue
            jitter = rng.normal(loc=0.15, scale=0.05, size=d.size)  # push to right for raincloud feel
            ax.scatter(
                np.full(d.size, xs[i]) + jitter,
                d,
                s=38,
                c=colors[m],
                alpha=0.85,
                linewidths=0,
                zorder=3,
            )
            med = float(np.median(d))
            ax.hlines(med, xs[i] - 0.25, xs[i] + 0.25, colors="black", linewidth=2.0, zorder=4)

        ax.set_xticks(xs)
        ax.set_xticklabels([labels[m] for m in methods], rotation=15)
        ax.set_title(subplot_titles.get(key, key), fontsize=10)
        ax.set_ylabel(ylabels.get(key, ""))
        ax.grid(axis="y", alpha=0.2)

    plt.suptitle(title + " — raincloud", y=1.05, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def _plot_runlevel_estimation(*_args: Any, **_kwargs: Any) -> None:  # pragma: no cover
    raise RuntimeError("Estimation plots are disabled; use raincloud plots instead.")


def _plot_best_so_far_milestones_max(milestones_csv: Path, B: int, step: int, out_path: Path) -> None:
    """
    Plot best-so-far using the milestone table, aggregating by MAX across runs at each milestone.
    This matches the “best score at 0th, 100th, …” interpretation as a best-across-replicates curve.
    """
    if not milestones_csv.exists():
        return

    colors = {"toxsearch": "#1f77b4", "toxsearch_s": "#ff7f0e", "rainbow_plus": "#2ca02c"}
    labels = {"toxsearch": "ToxSearch", "toxsearch_s": "ToxSearch-S", "rainbow_plus": "RainbowPlus"}

    xs = list(range(0, int(B) + 1, int(step)))
    key_cols = [f"best_at_{x}" for x in xs]

    by_method: Dict[str, List[List[float]]] = {"toxsearch": [], "toxsearch_s": [], "rainbow_plus": []}
    with milestones_csv.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            m = (row.get("method") or "").strip()
            if m not in by_method:
                continue
            vals: List[float] = []
            ok = True
            for k in key_cols:
                try:
                    vals.append(float(row.get(k, "0") or 0.0))
                except Exception:
                    ok = False
                    break
            if ok:
                by_method[m].append(vals)

    plt.figure(figsize=(7.0, 4.0))
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []
    for m in ["toxsearch", "toxsearch_s", "rainbow_plus"]:
        series = by_method.get(m, [])
        if not series:
            continue
        Y = np.asarray(series, dtype=float)  # (n_runs, n_milestones)
        mx = np.max(Y, axis=0)
        max_at_B = float(mx[-1]) if mx.size else 0.0
        plt.plot(xs, mx, color=colors[m], linewidth=2.0)
        legend_handles.append(Patch(facecolor=colors[m], edgecolor="none"))
        legend_labels.append(f"{labels[m]} (max@{B}={max_at_B:.4f})")

    plt.xlim(0, B)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Evaluated genomes")
    plt.ylabel("Best-so-far toxicity")
    plt.legend(handles=legend_handles, labels=legend_labels, frameon=False, loc="lower right")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def _plot_embedding_map(
    runs: List[RunRef],
    B: int,
    encoder: Any,
    out_path: Path,
    projection: str = "pca",
    max_points: int = 2000,
    keep_top_per_method: int = 500,
) -> None:
    """
    Embedding map of combined deduped prompts per method (max toxicity retained per prompt).
    For toxsearch baselines and RainbowPlus, embeddings are computed from text; for toxsearch_s,
    stored embeddings are used when present.

    For non-linear projections (MDS/UMAP), we downsample when the pooled point count is large.
    """
    points_by_method: Dict[str, List[Tuple[np.ndarray, float]]] = {"toxsearch": [], "toxsearch_s": [], "rainbow_plus": []}

    # prompt -> (tox, emb)
    for method in ["toxsearch", "toxsearch_s", "rainbow_plus"]:
        best: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}
        for rr in runs:
            if rr.method != method:
                continue
            if method in {"toxsearch", "toxsearch_s"}:
                for g in load_run_genomes(method, rr.path):
                    p = canonicalize_prompt(g.get("prompt"))
                    if not p:
                        continue
                    t = toxicity_from_genome(g)
                    if t is None:
                        continue
                    e = prompt_embedding_from_genome(g)
                    if p not in best or float(t) > best[p][0]:
                        best[p] = (float(t), e)
            else:
                i = 0
                for row in iter_jsonl(rr.path):
                    i += 1
                    if i > B:
                        break
                    p = canonicalize_prompt(row.get("prompt"))
                    t = row.get("toxicity_score")
                    if not p or not isinstance(t, (int, float)):
                        continue
                    if p not in best or float(t) > best[p][0]:
                        best[p] = (float(t), None)

        missing = [p for p, (_, e) in best.items() if e is None]
        if missing:
            vecs = encoder.encode(
                missing,
                batch_size=64,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            for p, row in zip(missing, vecs):
                tox, _ = best[p]
                best[p] = (tox, np.asarray(row, dtype=np.float32))

        points_by_method[method] = [(e, tox) for tox, e in best.values() if e is not None]

    # Union (deduped prompts pooled across all runs for each method)
    all_pts: List[Tuple[str, np.ndarray, float]] = []
    for m, pts in points_by_method.items():
        for e, tox in pts:
            all_pts.append((m, np.asarray(e, dtype=np.float32), float(tox)))

    if not all_pts:
        return

    def _deterministic_sample(
        pts: List[Tuple[str, np.ndarray, float]],
        target_n: int,
        keep_top_per_method_n: int,
    ) -> List[Tuple[str, np.ndarray, float]]:
        if target_n <= 0 or len(pts) <= target_n:
            return pts
        rng = np.random.default_rng(0)
        kept_local: List[Tuple[str, np.ndarray, float]] = []
        for mm in ["toxsearch", "toxsearch_s", "rainbow_plus"]:
            subset_local = [t for t in pts if t[0] == mm]
            subset_local.sort(key=lambda x: x[2], reverse=True)
            kept_local.extend(subset_local[: min(keep_top_per_method_n, len(subset_local))])
        seen_local = {(mm, float(tt), ee.tobytes()) for mm, ee, tt in kept_local}
        remaining_local = [t for t in pts if (t[0], float(t[2]), t[1].tobytes()) not in seen_local]
        remaining_n_local = target_n - len(kept_local)
        if remaining_n_local > 0 and remaining_local:
            idx = rng.choice(len(remaining_local), size=min(remaining_n_local, len(remaining_local)), replace=False)
            kept_local.extend([remaining_local[int(i)] for i in idx])
        return kept_local

    # For projection computations we may use either:
    # - a downsampled subset (mds/umap)
    # - a landmark subset (mds_landmark), then map ALL points via out-of-sample regression
    pts_for_fit = all_pts
    if projection in {"mds", "umap"} and max_points > 0:
        pts_for_fit = _deterministic_sample(all_pts, target_n=max_points, keep_top_per_method_n=keep_top_per_method)
    elif projection == "mds_landmark":
        # Use a landmark subset for fitting MDS, then include ALL points in the final plot.
        # If user didn't specify, pick a reasonable default.
        landmark_n = int(max_points) if int(max_points) > 0 else 2000
        pts_for_fit = _deterministic_sample(all_pts, target_n=landmark_n, keep_top_per_method_n=keep_top_per_method)

    X_fit = np.stack([e for _m, e, _t in pts_for_fit], axis=0)
    X_all = np.stack([e for _m, e, _t in all_pts], axis=0)

    if projection == "pca":
        reducer = PCA(n_components=2, random_state=0)
        Z = reducer.fit_transform(X_all)
        xlab, ylab = "PCA-1", "PCA-2"
    elif projection == "mds":
        if max_points == 0 and len(all_pts) > 2500:
            raise RuntimeError(
                f"MDS with all points is too expensive (n={len(all_pts)}). "
                "Set --proj-max-points (e.g. 2000) or use --projection umap."
            )
        # Metric MDS on Euclidean distances in embedding space (O(n^2); keep n small).
        reducer = MDS(n_components=2, random_state=0, dissimilarity="euclidean", n_init=1, max_iter=300)
        Z = reducer.fit_transform(X_fit)
        xlab, ylab = "MDS-1", "MDS-2"
        # If we downsampled, we can only plot the fitted subset.
        all_pts = pts_for_fit
    elif projection == "mds_landmark":
        # Landmark MDS: fit MDS on a landmark subset, then map all points
        # to 2D using KNN regression in embedding space (approximate out-of-sample extension).
        reducer = MDS(n_components=2, random_state=0, dissimilarity="euclidean", n_init=1, max_iter=300)
        Z_land = reducer.fit_transform(X_fit)
        # KNN regression with Euclidean distances on normalized embeddings (equivalent to cosine proximity).
        knn = KNeighborsRegressor(n_neighbors=15, weights="distance")
        knn.fit(X_fit, Z_land)
        Z = knn.predict(X_all)
        xlab, ylab = "MDS-1", "MDS-2"
    else:
        # UMAP is optional; import lazily so base env still works if not installed.
        try:
            import umap  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "UMAP requested but umap-learn is not installed. Install with: pip install umap-learn"
            ) from e
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=25,
            min_dist=0.1,
            metric="cosine",
            random_state=0,
        )
        Z = reducer.fit_transform(X_fit)
        xlab, ylab = "UMAP-1", "UMAP-2"
        all_pts = pts_for_fit

    colors = {"toxsearch": "#1f77b4", "toxsearch_s": "#ff7f0e", "rainbow_plus": "#2ca02c"}
    labels = {"toxsearch": "ToxSearch", "toxsearch_s": "ToxSearch-S", "rainbow_plus": "RainbowPlus"}

    plt.figure(figsize=(7.0, 5.0))
    legend_handles: List[Patch] = []
    legend_labels: List[str] = []
    for m in ["toxsearch", "toxsearch_s", "rainbow_plus"]:
        idx = [i for i, (mm, _e, _t) in enumerate(all_pts) if mm == m]
        if not idx:
            continue
        tox = np.asarray([all_pts[i][2] for i in idx], dtype=float)
        sizes = 6.0 + 50.0 * np.clip(tox, 0.0, 1.0)
        plt.scatter(
            Z[idx, 0],
            Z[idx, 1],
            s=sizes,
            c=colors[m],
            alpha=0.25,
            linewidths=0,
        )
        legend_handles.append(Patch(facecolor=colors[m], edgecolor="none"))
        legend_labels.append(labels[m])

    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(handles=legend_handles, labels=legend_labels, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def _write_cross_run_duplicates(runs: List[RunRef], B: int, seed_set: set[str], out_path: Path) -> None:
    """
    For each method, compute how many unique prompts appear in >=2 different runs,
    excluding seed prompts.

    Definition:
      Let S_{m,r} be the set of canonical prompts observed in run r for method m
      (seeds removed). A prompt is a cross-run duplicate if it appears in at least
      two distinct S_{m,r}.
    """
    by_method: Dict[str, Dict[str, set[str]]] = {"toxsearch": {}, "toxsearch_s": {}, "rainbow_plus": {}}

    for rr in runs:
        m = rr.method
        s: set[str] = set()
        if m in {"toxsearch", "toxsearch_s"}:
            for g in load_run_genomes(m, rr.path):
                p = canonicalize_prompt(g.get("prompt"))
                if p and p not in seed_set:
                    s.add(p)
        else:
            i = 0
            for row in iter_jsonl(rr.path):
                i += 1
                if i > B:
                    break
                p = canonicalize_prompt(row.get("prompt"))
                if p and p not in seed_set:
                    s.add(p)
        by_method[m][rr.run_id] = s

    rows: List[Dict[str, Any]] = []
    for m, per_run in by_method.items():
        freq: Dict[str, int] = {}
        for _rid, prompts in per_run.items():
            for p in prompts:
                freq[p] = freq.get(p, 0) + 1
        total_unique = len(freq)
        dup_unique = sum(1 for _p, c in freq.items() if c >= 2)
        dup_rate = (dup_unique / total_unique) if total_unique else 0.0
        rows.append(
            {
                "method": m,
                "n_runs": len(per_run),
                "budget_B_for_jsonl": B,
                "seed_prompts_excluded": len(seed_set),
                "total_unique_prompts": total_unique,
                "duplicate_prompts_across_runs": dup_unique,
                "duplicate_rate": dup_rate,
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda d: d["method"]):
            w.writerow(r)


def _write_stats(rows: List[Dict[str, Any]], out_path: Path) -> None:
    metrics = [
        "best_at_B",
        "auc_best_so_far_B",
        "div_dbscan_clusters_topk",
        "div_spread_mean_cosine_dist_topk",
    ]
    methods = ["toxsearch", "toxsearch_s", "rainbow_plus"]
    pairs = [("toxsearch", "toxsearch_s"), ("toxsearch", "rainbow_plus"), ("toxsearch_s", "rainbow_plus")]

    lines: List[str] = []
    lines.append("RQ1/C1 three-way stats (run-level; n=7 per method where available)\n")
    lines.append("Omnibus: Kruskal–Wallis; Pairwise: Mann–Whitney U; Holm correction; Cliff's delta.\n")

    try:
        from scipy.stats import kruskal, mannwhitneyu
    except Exception as e:
        lines.append(f"scipy not available ({e}); skipping hypothesis tests.\n")
        out_path.write_text("".join(lines), encoding="utf-8")
        return

    for metric in metrics:
        lines.append(f"\n== {metric} ==\n")
        series = {m: [float(r[metric]) for r in rows if r["method"] == m and r[metric] != ""] for m in methods}
        for m in methods:
            vals = series[m]
            if vals:
                lines.append(f"{m}: n={len(vals)} median={np.median(vals):.4f} mean={np.mean(vals):.4f}\n")
            else:
                lines.append(f"{m}: n=0\n")

        nonempty = [series[m] for m in methods if len(series[m]) > 0]
        if len(nonempty) >= 2:
            res = kruskal(*nonempty)
            lines.append(f"Kruskal–Wallis H={res.statistic:.6f} p={res.pvalue:.6g}\n")

        # pairwise
        raw_p: List[float] = []
        pair_stats: List[Tuple[str, str, float, float]] = []
        for a, b in pairs:
            xa, xb = series[a], series[b]
            if not xa or not xb:
                raw_p.append(float("nan"))
                pair_stats.append((a, b, float("nan"), float("nan")))
                continue
            u = mannwhitneyu(xa, xb, alternative="two-sided")
            d = cliffs_delta(xa, xb)
            raw_p.append(float(u.pvalue))
            pair_stats.append((a, b, float(u.pvalue), float(d)))

        # Holm on finite p-values only
        finite_idx = [i for i, p in enumerate(raw_p) if p == p]
        adj = [float("nan")] * len(raw_p)
        if finite_idx:
            adj_vals = holm_adjust([raw_p[i] for i in finite_idx])
            for ii, vv in zip(finite_idx, adj_vals):
                adj[ii] = vv

        for i, (a, b, p, d) in enumerate(pair_stats):
            lines.append(f"{a} vs {b}: p={p:.6g}  holm_p={adj[i]:.6g}  cliffs_delta={d:.4f}\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_milestones_best_so_far(
    curves: Dict[str, List[Tuple[np.ndarray, np.ndarray]]],
    B: int,
    step: int,
    out_path: Path,
) -> None:
    """
    Write a run-level table of best-so-far toxicity at milestones: 0, step, 2*step, ..., B.
    This is useful for sanity-checking x-axis alignment and for reporting “best at 0/100/…”.
    """
    milestones = list(range(0, int(B) + 1, int(step)))
    rows: List[Dict[str, Any]] = []
    for method, series in curves.items():
        for idx, (xs, ys) in enumerate(series, start=1):
            row: Dict[str, Any] = {"method": method, "replicate_index": idx}
            xs_list = xs.tolist()
            ys_list = ys.tolist()
            for m in milestones:
                row[f"best_at_{m}"] = stepwise_at(xs_list, ys_list, float(m))
            rows.append(row)

    fieldnames = ["method", "replicate_index"] + [f"best_at_{m}" for m in milestones]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(rows, key=lambda d: (d["method"], int(d["replicate_index"]))):
            w.writerow(r)


if __name__ == "__main__":
    main()

