#!/usr/bin/env python3
"""
C1 (two-way): ToxSearch vs ToxSearch-S — 3 publication-grade figures (no RainbowPlus).

Figures written to:
  experiments/comparison_results/c1_ppsn2026_two_way/figures/

Artifacts written to:
  experiments/comparison_results/c1_ppsn2026_two_way/

Outputs:
  1) trajectory_over_evaluated_genomes.pdf
     - milestones 0..1000 step 100
     - MAX across runs at each milestone (per method)
  2) diversity_dbscan_clusters_top50.pdf
     - 2D embedding map (PCA) of full combined population (all runs), blue vs orange

Run:
  python experiments/comparison_results/c1_two_way_figures.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None  # type: ignore

from sentence_transformers import SentenceTransformer

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances


# ------------------------------- configuration -------------------------------

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402

configure_matplotlib_embedded_fonts()

DATA = PROJ / "data" / "outputs" / "ppsn2026"
TOXSEARCH_DIR = DATA / "toxsearch"
TOXSEARCH_S_DIR = DATA / "toxsearch_s"

OUT = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_two_way"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

# publication-grade PDF font embedding
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

# Seaborn theme if available
if sns is not None:
    sns.set_theme(style="white", context="paper", font_scale=1.1)

TOPK_EMB = 50
# Same model family as ToxSearch-S logs; baseline JSON often omits prompt_embedding — we encode text.
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ------------------------------- IO/helpers ----------------------------------

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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


def discover_runs(method: str, root: Path) -> List[Tuple[str, str, Path]]:
    runs: List[Tuple[str, str, Path]] = []
    if not root.exists():
        return runs
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        if (p / "EvolutionTracker.json").exists():
            runs.append((method, p.name, p))
    return runs


def canonicalize_prompt(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def toxicity(genome: Dict[str, Any]) -> Optional[float]:
    mr = genome.get("moderation_result") or {}
    if not isinstance(mr, dict):
        mr = {}
    v = mr.get("google", {}).get("scores", {}).get("toxicity")
    if isinstance(v, (int, float)) and not math.isnan(v):
        return float(v)
    return None


def prompt_embedding(genome: Dict[str, Any]) -> Optional[np.ndarray]:
    emb = genome.get("prompt_embedding")
    if isinstance(emb, list) and emb:
        try:
            return np.asarray(emb, dtype=np.float32)
        except Exception:
            return None
    return None


# ------------------------------- figure 1 -----------------------------------

def best_so_far_vs_budget(tracker: Dict[str, Any], method: str) -> Tuple[np.ndarray, np.ndarray]:
    gens = tracker.get("generations") or []
    gens = sorted(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return np.asarray([0.0]), np.asarray([0.0])

    cum = 0.0
    run = 0.0
    x: List[float] = [0.0]
    y: List[float] = [0.0]

    if method == "toxsearch_s":
        g0 = gens[0]
        step0 = g0.get("evaluated_this_generation")
        step0 = float(step0) if isinstance(step0, (int, float)) else 0.0
        m0 = g0.get("max_score_variants")
        m0 = float(m0) if isinstance(m0, (int, float)) else 0.0
        run = max(run, m0)
        cum += max(0.0, step0)
        x.append(cum)  # typically 100
        y.append(run)
        rest = gens[1:]
    else:
        rest = gens

    for g in rest:
        m = g.get("max_score_variants")
        m = float(m) if isinstance(m, (int, float)) else 0.0
        run = max(run, m)

        if method == "toxsearch_s":
            step = g.get("evaluated_this_generation")
            step = float(step) if isinstance(step, (int, float)) else 0.0
        else:
            step = g.get("variants_created")
            step = float(step) if isinstance(step, (int, float)) else 0.0

        cum += max(0.0, step)
        if method == "toxsearch" and cum == 0.0:
            continue
        x.append(cum)
        y.append(run)

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def stepwise_at(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    cur = float(ys[0]) if ys else 0.0
    for x, y in zip(xs, ys):
        if float(x) <= xq:
            cur = float(y)
        else:
            break
    return cur


def plot_milestone_max_across_runs(
    runs: List[Tuple[str, str, Path]],
    out_path: Path,
    milestones: Sequence[int] = tuple([0] + list(range(100, 1100, 100))),
) -> None:
    by_method: Dict[str, List[Dict[int, float]]] = {}

    for method, _, run_dir in runs:
        tracker = load_json(run_dir / "EvolutionTracker.json")
        xs, ys = best_so_far_vs_budget(tracker, method)
        pairs = sorted(zip(xs.tolist(), ys.tolist()), key=lambda t: float(t[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        vals = {int(m): stepwise_at(xs2, ys2, float(m)) for m in milestones}
        by_method.setdefault(method, []).append(vals)

    plt.figure(figsize=(7.0, 4.0))
    for method in ["toxsearch", "toxsearch_s"]:
        if method not in by_method or not by_method[method]:
            continue
        y_max = [max(v[m] for v in by_method[method]) for m in milestones]
        plt.plot(list(milestones), y_max, marker="o", linewidth=2.5, label=method)

    plt.xlabel("Evaluated genomes")
    plt.ylabel("Best-so-far")
    plt.xlim(min(milestones), max(milestones))
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def plot_violin_box(
    values_by_method: Dict[str, Sequence[float]],
    out_path: Path,
    ylabel: str,
    ylim: Tuple[float, float] = (0.0, 1.0),
) -> None:
    methods = ["toxsearch", "toxsearch_s"]
    xs: List[str] = []
    ys: List[float] = []
    for m in methods:
        for v in values_by_method.get(m, []):
            xs.append(m)
            ys.append(float(v))

    plt.figure(figsize=(5.5, 3.8))
    if sns is not None:
        import pandas as pd  # local import to keep top deps minimal

        df = pd.DataFrame({"method": xs, "value": ys})
        ax = sns.violinplot(data=df, x="method", y="value", inner=None, cut=0)
        sns.boxplot(data=df, x="method", y="value", width=0.25, showcaps=True, boxprops={"zorder": 2}, ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel(ylabel)
    else:
        data = [values_by_method.get(m, []) for m in methods]
        plt.boxplot(data, labels=methods, showfliers=False)
        plt.ylabel(ylabel)

    plt.ylim(*ylim)
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()

# ------------------------------- figure 3 -----------------------------------

def load_run_genomes(method: str, run_dir: Path) -> List[Dict[str, Any]]:
    if method == "toxsearch_s":
        files = ["elites.json", "archive.json", "reserves.json"]
    else:
        files = ["elites.json", "non_elites.json", "under_performing.json"]
    genomes: List[Dict[str, Any]] = []
    for fn in files:
        genomes.extend(safe_load_population_list(run_dir / fn))
    return genomes


def topk_embeddings_dedup(genomes: Sequence[Dict[str, Any]], k: int) -> List[np.ndarray]:
    seen: set[str] = set()
    scored: List[Tuple[float, np.ndarray]] = []
    for g in genomes:
        p = canonicalize_prompt(g.get("prompt"))
        if not p or p in seen:
            continue
        t = toxicity(g)
        emb = prompt_embedding(g)
        if t is None or emb is None:
            continue
        seen.add(p)
        scored.append((float(t), emb))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [e for _, e in scored[:k]]

def gather_combined_population_points(
    method: str,
    run_tuples: Sequence[Tuple[str, str, Path]],
    encoder: Any,
) -> List[Tuple[np.ndarray, float]]:
    """
    All runs for `method` combined: load whole population files, dedupe by canonical prompt,
    keep max toxicity if the same prompt appears in multiple runs.

    Baseline ToxSearch outputs typically omit `prompt_embedding`; those prompts are encoded
    with `encoder` so both methods appear in the embedding map.
    """
    # prompt -> (max_toxicity, embedding or None if missing in JSON)
    best: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}
    for m, _run_id, run_dir in run_tuples:
        if m != method:
            continue
        for g in load_run_genomes(method, run_dir):
            p = canonicalize_prompt(g.get("prompt"))
            if not p:
                continue
            t = toxicity(g)
            if t is None:
                continue
            t = float(t)
            emb = prompt_embedding(g)
            emb_arr = np.asarray(emb, dtype=np.float32) if emb is not None else None
            if p not in best or t > best[p][0]:
                best[p] = (t, emb_arr)

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

    return [(e, tox) for tox, e in best.values() if e is not None]


def semantic_spread_mean_cosine_distance(embs: Sequence[np.ndarray]) -> float:
    """
    Diversity as semantic spread: mean pairwise cosine distance among embeddings.
    Range: [0, 2] in theory for cosine distance, but typically [0, 1] for normalized sentence embeddings.
    """
    if len(embs) < 2:
        return 0.0
    X = np.stack([np.asarray(e, dtype=np.float32) for e in embs], axis=0)
    D = cosine_distances(X)
    iu = np.triu_indices(D.shape[0], k=1)
    vals = D[iu]
    if vals.size == 0:
        return 0.0
    return float(np.mean(vals))


def plot_combined_embedding_map(
    points_by_method: Dict[str, List[Tuple[np.ndarray, float]]],
    out_path: Path,
) -> None:
    """
    Diversity visualization: 2D linear projection (PCA) of *all* prompt embeddings in a shared space.
      - toxsearch: blue; toxsearch_s: orange
      - point size scales with toxicity
    PCA is used so the full combined population (all runs) remains tractable; MDS does not scale to large N.
    """
    methods = ["toxsearch", "toxsearch_s"]
    colors = {"toxsearch": "#1f77b4", "toxsearch_s": "#ff7f0e"}  # blue, orange
    labels = {"toxsearch": "ToxSearch", "toxsearch_s": "ToxSearch-S"}

    X_list: List[np.ndarray] = []
    tox_list: List[float] = []
    method_list: List[str] = []
    for m in methods:
        for emb, tox in points_by_method.get(m, []):
            X_list.append(np.asarray(emb, dtype=np.float32))
            tox_list.append(float(tox))
            method_list.append(m)

    if not X_list:
        return

    X = np.stack(X_list, axis=0)
    coords = PCA(n_components=2, random_state=0).fit_transform(X)

    tox = np.clip(np.asarray(tox_list, dtype=float), 0.0, 1.0)
    size = 3.0 + 14.0 * tox

    plt.figure(figsize=(7.0, 5.0))
    for m in methods:
        idx = [i for i, mm in enumerate(method_list) if mm == m]
        if not idx:
            continue
        plt.scatter(
            coords[idx, 0],
            coords[idx, 1],
            s=size[idx],
            c=colors[m],
            alpha=0.55,
            edgecolors="none",
            label=labels[m],
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(False)
    plt.legend(frameon=False, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


# ------------------------------- orchestration ------------------------------

def write_run_manifest(runs: List[Tuple[str, str, Path]], out_csv: Path) -> None:
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method", "run_id", "run_path"])
        for method, run_id, run_dir in runs:
            w.writerow([method, run_id, str(run_dir)])


def main() -> int:
    runs: List[Tuple[str, str, Path]] = []
    runs.extend(discover_runs("toxsearch", TOXSEARCH_DIR))
    runs.extend(discover_runs("toxsearch_s", TOXSEARCH_S_DIR))

    # manifest (todo: manifest)
    write_run_manifest(runs, OUT / "run_manifest.csv")

    # Figure 1 (todo: fig1)
    plot_milestone_max_across_runs(runs, FIG / "trajectory_over_evaluated_genomes.pdf")

    encoder = SentenceTransformer(EMBED_MODEL_NAME)

    # Metrics per run for diversity (todo: diversity_metrics_fig3)
    metrics_rows: List[Dict[str, Any]] = []
    spread_by_method: Dict[str, List[float]] = {"toxsearch": [], "toxsearch_s": []}

    for method, run_id, run_dir in runs:
        genomes = load_run_genomes(method, run_dir)
        embs = topk_embeddings_dedup(genomes, TOPK_EMB)
        spread = semantic_spread_mean_cosine_distance(embs)

        spread_by_method.setdefault(method, []).append(float(spread))

        metrics_rows.append(
            {
                "method": method,
                "run_id": run_id,
                "semantic_spread_top50": float(spread),
                "n_points_top50": int(len(embs)),
            }
        )

    # write metrics CSV
    metrics_path = OUT / "metrics_two_way.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["method", "run_id", "semantic_spread_top50", "n_points_top50"]
        )
        w.writeheader()
        for r in metrics_rows:
            w.writerow(r)

    # Diversity: full combined population (all runs per method), shared PCA projection
    points_by_method = {
        "toxsearch": gather_combined_population_points("toxsearch", runs, encoder),
        "toxsearch_s": gather_combined_population_points("toxsearch_s", runs, encoder),
    }
    for mk, pts in points_by_method.items():
        print(f"Diversity map: {mk} unique prompts with embeddings = {len(pts)}")
    plot_combined_embedding_map(
        points_by_method,
        FIG / "diversity_dbscan_clusters_top50.pdf",
    )

    print(f"Wrote manifest: {OUT / 'run_manifest.csv'}")
    print(f"Wrote metrics:  {metrics_path}")
    print(f"Wrote figures:  {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

