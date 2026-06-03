"""Behaviour layout: UMAP, topic×axis heatmaps, cluster quality, centroid d_g vs d_p."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.behaviour_layout")

F0_FLAG = {"google": "on_global_f0_google", "openai": "on_global_f0_openai"}


def stamp_f0_flags(
    rows: Sequence[Dict[str, Any]],
    *,
    global_pareto_annotate,
) -> None:
    """Annotate per-evaluator global F₀ membership on row dicts (in-place)."""
    for evaluator, vec_key in (
        ("google", "objective_vector"),
        ("openai", "objective_vector_openai"),
    ):
        valid_idx = [i for i, r in enumerate(rows) if r.get(vec_key) is not None]
        if not valid_idx:
            continue
        F = np.vstack([rows[i][vec_key] for i in valid_idx])
        _, on_f0, _ = global_pareto_annotate(F)
        flag = F0_FLAG[evaluator]
        for j, li in enumerate(valid_idx):
            rows[li][flag] = bool(on_f0[j])


def _umap_coords(embeddings: np.ndarray) -> Tuple[np.ndarray, str]:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings), "umap"
    except ImportError:
        _logger.warning("umap-learn not installed; using PCA fallback")
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings), "pca"


def plot_umap_species(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    evaluator: str = "google",
    max_points: int = 3000,
    title: str = "",
) -> Optional[str]:
    vec_key = "objective_vector" if evaluator == "google" else "objective_vector_openai"
    f0_key = F0_FLAG.get(evaluator, "on_global_f0_google")
    pts = [
        (r, r["_embedding"])
        for r in rows
        if r.get("_embedding") is not None and r.get(vec_key) is not None
    ]
    if len(pts) < 10:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = [pts[i] for i in sorted(idx)]

    embs = np.vstack([p[1] for p in pts])
    coords, method = _umap_coords(embs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for (r, _), (x, y) in zip(pts, coords):
        sid = int(r.get("species_id") or 0)
        on_f0 = bool(r.get(f0_key))
        color = plt.cm.tab20(sid % 20) if sid > 0 else "#999999"
        marker = "o" if on_f0 else "x"
        ax.scatter(x, y, c=[color], marker=marker, s=18, alpha=0.7)
    ax.set_title(
        title
        or f"UMAP ({method}): species colour; circle=F₀ ({evaluator}), x=dominated"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def topic_axis_max_matrix(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    get_axis_order,
    topic_centroids,
    min_topic_size: int = 5,
) -> Tuple[List[int], List[str], np.ndarray]:
    axis_order = list(get_axis_order(evaluator))
    vec_key = "objective_vector" if evaluator == "google" else "objective_vector_openai"
    valid = [r for r in rows if vec_key in r]
    centroids = topic_centroids(valid, evaluator)
    sids = sorted(
        sid for sid, c in centroids.items() if len(c.get("members") or []) >= min_topic_size
    )
    mat = np.zeros((len(sids), len(axis_order)), dtype=np.float64)
    for i, sid in enumerate(sids):
        mat[i] = centroids[sid]["max_vector"]
    return sids, axis_order, mat


def write_topic_axis_csv(
    sids: Sequence[int],
    axes: Sequence[str],
    mat: np.ndarray,
    out_path: Path,
) -> str:
    import csv

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["species_id"] + list(axes))
        for sid, row in zip(sids, mat):
            w.writerow([sid] + [round(float(v), 6) for v in row])
    return str(out_path)


def plot_topic_axis_heatmap(
    sids: Sequence[int],
    axes: Sequence[str],
    mat: np.ndarray,
    out_path: Path,
    *,
    title: str = "",
) -> Optional[str]:
    if mat.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(axes) * 0.55), max(4, len(sids) * 0.45)))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(axes)))
    ax.set_xticklabels(list(axes), rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(sids)))
    ax.set_yticklabels([str(s) for s in sids])
    ax.set_ylabel("Species (topic)")
    ax.set_title(title or "Per-topic max axis scores")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_umap_species_dual(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    max_points: int = 3000,
    title: str = "UMAP of topics — Google vs OpenAI F₀ membership",
) -> Optional[str]:
    """Two-panel UMAP. Same embedding layout, F₀ marker switches per evaluator."""
    pts = [
        (r, r["_embedding"])
        for r in rows
        if r.get("_embedding") is not None
        and r.get("objective_vector") is not None
        and r.get("objective_vector_openai") is not None
    ]
    if len(pts) < 10:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = [pts[i] for i in sorted(idx)]

    embs = np.vstack([p[1] for p in pts])
    coords, method = _umap_coords(embs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, evaluator in zip(axes, ("google", "openai")):
        f0_key = F0_FLAG[evaluator]
        for (r, _), (x, y) in zip(pts, coords):
            sid = int(r.get("species_id") or 0)
            on_f0 = bool(r.get(f0_key))
            color = plt.cm.tab20(sid % 20) if sid > 0 else "#999999"
            marker = "o" if on_f0 else "x"
            ax.scatter(x, y, c=[color], marker=marker, s=14, alpha=0.65)
        ax.set_title(f"{method.upper()} — F₀ ({evaluator}); circle=on front, x=dominated")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_topic_axis_heatmap_dual(
    sids_g: Sequence[int],
    axes_g: Sequence[str],
    mat_g: np.ndarray,
    sids_o: Sequence[int],
    axes_o: Sequence[str],
    mat_o: np.ndarray,
    out_path: Path,
    *,
    title: str = "Per-topic max-score profiles — Google vs OpenAI",
) -> Optional[str]:
    """Two-panel topic×axis heatmap (Google left, OpenAI right)."""
    if mat_g.size == 0 and mat_o.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    panels = [(sids_g, axes_g, mat_g, "Google (Perspective)"),
              (sids_o, axes_o, mat_o, "OpenAI moderation")]
    n_axes = max(len(axes_g), len(axes_o))
    fig, axes_fig = plt.subplots(1, 2, figsize=(max(12, n_axes * 0.7), 6.2))
    for ax, (sids, axes_lbl, mat, sub) in zip(axes_fig, panels):
        if mat.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{sub}\n(no data)")
            continue
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(axes_lbl)))
        ax.set_xticklabels(list(axes_lbl), rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(sids)))
        ax.set_yticklabels([str(s) for s in sids])
        ax.set_ylabel("Species (topic)")
        ax.set_title(sub)
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_centroid_dg_dp_dual(
    rows: Sequence[Dict[str, Any]],
    summaries_by_eval: Dict[str, Sequence[Dict[str, Any]]],
    out_path: Path,
    *,
    topic_centroids,
    genotype_distance,
    phenotype_distance,
    title: str = "Topic centroid separation — d_g vs d_p (dual evaluator)",
) -> Optional[str]:
    """Two-panel d_g vs d_p centroid scatter (Google left, OpenAI right)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, axes_fig = plt.subplots(1, 2, figsize=(11, 4.6))
    drew = 0
    for ax, evaluator in zip(axes_fig, ("google", "openai")):
        vec_key = "objective_vector" if evaluator == "google" else "objective_vector_openai"
        valid = [r for r in rows if r.get(vec_key) is not None]
        if len(valid) < 2:
            ax.set_axis_off()
            continue
        centroids = topic_centroids(valid, evaluator)
        summaries = summaries_by_eval.get(evaluator, [])
        sids = [s["species_id"] for s in summaries if s.get("distinct_topic")]
        if len(sids) < 2:
            sids = [s["species_id"] for s in summaries]
        if len(sids) < 2:
            ax.set_axis_off()
            continue
        xs, ys, cols = [], [], []
        for i, a in enumerate(sids):
            for b in sids[i + 1:]:
                ca, cb = centroids.get(a), centroids.get(b)
                if not ca or not cb:
                    continue
                if ca.get("embedding_centroid") is None or cb.get("embedding_centroid") is None:
                    continue
                dg = genotype_distance(ca["embedding_centroid"], cb["embedding_centroid"])
                dp = phenotype_distance(ca["mean_vector"], cb["mean_vector"])
                xs.append(dg)
                ys.append(dp)
                fa = next((s for s in summaries if s["species_id"] == a), {})
                fb = next((s for s in summaries if s["species_id"] == b), {})
                dominated = fa.get("fully_dominated") or fb.get("fully_dominated")
                cols.append("#c0392b" if dominated else "#2980b9")
        if not xs:
            ax.set_axis_off()
            continue
        ax.scatter(xs, ys, c=cols, alpha=0.75)
        ax.set_xlabel("Genotype distance d_g (centroids)")
        ax.set_ylabel("Phenotype distance d_p (centroids)")
        ax.set_title(f"{evaluator} — red = involves fully-dominated topic")
        ax.grid(True, alpha=0.3)
        drew += 1
    if drew == 0:
        plt.close(fig)
        return None
    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def compute_cluster_quality(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    embs: List[np.ndarray] = []
    labels: List[int] = []
    for r in rows:
        emb = r.get("_embedding")
        sid = int(r.get("species_id") or 0)
        if emb is not None and sid > 0:
            embs.append(np.asarray(emb, dtype=np.float64).reshape(-1))
            labels.append(sid)
    if len(set(labels)) < 2 or len(labels) < 10:
        return {"num_samples": len(labels), "num_clusters": len(set(labels)), "skipped": True}
    X = np.vstack(embs)
    y = np.asarray(labels)
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    return {
        "silhouette_score": float(silhouette_score(X, y)),
        "davies_bouldin_index": float(davies_bouldin_score(X, y)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X, y)),
        "num_samples": int(len(labels)),
        "num_clusters": int(len(set(labels))),
        "skipped": False,
    }


def plot_centroid_dg_dp(
    rows: Sequence[Dict[str, Any]],
    summaries: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    evaluator: str,
    topic_centroids,
    genotype_distance,
    phenotype_distance,
) -> Optional[str]:
    vec_key = "objective_vector" if evaluator == "google" else "objective_vector_openai"
    valid = [r for r in rows if r.get(vec_key) is not None]
    if len(valid) < 2:
        return None
    centroids = topic_centroids(valid, evaluator)
    m = centroids[next(iter(centroids))]["mean_vector"].shape[0] if centroids else 0
    if m == 0:
        return None

    sids = [s["species_id"] for s in summaries if s.get("distinct_topic")]
    if len(sids) < 2:
        sids = [s["species_id"] for s in summaries]
    if len(sids) < 2:
        return None

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    xs, ys, cols = [], [], []
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            ca, cb = centroids.get(a), centroids.get(b)
            if not ca or not cb:
                continue
            if ca.get("embedding_centroid") is None or cb.get("embedding_centroid") is None:
                continue
            dg = genotype_distance(ca["embedding_centroid"], cb["embedding_centroid"])
            dp = phenotype_distance(ca["mean_vector"], cb["mean_vector"])
            xs.append(dg)
            ys.append(dp)
            fa = next((s for s in summaries if s["species_id"] == a), {})
            fb = next((s for s in summaries if s["species_id"] == b), {})
            dominated = fa.get("fully_dominated") or fb.get("fully_dominated")
            cols.append("#c0392b" if dominated else "#2980b9")

    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, c=cols, alpha=0.75)
    ax.set_xlabel("Genotype distance d_g (centroids)")
    ax.set_ylabel("Phenotype distance d_p (centroids)")
    ax.set_title(f"Topic centroid separation ({evaluator})")
    ax.grid(True, alpha=0.3)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "compute_cluster_quality",
    "plot_centroid_dg_dp",
    "plot_centroid_dg_dp_dual",
    "plot_topic_axis_heatmap",
    "plot_topic_axis_heatmap_dual",
    "plot_umap_species",
    "plot_umap_species_dual",
    "stamp_f0_flags",
    "topic_axis_max_matrix",
    "write_topic_axis_csv",
]
