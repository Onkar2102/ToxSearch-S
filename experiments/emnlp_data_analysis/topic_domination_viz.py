"""Topic-evaluator comparison plots and the topic-domination heatmap/graph."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.topic_domination_viz")


def build_topic_evaluator_comparison(
    google_summaries: Sequence[Dict[str, Any]],
    openai_summaries: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Pair per-species TDI / F₀ metrics across Google and OpenAI."""
    by_g = {int(s["species_id"]): s for s in google_summaries}
    by_o = {int(s["species_id"]): s for s in openai_summaries}
    species_ids = sorted(set(by_g) | set(by_o))
    rows: List[Dict[str, Any]] = []

    for sid in species_ids:
        g = by_g.get(sid, {})
        o = by_o.get(sid, {})
        tdi_g = float(g.get("tdi", float("nan")))
        tdi_o = float(o.get("tdi", float("nan")))
        fd_g = bool(g.get("fully_dominated", False))
        fd_o = bool(o.get("fully_dominated", False))
        if fd_g and fd_o:
            fd_pattern = "both"
        elif fd_g:
            fd_pattern = "google_only"
        elif fd_o:
            fd_pattern = "openai_only"
        else:
            fd_pattern = "neither"

        rows.append({
            "species_id": sid,
            "n_members": int(g.get("n_members") or o.get("n_members") or 0),
            "tdi_google": tdi_g,
            "tdi_openai": tdi_o,
            "tdi_delta_openai_minus_google": round(tdi_o - tdi_g, 4)
            if tdi_g == tdi_g and tdi_o == tdi_o
            else None,
            "n_on_f0_google": int(g.get("n_on_f0", 0)),
            "n_on_f0_openai": int(o.get("n_on_f0", 0)),
            "frac_on_f0_google": g.get("frac_on_f0"),
            "frac_on_f0_openai": o.get("frac_on_f0"),
            "fully_dominated_google": fd_g,
            "fully_dominated_openai": fd_o,
            "fully_dominated_pattern": fd_pattern,
            "distinct_topic": bool(g.get("distinct_topic") or o.get("distinct_topic")),
            "intra_cosine_mean": g.get("intra_cosine_mean") or o.get("intra_cosine_mean"),
            "inter_dg_min": g.get("inter_dg_min") or o.get("inter_dg_min"),
        })

    n_both_fd = sum(1 for r in rows if r["fully_dominated_pattern"] == "both")
    n_neither_fd = sum(1 for r in rows if r["fully_dominated_pattern"] == "neither")
    deltas = [
        r["tdi_delta_openai_minus_google"]
        for r in rows
        if r["tdi_delta_openai_minus_google"] is not None
    ]
    stats = {
        "n_topics": len(rows),
        "n_fully_dominated_both": n_both_fd,
        "n_fully_dominated_neither": n_neither_fd,
        "n_fully_dominated_google_only": sum(
            1 for r in rows if r["fully_dominated_pattern"] == "google_only"
        ),
        "n_fully_dominated_openai_only": sum(
            1 for r in rows if r["fully_dominated_pattern"] == "openai_only"
        ),
        "mean_tdi_delta_openai_minus_google": float(np.mean(deltas)) if deltas else None,
        "max_tdi_delta_openai_minus_google": float(np.max(deltas)) if deltas else None,
        "min_tdi_delta_openai_minus_google": float(np.min(deltas)) if deltas else None,
    }
    return rows, stats


def build_domination_matrix(
    summaries: Sequence[Dict[str, Any]],
    edges: Sequence[Dict[str, int]],
) -> Tuple[np.ndarray, List[int]]:
    """Matrix M[i,j]=1 if species_ids[i] max-vector dominates species_ids[j] (row dominates col)."""
    species_ids = sorted(int(s["species_id"]) for s in summaries)
    if not species_ids:
        return np.zeros((0, 0), dtype=np.int8), []
    idx = {sid: i for i, sid in enumerate(species_ids)}
    n = len(species_ids)
    mat = np.zeros((n, n), dtype=np.int8)
    for e in edges:
        i, j = idx[int(e["from"])], idx[int(e["to"])]
        mat[i, j] = 1
    return mat, species_ids


def plot_topic_evaluator_comparison(
    comparison_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Topic TDI: Google vs OpenAI",
) -> Optional[str]:
    """Grouped bar chart of TDI per species for both evaluators."""
    if not comparison_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    species_ids = [int(r["species_id"]) for r in comparison_rows]
    labels = [str(s) for s in species_ids]
    tdi_g = [float(r["tdi_google"]) for r in comparison_rows]
    tdi_o = [float(r["tdi_openai"]) for r in comparison_rows]
    x = np.arange(len(species_ids))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - w / 2, tdi_g, w, label="Google (Perspective)", color="#4C78A8", alpha=0.9)
    ax.bar(x + w / 2, tdi_o, w, label="OpenAI moderation", color="#F58518", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0)
    ax.set_xlabel("Species ID (topic)")
    ax.set_ylabel("TDI (fraction dominated in global G)")
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="0.5", linestyle=":", linewidth=0.8, alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    ax.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_domination_heatmap(
    matrix: np.ndarray,
    species_ids: Sequence[int],
    out_path: Path,
    *,
    title: str = "Topic max-vector domination",
) -> Optional[str]:
    """Heatmap: row topic's max score vector dominates column topic."""
    if matrix.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    labels = [str(s) for s in species_ids]
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Dominated topic (column)")
    ax.set_ylabel("Dominating topic (row)")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j]:
                ax.text(j, i, "1", ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dominates")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_domination_heatmap_dual(
    mat_g: np.ndarray,
    sids_g: Sequence[int],
    mat_o: np.ndarray,
    sids_o: Sequence[int],
    out_path: Path,
    *,
    title: str = "Topic max-vector domination — Google vs OpenAI",
) -> Optional[str]:
    """Two-panel heatmap (Google left, OpenAI right) for the hero figure budget."""
    if mat_g.size == 0 and mat_o.size == 0:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, mat, sids, sub in zip(
        axes,
        (mat_g, mat_o),
        (sids_g, sids_o),
        ("Google (Perspective)", "OpenAI moderation"),
    ):
        if mat.size == 0:
            ax.set_axis_off()
            ax.set_title(f"{sub}\n(no data)")
            continue
        labels = [str(s) for s in sids]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1, aspect="equal")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_xlabel("Dominated topic (column)")
        ax.set_ylabel("Dominating topic (row)")
        ax.set_title(sub)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if mat[i, j]:
                    ax.text(j, i, "1", ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Dominates")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_domination_graph(
    edges: Sequence[Dict[str, int]],
    species_ids: Sequence[int],
    out_path: Path,
    *,
    title: str = "Topic domination (max-vector)",
) -> Optional[str]:
    """Directed graph: arrow from dominator → dominated topic."""
    if not species_ids:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    n = len(species_ids)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos = {sid: (np.cos(a), np.sin(a)) for sid, a in zip(species_ids, angles)}

    fig, ax = plt.subplots(figsize=(8, 8))
    for sid in species_ids:
        x, y = pos[sid]
        ax.scatter([x], [y], s=320, c="#BFD3E6", edgecolors="#2C5F7A", linewidths=1.2, zorder=2)
        ax.text(x, y, str(sid), ha="center", va="center", fontsize=9, fontweight="bold", zorder=3)

    for e in edges:
        u, v = int(e["from"]), int(e["to"])
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.4, shrinkA=14, shrinkB=14),
            zorder=1,
        )

    ax.set_title(title)
    ax.axis("off")
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "build_domination_matrix",
    "build_topic_evaluator_comparison",
    "plot_domination_graph",
    "plot_domination_heatmap",
    "plot_domination_heatmap_dual",
    "plot_topic_evaluator_comparison",
]
