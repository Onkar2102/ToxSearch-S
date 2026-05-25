"""Phase 11: NSI/TDI tables, single-axis survival, fig11_* plots."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.niche_dominance")


GOOGLE_VEC = "objective_vector"
OPENAI_VEC = "objective_vector_openai"


def _topic_member_indices(
    rows: Sequence[Dict[str, Any]],
    *,
    min_topic_size: int,
) -> Dict[int, List[int]]:
    """Return {species_id -> indices into rows}, dropping topics smaller than min_topic_size."""
    by_topic: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic[sid].append(i)
    return {sid: idxs for sid, idxs in by_topic.items() if len(idxs) >= min_topic_size}


def _axis_score_matrix(
    rows: Sequence[Dict[str, Any]], evaluator: str
) -> Tuple[np.ndarray, List[int]]:
    """Stack (genome_index, score) for genomes that have the evaluator vector."""
    vec_key = GOOGLE_VEC if evaluator == "google" else OPENAI_VEC
    keep = [i for i, r in enumerate(rows) if r.get(vec_key) is not None]
    if not keep:
        return np.zeros((0, 0), dtype=np.float64), []
    F = np.vstack([np.asarray(rows[i][vec_key], dtype=np.float64) for i in keep])
    return F, keep


def _normalised_profile(profile: np.ndarray) -> np.ndarray:
    s = float(profile.sum())
    if s <= 0:
        return np.full_like(profile, 1.0 / len(profile), dtype=np.float64)
    return profile / s


def niche_specialization_index(profile: np.ndarray) -> float:
    """Shannon entropy of the normalised mean profile, in [0, 1].

    High = well-rounded across axes (a "multi-metric" niche).
    Low  = specialised on a single axis (a "single-metric" niche).
    """
    p = _normalised_profile(np.asarray(profile, dtype=np.float64))
    p = np.clip(p, 1e-12, 1.0)
    h = -float(np.sum(p * np.log(p)))
    h_max = math.log(len(p)) if len(p) > 1 else 1.0
    return float(h / h_max) if h_max > 0 else 0.0


def per_topic_silhouette(
    rows: Sequence[Dict[str, Any]],
    member_indices: Dict[int, List[int]],
) -> Dict[int, Optional[float]]:
    """Per-topic mean silhouette in the sentence-embedding space.

    Uses sklearn.metrics.silhouette_samples and averages within each topic.
    Returns None for the topic if fewer than 2 topics or fewer than 2 samples
    are available globally.
    """
    embs: List[np.ndarray] = []
    labels: List[int] = []
    label_to_indices: Dict[int, List[int]] = defaultdict(list)
    for sid, idxs in member_indices.items():
        for i in idxs:
            emb = rows[i].get("_embedding")
            if emb is None:
                continue
            label_to_indices[sid].append(len(embs))
            embs.append(np.asarray(emb, dtype=np.float64).reshape(-1))
            labels.append(sid)

    if len(set(labels)) < 2 or len(labels) < 10:
        return {sid: None for sid in member_indices}

    X = np.vstack(embs)
    y = np.asarray(labels)
    from sklearn.metrics import silhouette_samples

    s_all = silhouette_samples(X, y)
    out: Dict[int, Optional[float]] = {}
    for sid, idxs in label_to_indices.items():
        if not idxs:
            out[sid] = None
            continue
        out[sid] = float(np.mean([s_all[i] for i in idxs]))
    return out


def per_topic_min_dg(
    centroids: Dict[int, Dict[str, Any]],
    genotype_distance: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[int, Optional[float]]:
    """Min embedding-centroid distance to any other topic's centroid."""
    out: Dict[int, Optional[float]] = {}
    sids = [sid for sid in centroids if centroids[sid].get("embedding_centroid") is not None]
    for sid in sids:
        ca = centroids[sid]["embedding_centroid"]
        best = None
        for sid_b in sids:
            if sid_b == sid:
                continue
            cb = centroids[sid_b].get("embedding_centroid")
            if cb is None:
                continue
            d = genotype_distance(ca, cb)
            if best is None or d < best:
                best = d
        out[sid] = float(best) if best is not None else None
    for sid in centroids:
        out.setdefault(sid, None)
    return out


def topic_global_f0_counts(
    rows: Sequence[Dict[str, Any]],
    member_indices: Dict[int, List[int]],
    on_f0_flag: str,
) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for sid, idxs in member_indices.items():
        out[sid] = int(sum(1 for i in idxs if rows[i].get(on_f0_flag)))
    return out


def topic_pareto_contribution_share(
    rows: Sequence[Dict[str, Any]],
    member_indices: Dict[int, List[int]],
    on_f0_flag: str,
    n_f0: int,
) -> Dict[int, float]:
    if n_f0 <= 0:
        return {sid: 0.0 for sid in member_indices}
    counts = topic_global_f0_counts(rows, member_indices, on_f0_flag)
    return {sid: c / n_f0 for sid, c in counts.items()}


def single_axis_topk_topics(
    F: np.ndarray,
    keep_genome_idx: Sequence[int],
    rows: Sequence[Dict[str, Any]],
    member_indices: Dict[int, List[int]],
    *,
    axis: int,
    k: int,
) -> List[int]:
    """Top-k genomes by axis score; return species_ids whose topic intersects the keep set."""
    if F.size == 0 or k <= 0:
        return []
    n = F.shape[0]
    k_eff = min(k, n)
    order = np.argsort(F[:, axis])[::-1][:k_eff]
    kept = {int(keep_genome_idx[i]) for i in order}
    survived: List[int] = []
    for sid, idxs in member_indices.items():
        if any(i in kept for i in idxs):
            survived.append(int(sid))
    return sorted(survived)


def global_mo_topics(
    rows: Sequence[Dict[str, Any]],
    member_indices: Dict[int, List[int]],
    on_f0_flag: str,
) -> List[int]:
    out: List[int] = []
    for sid, idxs in member_indices.items():
        if any(rows[i].get(on_f0_flag) for i in idxs):
            out.append(int(sid))
    return sorted(out)


def speciated_oracle_topics(member_indices: Dict[int, List[int]]) -> List[int]:
    return sorted(int(sid) for sid in member_indices)


def build_advanced_analytics(
    rows: Sequence[Dict[str, Any]],
    *,
    min_topic_size: int,
    google_axis_order: Sequence[str],
    openai_axis_order: Sequence[str],
    google_summaries: Sequence[Dict[str, Any]],
    openai_summaries: Sequence[Dict[str, Any]],
    google_centroids: Dict[int, Dict[str, Any]],
    openai_centroids: Dict[int, Dict[str, Any]],
    genotype_distance: Callable[[np.ndarray, np.ndarray], float],
) -> Dict[str, Any]:
    """Compute the per-topic table + survival CSV + summary numbers."""
    member_indices = _topic_member_indices(rows, min_topic_size=min_topic_size)
    summary_by_g = {int(s["species_id"]): s for s in google_summaries}
    summary_by_o = {int(s["species_id"]): s for s in openai_summaries}
    silh = per_topic_silhouette(rows, member_indices)
    min_dg_g = per_topic_min_dg(google_centroids, genotype_distance)
    min_dg_o = per_topic_min_dg(openai_centroids, genotype_distance)

    # Score matrices per evaluator.
    F_g, idx_g = _axis_score_matrix(rows, "google")
    F_o, idx_o = _axis_score_matrix(rows, "openai")
    n_f0_g = int(sum(1 for r in rows if r.get("on_global_f0_google")))
    n_f0_o = int(sum(1 for r in rows if r.get("on_global_f0_openai")))

    contrib_g = topic_pareto_contribution_share(
        rows, member_indices, "on_global_f0_google", n_f0_g
    )
    contrib_o = topic_pareto_contribution_share(
        rows, member_indices, "on_global_f0_openai", n_f0_o
    )
    on_f0_g = topic_global_f0_counts(rows, member_indices, "on_global_f0_google")
    on_f0_o = topic_global_f0_counts(rows, member_indices, "on_global_f0_openai")

    # Topics-on-front-or-not under each policy.
    google_axis_survival: Dict[str, List[int]] = {}
    openai_axis_survival: Dict[str, List[int]] = {}
    for j, axis in enumerate(google_axis_order):
        google_axis_survival[axis] = single_axis_topk_topics(
            F_g, idx_g, rows, member_indices, axis=j, k=n_f0_g
        )
    for j, axis in enumerate(openai_axis_order):
        openai_axis_survival[axis] = single_axis_topk_topics(
            F_o, idx_o, rows, member_indices, axis=j, k=n_f0_o
        )
    google_mo_survival = global_mo_topics(rows, member_indices, "on_global_f0_google")
    openai_mo_survival = global_mo_topics(rows, member_indices, "on_global_f0_openai")
    oracle_survival = speciated_oracle_topics(member_indices)

    # Per-topic table.
    topic_rows: List[Dict[str, Any]] = []
    for sid in sorted(member_indices):
        idxs = member_indices[sid]
        sg = summary_by_g.get(sid, {})
        so = summary_by_o.get(sid, {})
        cg = google_centroids.get(sid, {})
        co = openai_centroids.get(sid, {})

        mean_g = (
            np.asarray(cg.get("mean_vector"))
            if cg.get("mean_vector") is not None
            else np.zeros(len(google_axis_order))
        )
        mean_o = (
            np.asarray(co.get("mean_vector"))
            if co.get("mean_vector") is not None
            else np.zeros(len(openai_axis_order))
        )
        max_g = (
            np.asarray(cg.get("max_vector"))
            if cg.get("max_vector") is not None
            else np.zeros(len(google_axis_order))
        )
        max_o = (
            np.asarray(co.get("max_vector"))
            if co.get("max_vector") is not None
            else np.zeros(len(openai_axis_order))
        )
        spec_g = niche_specialization_index(mean_g)
        spec_o = niche_specialization_index(mean_o)

        # Domination flip pattern across evaluators.
        fd_g = bool(sg.get("fully_dominated", False))
        fd_o = bool(so.get("fully_dominated", False))
        if fd_g and fd_o:
            flip = "both_dominated"
        elif fd_g:
            flip = "google_only"
        elif fd_o:
            flip = "openai_only"
        else:
            flip = "neither"

        # Single-axis policies that retain this topic, per evaluator.
        survives_g = sorted(
            ax for ax, sids in google_axis_survival.items() if sid in sids
        )
        survives_o = sorted(
            ax for ax, sids in openai_axis_survival.items() if sid in sids
        )

        row: Dict[str, Any] = {
            "species_id": sid,
            "size": len(idxs),
            "tdi_google": float(sg.get("tdi", float("nan"))),
            "tdi_openai": float(so.get("tdi", float("nan"))),
            "fully_dominated_google": fd_g,
            "fully_dominated_openai": fd_o,
            "domination_flip_pattern": flip,
            "on_f0_google": on_f0_g.get(sid, 0),
            "on_f0_openai": on_f0_o.get(sid, 0),
            "pareto_contribution_share_google": round(contrib_g.get(sid, 0.0), 4),
            "pareto_contribution_share_openai": round(contrib_o.get(sid, 0.0), 4),
            "embedding_silhouette": (
                round(float(silh[sid]), 4) if silh.get(sid) is not None else None
            ),
            "min_dg_google": (
                round(float(min_dg_g.get(sid)), 4)
                if min_dg_g.get(sid) is not None
                else None
            ),
            "min_dg_openai": (
                round(float(min_dg_o.get(sid)), 4)
                if min_dg_o.get(sid) is not None
                else None
            ),
            "niche_specialization_google": round(spec_g, 4),
            "niche_specialization_openai": round(spec_o, 4),
            "survives_global_mo_google": int(sid in google_mo_survival),
            "survives_global_mo_openai": int(sid in openai_mo_survival),
            "survives_speciated_oracle": int(sid in oracle_survival),
            "single_axis_survivors_google": "|".join(survives_g),
            "single_axis_survivors_openai": "|".join(survives_o),
            "n_single_axis_google": len(survives_g),
            "n_single_axis_openai": len(survives_o),
        }
        for j, axis in enumerate(google_axis_order):
            row[f"axis_mean_g_{axis}"] = round(float(mean_g[j]), 4)
            row[f"axis_max_g_{axis}"] = round(float(max_g[j]), 4)
            row[f"survives_g_axis_{axis}"] = int(sid in google_axis_survival[axis])
        for j, axis in enumerate(openai_axis_order):
            safe = axis.replace("/", "_")
            row[f"axis_mean_o_{safe}"] = round(float(mean_o[j]), 4)
            row[f"axis_max_o_{safe}"] = round(float(max_o[j]), 4)
            row[f"survives_o_axis_{safe}"] = int(sid in openai_axis_survival[axis])

        topic_rows.append(row)

    # Long counterfactual survival table.
    survival_rows: List[Dict[str, Any]] = []
    n_topics = len(member_indices)

    def _record(evaluator: str, policy: str, sids: Sequence[int]) -> Dict[str, Any]:
        sids_set = sorted(int(s) for s in sids)
        return {
            "evaluator": evaluator,
            "policy": policy,
            "topics_total": n_topics,
            "topics_survived": len(sids_set),
            "survival_rate": round(
                len(sids_set) / n_topics if n_topics else 0.0, 4
            ),
            "extinction_rate": round(
                (n_topics - len(sids_set)) / n_topics if n_topics else 0.0, 4
            ),
            "topic_set": "|".join(str(s) for s in sids_set),
        }

    for axis in google_axis_order:
        survival_rows.append(_record("google", f"single_axis:{axis}", google_axis_survival[axis]))
    for axis in openai_axis_order:
        survival_rows.append(_record("openai", f"single_axis:{axis}", openai_axis_survival[axis]))
    survival_rows.append(_record("google", "global_mo", google_mo_survival))
    survival_rows.append(_record("openai", "global_mo", openai_mo_survival))
    survival_rows.append(_record("google", "speciated_oracle", oracle_survival))
    survival_rows.append(_record("openai", "speciated_oracle", oracle_survival))

    summary = {
        "n_topics": n_topics,
        "n_f0_google": n_f0_g,
        "n_f0_openai": n_f0_o,
        "n_topicless_global_f0_google": int(
            n_f0_g - sum(on_f0_g.get(sid, 0) for sid in member_indices)
        ),
        "n_topicless_global_f0_openai": int(
            n_f0_o - sum(on_f0_o.get(sid, 0) for sid in member_indices)
        ),
        "best_single_axis_google": max(
            google_axis_survival.items(),
            key=lambda kv: len(kv[1]),
            default=("", []),
        ),
        "best_single_axis_openai": max(
            openai_axis_survival.items(),
            key=lambda kv: len(kv[1]),
            default=("", []),
        ),
        "worst_single_axis_google": min(
            google_axis_survival.items(),
            key=lambda kv: len(kv[1]),
            default=("", []),
        ),
        "worst_single_axis_openai": min(
            openai_axis_survival.items(),
            key=lambda kv: len(kv[1]),
            default=("", []),
        ),
        "n_topics_no_single_axis_recovers_google": sum(
            1 for r in topic_rows if r["n_single_axis_google"] == 0
        ),
        "n_topics_no_single_axis_recovers_openai": sum(
            1 for r in topic_rows if r["n_single_axis_openai"] == 0
        ),
        "mean_specialization_google": round(
            float(np.mean([r["niche_specialization_google"] for r in topic_rows])), 4
        )
        if topic_rows
        else None,
        "mean_specialization_openai": round(
            float(np.mean([r["niche_specialization_openai"] for r in topic_rows])), 4
        )
        if topic_rows
        else None,
        "mean_silhouette": (
            round(
                float(
                    np.mean(
                        [r["embedding_silhouette"] for r in topic_rows
                         if r["embedding_silhouette"] is not None]
                    )
                ),
                4,
            )
            if any(r["embedding_silhouette"] is not None for r in topic_rows)
            else None
        ),
        "google_mo_topics_survived": len(google_mo_survival),
        "openai_mo_topics_survived": len(openai_mo_survival),
        "speciated_oracle_topics_survived": len(oracle_survival),
    }
    # Tuples are not JSON-friendly when nested; flatten the best/worst entries.
    for key in (
        "best_single_axis_google",
        "best_single_axis_openai",
        "worst_single_axis_google",
        "worst_single_axis_openai",
    ):
        axis, sids = summary[key]
        summary[key] = {"axis": axis, "topics_survived": len(sids), "topic_set": "|".join(str(s) for s in sids)}

    return {
        "topic_rows": topic_rows,
        "survival_rows": survival_rows,
        "summary": summary,
        "axis_survival_google": google_axis_survival,
        "axis_survival_openai": openai_axis_survival,
    }


def _figsize_for_axes(n_axes: int) -> Tuple[float, float]:
    return (max(10.0, 1.0 + 0.6 * n_axes), 5.6)


def plot_single_axis_survival(
    axis_survival_google: Dict[str, List[int]],
    axis_survival_openai: Dict[str, List[int]],
    google_mo: int,
    openai_mo: int,
    oracle: int,
    n_topics: int,
    out_path: Path,
    *,
    title: str = "Topics retained by selection policy",
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    google_policies = list(axis_survival_google.keys())
    openai_policies = list(axis_survival_openai.keys())
    g_vals = [len(axis_survival_google[a]) for a in google_policies]
    o_vals = [len(axis_survival_openai[a]) for a in openai_policies]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=True)

    g_x = np.arange(len(google_policies) + 2)
    g_y = g_vals + [google_mo, oracle]
    g_labels = [f"axis: {a}" for a in google_policies] + ["global Pareto", "speciated oracle"]
    g_colors = ["#7B9CC2"] * len(google_policies) + ["#C44E52", "#2C8C5C"]
    bars_g = axes[0].bar(g_x, g_y, color=g_colors, edgecolor="0.3", alpha=0.95)
    axes[0].set_xticks(g_x)
    axes[0].set_xticklabels(g_labels, rotation=45, ha="right", fontsize=8)
    axes[0].set_ylabel(f"Topics retained (of {n_topics})")
    axes[0].set_ylim(0, n_topics + 1)
    axes[0].set_title("Google (Perspective)")
    axes[0].axhline(n_topics, color="0.5", ls=":", lw=1, alpha=0.7)
    for bar, v in zip(bars_g, g_y):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v),
                     ha="center", va="bottom", fontsize=8)

    o_x = np.arange(len(openai_policies) + 2)
    o_y = o_vals + [openai_mo, oracle]
    o_labels = [f"axis: {a}" for a in openai_policies] + ["global Pareto", "speciated oracle"]
    o_colors = ["#E0A26F"] * len(openai_policies) + ["#C44E52", "#2C8C5C"]
    bars_o = axes[1].bar(o_x, o_y, color=o_colors, edgecolor="0.3", alpha=0.95)
    axes[1].set_xticks(o_x)
    axes[1].set_xticklabels(o_labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_title("OpenAI moderation")
    axes[1].axhline(n_topics, color="0.5", ls=":", lw=1, alpha=0.7)
    for bar, v in zip(bars_o, o_y):
        axes[1].text(bar.get_x() + bar.get_width() / 2, v + 0.1, str(v),
                     ha="center", va="bottom", fontsize=8)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_topic_profile_vs_domination(
    topic_rows: Sequence[Dict[str, Any]],
    google_axis_order: Sequence[str],
    out_path: Path,
    *,
    title: str = "Per-topic profile vs domination",
) -> Optional[str]:
    if not topic_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    sids = [r["species_id"] for r in topic_rows]
    profile = np.array(
        [[r[f"axis_mean_g_{a}"] for a in google_axis_order] for r in topic_rows],
        dtype=np.float64,
    )
    tdi_g = np.array([r["tdi_google"] for r in topic_rows], dtype=np.float64)
    tdi_o = np.array([r["tdi_openai"] for r in topic_rows], dtype=np.float64)
    spec = np.array(
        [r["niche_specialization_google"] for r in topic_rows], dtype=np.float64
    )

    fig, axes = plt.subplots(
        1, 4, figsize=(13, max(4.5, 0.55 * len(sids) + 1.5)),
        gridspec_kw={"width_ratios": [4, 0.6, 0.6, 0.6]},
    )

    im = axes[0].imshow(profile, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(google_axis_order)))
    axes[0].set_xticklabels(list(google_axis_order), rotation=45, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(sids)))
    axes[0].set_yticklabels([str(s) for s in sids])
    axes[0].set_xlabel("Google axis")
    axes[0].set_ylabel("Species (topic)")
    axes[0].set_title("Mean axis profile (Google)")
    fig.colorbar(im, ax=axes[0], fraction=0.04, pad=0.02)

    for ax, vals, lab, cmap, vmin, vmax in (
        (axes[1], tdi_g, "TDI (G)", "Greys", 0, 1),
        (axes[2], tdi_o, "TDI (O)", "Greys", 0, 1),
        (axes[3], spec, "Specialization (entropy)", "Blues", 0, 1),
    ):
        im_a = ax.imshow(vals.reshape(-1, 1), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks(range(len(sids)))
        ax.set_yticklabels([])
        ax.set_title(lab, fontsize=9)
        for i, v in enumerate(vals):
            ax.text(0, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > 0.5 else "black", fontsize=7)

    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_distinct_but_dominated(
    topic_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "Distinct-but-dominated topics",
) -> Optional[str]:
    if not topic_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    sids = [r["species_id"] for r in topic_rows]
    sizes = np.array([r["size"] for r in topic_rows], dtype=np.float64)
    silh = np.array(
        [r["embedding_silhouette"] if r["embedding_silhouette"] is not None else 0.0
         for r in topic_rows],
        dtype=np.float64,
    )
    tdi_g = np.array([r["tdi_google"] for r in topic_rows], dtype=np.float64)
    tdi_o = np.array([r["tdi_openai"] for r in topic_rows], dtype=np.float64)
    spec = np.array(
        [r["niche_specialization_google"] for r in topic_rows], dtype=np.float64
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, tdi, sub in zip(axes, (tdi_g, tdi_o), ("Google TDI", "OpenAI TDI")):
        sc = ax.scatter(
            silh, tdi, s=20 + 4 * sizes, c=spec, cmap="viridis", vmin=0, vmax=1,
            edgecolors="0.2", linewidths=0.5, alpha=0.92,
        )
        for i, sid in enumerate(sids):
            ax.text(silh[i], tdi[i], f" {sid}", fontsize=8, va="center")
        ax.axhline(1.0, color="0.5", ls=":", lw=1)
        ax.set_xlabel("Per-topic embedding silhouette")
        ax.set_ylabel(sub)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(sub + " vs embedding silhouette")
        ax.grid(True, alpha=0.3)
    cbar = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Niche specialization (entropy of profile, Google)")
    fig.suptitle(title + " — top-right = behaviourally distinct yet fully dominated", fontsize=11, y=1.02)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_niche_specialization_vs_tdi(
    topic_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    labels_by_sid: Optional[Dict[int, str]] = None,
    title: str = "Niche specialization vs topic domination index",
) -> Optional[str]:
    """Figure 5 (paper): NSI vs TDI scatter with optional Phase 12 short labels.

    Two panels (Google, OpenAI). Each point is a topic; annotations come from
    ``labels_by_sid`` if provided. Bubble size encodes topic size; points in
    the upper-right (high TDI ∧ high NSI) are well-rounded yet fully dominated.
    """
    if not topic_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    sids = [int(r["species_id"]) for r in topic_rows]
    sizes = np.asarray([float(r["size"]) for r in topic_rows], dtype=np.float64)
    nsi_g = np.asarray(
        [float(r["niche_specialization_google"]) for r in topic_rows],
        dtype=np.float64,
    )
    nsi_o = np.asarray(
        [float(r["niche_specialization_openai"]) for r in topic_rows],
        dtype=np.float64,
    )
    tdi_g = np.asarray([float(r["tdi_google"]) for r in topic_rows], dtype=np.float64)
    tdi_o = np.asarray([float(r["tdi_openai"]) for r in topic_rows], dtype=np.float64)
    survives_g = np.asarray(
        [int(r.get("survives_global_mo_google", 0)) for r in topic_rows],
        dtype=np.int32,
    )
    survives_o = np.asarray(
        [int(r.get("survives_global_mo_openai", 0)) for r in topic_rows],
        dtype=np.int32,
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    panels = (
        (axes[0], nsi_g, tdi_g, survives_g, "Google (Perspective)"),
        (axes[1], nsi_o, tdi_o, survives_o, "OpenAI moderation"),
    )
    for ax, nsi, tdi, surv, sub in panels:
        colors = ["#2C8C5C" if s else "#C44E52" for s in surv]
        ax.scatter(
            nsi, tdi,
            s=30 + 4 * sizes,
            c=colors,
            edgecolors="0.2",
            linewidths=0.5,
            alpha=0.92,
        )
        for i, sid in enumerate(sids):
            label = (labels_by_sid or {}).get(int(sid)) or f"sid={sid}"
            label = label if len(label) <= 32 else (label[:30] + "…")
            ax.annotate(
                f"{sid}: {label}",
                (nsi[i], tdi[i]),
                xytext=(6, 4),
                textcoords="offset points",
                fontsize=8,
                color="0.15",
            )
        ax.axhline(1.0, color="0.5", ls=":", lw=1, alpha=0.7)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Niche specialization (entropy of profile)")
        ax.set_title(sub)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Topic Domination Index (TDI)")

    legend_elems = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#2C8C5C",
                   markersize=8, label="survives global MO"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#C44E52",
                   markersize=8, label="extinguished by global MO"),
    ]
    axes[1].legend(handles=legend_elems, loc="lower left", fontsize=8, frameon=True)
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "build_advanced_analytics",
    "niche_specialization_index",
    "per_topic_min_dg",
    "per_topic_silhouette",
    "plot_distinct_but_dominated",
    "plot_niche_specialization_vs_tdi",
    "plot_single_axis_survival",
    "plot_topic_profile_vs_domination",
    "single_axis_topk_topics",
]
