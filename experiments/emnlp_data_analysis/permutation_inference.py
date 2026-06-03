"""Permutation tests, rank contingency heatmap, and axis-score correlations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.permutation_inference")

VECTOR_KEY = {"google": "objective_vector", "openai": "objective_vector_openai"}
F0_FLAG = {"google": "on_global_f0_google", "openai": "on_global_f0_openai"}


def stamp_tdi_members(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    min_topic_size: int,
    member_dominated_mask,
) -> None:
    """Mark dominated-in-topic members (not on evaluator F₀) for permutation contrast."""
    vec_key = VECTOR_KEY[evaluator]
    valid = [r for r in rows if r.get(vec_key) is not None]
    if not valid:
        return
    F = np.vstack([r[vec_key] for r in valid])
    dominated = member_dominated_mask(F)
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)
    key = f"tdi_member_{evaluator}"
    f0_key = F0_FLAG[evaluator]
    for r in rows:
        r[key] = False
    for idxs in by_topic.values():
        if len(idxs) < min_topic_size:
            continue
        for li in idxs:
            r = valid[li]
            if dominated[li] and not r.get(f0_key):
                r[key] = True


def holm_bonferroni(
    p_values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """Holm-Bonferroni; returns {p_raw, p_holm, reject_at_0p05, rank} per input."""
    n = len(p_values)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: p_values[i])
    adjusted = [0.0] * n
    running = 0.0
    for rank, i in enumerate(order):
        m = n - rank
        cand = float(p_values[i]) * m
        running = max(running, cand)
        adjusted[i] = min(1.0, running)
    out: List[Dict[str, Any]] = []
    inv_rank = {i: r for r, i in enumerate(order)}
    for i, p in enumerate(p_values):
        out.append({
            "p_raw": float(p),
            "p_holm": round(float(adjusted[i]), 6),
            "reject_at_0p05": bool(adjusted[i] <= alpha),
            "rank": int(inv_rank[i]) + 1,
        })
    return out


def per_topic_permutation_separation(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    genotype_distance,
    n_perm: int = 5000,
    seed: int = 42,
    min_topic_size: int = 5,
    max_pool_size: int = 200,
) -> List[Dict[str, Any]]:
    """Permutation test: mean d_g(TDI members) vs global F0 pool (per topic)."""
    f0_key = F0_FLAG[evaluator]
    rng = np.random.default_rng(seed)
    f0_pool: List[np.ndarray] = []
    for r in rows:
        if r.get(f0_key) and r.get("_embedding") is not None:
            f0_pool.append(np.asarray(r["_embedding"], dtype=np.float64).ravel())
    if len(f0_pool) < 2:
        return []

    by_topic: Dict[int, List[np.ndarray]] = {}
    for r in rows:
        sid = int(r.get("species_id") or 0)
        if sid <= 0:
            continue
        emb = r.get("_embedding")
        if emb is None:
            continue
        if r.get(f0_key):
            continue  # only dominated members per topic
        by_topic.setdefault(sid, []).append(
            np.asarray(emb, dtype=np.float64).ravel()
        )

    def _stat(g0: List[np.ndarray], g1: List[np.ndarray]) -> float:
        if not g0 or not g1:
            return 0.0
        a = g0[: max_pool_size]
        b = g1[: max_pool_size]
        d_sum = 0.0
        n_pairs = 0
        for x in a:
            for y in b:
                d_sum += genotype_distance(x, y)
                n_pairs += 1
        return d_sum / n_pairs if n_pairs else 0.0

    out: List[Dict[str, Any]] = []
    for sid, members in sorted(by_topic.items()):
        if len(members) < min_topic_size:
            continue
        g0 = list(members)
        g1 = list(f0_pool)
        if len(g0) > max_pool_size:
            idx = rng.choice(len(g0), size=max_pool_size, replace=False)
            g0 = [g0[i] for i in idx]
        if len(g1) > max_pool_size:
            idx = rng.choice(len(g1), size=max_pool_size, replace=False)
            g1 = [g1[i] for i in idx]
        obs = _stat(g0, g1)
        pool = g0 + g1
        n0 = len(g0)
        ge = 0
        for _ in range(n_perm):
            perm_idx = rng.permutation(len(pool))
            shuf = [pool[k] for k in perm_idx]
            if _stat(shuf[:n0], shuf[n0:]) >= obs:
                ge += 1
        p_raw = (ge + 1) / (n_perm + 1)
        out.append({
            "evaluator": evaluator,
            "species_id": int(sid),
            "n_topic_dominated": len(members),
            "n_f0_global": len(f0_pool),
            "obs_stat": round(float(obs), 6),
            "p_raw": round(float(p_raw), 6),
            "n_perm": int(n_perm),
        })
    return out


def annotate_holm_bonferroni(
    rows: List[Dict[str, Any]],
    *,
    p_key: str = "p_raw",
    alpha: float = 0.05,
) -> List[Dict[str, Any]]:
    """In-place: append Holm-Bonferroni adjusted p and rejection flag to rows."""
    if not rows:
        return rows
    pvals = [float(r.get(p_key, 1.0)) for r in rows]
    adj = holm_bonferroni(pvals, alpha=alpha)
    for r, info in zip(rows, adj):
        r["p_holm"] = info["p_holm"]
        r["reject_at_0p05"] = info["reject_at_0p05"]
        r["holm_rank"] = info["rank"]
    return rows


def permutation_embedding_separation(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    genotype_distance,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    f0_key = F0_FLAG[evaluator]
    tdi_key = f"tdi_member_{evaluator}"
    labeled = []
    for r in rows:
        emb = r.get("_embedding")
        if emb is None:
            continue
        if r.get(f0_key):
            labeled.append((emb, 1))
        elif r.get(tdi_key) and int(r.get("species_id") or 0) > 0:
            labeled.append((emb, 0))

    if len(labeled) < 20:
        return {
            "evaluator": evaluator,
            "p_value": 1.0,
            "obs_stat": 0.0,
            "n": len(labeled),
            "skipped": True,
        }

    def stat(pairs):
        g0 = [e for e, y in pairs if y == 0]
        g1 = [e for e, y in pairs if y == 1]
        if not g0 or not g1:
            return 0.0
        dists = []
        for a in g0[:50]:
            for b in g1[:50]:
                dists.append(genotype_distance(a, b))
        return float(np.mean(dists)) if dists else 0.0

    obs = stat(labeled)
    rng = np.random.default_rng(seed)
    count = 0
    labels = [y for _, y in labeled]
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        pairs = [(labeled[i][0], int(perm_labels[i])) for i in range(len(labeled))]
        if stat(pairs) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {
        "evaluator": evaluator,
        "p_value": round(p, 6),
        "obs_stat": round(obs, 6),
        "n": len(labeled),
        "n_f0": sum(1 for _, y in labeled if y == 1),
        "n_dominated_member": sum(1 for _, y in labeled if y == 0),
        "skipped": False,
    }


def plot_rank_contingency_heatmap(
    contingency: Dict[str, int],
    out_path: Path,
    *,
    cap: int = 12,
    title: str = "",
) -> Optional[str]:
    if not contingency:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    mat = np.zeros((cap + 1, cap + 1), dtype=np.int32)
    for key, cnt in contingency.items():
        parts = key.split("_")
        try:
            rg = int(parts[0][1:])
            ro = int(parts[1][1:])
        except (IndexError, ValueError):
            continue
        if rg <= cap and ro <= cap:
            mat[rg, ro] += int(cnt)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, aspect="auto", cmap="Blues", origin="lower")
    ax.set_xlabel("OpenAI global rank")
    ax.set_ylabel("Google global rank")
    ax.set_title(title or f"Rank contingency (ranks 0–{cap})")
    fig.colorbar(im, ax=ax, fraction=0.03, label="count")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def axis_score_correlation(
    rows: Sequence[Dict[str, Any]],
    *,
    get_axis_order,
    openai_metric_aliases: Dict[str, str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    g_axes = list(get_axis_order("google"))
    o_axes = list(get_axis_order("openai"))
    o_index = {a: i for i, a in enumerate(o_axes)}
    pairs = []
    for g_axis in g_axes:
        o_axis = openai_metric_aliases.get(g_axis)
        if not o_axis or o_axis not in o_index:
            continue
        gi, oi = g_axes.index(g_axis), o_index[o_axis]
        vals_g, vals_o = [], []
        for r in rows:
            if "objective_vector" not in r or "objective_vector_openai" not in r:
                continue
            vals_g.append(float(r["objective_vector"][gi]))
            vals_o.append(float(r["objective_vector_openai"][oi]))
        if len(vals_g) < 10:
            continue
        spearman_r = float("nan")
        pearson_r = float("nan")
        try:
            from scipy.stats import pearsonr, spearmanr

            sr = spearmanr(vals_g, vals_o)
            pr = pearsonr(vals_g, vals_o)
            spearman_r = float(sr.correlation) if sr.correlation is not None else float("nan")
            pearson_r = float(pr.statistic) if pr.statistic is not None else float("nan")
        except Exception:
            pass
        pairs.append({
            "google_axis": g_axis,
            "openai_axis": o_axis,
            "n": len(vals_g),
            "spearman": round(spearman_r, 4) if spearman_r == spearman_r else None,
            "pearson": round(pearson_r, 4) if pearson_r == pearson_r else None,
        })

    meta = {
        "n_mappable_pairs": len(pairs),
        "n_genomes": sum(p["n"] for p in pairs[:1]) if pairs else 0,
    }
    return pairs, meta


__all__ = [
    "annotate_holm_bonferroni",
    "axis_score_correlation",
    "holm_bonferroni",
    "per_topic_permutation_separation",
    "permutation_embedding_separation",
    "plot_rank_contingency_heatmap",
    "stamp_tdi_members",
]
