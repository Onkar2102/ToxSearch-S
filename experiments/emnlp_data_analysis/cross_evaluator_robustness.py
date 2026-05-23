"""Cross-evaluator robustness: Jaccard metrics and hierarchy checks across evaluators."""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_logger = logging.getLogger("emnlp.cross_evaluator_robustness")


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def _load_topic_summaries(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_counterfactual_combined(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _survived_topic_sets(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    policy: str,
    *,
    min_topic_size: int,
    row_list: Sequence[Dict[str, Any]],
    policy_fn,
) -> set:
    """Topics with ≥1 member kept under policy (recompute if CSV lacks topic ids)."""
    from counterfactual_survival import VECTOR_KEY

    vec_key = VECTOR_KEY[evaluator]
    cf_row = next(
        (r for r in rows if r.get("evaluator") == evaluator and r.get("policy") == policy),
        None,
    )
    if cf_row is None:
        return set()
    k = int(float(cf_row.get("k") or cf_row.get("kept_size") or 0))
    if policy == "scalar":
        from counterfactual_survival import policy_scalar

        kept = policy_scalar(row_list, k, vec_key)
    elif policy == "global_mo":
        from counterfactual_survival import policy_global_mo

        kept = policy_fn(row_list, k, vec_key)
    else:
        from counterfactual_survival import policy_speciated_oracle

        kept = policy_fn(row_list, min_topic_size, vec_key)
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(row_list):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)
    return {
        sid
        for sid, idxs in by_topic.items()
        if len(idxs) >= min_topic_size and any(i in kept for i in idxs)
    }


def build_cross_evaluator_robustness(
    *,
    phase3_dir: Path,
    phase5_dir: Path,
    rank_summary: Dict[str, Any],
    topic_comparison_rows: Sequence[Dict[str, Any]],
    counterfactual_combined: Sequence[Dict[str, Any]],
    row_list: Optional[Sequence[Dict[str, Any]]] = None,
    min_topic_size: int = 5,
    fast_non_dominated_sort=None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    g_sum = _load_topic_summaries(phase3_dir / "google" / "topic_domination_summary.csv")
    o_sum = _load_topic_summaries(phase3_dir / "openai" / "topic_domination_summary.csv")

    g_fully_dom = {int(s["species_id"]) for s in g_sum if s.get("fully_dominated") in ("True", True, "1", 1)}
    o_fully_dom = {int(s["species_id"]) for s in o_sum if s.get("fully_dominated") in ("True", True, "1", 1)}

    metrics: List[Dict[str, Any]] = []
    metrics.append({
        "metric": "jaccard_fully_dominated_topics",
        "value": round(jaccard(g_fully_dom, o_fully_dom), 4),
        "google_n": len(g_fully_dom),
        "openai_n": len(o_fully_dom),
    })

    if rank_summary.get("jaccard_f0_sets") is not None:
        metrics.append({
            "metric": "jaccard_global_f0_genomes",
            "value": rank_summary.get("jaccard_f0_sets"),
            "google_n": rank_summary.get("n_google_f0"),
            "openai_n": rank_summary.get("n_openai_f0"),
        })

    pattern_match = 0
    pattern_total = 0
    for row in topic_comparison_rows:
        pattern_total += 1
        pat = row.get("fully_dominated_pattern")
        g_fd = row.get("fully_dominated_google") in ("True", True, "1", 1)
        o_fd = row.get("fully_dominated_openai") in ("True", True, "1", 1)
        if pat in ("both", "neither") or (g_fd == o_fd):
            pattern_match += 1
    if pattern_total:
        metrics.append({
            "metric": "frac_fully_dominated_agreement",
            "value": round(pattern_match / pattern_total, 4),
            "n_topics": pattern_total,
        })

    g_rates = []
    o_rates = []
    for policy in ("scalar", "global_mo", "speciated_oracle"):
        g = next(
            (r for r in counterfactual_combined if r.get("evaluator") == "google" and r.get("policy") == policy),
            {},
        )
        o = next(
            (r for r in counterfactual_combined if r.get("evaluator") == "openai" and r.get("policy") == policy),
            {},
        )
        gr = float(g.get("survival_rate") or 0)
        or_ = float(o.get("survival_rate") or 0)
        g_rates.append(gr)
        o_rates.append(or_)
        metrics.append({
            "metric": f"counterfactual_survival_{policy}",
            "google": gr,
            "openai": or_,
            "delta_openai_minus_google": round(or_ - gr, 4),
        })

    metrics.append({
        "metric": "counterfactual_hierarchy_scalar_lt_global_lt_oracle",
        "google": g_rates == sorted(g_rates) if len(g_rates) == 3 else None,
        "openai": o_rates == sorted(o_rates) if len(o_rates) == 3 else None,
    })

    summary = {
        "n_metrics": len(metrics),
        "jaccard_fully_dominated_topics": metrics[0]["value"] if metrics else None,
        "rank_agreement": rank_summary,
    }
    return metrics, summary


def plot_cross_evaluator_summary(
    metrics: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    j_rows = [m for m in metrics if m.get("metric", "").startswith("jaccard") or m.get("metric", "").startswith("frac")]
    if not j_rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = [m["metric"].replace("jaccard_", "").replace("frac_", "")[:24] for m in j_rows]
    vals = [float(m.get("value") or 0) for m in j_rows]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(labels, vals, color="#4C78A8")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Agreement / Jaccard")
    ax.set_title("Cross-evaluator robustness metrics")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = ["build_cross_evaluator_robustness", "jaccard", "plot_cross_evaluator_summary"]
