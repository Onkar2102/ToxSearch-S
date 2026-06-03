"""Sensitivity sweeps: epsilon, threshold, and population-ablation counterfactuals."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from counterfactual_survival import policy_global_mo, topic_survival

_logger = logging.getLogger("emnlp.sensitivity_sweeps")

VECTOR_KEY = {"google": "objective_vector", "openai": "objective_vector_openai"}


def epsilon_dominates(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    return bool(np.all(a >= b + eps) and np.any(a > b + eps))


def epsilon_dominated_f0_mask(
    F: np.ndarray,
    eps: float,
    *,
    global_pareto_annotate,
) -> np.ndarray:
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    if eps <= 0:
        _, on_f0, _ = global_pareto_annotate(F)
        return on_f0
    ge = np.all(F[:, None, :] >= F[None, :, :] + eps, axis=2)
    gt = np.any(F[:, None, :] > F[None, :, :] + eps, axis=2)
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return ~dom.any(axis=0)


def epsilon_sweep(
    rows: Sequence[Dict[str, Any]],
    grid: Sequence[float],
    evaluator: str,
    *,
    min_topic_size: int,
    global_pareto_annotate,
    policy_global_mo,
    fast_non_dominated_sort,
) -> List[Dict[str, Any]]:
    vec_key = VECTOR_KEY[evaluator]
    valid = [r for r in rows if r.get(vec_key) is not None]
    if not valid:
        return []
    F = np.vstack([r[vec_key] for r in valid])
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)

    row_list = list(rows)
    out: List[Dict[str, Any]] = []
    for eps in grid:
        f0_mask = epsilon_dominated_f0_mask(F, eps, global_pareto_annotate=global_pareto_annotate)
        n_fully_dom = 0
        for sid, idxs in by_topic.items():
            if len(idxs) < min_topic_size:
                continue
            if not any(f0_mask[i] for i in idxs):
                n_fully_dom += 1
        k = int(np.sum(f0_mask)) or len(valid)
        kept = policy_global_mo(
            row_list,
            k,
            vec_key,
            fast_non_dominated_sort=fast_non_dominated_sort,
        )
        surv = topic_survival(kept, row_list, min_topic_size)
        out.append({
            "evaluator": evaluator,
            "epsilon": eps,
            "n_f0": int(np.sum(f0_mask)),
            "n_fully_dominated_topics": n_fully_dom,
            "global_mo_survival_rate": surv["survival_rate"],
        })
    return out


def threshold_sweep(
    rows: Sequence[Dict[str, Any]],
    tau_intra_grid: Sequence[float],
    tau_inter_grid: Sequence[float],
    *,
    min_topic_size: int,
    global_pareto_annotate,
    topic_centroids,
    distinct_topic_label,
) -> List[Dict[str, Any]]:
    valid = [r for r in rows if r.get("objective_vector") is not None]
    if not valid:
        return []
    F = np.vstack([r["objective_vector"] for r in valid])
    _, on_f0, _ = global_pareto_annotate(F)
    centroids = topic_centroids(valid, "google")

    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)

    base_topics = []
    for sid, idxs in sorted(by_topic.items()):
        if len(idxs) < min_topic_size:
            continue
        n_on_f0 = int(np.sum(on_f0[idxs]))
        max_vec = F[idxs].max(axis=0)
        f0_max = F[on_f0].max(axis=0) if np.any(on_f0) else F.max(axis=0)
        axis_exclusive = sum(1 for i in range(F.shape[1]) if max_vec[i] > f0_max[i] + 1e-9)
        base_topics.append({
            "species_id": sid,
            "fully_dominated": n_on_f0 == 0,
            "n_axis_exclusive": axis_exclusive,
        })

    out: List[Dict[str, Any]] = []
    for ti in tau_intra_grid:
        for te in tau_inter_grid:
            n_distinct_dominated = 0
            n_axis_exclusive_dominated = 0
            for topic in base_topics:
                sid = topic["species_id"]
                distinct, _, _ = distinct_topic_label(
                    sid, centroids, tau_intra=ti, tau_inter=te
                )
                if topic["fully_dominated"] and distinct:
                    n_distinct_dominated += 1
                if topic["fully_dominated"] and topic["n_axis_exclusive"] > 0:
                    n_axis_exclusive_dominated += 1
            out.append({
                "tau_intra": ti,
                "tau_inter": te,
                "n_distinct_dominated": n_distinct_dominated,
                "n_axis_exclusive_dominated": n_axis_exclusive_dominated,
            })
    return out


def population_ablation_counterfactual(
    rows: Sequence[Dict[str, Any]],
    population: str,
    evaluator: str,
    *,
    min_topic_size: int,
    run_counterfactual_for_evaluator,
    fast_non_dominated_sort,
    global_pareto_annotate,
) -> List[Dict[str, Any]]:
    if population == "elites":
        sub = [r for r in rows if r.get("source_file") == "elites.json"]
    elif population == "no_reserves":
        sub = [r for r in rows if r.get("source_file") != "reserves.json"]
    else:
        sub = list(rows)
    results = run_counterfactual_for_evaluator(
        sub,
        evaluator,
        min_topic_size=min_topic_size,
        fast_non_dominated_sort=fast_non_dominated_sort,
        global_pareto_annotate=global_pareto_annotate,
    )
    for r in results:
        r["population"] = population
    return results


def plot_epsilon_sweep_topics(
    google_rows: Sequence[Dict[str, Any]],
    openai_rows: Sequence[Dict[str, Any]],
    out_path,
    *,
    title: str = "",
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    for rows, label, color in (
        (google_rows, "Google", "#4C78A8"),
        (openai_rows, "OpenAI", "#F58518"),
    ):
        if not rows:
            continue
        eps = [r["epsilon"] for r in rows]
        n_dom = [r["n_fully_dominated_topics"] for r in rows]
        ax.plot(eps, n_dom, marker="o", label=label, color=color)
    ax.set_xlabel("ε (epsilon-dominance)")
    ax.set_ylabel("Fully dominated topics (count)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    ax.set_title(title or "Fully dominated topic count vs ε")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "epsilon_sweep",
    "plot_epsilon_sweep_topics",
    "population_ablation_counterfactual",
    "threshold_sweep",
]
