"""Counterfactual topic survival under scalar / global-MO / speciated-oracle policies."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

_logger = logging.getLogger("emnlp.counterfactual_survival")

EVALUATORS = ("google", "openai")
VECTOR_KEY = {"google": "objective_vector", "openai": "objective_vector_openai"}


def _crowding_distance(F: np.ndarray, front: List[int]) -> np.ndarray:
    n = len(front)
    cd = np.zeros(n, dtype=np.float64)
    if n <= 2:
        cd[:] = np.inf
        return cd
    for m in range(F.shape[1]):
        vals = F[front, m]
        order = np.argsort(vals)
        vmin, vmax = vals[order[0]], vals[order[-1]]
        span = vmax - vmin
        if span < 1e-12:
            continue
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        for i in range(1, n - 1):
            cd[order[i]] += (vals[order[i + 1]] - vals[order[i - 1]]) / span
    return cd


def _indices_by_topic(
    row_list: Sequence[Dict[str, Any]],
    min_topic_size: int,
) -> Dict[int, List[int]]:
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(row_list):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)
    return {sid: idxs for sid, idxs in by_topic.items() if len(idxs) >= min_topic_size}


def policy_scalar(
    row_list: Sequence[Dict[str, Any]],
    k: int,
    vec_key: str,
) -> Set[int]:
    scored = [
        (i, float(row_list[i][vec_key][0]))
        for i, r in enumerate(row_list)
        if vec_key in r
    ]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return {i for i, _ in scored[:k]}


def policy_global_mo(
    row_list: Sequence[Dict[str, Any]],
    cap: int,
    vec_key: str,
    *,
    fast_non_dominated_sort,
) -> Set[int]:
    valid_idx = [i for i, r in enumerate(row_list) if vec_key in r]
    if not valid_idx:
        return set()
    F = np.vstack([row_list[i][vec_key] for i in valid_idx])
    fronts = fast_non_dominated_sort(F)
    f0_local = fronts[0] if fronts else list(range(len(valid_idx)))
    f0_global = [valid_idx[i] for i in f0_local]
    if len(f0_global) <= cap:
        return set(f0_global)
    F0 = F[f0_local]
    cd = _crowding_distance(F0, list(range(len(f0_local))))
    order = sorted(range(len(f0_local)), key=lambda k: (-cd[k], f0_global[k]))
    return {f0_global[i] for i in order[:cap]}


def policy_speciated_oracle(
    row_list: Sequence[Dict[str, Any]],
    min_topic_size: int,
    vec_key: str,
    *,
    fast_non_dominated_sort,
) -> Set[int]:
    by_topic = _indices_by_topic(row_list, min_topic_size)
    kept: Set[int] = set()
    for idxs in by_topic.values():
        F = np.vstack([row_list[i][vec_key] for i in idxs])
        fronts = fast_non_dominated_sort(F)
        if fronts:
            kept.update(idxs[i] for i in fronts[0])
    return kept


def topic_survival(
    kept: Set[int],
    row_list: Sequence[Dict[str, Any]],
    min_topic_size: int,
) -> Dict[str, Any]:
    by_topic = _indices_by_topic(row_list, min_topic_size)
    total = len(by_topic)
    survived = sum(1 for sid, idxs in by_topic.items() if any(i in kept for i in idxs))
    rate = survived / total if total else 0.0
    return {
        "topics_total": total,
        "topics_survived": survived,
        "survival_rate": round(rate, 4),
        "extinction_rate": round(1.0 - rate, 4),
    }


def wilson_interval(
    successes: int,
    n: int,
    *,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        return (0.0, 0.0)
    # Inverse normal CDF for two-sided alpha. Use scipy if available, else
    # the textbook 95% z=1.96 fallback (only 95% supported in fallback).
    try:
        from scipy.stats import norm  # type: ignore

        z = float(norm.ppf(1.0 - (1.0 - confidence) / 2.0))
    except ImportError:
        z = 1.959963984540054 if abs(confidence - 0.95) < 1e-6 else 1.959963984540054
    p_hat = successes / n
    denom = 1.0 + (z * z) / n
    center = (p_hat + (z * z) / (2.0 * n)) / denom
    half = (
        z
        * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z * z) / (4.0 * n * n))
        / denom
    )
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return (float(lo), float(hi))


def bootstrap_topic_survival_ci(
    row_list: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    policy: str,
    min_topic_size: int,
    fast_non_dominated_sort,
    global_pareto_annotate,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Bootstrap CI on topic-survival rate by resampling genomes with replacement.

    Each replicate draws ``len(row_list)`` row indices with replacement and
    recomputes the policy's kept set (using the same row-indexing space) and
    topic survival on the resampled population.
    """
    vec_key = VECTOR_KEY[evaluator]
    rng = np.random.default_rng(seed)
    n = len(row_list)
    if n == 0 or n_bootstrap <= 0:
        return {"bootstrap_ci_lower": 0.0, "bootstrap_ci_upper": 0.0}

    # Default k corresponds to the full population's |F0|.
    valid_idx_full = [i for i, r in enumerate(row_list) if vec_key in r]
    if not valid_idx_full:
        return {"bootstrap_ci_lower": 0.0, "bootstrap_ci_upper": 0.0}
    F_full = np.vstack([row_list[i][vec_key] for i in valid_idx_full])
    _, on_f0_full, _ = global_pareto_annotate(F_full)
    k_default = int(np.sum(on_f0_full)) if np.any(on_f0_full) else len(valid_idx_full)

    rates: List[float] = []
    for _ in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        sub = [row_list[i] for i in sample_idx]
        if policy == "scalar":
            kept = policy_scalar(sub, k_default, vec_key)
        elif policy == "global_mo":
            kept = policy_global_mo(
                sub, k_default, vec_key,
                fast_non_dominated_sort=fast_non_dominated_sort,
            )
        elif policy == "speciated_oracle":
            kept = policy_speciated_oracle(
                sub, min_topic_size, vec_key,
                fast_non_dominated_sort=fast_non_dominated_sort,
            )
        else:
            raise ValueError(f"Unknown policy {policy!r}")
        stats = topic_survival(kept, sub, min_topic_size)
        rates.append(float(stats["survival_rate"]))

    arr = np.asarray(rates, dtype=np.float64)
    alpha = (1.0 - confidence) / 2.0
    lo = float(np.quantile(arr, alpha))
    hi = float(np.quantile(arr, 1.0 - alpha))
    return {
        "bootstrap_ci_lower": round(lo, 4),
        "bootstrap_ci_upper": round(hi, 4),
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_mean": round(float(arr.mean()), 4),
    }


def annotate_survival_with_cis(
    rows: List[Dict[str, Any]],
    row_list: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    min_topic_size: int,
    fast_non_dominated_sort,
    global_pareto_annotate,
    bootstrap_policies: Sequence[str] = ("global_mo",),
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Append Wilson interval to every row and bootstrap CI for selected policies."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("evaluator") != evaluator:
            out.append(r)
            continue
        n_total = int(r.get("topics_total") or 0)
        n_surv = int(r.get("topics_survived") or 0)
        wlo, whi = wilson_interval(n_surv, n_total, confidence=confidence)
        new_row = dict(r)
        new_row["wilson_ci_lower"] = round(wlo, 4)
        new_row["wilson_ci_upper"] = round(whi, 4)
        new_row["confidence"] = float(confidence)
        if r.get("policy") in bootstrap_policies:
            ci = bootstrap_topic_survival_ci(
                row_list,
                evaluator,
                policy=str(r["policy"]),
                min_topic_size=min_topic_size,
                fast_non_dominated_sort=fast_non_dominated_sort,
                global_pareto_annotate=global_pareto_annotate,
                n_bootstrap=n_bootstrap,
                confidence=confidence,
                seed=seed,
            )
            new_row.update(ci)
        out.append(new_row)
    return out


def run_counterfactual_for_evaluator(
    row_list: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    min_topic_size: int = 5,
    k_override: Optional[int] = None,
    fast_non_dominated_sort,
    global_pareto_annotate,
) -> List[Dict[str, Any]]:
    vec_key = VECTOR_KEY[evaluator]
    valid = [r for r in row_list if vec_key in r]
    if not valid:
        return []

    valid_idx = [i for i, r in enumerate(row_list) if vec_key in r]
    F = np.vstack([row_list[i][vec_key] for i in valid_idx])
    _, on_f0, _ = global_pareto_annotate(F)
    k_default = int(np.sum(on_f0)) if np.any(on_f0) else len(valid_idx)
    k = k_override if k_override is not None else k_default

    results: List[Dict[str, Any]] = []
    for policy_name, kept in [
        (
            "scalar",
            policy_scalar(row_list, k, vec_key),
        ),
        (
            "global_mo",
            policy_global_mo(row_list, k, vec_key, fast_non_dominated_sort=fast_non_dominated_sort),
        ),
        (
            "speciated_oracle",
            policy_speciated_oracle(
                row_list,
                min_topic_size,
                vec_key,
                fast_non_dominated_sort=fast_non_dominated_sort,
            ),
        ),
    ]:
        stats = topic_survival(kept, row_list, min_topic_size)
        stats.update({
            "evaluator": evaluator,
            "policy": policy_name,
            "kept_size": len(kept),
            "k": k,
            "n_f0": k_default,
        })
        results.append(stats)
    return results


def run_counterfactual_sensitivity_k(
    row_list: Sequence[Dict[str, Any]],
    evaluator: str,
    k_grid: Sequence[int],
    *,
    min_topic_size: int,
    fast_non_dominated_sort,
    global_pareto_annotate,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for k in k_grid:
        for policy in ("scalar", "global_mo"):
            batch = run_counterfactual_for_evaluator(
                row_list,
                evaluator,
                min_topic_size=min_topic_size,
                k_override=k,
                fast_non_dominated_sort=fast_non_dominated_sort,
                global_pareto_annotate=global_pareto_annotate,
            )
            for r in batch:
                if r["policy"] == policy:
                    rows.append({**r, "sensitivity_k": k})
    return rows


def compare_evaluators_survival(
    google_rows: Sequence[Dict[str, Any]],
    openai_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"policies": {}}
    for policy in ("scalar", "global_mo", "speciated_oracle"):
        g = next((r for r in google_rows if r["policy"] == policy), {})
        o = next((r for r in openai_rows if r["policy"] == policy), {})
        out["policies"][policy] = {
            "google_survival_rate": g.get("survival_rate"),
            "openai_survival_rate": o.get("survival_rate"),
            "google_extinction_rate": g.get("extinction_rate"),
            "openai_extinction_rate": o.get("extinction_rate"),
            "google_topics_survived": g.get("topics_survived"),
            "openai_topics_survived": o.get("topics_survived"),
        }
    g_rates = [r["survival_rate"] for r in google_rows]
    o_rates = [r["survival_rate"] for r in openai_rows]
    out["hierarchy_google"] = g_rates == sorted(g_rates)
    out["hierarchy_openai"] = o_rates == sorted(o_rates)
    return out


def plot_counterfactual_bars(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "",
) -> Optional[str]:
    if not rows:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    order = ["scalar", "global_mo", "speciated_oracle"]
    labels = ["Scalar top-k", "Global MO", "Speciated oracle"]
    by_policy = {r["policy"]: float(r["survival_rate"]) for r in rows}

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(labels, [by_policy.get(p, 0.0) for p in order], color=["#888888", "#4C78A8", "#54A24B"])
    ax.set_ylabel("Topic survival rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title or "Counterfactual retention")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_counterfactual_dual(
    google_rows: Sequence[Dict[str, Any]],
    openai_rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    title: str = "",
) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    policies = ["scalar", "global_mo", "speciated_oracle"]
    labels = ["Scalar", "Global MO", "Oracle"]
    x = np.arange(len(policies))
    w = 0.35
    g = [float(next(r["survival_rate"] for r in google_rows if r["policy"] == p)) for p in policies]
    o = [float(next(r["survival_rate"] for r in openai_rows if r["policy"] == p)) for p in policies]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - w / 2, g, w, label="Google", color="#4C78A8")
    ax.bar(x + w / 2, o, w, label="OpenAI", color="#F58518")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Topic survival rate")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(title or "Counterfactual retention by evaluator")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "EVALUATORS",
    "annotate_survival_with_cis",
    "bootstrap_topic_survival_ci",
    "compare_evaluators_survival",
    "plot_counterfactual_bars",
    "plot_counterfactual_dual",
    "run_counterfactual_for_evaluator",
    "run_counterfactual_sensitivity_k",
    "wilson_interval",
]
