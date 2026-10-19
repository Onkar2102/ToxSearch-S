#!/usr/bin/env python3
"""Simulate IncDBSCAN ε-neighborhoods as |D_t| grows to N_max.

Compares insert regimes against **exact** IncDBSCAN on the same prefix at
each checkpoint (no batch / sklearn DBSCAN):

  1. exact IncDBSCAN  — true ε-balls (ToxSearch insert semantics)
  2. approx IncDBSCAN — each true ε-neighbor kept w.p. ``recall`` (ANN-style miss)

Data: real unit-normalized embeddings from an IncDBSCAN run, extended to N_max
with on-sphere perturbations of the empirical cloud (variant-like) plus a small
fraction of fresh uniform directions (exploration).

Outputs under ``results/incdbscan_neighborhood_sim/``:
  metrics.csv, summary.json, plots (ARI / Jaccard / clusters / time).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Distance (ToxSearch d_E)
# ---------------------------------------------------------------------------


def unit_normalize_rows(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(nrm, 1e-12)


def pairwise_d_E(X: np.ndarray) -> np.ndarray:
    """Pairwise embedding distance (1 - cos) / 2 on L2-normalized rows."""
    sims = np.clip(X @ X.T, -1.0, 1.0)
    D = (1.0 - sims) / 2.0
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def d_E_to_query(query: np.ndarray, X: np.ndarray) -> np.ndarray:
    sims = np.clip(X @ query, -1.0, 1.0)
    return (1.0 - sims) / 2.0


# ---------------------------------------------------------------------------
# Incremental DBSCAN (insert-only), exact or approximate neighborhoods
# ---------------------------------------------------------------------------


@dataclass
class IncState:
    eps: float
    min_samples: int
    capacity: int
    embeddings: Optional[np.ndarray] = None  # (capacity, d), filled [:n]
    n: int = 0
    labels: List[int] = field(default_factory=list)  # -1 noise, ≥0 cluster
    is_core: List[bool] = field(default_factory=list)
    neighbors: List[Set[int]] = field(default_factory=list)
    next_cid: int = 0

    def n_points(self) -> int:
        return self.n


def _expand(state: IncState, seed: int, cid: int) -> None:
    q = deque([seed])
    seen = {seed}
    while q:
        i = q.popleft()
        state.labels[i] = cid
        if not state.is_core[i]:
            continue
        for j in state.neighbors[i]:
            if j in seen:
                continue
            seen.add(j)
            # Absorb unlabeled / noise into this cluster (Ester-style insert expand)
            if state.labels[j] == -1 or state.labels[j] == cid:
                q.append(j)


def insert_point(
    state: IncState,
    emb: np.ndarray,
    *,
    recall: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> None:
    """Insert one point. ``recall`` < 1 randomly drops true ε-neighbors (ANN sim)."""
    n_old = state.n_points()
    if state.embeddings is None:
        state.embeddings = np.empty((state.capacity, emb.shape[0]), dtype=np.float64)

    if n_old == 0:
        state.embeddings[0] = emb
        state.n = 1
        state.labels.append(-1)
        state.neighbors.append({0})
        state.is_core.append(1 >= state.min_samples)
        if state.is_core[0]:
            state.labels[0] = state.next_cid
            state.next_cid += 1
        return

    dists = d_E_to_query(emb, state.embeddings[:n_old])
    true_nbrs = {int(j) for j in np.flatnonzero(dists <= state.eps)}

    if recall >= 1.0 - 1e-15:
        nbrs = set(true_nbrs)
    else:
        assert rng is not None
        nbrs = {j for j in true_nbrs if rng.random() < recall}

    new_idx = n_old
    state.embeddings[new_idx] = emb
    state.n = n_old + 1
    state.labels.append(-1)
    state.neighbors.append(set(nbrs) | {new_idx})
    state.is_core.append(False)

    for j in nbrs:
        state.neighbors[j].add(new_idx)

    affected = set(nbrs) | {new_idx}
    for i in affected:
        state.is_core[i] = len(state.neighbors[i]) >= state.min_samples

    core_ids: Set[int] = set()
    for j in nbrs:
        if state.is_core[j] and state.labels[j] >= 0:
            core_ids.add(int(state.labels[j]))

    new_is_core = state.is_core[new_idx]

    if not new_is_core and not core_ids:
        return  # noise

    if not new_is_core and core_ids:
        state.labels[new_idx] = min(core_ids)
        return

    # new point is core
    if not core_ids:
        cid = state.next_cid
        state.next_cid += 1
        _expand(state, new_idx, cid)
        return

    if len(core_ids) == 1:
        cid = next(iter(core_ids))
        _expand(state, new_idx, cid)
        return

    # Merge: survivor = lowest id
    survivor = min(core_ids)
    _expand(state, new_idx, survivor)
    for i, lab in enumerate(state.labels):
        if lab in core_ids:
            state.labels[i] = survivor


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def contingency_ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Adjusted Rand Index without sklearn."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    n = len(y_true)
    if n < 2:
        return 1.0

    def _comb2(x: np.ndarray) -> np.ndarray:
        return x * (x - 1) / 2.0

    # Contingency
    classes_t, t_inv = np.unique(y_true, return_inverse=True)
    classes_p, p_inv = np.unique(y_pred, return_inverse=True)
    C = np.zeros((len(classes_t), len(classes_p)), dtype=np.float64)
    for i in range(n):
        C[t_inv[i], p_inv[i]] += 1.0

    sum_comb_c = _comb2(C).sum()
    sum_comb_t = _comb2(C.sum(axis=1)).sum()
    sum_comb_p = _comb2(C.sum(axis=0)).sum()
    comb_n = n * (n - 1) / 2.0
    if comb_n == 0:
        return 1.0
    expected = sum_comb_t * sum_comb_p / comb_n
    max_index = 0.5 * (sum_comb_t + sum_comb_p)
    denom = max_index - expected
    if denom == 0:
        return 1.0
    return float((sum_comb_c - expected) / denom)


def mean_neighborhood_jaccard(
    nbrs_a: Sequence[Set[int]],
    nbrs_b: Sequence[Set[int]],
) -> float:
    n = min(len(nbrs_a), len(nbrs_b))
    if n == 0:
        return 1.0
    acc = 0.0
    for i in range(n):
        a, b = nbrs_a[i], nbrs_b[i]
        inter = len(a & b)
        union = len(a | b)
        acc += 1.0 if union == 0 else inter / union
    return acc / n


def n_clusters_and_noise(labels: np.ndarray) -> Tuple[int, float]:
    labs = np.asarray(labels)
    noise = float(np.mean(labs < 0))
    pos = labs[labs >= 0]
    k = int(len(np.unique(pos))) if len(pos) else 0
    return k, noise


# ---------------------------------------------------------------------------
# Data: load + extend to N_max
# ---------------------------------------------------------------------------


def load_base_embeddings(state_json: Path) -> Tuple[np.ndarray, float, int]:
    with open(state_json) as f:
        s = json.load(f)
    E = unit_normalize_rows(np.asarray(s["embeddings"], dtype=np.float64))
    eps = float(s.get("eps", 0.25))
    min_samples = int(s.get("min_samples", 2))
    return E, eps, min_samples


def extend_embeddings(
    E0: np.ndarray,
    n_max: int,
    *,
    rng: np.random.Generator,
    local_frac: float = 0.85,
    local_noise: float = 0.08,
) -> np.ndarray:
    """Grow to n_max: mostly local variants of existing points + some uniform."""
    n0, d = E0.shape
    if n_max <= n0:
        return E0[:n_max].copy()
    out = np.empty((n_max, d), dtype=np.float64)
    out[:n0] = E0
    n_local = int(round((n_max - n0) * local_frac))
    n_fresh = (n_max - n0) - n_local
    # Local: pick parent + isotropic noise, re-normalize
    parents = rng.integers(0, n0, size=n_local)
    noise = rng.normal(0.0, local_noise, size=(n_local, d))
    local = E0[parents] + noise
    out[n0 : n0 + n_local] = unit_normalize_rows(local)
    if n_fresh > 0:
        fresh = rng.normal(0.0, 1.0, size=(n_fresh, d))
        out[n0 + n_local :] = unit_normalize_rows(fresh)
    return out


def generation_schedule(n_max: int, delta: int, seed_n: int) -> List[int]:
    """Cumulative sizes after each generation (seed then +delta)."""
    sizes = [min(seed_n, n_max)]
    while sizes[-1] < n_max:
        sizes.append(min(sizes[-1] + delta, n_max))
    return sizes


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_simulation(
    X: np.ndarray,
    *,
    eps: float,
    min_samples: int,
    delta: int,
    seed_n: int,
    checkpoints: Sequence[int],
    recalls: Sequence[float],
    rng: np.random.Generator,
) -> List[dict]:
    n_max = X.shape[0]
    sizes = generation_schedule(n_max, delta, seed_n)
    checkpoint_set = set(int(c) for c in checkpoints if c <= n_max)
    checkpoint_set.add(n_max)

    # Optional: time full pairwise d_E (not used for clustering)
    t0 = time.perf_counter()
    _ = pairwise_d_E(X)
    t_dist = time.perf_counter() - t0

    # Incremental states: exact + approx per recall
    states: Dict[str, IncState] = {
        "inc_exact": IncState(eps=eps, min_samples=min_samples, capacity=n_max),
    }
    for r in recalls:
        if abs(r - 1.0) < 1e-15:
            continue
        states[f"inc_approx_r{r:.2f}"] = IncState(
            eps=eps, min_samples=min_samples, capacity=n_max
        )

    rows: List[dict] = []
    next_insert = 0
    gen = 0
    t_inc: Dict[str, float] = {k: 0.0 for k in states}

    for target_n in sizes:
        # Insert [next_insert, target_n)
        for i in range(next_insert, target_n):
            emb = X[i]
            for name, st in states.items():
                t1 = time.perf_counter()
                if name == "inc_exact":
                    insert_point(st, emb, recall=1.0)
                else:
                    r = float(name.split("r")[-1])
                    insert_point(st, emb, recall=r, rng=rng)
                t_inc[name] += time.perf_counter() - t1
        next_insert = target_n
        gen += 1

        if target_n not in checkpoint_set:
            continue

        # Reference = exact IncDBSCAN state at this prefix
        ref = states["inc_exact"]
        ref_labels = np.asarray(ref.labels, dtype=np.int32)
        ref_nbrs = ref.neighbors
        k_ref, noise_ref = n_clusters_and_noise(ref_labels)

        for name, st in states.items():
            inc_labels = np.asarray(st.labels, dtype=np.int32)
            assert len(inc_labels) == target_n
            ari = contingency_ari(ref_labels, inc_labels) if name != "inc_exact" else 1.0
            jacc = mean_neighborhood_jaccard(st.neighbors, ref_nbrs)
            k_i, noise_i = n_clusters_and_noise(inc_labels)
            # Neighborhood recall vs exact IncDBSCAN
            recall_emp = []
            for i in range(target_n):
                true = ref_nbrs[i]
                got = st.neighbors[i]
                if len(true) <= 1:
                    recall_emp.append(1.0)
                else:
                    recall_emp.append(len(got & true) / len(true))
            rows.append(
                {
                    "generation": gen,
                    "n": target_n,
                    "method": name,
                    "ari_vs_exact": ari,
                    "mean_nbr_jaccard": jacc,
                    "mean_nbr_recall": float(np.mean(recall_emp)),
                    "n_clusters_inc": k_i,
                    "n_clusters_exact": k_ref,
                    "noise_frac_inc": noise_i,
                    "noise_frac_exact": noise_ref,
                    "cum_insert_seconds": t_inc[name],
                    "pairwise_dE_seconds_full": t_dist,
                    "eps": eps,
                    "min_samples": min_samples,
                }
            )
    return rows


def plot_results(rows: List[dict], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = sorted({r["method"] for r in rows})
    # Color map
    cmap = {
        "inc_exact": "#1b9e77",
        "inc_approx_r0.95": "#d95f02",
        "inc_approx_r0.90": "#7570b3",
        "inc_approx_r0.70": "#e7298a",
    }

    def series(method: str, key: str):
        pts = sorted([r for r in rows if r["method"] == method], key=lambda x: x["n"])
        return [p["n"] for p in pts], [p[key] for p in pts]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    for m in methods:
        if m == "inc_exact":
            continue
        xs, ys = series(m, "ari_vs_exact")
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=cmap.get(m, None), label=m)
    ax.set_xlabel(r"$n = |D_t|$")
    ax.set_ylabel("ARI vs exact IncDBSCAN")
    ax.set_title("Partition agreement (approx vs exact)")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    for m in methods:
        xs, ys = series(m, "mean_nbr_jaccard")
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=cmap.get(m, None), label=m)
    ax.set_xlabel(r"$n = |D_t|$")
    ax.set_ylabel(r"Mean Jaccard($N_\varepsilon, N_\varepsilon^{\mathrm{exact}}$)")
    ax.set_title("ε-neighborhood agreement")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for m in methods:
        xs, ys = series(m, "n_clusters_inc")
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=cmap.get(m, None), label=m)
    ax.set_xlabel(r"$n = |D_t|$")
    ax.set_ylabel("# clusters (non-noise)")
    ax.set_title("Cluster count")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    ax = axes[1, 1]
    for m in methods:
        xs, ys = series(m, "cum_insert_seconds")
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.6, color=cmap.get(m, None), label=f"{m} cum.")
    ax.set_xlabel(r"$n = |D_t|$")
    ax.set_ylabel("Seconds")
    ax.set_title("Cumulative insert runtime")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7)

    fig.suptitle(
        f"IncDBSCAN neighborhoods (exact vs approx; eps={rows[0]['eps']}, "
        f"min_samples={rows[0]['min_samples']})",
        fontsize=11,
    )
    fig.savefig(out_dir / "neighborhood_sim.png", dpi=160)
    fig.savefig(out_dir / "neighborhood_sim.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--state-json",
        type=Path,
        default=Path(
            "data/outputs/slurm_seq_20261014_031959_google_embedding_inc_dbscan_s2027_g1000"
            "/inc_dbscan_state.json"
        ),
    )
    ap.add_argument("--n-max", type=int, default=10000)
    ap.add_argument("--delta", type=int, default=50, help="New genomes per simulated generation")
    ap.add_argument("--seed-n", type=int, default=100, help="Gen-0 seed size")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--min-samples", type=int, default=None)
    ap.add_argument(
        "--checkpoints",
        type=str,
        default="100,250,500,750,1000,1500,2000,3000,4000,5000,6000,7000,8000,9000,10000",
    )
    ap.add_argument("--recalls", type=str, default="0.95,0.90,0.70")
    ap.add_argument("--rng-seed", type=int, default=2027)
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/incdbscan_neighborhood_sim"),
    )
    args = ap.parse_args()

    os.environ.setdefault("MPLCONFIGDIR", str(args.out_dir / ".mplconfig"))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.rng_seed)
    E0, eps0, ms0 = load_base_embeddings(args.state_json)
    eps = float(args.eps if args.eps is not None else eps0)
    min_samples = int(args.min_samples if args.min_samples is not None else ms0)

    print(f"Loaded base embeddings {E0.shape}; extending to n_max={args.n_max} …")
    X = extend_embeddings(E0, args.n_max, rng=rng)
    checkpoints = [int(x) for x in args.checkpoints.split(",") if x.strip()]
    recalls = [float(x) for x in args.recalls.split(",") if x.strip()]

    print(
        f"Running simulation: eps={eps}, min_samples={min_samples}, "
        f"delta={args.delta}, recalls={recalls}"
    )
    t0 = time.perf_counter()
    rows = run_simulation(
        X,
        eps=eps,
        min_samples=min_samples,
        delta=args.delta,
        seed_n=min(args.seed_n, args.n_max),
        checkpoints=checkpoints,
        recalls=recalls,
        rng=rng,
    )
    elapsed = time.perf_counter() - t0
    print(f"Simulation finished in {elapsed:.1f}s; {len(rows)} metric rows")

    csv_path = args.out_dir / "metrics.csv"
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Compact summary at n_max
    final = [r for r in rows if r["n"] == args.n_max]
    summary = {
        "n_max": args.n_max,
        "eps": eps,
        "min_samples": min_samples,
        "delta": args.delta,
        "seed_n": args.seed_n,
        "base_embeddings": str(args.state_json),
        "base_n": int(E0.shape[0]),
        "recalls": recalls,
        "wall_seconds": elapsed,
        "final": final,
        "note": (
            "Exact IncDBSCAN uses true ε-neighborhoods (Ester-style insert). "
            "Approx variants randomly drop true neighbors with keep-prob=recall "
            "to simulate ANN range-query misses. Metrics are vs exact IncDBSCAN "
            "(no batch DBSCAN)."
        ),
    }
    with open(args.out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_results(rows, args.out_dir)
    print(f"Wrote {csv_path}")
    print(f"Wrote {args.out_dir / 'neighborhood_sim.png'}")
    for r in final:
        print(
            f"  {r['method']}: ARI={r['ari_vs_exact']:.4f}  "
            f"Jaccard={r['mean_nbr_jaccard']:.4f}  "
            f"nbr_recall={r['mean_nbr_recall']:.4f}  "
            f"k_inc={r['n_clusters_inc']} k_exact={r['n_clusters_exact']}"
        )


if __name__ == "__main__":
    main()
