#!/usr/bin/env python3
"""Compare grow-only IncDBSCAN partitions against batch DBSCAN.

Two regimes
-----------
1. **Archive** (RQ1 calibration matrices): insert in fitness order with the
   same offline ``seed_from_precomputed_matrix`` used for the threshold sweep,
   then compare against sklearn DBSCAN on the same precomputed matrix.
2. **Online** (live IncDBSCAN runs): rebuild the ε-neighborhood graph from
   ``inc_dbscan_state.json`` and run batch DBSCAN on that *exact* graph.
   This bounds how far the final live labels deviate from batch DBSCAN; it
   cannot replay the evolutionary trajectory.

At MinPts=2 the two must match (connected-component niching). At MinPts≥3
the live insert path under-merges relative to batch.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/incdbscan_exactness.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from speciation.clustering.inc_dbscan import seed_from_precomputed_matrix  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
OUT = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
MAT_E = ROOT / "experiments" / "evostar_2027" / "rq1" / "mats" / "d_E_3139.npy"
MAT_NLI = ROOT / "experiments" / "evostar_2027" / "data" / "mats" / "B3139" / "d_negE.npy"
POOL = ROOT / "experiments" / "evostar_2027" / "data" / "pool" / "p0_full.pkl"

THRESHOLDS = (0.05, 0.10, 0.15)
MIN_SAMPLES = (2, 3, 5)
ONLINE_CELLS = (
    ("embedding", "inc_embedding"),
    ("nli_entail", "inc_nli"),
)


def _load_fitness_order(n: int) -> np.ndarray:
    """Fitness-descending insertion order used by RQ1 (fallback: identity)."""
    if POOL.is_file():
        import pickle

        with POOL.open("rb") as f:
            pool = pickle.load(f)
        fits: List[Tuple[float, int]] = []
        if isinstance(pool, dict) and "rows" in pool:
            rows = pool["rows"]
        elif isinstance(pool, list):
            rows = pool
        else:
            rows = []
        for i, rec in enumerate(rows[:n]):
            if isinstance(rec, dict):
                fit = rec.get("fitness")
                if fit is None:
                    fit = rec.get("illicit")
                try:
                    fits.append((float(fit), i))
                except (TypeError, ValueError):
                    fits.append((0.0, i))
            else:
                fits.append((0.0, i))
        if len(fits) == n:
            fits.sort(key=lambda t: (-t[0], t[1]))
            return np.asarray([i for _, i in fits], dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _partition_stats(labels: np.ndarray) -> Dict[str, float]:
    labs = np.asarray(labels)
    n = int(labs.size)
    assigned = labs >= 0
    n_unass = int((~assigned).sum())
    counts = Counter(int(x) for x in labs[assigned].tolist())
    n_mat = sum(1 for c in counts.values() if c >= 2)
    # Drop size-1 "clusters" from assigned dominance denominator if any remain
    mature = {k: v for k, v in counts.items() if v >= 2}
    n_assigned = int(sum(mature.values()))
    largest = max(mature.values()) if mature else 0
    d_assigned = (largest / n_assigned) if n_assigned > 0 else float("nan")
    return {
        "n": n,
        "n_mat": float(n_mat),
        "n_unassigned": float(n_unass),
        "unassigned_frac": n_unass / n if n else float("nan"),
        "n_assigned": float(n_assigned),
        "largest": float(largest),
        "d_assigned": float(d_assigned),
    }


def _batch_dbscan(D: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    from sklearn.cluster import DBSCAN

    model = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    raw = model.fit_predict(D)
    # sklearn: -1 = noise; remapped mature labels stay >=0
    return np.asarray(raw, dtype=np.int32)


def _ari(a: np.ndarray, b: np.ndarray) -> float:
    from sklearn.metrics import adjusted_rand_score

    return float(adjusted_rand_score(a, b))


def archive_comparison(
    mats: Dict[str, Path],
    thresholds: Sequence[float],
    min_samples_list: Sequence[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for name, path in mats.items():
        if not path.is_file():
            print(f"[warn] missing matrix {path}")
            continue
        D = np.load(path)
        n = int(D.shape[0])
        order = _load_fitness_order(n)
        print(f"Archive {name}: N={n}")
        for eps in thresholds:
            for ms in min_samples_list:
                inc = seed_from_precomputed_matrix(D, order, eps=eps, min_samples=ms)
                batch = _batch_dbscan(D, eps=eps, min_samples=ms)
                s_inc = _partition_stats(inc)
                s_batch = _partition_stats(batch)
                # Align noise: both use -1 for unassigned
                ari = _ari(inc, batch)
                rows.append(
                    {
                        "regime": "archive",
                        "distance": name,
                        "eps": eps,
                        "min_samples": ms,
                        "n": n,
                        "inc_n_mat": s_inc["n_mat"],
                        "inc_d_assigned": s_inc["d_assigned"],
                        "inc_unassigned_frac": s_inc["unassigned_frac"],
                        "batch_n_mat": s_batch["n_mat"],
                        "batch_d_assigned": s_batch["d_assigned"],
                        "batch_unassigned_frac": s_batch["unassigned_frac"],
                        "ari": ari,
                        "exact_match": int(np.array_equal(inc, batch)),
                    }
                )
                print(
                    f"  {name} t={eps} MinPts={ms}: "
                    f"inc n_mat={s_inc['n_mat']:.0f} vs batch {s_batch['n_mat']:.0f}; "
                    f"ARI={ari:.4f}"
                )
    return rows


def _labels_from_state(state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    labels = np.asarray(state["labels"], dtype=np.int32)
    is_core = np.asarray(state["is_core"], dtype=bool)
    neighbors = [list(nbrs) for nbrs in state["neighbors"]]
    return labels, is_core, neighbors


def _batch_from_graph(
    neighbors: Sequence[Sequence[int]],
    min_samples: int,
) -> np.ndarray:
    """Run classical DBSCAN on an explicit undirected neighborhood graph.

    Neighbor lists already include self. Core = degree >= min_samples.
    Clusters = core-connected components plus adjacent border points.
    Border ties go to the lowest core-component ID (matches live code).
    """
    n = len(neighbors)
    nbr_sets = [set(int(j) for j in nbrs) for nbrs in neighbors]
    is_core = [len(nbr_sets[i]) >= min_samples for i in range(n)]
    # BFS over core-core edges
    core_comp = [-1] * n
    next_id = 0
    for i in range(n):
        if not is_core[i] or core_comp[i] >= 0:
            continue
        stack = [i]
        core_comp[i] = next_id
        while stack:
            u = stack.pop()
            for v in nbr_sets[u]:
                if is_core[v] and core_comp[v] < 0:
                    core_comp[v] = next_id
                    stack.append(v)
        next_id += 1
    # Assign borders to lowest adjacent core component
    out = np.full(n, -1, dtype=np.int32)
    for i in range(n):
        if is_core[i]:
            out[i] = core_comp[i]
            continue
        adj = sorted(
            {core_comp[j] for j in nbr_sets[i] if is_core[j] and core_comp[j] >= 0}
        )
        if adj:
            out[i] = adj[0]
    # Demote size-1 components to noise (matches grow-only IncDBSCAN)
    counts: Dict[int, int] = defaultdict(int)
    for lab in out.tolist():
        if lab >= 0:
            counts[int(lab)] += 1
    size1 = {sid for sid, c in counts.items() if c == 1}
    if size1:
        for i, lab in enumerate(out.tolist()):
            if lab in size1:
                out[i] = -1
    return out


def _canonicalize(labels: np.ndarray) -> np.ndarray:
    """Remap mature IDs to 0..K-1 in order of appearance; keep -1/0 as unassigned."""
    out = np.full(labels.shape, -1, dtype=np.int32)
    remap: Dict[int, int] = {}
    next_id = 0
    for i, lab in enumerate(labels.tolist()):
        lab = int(lab)
        if lab <= 0:
            out[i] = -1
            continue
        if lab not in remap:
            remap[lab] = next_id
            next_id += 1
        out[i] = remap[lab]
    return out


def online_comparison(live_root: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for distance, folder in ONLINE_CELLS:
        for seed in seeds:
            run_dir = live_root / folder / f"rep{seed}_s{seed}"
            state_path = run_dir / "inc_dbscan_state.json"
            cfg_path = run_dir / "run_config.json"
            if not state_path.is_file():
                print(f"[warn] missing {state_path}")
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            spec = cfg.get("speciation") or {}
            ms = int(spec.get("dbscan_min_samples") or state.get("min_samples") or 2)
            eps = float(spec.get("theta_sim") or state.get("eps") or 0.1)
            live_labels, _is_core, neighbors = _labels_from_state(state)
            batch = _batch_from_graph(neighbors, min_samples=ms)
            live_c = _canonicalize(live_labels)
            batch_c = _canonicalize(batch)
            s_live = _partition_stats(live_c)
            s_batch = _partition_stats(batch_c)
            ari = _ari(live_c, batch_c)
            n_edges = sum(len(nbrs) for nbrs in neighbors)
            # Undirected edge count excluding self-loops
            n_undirected = (n_edges - len(neighbors)) // 2
            rows.append(
                {
                    "regime": "online",
                    "distance": distance,
                    "folder": folder,
                    "seed": seed,
                    "eps": eps,
                    "min_samples": ms,
                    "n": int(live_labels.size),
                    "n_edges": n_undirected,
                    "inc_n_mat": s_live["n_mat"],
                    "inc_d_assigned": s_live["d_assigned"],
                    "inc_unassigned_frac": s_live["unassigned_frac"],
                    "batch_n_mat": s_batch["n_mat"],
                    "batch_d_assigned": s_batch["d_assigned"],
                    "batch_unassigned_frac": s_batch["unassigned_frac"],
                    "ari": ari,
                    "exact_match": int(np.array_equal(live_c, batch_c)),
                }
            )
            print(
                f"Online {folder} seed={seed} MinPts={ms}: "
                f"live n_mat={s_live['n_mat']:.0f} vs batch {s_batch['n_mat']:.0f}; "
                f"ARI={ari:.4f}"
            )
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_summary(path: Path, archive: List[Dict[str, Any]], online: List[Dict[str, Any]]) -> None:
    lines = [
        "# IncDBSCAN exactness audit",
        "",
        "Compares grow-only IncDBSCAN partitions against classical batch DBSCAN.",
        "At MinPts=2 the methods must match (connected-component niching).",
        "At MinPts≥3 the live insert path under-merges relative to batch.",
        "",
        "## Archive (calibration matrices)",
        "",
        "| distance | t | MinPts | inc n_mat | batch n_mat | ARI | exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in archive:
        lines.append(
            f"| {r['distance']} | {r['eps']} | {r['min_samples']} | "
            f"{r['inc_n_mat']:.0f} | {r['batch_n_mat']:.0f} | "
            f"{r['ari']:.4f} | {r['exact_match']} |"
        )
    lines += ["", "## Online (final live graphs)", "", 
              "| cell | seed | MinPts | live n_mat | batch n_mat | ARI | exact |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for r in online:
        lines.append(
            f"| {r['folder']} | {r['seed']} | {r['min_samples']} | "
            f"{r['inc_n_mat']:.0f} | {r['batch_n_mat']:.0f} | "
            f"{r['ari']:.4f} | {r['exact_match']} |"
        )
    # Aggregate online ARI
    by_cell: Dict[str, List[float]] = defaultdict(list)
    for r in online:
        by_cell[str(r["folder"])].append(float(r["ari"]))
    lines += ["", "## Online ARI summary (median [min, max])", ""]
    for folder, vals in sorted(by_cell.items()):
        med = float(np.median(vals))
        lines.append(f"- `{folder}`: {med:.4f} [{min(vals):.4f}, {max(vals):.4f}] (n={len(vals)})")
    lines += [
        "",
        "## Caveat",
        "",
        "The online comparison applies batch DBSCAN to the *final* ε-graph and",
        "therefore bounds final-partition disagreement. It does not replay the",
        "evolutionary trajectory under exact IncrementalDBSCAN updates.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="IncDBSCAN vs batch DBSCAN exactness audit")
    p.add_argument("--live-dir", type=Path, default=LIVE)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--skip-archive", action="store_true")
    p.add_argument("--skip-online", action="store_true")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    mats = {"d_E": MAT_E, "d_negE": MAT_NLI}
    archive: List[Dict[str, Any]] = []
    online: List[Dict[str, Any]] = []
    if not args.skip_archive:
        archive = archive_comparison(mats, THRESHOLDS, MIN_SAMPLES)
    if not args.skip_online:
        online = online_comparison(args.live_dir, seeds)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "incdbscan_exactness_archive.csv", archive)
    write_csv(args.out_dir / "incdbscan_exactness_online.csv", online)
    write_summary(args.out_dir / "incdbscan_exactness_summary.md", archive, online)
    meta = {
        "thresholds": list(THRESHOLDS),
        "min_samples": list(MIN_SAMPLES),
        "mats": {k: str(v) for k, v in mats.items()},
        "n_archive_rows": len(archive),
        "n_online_rows": len(online),
        "seeds": seeds,
    }
    (args.out_dir / "incdbscan_exactness_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote reports under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
