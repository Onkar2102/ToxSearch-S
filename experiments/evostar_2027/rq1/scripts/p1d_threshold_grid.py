#!/usr/bin/env python3
"""Part 1d⁺: step-wise threshold grid for BOTH speciation methods.

Independent grid per distance (default start 0.1, step 0.05):

- **leader_follower**: offline LF on the distance matrix (fitness-desc order;
  join nearest leader if ``d < theta``; else new leader / singleton).
- **incdbscan**: true IncDBSCAN incremental seed on the precomputed mat
  (fitness-desc sequential inserts; same insert logic as live ``clustering_method=dbscan``).

Also records LF neighbor fraction and prompt samples from largest niches.

Per-(method, distance, threshold) species artifacts under ``{sweeps-dir}/species/``::

    {method}__{distance}__t{t:.2f}[_ms{ms}]_labels.npy
    {method}__{distance}__t{t:.2f}[_ms{ms}]_summary.json
    {method}__{distance}__t{t:.2f}[_ms{ms}]_prompts.json

Example (B500)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_threshold_grid.py \\
      --mats-dir experiments/evostar_2027/data/mats/B500 \\
      --p0 experiments/evostar_2027/data/batches/B500/p0.pkl \\
      --sweeps-dir experiments/evostar_2027/rq1/sweeps/B500 \\
      --t-min 0.1 --t-max 0.5 --t-step 0.05
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from experiments.stage_analysis import stage_a
from speciation.clustering.inc_dbscan import seed_from_precomputed_matrix

PLAN_TO_CLI = {
    "d_E": "embedding",
    "d_P": "objective",
    "d_EP": "embedding+objective",
    "d_N": "nli",
    "d_negE": "nli_entail",
}
DEFAULT_DISTANCES = ("d_E", "d_EP", "d_N", "d_negE")


def _load_plain_mats(mats_dir: Path, names: Sequence[str]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name in names:
        path = mats_dir / f"{name}.npy"
        if not path.is_file():
            print(f"  skip missing {path}")
            continue
        m = np.load(path)
        if m.ndim != 2 or m.shape[0] != m.shape[1]:
            raise SystemExit(f"{path} is not square: {m.shape}")
        out[name] = np.asarray(m, dtype=np.float64)
        print(f"  loaded {name} N={m.shape[0]}")
    if not out:
        raise SystemExit(f"No mats found in {mats_dir}")
    return out


def _threshold_grid(t_min: float, t_max: float, t_step: float) -> List[float]:
    if t_step <= 0:
        raise SystemExit("--t-step must be > 0")
    vals: List[float] = []
    t = t_min
    while t <= t_max + 1e-9:
        vals.append(round(float(t), 10))
        t += t_step
    return vals


def _summarize_labels(labels: np.ndarray, *, noise_label: int = -1) -> Dict[str, float]:
    n = len(labels)
    noise = float(np.mean(labels == noise_label)) if n else 0.0
    cluster_ids = [int(c) for c in set(labels.tolist()) if c != noise_label]
    sizes = [int(np.sum(labels == c)) for c in cluster_ids]
    mature = [s for s in sizes if s >= 2]
    singletons = [s for s in sizes if s == 1]
    return {
        "n_clusters": len(cluster_ids),
        "n_mature": len(mature),
        "n_singletons": len(singletons),
        "noise_frac": noise,
        "max_cluster": float(max(sizes) if sizes else 0),
        "median_cluster": float(np.median(sizes) if sizes else 0),
        "median_mature": float(np.median(mature) if mature else 0),
        "singleton_frac": float(len(singletons) / max(len(cluster_ids), 1)),
        "frac_in_mature": float(sum(mature) / n) if n else 0.0,
    }


def lf_offline_labels(
    D: np.ndarray,
    theta: float,
    order: np.ndarray,
) -> np.ndarray:
    """Offline Leader–Follower on a static distance matrix.

    Process indices in ``order`` (fitness-descending). Each point joins the
    nearest existing leader with ``d < theta`` (strict, matching ToxSearch LF);
    otherwise it becomes a new leader (singleton until someone joins).
    """
    n = D.shape[0]
    labels = np.full(n, -1, dtype=np.int32)
    leaders: List[int] = []  # leader index per species id
    next_sid = 0
    for i in order:
        i = int(i)
        best_sid = -1
        best_d = float("inf")
        for sid, lid in enumerate(leaders):
            d = float(D[i, lid])
            if d < theta and d < best_d:
                best_d = d
                best_sid = sid
        if best_sid >= 0:
            labels[i] = best_sid
        else:
            labels[i] = next_sid
            leaders.append(i)
            next_sid += 1
    return labels


def incdbscan_labels(
    D: np.ndarray,
    eps: float,
    min_samples: int,
    order: np.ndarray,
) -> np.ndarray:
    """True IncDBSCAN seed on precomputed distances (analysis labels: non-mature=-1)."""
    return seed_from_precomputed_matrix(D, order, eps=eps, min_samples=min_samples)


def _cluster_prompt_samples(
    labels: np.ndarray,
    prompts: Sequence[str],
    fitness: Optional[Sequence[float]],
    *,
    top_k_clusters: int,
    prompts_per_cluster: int,
    noise_label: int = -1,
) -> List[Dict[str, Any]]:
    cluster_ids = sorted({int(c) for c in labels if c != noise_label})
    if not cluster_ids:
        return []
    sizes = {c: int(np.sum(labels == c)) for c in cluster_ids}
    ordered = sorted(cluster_ids, key=lambda c: sizes[c], reverse=True)[:top_k_clusters]
    samples: List[Dict[str, Any]] = []
    for c in ordered:
        idxs = np.where(labels == c)[0]
        if fitness is not None:
            idxs = idxs[np.argsort([-float(fitness[i]) for i in idxs])]
        take = idxs[:prompts_per_cluster]
        samples.append({
            "cluster_id": int(c),
            "size": sizes[c],
            "prompts": [
                {
                    "i": int(i),
                    "fitness": None if fitness is None else float(fitness[i]),
                    "prompt": str(prompts[i])[:500],
                }
                for i in take
            ],
        })
    return samples


def _species_stem(method: str, distance: str, t: float, ms: Optional[int] = None) -> str:
    stem = f"{method}__{distance}__t{t:.2f}"
    if ms is not None:
        stem = f"{stem}_ms{ms}"
    return stem


def _write_species_artifacts(
    species_dir: Path,
    *,
    method: str,
    distance: str,
    t: float,
    labels: np.ndarray,
    summary: Dict[str, Any],
    clusters: Optional[List[Dict[str, Any]]],
    ms: Optional[int] = None,
) -> Dict[str, str]:
    """Write labels.npy, summary.json, prompts.json; return relative paths."""
    species_dir.mkdir(parents=True, exist_ok=True)
    stem = _species_stem(method, distance, t, ms)
    labels_path = species_dir / f"{stem}_labels.npy"
    summary_path = species_dir / f"{stem}_summary.json"
    prompts_path = species_dir / f"{stem}_prompts.json"

    np.save(labels_path, np.asarray(labels, dtype=np.int32))
    summary_payload = {
        "method": method,
        "distance": distance,
        "threshold": t,
        "min_samples": ms,
        **summary,
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    prompts_payload = {
        "method": method,
        "distance": distance,
        "threshold": t,
        "min_samples": ms,
        "clusters": clusters or [],
    }
    prompts_path.write_text(json.dumps(prompts_payload, indent=2), encoding="utf-8")

    return {
        "stem": stem,
        "labels": str(labels_path),
        "summary": str(summary_path),
        "prompts": str(prompts_path),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 1d⁺: LF + IncDBSCAN threshold grid")
    p.add_argument("--mats-dir", type=Path, required=True)
    p.add_argument("--p0", type=Path, default=None)
    p.add_argument("--sweeps-dir", type=Path, required=True)
    p.add_argument("--distances", type=str, default=",".join(DEFAULT_DISTANCES))
    p.add_argument("--t-min", type=float, default=0.1)
    p.add_argument("--t-max", type=float, default=0.5)
    p.add_argument("--t-step", type=float, default=0.05)
    p.add_argument("--min-samples", type=int, nargs="+", default=[2],
                   help="IncDBSCAN/DBSCAN min_samples values")
    p.add_argument("--top-k-clusters", type=int, default=5)
    p.add_argument("--prompts-per-cluster", type=int, default=5)
    args = p.parse_args()

    names = tuple(x.strip() for x in args.distances.split(",") if x.strip())
    mats = _load_plain_mats(args.mats_dir, names)
    grid = _threshold_grid(args.t_min, args.t_max, args.t_step)
    print(f"threshold grid ({len(grid)}): {grid}")
    print("methods: leader_follower, incdbscan")
    t_grid0 = time.time()

    prompts: Optional[List[str]] = None
    fitness_list: Optional[List[float]] = None
    order = None
    if args.p0 is not None and args.p0.is_file():
        df = pd.read_pickle(args.p0)
        n0 = len(df)
        for name, m in mats.items():
            if m.shape[0] != n0:
                raise SystemExit(f"{name} N={m.shape[0]} != p0 rows={n0}")
        prompts = df["prompt"].astype(str).tolist()
        fitness_list = pd.to_numeric(df["fitness"], errors="coerce").fillna(0.0).tolist()
        order = np.argsort(-np.asarray(fitness_list, dtype=np.float64))
        print(f"LF order = fitness-desc from {args.p0} (N={n0})")
    else:
        # fallback: matrix index order
        n = next(iter(mats.values())).shape[0]
        order = np.arange(n)
        print("WARN: no p0 — LF uses index order, not fitness")

    rows: List[Dict[str, Any]] = []
    samples_out: Dict[str, Any] = {
        "t_min": args.t_min,
        "t_max": args.t_max,
        "t_step": args.t_step,
        "grid": grid,
        "methods": ["leader_follower", "incdbscan"],
        "by_distance": {},
    }
    sweeps_dir = args.sweeps_dir
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    species_dir = sweeps_dir / "species"
    species_index: List[Dict[str, Any]] = []

    for name, m in mats.items():
        print(f"=== {name} ===")
        dist_samples: Dict[str, Any] = {"leader_follower": {}, "incdbscan": {}}
        for t in grid:
            neigh = stage_a.neighbor_fraction(m, t, strict=True)

            # --- Leader–Follower ---
            lf_lab = lf_offline_labels(m, t, order)
            lf_sum = _summarize_labels(lf_lab, noise_label=-1)
            lf_sum["noise_frac"] = 0.0  # LF assigns every point (singletons = size-1 niches)

            row_lf = {
                "distance": name,
                "cli_method": PLAN_TO_CLI.get(name, name),
                "method": "leader_follower",
                "N": int(m.shape[0]),
                "threshold": t,
                "theta_sim": t,
                "eps": None,
                "min_samples": None,
                "neighbor_frac_lf": neigh,
                **lf_sum,
            }
            rows.append(row_lf)
            key = f"t={t:.2f}"
            lf_clusters = None
            if prompts is not None:
                lf_clusters = _cluster_prompt_samples(
                    lf_lab, prompts, fitness_list,
                    top_k_clusters=args.top_k_clusters,
                    prompts_per_cluster=args.prompts_per_cluster,
                    noise_label=-1,
                )
                dist_samples["leader_follower"][key] = {
                    "threshold": t,
                    "neighbor_frac_lf": neigh,
                    **{k: lf_sum[k] for k in lf_sum},
                    "clusters": lf_clusters,
                }
            art = _write_species_artifacts(
                species_dir,
                method="leader_follower",
                distance=name,
                t=t,
                labels=lf_lab,
                summary={**lf_sum, "neighbor_frac_lf": neigh, "N": int(m.shape[0])},
                clusters=lf_clusters,
            )
            species_index.append({
                "method": "leader_follower",
                "distance": name,
                "threshold": t,
                "min_samples": None,
                **art,
            })
            print(
                f"  LF  t={t:.2f}: mature={lf_sum['n_mature']} singletons={lf_sum['n_singletons']} "
                f"max={lf_sum['max_cluster']:.0f} frac_mature={lf_sum['frac_in_mature']:.3f} "
                f"neigh={neigh:.3f}"
            )

            # --- IncDBSCAN (true incremental seed on precomputed mat) ---
            for ms in args.min_samples:
                db_lab = incdbscan_labels(m, t, ms, order)
                db_sum = _summarize_labels(db_lab, noise_label=-1)
                row_db = {
                    "distance": name,
                    "cli_method": PLAN_TO_CLI.get(name, name),
                    "method": "incdbscan",
                    "N": int(m.shape[0]),
                    "threshold": t,
                    "theta_sim": None,
                    "eps": t,
                    "min_samples": ms,
                    "neighbor_frac_lf": neigh,
                    **db_sum,
                }
                rows.append(row_db)
                key_ms = f"t={t:.2f}_ms={ms}"
                db_clusters = None
                if prompts is not None:
                    db_clusters = _cluster_prompt_samples(
                        db_lab, prompts, fitness_list,
                        top_k_clusters=args.top_k_clusters,
                        prompts_per_cluster=args.prompts_per_cluster,
                        noise_label=-1,
                    )
                    dist_samples["incdbscan"][key_ms] = {
                        "threshold": t,
                        "min_samples": ms,
                        "neighbor_frac_lf": neigh,
                        **{k: db_sum[k] for k in db_sum},
                        "clusters": db_clusters,
                    }
                art = _write_species_artifacts(
                    species_dir,
                    method="incdbscan",
                    distance=name,
                    t=t,
                    labels=db_lab,
                    summary={**db_sum, "neighbor_frac_lf": neigh, "N": int(m.shape[0])},
                    clusters=db_clusters,
                    ms=ms,
                )
                species_index.append({
                    "method": "incdbscan",
                    "distance": name,
                    "threshold": t,
                    "min_samples": ms,
                    **art,
                })
                print(
                    f"  DB  t={t:.2f} ms={ms}: clusters={db_sum['n_clusters']} "
                    f"mature={db_sum['n_mature']} noise={db_sum['noise_frac']:.3f} "
                    f"max={db_sum['max_cluster']:.0f}"
                )

        samples_out["by_distance"][name] = dist_samples

    grid_path = sweeps_dir / "threshold_grid.csv"
    pd.DataFrame(rows).to_csv(grid_path, index=False)
    print(f"Wrote {grid_path} ({len(rows)} rows)")

    samples_path = sweeps_dir / "threshold_grid_prompt_samples.json"
    samples_path.write_text(json.dumps(samples_out, indent=2), encoding="utf-8")
    print(f"Wrote {samples_path}")

    index_path = species_dir / "index.json"
    index_path.write_text(
        json.dumps({"n_entries": len(species_index), "entries": species_index}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {index_path} ({len(species_index)} species configs)")

    wall_s = time.time() - t_grid0
    meta = {
        "mats_dir": str(args.mats_dir),
        "p0": str(args.p0) if args.p0 else None,
        "distances": list(mats.keys()),
        "methods": ["leader_follower", "incdbscan"],
        "t_min": args.t_min,
        "t_max": args.t_max,
        "t_step": args.t_step,
        "min_samples": list(args.min_samples),
        "lf_semantics": "offline fitness-desc; join nearest leader if d < theta; else new leader",
        "incdbscan_semantics": (
            "true IncDBSCAN seed_from_precomputed_matrix; fitness-desc inserts; "
            "eps=t; min_samples as given; same insert logic as live clustering_method=dbscan"
        ),
        "timing": {
            "wall_seconds": round(wall_s, 3),
            "wall_hours": round(wall_s / 3600.0, 4),
            "n_grid_rows": len(rows),
            "finished_at": datetime.now(timezone.utc).isoformat(),
        },
        "files": {
            "threshold_grid": str(grid_path),
            "prompt_samples": str(samples_path),
            "species_dir": str(species_dir),
            "species_index": str(index_path),
        },
    }
    meta_path = sweeps_dir / "threshold_grid_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path} (grid wall={wall_s:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
