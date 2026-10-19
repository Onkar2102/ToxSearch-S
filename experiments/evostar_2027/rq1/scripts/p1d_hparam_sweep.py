#!/usr/bin/env python3
"""Part 1d: lock θ_sim / ε / min_samples; structure probes; propose shortlist.

Uses whatever mats exist under rq1/mats (cheap always; NLI if present).
If only smoke-sized d_N / d_negE exist, calibrates those at their N and notes it.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_hparam_sweep.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from experiments.stage_analysis import stage_a

DEFAULT_MATS = Path("experiments/evostar_2027/rq1/mats")
DEFAULT_SWEEPS = Path("experiments/evostar_2027/rq1/sweeps")
REFERENCE_RADIUS = 0.25
MIN_SAMPLES_GRID = (2, 3, 5)


# Plan name → CLI distance_method
PLAN_TO_CLI = {
    "d_E": "embedding",
    "d_P": "objective",
    "d_EP": "embedding+objective",
    "d_N": "nli",
    "d_negE": "nli_entail",
}


def _load_mat(mats_dir: Path, stem: str) -> Optional[Tuple[np.ndarray, int, Path]]:
    """Load largest available ``{stem}_{N}.npy`` (prefer N=2000)."""
    matches = sorted(mats_dir.glob(f"{stem}_*.npy"), key=lambda p: p.stat().st_mtime, reverse=True)
    # Prefer exact analysis size if present
    by_n: Dict[int, Path] = {}
    for p in mats_dir.glob(f"{stem}_*.npy"):
        try:
            n = int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            continue
        by_n[n] = p
    if not by_n:
        return None
    n = max(by_n)
    # Prefer 2000 when available
    if 2000 in by_n:
        n = 2000
    path = by_n[n]
    m = np.load(path)
    return m, n, path


def _dbscan_structure(m: np.ndarray, eps: float, min_samples: int) -> Dict[str, float]:
    labels = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed").fit_predict(m)
    n = len(labels)
    noise = float(np.mean(labels < 0)) if n else 0.0
    clusters = int(len(set(labels) - {-1}))
    sizes = [int(np.sum(labels == c)) for c in set(labels) if c >= 0]
    return {
        "n_clusters": clusters,
        "noise_frac": noise,
        "max_cluster": float(max(sizes) if sizes else 0),
        "median_cluster": float(np.median(sizes) if sizes else 0),
        "singleton_frac": float(sum(1 for s in sizes if s == 1) / max(clusters, 1)),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1d: hparam sweep + shortlist")
    p.add_argument("--mats-dir", type=Path, default=DEFAULT_MATS)
    p.add_argument("--sweeps-dir", type=Path, default=DEFAULT_SWEEPS)
    p.add_argument("--reference-radius", type=float, default=REFERENCE_RADIUS)
    p.add_argument("--min-samples", type=int, nargs="+", default=list(MIN_SAMPLES_GRID))
    args = p.parse_args()

    mats_dir = args.mats_dir
    if not mats_dir.is_dir():
        raise SystemExit(f"Missing mats dir {mats_dir}; run p1b first")

    loaded: Dict[str, Tuple[np.ndarray, int, Path]] = {}
    for stem in ("d_E", "d_P", "d_EP", "d_N", "d_negE"):
        hit = _load_mat(mats_dir, stem)
        if hit is not None:
            loaded[stem] = hit
            print(f"  loaded {stem} N={hit[1]} from {hit[2]}")

    if "d_E" not in loaded:
        raise SystemExit("Need at least d_E mat from p1b")

    # Align to common N for multi-mat tables (use min N among loaded for paired stats)
    notes: List[str] = []
    n_sizes = {k: v[1] for k, v in loaded.items()}
    if len(set(n_sizes.values())) > 1:
        notes.append(
            f"Mat sizes differ {n_sizes}; calibration uses each mat at its own N. "
            "Shortlist prefers distances with N matching d_E when available."
        )

    calib_rows = []
    structure_rows = []
    ref_m, ref_n, _ = loaded["d_E"]
    target_frac = stage_a.neighbor_fraction(ref_m, args.reference_radius, strict=True)

    for name, (m, n, path) in loaded.items():
        theta = stage_a.calibrate_radius(m, target_frac, strict=True)
        eps = stage_a.calibrate_radius(m, target_frac, strict=False)
        calib_rows.append({
            "distance": name,
            "cli_method": PLAN_TO_CLI[name],
            "N": n,
            "path": str(path),
            "target_fraction": target_frac,
            "reference": "d_E",
            "reference_radius": args.reference_radius,
            "theta_sim": theta,
            "eps": eps,
            "dbscan_eps": eps,
            "smoke_sized": n < ref_n,
        })
        for ms in args.min_samples:
            st = _dbscan_structure(m, eps, ms)
            structure_rows.append({
                "distance": name,
                "N": n,
                "eps": eps,
                "min_samples": ms,
                **st,
            })

    sweeps_dir = args.sweeps_dir
    sweeps_dir.mkdir(parents=True, exist_ok=True)
    calib_df = pd.DataFrame(calib_rows)
    struct_df = pd.DataFrame(structure_rows)
    calib_path = sweeps_dir / "calibration.csv"
    struct_path = sweeps_dir / "structure.csv"
    calib_df.to_csv(calib_path, index=False)
    struct_df.to_csv(struct_path, index=False)

    # Shortlist: always include d_E + d_EP; add d_N / d_negE if present (even smoke);
    # drop raw d_P from pilot shortlist (component of d_EP) unless only cheap mats.
    prefer = ["d_E", "d_EP", "d_N", "d_negE"]
    shortlist = [d for d in prefer if d in loaded]
    if not shortlist:
        shortlist = list(loaded.keys())

    # Default min_samples: prefer 2 (code default); note if structure collapses
    locked_min_samples = 2
    for d in shortlist:
        row = struct_df[(struct_df["distance"] == d) & (struct_df["min_samples"] == 2)]
        if len(row) and float(row.iloc[0]["noise_frac"]) > 0.95:
            notes.append(f"{d}: high noise at min_samples=2; consider 2 still (default) for pilots")

    proposal = {
        "shortlist_plan_names": shortlist,
        "shortlist_cli_methods": [PLAN_TO_CLI[d] for d in shortlist],
        "clustering_to_pilot": ["leader_follower", "dbscan"],
        "locked_min_samples": locked_min_samples,
        "reference_radius_d_E": args.reference_radius,
        "target_neighbor_fraction": target_frac,
        "per_distance": {
            r["distance"]: {
                "cli_method": r["cli_method"],
                "theta_sim": r["theta_sim"],
                "eps": r["eps"],
                "dbscan_eps": r["dbscan_eps"],
                "min_samples": locked_min_samples,
                "N_used": r["N"],
                "smoke_sized": r["smoke_sized"],
            }
            for r in calib_rows
            if r["distance"] in shortlist
        },
        "notes": notes,
        "files": {"calibration": str(calib_path), "structure": str(struct_path)},
    }
    prop_path = sweeps_dir / "shortlist_proposal.json"
    prop_path.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    print(f"Wrote {calib_path}")
    print(f"Wrote {struct_path}")
    print(f"Wrote {prop_path}")
    print(f"Shortlist CLI: {proposal['shortlist_cli_methods']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
