#!/usr/bin/env python3
"""Part 1d⁺: embedding-canvas cohesion for offline species labels.

For each ``species/*_labels.npy`` entry, score topic-clustering quality on the
shared embedding canvas ``d_E`` (no gold topics):

- mean **intra-cluster** pairwise ``d_E`` (mature clusters, size ≥ 2)
- mean **inter-cluster** ``d_E`` (random cross-cluster pairs)
- ``separation = inter / intra`` (higher ⇒ better topic separation)
- structural health flags (collapse / over-fragmentation)

Example (B500)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_cluster_coherence.py \\
      --species-dir experiments/evostar_2027/rq1/sweeps/B500/species \\
      --d-e experiments/evostar_2027/data/mats/B500/d_E.npy \\
      --out experiments/evostar_2027/rq1/sweeps/B500/coherence.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _mean_pairwise(D: np.ndarray, idxs: np.ndarray) -> Optional[float]:
    if len(idxs) < 2:
        return None
    sub = D[np.ix_(idxs, idxs)]
    iu = np.triu_indices(len(idxs), k=1)
    vals = sub[iu]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def _intra_inter(
    D: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
    max_inter_pairs: int = 5000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Optional[float], Optional[float], int, int]:
    """Return (intra_mean, inter_mean, n_mature, n_pairs_inter)."""
    rng = rng or np.random.default_rng(0)
    cluster_ids = sorted({int(c) for c in labels if c != noise_label})
    mature = [c for c in cluster_ids if int(np.sum(labels == c)) >= 2]

    # Unweighted mean of per-cluster means (stable ranking across uneven sizes)
    per_cluster_intra: List[float] = []
    for c in mature:
        idxs = np.where(labels == c)[0]
        m = _mean_pairwise(D, idxs)
        if m is not None:
            per_cluster_intra.append(m)
    intra = float(np.mean(per_cluster_intra)) if per_cluster_intra else None

    # inter: sample pairs from different mature clusters (or all non-noise)
    labeled = np.where(labels != noise_label)[0]
    if len(labeled) < 2 or len(mature) < 2:
        return intra, None, len(mature), 0

    inter_vals: List[float] = []
    attempts = 0
    max_attempts = max_inter_pairs * 20
    while len(inter_vals) < max_inter_pairs and attempts < max_attempts:
        attempts += 1
        i, j = rng.choice(labeled, size=2, replace=False)
        if labels[i] == labels[j]:
            continue
        if labels[i] == noise_label or labels[j] == noise_label:
            continue
        inter_vals.append(float(D[int(i), int(j)]))
    inter = float(np.mean(inter_vals)) if inter_vals else None
    return intra, inter, len(mature), len(inter_vals)


def _health_flags(
    summary: Dict[str, Any],
    N: int,
    *,
    collapse_max_frac: float = 0.7,
    min_mature: int = 3,
    min_frac_mature: float = 0.3,
) -> Tuple[bool, str]:
    max_c = float(summary.get("max_cluster", 0) or 0)
    n_mature = int(summary.get("n_mature", 0) or 0)
    frac_mature = float(summary.get("frac_in_mature", 0) or 0)
    noise = float(summary.get("noise_frac", 0) or 0)

    if N > 0 and max_c / N >= collapse_max_frac:
        return False, "collapse"
    if n_mature < min_mature:
        return False, "too_few_mature"
    if frac_mature < min_frac_mature and noise < 0.5:
        # LF with many singletons / DBSCAN with little coverage
        return False, "over_fragmented"
    return True, "ok"


def score_species_dir(
    species_dir: Path,
    d_e: np.ndarray,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    index_path = species_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries", [])
    N = int(d_e.shape[0])
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []

    for ent in entries:
        labels_path = Path(ent["labels"])
        summary_path = Path(ent["summary"])
        if not labels_path.is_file():
            print(f"  skip missing labels {labels_path}")
            continue
        labels = np.load(labels_path)
        if labels.shape[0] != N:
            raise SystemExit(f"{labels_path} N={labels.shape[0]} != d_E N={N}")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

        intra, inter, n_mature_scored, n_inter = _intra_inter(d_e, labels, rng=rng)
        healthy, health = _health_flags(summary, N)
        sep = (inter / intra) if (intra is not None and inter is not None and intra > 1e-12) else None

        rows.append({
            "method": ent.get("method"),
            "distance": ent.get("distance"),
            "threshold": ent.get("threshold"),
            "min_samples": ent.get("min_samples"),
            "stem": ent.get("stem"),
            "N": N,
            "n_clusters": summary.get("n_clusters"),
            "n_mature": summary.get("n_mature"),
            "n_singletons": summary.get("n_singletons"),
            "noise_frac": summary.get("noise_frac"),
            "max_cluster": summary.get("max_cluster"),
            "frac_in_mature": summary.get("frac_in_mature"),
            "intra_d_E": intra,
            "inter_d_E": inter,
            "separation": sep,
            "n_mature_scored": n_mature_scored,
            "n_inter_pairs": n_inter,
            "structurally_healthy": healthy,
            "health": health,
            "labels": str(labels_path),
            "prompts": ent.get("prompts"),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "separation" in df.columns:
        # Rank healthy configs by separation descending; unhealthy last
        df["_rank_key"] = df.apply(
            lambda r: (0 if r["structurally_healthy"] else 1, -(r["separation"] or -1.0)),
            axis=1,
        )
        df = df.sort_values("_rank_key").drop(columns=["_rank_key"]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1: canvas d_E cohesion for species labels")
    p.add_argument("--species-dir", type=Path, required=True)
    p.add_argument("--d-e", type=Path, required=True, help="Path to d_E.npy canvas")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    d_e = np.load(args.d_e)
    if d_e.ndim != 2 or d_e.shape[0] != d_e.shape[1]:
        raise SystemExit(f"{args.d_e} is not square: {d_e.shape}")
    print(f"d_E N={d_e.shape[0]} from {args.d_e}")
    print(f"scoring species in {args.species_dir}")

    df = score_species_dir(args.species_dir, d_e, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")

    healthy = df[df["structurally_healthy"]] if not df.empty else df
    if not healthy.empty:
        print("Top healthy by separation:")
        cols = ["rank", "method", "distance", "threshold", "separation", "intra_d_E", "inter_d_E", "max_cluster", "n_mature"]
        print(healthy.head(10)[cols].to_string(index=False))
    else:
        print("No structurally healthy configs found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
