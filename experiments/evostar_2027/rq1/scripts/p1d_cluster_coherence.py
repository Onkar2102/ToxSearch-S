#!/usr/bin/env python3
"""Part 1d⁺: multi-canvas cohesion for offline species labels.

For each ``species/*_labels.npy`` entry (niches formed under one distance),
score topic-clustering quality on **every available** distance canvas
(``d_E``, ``d_EP``, ``d_N``, ``d_negE``):

- mean **intra-cluster** pairwise distance (mature clusters, size ≥ 2)
- mean **inter-cluster** pairwise distance (random cross-cluster pairs)
- ``separation = inter / intra``
- **silhouette** on that precomputed canvas (noise / tiny clusters dropped)
- **Davies–Bouldin** / **Calinski–Harabasz** on that canvas (medoids; lower / higher better)

Primary ranking columns stay on the held-out embedding canvas ``d_E``
(``intra_d_E``, ``inter_d_E``, ``separation``, ``silhouette_d_E``).
Cross columns are ``{intra,inter,separation,silhouette,davies_bouldin,calinski_harabasz}_on_<canvas>``.

Feature-space Davies–Bouldin / Calinski–Harabasz still use MiniLM embedding vectors
(``davies_bouldin``, ``calinski_harabasz``).

Example (B500)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_cluster_coherence.py \\
      --species-dir experiments/evostar_2027/rq1/sweeps/B500/species \\
      --mats-dir experiments/evostar_2027/data/mats/B500 \\
      --p0 experiments/evostar_2027/data/batches/B500/p0.pkl \\
      --out experiments/evostar_2027/rq1/sweeps/B500/coherence.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

CANVAS_KEYS = ("d_E", "d_EP", "d_N", "d_negE")


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

    per_cluster_intra: List[float] = []
    for c in mature:
        idxs = np.where(labels == c)[0]
        m = _mean_pairwise(D, idxs)
        if m is not None:
            per_cluster_intra.append(m)
    intra = float(np.mean(per_cluster_intra)) if per_cluster_intra else None

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


def _mature_mask(
    labels: np.ndarray,
    *,
    noise_label: int = -1,
    min_size: int = 2,
) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=bool)
    for c in set(int(x) for x in labels):
        if c == noise_label:
            continue
        idxs = labels == c
        if int(np.sum(idxs)) >= min_size:
            mask |= idxs
    return mask


def _silhouette_precomputed(
    D: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
) -> Optional[float]:
    try:
        from sklearn.metrics import silhouette_score
    except ImportError:
        return None
    mask = _mature_mask(labels, noise_label=noise_label)
    if int(np.sum(mask)) < 4:
        return None
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return None
    sub = D[np.ix_(mask, mask)]
    sub = 0.5 * (sub + sub.T)
    np.fill_diagonal(sub, 0.0)
    try:
        return float(silhouette_score(sub, labs, metric="precomputed"))
    except Exception:
        return None


def _davies_bouldin_emb(
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
) -> Optional[float]:
    try:
        from sklearn.metrics import davies_bouldin_score
    except ImportError:
        return None
    mask = labels != noise_label
    if int(np.sum(mask)) < 2:
        return None
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return None
    try:
        return float(davies_bouldin_score(emb[mask], labs))
    except Exception:
        return None


def _calinski_harabasz_emb(
    emb: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
) -> Optional[float]:
    try:
        from sklearn.metrics import calinski_harabasz_score
    except ImportError:
        return None
    mask = labels != noise_label
    if int(np.sum(mask)) < 2:
        return None
    labs = labels[mask]
    if len(np.unique(labs)) < 2:
        return None
    try:
        return float(calinski_harabasz_score(emb[mask], labs))
    except Exception:
        return None


def _cluster_medoid(D: np.ndarray, idxs: np.ndarray) -> int:
    if len(idxs) == 1:
        return int(idxs[0])
    sub = D[np.ix_(idxs, idxs)]
    return int(idxs[int(np.argmin(sub.sum(axis=1)))])


def _davies_bouldin_precomputed(
    D: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
) -> Optional[float]:
    """Medoid Davies–Bouldin on a precomputed dissimilarity canvas."""
    ids = sorted({int(c) for c in labels if int(c) != noise_label})
    if len(ids) < 2:
        return None
    medoids: Dict[int, int] = {}
    spreads: Dict[int, float] = {}
    for c in ids:
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        m = _cluster_medoid(D, idxs)
        medoids[c] = m
        spreads[c] = 0.0 if len(idxs) == 1 else float(np.mean(D[idxs, m]))
    if len(medoids) < 2:
        return None
    ratios: List[float] = []
    for ci in medoids:
        best = 0.0
        found = False
        for cj in medoids:
            if cj == ci:
                continue
            dij = float(D[medoids[ci], medoids[cj]])
            if dij <= 1e-12:
                continue
            found = True
            best = max(best, (spreads[ci] + spreads[cj]) / dij)
        if found:
            ratios.append(best)
    if not ratios:
        return None
    return float(np.mean(ratios))


def _calinski_harabasz_precomputed(
    D: np.ndarray,
    labels: np.ndarray,
    *,
    noise_label: int = -1,
) -> Optional[float]:
    """Medoid Calinski–Harabasz on a precomputed dissimilarity canvas."""
    idxs_all = np.where(labels != noise_label)[0]
    ids = sorted({int(c) for c in labels[idxs_all]})
    n = int(len(idxs_all))
    k = int(len(ids))
    if k < 2 or n <= k:
        return None
    gmed = _cluster_medoid(D, idxs_all)
    within = 0.0
    between = 0.0
    for c in ids:
        idxs = np.where(labels == c)[0]
        m = _cluster_medoid(D, idxs)
        within += float(np.sum(np.square(D[idxs, m])))
        between += float(len(idxs)) * float(D[m, gmed] ** 2)
    if within <= 1e-12:
        return None
    return (between / (k - 1)) / (within / (n - k))


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
        return False, "over_fragmented"
    return True, "ok"


def _load_embeddings(*, p0: Optional[Path], emb_path: Optional[Path], N: int) -> np.ndarray:
    if emb_path is not None:
        X = np.load(emb_path)
        if X.ndim != 2 or X.shape[0] != N:
            raise SystemExit(f"{emb_path} shape {X.shape} incompatible with N={N}")
        return np.asarray(X, dtype=np.float64)
    if p0 is None:
        raise SystemExit("Need --p0 or --emb for Davies–Bouldin / Calinski–Harabasz")
    if not p0.is_file():
        raise SystemExit(f"Missing p0 pickle: {p0}")
    df = pd.read_pickle(p0)
    if "emb" not in df.columns:
        raise SystemExit(f"{p0} missing 'emb' column")
    if len(df) != N:
        raise SystemExit(f"{p0} has {len(df)} rows, expected N={N}")
    X = np.vstack(df["emb"].to_numpy())
    if X.shape[0] != N:
        raise SystemExit(f"Stacked emb shape {X.shape} != N={N}")
    return np.asarray(X, dtype=np.float64)


def _load_canvases(
    *,
    mats_dir: Optional[Path],
    d_e_path: Optional[Path],
) -> Dict[str, np.ndarray]:
    """Load square distance mats; d_E is required."""
    canvases: Dict[str, np.ndarray] = {}
    if mats_dir is not None:
        if not mats_dir.is_dir():
            raise SystemExit(f"Missing mats dir {mats_dir}")
        for key in CANVAS_KEYS:
            path = mats_dir / f"{key}.npy"
            if not path.is_file():
                print(f"  [warn] missing canvas {path} — skip scoring under {key}")
                continue
            D = np.load(path)
            if D.ndim != 2 or D.shape[0] != D.shape[1]:
                raise SystemExit(f"{path} is not square: {D.shape}")
            canvases[key] = D
    elif d_e_path is not None:
        D = np.load(d_e_path)
        if D.ndim != 2 or D.shape[0] != D.shape[1]:
            raise SystemExit(f"{d_e_path} is not square: {D.shape}")
        canvases["d_E"] = D
    else:
        raise SystemExit("Need --mats-dir or --d-e")

    if "d_E" not in canvases:
        raise SystemExit("d_E canvas is required (mats-dir/d_E.npy or --d-e)")

    N = int(canvases["d_E"].shape[0])
    for key, D in list(canvases.items()):
        if D.shape[0] != N:
            raise SystemExit(f"{key} N={D.shape[0]} != d_E N={N}")
    return canvases


def _score_on_canvas(
    D: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    intra, inter, n_mature_scored, n_inter = _intra_inter(D, labels, rng=rng)
    sep = (inter / intra) if (intra is not None and inter is not None and intra > 1e-12) else None
    sil = _silhouette_precomputed(D, labels)
    return {
        "intra": intra,
        "inter": inter,
        "separation": sep,
        "silhouette": sil,
        "davies_bouldin": _davies_bouldin_precomputed(D, labels),
        "calinski_harabasz": _calinski_harabasz_precomputed(D, labels),
        "n_mature_scored": n_mature_scored,
        "n_inter_pairs": n_inter,
    }


def score_species_dir(
    species_dir: Path,
    canvases: Dict[str, np.ndarray],
    emb: np.ndarray,
    *,
    seed: int = 0,
) -> pd.DataFrame:
    index_path = species_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path}")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("entries", [])
    d_e = canvases["d_E"]
    N = int(d_e.shape[0])
    if emb.shape[0] != N:
        raise SystemExit(f"emb N={emb.shape[0]} != d_E N={N}")
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    skip_stems: set = set()
    existing_path = getattr(score_species_dir, "_existing_out", None)
    if existing_path and Path(existing_path).is_file():
        try:
            prev = pd.read_csv(existing_path)
            if "stem" in prev.columns:
                skip_stems = set(prev["stem"].astype(str))
                rows.extend(prev.to_dict(orient="records"))
                print(f"  reuse {len(skip_stems)} scored stems from {existing_path}")
        except Exception as e:
            print(f"  [warn] could not reuse {existing_path}: {e}")

    for ent in entries:
        if ent.get("stem") in skip_stems:
            continue
        labels_path = Path(ent["labels"])
        summary_path = Path(ent["summary"])
        if not labels_path.is_file():
            print(f"  skip missing labels {labels_path}")
            continue
        labels = np.load(labels_path)
        if labels.shape[0] != N:
            raise SystemExit(f"{labels_path} N={labels.shape[0]} != d_E N={N}")
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}

        healthy, health = _health_flags(summary, N)
        db = _davies_bouldin_emb(emb, labels)
        ch = _calinski_harabasz_emb(emb, labels)

        # Score under every available canvas (independent RNG streams via same rng).
        scored: Dict[str, Dict[str, Any]] = {}
        for key in CANVAS_KEYS:
            if key not in canvases:
                continue
            scored[key] = _score_on_canvas(canvases[key], labels, rng)

        # Primary columns: held-out embedding canvas (backward compatible).
        se = scored["d_E"]
        row: Dict[str, Any] = {
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
            "intra_d_E": se["intra"],
            "inter_d_E": se["inter"],
            "separation": se["separation"],
            "silhouette_d_E": se["silhouette"],
            "davies_bouldin": db,
            "calinski_harabasz": ch,
            "n_mature_scored": se["n_mature_scored"],
            "n_inter_pairs": se["n_inter_pairs"],
            "structurally_healthy": healthy,
            "health": health,
            "labels": str(labels_path),
            "prompts": ent.get("prompts"),
            "canvases_scored": ",".join(scored.keys()),
        }
        for key, s in scored.items():
            row[f"intra_on_{key}"] = s["intra"]
            row[f"inter_on_{key}"] = s["inter"]
            row[f"separation_on_{key}"] = s["separation"]
            row[f"silhouette_on_{key}"] = s["silhouette"]
            row[f"davies_bouldin_on_{key}"] = s["davies_bouldin"]
            row[f"calinski_harabasz_on_{key}"] = s["calinski_harabasz"]
        # Forming-distance (diagonal) convenience columns
        form = str(ent.get("distance") or "")
        if form in scored:
            row["separation_on_forming"] = scored[form]["separation"]
            row["silhouette_on_forming"] = scored[form]["silhouette"]
            row["intra_on_forming"] = scored[form]["intra"]
            row["inter_on_forming"] = scored[form]["inter"]
            row["davies_bouldin_on_forming"] = scored[form]["davies_bouldin"]
            row["calinski_harabasz_on_forming"] = scored[form]["calinski_harabasz"]
        else:
            row["separation_on_forming"] = None
            row["silhouette_on_forming"] = None
            row["intra_on_forming"] = None
            row["inter_on_forming"] = None
            row["davies_bouldin_on_forming"] = None
            row["calinski_harabasz_on_forming"] = None

        rows.append(row)

    df = pd.DataFrame(rows)
    if "rank" in df.columns:
        df = df.drop(columns=["rank"])
    if not df.empty and "separation" in df.columns:
        df["_rank_key"] = df.apply(
            lambda r: (0 if r["structurally_healthy"] else 1, -(r["separation"] or -1.0)),
            axis=1,
        )
        df = df.sort_values("_rank_key").drop(columns=["_rank_key"]).reset_index(drop=True)
        df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def backfill_canvas_cluster_metrics(
    out_csv: Path,
    canvases: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """Add per-canvas medoid DB/CH to an existing coherence.csv."""
    df = pd.read_csv(out_csv)
    if "labels" not in df.columns:
        raise SystemExit(f"{out_csv} has no labels column")
    n_ok = 0
    for i, row in df.iterrows():
        labels_path = Path(str(row["labels"]))
        if not labels_path.is_file():
            print(f"  skip missing labels {labels_path}")
            continue
        labels = np.load(labels_path)
        for key, D in canvases.items():
            if labels.shape[0] != D.shape[0]:
                raise SystemExit(f"{labels_path} N={labels.shape[0]} != {key} N={D.shape[0]}")
            df.at[i, f"davies_bouldin_on_{key}"] = _davies_bouldin_precomputed(D, labels)
            df.at[i, f"calinski_harabasz_on_{key}"] = _calinski_harabasz_precomputed(D, labels)
        form = str(row.get("distance") or "")
        db_col = f"davies_bouldin_on_{form}"
        ch_col = f"calinski_harabasz_on_{form}"
        df.at[i, "davies_bouldin_on_forming"] = df.at[i, db_col] if db_col in df.columns else None
        df.at[i, "calinski_harabasz_on_forming"] = df.at[i, ch_col] if ch_col in df.columns else None
        n_ok += 1
        if n_ok % 20 == 0:
            print(f"  scored {n_ok}/{len(df)} labelings")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1: multi-canvas cohesion for species labels")
    p.add_argument("--species-dir", type=Path, required=True)
    p.add_argument("--mats-dir", type=Path, default=None, help="Dir with d_E/d_EP/d_N/d_negE .npy")
    p.add_argument("--d-e", type=Path, default=None, help="Legacy: only d_E.npy canvas")
    p.add_argument("--p0", type=Path, default=None, help="Batch p0.pkl with emb column")
    p.add_argument("--emb", type=Path, default=None, help="Optional (N, D) embeddings .npy")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--reuse-out",
        action="store_true",
        default=True,
        help="Skip stems already present in --out (default on)",
    )
    p.add_argument(
        "--backfill-dbch",
        action="store_true",
        help="Add per-canvas medoid DB/CH onto an existing --out CSV and exit",
    )
    args = p.parse_args()

    canvases = _load_canvases(mats_dir=args.mats_dir, d_e_path=args.d_e)
    if args.backfill_dbch:
        if not args.out.is_file():
            raise SystemExit(f"--backfill-dbch needs an existing {args.out}")
        print(f"backfill DB/CH canvases={list(canvases.keys())} into {args.out}")
        df = backfill_canvas_cluster_metrics(args.out, canvases)
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out} ({len(df)} rows)")
        return 0
    emb = _load_embeddings(p0=args.p0, emb_path=args.emb, N=int(canvases["d_E"].shape[0]))
    print(f"canvases={list(canvases.keys())} N={canvases['d_E'].shape[0]}")
    print(f"emb shape={emb.shape} from {args.emb or args.p0}")
    print(f"scoring species in {args.species_dir}")

    score_species_dir._existing_out = str(args.out) if args.reuse_out else None
    df = score_species_dir(args.species_dir, canvases, emb, seed=args.seed)
    score_species_dir._existing_out = None
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out} ({len(df)} rows)")

    healthy = df[df["structurally_healthy"]] if not df.empty else df
    if not healthy.empty:
        print("Top healthy by Separation@d_E:")
        cols = [
            "rank", "method", "distance", "threshold", "separation",
            "separation_on_forming", "silhouette_d_E", "silhouette_on_forming",
            "davies_bouldin", "calinski_harabasz", "max_cluster", "n_mature",
        ]
        cols = [c for c in cols if c in healthy.columns]
        print(healthy.head(10)[cols].to_string(index=False))
    else:
        print("No structurally healthy configs found")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
