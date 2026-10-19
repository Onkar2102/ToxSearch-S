#!/usr/bin/env python3
"""RQ1 Stage A over nested held-out prefixes (Phase 1: B1=1000, B2=2000).

Example::

    PYTHONPATH=src:. python experiments/stage_analysis/run_rq1_batches.py \\
      --p0 results/rq1_heldout/p0_2000.pkl --out-dir results/rq1_heldout \\
      --batches 1000 2000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from experiments.stage_analysis import stage_a


def _mats_dir(out_dir: Path) -> Path:
    d = out_dir / "mats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def compute_cheap_mats(p0: pd.DataFrame, max_n: int, mats_dir: Path) -> Dict[str, np.ndarray]:
    """d_E, d_P[google], d_EP[google] for the full prefix; persist .npy."""
    chunk = p0.iloc[:max_n].reset_index(drop=True)
    mats = stage_a.distance_matrices(chunk, nli=False, evaluators=("google",))
    mapping = {
        "d_E": mats["d_E"],
        "d_P_google": mats["d_P[google]"],
        "d_EP_google": mats["d_EP[google]"],
    }
    for name, m in mapping.items():
        path = mats_dir / f"{name}_{max_n}.npy"
        np.save(path, m)
        print(f"  wrote {path} shape={m.shape}")
    return mats


def pairwise_nli_checkpointed(
    texts: Sequence[str],
    out_path: Path,
    *,
    checkpoint_every: int = 10,
    batch_size: int = 64,
) -> np.ndarray:
    """Symmetric d_N with resume support (fills upper triangle row-by-row)."""
    from speciation.distance_metrics.nli import get_nli_stats, nli_distances_from_query_torch

    n = len(texts)
    if out_path.is_file():
        dist = np.load(out_path)
        if dist.shape != (n, n):
            raise ValueError(f"Checkpoint shape {dist.shape} != ({n}, {n})")
        print(f"  resumed d_N from {out_path}")
    else:
        dist = np.zeros((n, n), dtype=np.float64)

    progress_path = out_path.with_suffix(".progress.json")
    start_i = 0
    if progress_path.is_file():
        start_i = int(json.loads(progress_path.read_text()).get("next_i", 0))

    t0 = time.time()
    pairs_at_start = sum(n - 1 - r for r in range(start_i))
    for i in range(start_i, n):
        others = [texts[j] for j in range(i + 1, n)]
        if others:
            row = nli_distances_from_query_torch(texts[i], others, batch_size=batch_size)
            for k, j in enumerate(range(i + 1, n)):
                dist[i, j] = row[k]
                dist[j, i] = row[k]
        if (i + 1) % 5 == 0:
            try:
                import torch
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        if (i + 1) % checkpoint_every == 0 or i + 1 == n:
            np.save(out_path, dist)
            progress_path.write_text(json.dumps({"next_i": i + 1, "n": n}), encoding="utf-8")
            stats = get_nli_stats()
            elapsed = time.time() - t0
            pairs_done = sum(n - 1 - r for r in range(i + 1))
            pairs_total = n * (n - 1) // 2
            pairs_this_session = pairs_done - pairs_at_start
            rate = pairs_this_session / elapsed if elapsed > 0 else 0.0
            eta = (pairs_total - pairs_done) / rate if rate > 0 else float("inf")
            print(
                f"  d_N checkpoint row {i + 1}/{n} pairs={pairs_done}/{pairs_total} "
                f"({100 * pairs_done / pairs_total:.1f}%) "
                f"rate={rate:.1f}/s ETA={eta / 3600:.1f}h "
                f"model_passes={stats.get('model_passes')}",
                flush=True,
            )
    progress_path.write_text(json.dumps({"next_i": n, "n": n, "complete": True}), encoding="utf-8")
    return dist


def load_or_compute_mats(
    p0: pd.DataFrame,
    max_n: int,
    mats_dir: Path,
    *,
    nli: bool,
    checkpoint_every: int,
    batch_size: int = 64,
) -> Dict[str, np.ndarray]:
    paths = {
        "d_E": mats_dir / f"d_E_{max_n}.npy",
        "d_P[google]": mats_dir / f"d_P_google_{max_n}.npy",
        "d_EP[google]": mats_dir / f"d_EP_google_{max_n}.npy",
    }
    if all(p.is_file() for p in paths.values()):
        mats = {k: np.load(v) for k, v in paths.items()}
        print(f"  loaded cheap mats from {mats_dir}")
    else:
        print(f"  computing cheap mats for N={max_n}")
        mats = compute_cheap_mats(p0, max_n, mats_dir)

    if nli:
        nli_path = mats_dir / f"d_N_{max_n}.npy"
        texts = p0.iloc[:max_n]["prompt"].tolist()
        if nli_path.is_file() and (mats_dir / f"d_N_{max_n}.progress.json").is_file():
            prog = json.loads((mats_dir / f"d_N_{max_n}.progress.json").read_text())
            if prog.get("complete") and prog.get("n") == max_n:
                mats["d_N"] = np.load(nli_path)
                print(f"  loaded complete d_N from {nli_path}")
                return mats
        print(f"  computing d_N for N={max_n} (checkpoint every {checkpoint_every} rows)")
        mats["d_N"] = pairwise_nli_checkpointed(
            texts,
            nli_path,
            checkpoint_every=checkpoint_every,
            batch_size=batch_size,
        )
    return mats


def slice_mats(mats: Dict[str, np.ndarray], n: int) -> Dict[str, np.ndarray]:
    return {k: v[:n, :n].copy() for k, v in mats.items()}


def analyze_batch(
    mats: Dict[str, np.ndarray],
    out_dir: Path,
    *,
    knn_k: int,
    reference_radius: float,
    n_boot: int,
    seed: int,
) -> Dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Primary RQ1 distances (drop raw d_P from correlation tables to match paper focus)
    primary_keys = [k for k in ("d_E", "d_N", "d_EP[google]") if k in mats]
    primary = {k: mats[k] for k in primary_keys}
    # Keep d_P available for output_term_share
    full = dict(mats)

    spearman = stage_a.spearman_table(primary)
    knn = stage_a.knn_table(primary, k=knn_k)
    quantiles = stage_a.quantile_table(primary)
    calib = stage_a.calibration_table(primary, reference="d_E", reference_radius=reference_radius)
    share = stage_a.output_term_share(full, "google") if "d_EP[google]" in full and "d_P[google]" in full else {}
    boot = stage_a.prompt_resample_cis(primary, knn_k=knn_k, n_boot=n_boot, seed=seed)

    spearman.to_csv(out_dir / "spearman.csv")
    knn.to_csv(out_dir / "knn_jaccard.csv")
    quantiles.to_csv(out_dir / "quantiles.csv")
    calib.to_csv(out_dir / "calibration.csv", index=False)
    (out_dir / "output_term_share.json").write_text(json.dumps(share, indent=2), encoding="utf-8")
    (out_dir / "bootstrap_ci.json").write_text(json.dumps(boot, indent=2), encoding="utf-8")
    print(f"  wrote Stage A tables -> {out_dir}")
    return {
        "spearman": spearman,
        "knn": knn,
        "quantiles": quantiles,
        "calibration": calib,
        "share": share,
        "bootstrap": boot,
    }


def compare_batches(results: Dict[int, Dict[str, object]], out_dir: Path) -> pd.DataFrame:
    rows: List[Dict] = []
    for n, res in sorted(results.items()):
        sp: pd.DataFrame = res["spearman"]  # type: ignore
        kn: pd.DataFrame = res["knn"]  # type: ignore
        qu: pd.DataFrame = res["quantiles"]  # type: ignore
        ca: pd.DataFrame = res["calibration"]  # type: ignore
        share: Dict = res["share"]  # type: ignore

        def _sp(a: str, b: str) -> float:
            if a in sp.index and b in sp.columns:
                return float(sp.loc[a, b])
            return float("nan")

        def _kn(a: str, b: str) -> float:
            if a in kn.index and b in kn.columns:
                return float(kn.loc[a, b])
            return float("nan")

        row: Dict = {
            "N": n,
            "spearman_dE_dN": _sp("d_E", "d_N"),
            "spearman_dE_dEP": _sp("d_E", "d_EP[google]"),
            "spearman_dN_dEP": _sp("d_N", "d_EP[google]"),
            "knn_dE_dN": _kn("d_E", "d_N"),
            "knn_dE_dEP": _kn("d_E", "d_EP[google]"),
            "knn_dN_dEP": _kn("d_N", "d_EP[google]"),
            "share_mean": share.get("mean"),
            "share_median": share.get("median"),
            "share_p90": share.get("p90"),
        }
        for dist in ("d_E", "d_N", "d_EP[google]"):
            if dist in qu.index:
                for q in (0.05, 0.25, 0.5):
                    if q in qu.columns:
                        row[f"q{q}_{dist}"] = float(qu.loc[dist, q])
            hit = ca.loc[ca["distance"] == dist]
            if len(hit):
                row[f"theta_sim_{dist}"] = float(hit["theta_sim"].iloc[0])
                row[f"eps_{dist}"] = float(hit["eps"].iloc[0])
                row["target_fraction"] = float(hit["target_fraction"].iloc[0])
        rows.append(row)

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "b1_b2.csv"
    df.to_csv(path, index=False)
    print(f"Wrote comparison -> {path}")
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 Stage A over held-out prefixes")
    p.add_argument("--p0", type=Path, default=Path("results/rq1_heldout/p0_2000.pkl"))
    p.add_argument("--out-dir", type=Path, default=Path("results/rq1_heldout"))
    p.add_argument("--batches", type=int, nargs="+", default=[1000, 2000])
    p.add_argument("--no-nli", action="store_true")
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--reference-radius", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-n", type=int, default=None, help="Override matrix size (default=max batch)")
    args = p.parse_args()

    p0 = stage_a.load_p0(str(args.p0), require_openai=False)
    max_n = args.max_n or max(args.batches)
    if len(p0) < max_n:
        raise SystemExit(f"P0 has {len(p0)} rows < max_n={max_n}")
    for n in args.batches:
        if n > max_n:
            raise SystemExit(f"batch {n} > max_n {max_n}")

    mats_dir = _mats_dir(args.out_dir)
    print(f"Matrices for N={max_n} -> {mats_dir}")
    full_mats = load_or_compute_mats(
        p0,
        max_n,
        mats_dir,
        nli=not args.no_nli,
        checkpoint_every=args.checkpoint_every,
        batch_size=args.batch_size,
    )

    results: Dict[int, Dict[str, object]] = {}
    for n in sorted(args.batches):
        print(f"Analyzing B{n}")
        sliced = slice_mats(full_mats, n)
        results[n] = analyze_batch(
            sliced,
            args.out_dir / f"B{n}",
            knn_k=args.knn_k,
            reference_radius=args.reference_radius,
            n_boot=args.n_boot,
            seed=args.seed,
        )

    if len(results) >= 2:
        compare_batches(results, args.out_dir / "comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
