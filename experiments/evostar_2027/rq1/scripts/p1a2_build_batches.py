#!/usr/bin/env python3
"""Build nested RQ1 batches B1000 ⊂ B2000 ⊂ B3139 from fitness-sorted p0_full.

Writes::

    experiments/evostar_2027/rq1/batches/
      batch_index.json
      B1000/p0.pkl
      B2000/p0.pkl
      B3139/p0.pkl

Cheap mats ``d_E`` / ``d_P`` / ``d_EP`` are built once at full N and sliced::

    experiments/evostar_2027/rq1/mats/d_{E,P,EP}_{1000,2000,3139}.npy

NLI (``d_N`` / ``d_negE``): not computed at 3139. If ``d_*_2000.npy`` exists
(from p1c --max-n 2000), this script slices B1000. Otherwise see
``batch_index.json`` → ``nli_commands``.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1a2_build_batches.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

DEFAULT_P0 = Path("experiments/evostar_2027/rq1/p0")
DEFAULT_BATCHES = Path("experiments/evostar_2027/rq1/batches")
DEFAULT_MATS = Path("experiments/evostar_2027/rq1/mats")
BATCH_SIZES: Tuple[int, ...] = (1000, 2000, 3139)
W_G, W_P = 0.7, 0.3


def _d_e_matrix(emb: np.ndarray) -> np.ndarray:
    """ToxSearch d_E = (1 - cos) / 2 on L2-normalized rows."""
    X = np.asarray(emb, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)
    sims = np.clip(X @ X.T, -1.0, 1.0)
    D = (1.0 - sims) / 2.0
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def _d_p_matrix(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)) / np.sqrt(x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def _cheap_mats_from_df(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    emb = np.vstack(df["emb"].map(np.asarray).to_list())
    scores = np.vstack(df["scores_google"].map(np.asarray).to_list())
    d_e = _d_e_matrix(emb)
    d_p = _d_p_matrix(scores)
    return {"d_E": d_e, "d_P": d_p, "d_EP": W_G * d_e + W_P * d_p}


def _batch_id(n: int) -> str:
    return f"B{n}"


def write_p0_batches(
    df_full: pd.DataFrame,
    batches_dir: Path,
    sizes: Sequence[int],
) -> List[Dict[str, Any]]:
    batches_dir.mkdir(parents=True, exist_ok=True)
    n_full = len(df_full)
    meta_batches: List[Dict[str, Any]] = []
    for n in sizes:
        if n > n_full:
            raise SystemExit(f"Batch N={n} > p0_full rows={n_full}")
        bid = _batch_id(n)
        out_dir = batches_dir / bid
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = df_full.iloc[:n].reset_index(drop=True)
        pkl = out_dir / "p0.pkl"
        sub.to_pickle(pkl)
        fit = sub["fitness"].astype(float)
        meta_batches.append({
            "batch_id": bid,
            "N": n,
            "p0_path": str(pkl),
            "fitness_min": float(fit.min()),
            "fitness_max": float(fit.max()),
            "fitness_median": float(fit.median()),
            "nested_prefix_of": _batch_id(n_full) if n < n_full else None,
        })
        print(f"Wrote {pkl} ({n} rows)")
    return meta_batches


def build_cheap_mats_full_and_slice(
    df_full: pd.DataFrame,
    mats_dir: Path,
    sizes: Sequence[int],
) -> Dict[str, Any]:
    """Compute d_E/d_P/d_EP at max N; save each batch size as a leading principal submatrix."""
    mats_dir.mkdir(parents=True, exist_ok=True)
    n_full = len(df_full)
    print(f"Building cheap mats at N={n_full} …")
    full = {k: np.asarray(v, dtype=np.float64) for k, v in _cheap_mats_from_df(df_full).items()}
    files: Dict[str, Dict[str, str]] = {str(n): {} for n in sizes}
    for n in sizes:
        for name, M in full.items():
            path = mats_dir / f"{name}_{n}.npy"
            np.save(path, M[:n, :n].copy())
            files[str(n)][name] = str(path)
            print(f"  wrote {path} shape=({n},{n})")

    # Keep legacy names expected by p1d when N=2000
    for name in ("d_E", "d_P", "d_EP"):
        legacy = mats_dir / f"{name}_2000.npy"
        if 2000 in sizes and legacy.resolve() != Path(files["2000"][name]).resolve():
            np.save(legacy, full[name][:2000, :2000].copy())

    meta = {
        "N_full": n_full,
        "sizes": list(sizes),
        "w_genotype": W_G,
        "w_phenotype": W_P,
        "files_by_N": files,
        "source_p0": "p0_full leading prefixes (fitness-sorted)",
    }
    meta_path = mats_dir / "cheap_mats_batches_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")
    return meta


def slice_nli_if_present(mats_dir: Path, sizes_nli: Sequence[int] = (1000, 2000)) -> Dict[str, Any]:
    """If d_N_2000 / d_negE_2000 exist, write B1000 slices. Never invent N=3139 NLI."""
    out: Dict[str, Any] = {"sliced": {}, "missing_sources": [], "commands": []}
    max_n = max(sizes_nli)
    for label in ("d_N", "d_negE"):
        src = mats_dir / f"{label}_{max_n}.npy"
        if not src.is_file():
            # try smoke leftover — do not slice incomplete smoke as if full
            out["missing_sources"].append(str(src))
            continue
        M = np.load(src)
        if M.shape[0] < max_n:
            out["missing_sources"].append(f"{src} shape={M.shape} < {max_n}")
            continue
        for n in sizes_nli:
            if n > M.shape[0]:
                continue
            path = mats_dir / f"{label}_{n}.npy"
            np.save(path, M[:n, :n].copy())
            out["sliced"].setdefault(label, {})[str(n)] = str(path)
            print(f"  sliced NLI {path} from {src.name}")

    out["commands"] = [
        (
            "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py "
            "--max-n 2000 --which both --checkpoint-every 10 --batch-size 256"
        ),
        (
            "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1a2_build_batches.py "
            "--skip-cheap-mats --slice-nli-only"
        ),
        "No d_N / d_negE at N=3139 (compute budget).",
    ]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1a2: nested batches B1000/B2000/B3139")
    p.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    p.add_argument("--batches-dir", type=Path, default=DEFAULT_BATCHES)
    p.add_argument("--mats-dir", type=Path, default=DEFAULT_MATS)
    p.add_argument(
        "--sizes",
        type=str,
        default="1000,2000,3139",
        help="Comma-separated batch sizes (nested prefixes of p0_full)",
    )
    p.add_argument("--skip-cheap-mats", action="store_true")
    p.add_argument(
        "--slice-nli-only",
        action="store_true",
        help="Only slice NLI mats from d_*_2000.npy (skip P0 rewrite if already present)",
    )
    args = p.parse_args()

    sizes = tuple(int(x.strip()) for x in args.sizes.split(",") if x.strip())
    p0_full_path = args.p0_dir / "p0_full.pkl"
    if not p0_full_path.is_file():
        raise SystemExit(f"Missing {p0_full_path}; run p1a first")

    df_full = pd.read_pickle(p0_full_path)
    if not isinstance(df_full, pd.DataFrame):
        raise SystemExit("p0_full.pkl is not a DataFrame")
    n_full = len(df_full)
    print(f"Loaded {p0_full_path} N={n_full}")
    if sizes[-1] != n_full and max(sizes) != n_full:
        # allow last size to mean "full" if it matches len
        pass
    if max(sizes) > n_full:
        raise SystemExit(f"Requested max size {max(sizes)} > p0_full {n_full}")

    # Normalize: ensure full pool size is included as largest batch id
    if n_full not in sizes:
        sizes = tuple(sorted(set(sizes) | {n_full}))

    if not args.slice_nli_only:
        batch_meta = write_p0_batches(df_full, args.batches_dir, sizes)
    else:
        batch_meta = []
        for n in sizes:
            bid = _batch_id(n)
            pkl = args.batches_dir / bid / "p0.pkl"
            if pkl.is_file():
                batch_meta.append({"batch_id": bid, "N": n, "p0_path": str(pkl)})

    cheap_meta: Dict[str, Any] = {}
    if not args.skip_cheap_mats and not args.slice_nli_only:
        cheap_meta = build_cheap_mats_full_and_slice(df_full, args.mats_dir, sizes)
    elif args.skip_cheap_mats:
        print("Skipping cheap mats (--skip-cheap-mats)")

    nli_meta = slice_nli_if_present(args.mats_dir, sizes_nli=[s for s in sizes if s <= 2000])

    index = {
        "ordering": "fitness_desc_then_run_genome_id (same as p0_full)",
        "p0_full": str(p0_full_path),
        "n_full": n_full,
        "batches": batch_meta,
        "nested": "B1000 ⊂ B2000 ⊂ B3139" if n_full == 3139 else f"prefixes of N={n_full}",
        "cheap_mats": cheap_meta,
        "nli": nli_meta,
        "nli_policy": (
            "Compute d_N and d_negE only up to N=2000 via p1c; "
            "B1000 is a leading slice. No NLI matrix at full N=3139."
        ),
    }
    index_path = args.batches_dir / "batch_index.json"
    args.batches_dir.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {index_path}")
    if nli_meta.get("missing_sources"):
        print("NLI sources missing — run p1c at --max-n 2000, then re-run with --slice-nli-only:")
        for cmd in nli_meta.get("commands", []):
            print(f"  {cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
