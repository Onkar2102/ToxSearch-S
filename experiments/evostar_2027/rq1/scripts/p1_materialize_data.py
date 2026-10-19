#!/usr/bin/env python3
"""Materialize Part-1 pool + per-batch distance datasets under experiments/evostar_2027/data.

Layout::

    experiments/evostar_2027/data/
      pool/                 # p0_full + index (+ analysis_2000)
      batches/B{N}/p0.pkl
      mats/B{N}/            # d_E, d_P, d_EP [, d_N, d_negE]
        dataset.json        # provisional theta_sim=eps=0.1; 1d held
      provisional_radii.json
      PART1_STATUS.json

Cheap mats (1b) are built/sliced immediately. NLI (1c) is written when sources
exist under --nli-source-dir (or after p1c finishes). Does **not** run 1d.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1_materialize_data.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

# Reuse cheap-mat helpers from p1a2
from experiments.evostar_2027.rq1.scripts.p1a2_build_batches import (  # noqa: E402
    BATCH_SIZES,
    W_G,
    W_P,
    _cheap_mats_from_df,
    write_p0_batches,
)

DEFAULT_RQ1_P0 = Path("experiments/evostar_2027/rq1/p0")
DEFAULT_RQ1_MATS = Path("experiments/evostar_2027/rq1/mats")
DEFAULT_DATA = Path("experiments/evostar_2027/data")
PROVISIONAL_THETA = 0.1
PROVISIONAL_EPS = 0.1
DISTANCES_CHEAP = ("d_E", "d_P", "d_EP")
DISTANCES_NLI = ("d_N", "d_negE")


def _copy_pool(src_p0: Path, data_dir: Path) -> Dict[str, Any]:
    pool_dir = data_dir / "pool"
    pool_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}
    for name in ("p0_full.pkl", "p0_analysis_2000.pkl", "index.json"):
        src = src_p0 / name
        if not src.is_file():
            continue
        dst = pool_dir / name
        if src.resolve() == dst.resolve():
            copied[name] = str(dst)
            continue
        shutil.copy2(src, dst)
        copied[name] = str(dst)
    # Rewrite index paths to data/pool
    idx_path = pool_dir / "index.json"
    if idx_path.is_file():
        idx = json.loads(idx_path.read_text(encoding="utf-8"))
        if (pool_dir / "p0_full.pkl").is_file():
            idx["p0_full"] = str(pool_dir / "p0_full.pkl")
        if (pool_dir / "p0_analysis_2000.pkl").is_file():
            idx["p0_analysis"] = str(pool_dir / "p0_analysis_2000.pkl")
        idx["home"] = str(pool_dir)
        idx_path.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    return {"pool_dir": str(pool_dir), "files": copied}


def _save_mat(path: Path, M: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(M, dtype=np.float64))


def _write_dataset_json(
    batch_dir: Path,
    *,
    batch_id: str,
    n: int,
    p0_path: Path,
    files: Dict[str, str],
    missing: List[str],
) -> Path:
    meta = {
        "batch_id": batch_id,
        "N": n,
        "p0_path": str(p0_path),
        "distances": files,
        "missing_distances": missing,
        "provisional_radii": {
            "theta_sim": PROVISIONAL_THETA,
            "eps": PROVISIONAL_EPS,
            "dbscan_eps": PROVISIONAL_EPS,
            "note": "Provisional for Part 1 until 1d; not from neighbor-fraction calibration.",
        },
        "part1d": "held",
        "w_genotype": W_G,
        "w_phenotype": W_P,
    }
    path = batch_dir / "dataset.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def materialize_cheap(
    df_full: pd.DataFrame,
    mats_root: Path,
    sizes: Sequence[int],
    *,
    reuse_rq1_mats: Path | None,
) -> Dict[str, np.ndarray]:
    """Return full-size cheap mats; write per-batch slices under mats_root/B{N}/."""
    n_full = len(df_full)
    full: Dict[str, np.ndarray] = {}
    # Prefer rebuild for alignment with current p0_full; optionally verify reuse
    if reuse_rq1_mats is not None:
        ok = True
        for name in DISTANCES_CHEAP:
            src = reuse_rq1_mats / f"{name}_{n_full}.npy"
            if not src.is_file():
                ok = False
                break
            full[name] = np.load(src)
            if full[name].shape != (n_full, n_full):
                ok = False
                break
        if not ok:
            full = {}
    if not full:
        print(f"Building cheap mats at N={n_full} …")
        full = {k: np.asarray(v, dtype=np.float64) for k, v in _cheap_mats_from_df(df_full).items()}
    else:
        print(f"Reusing cheap mats from {reuse_rq1_mats} at N={n_full}")

    for n in sizes:
        bid = f"B{n}"
        bdir = mats_root / bid
        bdir.mkdir(parents=True, exist_ok=True)
        for name, M in full.items():
            _save_mat(bdir / f"{name}.npy", M[:n, :n])
            print(f"  wrote {bdir / (name + '.npy')} shape=({n},{n})")
    return full


def attach_nli_slices(
    mats_root: Path,
    sizes: Sequence[int],
    nli_source_dir: Path,
) -> Dict[str, Any]:
    """Copy/slice d_N / d_negE into mats/B{N}/ when source npy exists."""
    report: Dict[str, Any] = {"attached": {}, "missing": []}
    for label in DISTANCES_NLI:
        # Prefer largest available complete source among requested sizes
        sources = []
        for n in sorted(sizes, reverse=True):
            # Prefer per-batch already in data, then flat {label}_{n}.npy in source dir
            cand = [
                nli_source_dir / f"B{n}" / f"{label}.npy",
                nli_source_dir / f"{label}_{n}.npy",
                nli_source_dir / "_flat" / f"{label}_{n}.npy",
                Path("experiments/evostar_2027/data/mats/_flat") / f"{label}_{n}.npy",
                Path("experiments/evostar_2027/rq1/mats") / f"{label}_{n}.npy",
            ]
            for c in cand:
                if not c.is_file():
                    continue
                # Require complete checkpoint when a sibling .progress.json exists
                prog = c.with_suffix(".progress.json")
                if prog.is_file():
                    meta = json.loads(prog.read_text(encoding="utf-8"))
                    if not meta.get("complete"):
                        continue
                sources.append((n, c))
                break
        if not sources:
            report["missing"].append(label)
            continue
        src_n, src_path = sources[0]
        M = np.load(src_path)
        report["attached"][label] = {"source": str(src_path), "source_N": int(M.shape[0])}
        for n in sizes:
            if n > M.shape[0]:
                report.setdefault("pending_larger", {}).setdefault(label, []).append(n)
                continue
            path = mats_root / f"B{n}" / f"{label}.npy"
            _save_mat(path, M[:n, :n])
            print(f"  NLI {path} from {src_path.name} ([:{n},:{n}])")
    return report


def write_batch_dataset_jsons(
    data_dir: Path,
    sizes: Sequence[int],
) -> None:
    mats_root = data_dir / "mats"
    batches_dir = data_dir / "batches"
    for n in sizes:
        bid = f"B{n}"
        bdir = mats_root / bid
        p0_path = batches_dir / bid / "p0.pkl"
        files: Dict[str, str] = {}
        missing: List[str] = []
        for name in DISTANCES_CHEAP + DISTANCES_NLI:
            p = bdir / f"{name}.npy"
            if p.is_file():
                files[name] = str(p)
            else:
                missing.append(name)
        _write_dataset_json(bdir, batch_id=bid, n=n, p0_path=p0_path, files=files, missing=missing)
        print(f"  wrote {bdir / 'dataset.json'} (missing={missing})")


def main() -> int:
    p = argparse.ArgumentParser(description="Materialize evostar_2027/data pool + batch mats")
    p.add_argument("--src-p0", type=Path, default=DEFAULT_RQ1_P0)
    p.add_argument("--src-mats", type=Path, default=DEFAULT_RQ1_MATS)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--sizes", type=str, default="1000,2000,3139")
    p.add_argument("--rebuild-cheap", action="store_true")
    args = p.parse_args()

    sizes = tuple(int(x.strip()) for x in args.sizes.split(",") if x.strip())
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    pool_info = _copy_pool(args.src_p0, data_dir)
    p0_full = data_dir / "pool" / "p0_full.pkl"
    if not p0_full.is_file():
        raise SystemExit(f"Missing {p0_full}")
    df_full = pd.read_pickle(p0_full)
    n_full = len(df_full)
    print(f"Pool N={n_full} → {data_dir / 'pool'}")

    if n_full not in sizes:
        sizes = tuple(sorted(set(sizes) | {n_full}))

    batches_dir = data_dir / "batches"
    batch_meta = write_p0_batches(df_full, batches_dir, sizes)
    (batches_dir / "batch_index.json").write_text(
        json.dumps(
            {
                "ordering": "fitness_desc_then_run_genome_id",
                "p0_full": str(p0_full),
                "n_full": n_full,
                "batches": batch_meta,
                "nested": "B1000 ⊂ B2000 ⊂ B3139",
                "provisional_radii": {"theta_sim": PROVISIONAL_THETA, "eps": PROVISIONAL_EPS},
                "part1d": "held",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mats_root = data_dir / "mats"
    reuse = None if args.rebuild_cheap else args.src_mats
    materialize_cheap(df_full, mats_root, sizes, reuse_rq1_mats=reuse)

    nli_report = attach_nli_slices(mats_root, sizes, args.src_mats)
    write_batch_dataset_jsons(data_dir, sizes)

    radii = {
        "theta_sim": PROVISIONAL_THETA,
        "eps": PROVISIONAL_EPS,
        "dbscan_eps": PROVISIONAL_EPS,
        "applies_to": "all batches / all distances until Part 1d",
        "part1d": "held",
    }
    (data_dir / "provisional_radii.json").write_text(json.dumps(radii, indent=2), encoding="utf-8")

    status = {
        "home": str(data_dir),
        "pool": pool_info,
        "batches": [f"B{n}" for n in sizes],
        "distances_cheap": list(DISTANCES_CHEAP),
        "distances_nli": list(DISTANCES_NLI),
        "provisional_radii": radii,
        "nli": nli_report,
        "part1d": "held — do not run p1d_hparam_sweep until NLI mats ready and review begins",
        "next_nli_commands": [
            (
                "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py "
                f"--p0-dir {data_dir / 'pool'} --mats-dir {data_dir / 'mats' / '_flat'} "
                "--max-n 2000 --which both --checkpoint-every 10 --batch-size 256"
            ),
            (
                "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py "
                f"--p0-dir {data_dir / 'pool'} --mats-dir {data_dir / 'mats' / '_flat'} "
                "--max-n 3139 --allow-large-n --which both --checkpoint-every 10 --batch-size 256"
            ),
            (
                "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1_materialize_data.py "
                f"--data-dir {data_dir}  # re-attach NLI slices after p1c"
            ),
        ],
    }
    (data_dir / "PART1_STATUS.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {data_dir / 'PART1_STATUS.json'}")
    print("Part 1d: HELD")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
