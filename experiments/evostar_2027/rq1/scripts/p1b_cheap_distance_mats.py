#!/usr/bin/env python3
"""Part 1b: cheap distance mats d_E / d_P / d_EP on the analysis index.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1b_cheap_distance_mats.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.stage_analysis import stage_a

DEFAULT_P0 = Path("experiments/evostar_2027/rq1/p0")
DEFAULT_MATS = Path("experiments/evostar_2027/rq1/mats")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1b: cheap distance matrices")
    p.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    p.add_argument("--mats-dir", type=Path, default=DEFAULT_MATS)
    p.add_argument("--analysis-pkl", type=Path, default=None)
    args = p.parse_args()

    index_path = args.p0_dir / "index.json"
    if not index_path.is_file():
        raise SystemExit(f"Missing {index_path}; run p1a first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    pkl = args.analysis_pkl or Path(index["p0_analysis"])
    df = stage_a.load_p0(str(pkl), require_openai=False)
    n = len(df)
    print(f"Building cheap mats for N={n} from {pkl}")

    mats = stage_a.distance_matrices(df, nli=False, evaluators=("google",))
    mats_dir = args.mats_dir
    mats_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "d_E": mats["d_E"],
        "d_P": mats["d_P[google]"],
        "d_EP": mats["d_EP[google]"],
    }
    meta = {
        "N": n,
        "p0_analysis": str(pkl),
        "index_json": str(index_path),
        "files": {},
        "w_genotype": stage_a.W_G,
        "w_phenotype": stage_a.W_P,
    }
    for name, m in mapping.items():
        path = mats_dir / f"{name}_{n}.npy"
        np.save(path, m.astype(np.float64))
        meta["files"][name] = str(path)
        print(f"  wrote {path} shape={m.shape}")

    # Shared analysis row order (aligned with mats)
    shared = {
        "N": n,
        "p0_analysis": str(pkl),
        "rows": [
            {
                "i": int(i),
                "run": row.run if hasattr(row, "run") else None,
                "genome_id": int(row.genome_id) if pd.notna(row.genome_id) else row.genome_id,
                "fitness": float(row.fitness) if hasattr(row, "fitness") else None,
                "prompt": row.prompt,
            }
            for i, row in df.iterrows()
        ],
    }
    shared_path = mats_dir / "index.json"
    shared_path.write_text(json.dumps(shared, indent=2), encoding="utf-8")
    meta_path = mats_dir / "cheap_mats_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {shared_path}")
    print(f"Wrote {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
