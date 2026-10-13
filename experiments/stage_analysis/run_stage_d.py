#!/usr/bin/env python3
"""Stage D CLI: re-score exported top-k prompts under both live oracles.

Example::

    PYTHONPATH=src:. python experiments/stage_analysis/run_stage_d.py \\
      --topk results/stagec/rq4_topk_for_cross_scoring.csv \\
      --out results/stageD/cross_scores.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.stage_analysis import rq


def main() -> int:
    p = argparse.ArgumentParser(description="Stage D: top-k cross-oracle rescoring")
    p.add_argument("--topk", type=Path, required=True, help="CSV from run_all.py export_topk")
    p.add_argument("--out", type=Path, default=Path("results/stageD/cross_scores.csv"))
    p.add_argument("--rg-config", default="config/RGConfig.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--openai-model", default="omni-moderation-latest")
    p.add_argument(
        "--measure-oracles",
        nargs="+",
        default=["google", "openai"],
        help="Measurement oracles (default: google openai)",
    )
    p.add_argument("--matrix-out", type=Path, default=None, help="Optional pivot CSV path")
    args = p.parse_args()

    df = rq.rescore_topk(
        args.topk,
        args.out,
        measure_oracles=args.measure_oracles,
        rg_config=args.rg_config,
        seed=args.seed,
        openai_model=args.openai_model,
    )
    print(f"Wrote {len(df)} cross-score rows -> {args.out}")

    matrix_path = args.matrix_out or args.out.with_name(args.out.stem + "_matrix.csv")
    mat = rq.rq4_cross_scoring(args.out)
    mat.to_csv(matrix_path)
    print(f"Wrote cross-score matrix -> {matrix_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
