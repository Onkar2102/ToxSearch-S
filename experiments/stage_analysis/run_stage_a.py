#!/usr/bin/env python3
"""Stage A CLI: build P0 and/or write intrinsic agreement tables.

Examples::

    PYTHONPATH=src:. python experiments/stage_analysis/run_stage_a.py \\
      --build-p0 --prompts data/prompt_heldout.csv --out results/stagea/p0.parquet

    PYTHONPATH=src:. python experiments/stage_analysis/run_stage_a.py \\
      --p0 results/stagea/p0.parquet --out-dir results/stagea
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.stage_analysis import stage_a


def main() -> int:
    p = argparse.ArgumentParser(description="Stage A: P0 build + intrinsic distance tables")
    p.add_argument("--build-p0", action="store_true", help="Generate responses, dual-oracle scores, embeddings")
    p.add_argument("--prompts", type=Path, help="CSV with a questions column (required with --build-p0)")
    p.add_argument("--prompt-column", default="questions")
    p.add_argument("--out", type=Path, help="P0 output path (.parquet/.pkl/.json)")
    p.add_argument("--p0", type=Path, help="Existing P0 table to analyse")
    p.add_argument("--out-dir", type=Path, default=Path("results/stagea"))
    p.add_argument("--rg-config", default="config/RGConfig.yaml")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    p.add_argument("--openai-model", default="omni-moderation-latest")
    p.add_argument("--reference-radius", type=float, default=0.25)
    p.add_argument("--no-nli", action="store_true", help="Skip d_N matrices (faster smoke)")
    p.add_argument("--knn-k", type=int, default=10)
    args = p.parse_args()

    p0_path = args.p0
    if args.build_p0:
        if args.prompts is None or args.out is None:
            p.error("--build-p0 requires --prompts and --out")
        prompts = stage_a.load_prompts_csv(str(args.prompts), args.prompt_column)
        stage_a.build_p0(
            prompts,
            str(args.out),
            rg_config=args.rg_config,
            seed=args.seed,
            embedding_model=args.embedding_model,
            openai_model=args.openai_model,
        )
        print(f"Wrote P0 ({len(prompts)} prompts) -> {args.out}")
        p0_path = args.out

    if p0_path is None:
        p.error("Provide --p0 or --build-p0")

    df = stage_a.load_p0(str(p0_path))
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mats = stage_a.distance_matrices(df, nli=not args.no_nli)
    spearman = stage_a.spearman_table(mats)
    knn = stage_a.knn_table(mats, k=args.knn_k)
    quantiles = stage_a.quantile_table(mats)
    calib = stage_a.calibration_table(mats, reference="d_E", reference_radius=args.reference_radius)
    shares = {
        "google": stage_a.output_term_share(mats, "google"),
        "openai": stage_a.output_term_share(mats, "openai"),
    }
    agree = stage_a.oracle_agreement(df)

    spearman.to_csv(out_dir / "spearman.csv")
    knn.to_csv(out_dir / "knn_jaccard.csv")
    quantiles.to_csv(out_dir / "quantiles.csv")
    calib.to_csv(out_dir / "calibration.csv", index=False)
    (out_dir / "output_term_share.json").write_text(json.dumps(shares, indent=2), encoding="utf-8")
    (out_dir / "oracle_agreement.json").write_text(json.dumps(agree, indent=2), encoding="utf-8")
    print(f"Wrote Stage A tables -> {out_dir}")
    print("calibration:\n", calib.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
