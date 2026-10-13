"""Stage C/D analysis CLI.

    PYTHONPATH=src:. python experiments/stage_analysis/run_all.py \
        --roots data/outputs/stagec --out results/stagec [--forming] [--topics] [--min-seeds 5]

Writes runs.csv, design_check.csv, run_metrics.csv, and rq1_*/rq3_*/rq4_* tables.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.stage_analysis import rq
from experiments.stage_analysis.metrics import load_final_population, run_metrics, topic_coverage
from experiments.stage_analysis.runs import check_design, index_runs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--roots", nargs="+", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--min-seeds", type=int, default=5)
    p.add_argument("--forming", action="store_true", help="Also compute niche geometry under the forming d (slow for d_N)")
    p.add_argument("--topics", action="store_true", help="Pooled BERTopic coverage (N1, K_topic)")
    p.add_argument("--include-incomplete", action="store_true")
    p.add_argument("--topk", type=int, default=50, help="Top-k prompts per run exported for cross-scoring")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    runs = index_runs(args.roots)
    if runs.empty:
        print("No runs found.")
        return 1
    runs.to_csv(args.out / "runs.csv", index=False)
    design = check_design(runs, args.min_seeds)
    design.to_csv(args.out / "design_check.csv", index=False)
    print(design.to_string(index=False))
    if not args.include_incomplete:
        runs = runs[runs.status == "complete"]

    metrics = pd.DataFrame([run_metrics(r, forming=args.forming) for r in runs.to_dict("records")])
    df = runs.merge(metrics, on="run_dir", how="left")

    if args.topics:
        prompts, embs = {}, {}
        for r in runs.itertuples(index=False):
            pop = [g for g in load_final_population(Path(r.run_dir), r.clustering_method) if g.get("prompt_embedding") is not None]
            prompts[r.run_dir] = [g["prompt"] for g in pop]
            embs[r.run_dir] = [g["prompt_embedding"] for g in pop]
        cov = topic_coverage(prompts, embs)
        df["N1"] = df.run_dir.map(lambda k: cov.get(k, {}).get("N1"))
        df["K_topic"] = df.run_dir.map(lambda k: cov.get(k, {}).get("K_topic"))

    df.to_csv(args.out / "run_metrics.csv", index=False)
    rq.write_tables(rq.rq1_rq2(df), args.out, "rq1")
    rq.write_tables(rq.rq3(df), args.out, "rq3")
    rq.write_tables({"rank_agreement": rq.rq4_rank_agreement(df), "sign_agreement": rq.rq4_sign_agreement(df)}, args.out, "rq4")
    rq.export_topk_for_cross_scoring(df, args.topk, args.out / "rq4_topk_for_cross_scoring.csv")
    print(f"Wrote tables to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
