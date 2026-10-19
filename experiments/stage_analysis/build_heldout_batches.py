#!/usr/bin/env python3
"""Build held-out P0 prefixes from a completed run (RQ1 Stage A).

Example::

    PYTHONPATH=src:. python experiments/stage_analysis/build_heldout_batches.py \\
      --run-dir data/outputs/20260211_2122 --max-n 2000 --out-dir results/rq1_heldout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd

from utils.evaluator_profiles import GOOGLE_PHENOTYPE_SCORE_ORDER

POP_FILES = ("elites.json", "archive.json", "temp.json", "reserves.json")
DEFAULT_BATCHES = (1000, 2000)


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size < 3:
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [g for g in data if g]
    if isinstance(data, dict):
        return [g for g in data.values() if isinstance(g, dict)]
    return []


def load_genomes_by_id(run_dir: Path) -> List[Dict[str, Any]]:
    """Deduped genomes from population files, sorted by ascending genome id."""
    by_id: Dict[int, Dict[str, Any]] = {}
    for fname in POP_FILES:
        for g in _load_json_list(run_dir / fname):
            gid = g.get("id")
            if gid is None:
                continue
            try:
                iid = int(gid)
            except (TypeError, ValueError):
                continue
            if iid not in by_id:
                by_id[iid] = g
    return [by_id[k] for k in sorted(by_id)]


def _google_scores(g: Dict[str, Any]) -> List[float]:
    mr = g.get("moderation_result") or {}
    block = mr.get("google") if isinstance(mr, dict) else None
    scores = block.get("scores") if isinstance(block, dict) else None
    if not isinstance(scores, dict):
        raise ValueError(f"genome {g.get('id')}: missing google scores")
    missing = [k for k in GOOGLE_PHENOTYPE_SCORE_ORDER if k not in scores or scores[k] is None]
    if missing:
        raise ValueError(f"genome {g.get('id')}: missing phenotype keys {missing}")
    return [float(scores[k]) for k in GOOGLE_PHENOTYPE_SCORE_ORDER]


def _embedding(g: Dict[str, Any]) -> List[float]:
    emb = g.get("prompt_embedding")
    if emb is None:
        emb = g.get("embedding")
    if emb is None:
        raise ValueError(f"genome {g.get('id')}: missing embedding")
    return list(map(float, emb))


def genomes_to_p0(genomes: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for g in genomes:
        rows.append({
            "genome_id": int(g["id"]),
            "generation": int(g.get("generation") or 0),
            "variant_type": str(g.get("variant_type") or ""),
            "prompt": str(g.get("prompt") or ""),
            "response": str(g.get("generated_output") or g.get("response") or ""),
            "emb": _embedding(g),
            "scores_google": _google_scores(g),
            "scores_openai": None,
        })
    return pd.DataFrame(rows)


def batch_meta(df: pd.DataFrame, n: int) -> Dict[str, Any]:
    chunk = df.iloc[:n]
    return {
        "batch_id": f"B{n}",
        "N": int(n),
        "max_id": int(chunk["genome_id"].iloc[-1]),
        "max_generation": int(chunk["generation"].max()),
        "n_unique_prompts": int(chunk["prompt"].nunique()),
        "n_initial": int((chunk["variant_type"] == "initial").sum()),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Build RQ1 held-out P0 prefixes from a run")
    p.add_argument("--run-dir", type=Path, default=Path("data/outputs/20260211_2122"))
    p.add_argument("--max-n", type=int, default=2000, help="Largest prefix to materialize")
    p.add_argument("--batches", type=int, nargs="+", default=list(DEFAULT_BATCHES))
    p.add_argument("--out-dir", type=Path, default=Path("results/rq1_heldout"))
    args = p.parse_args()

    genomes = load_genomes_by_id(args.run_dir)
    if len(genomes) < args.max_n:
        raise SystemExit(f"Need >= {args.max_n} genomes, found {len(genomes)} in {args.run_dir}")

    for n in args.batches:
        if n > args.max_n:
            raise SystemExit(f"batch size {n} exceeds --max-n {args.max_n}")

    df = genomes_to_p0(genomes[: args.max_n])
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p0_path = out_dir / f"p0_{args.max_n}.pkl"
    df.to_pickle(p0_path)

    index = {
        "run_dir": str(args.run_dir),
        "max_n": args.max_n,
        "n_genomes_available": len(genomes),
        "p0_path": str(p0_path),
        "batches": [batch_meta(df, n) for n in sorted(set(args.batches))],
    }
    index_path = out_dir / "batch_index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {p0_path} ({len(df)} rows)")
    print(f"Wrote {index_path}")
    for b in index["batches"]:
        print(f"  {b['batch_id']}: N={b['N']} max_id={b['max_id']} "
              f"unique_prompts={b['n_unique_prompts']} n_initial={b['n_initial']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
