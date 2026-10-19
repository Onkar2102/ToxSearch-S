#!/usr/bin/env python3
"""Part 1a: pool *_speciated genomes, dedupe by prompt, fill embeddings → rq1/p0.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1a_build_p0_pool.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils.evaluator_profiles import GOOGLE_PHENOTYPE_SCORE_ORDER

POP_FILES = ("elites.json", "archive.json", "reserves.json", "temp.json", "non_elites.json")
DEFAULT_SOURCE = Path("data/outputs/toxsearch_s_outputs copy")
DEFAULT_OUT = Path("experiments/evostar_2027/data/pool")
ANALYSIS_N = 2000


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file() or path.stat().st_size < 3:
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    if isinstance(data, dict):
        return [g for g in data.values() if isinstance(g, dict)]
    return []


def _fitness(g: Dict[str, Any]) -> float:
    for key in ("north_star_score", "fitness", "toxicity"):
        v = g.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                pass
    mr = g.get("moderation_result") or {}
    block = mr.get("google") if isinstance(mr, dict) else None
    scores = block.get("scores") if isinstance(block, dict) else None
    if isinstance(scores, dict) and scores.get("toxicity") is not None:
        return float(scores["toxicity"])
    return float("-inf")


def _google_scores(g: Dict[str, Any]) -> Optional[List[float]]:
    mr = g.get("moderation_result") or {}
    block = mr.get("google") if isinstance(mr, dict) else None
    scores = block.get("scores") if isinstance(block, dict) else None
    if not isinstance(scores, dict):
        return None
    try:
        return [float(scores[k]) for k in GOOGLE_PHENOTYPE_SCORE_ORDER]
    except (KeyError, TypeError, ValueError):
        return None


def _embedding(g: Dict[str, Any]) -> Optional[List[float]]:
    emb = g.get("prompt_embedding")
    if emb is None:
        emb = g.get("embedding")
    if emb is None or len(emb) == 0:
        return None
    return list(map(float, emb))


def discover_speciated_runs(source_root: Path) -> List[Path]:
    runs = sorted(
        p for p in source_root.iterdir()
        if p.is_dir() and p.name.endswith("_speciated") and not p.name.endswith("_comb")
    )
    if not runs:
        raise SystemExit(f"No *_speciated dirs under {source_root}")
    return runs


def pool_genomes(run_dirs: List[Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        for fname in POP_FILES:
            for g in _load_json_list(run_dir / fname):
                prompt = str(g.get("prompt") or "").strip()
                if not prompt:
                    continue
                rows.append({
                    "run": run_dir.name,
                    "source_file": fname,
                    "genome_id": g.get("id"),
                    "generation": int(g.get("generation") or 0),
                    "variant_type": str(g.get("variant_type") or ""),
                    "prompt": prompt,
                    "response": str(g.get("generated_output") or g.get("response") or ""),
                    "fitness": _fitness(g),
                    "emb": _embedding(g),
                    "scores_google": _google_scores(g),
                    "raw": g,
                })
    return rows


def dedupe_by_prompt(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep highest fitness; tie-break: run name, genome id (stable)."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r["prompt"]
        cur = best.get(key)
        if cur is None:
            best[key] = r
            continue
        gid = r["genome_id"]
        cur_gid = cur["genome_id"]
        try:
            gid_t = (int(gid) if gid is not None else -1,)
        except (TypeError, ValueError):
            gid_t = (str(gid),)
        try:
            cur_gid_t = (int(cur_gid) if cur_gid is not None else -1,)
        except (TypeError, ValueError):
            cur_gid_t = (str(cur_gid),)
        # higher fitness wins; ties → lexicographically smaller (run, genome_id)
        cand = (r["fitness"], r["run"], gid_t)
        prev = (cur["fitness"], cur["run"], cur_gid_t)
        # Prefer higher fitness; on equal fitness prefer smaller run/id for stability
        if cand[0] > prev[0] or (cand[0] == prev[0] and (cand[1], cand[2]) < (prev[1], prev[2])):
            best[key] = r
    return list(best.values())


def fill_embeddings(
    winners: List[Dict[str, Any]],
    *,
    model_name: str,
    batch_size: int,
) -> int:
    missing = [r for r in winners if r["emb"] is None]
    if not missing:
        return 0
    from speciation.embeddings import get_embedding_model

    model = get_embedding_model(model_name=model_name)
    prompts = [r["prompt"] for r in missing]
    print(f"Re-encoding {len(prompts)} missing embeddings with {model_name}...")
    embs = model.encode(prompts, batch_size=batch_size, show_progress=True)
    for r, e in zip(missing, embs):
        r["emb"] = list(map(float, e))
    return len(missing)


def to_dataframe(winners: List[Dict[str, Any]]) -> pd.DataFrame:
    records = []
    for r in winners:
        if r["emb"] is None:
            raise ValueError(f"missing embedding after fill: run={r['run']} id={r['genome_id']}")
        if r["scores_google"] is None:
            raise ValueError(f"missing google scores: run={r['run']} id={r['genome_id']}")
        records.append({
            "run": r["run"],
            "source_file": r["source_file"],
            "genome_id": r["genome_id"],
            "generation": r["generation"],
            "variant_type": r["variant_type"],
            "prompt": r["prompt"],
            "response": r["response"],
            "fitness": float(r["fitness"]),
            "emb": r["emb"],
            "scores_google": r["scores_google"],
            "scores_openai": None,
        })
    df = pd.DataFrame(records)
    # Sort for analysis index: fitness desc, then run, genome_id
    def _gid_key(x: Any) -> Tuple:
        try:
            return (0, int(x))
        except (TypeError, ValueError):
            return (1, str(x))

    df = df.sort_values(
        by=["fitness", "run", "genome_id"],
        ascending=[False, True, True],
        key=lambda col: col.map(_gid_key) if col.name == "genome_id" else col,
    ).reset_index(drop=True)
    return df


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1a: build P0 pool + analysis index")
    p.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--analysis-n", type=int, default=ANALYSIS_N)
    p.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--skip-encode", action="store_true", help="Fail if embeddings missing")
    args = p.parse_args()

    runs = discover_speciated_runs(args.source_root)
    print(f"Pooling {len(runs)} speciated runs from {args.source_root}")
    pooled = pool_genomes(runs)
    print(f"  raw genomes with prompt: {len(pooled)}")
    winners = dedupe_by_prompt(pooled)
    print(f"  unique prompts after dedupe: {len(winners)}")
    n_missing = sum(1 for r in winners if r["emb"] is None)
    print(f"  missing embeddings: {n_missing}")
    if n_missing and args.skip_encode:
        raise SystemExit("--skip-encode set but embeddings missing")
    if n_missing:
        fill_embeddings(winners, model_name=args.embedding_model, batch_size=args.batch_size)

    df = to_dataframe(winners)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    p0_path = out_dir / "p0_full.pkl"
    df.to_pickle(p0_path)

    n = min(args.analysis_n, len(df))
    analysis = df.iloc[:n].reset_index(drop=True)
    analysis_path = out_dir / f"p0_analysis_{n}.pkl"
    analysis.to_pickle(analysis_path)

    index = {
        "source_root": str(args.source_root),
        "runs": [r.name for r in runs],
        "n_pooled_raw": len(pooled),
        "n_deduped": len(df),
        "n_missing_embeddings_filled": int(n_missing),
        "embedding_model": args.embedding_model,
        "analysis_n": n,
        "selection": "top_N_by_fitness_desc_then_run_genome_id",
        "p0_full": str(p0_path),
        "p0_analysis": str(analysis_path),
        "analysis_index": [
            {
                "i": int(i),
                "run": row.run,
                "genome_id": row.genome_id,
                "fitness": float(row.fitness),
            }
            for i, row in analysis.iterrows()
        ],
        "fitness_stats": {
            "min": float(df["fitness"].min()),
            "median": float(df["fitness"].median()),
            "max": float(df["fitness"].max()),
            "analysis_min": float(analysis["fitness"].min()),
            "analysis_max": float(analysis["fitness"].max()),
        },
    }
    index_path = out_dir / "index.json"
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"Wrote {p0_path} ({len(df)} rows)")
    print(f"Wrote {analysis_path} ({n} rows)")
    print(f"Wrote {index_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
