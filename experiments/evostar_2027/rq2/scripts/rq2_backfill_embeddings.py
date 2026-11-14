#!/usr/bin/env python3
"""Backfill MiniLM prompt embeddings for Global EA (no_speciation) runs.

Writes ``prompt_embeddings.npz`` beside each run's elites (ids + (n, 384)
L2-normalized vectors). Does not rewrite elites.json.

Example::

    PYTHONPATH=src:. venv/bin/python experiments/evostar_2027/rq2/scripts/rq2_backfill_embeddings.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "src"))

from rq2_metrics import (  # noqa: E402
    EMBEDDING_CACHE_NAME,
    load_genomes,
    save_embedding_cache,
)

LIVE = ROOT / "data" / "outputs" / "evostar2027"
DEFAULT_CELL = "no_speciation"
MODEL_NAME = "all-MiniLM-L6-v2"


def backfill_run(run_dir: Path, *, model_name: str, batch_size: int, force: bool) -> int:
    cache_path = run_dir / EMBEDDING_CACHE_NAME
    genomes = load_genomes(run_dir, "none")
    if not genomes:
        print(f"skip empty {run_dir}", flush=True)
        return 0

    need_ids: List[str] = []
    need_prompts: List[str] = []
    existing = {}
    if cache_path.is_file() and not force:
        data = np.load(cache_path, allow_pickle=False)
        ids = [str(x) for x in data["ids"].tolist()]
        embs = np.asarray(data["embeddings"], dtype=np.float32)
        existing = {i: embs[j] for j, i in enumerate(ids)}

    for g in genomes:
        rid = str(g.get("id"))
        e = g.get("prompt_embedding")
        if isinstance(e, list) and len(e) == 384:
            existing[rid] = np.asarray(e, dtype=np.float32)
            continue
        if rid in existing and not force:
            continue
        prompt = g.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            continue
        need_ids.append(rid)
        need_prompts.append(prompt)

    if need_prompts:
        from speciation.embeddings import get_embedding_model

        print(f"encode {run_dir.name}: {len(need_prompts)} prompts", flush=True)
        model = get_embedding_model(model_name=model_name)
        arr = model.encode(need_prompts, batch_size=batch_size, show_progress=True)
        arr = np.asarray(arr, dtype=np.float32)
        for rid, vec in zip(need_ids, arr):
            existing[rid] = vec
    elif cache_path.is_file() and not force:
        print(f"ok cached {run_dir.name}: {len(existing)}", flush=True)
        return len(existing)

    # Preserve genome order for determinism.
    ordered_ids = [str(g.get("id")) for g in genomes if str(g.get("id")) in existing]
    ordered = np.stack([existing[i] for i in ordered_ids], axis=0)
    save_embedding_cache(run_dir, ordered_ids, ordered)
    print(f"wrote {cache_path} n={len(ordered_ids)}", flush=True)
    return len(ordered_ids)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cell", default=DEFAULT_CELL, help="Folder under data/outputs/evostar2027")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--force", action="store_true", help="Recompute even if cache exists")
    ap.add_argument("--seeds", type=int, nargs="*", default=list(range(10)))
    args = ap.parse_args()

    cell = LIVE / args.cell
    if not cell.is_dir():
        raise SystemExit(f"missing cell dir: {cell}")
    total = 0
    for seed in args.seeds:
        run_dir = cell / f"rep{seed}_s{seed}"
        if not run_dir.is_dir():
            print(f"missing {run_dir}", flush=True)
            continue
        total += backfill_run(
            run_dir,
            model_name=args.model,
            batch_size=args.batch_size,
            force=args.force,
        )
    print(f"done total_cached_vectors≈{total}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
