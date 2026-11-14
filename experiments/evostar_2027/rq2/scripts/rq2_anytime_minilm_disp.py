#!/usr/bin/env python3
"""Anytime top-50 MiniLM dispersion trajectories (sensitivity; uses stored embeddings)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import illicit_of, load_genomes, moderated_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
OUT_NPZ = REPORTS / "rq2_anytime_minilm_curves.npz"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
    ("none", "none"): "no_speciation",
}
ELITE_K = 50
GRID = np.arange(ELITE_K, 2001, 25, dtype=int)


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("status")) == "complete"
    except Exception:
        return False


def mean_pairwise_disp(emb: np.ndarray) -> float:
    X = np.asarray(emb, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
    sims = X @ X.T
    tri = sims[np.triu_indices(sims.shape[0], k=1)]
    return float(np.mean((1.0 - tri) / 2.0))


def embedding_of(rec: Dict[str, Any]) -> np.ndarray | None:
    e = rec.get("prompt_embedding")
    if isinstance(e, list) and len(e) == 384:
        return np.asarray(e, dtype=np.float32)
    if isinstance(e, np.ndarray) and e.shape == (384,):
        return e.astype(np.float32)
    return None


def curve_for_run(genomes: Sequence[Dict[str, Any]]) -> np.ndarray:
    best: List[Tuple[float, str]] = []
    emb_by_id: Dict[str, np.ndarray] = {}
    out = np.full(GRID.shape, np.nan, dtype=float)
    n = 0
    gi = 0
    for rec in genomes:
        s = illicit_of(rec)
        if s is None:
            continue
        vec = embedding_of(rec)
        if vec is None:
            continue
        n += 1
        rid = str(rec.get("id"))
        emb_by_id[rid] = vec
        best.append((float(s), rid))
        best.sort(key=lambda t: (-t[0], t[1]))
        if len(best) > ELITE_K:
            best = best[:ELITE_K]
        while gi < len(GRID) and n >= int(GRID[gi]):
            ids = [rid for _, rid in best]
            if len(ids) >= 2:
                out[gi] = mean_pairwise_disp(np.stack([emb_by_id[i] for i in ids], axis=0))
            gi += 1
    return out


def main() -> int:
    arrays: Dict[str, np.ndarray] = {"grid": GRID.astype(float)}
    for distance, clustering in LIVE_DIR:
        cell = LIVE_DIR[(distance, clustering)]
        curves = []
        for seed in range(10):
            run = LIVE / cell / f"rep{seed}_s{seed}"
            if not complete(run):
                raise RuntimeError(f"incomplete {run}")
            gens = moderated_genomes(load_genomes(run, clustering, attach_embeddings=True))
            curve = curve_for_run(gens)
            curves.append(curve)
            print(
                f"minilm disp {distance} {clustering} seed{seed} D2000={curve[-1]:.4f}",
                flush=True,
            )
        arrays[f"{distance}__{clustering}"] = np.asarray(curves, dtype=float)
    REPORTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    print(f"wrote {OUT_NPZ}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
