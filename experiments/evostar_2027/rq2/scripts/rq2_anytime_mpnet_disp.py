#!/usr/bin/env python3
"""Anytime top-50 MPNet dispersion trajectories for RQ2 (no new EA runs).

For each of the 50 completed runs, reconstructs E_{r,50}(b) at evaluation
checkpoints, embeds the union of those elites once with all-mpnet-base-v2,
and stores mean pairwise (1-cos)/2 dispersion vs budget.

Outputs:
  experiments/evostar_2027/rq2/reports/rq2_anytime_mpnet_curves.npz
  experiments/evostar_2027/rq2/reports/rq2_anytime_mpnet_runs.csv
  per-run embedding caches under mpnet_disp_cache/anytime_<cell>_s<seed>.npz
"""

from __future__ import annotations

import csv
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
CACHE_DIR = REPORTS / "mpnet_disp_cache"
ENDPOINT_CSV = REPORTS / "rq2_mpnet_disp_runs.csv"
OUT_NPZ = REPORTS / "rq2_anytime_mpnet_curves.npz"
OUT_CSV = REPORTS / "rq2_anytime_mpnet_runs.csv"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
    ("none", "none"): "no_speciation",
}
CELLS: List[Tuple[str, str]] = list(LIVE_DIR.keys())
ELITE_K = 50
# Align with existing anytime fitness grid so Fig.1 panels share x-axis.
GRID = np.arange(ELITE_K, 2001, 25, dtype=int)
MPNET_NAME = "sentence-transformers/all-mpnet-base-v2"


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text(encoding="utf-8")).get("status")) == "complete"
    except Exception:
        return False


def mean_pairwise_disp(emb: np.ndarray) -> float:
    """Mean pairwise (1 - cos)/2 on L2-normalized rows."""
    X = np.asarray(emb, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        return float("nan")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.clip(norms, 1e-12, None)
    sims = X @ X.T
    n = sims.shape[0]
    tri = sims[np.triu_indices(n, k=1)]
    return float(np.mean((1.0 - tri) / 2.0))


def topk_membership_over_budget(
    genomes: Sequence[Dict[str, Any]],
    grid: np.ndarray,
    k: int = ELITE_K,
) -> Tuple[Dict[int, List[str]], Dict[str, str], Dict[str, float]]:
    """Return checkpoint -> top-k ids, id->prompt, id->fitness among moderated genomes."""
    best: List[Tuple[float, str]] = []
    prompts: Dict[str, str] = {}
    scores: Dict[str, float] = {}
    out: Dict[int, List[str]] = {}
    n = 0
    gi = 0
    for rec in genomes:
        s = illicit_of(rec)
        if s is None:
            continue
        n += 1
        rid = str(rec.get("id"))
        prompts[rid] = str(rec.get("prompt") or "")
        scores[rid] = float(s)
        best.append((float(s), rid))
        best.sort(key=lambda t: (-t[0], t[1]))
        if len(best) > k:
            best = best[:k]
        while gi < len(grid) and n >= int(grid[gi]):
            out[int(grid[gi])] = [rid for _, rid in best]
            gi += 1
    return out, prompts, scores


def load_endpoint_csv() -> Dict[Tuple[str, str, int], float]:
    if not ENDPOINT_CSV.is_file():
        return {}
    with ENDPOINT_CSV.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {
        (r["distance"], r["clustering"], int(r["seed"])): float(r["disp_mpnet_top50_b2000"])
        for r in rows
    }


def get_encoder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MPNET_NAME)


def embed_prompts(model, texts: Sequence[str]) -> np.ndarray:
    emb = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return np.asarray(emb, dtype=np.float32)


def process_run(
    model,
    distance: str,
    clustering: str,
    seed: int,
    *,
    force: bool = False,
) -> Tuple[np.ndarray, float, float]:
    cell = LIVE_DIR[(distance, clustering)]
    run_dir = LIVE / cell / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        raise RuntimeError(f"incomplete run {run_dir}")

    cache_path = CACHE_DIR / f"anytime_{cell}_s{seed}.npz"
    genomes = moderated_genomes(load_genomes(run_dir, clustering, attach_embeddings=False))
    membership, prompts, _scores = topk_membership_over_budget(genomes, GRID, ELITE_K)
    union_ids = sorted({rid for ids in membership.values() for rid in ids})
    if not union_ids:
        raise RuntimeError(f"empty elite union for {cell} seed{seed}")

    if cache_path.is_file() and not force:
        z = np.load(cache_path, allow_pickle=True)
        cached_ids = [str(x) for x in z["ids"].tolist()]
        cached_emb = np.asarray(z["embeddings"], dtype=np.float32)
        id_to_emb = {cached_ids[i]: cached_emb[i] for i in range(len(cached_ids))}
        missing = [rid for rid in union_ids if rid not in id_to_emb]
        if missing:
            texts = [prompts[rid] for rid in missing]
            new_emb = embed_prompts(model, texts)
            for i, rid in enumerate(missing):
                id_to_emb[rid] = new_emb[i]
            ids_out = sorted(id_to_emb)
            emb_out = np.stack([id_to_emb[i] for i in ids_out], axis=0)
            np.savez_compressed(cache_path, ids=np.asarray(ids_out), embeddings=emb_out)
        else:
            ids_out = cached_ids
            emb_out = cached_emb
            id_to_emb = {ids_out[i]: emb_out[i] for i in range(len(ids_out))}
    else:
        texts = [prompts[rid] for rid in union_ids]
        emb_out = embed_prompts(model, texts)
        ids_out = union_ids
        id_to_emb = {ids_out[i]: emb_out[i] for i in range(len(ids_out))}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ids=np.asarray(ids_out), embeddings=emb_out)

    curve = np.full(GRID.shape, np.nan, dtype=float)
    for j, b in enumerate(GRID):
        ids = membership.get(int(b))
        if not ids or len(ids) < 2:
            continue
        emb = np.stack([id_to_emb[rid] for rid in ids], axis=0)
        curve[j] = mean_pairwise_disp(emb)

    end_disp = float(curve[-1]) if np.isfinite(curve[-1]) else float("nan")
    # Also compute from endpoint-only cache if present (sanity).
    end_cache = CACHE_DIR / f"{cell}_s{seed}_top50.npz"
    end_cache_disp = float("nan")
    if end_cache.is_file():
        z = np.load(end_cache)
        end_cache_disp = mean_pairwise_disp(z["embeddings"])
    return curve, end_disp, end_cache_disp


def main() -> int:
    force = "--force" in sys.argv
    endpoint = load_endpoint_csv()
    print(f"loading encoder {MPNET_NAME}...", flush=True)
    model = get_encoder()

    arrays: Dict[str, np.ndarray] = {"grid": GRID.astype(float)}
    rows: List[Dict[str, Any]] = []
    mismatches = []

    for distance, clustering in CELLS:
        cell_curves = []
        for seed in range(10):
            print(f"anytime mpnet {distance} {clustering} seed{seed}", flush=True)
            curve, end_disp, end_cache_disp = process_run(
                model, distance, clustering, seed, force=force
            )
            cell_curves.append(curve)
            csv_disp = endpoint.get((distance, clustering, seed), float("nan"))
            row = {
                "distance": distance,
                "clustering": clustering,
                "seed": seed,
                "disp_mpnet_top50_b2000": end_disp,
                "endpoint_csv": csv_disp,
                "endpoint_cache": end_cache_disp,
                "abs_diff_csv": abs(end_disp - csv_disp) if np.isfinite(csv_disp) else float("nan"),
            }
            rows.append(row)
            if np.isfinite(csv_disp) and abs(end_disp - csv_disp) > 1e-3:
                mismatches.append(row)
            print(
                f"  D(2000)={end_disp:.6f} csv={csv_disp:.6f} cache={end_cache_disp:.6f}",
                flush=True,
            )
        arrays[f"{distance}__{clustering}"] = np.asarray(cell_curves, dtype=float)

    REPORTS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_NPZ, **arrays)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"wrote {OUT_NPZ}")
    print(f"wrote {OUT_CSV}")
    print(f"endpoint mismatches (>1e-3): {len(mismatches)}")
    for r in mismatches[:10]:
        print(
            f"  {r['distance']} {r['clustering']} s{r['seed']}: "
            f"new={r['disp_mpnet_top50_b2000']:.6f} csv={r['endpoint_csv']:.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
