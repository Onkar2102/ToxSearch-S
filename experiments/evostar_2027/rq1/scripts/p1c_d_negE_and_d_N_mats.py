#!/usr/bin/env python3
"""Part 1c: checkpointed d_N and d_¬E (nli / nli_entail) on the analysis index.

Currently writes only final distances:

- ``d_N`` = symmetrized Qiu Eq. 9: ``P̄_C + ½ P̄_N``
- ``d_negE`` = ``1 − P̄_E``

Follow-up (not implemented here — avoid a full NLI recompute unless needed):
checkpoint raw class probabilities ``P_E / P_N / P_C`` (or a clustering-oriented
``d_biE`` from Kuhn-style bidirectional entailment / soft
``1 − min(P_E^{ab}, P_E^{ba})``). Topic clustering wants meaning-equivalence,
not UQ-style continuous semantic density; without class probs we cannot build
that third formula from existing ``d_N`` / ``d_negE`` alone.

Smoke (recommended first)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py \\
      --max-n 50 --which both

Full N=2000 (long; resumes from checkpoint)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py \\
      --max-n 2000 --which both --checkpoint-every 10 --batch-size 256

Then slice B1000 from the N=2000 mats (no NLI at N=3139)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1a2_build_batches.py \\
      --skip-cheap-mats --slice-nli-only

Artifacts land under experiments/evostar_2027/rq1/mats/:
  d_N_{N}.npy + d_N_{N}.progress.json
  d_negE_{N}.npy + d_negE_{N}.progress.json
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

from experiments.stage_analysis import stage_a

DEFAULT_P0 = Path("experiments/evostar_2027/rq1/p0")
DEFAULT_MATS = Path("experiments/evostar_2027/rq1/mats")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_progress(progress_path: Path) -> Dict[str, Any]:
    if not progress_path.is_file():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _timing_payload(
    *,
    n: int,
    label: str,
    next_i: int,
    prior_wall: float,
    session_wall: float,
    pairs_done: int,
    pairs_total: int,
    complete: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    wall = float(prior_wall) + float(session_wall)
    rate = (pairs_done / wall) if wall > 0 else 0.0
    payload: Dict[str, Any] = {
        "next_i": int(next_i),
        "n": int(n),
        "label": label,
        "wall_seconds": round(wall, 3),
        "wall_hours": round(wall / 3600.0, 4),
        "session_wall_seconds": round(float(session_wall), 3),
        "prior_wall_seconds": round(float(prior_wall), 3),
        "pairs_done": int(pairs_done),
        "pairs_total": int(pairs_total),
        "pairs_per_sec": round(rate, 2),
        "updated_at": _utc_now(),
    }
    if complete:
        payload["complete"] = True
        payload["finished_at"] = _utc_now()
    if extra:
        payload.update(extra)
    return payload


def pairwise_nli_checkpointed(
    texts: Sequence[str],
    out_path: Path,
    *,
    row_fn: Callable[..., np.ndarray],
    checkpoint_every: int = 10,
    batch_size: int = 256,
    label: str = "d_N",
) -> np.ndarray:
    """Fill symmetric matrix row-by-row with resume support."""
    from speciation.distance_metrics.nli import get_nli_stats

    n = len(texts)
    if out_path.is_file():
        dist = np.load(out_path)
        if dist.shape != (n, n):
            raise ValueError(f"Checkpoint shape {dist.shape} != ({n}, {n})")
        print(f"  resumed {label} from {out_path}")
    else:
        dist = np.zeros((n, n), dtype=np.float64)

    progress_path = out_path.with_suffix(".progress.json")
    prev = _read_progress(progress_path)
    start_i = int(prev.get("next_i", 0))
    # Accumulated wall across prior resume sessions (do not reset on resume).
    prior_wall = float(prev.get("wall_seconds", 0.0) or 0.0)

    t0 = time.time()
    pairs_at_start = sum(n - 1 - r for r in range(start_i))
    pairs_total = n * (n - 1) // 2
    for i in range(start_i, n):
        others = [texts[j] for j in range(i + 1, n)]
        if others:
            row = row_fn(texts[i], others, batch_size=batch_size)
            for k, j in enumerate(range(i + 1, n)):
                dist[i, j] = row[k]
                dist[j, i] = row[k]
        if (i + 1) % 5 == 0:
            try:
                import torch
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
            except Exception:
                pass
        if (i + 1) % checkpoint_every == 0 or i + 1 == n:
            np.save(out_path, dist)
            session_wall = time.time() - t0
            pairs_done = sum(n - 1 - r for r in range(i + 1))
            pairs_this = pairs_done - pairs_at_start
            rate = pairs_this / session_wall if session_wall > 0 else 0.0
            eta = (pairs_total - pairs_done) / rate if rate > 0 else float("inf")
            payload = _timing_payload(
                n=n,
                label=label,
                next_i=i + 1,
                prior_wall=prior_wall,
                session_wall=session_wall,
                pairs_done=pairs_done,
                pairs_total=pairs_total,
                complete=(i + 1 == n),
                extra={"batch_size": int(batch_size)},
            )
            progress_path.write_text(json.dumps(payload), encoding="utf-8")
            stats = get_nli_stats()
            print(
                f"  {label} checkpoint row {i + 1}/{n} pairs={pairs_done}/{pairs_total} "
                f"({100 * pairs_done / pairs_total:.1f}%) "
                f"rate={rate:.1f}/s ETA={eta / 3600:.1f}h "
                f"wall={payload['wall_hours']:.2f}h "
                f"model_passes={stats.get('model_passes')}",
                flush=True,
            )
    return dist


def _already_complete(path: Path, n: int) -> bool:
    prog = path.with_suffix(".progress.json")
    if not path.is_file() or not prog.is_file():
        return False
    meta = json.loads(prog.read_text(encoding="utf-8"))
    return bool(meta.get("complete")) and int(meta.get("n", -1)) == n


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1c: d_N + d_¬E mats (checkpointed)")
    p.add_argument("--p0-dir", type=Path, default=DEFAULT_P0)
    p.add_argument("--mats-dir", type=Path, default=DEFAULT_MATS)
    p.add_argument(
        "--max-n",
        type=int,
        default=2000,
        help="Leading prefix of fitness-sorted pool (default 2000)",
    )
    p.add_argument(
        "--allow-large-n",
        action="store_true",
        help="Allow --max-n > 2000 (full pool). Long/expensive; requires checkpoints.",
    )
    p.add_argument(
        "--p0-pkl",
        type=Path,
        default=None,
        help="Override P0 pickle (default: p0_full.pkl if present else index p0_analysis)",
    )
    p.add_argument(
        "--which",
        choices=["both", "d_N", "d_negE"],
        default="both",
        help="Which matrices to (re)compute",
    )
    p.add_argument("--checkpoint-every", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--nli-model", type=str, default=None)
    p.add_argument("--force", action="store_true", help="Ignore complete checkpoints")
    args = p.parse_args()

    if args.max_n > 2000 and not args.allow_large_n:
        raise SystemExit(
            "Plan default cap: --max-n ≤ 2000. Pass --allow-large-n for full-pool NLI."
        )

    p0_dir = args.p0_dir
    if args.p0_pkl is not None:
        pkl = args.p0_pkl
    elif (p0_dir / "p0_full.pkl").is_file():
        pkl = p0_dir / "p0_full.pkl"
    else:
        index_path = p0_dir / "index.json"
        if not index_path.is_file():
            raise SystemExit(f"Missing {index_path}; run p1a first")
        index = json.loads(index_path.read_text(encoding="utf-8"))
        pkl = Path(index["p0_analysis"])
    df = stage_a.load_p0(str(pkl), require_openai=False)
    n = min(args.max_n, len(df))
    texts = df.iloc[:n]["prompt"].tolist()
    mats_dir = args.mats_dir
    mats_dir.mkdir(parents=True, exist_ok=True)

    if args.nli_model:
        from speciation.distance_metrics.nli import set_nli_model_name
        set_nli_model_name(args.nli_model)

    from speciation.distance_metrics.nli import (
        nli_distances_from_query_torch,
        nli_entail_distances_from_query_torch,
    )

    jobs = []
    if args.which in ("both", "d_N"):
        jobs.append(("d_N", mats_dir / f"d_N_{n}.npy", nli_distances_from_query_torch))
    if args.which in ("both", "d_negE"):
        jobs.append(("d_negE", mats_dir / f"d_negE_{n}.npy", nli_entail_distances_from_query_torch))

    print(f"NLI mats for N={n} → {mats_dir}")
    print(
        "Resume tip: re-run the same command; incomplete .npy/.progress.json continue "
        "from next_i. For full analysis index use --max-n 2000."
    )
    timing: Dict[str, Any] = {}
    for label, path, fn in jobs:
        if not args.force and _already_complete(path, n):
            print(f"  skip {label}: already complete at {path}")
            prog = _read_progress(path.with_suffix(".progress.json"))
            if prog:
                timing[label] = {
                    "wall_seconds": prog.get("wall_seconds"),
                    "wall_hours": prog.get("wall_hours"),
                    "pairs_per_sec": prog.get("pairs_per_sec"),
                    "pairs_total": prog.get("pairs_total"),
                    "complete": True,
                    "source": "progress.json",
                }
            continue
        print(f"  computing {label} → {path}")
        pairwise_nli_checkpointed(
            texts,
            path,
            row_fn=fn,
            checkpoint_every=args.checkpoint_every,
            batch_size=args.batch_size,
            label=label,
        )
        prog = _read_progress(path.with_suffix(".progress.json"))
        timing[label] = {
            "wall_seconds": prog.get("wall_seconds"),
            "wall_hours": prog.get("wall_hours"),
            "pairs_per_sec": prog.get("pairs_per_sec"),
            "pairs_total": prog.get("pairs_total"),
            "complete": bool(prog.get("complete")),
            "source": "progress.json",
        }

    total_wall = sum(
        float(v["wall_seconds"])
        for v in timing.values()
        if v.get("wall_seconds") is not None
    )
    meta = {
        "N": n,
        "p0_analysis": str(pkl),
        "which": args.which,
        "batch_size": int(args.batch_size),
        "checkpoint_every": int(args.checkpoint_every),
        "nli_model": args.nli_model,
        "files": {
            "d_N": str(mats_dir / f"d_N_{n}.npy"),
            "d_negE": str(mats_dir / f"d_negE_{n}.npy"),
        },
        "timing": timing,
        "timing_total_wall_seconds": round(total_wall, 3) if timing else None,
        "timing_total_wall_hours": round(total_wall / 3600.0, 4) if timing else None,
        "updated_at": _utc_now(),
        "resume_full_n2000": (
            "PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/"
            "p1c_d_negE_and_d_N_mats.py --max-n 2000 --which both "
            "--checkpoint-every 10 --batch-size 256"
        ),
    }
    (mats_dir / f"nli_mats_meta_{n}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(
        f"Wrote {mats_dir / f'nli_mats_meta_{n}.json'} "
        f"(total NLI wall ≈ {meta.get('timing_total_wall_hours')} h)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
