#!/usr/bin/env python3
"""Fill wall_seconds into NLI progress/meta when missing (e.g. pre-instrumentation runs).

Estimates from progress pairs_per_sec in the log when available; otherwise from
pairs_total / typical rate. Safe to re-run; does not overwrite existing wall_seconds
unless --force.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

FLAT_DEFAULT = Path("experiments/evostar_2027/data/mats/_flat")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _rate_from_log(log_path: Path, label: str) -> Optional[float]:
    if not log_path.is_file():
        return None
    pat = re.compile(
        rf"\s*{re.escape(label)} checkpoint row .* rate=([0-9.]+)/s"
    )
    last = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if m:
            last = float(m.group(1))
    return last


def backfill_label(
    flat: Path,
    n: int,
    label: str,
    *,
    force: bool = False,
    log_path: Optional[Path] = None,
) -> Optional[Dict[str, Any]]:
    prog_path = flat / f"{label}_{n}.progress.json"
    prog = _load(prog_path)
    if not prog.get("complete"):
        return None
    if prog.get("wall_seconds") is not None and not force:
        return prog
    pairs = int(prog.get("pairs_total") or n * (n - 1) // 2)
    rate = prog.get("pairs_per_sec")
    if rate is None and log_path is not None:
        rate = _rate_from_log(log_path, label)
    if rate is None or float(rate) <= 0:
        rate = 100.0  # conservative fallback
    wall = pairs / float(rate)
    prog.update(
        {
            "wall_seconds": round(wall, 3),
            "wall_hours": round(wall / 3600.0, 4),
            "pairs_done": pairs,
            "pairs_total": pairs,
            "pairs_per_sec": round(float(rate), 2),
            "updated_at": _utc_now(),
            "timing_note": "Backfilled by backfill_nli_timing.py from pairs/rate.",
            "complete": True,
            "next_i": n,
            "n": n,
            "label": label,
        }
    )
    prog_path.write_text(json.dumps(prog, indent=2), encoding="utf-8")
    print(f"  {label}: wall={prog['wall_hours']:.3f}h rate={rate}/s")
    return prog


def rewrite_meta(flat: Path, n: int) -> None:
    timing: Dict[str, Any] = {}
    total = 0.0
    for label in ("d_N", "d_negE"):
        prog = _load(flat / f"{label}_{n}.progress.json")
        if not prog.get("complete") or prog.get("wall_seconds") is None:
            continue
        timing[label] = {
            "wall_seconds": prog["wall_seconds"],
            "wall_hours": prog["wall_hours"],
            "pairs_per_sec": prog.get("pairs_per_sec"),
            "pairs_total": prog.get("pairs_total"),
            "complete": True,
            "source": "progress.json",
        }
        total += float(prog["wall_seconds"])
    meta_path = flat / f"nli_mats_meta_{n}.json"
    meta = _load(meta_path)
    meta.update(
        {
            "N": n,
            "timing": timing,
            "timing_total_wall_seconds": round(total, 3) if timing else None,
            "timing_total_wall_hours": round(total / 3600.0, 4) if timing else None,
            "updated_at": _utc_now(),
            "files": {
                "d_N": str(flat / f"d_N_{n}.npy"),
                "d_negE": str(flat / f"d_negE_{n}.npy"),
            },
        }
    )
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path} total_h={meta.get('timing_total_wall_hours')}")


def main() -> int:
    p = argparse.ArgumentParser(description="Backfill NLI wall timing into progress/meta")
    p.add_argument("--mats-dir", type=Path, default=FLAT_DEFAULT)
    p.add_argument("--max-n", type=int, default=2000)
    p.add_argument("--force", action="store_true")
    p.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Optional p1c log for rate extraction (default: mats-dir/p1c_n{N}_resume.log)",
    )
    args = p.parse_args()
    flat = args.mats_dir
    log_path = args.log or flat / f"p1c_n{args.max_n}_resume.log"
    if not log_path.is_file():
        log_path = flat / f"p1c_n{args.max_n}.log"
    print(f"Backfill N={args.max_n} under {flat} (log={log_path})")
    for label in ("d_N", "d_negE"):
        backfill_label(flat, args.max_n, label, force=args.force, log_path=log_path)
    rewrite_meta(flat, args.max_n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
