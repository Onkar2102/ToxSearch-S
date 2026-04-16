#!/usr/bin/env python3
"""
C3 bridge: compare ToxSearch-S species structure (EvolutionTracker) to RainbowPlus archive cells (JSONL).

Reads the last generation's speciation block from EvolutionTracker.json and optionally counts
unique (category, style) keys from a RainbowPlus all_genomes.jsonl. Writes a one-row CSV for RQ tables.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from rainbowplus_io import archive_cell_keys, iter_jsonl


def _last_generation(tracker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gens = tracker.get("generations") or []
    if not gens:
        return None
    return max(gens, key=lambda g: g.get("generation_number", 0))


FIELDNAMES = [
    "tracker_path",
    "last_generation_number",
    "species_count",
    "active_species_count",
    "rainbow_jsonl",
    "rainbow_unique_cells",
    "rainbow_rows",
]


def row_from_tracker(tracker_path: Path, rainbow_jsonl: Optional[Path]) -> Dict[str, Any]:
    with tracker_path.open("r", encoding="utf-8") as f:
        tracker = json.load(f)
    last = _last_generation(tracker)
    spec = (last or {}).get("speciation") or {}
    species_count = spec.get("species_count")
    active = spec.get("active_species_count")
    gen_no = (last or {}).get("generation_number")

    rb_cells = None
    rb_rows = None
    rj = rainbow_jsonl
    if rj and rj.exists():
        cells = archive_cell_keys(rj)
        rb_cells = len(cells)
        rb_rows = sum(1 for _ in iter_jsonl(rj))

    return {
        "tracker_path": str(tracker_path),
        "last_generation_number": gen_no,
        "species_count": species_count,
        "active_species_count": active,
        "rainbow_jsonl": str(rj) if rj else "",
        "rainbow_unique_cells": rb_cells if rb_cells is not None else "",
        "rainbow_rows": rb_rows if rb_rows is not None else "",
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker", type=Path, default=None, help="Path to EvolutionTracker.json (single-run mode)")
    p.add_argument("--rainbow-jsonl", type=Path, default=None, help="Optional RainbowPlus JSONL (e.g. all_genomes.jsonl)")
    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: cwd bridge.csv)")
    p.add_argument(
        "--batch-csv",
        type=Path,
        default=None,
        help="CSV with columns tracker_path,rainbow_jsonl (optional) — one bridge row per line; writes --out-csv",
    )
    args = p.parse_args()

    if args.batch_csv:
        out = args.out_csv or Path("c3_species_bridge_batch.csv")
        out.parent.mkdir(parents=True, exist_ok=True)
        with args.batch_csv.open("r", encoding="utf-8", newline="") as bf:
            reader = csv.DictReader(bf)
            rows_out: List[Dict[str, Any]] = []
            for line in reader:
                tp = (line.get("tracker_path") or "").strip()
                rj = (line.get("rainbow_jsonl") or "").strip()
                if not tp:
                    continue
                tpath = Path(tp)
                rpath = Path(rj) if rj else None
                rows_out.append(row_from_tracker(tpath, rpath))
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDNAMES)
            w.writeheader()
            for row in rows_out:
                w.writerow(row)
        print(f"Wrote {out} ({len(rows_out)} rows)")
        return

    if not args.tracker:
        p.error("Provide --tracker or --batch-csv")

    out = args.out_csv or Path("c3_species_bridge_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    row = row_from_tracker(args.tracker, args.rainbow_jsonl)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        w.writeheader()
        w.writerow(row)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
