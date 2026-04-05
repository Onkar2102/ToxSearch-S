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
from typing import Any, Dict

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from rainbowplus_io import archive_cell_keys, iter_jsonl


def _last_generation(tracker: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gens = tracker.get("generations") or []
    if not gens:
        return None
    return max(gens, key=lambda g: g.get("generation_number", 0))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tracker", type=Path, required=True, help="Path to EvolutionTracker.json")
    p.add_argument("--rainbow-jsonl", type=Path, default=None, help="Optional RainbowPlus JSONL (e.g. all_genomes.jsonl)")
    p.add_argument("--out-csv", type=Path, default=None, help="Output CSV path (default: stdout row to cwd bridge.csv)")
    args = p.parse_args()

    with args.tracker.open("r", encoding="utf-8") as f:
        tracker = json.load(f)
    last = _last_generation(tracker)
    spec = (last or {}).get("speciation") or {}
    species_count = spec.get("species_count")
    active = spec.get("active_species_count")
    gen_no = (last or {}).get("generation_number")

    rb_cells = None
    rb_rows = None
    if args.rainbow_jsonl and args.rainbow_jsonl.exists():
        cells = archive_cell_keys(args.rainbow_jsonl)
        rb_cells = len(cells)
        rb_rows = sum(1 for _ in iter_jsonl(args.rainbow_jsonl))

    out = args.out_csv or Path("c3_species_bridge_summary.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "tracker_path",
                "last_generation_number",
                "species_count",
                "active_species_count",
                "rainbow_jsonl",
                "rainbow_unique_cells",
                "rainbow_rows",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "tracker_path": str(args.tracker),
                "last_generation_number": gen_no,
                "species_count": species_count,
                "active_species_count": active,
                "rainbow_jsonl": str(args.rainbow_jsonl) if args.rainbow_jsonl else "",
                "rainbow_unique_cells": rb_cells if rb_cells is not None else "",
                "rainbow_rows": rb_rows if rb_rows is not None else "",
            }
        )
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
