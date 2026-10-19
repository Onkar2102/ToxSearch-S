#!/usr/bin/env python3
"""Part 1e: write immutable frozen_config.json for Part 2 pilots.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1e_freeze_config.py
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SWEEPS = Path("experiments/evostar_2027/rq1/sweeps")
DEFAULT_OUT = Path("experiments/evostar_2027/rq1/frozen_config.json")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 p1e: freeze Part 1 geometry/shortlist")
    p.add_argument("--sweeps-dir", type=Path, default=DEFAULT_SWEEPS)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing frozen_config.json",
    )
    p.add_argument(
        "--max-total-genomes",
        type=int,
        default=150,
        help="Pilot EA budget written into frozen_config (default 150)",
    )
    args = p.parse_args()

    prop_path = args.sweeps_dir / "shortlist_proposal.json"
    if not prop_path.is_file():
        raise SystemExit(f"Missing {prop_path}; run p1d first")
    if args.out.is_file() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    proposal = json.loads(prop_path.read_text(encoding="utf-8"))
    distances = {}
    for plan_name, cfg in proposal["per_distance"].items():
        cli = cfg["cli_method"]
        distances[cli] = {
            "plan_name": plan_name,
            "theta_sim": float(cfg["theta_sim"]),
            "eps": float(cfg["eps"]),
            "dbscan_eps": float(cfg["dbscan_eps"]),
            "min_samples": int(cfg["min_samples"]),
            "dbscan_min_samples": int(cfg["min_samples"]),
            "N_calibrated_on": int(cfg["N_used"]),
            "smoke_sized_calibration": bool(cfg.get("smoke_sized", False)),
        }

    frozen = {
        "schema_version": 1,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source": "experiments/evostar_2027/rq1 Part 1",
        "seed_file": "data/prompt.csv",
        "max_total_genomes": int(args.max_total_genomes),
        "shortlist_cli_methods": list(proposal["shortlist_cli_methods"]),
        "clustering_methods": list(proposal["clustering_to_pilot"]),
        "temporal_knobs": "code_defaults_untouched",
        "distances": distances,
        "notes": list(proposal.get("notes") or []),
        "proposal_path": str(prop_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  methods: {frozen['shortlist_cli_methods']}")
    print(f"  clustering: {frozen['clustering_methods']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
