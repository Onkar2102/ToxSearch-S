#!/usr/bin/env python3
"""Part 1e: write immutable frozen_config.json for Part 2 / RQ2 live cells.

Example (RQ2 factorial)::

    PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1e_freeze_config.py \\
      --sweeps-dir experiments/evostar_2027/rq1/sweeps/B500 \\
      --cli-methods embedding,nli_entail \\
      --max-total-genomes 2000 --force
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

DEFAULT_SWEEPS = Path("experiments/evostar_2027/rq1/sweeps")
DEFAULT_OUT = Path("experiments/evostar_2027/rq1/frozen_config.json")


def _parse_csv_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None or str(raw).strip() == "":
        return None
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1/RQ2 p1e: freeze Part 1 geometry/shortlist")
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
        default=2000,
        help="Live EA budget written into frozen_config (default 2000 for RQ2)",
    )
    p.add_argument(
        "--cli-methods",
        type=str,
        default="",
        help="Comma CLI distance methods to keep (default: all in proposal). "
        "RQ2: embedding,nli_entail",
    )
    p.add_argument(
        "--clustering-methods",
        type=str,
        default="",
        help="Comma clustering methods (default: proposal clustering_to_pilot). "
        "RQ2: leader_follower,dbscan",
    )
    p.add_argument(
        "--pilot-seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Documented seed list for the live factorial (default 0..9)",
    )
    args = p.parse_args()

    prop_path = args.sweeps_dir / "shortlist_proposal.json"
    if not prop_path.is_file():
        raise SystemExit(f"Missing {prop_path}; run p1d first")
    if args.out.is_file() and not args.force:
        raise SystemExit(f"{args.out} exists; pass --force to overwrite")

    proposal = json.loads(prop_path.read_text(encoding="utf-8"))
    want_cli = _parse_csv_list(args.cli_methods)
    want_clus = _parse_csv_list(args.clustering_methods)

    distances = {}
    for plan_name, cfg in proposal["per_distance"].items():
        cli = cfg["cli_method"]
        if want_cli is not None and cli not in want_cli:
            continue
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

    if want_cli is not None:
        missing = [m for m in want_cli if m not in distances]
        if missing:
            raise SystemExit(f"Requested cli methods not in proposal: {missing}")
        shortlist = list(want_cli)
    else:
        shortlist = list(proposal["shortlist_cli_methods"])
        shortlist = [m for m in shortlist if m in distances]

    if want_clus is not None:
        clustering = list(want_clus)
    else:
        clustering = list(proposal["clustering_to_pilot"])

    seeds = [int(x) for x in str(args.pilot_seeds).split(",") if x.strip() != ""]

    frozen = {
        "schema_version": 2,
        "campaign": "rq2_factorial",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "source": "experiments/evostar_2027/rq1 Part 1 offline (B500 radii)",
        "seed_file": "data/prompt.csv",
        "max_total_genomes": int(args.max_total_genomes),
        "pilot_seeds": seeds,
        "n_cells": int(len(shortlist) * len(clustering) * len(seeds)),
        "shortlist_cli_methods": shortlist,
        "clustering_methods": clustering,
        "temporal_knobs": "code_defaults_untouched",
        "distances": distances,
        "notes": list(proposal.get("notes") or [])
        + [
            "RQ2 live factorial: d_E (embedding) vs d_negE (nli_entail) × LF vs IncDBSCAN; "
            "d_EP and full d_N dropped from live grid after offline RQ1.",
        ],
        "proposal_path": str(prop_path),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(frozen, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")
    print(f"  methods: {frozen['shortlist_cli_methods']}")
    print(f"  clustering: {frozen['clustering_methods']}")
    print(f"  N_max={frozen['max_total_genomes']} seeds={frozen['pilot_seeds']} cells={frozen['n_cells']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
