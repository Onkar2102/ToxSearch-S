#!/usr/bin/env python3
"""Offline smoke gate before any OpenAI scoring.

Sanity-check a frozen evolution run (counts, topics, objective coverage) and
write the result to ``results/gate0_smoke.json``. Anything downstream — the
unified dataset, the notebook — assumes this passes.

Examples
--------
  python experiments/emnlp_data_analysis/smoke_gate.py
  python experiments/emnlp_data_analysis/smoke_gate.py --run-path data/outputs/20260211_2122
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analysis_utils import (  # noqa: E402
    DEFAULT_PRIMARY_RUN,
    RESULTS_DIR,
    smoke_validate_run,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Offline smoke gate for a frozen run")
    parser.add_argument(
        "--run-path",
        type=Path,
        default=DEFAULT_PRIMARY_RUN,
        help="Evolution run directory (elites/reserves/archive JSON)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Analysis results root",
    )
    parser.add_argument("--min-genomes", type=int, default=5_000)
    parser.add_argument("--min-topics", type=int, default=5)
    parser.add_argument("--min-topic-size", type=int, default=5)
    parser.add_argument("--min-objective-frac", type=float, default=0.95)
    args = parser.parse_args()

    gate0 = smoke_validate_run(
        args.run_path,
        min_genomes=args.min_genomes,
        min_topics=args.min_topics,
        min_topic_size=args.min_topic_size,
        min_objective_frac=args.min_objective_frac,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.results_dir / "gate0_smoke.json"
    out_path.write_text(json.dumps(gate0, indent=2), encoding="utf-8")

    status = "PASS" if gate0["pass"] else "FAIL"
    print(f"Gate 0: {status}")
    print(f"  n_genomes={gate0['n_genomes']}, n_species={gate0['n_species']}")
    print(f"  frac_with_objectives={gate0['frac_with_objectives']:.4f}")
    print(f"Wrote {out_path}")

    if not gate0["pass"]:
        failed = [k for k, v in gate0["checks"].items() if not v]
        print("Failed checks:", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
