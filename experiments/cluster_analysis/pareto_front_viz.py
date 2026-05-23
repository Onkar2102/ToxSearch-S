#!/usr/bin/env python3
"""Thin shim around `utils.post_hoc` for backwards compatibility.

The canonical implementation now lives at [`src/utils/post_hoc/`](../../src/utils/post_hoc/).
Existing invocations like ``python experiments/cluster_analysis/pareto_front_viz.py
--run-dir data/outputs/<run>`` still work; they simply re-route into the
package's CLI orchestrator.

Differences vs the historical script:

- Axis order is read from :data:`utils.objectives.PERSPECTIVE_AXIS_ORDER`
  (matches production telemetry).
- ``--leaders-only``, ``--full``, and ``--max-size`` are accepted but no longer
  customize the post-hoc pipeline. All elites + reserves rows are used as-is;
  cohort fronts are recomputed from the canonical objective vectors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> None:
    from utils.post_hoc.runner import run_post_hoc_analysis

    parser = argparse.ArgumentParser(description="Pareto post-hoc analysis (compat shim).")
    parser.add_argument("--run-dir", type=Path,
                        default=_REPO_ROOT / "data" / "outputs" / "20260211_2122")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None, metavar="N")
    # Accept (and ignore) the legacy flags so old invocations don't error out.
    parser.add_argument("--leaders-only", action="store_true",
                        help="(legacy, ignored)")
    parser.add_argument("--full", action="store_true",
                        help="(legacy, ignored)")
    parser.add_argument("--max-size", type=int, default=None, metavar="N",
                        help="(legacy, ignored)")
    args = parser.parse_args()
    results = run_post_hoc_analysis(args.run_dir, generation=args.generation, out_dir=args.out_dir)
    print(f"Wrote post-hoc figures to: {results.get('out_dir')}")


if __name__ == "__main__":
    main()
