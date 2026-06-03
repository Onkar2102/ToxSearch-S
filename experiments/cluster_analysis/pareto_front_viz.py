#!/usr/bin/env python3
"""Wrapper: delegates to ``src/utils/post_hoc/runner.py``."""

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

    parser = argparse.ArgumentParser(description="Pareto post-hoc figures")
    parser.add_argument("--run-dir", type=Path,
                        default=_REPO_ROOT / "data" / "outputs" / "20260211_2122")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None, metavar="N")
    # legacy flags kept so old CLI calls still parse
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
