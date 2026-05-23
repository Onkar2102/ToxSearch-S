#!/usr/bin/env python3
"""Thin shim around `utils.post_hoc.pymoo_viz`.

The canonical implementation now lives at
[`src/utils/post_hoc/pymoo_viz.py`](../../src/utils/post_hoc/pymoo_viz.py).
Invoking this script reuses the post-hoc loaders, axis order, and Pareto
helpers from ``utils.objectives``. ``pymoo`` is a soft dependency: when it is
not installed the script logs a warning and exits cleanly.
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
    from utils.post_hoc._loaders import load_objective_rows
    from utils.post_hoc.pymoo_viz import generate_pymoo_viz

    parser = argparse.ArgumentParser(description="pymoo MO viz (compat shim).")
    parser.add_argument("--run-dir", type=Path,
                        default=_REPO_ROOT / "data" / "outputs" / "20260211_2122")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--generation", type=int, default=None, metavar="N")
    parser.add_argument("--leaders-only", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--full", action="store_true", help="(legacy, ignored)")
    parser.add_argument("--max-size", type=int, default=None, help="(legacy, ignored)")
    args = parser.parse_args()

    out_dir = args.out_dir
    if out_dir is None:
        base = args.run_dir / "figures"
        if args.generation is not None:
            base = base / f"gen_{args.generation}"
        out_dir = base / "post_hoc"
    rows = load_objective_rows(args.run_dir, generation=args.generation)
    title = f"gen {args.generation}" if args.generation is not None else "final"
    results = generate_pymoo_viz(rows, out_dir, title_suffix=title)
    for name, path in results.items():
        if path:
            print(f"Wrote {name}: {path}")


if __name__ == "__main__":
    main()
