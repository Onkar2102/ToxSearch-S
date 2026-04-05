#!/usr/bin/env python3
"""
C1 omnibus check: Kruskal–Wallis on run-level best toxicity across three groups.

Each --group takes a name followed by one or more paths. Paths ending in .jsonl are treated as
RainbowPlus genome logs (max toxicity-like field). Other paths are read as EvolutionTracker.json
(uses population_max_toxicity).

Example:
  python experiments/rq1_three_way.py \\
    --group toxsearch_pool outputs/pool/run1/EvolutionTracker.json \\
    --group toxsearch_s outputs/spec/run1/EvolutionTracker.json \\
    --group rainbow runs/rp1/all_genomes.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

_EXP = Path(__file__).resolve().parent
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from rainbowplus_io import best_scalar_field  # noqa: E402


def _best_from_tracker(path: Path) -> float:
    with path.open("r", encoding="utf-8") as f:
        t = json.load(f)
    v = t.get("population_max_toxicity")
    if v is not None:
        return float(v)
    gens = t.get("generations") or []
    if not gens:
        return 0.0
    last = max(gens, key=lambda g: g.get("generation_number", 0))
    return float(last.get("best_fitness", 0) or 0)


def _score_path(path: Path) -> float:
    if path.suffix.lower() == ".jsonl":
        b = best_scalar_field(path)
        return float(b) if b is not None else float("nan")
    return _best_from_tracker(path)


def _parse_groups(argv: Sequence[str]) -> List[Tuple[str, List[Path]]]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--group",
        action="append",
        nargs="+",
        metavar=("NAME", "PATH"),
        help="Group name and one or more tracker .json or RainbowPlus .jsonl paths",
        required=True,
    )
    args = parser.parse_args(list(argv))
    groups: List[Tuple[str, List[Path]]] = []
    for block in args.group:
        if len(block) < 2:
            print("--group needs NAME and at least one PATH", file=sys.stderr)
            sys.exit(2)
        name, paths = block[0], [Path(p) for p in block[1:]]
        for p in paths:
            if not p.exists():
                print(f"Missing path for group {name}: {p}", file=sys.stderr)
                sys.exit(2)
        groups.append((name, paths))
    return groups


def main(argv: Sequence[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    groups = _parse_groups(argv)
    if len(groups) < 2:
        print("Need at least two --group blocks.", file=sys.stderr)
        sys.exit(2)

    series: List[Tuple[str, List[float]]] = []
    for name, paths in groups:
        vals = [_score_path(p) for p in paths]
        vals = [v for v in vals if v == v]  # drop NaN
        series.append((name, vals))
        print(f"{name}: n={len(vals)}  per_run={vals}")

    try:
        from scipy.stats import kruskal
    except ImportError:
        print("scipy not installed; pip install scipy for Kruskal–Wallis.", file=sys.stderr)
        return

    nonempty = [v for _, v in series if len(v) > 0]
    if len(nonempty) < 2:
        print("Need at least two non-empty groups.", file=sys.stderr)
        return
    res = kruskal(*nonempty)
    print(f"Kruskal–Wallis H={res.statistic:.6f} p={res.pvalue:.6g}")


if __name__ == "__main__":
    main()
