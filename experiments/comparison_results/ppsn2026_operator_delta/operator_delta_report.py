#!/usr/bin/env python3
"""
PPSN2026: operator effectiveness as mean (child score − parent score) per operator,
aggregated across runs for three execution settings:

  - ToxSearch (non-speciated): toxsearch runs from c1_ppsn2026_three_way manifest
  - ToxSearch-S (sequential): toxsearch_s from the same manifest

Child score = north-star metric (default toxicity) via population_io._extract_north_star_score.
Parent score = genome.parent_score when set; otherwise parents[0].score (MPI parallel runs).
Same delta idea as src/utils/operator_effectiveness.calculate_table4_metrics.

Writes:
  operator_delta_summary.csv
  figures/operator_delta_table.pdf

Run from repo root:
  python experiments/comparison_results/ppsn2026_operator_delta/operator_delta_report.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
# population_io uses `from utils...` (expects src on path)
_src = PROJ / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402
from utils.population_io import _extract_north_star_score  # noqa: E402

configure_matplotlib_embedded_fonts()

DATA = PROJ / "data" / "outputs" / "ppsn2026"
C1_MANIFEST = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_three_way" / "run_manifest.csv"

OUT = PROJ / "experiments" / "comparison_results" / "ppsn2026_operator_delta"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

NORTH_STAR = "toxicity"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_manifest_methods(method: str) -> List[Tuple[str, Path]]:
    if not C1_MANIFEST.exists():
        return []
    out: List[Tuple[str, Path]] = []
    with C1_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("method") or "").strip() != method:
                continue
            rid = (row.get("run_id") or "").strip()
            rp = (row.get("run_path") or "").strip()
            if not rid or not rp:
                continue
            p = Path(rp)
            if (p / "EvolutionTracker.json").exists():
                out.append((rid, p))
    return out


def iter_genomes(method: str, run_dir: Path) -> Sequence[Dict[str, Any]]:
    if method == "toxsearch":
        fnames = ("elites.json", "non_elites.json", "under_performing.json")
    elif method == "toxsearch_s":
        fnames = ("elites.json", "reserves.json", "archive.json")
    else:
        raise ValueError(f"Unknown method: {method}")

    for fname in fnames:
        p = run_dir / fname
        if not p.exists():
            continue
        data = load_json(p)
        if isinstance(data, list):
            for g in data:
                if isinstance(g, dict):
                    yield g


def resolve_parent_score(g: Dict[str, Any]) -> Optional[float]:
    """
    Sequential runs set parent_score; MPI parallel runs often omit it but store scores on
    parents[0].score (see evolution outputs).
    """
    ps = g.get("parent_score")
    if ps is not None:
        return float(ps)
    parents = g.get("parents")
    if not parents:
        return None
    first = parents[0]
    if isinstance(first, dict) and first.get("score") is not None:
        return float(first["score"])
    return None


def extract_operator_deltas(method: str, run_dir: Path) -> List[Tuple[str, float]]:
    """Pairs (operator_name, child_score - parent_score) for operator-created variants (generation >= 1)."""
    out: List[Tuple[str, float]] = []
    for g in iter_genomes(method, run_dir):
        gen = g.get("generation")
        if gen is None:
            continue
        try:
            gi = int(gen)
        except (TypeError, ValueError):
            continue
        if gi <= 0:
            continue
        op = g.get("operator")
        if not op or op in ("Unknown", "Initial Seed"):
            continue
        ps = resolve_parent_score(g)
        if ps is None:
            continue
        cur = _extract_north_star_score(g, NORTH_STAR)
        if cur is None:
            continue
        delta = float(cur) - float(ps)
        if math.isnan(delta):
            continue
        out.append((str(op), delta))
    return out


def pool_by_operator(method: str, runs: List[Tuple[str, Path]]) -> DefaultDict[str, List[float]]:
    by_op: DefaultDict[str, List[float]] = defaultdict(list)
    for _, rd in runs:
        for op, d in extract_operator_deltas(method, rd):
            by_op[op].append(d)
    return by_op


def mean_sd_str(vals: List[float]) -> str:
    if not vals:
        return "—"
    a = np.asarray(vals, dtype=float)
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    return f"{m:.4f} ± {sd:.4f}"


def main() -> int:
    tox_runs = load_manifest_methods("toxsearch")
    ts_runs = load_manifest_methods("toxsearch_s")

    pools = {
        "ToxSearch": pool_by_operator("toxsearch", tox_runs),
        "ToxSearch-S (seq.)": pool_by_operator("toxsearch_s", ts_runs),
    }

    all_ops = sorted(set().union(*(p.keys() for p in pools.values())))

    rows_csv: List[Dict[str, Any]] = []
    cell_rows: List[List[str]] = []
    for op in all_ops:
        row: Dict[str, Any] = {"operator": op}
        line = [op]
        for label, key in [
            ("ToxSearch", "toxsearch"),
            ("ToxSearch-S (seq.)", "toxsearch_s_seq"),
        ]:
            vals = pools[label].get(op, [])
            row[f"{key}_mean"] = float(np.mean(vals)) if vals else ""
            row[f"{key}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else (0.0 if vals else "")
            row[f"{key}_n"] = len(vals)
            line.append(mean_sd_str(vals))
        rows_csv.append(row)
        cell_rows.append(line)

    fieldnames = [
        "operator",
        "toxsearch_mean",
        "toxsearch_std",
        "toxsearch_n",
        "toxsearch_s_seq_mean",
        "toxsearch_s_seq_std",
        "toxsearch_s_seq_n",
    ]
    with (OUT / "operator_delta_summary.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows_csv:
            w.writerow(row)

    # --- PDF table ---
    col_labels = [
        "Operator",
        "ToxSearch\nΔμ ± SD",
        "ToxSearch-S\n(seq.) Δμ ± SD",
        "ToxSearch-S\n(2w) Δμ ± SD",
    ]
    fig_h = 0.38 * (len(cell_rows) + 1) + 0.5
    fig, ax = plt.subplots(figsize=(12.5, min(fig_h, 16.0)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_rows,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.06)
        else:
            cell.set_height(0.035)
        if c > 0:
            cell.get_text().set_horizontalalignment("center")
    tbl.scale(1, 1.4)
    plt.savefig(FIG / "operator_delta_table.pdf", format="pdf")
    plt.close()

    print(f"Wrote {OUT}")
    print(f"  runs: toxsearch={len(tox_runs)}, toxsearch_s={len(ts_runs)}")
    print(f"  operators: {len(all_ops)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
