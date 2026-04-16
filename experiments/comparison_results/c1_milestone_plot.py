#!/usr/bin/env python3
"""
Generate ONLY the C1 milestone plot:
  `experiments/comparison_results/c1_ppsn2026_two_way/figures/trajectory_over_evaluated_genomes.pdf`

Plot definition:
  - x-axis: 0..1000, milestones every 100
  - y-axis: best-so-far (fixed 0.0..1.0)
  - aggregation: at each milestone, take the MAX best-so-far across runs (per method)
  - styling: no title, no grid lines, keep legend, embedded TrueType fonts (Type 42)

Run:
  python experiments/comparison_results/c1_milestone_plot.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt


PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent

if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402

configure_matplotlib_embedded_fonts()

DATA = PROJ / "data" / "outputs" / "ppsn2026"
TOXSEARCH_DIR = DATA / "toxsearch"
TOXSEARCH_S_DIR = DATA / "toxsearch_s"

OUT = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_two_way"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_runs(method: str, root: Path) -> List[Tuple[str, str, Path]]:
    runs: List[Tuple[str, str, Path]] = []
    if not root.exists():
        return runs
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        if (p / "EvolutionTracker.json").exists():
            runs.append((method, p.name, p))
    return runs


def best_so_far_vs_budget(tracker: Dict[str, Any], method: str) -> Tuple[np.ndarray, np.ndarray]:
    gens = tracker.get("generations") or []
    gens = sorted(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return np.asarray([0.0]), np.asarray([0.0])

    cum = 0.0
    run = 0.0
    x: List[float] = [0.0]
    y: List[float] = [0.0]

    if method == "toxsearch_s":
        g0 = gens[0]
        step0 = g0.get("evaluated_this_generation")
        step0 = float(step0) if isinstance(step0, (int, float)) else 0.0
        m0 = g0.get("max_score_variants")
        m0 = float(m0) if isinstance(m0, (int, float)) else 0.0
        run = max(run, m0)
        cum += max(0.0, step0)
        x.append(cum)
        y.append(run)
        rest = gens[1:]
    else:
        rest = gens

    for g in rest:
        m = g.get("max_score_variants")
        m = float(m) if isinstance(m, (int, float)) else 0.0
        run = max(run, m)

        if method == "toxsearch_s":
            step = g.get("evaluated_this_generation")
            step = float(step) if isinstance(step, (int, float)) else 0.0
        else:
            step = g.get("variants_created")
            step = float(step) if isinstance(step, (int, float)) else 0.0

        cum += max(0.0, step)
        if method == "toxsearch" and cum == 0.0:
            continue
        x.append(cum)
        y.append(run)

    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def stepwise_at(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    cur = float(ys[0]) if ys else 0.0
    for x, y in zip(xs, ys):
        if float(x) <= xq:
            cur = float(y)
        else:
            break
    return cur


def plot_milestone_max_across_runs(
    runs: List[Tuple[str, str, Path]],
    out_path: Path,
    milestones: Sequence[int] = tuple([0] + list(range(100, 1100, 100))),
) -> None:
    by_method: Dict[str, List[Dict[int, float]]] = {}

    for method, _, run_dir in runs:
        tracker = load_json(run_dir / "EvolutionTracker.json")
        xs, ys = best_so_far_vs_budget(tracker, method)
        pairs = sorted(zip(xs.tolist(), ys.tolist()), key=lambda t: float(t[0]))
        xs2 = [float(p[0]) for p in pairs]
        ys2 = [float(p[1]) for p in pairs]
        vals = {int(m): stepwise_at(xs2, ys2, float(m)) for m in milestones}
        by_method.setdefault(method, []).append(vals)

    plt.figure(figsize=(7.5, 4.5))
    for method in ["toxsearch", "toxsearch_s"]:
        if method not in by_method or not by_method[method]:
            continue
        y_max = [max(v[m] for v in by_method[method]) for m in milestones]
        plt.plot(list(milestones), y_max, marker="o", linewidth=2.5, label=method)

    plt.xlabel("Evaluated genomes")
    plt.ylabel("Best-so-far")
    plt.xlim(min(milestones), max(milestones))
    plt.ylim(0.0, 1.0)
    plt.grid(False)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def main() -> int:
    runs: List[Tuple[str, str, Path]] = []
    runs.extend(discover_runs("toxsearch", TOXSEARCH_DIR))
    runs.extend(discover_runs("toxsearch_s", TOXSEARCH_S_DIR))

    out_pdf = FIG / "trajectory_over_evaluated_genomes.pdf"
    plot_milestone_max_across_runs(runs, out_pdf)
    print(f"Wrote figure: {out_pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

