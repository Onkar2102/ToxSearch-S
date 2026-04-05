#!/usr/bin/env python3
"""
Plot scaling curves from EvolutionTracker.json for distribution research.

Loads one or more EvolutionTracker.json files (e.g. from runs with different
num_workers), and plots:
  - generation_duration_seconds vs generation
  - genomes_per_second vs generation
  - species_count vs generation (from speciation block)
  - population_max_toxicity (best-so-far) vs generation

Each run is labeled by run_metadata.num_workers (or path name if missing).

Usage:
  python experiments/plot_scaling_curves.py \
    data/outputs/run_1w/EvolutionTracker.json \
    data/outputs/run_2w/EvolutionTracker.json \
    data/outputs/run_4w/EvolutionTracker.json \
    -o experiments/scaling_curves.png

Requires: matplotlib, pandas (optional).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

def load_tracker(path: Path) -> Optional[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = str(path)
    return data

def extract_series(tracker: Dict[str, Any]) -> Dict[str, List[Any]]:
    gens = tracker.get("generations") or []
    n_workers = (tracker.get("run_metadata") or {}).get("num_workers") or tracker.get("_path", "?")
    label = f"workers={n_workers}"
    out = {
        "generation": [],
        "generation_duration_seconds": [],
        "genomes_per_second": [],
        "species_count": [],
        "population_max_toxicity": [],
    }
    for g in sorted(gens, key=lambda x: x.get("generation_number", 0)):
        gen_num = g.get("generation_number")
        if gen_num is None:
            continue
        out["generation"].append(gen_num)
        out["generation_duration_seconds"].append(g.get("generation_duration_seconds"))
        out["genomes_per_second"].append(g.get("genomes_per_second"))
        spec = g.get("speciation") or {}
        out["species_count"].append(spec.get("species_count"))
        out["population_max_toxicity"].append(g.get("best_fitness"))
    return out, label

def main():
    ap = argparse.ArgumentParser(description="Plot scaling curves from EvolutionTracker.json")
    ap.add_argument("trackers", nargs="+", help="Paths to EvolutionTracker.json")
    ap.add_argument("-o", "--output", default="experiments/scaling_curves.png", help="Output plot path")
    ap.add_argument("--no-plot", action="store_true", help="Only print summary, do not plot")
    args = ap.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        _proj = Path(__file__).resolve().parents[1]
        if str(_proj / "src") not in sys.path:
            sys.path.insert(0, str(_proj / "src"))
        from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

        configure_matplotlib_embedded_fonts()
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not found; install it to generate plots. Printing summary only.")
        args.no_plot = True

    series_by_label: Dict[str, Dict[str, List[Any]]] = {}
    for p in args.trackers:
        path = Path(p)
        if not path.exists():
            print(f"Skip (not found): {path}")
            continue
        tracker = load_tracker(path)
        if not tracker:
            continue
        series, label = extract_series(tracker)
        if label in series_by_label:
            label = f"{label}_{path.parent.name}"
        series_by_label[label] = series
        print(f"Loaded {path}: {len(series['generation'])} generations, label={label}")

    if args.no_plot:
        return
    if not series_by_label:
        print("No trackers loaded.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for label, series in series_by_label.items():
        gen = series["generation"]
        for ax, key, ylabel in [
            (axes[0, 0], "generation_duration_seconds", "Generation duration (s)"),
            (axes[0, 1], "genomes_per_second", "Genomes/s"),
            (axes[1, 0], "species_count", "Species count"),
            (axes[1, 1], "population_max_toxicity", "Max toxicity (best-so-far)"),
        ]:
            y = series[key]
            valid = [(a, b) for a, b in zip(gen, y) if b is not None]
            if valid:
                xs, ys = zip(*valid)
                ax.plot(xs, ys, label=label, marker=".", markersize=4)
    for ax in axes.flat:
        ax.legend()
        ax.set_xlabel("Generation")
    axes[0, 0].set_ylabel("Generation duration (s)")
    axes[0, 1].set_ylabel("Genomes/s")
    axes[1, 0].set_ylabel("Species count")
    axes[1, 1].set_ylabel("Max toxicity")
    plt.suptitle("Scaling: by number of workers")
    plt.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    main()
