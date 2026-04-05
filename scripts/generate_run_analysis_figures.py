#!/usr/bin/env python3
"""
Generate per-run analysis figures, including fig3_toxicity_by_species.

Figure 3 uses species labels (c-TF-IDF from speciation_state.json) on the y-axis
instead of species_id. Run from project root with:

  PYTHONPATH=src python scripts/generate_run_analysis_figures.py <run_dir>

Example:
  PYTHONPATH=src python scripts/generate_run_analysis_figures.py data/outputs/20260211_2122

Outputs:
  <run_dir>/analysis/figures/fig3_toxicity_by_species.png
  <run_dir>/analysis/figures/fig3_toxicity_by_species.pdf
"""

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

# Project root
PROJ = Path(__file__).resolve().parents[1]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))
if str(PROJ / "src") not in sys.path:
    sys.path.insert(0, str(PROJ / "src"))
from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

configure_matplotlib_embedded_fonts()
import matplotlib.pyplot as plt
import seaborn as sns


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_toxicity(genome: Dict[str, Any]) -> Optional[float]:
    """Extract toxicity from genome (exclude refusals)."""
    if not isinstance(genome, dict):
        return None
    if genome.get("is_refusal") == 1:
        return None
    for key in ("toxicity", "north_star_score", "fitness"):
        v = genome.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    try:
        v = genome.get("moderation_result", {}).get("google", {}).get("scores", {}).get("toxicity")
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    except Exception:
        pass
    return None


def is_valid_species_id(sid: Any) -> bool:
    sid_str = str(sid) if sid is not None else ""
    return sid_str not in ("", "0", "-1", "None", "unknown")


def generate_fig3_toxicity_by_species(run_dir: Path) -> bool:
    """
    Generate fig3_toxicity_by_species with species LABELS on y-axis (not species_id).
    Uses speciation_state.json labels (c-TF-IDF) for each species.
    """
    run_dir = Path(run_dir).resolve()
    spec_state = load_json(run_dir / "speciation_state.json")
    elites = load_json(run_dir / "elites.json") or []
    reserves = load_json(run_dir / "reserves.json") or []

    if not spec_state or "species" not in spec_state:
        print(f"[fig3] No speciation_state or species in {run_dir}; skipping.")
        return False

    # Build species_id -> display label (from c-TF-IDF labels in speciation_state)
    species_id_to_label: Dict[str, str] = {}
    mature_species_ids = set()
    for sid, sdata in spec_state["species"].items():
        state = sdata.get("species_state", "")
        if state not in ("active", "frozen"):
            continue
        labels = sdata.get("labels", [])
        if not labels:
            labels = [f"Species {sid}"]
        clean = [str(l).strip() for l in labels if l and str(l).strip()]
        if not clean:
            clean = [f"Species {sid}"]
        species_id_to_label[str(sid)] = ", ".join(clean[:10])  # up to 10 label words
        mature_species_ids.add(str(sid))

    # Collect toxicity per species (only mature species with labels)
    rows: List[Dict[str, Any]] = []
    for genome in elites + reserves:
        sid = str(genome.get("species_id", ""))
        if sid not in mature_species_ids or sid not in species_id_to_label or not is_valid_species_id(sid):
            continue
        tox = get_toxicity(genome)
        if tox is None:
            continue
        rows.append({
            "species_id": sid,
            "species_label": species_id_to_label[sid],
            "toxicity": tox,
        })

    if not rows:
        print(f"[fig3] No toxicity data for mature species in {run_dir}; skipping.")
        return False

    df = pd.DataFrame(rows)
    # Order by max toxicity per species (descending)
    order = df.groupby("species_label")["toxicity"].max().sort_values(ascending=False).index.tolist()
    df = df[df["species_label"].isin(order)]

    fig, ax = plt.subplots(figsize=(12, max(6, len(order) * 0.45)))
    sns.boxplot(
        data=df, y="species_label", x="toxicity", order=order,
        hue="species_label", palette="tab10", ax=ax, orient="h",
        linewidth=1.5, width=0.6, legend=False,
    )
    sns.stripplot(
        data=df, y="species_label", x="toxicity", order=order,
        color="black", alpha=0.5, size=3, ax=ax, jitter=0.3, orient="h", linewidth=0.5,
    )
    ax.set_xlabel("Toxicity Score", fontsize=14, fontweight="bold")
    ax.set_ylabel("Species (by semantic label)", fontsize=14, fontweight="bold")
    ax.set_title("Toxicity Distribution by Species", fontsize=16)
    ax.set_xlim(0, 1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()

    out_dir = run_dir / "analysis" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("png", 300), ("pdf", None)]:
        path = out_dir / f"fig3_toxicity_by_species.{ext}"
        plt.savefig(path, dpi=dpi or 150, bbox_inches="tight")
        print(f"Saved: {path}")
    plt.close()
    return True


def main():
    import argparse
    p = argparse.ArgumentParser(description="Generate per-run analysis figures (fig3: toxicity by species with labels).")
    p.add_argument("run_dir", type=Path, help="Run output directory (e.g. data/outputs/20260211_2122)")
    args = p.parse_args()
    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"Error: not a directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    ok = generate_fig3_toxicity_by_species(run_dir)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
