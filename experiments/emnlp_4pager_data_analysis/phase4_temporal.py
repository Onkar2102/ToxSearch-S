"""Phase 4: temporal species dynamics from EvolutionTracker.json."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_logger = logging.getLogger("emnlp_4pager.phase4")


def load_evolution_tracker_timeseries(
    run_path: Path,
    *,
    tracker_name: str = "EvolutionTracker.json",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Parse per-generation species counts and scalar fitness from EvolutionTracker."""
    path = Path(run_path) / tracker_name
    if not path.is_file():
        return [], {"error": f"missing {path}"}

    raw = json.loads(path.read_text(encoding="utf-8"))
    generations = raw.get("generations") or []
    rows: List[Dict[str, Any]] = []

    for g in generations:
        sp = g.get("speciation") or {}
        gen = int(g.get("generation_number", len(rows)))
        active = int(sp.get("active_species_count") or 0)
        total_sp = int(sp.get("species_count") or active)
        max_var = g.get("max_score_variants")
        rows.append({
            "generation": gen,
            "active_species_count": active,
            "species_count": total_sp,
            "max_score_variants": float(max_var) if max_var is not None else None,
            "avg_fitness": float(g["avg_fitness"]) if g.get("avg_fitness") is not None else None,
            "avg_fitness_variants": float(g["avg_fitness_variants"])
            if g.get("avg_fitness_variants") is not None
            else None,
            "elites_count": int(g.get("elites_count") or 0),
            "archived_count": int(g.get("archived_count") or 0),
            "speciation_events": int(sp.get("speciation_events") or 0),
            "merge_events": int(sp.get("merge_events") or 0),
            "extinction_events": int(sp.get("extinction_events") or 0),
            "collapsed_to_one": active == 1,
        })

    rows.sort(key=lambda r: r["generation"])
    active = np.array([r["active_species_count"] for r in rows], dtype=np.int32)
    collapse_gens = [int(r["generation"]) for r in rows if r["collapsed_to_one"]]

    stats: Dict[str, Any] = {
        "tracker_path": str(path),
        "status": raw.get("status"),
        "total_generations_tracker": raw.get("total_generations"),
        "n_generations_parsed": len(rows),
        "active_species_min": int(active.min()) if len(active) else 0,
        "active_species_max": int(active.max()) if len(active) else 0,
        "final_active_species_count": int(rows[-1]["active_species_count"]) if rows else 0,
        "final_species_count": int(rows[-1]["species_count"]) if rows else 0,
        "n_generations_collapsed_to_one": len(collapse_gens),
        "collapsed_generations": collapse_gens,
        "selection_mode_final": raw.get("selection_mode"),
    }
    return rows, stats


def plot_temporal_species(
    timeseries: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    run_id: str = "",
    reference_species: int = 9,
) -> Optional[str]:
    """Fig 1A style: active species count (+ optional max variant score) vs generation."""
    if not timeseries:
        return None
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        _logger.warning("matplotlib unavailable: %s", e)
        return None

    gens = [int(r["generation"]) for r in timeseries]
    active = [int(r["active_species_count"]) for r in timeseries]
    max_scores = [r.get("max_score_variants") for r in timeseries]
    has_scores = any(v is not None for v in max_scores)
    collapse_gens = [int(r["generation"]) for r in timeseries if r.get("collapsed_to_one")]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(gens, active, color="#4C78A8", linewidth=1.8, label="Active species")
    ax1.axhline(reference_species, color="#54A24B", linestyle="--", linewidth=1.2, alpha=0.8,
                label=f"Reference ({reference_species} topics)")
    for g in collapse_gens:
        ax1.axvline(g, color="#E45756", alpha=0.12, linewidth=0.8)
    if collapse_gens:
        ax1.scatter(
            collapse_gens,
            [1] * len(collapse_gens),
            color="#E45756",
            s=28,
            zorder=4,
            label=f"Collapse to 1 ({len(collapse_gens)} gens)",
        )
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Active species count")
    ax1.set_ylim(bottom=0)
    ax1.grid(True, alpha=0.25)

    lines1, labels1 = ax1.get_legend_handles_labels()
    if has_scores:
        ax2 = ax1.twinx()
        scores = [float(v) if v is not None else np.nan for v in max_scores]
        ax2.plot(gens, scores, color="#B279A2", linewidth=1.0, alpha=0.65, label="Max variant score")
        ax2.set_ylabel("Max variant score (generation)")
        ax2.set_ylim(0, 1.05)
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8)
    else:
        ax1.legend(loc="upper left", fontsize=8)

    title = f"{run_id}: species count over evolution" if run_id else "Species count over evolution"
    ax1.set_title(title)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


__all__ = [
    "load_evolution_tracker_timeseries",
    "plot_temporal_species",
]
