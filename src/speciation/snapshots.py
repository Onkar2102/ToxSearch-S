"""Per-generation species snapshots (``species_snapshots.jsonl``).

``genome_tracker.json`` keeps only each genome's *current* species, so niche
membership over time cannot be rebuilt after a run. One JSON line per
generation records the live partition, leaders, and speciation cost counters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

SNAPSHOT_FILE = "species_snapshots.jsonl"


def _species_id(genome: Dict[str, Any], is_inc: bool) -> int:
    if is_inc:
        return int(genome.get("density_label", genome.get("species_id", 0)) or 0)
    sid = genome.get("species_id")
    return int(sid) if sid is not None else -1


def append_species_snapshot(
    outputs_path: Path,
    generation: int,
    *,
    is_inc: bool,
    speciation_result: Optional[Dict[str, Any]] = None,
    logger=None,
) -> None:
    from utils.population_io import load_analysis_population
    from .distance_metrics.nli import get_nli_stats

    outputs_path = Path(outputs_path)
    pop = load_analysis_population(outputs_path)
    membership = {str(g.get("id")): _species_id(g, is_inc) for g in pop if g and g.get("id") is not None}

    leaders: Dict[str, Any] = {}
    state_path = outputs_path / "speciation_state.json"
    if state_path.exists():
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                st = json.load(f)
            for sid, sp in (st.get("species") or {}).items():
                leaders[str(sid)] = {
                    "leader_id": sp.get("leader_id"),
                    "leader_fitness": sp.get("leader_fitness"),
                    "species_state": sp.get("species_state"),
                    "created_at": sp.get("created_at"),
                    "cluster_origin": sp.get("cluster_origin"),
                    "parent_ids": sp.get("parent_ids"),
                }
        except Exception as e:
            if logger is not None:
                logger.warning("Species snapshot: could not read speciation_state.json: %s", e)

    res = speciation_result or {}
    row = {
        "generation": int(generation),
        "population_mode": "inc_dbscan" if is_inc else "leader_follower",
        "n_genomes": len(membership),
        "membership": membership,
        "leaders": leaders,
        "speciation_duration_seconds": res.get("speciation_duration_seconds"),
        "nli": get_nli_stats(),
    }
    with open(outputs_path / SNAPSHOT_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
