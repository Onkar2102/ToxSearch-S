"""Global-EA path: no niche clustering / mating restriction (``clustering_method=none``).

Uses the Leader–Follower file layout (``elites.json`` + empty ``archive.json``).
Every evaluated individual is assigned ``species_id=1`` (single synthetic niche
for tracker compatibility). Newcomers in ``temp.json`` are merged into elites
each generation; ``temp.json`` is then cleared. No embeddings, distance
matrices, capacity archive, merge, or freeze.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import SpeciationConfig

from utils import get_custom_logging
from utils import get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()

GLOBAL_SPECIES_ID = 1


def _load_json_list(path: Path, logger) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return []
    if isinstance(data, list):
        return [g for g in data if isinstance(g, dict)]
    return []


def _write_json_atomic(path: Path, data: Any, logger, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)
    logger.debug("Wrote %s (%s)", label, path.name)


def _dedupe_by_id(genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for g in genomes:
        gid = g.get("id")
        key = str(gid) if gid is not None and str(gid) != "" else None
        if key is None:
            out.append(g)
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(g)
    return out


def _north_star_score(genome: Dict[str, Any], metric: str) -> float:
    from utils.population_io import _extract_north_star_score

    try:
        return float(_extract_north_star_score(genome, metric))
    except Exception:
        return float("-inf")


def _write_speciation_state(
    outputs_path: Path,
    *,
    n_elites: int,
    current_generation: int,
    north_star_metric: str,
    best_fitness: float,
    logger,
) -> None:
    state = {
        "clustering_method": "none",
        "population_mode": "global",
        "generation": current_generation,
        "north_star_metric": north_star_metric,
        "species": {
            str(GLOBAL_SPECIES_ID): {
                "species_id": GLOBAL_SPECIES_ID,
                "species_state": "active",
                "size": n_elites,
                "best_fitness": best_fitness,
                "leader_id": None,
            }
        },
        "config": {"clustering_method": "none"},
    }
    _write_json_atomic(outputs_path / "speciation_state.json", state, logger, "speciation_state")


def run_global_speciation(
    temp_path: Optional[str] = None,
    current_generation: int = 0,
    config: Optional[SpeciationConfig] = None,
    log_file: Optional[str] = None,
    north_star_metric: str = "toxicity",
) -> Dict[str, Any]:
    """Integrate evaluated newcomers into a single global elite pool."""
    del config  # accepted for API parity; niche geometry is unused
    logger = get_logger("RunGlobal", log_file)
    t0 = time.perf_counter()
    outputs_path = get_outputs_path()
    if temp_path is None:
        temp_path = str(outputs_path / "temp.json")

    temp_path_obj = Path(temp_path)
    elites_path = outputs_path / "elites.json"
    archive_path = outputs_path / "archive.json"

    elites = _load_json_list(elites_path, logger)
    newcomers = _load_json_list(temp_path_obj, logger)

    for g in newcomers:
        if g.get("generation") is None:
            g["generation"] = current_generation

    merged = _dedupe_by_id(list(elites) + list(newcomers))
    for g in merged:
        g["species_id"] = GLOBAL_SPECIES_ID

    # Prefer higher fitness when sorting for stable elites ordering
    merged.sort(
        key=lambda g: _north_star_score(g, north_star_metric),
        reverse=True,
    )

    _write_json_atomic(elites_path, merged, logger, "elites.json")
    if not archive_path.exists():
        _write_json_atomic(archive_path, [], logger, "archive.json")

    with open(temp_path_obj, "w", encoding="utf-8") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    best = _north_star_score(merged[0], north_star_metric) if merged else 0.0
    if best == float("-inf"):
        best = 0.0

    _write_speciation_state(
        outputs_path,
        n_elites=len(merged),
        current_generation=current_generation,
        north_star_metric=north_star_metric,
        best_fitness=float(best),
        logger=logger,
    )

    elapsed = time.perf_counter() - t0
    logger.info(
        "Global-EA integrate gen=%d: elites=%d newcomers=%d (%.3fs)",
        current_generation,
        len(merged),
        len(newcomers),
        elapsed,
    )

    result = {
        "success": True,
        "global": True,
        "population_mode": "global",
        "clustering_method": "none",
        "species_count": 1 if merged else 0,
        "active_species_count": 1 if merged else 0,
        "frozen_species_count": 0,
        "archive_size": 0,
        "archived_size": 0,
        "archived_count": 0,
        "density_size": len(merged),
        "noise_count": 0,
        "singleton_species_count": 0,
        "clustered_count": len(merged),
        "largest_species_size": len(merged),
        "average_species_size": float(len(merged)),
        "speciation_events": 0,
        "merge_events": 0,
        "extinction_events": 0,
        "elites_moved": len(merged),
        "archive_moved": 0,
        "genomes_updated": len(newcomers),
        "newcomers": len(newcomers),
        "elapsed_seconds": elapsed,
        "speciation_duration_seconds": round(elapsed, 3),
    }
    try:
        from .run_speciation import update_evolution_tracker_with_speciation

        update_evolution_tracker_with_speciation(
            evolution_tracker_path=str(outputs_path / "EvolutionTracker.json"),
            current_generation=current_generation,
            speciation_result=result,
            speciation_stats={"initialized": True, "metrics_summary": {}},
            logger=logger,
        )
    except Exception as e:
        logger.error(
            "Failed to update EvolutionTracker with Global-EA data: %s",
            e,
            exc_info=True,
        )
    return result
