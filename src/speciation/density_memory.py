"""Grow-only density memory for Incremental DBSCAN.

For ``clustering_method=dbscan``, ``temp.json`` *is* D_t: all discovered
genomes (density clusters + noise). Never delete rows. Leader-follower continues
to treat ``temp.json`` as staging and is unaffected by these helpers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from utils import get_custom_logging, get_system_utils

get_logger, _, _, _ = get_custom_logging()
_, _, _, get_outputs_path, _, _, _ = get_system_utils()

NOISE_LABEL = -1
DENSITY_LABEL_KEY = "density_label"


def density_memory_path(outputs_path: Optional[str] = None) -> Path:
    """Persistent D_t path — ``temp.json`` for the IncDBSCAN arm."""
    base = Path(outputs_path) if outputs_path else get_outputs_path()
    return base / "temp.json"


def load_density_memory(path: Optional[str] = None, logger=None) -> List[Dict[str, Any]]:
    p = Path(path) if path else density_memory_path()
    if not p.exists():
        return []
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception as e:
        log = logger or get_logger("DensityMemory")
        log.warning("Failed to load density memory %s: %s", p, e)
        return []


def save_density_memory(
    genomes: Sequence[Dict[str, Any]],
    path: Optional[str] = None,
    logger=None,
) -> None:
    p = Path(path) if path else density_memory_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(list(genomes), f, indent=2, ensure_ascii=False)
    log = logger or get_logger("DensityMemory")
    log.debug("Saved density memory n=%d → %s", len(genomes), p)


def is_labeled(genome: Dict[str, Any]) -> bool:
    """True if genome already has an IncDBSCAN density assignment."""
    if DENSITY_LABEL_KEY in genome and genome[DENSITY_LABEL_KEY] is not None:
        return True
    # After seed/insert we always set density_label; species_id alone is not enough
    # for newcomers that might carry a stale L–F species_id.
    return False


def get_density_label(genome: Dict[str, Any]) -> Optional[int]:
    if DENSITY_LABEL_KEY in genome and genome[DENSITY_LABEL_KEY] is not None:
        return int(genome[DENSITY_LABEL_KEY])
    return None


def set_density_label(genome: Dict[str, Any], label: int) -> None:
    """Set persistent density label and mirror onto ``species_id`` for selectors."""
    lab = int(label)
    genome[DENSITY_LABEL_KEY] = lab
    genome["species_id"] = lab


def unlabeled_indices(genomes: Sequence[Dict[str, Any]]) -> List[int]:
    return [i for i, g in enumerate(genomes) if not is_labeled(g)]


def append_genomes(
    existing: List[Dict[str, Any]],
    newcomers: Sequence[Dict[str, Any]],
    *,
    logger=None,
) -> List[Dict[str, Any]]:
    """Append newcomers by id (grow-only; never removes existing rows)."""
    seen: Set[str] = {str(g.get("id")) for g in existing if g.get("id") is not None}
    out = list(existing)
    added = 0
    for g in newcomers:
        key = str(g.get("id")) if g.get("id") is not None else None
        if key is None or key in seen:
            continue
        seen.add(key)
        out.append(g)
        added += 1
    log = logger or get_logger("DensityMemory")
    if added:
        log.info("Density memory append: +%d (total=%d)", added, len(out))
    return out


def prompts_in_memory(genomes: Sequence[Dict[str, Any]]) -> Set[str]:
    return {g["prompt"] for g in genomes if isinstance(g, dict) and g.get("prompt")}
