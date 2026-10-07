"""Detect clustering population mode without importing heavy speciation stacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def is_inc_dbscan_mode(
    clustering_method: Optional[str] = None,
    config: Any = None,
    outputs_path: Optional[str] = None,
) -> bool:
    """True when CLI/config uses Incremental DBSCAN (``clustering_method=dbscan``)."""
    if clustering_method is not None:
        m = str(clustering_method).strip().lower().replace("-", "_")
        return m == "dbscan"
    if config is not None:
        m = getattr(config, "clustering_method", None)
        if m is not None:
            return is_inc_dbscan_mode(clustering_method=m)
        if isinstance(config, dict) and config.get("clustering_method"):
            return is_inc_dbscan_mode(clustering_method=config["clustering_method"])
    if outputs_path is not None:
        base = Path(outputs_path)
        for name in ("run_config.json", "speciation_state.json"):
            p = base / name
            if not p.exists():
                continue
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if name == "run_config.json":
                    m = (data.get("speciation") or {}).get("clustering_method")
                else:
                    m = data.get("clustering_method") or (data.get("config") or {}).get("clustering_method")
                if m is not None:
                    return is_inc_dbscan_mode(clustering_method=m)
            except Exception:
                continue
    return False
