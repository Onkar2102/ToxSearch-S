"""Detect clustering population mode without importing heavy speciation stacks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def _normalize_method(method: Optional[str]) -> Optional[str]:
    if method is None:
        return None
    m = str(method).strip().lower().replace("-", "_")
    if m in ("lf",):
        return "leader_follower"
    if m in ("global",):
        return "none"
    return m


def _method_from_outputs(outputs_path: Optional[str]) -> Optional[str]:
    if outputs_path is None:
        return None
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
                m = data.get("clustering_method") or (data.get("config") or {}).get(
                    "clustering_method"
                )
            if m is not None:
                return _normalize_method(m)
        except Exception:
            continue
    return None


def _resolve_method(
    clustering_method: Optional[str] = None,
    config: Any = None,
    outputs_path: Optional[str] = None,
) -> Optional[str]:
    if clustering_method is not None:
        return _normalize_method(clustering_method)
    if config is not None:
        m = getattr(config, "clustering_method", None)
        if m is not None:
            return _normalize_method(m)
        if isinstance(config, dict) and config.get("clustering_method"):
            return _normalize_method(config["clustering_method"])
    return _method_from_outputs(outputs_path)


def is_inc_dbscan_mode(
    clustering_method: Optional[str] = None,
    config: Any = None,
    outputs_path: Optional[str] = None,
) -> bool:
    """True when CLI/config uses Incremental DBSCAN (``clustering_method=dbscan``)."""
    return _resolve_method(clustering_method, config, outputs_path) == "dbscan"


def is_global_mode(
    clustering_method: Optional[str] = None,
    config: Any = None,
    outputs_path: Optional[str] = None,
) -> bool:
    """True when CLI/config uses Global-EA (``clustering_method=none``)."""
    return _resolve_method(clustering_method, config, outputs_path) == "none"
