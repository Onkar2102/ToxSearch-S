#!/usr/bin/env python3
"""RainbowPlus JSONL helpers for C1 three-way analysis (ToxSearch vs ToxSearch-S vs RainbowPlus)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    """Yield decoded objects from a UTF-8 JSONL file (one JSON object per line)."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def count_jsonl_rows(path: Path) -> int:
    return sum(1 for _ in iter_jsonl(path))


def best_scalar_field(
    path: Path,
    field_names: Tuple[str, ...] = ("toxicity", "score", "fitness", "north_star", "max_score"),
) -> Optional[float]:
    """Best (max) of the first present numeric field among field_names per row."""
    best: Optional[float] = None
    for row in iter_jsonl(path):
        for k in field_names:
            if k not in row:
                continue
            try:
                v = float(row[k])
            except (TypeError, ValueError):
                continue
            best = v if best is None else max(best, v)
            break
    return best


def archive_cell_keys(path: Path) -> List[Tuple[Any, ...]]:
    """
    Collect unique archive-style keys from rows if present.
    RainbowPlus rows often include category / style or similar; keys are normalized to tuples for hashing.
    """
    keys: List[Tuple[Any, ...]] = []
    seen = set()
    for row in iter_jsonl(path):
        cat = row.get("category", row.get("cell_category"))
        sty = row.get("style", row.get("cell_style"))
        if cat is None and sty is None:
            continue
        key = (cat, sty)
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys
