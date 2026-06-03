#!/usr/bin/env python3
"""
Build one unified table of prompt, all Perspective attribute scores, and source JSON path
for a single run directory: data/outputs/20260211_2122 (no model response text).

Score columns follow the API (moderation_result.google.scores): severe_toxicity,
identity_attack, toxicity, flirtation, threat, sexually_explicit, profanity, insult.

Reads elites.json, reserves.json, archive.json, and temp.json when present in that folder.
Run from repo root (optional: output CSV, then run directory):

  python experiments/cluster_analysis/combine_toxsearch_s_prompts_responses.py
  python experiments/cluster_analysis/combine_toxsearch_s_prompts_responses.py path/to/out.csv
  python experiments/cluster_analysis/combine_toxsearch_s_prompts_responses.py path/to/out.csv path/to/run_dir
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils.evaluator_profiles import PERSPECTIVE_AXIS_ORDER  # noqa: E402

DEFAULT_RUN_DIR = REPO_ROOT / "data" / "outputs" / "20260211_2122"
POP_FILES = ("elites.json", "reserves.json", "archive.json", "temp.json")
DEFAULT_OUT = Path(__file__).resolve().parent / "toxsearch_s_unified_prompts_responses_scores.csv"

# Canonical axis order (matches src/utils/evaluator_profiles.py).
PERSPECTIVE_ATTRS = PERSPECTIVE_AXIS_ORDER


def _perspective_scores(genome: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Extract all attribute scores from moderation_result (google.scores or top-level scores)."""
    mr = genome.get("moderation_result")
    raw: Dict[str, Any] = {}
    if isinstance(mr, dict):
        ggl = mr.get("google")
        if isinstance(ggl, dict) and isinstance(ggl.get("scores"), dict):
            raw = ggl["scores"]
        elif isinstance(mr.get("scores"), dict):
            raw = mr["scores"]
    out: Dict[str, Optional[float]] = {}
    for k in PERSPECTIVE_ATTRS:
        v = raw.get(k) if isinstance(raw, dict) else None
        if v is None:
            out[k] = None
        else:
            try:
                out[k] = float(v)
            except (TypeError, ValueError):
                out[k] = None
    return out


def load_genome_list(path: Path) -> List[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def build_rows(run_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not run_dir.is_dir():
        return rows
    for fname in POP_FILES:
        fp = run_dir / fname
        if not fp.is_file():
            continue
        genomes = load_genome_list(fp)
        rel = fp.relative_to(REPO_ROOT).as_posix()
        for g in genomes:
            prompt = g.get("prompt")
            if prompt is None:
                continue
            prompt = str(prompt).strip()
            if not prompt:
                continue
            row: Dict[str, Any] = {
                "prompt": prompt,
                "source_file": rel,
            }
            row.update(_perspective_scores(g))
            rows.append(row)
    return rows


def main() -> None:
    out_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUT
    run_dir = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else DEFAULT_RUN_DIR
    rows = build_rows(run_dir)
    if not rows:
        print(f"No rows collected; check run directory: {run_dir}", file=sys.stderr)
        sys.exit(1)
    columns = ["prompt", *PERSPECTIVE_ATTRS, "source_file"]
    df = pd.DataFrame(rows, columns=columns)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
