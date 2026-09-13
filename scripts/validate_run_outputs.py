#!/usr/bin/env python3
"""Validate MOEA run outputs for google (M=8) or openai (M=13) evaluator profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def validate_run(run_dir: Path, expected_evaluator: str) -> list[str]:
    errors: list[str] = []
    expected_m = 8 if expected_evaluator == "google" else 13
    backend = expected_evaluator

    tracker_path = run_dir / "EvolutionTracker.json"
    tracker = _load_json(tracker_path)
    if not tracker:
        errors.append(f"Missing or invalid {tracker_path}")
        return errors

    meta = tracker.get("run_metadata") or {}
    if meta.get("evaluator") != expected_evaluator:
        errors.append(f"run_metadata.evaluator={meta.get('evaluator')!r} expected {expected_evaluator!r}")
    m_meta = meta.get("num_objectives")
    if m_meta is not None and int(m_meta) != expected_m:
        errors.append(f"run_metadata.num_objectives={m_meta} expected {expected_m}")
    axis = meta.get("axis_order")
    if isinstance(axis, list) and len(axis) != expected_m:
        errors.append(f"run_metadata.axis_order length {len(axis)} expected {expected_m}")

    gens = tracker.get("generations") or []
    if gens:
        latest = max(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
        for fld in ("max_per_objective", "avg_per_objective"):
            row = latest.get(fld)
            if row is not None and isinstance(row, list) and len(row) != expected_m:
                errors.append(
                    f"gen {latest.get('generation_number')} {fld} length {len(row)} expected {expected_m}"
                )

    total_pop = 0
    for fname in ("elites.json", "reserves.json", "archive.json"):
        pop = _load_json(run_dir / fname)
        if not isinstance(pop, list):
            continue
        total_pop += len(pop)
        checked = 0
        bad = 0
        for g in pop[:50]:
            if not g:
                continue
            ev = g.get("evaluator")
            if ev and ev != expected_evaluator:
                bad += 1
            ov = g.get("objective_vector")
            if ov is not None and len(ov) != expected_m:
                bad += 1
            mr = g.get("moderation_result") or {}
            if backend not in mr or "scores" not in (mr.get(backend) or {}):
                bad += 1
            checked += 1
        if checked and bad:
            errors.append(f"{fname}: {bad}/{checked} sample genomes failed evaluator/vector/backend checks")

    pm = _load_json(run_dir / "p_mating.json")
    if isinstance(pm, list) and pm:
        bad_pm = sum(
            1 for r in pm
            if r.get("objective_vector") is not None and len(r["objective_vector"]) != expected_m
        )
        if bad_pm:
            errors.append(f"p_mating.json: {bad_pm} rows with wrong objective_vector length")

    figures = run_dir / "figures"
    if not figures.is_dir():
        errors.append("figures/ directory missing")
    else:
        for fig in ("objective_evolution.png", "alpha_knob.png"):
            if not (figures / fig).is_file():
                errors.append(f"missing figure: figures/{fig}")

    status = tracker.get("status")
    print(f"  status={status!r}  generations={len(gens)}  population_files_total={total_pop}")
    print(f"  run_metadata: evaluator={meta.get('evaluator')} M={m_meta or len(axis or [])}")

    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--evaluator", choices=["google", "openai"], required=True)
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    print(f"Validating {run_dir} (expected evaluator={args.evaluator})")
    errors = validate_run(run_dir, args.evaluator)
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: all checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
