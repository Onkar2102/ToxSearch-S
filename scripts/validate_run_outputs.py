#!/usr/bin/env python3
"""Validate MOEA run outputs across clustering / distance / evaluator configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional


def _load_json(path: Path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _expected_m(evaluator: str) -> int:
    return 8 if evaluator == "google" else 13


def validate_run(
    run_dir: Path,
    expected_evaluator: Optional[str] = None,
    *,
    require_run_config: bool = True,
) -> list[str]:
    errors: list[str] = []

    run_cfg = _load_json(run_dir / "run_config.json")
    if require_run_config and not run_cfg:
        errors.append("Missing or invalid run_config.json")
    speciation_cfg = (run_cfg or {}).get("speciation") or {}
    cfg_evaluator = (run_cfg or {}).get("evaluator")
    if expected_evaluator is None:
        expected_evaluator = cfg_evaluator or "google"
    if cfg_evaluator and cfg_evaluator != expected_evaluator:
        errors.append(
            f"run_config.evaluator={cfg_evaluator!r} != --evaluator {expected_evaluator!r}"
        )

    expected_m = _expected_m(expected_evaluator)
    backend = expected_evaluator
    clustering = speciation_cfg.get("clustering_method")
    distance = speciation_cfg.get("distance_method")
    is_inc = clustering == "dbscan"

    tracker_path = run_dir / "EvolutionTracker.json"
    tracker = _load_json(tracker_path)
    if not tracker:
        errors.append(f"Missing or invalid {tracker_path}")
        return errors

    meta = tracker.get("run_metadata") or {}
    if meta.get("evaluator") and meta.get("evaluator") != expected_evaluator:
        errors.append(
            f"run_metadata.evaluator={meta.get('evaluator')!r} expected {expected_evaluator!r}"
        )
    m_meta = meta.get("num_objectives")
    if m_meta is not None and int(m_meta) != expected_m:
        errors.append(f"run_metadata.num_objectives={m_meta} expected {expected_m}")
    axis = meta.get("axis_order")
    if isinstance(axis, list) and len(axis) != expected_m:
        errors.append(f"run_metadata.axis_order length {len(axis)} expected {expected_m}")

    # Cross-check matrix knobs when present in both places.
    if clustering and meta.get("clustering_method") and meta["clustering_method"] != clustering:
        errors.append(
            f"clustering_method mismatch: run_config={clustering!r} "
            f"run_metadata={meta.get('clustering_method')!r}"
        )
    if distance and meta.get("distance_method") and meta["distance_method"] != distance:
        errors.append(
            f"distance_method mismatch: run_config={distance!r} "
            f"run_metadata={meta.get('distance_method')!r}"
        )

    summary = tracker.get("speciation_summary") or {}
    mode = summary.get("population_mode") or meta.get("population_mode")
    if is_inc and mode and mode != "inc_dbscan":
        errors.append(f"IncDBSCAN run but tracker population_mode={mode!r}")
    if clustering == "leader_follower" and mode == "inc_dbscan":
        errors.append("LF clustering_method but tracker reports inc_dbscan")

    gens = tracker.get("generations") or []
    if gens:
        latest = max(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
        for fld in ("max_per_objective", "avg_per_objective"):
            row = latest.get(fld)
            if row is not None and isinstance(row, list) and len(row) != expected_m:
                errors.append(
                    f"gen {latest.get('generation_number')} {fld} length {len(row)} expected {expected_m}"
                )
        if is_inc:
            dens = latest.get("density_size")
            clustered = latest.get("clustered_count")
            single = latest.get("singleton_species_count")
            noise = latest.get("noise_count")
            residual = latest.get("residual_dbscan_noise_count")
            # Canonical: noise_count = singleton + residual (all non-mature).
            # Identity: density == clustered + noise_count.
            if dens is not None and clustered is not None and noise is not None:
                if int(dens) != int(clustered) + int(noise):
                    errors.append(
                        f"density identity broken: density={dens} != "
                        f"clustered({clustered})+noise({noise})"
                    )
            if (
                single is not None
                and residual is not None
                and noise is not None
                and int(noise) != int(single) + int(residual)
            ):
                errors.append(
                    f"noise decomposition broken: noise={noise} != "
                    f"singleton({single})+residual({residual})"
                )

    total_pop = 0
    pop_files = ("temp.json",) if is_inc else ("elites.json", "archive.json", "temp.json")
    found_primary = False
    for fname in pop_files:
        pop = _load_json(run_dir / fname)
        if not isinstance(pop, list):
            continue
        if fname in ("elites.json", "temp.json") and pop:
            found_primary = True
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

    if is_inc:
        if not (run_dir / "temp.json").is_file():
            errors.append("IncDBSCAN run missing temp.json (density memory D_t)")
        if not (run_dir / "inc_dbscan_state.json").is_file():
            errors.append("IncDBSCAN run missing inc_dbscan_state.json")
        else:
            st = _load_json(run_dir / "inc_dbscan_state.json") or {}
            if distance and st.get("distance_method") and st["distance_method"] != distance:
                errors.append(
                    f"inc_dbscan_state.distance_method={st.get('distance_method')!r} "
                    f"!= run_config {distance!r}"
                )
            if distance and "nli" in str(distance) and not st.get("nli_model"):
                print(
                    "  WARN: nli distance but inc_dbscan_state.json missing nli_model "
                    "(expected on new runs; legacy states omit it)"
                )
        if not found_primary:
            errors.append("IncDBSCAN: temp.json empty or missing genomes")
    else:
        if not (run_dir / "elites.json").is_file():
            errors.append("LF run missing elites.json")
        elif not found_primary and total_pop == 0:
            errors.append("no genomes found in elites.json / archive.json / temp.json")

    if total_pop == 0 and not is_inc:
        errors.append("no genomes found in elites.json / archive.json / temp.json")

    snap_path = run_dir / "species_snapshots.jsonl"
    if not snap_path.is_file():
        print("  WARN: species_snapshots.jsonl missing (expected on new runs)")
    else:
        snap_gens = []
        with snap_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    snap_gens.append(int(json.loads(line).get("generation", -1)))
        tracker_gens = {int(g.get("generation_number", -1)) for g in gens}
        missing = sorted(tracker_gens - set(snap_gens))
        if missing:
            errors.append(f"species_snapshots.jsonl missing generations {missing[:10]}")

    figures = run_dir / "figures"
    if not figures.is_dir():
        errors.append("figures/ directory missing")
    else:
        for fig in ("fitness_evolution.png", "speciation_evolution.png"):
            if not (figures / fig).is_file():
                errors.append(f"missing figure: figures/{fig}")

    status = tracker.get("status")
    print(f"  status={status!r}  generations={len(gens)}  population_files_total={total_pop}")
    print(
        f"  run_metadata: evaluator={meta.get('evaluator')} M={m_meta or len(axis or [])} "
        f"clustering={meta.get('clustering_method') or clustering} "
        f"distance={meta.get('distance_method') or distance}"
    )
    if run_cfg:
        print(
            f"  run_config: evaluator={cfg_evaluator} "
            f"clustering={clustering} distance={distance} "
            f"nli_model={speciation_cfg.get('nli_model')}"
        )

    return errors


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument(
        "--evaluator",
        choices=["google", "openai"],
        default=None,
        help="Expected evaluator (default: read from run_config.json)",
    )
    p.add_argument(
        "--allow-missing-run-config",
        action="store_true",
        help="Do not require run_config.json (legacy runs)",
    )
    args = p.parse_args()
    run_dir = args.run_dir.resolve()
    print(f"Validating {run_dir} (expected evaluator={args.evaluator or 'from run_config'})")
    errors = validate_run(
        run_dir,
        args.evaluator,
        require_run_config=not args.allow_missing_run_config,
    )
    if errors:
        print("FAIL:")
        for e in errors:
            print(f"  - {e}")
        return 1
    print("PASS: all checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
