#!/usr/bin/env python3
"""Fast verification for phases 2–12 (skips 0/1). Phase 5 uses n_bootstrap=50."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

from verify_phases import RUN_ID, MIN_TOPIC_SIZE, RESULTS_DIR, load_unified_rows  # noqa: E402

import analysis_utils  # noqa: E402
import counterfactual_survival  # noqa: E402


def _run(name: str, fn: Callable[[], Any]) -> Tuple[str, str]:
    try:
        out = fn()
        if isinstance(out, dict) and out.get("meta", {}).get("error"):
            return "FAIL", str(out["meta"]["error"])
        n = len(out.get("artifacts", out)) if isinstance(out, dict) else 0
        return "PASS", f"{n} artifacts"
    except Exception as exc:
        return "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def save_phase5_fast(rows, out_dir, run_id):
    """Phase 5 without 2000-iteration bootstrap (Wilson CIs only)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    all_rows: List[Dict[str, Any]] = []
    by_ev: Dict[str, List[Dict[str, Any]]] = {}
    for evaluator in counterfactual_survival.EVALUATORS:
        ev_rows = counterfactual_survival.run_counterfactual_for_evaluator(
            rows,
            evaluator,
            min_topic_size=MIN_TOPIC_SIZE,
            fast_non_dominated_sort=analysis_utils.fast_non_dominated_sort,
            global_pareto_annotate=analysis_utils.global_pareto_annotate,
        )
        ev_rows = counterfactual_survival.annotate_survival_with_cis(
            ev_rows,
            rows,
            evaluator,
            min_topic_size=MIN_TOPIC_SIZE,
            fast_non_dominated_sort=analysis_utils.fast_non_dominated_sort,
            global_pareto_annotate=analysis_utils.global_pareto_annotate,
            bootstrap_policies=(),  # Wilson only — fast
            n_bootstrap=0,
        )
        by_ev[evaluator] = ev_rows
        all_rows.extend(ev_rows)
    if not all_rows:
        raise RuntimeError("counterfactual produced no rows")
    artifacts["n_policies"] = str(len(all_rows))
    return {"artifacts": artifacts, "meta": {"run_id": run_id, "n_rows": len(all_rows)}}


def main() -> int:
    manifest = json.loads((RESULTS_DIR / "phase1_manifest.json").read_text(encoding="utf-8"))
    run_path = Path(manifest["run_path"])

    print("Loading unified rows...", flush=True)
    rows = load_unified_rows()
    print(f"  {len(rows)} rows\n", flush=True)

    stats = analysis_utils.annotate_pareto_ranks(rows)

    phases = [
        ("phase2", lambda: analysis_utils.save_phase2_artifacts(
            rows, stats, RESULTS_DIR / "phase2", RUN_ID)),
        ("phase3", lambda: analysis_utils.save_phase3_artifacts(
            rows, RESULTS_DIR / "phase3", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase4", lambda: analysis_utils.save_phase4_artifacts(
            run_path, RESULTS_DIR / "phase4", RUN_ID)),
        ("phase5", lambda: save_phase5_fast(rows, RESULTS_DIR / "phase5", RUN_ID)),
        ("phase6", lambda: analysis_utils.save_phase6_artifacts(
            rows, RESULTS_DIR / "phase6", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase7", lambda: analysis_utils.save_phase7_artifacts(
            rows, RESULTS_DIR / "phase7", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase8", lambda: analysis_utils.save_phase8_artifacts(
            rows, RESULTS_DIR / "phase8", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase9", lambda: analysis_utils.save_phase9_artifacts(
            rows, RESULTS_DIR / "phase9", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase11", lambda: analysis_utils.save_phase11_artifacts(
            rows, RESULTS_DIR / "phase11", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase12", lambda: analysis_utils.save_phase12_artifacts(
            rows, RESULTS_DIR / "phase12", RUN_ID, min_topic_size=MIN_TOPIC_SIZE)),
        ("phase10", lambda: analysis_utils.save_phase10_artifacts(RESULTS_DIR, RUN_ID)),
    ]

    results: Dict[str, Tuple[str, str]] = {}
    failures = 0
    for name, fn in phases:
        print(f"--- {name} ---", flush=True)
        status, detail = _run(name, fn)
        results[name] = (status, detail)
        print(f"  {status}: {detail.split(chr(10))[0]}", flush=True)
        if status == "FAIL":
            failures += 1
            print(detail, flush=True)

    print("\nSUMMARY", flush=True)
    for name, (st, _) in results.items():
        print(f"  {name}: {st}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
