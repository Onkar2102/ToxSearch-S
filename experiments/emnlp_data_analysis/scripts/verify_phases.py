#!/usr/bin/env python3
"""Run phases 2–12 against unified artifacts and report pass/fail (skips phase 0/1)."""

from __future__ import annotations

import csv
import json
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

import analysis_utils  # noqa: E402

RUN_ID = "20260211_2122"
RESULTS_DIR = THIS_DIR / "results"
UNIFIED_DIR = RESULTS_DIR / "unified"
MIN_TOPIC_SIZE = 5


def load_unified_rows() -> List[Dict[str, Any]]:
    """Reconstruct row dicts from phase-1 unified CSV + embeddings (read-only)."""
    csv_path = UNIFIED_DIR / f"{RUN_ID}_genomes.csv"
    emb_path = UNIFIED_DIR / f"{RUN_ID}_embeddings.npy"
    ids_path = UNIFIED_DIR / f"{RUN_ID}_genome_ids.json"
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing unified CSV: {csv_path}")

    embeddings = np.load(emb_path, allow_pickle=False) if emb_path.is_file() else None
    ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.is_file() else []
    id_to_emb_index = {str(g): k for k, g in enumerate(ids)}

    google_axes = list(analysis_utils.get_axis_order("google"))
    openai_axes = list(analysis_utils.get_axis_order("openai"))

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            row: Dict[str, Any] = {
                "run_id": r.get("run_id") or RUN_ID,
                "genome_id": int(r["genome_id"])
                if str(r.get("genome_id") or "").isdigit()
                else r.get("genome_id"),
                "species_id": int(r.get("species_id") or 0),
                "generation": int(r.get("generation") or 0),
                "source_file": r.get("source_file") or "",
                "prompt": str(r.get("prompt") or ""),
                "has_embedding": str(r.get("has_embedding")).lower() == "true",
            }

            g_vec: List[float] = []
            g_ok = True
            for ax in google_axes:
                col = analysis_utils.score_column_name("google", ax)
                v = r.get(col)
                if v in (None, ""):
                    g_ok = False
                    break
                try:
                    g_vec.append(float(v))
                except (TypeError, ValueError):
                    g_ok = False
                    break
            if g_ok and len(g_vec) == len(google_axes):
                row["objective_vector"] = np.asarray(g_vec, dtype=np.float64)

            o_vec: List[float] = []
            o_ok = True
            for ax in openai_axes:
                col = analysis_utils.score_column_name("openai", ax)
                v = r.get(col)
                if v in (None, ""):
                    o_ok = False
                    break
                try:
                    o_vec.append(float(v))
                except (TypeError, ValueError):
                    o_ok = False
                    break
            if o_ok and len(o_vec) == len(openai_axes):
                row["objective_vector_openai"] = np.asarray(o_vec, dtype=np.float64)

            gid = str(r.get("genome_id") or "")
            if embeddings is not None and gid in id_to_emb_index:
                k = id_to_emb_index[gid]
                if 0 <= k < embeddings.shape[0]:
                    row["_embedding"] = np.asarray(embeddings[k], dtype=np.float64)

            rows.append(row)
    return rows


def _run_phase(name: str, fn: Callable[[], Any]) -> Tuple[str, str]:
    try:
        out = fn()
        n_keys = len(out) if isinstance(out, dict) else 0
        return "PASS", f"{n_keys} artifact keys returned"
    except Exception as exc:
        return "FAIL", f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def main() -> int:
    manifest = json.loads((RESULTS_DIR / "phase1_manifest.json").read_text(encoding="utf-8"))
    run_path = Path(manifest["run_path"])

    print("Loading unified rows...")
    rows = load_unified_rows()
    n_g = sum(1 for r in rows if "objective_vector" in r)
    n_o = sum(1 for r in rows if "objective_vector_openai" in r)
    n_e = sum(1 for r in rows if "_embedding" in r)
    print(f"  {len(rows)} rows | google={n_g} openai={n_o} embeddings={n_e}")
    if n_g < len(rows) or n_o < len(rows):
        print("WARNING: incomplete scorer coverage — some phases may degrade")

    results: Dict[str, Tuple[str, str]] = {}

    def phase2():
        stats = analysis_utils.annotate_pareto_ranks(rows)
        return analysis_utils.save_phase2_artifacts(
            rows, stats, RESULTS_DIR / "phase2", RUN_ID
        )

    def phase3():
        return analysis_utils.save_phase3_artifacts(
            rows, RESULTS_DIR / "phase3", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase4():
        return analysis_utils.save_phase4_artifacts(
            run_path, RESULTS_DIR / "phase4", RUN_ID
        )

    def phase5():
        return analysis_utils.save_phase5_artifacts(
            rows, RESULTS_DIR / "phase5", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase6():
        return analysis_utils.save_phase6_artifacts(
            rows, RESULTS_DIR / "phase6", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase7():
        return analysis_utils.save_phase7_artifacts(
            rows, RESULTS_DIR / "phase7", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase8():
        return analysis_utils.save_phase8_artifacts(
            rows, RESULTS_DIR / "phase8", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase9():
        return analysis_utils.save_phase9_artifacts(
            rows, RESULTS_DIR / "phase9", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase11():
        return analysis_utils.save_phase11_artifacts(
            rows, RESULTS_DIR / "phase11", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase12():
        return analysis_utils.save_phase12_artifacts(
            rows, RESULTS_DIR / "phase12", RUN_ID, min_topic_size=MIN_TOPIC_SIZE
        )

    def phase10():
        return analysis_utils.save_phase10_artifacts(RESULTS_DIR, RUN_ID)

    order = [
        ("phase2", phase2),
        ("phase3", phase3),
        ("phase4", phase4),
        ("phase5", phase5),
        ("phase6", phase6),
        ("phase7", phase7),
        ("phase8", phase8),
        ("phase9", phase9),
        ("phase11", phase11),
        ("phase12", phase12),
        ("phase10", phase10),
    ]

    print("\nRunning phases (2–12, excluding 0/1)...\n")
    failures = 0
    for name, fn in order:
        print(f"--- {name} ---")
        status, detail = _run_phase(name, fn)
        results[name] = (status, detail)
        print(f"  {status}: {detail.split(chr(10))[0]}")
        if status == "FAIL":
            failures += 1
            print(detail)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for name, (status, _) in results.items():
        print(f"  {name}: {status}")
    print("=" * 60)

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
