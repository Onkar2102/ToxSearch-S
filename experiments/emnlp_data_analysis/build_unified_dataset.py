#!/usr/bin/env python3
"""Build the unified Google + OpenAI table the rest of the analysis reads.

The workflow is split so slow OpenAI scoring never forces rebuilding from the
raw run JSON:

  1. ``build``        merge run JSON → CSV + embeddings (Google ``f_*`` only).
  2. ``score-openai`` read the CSV, score via API or local cache, write
                      ``oai_*`` columns back into the same CSV.

Requires the smoke gate (``results/gate0_smoke.json``) to have passed.

Examples
--------
  python build_unified_dataset.py build
  python build_unified_dataset.py score-openai
  python build_unified_dataset.py score-openai --no-fetch   # cache only → CSV
  python build_unified_dataset.py all                       # build + score-openai
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from analysis_utils import (  # noqa: E402
    DEFAULT_PRIMARY_RUN,
    RESULTS_DIR,
    build_unified_google_table,
    run_id_from_path,
    rows_from_unified_csv,
    score_openai_and_update_unified_csv,
    unified_artifacts_dir,
    unified_csv_path,
    write_phase1_manifest,
)


def _require_gate0(results_dir: Path) -> None:
    gate0_path = results_dir / "gate0_smoke.json"
    if not gate0_path.is_file():
        raise SystemExit(f"Run smoke_gate.py first — missing {gate0_path}")
    gate0 = json.loads(gate0_path.read_text(encoding="utf-8"))
    if not gate0.get("pass"):
        raise SystemExit(f"Gate 0 FAIL — fix run or thresholds ({gate0_path})")


def cmd_build(args: argparse.Namespace) -> int:
    _require_gate0(args.results_dir)
    rows, stats, paths = build_unified_google_table(
        args.run_path, results_dir=args.results_dir
    )
    run_id = stats["run_id"]
    manifest_path = write_phase1_manifest(
        args.results_dir,
        run_id=run_id,
        run_path=args.run_path,
        rows=rows,
        artifact_paths=paths,
        stats=stats,
        phase="google_only",
        fetch_openai_missing=False,
    )
    print(f"Built unified table |G|={len(rows):,}")
    print(f"  Google scores: {stats['n_with_google_objectives']:,}")
    print(f"  Embeddings: {stats.get('n_with_embedding', 0):,}")
    print(f"  CSV: {paths['csv']}")
    print(f"Manifest (google_only): {manifest_path}")
    print("Next: python build_unified_dataset.py score-openai")
    return 0


def cmd_score_openai(args: argparse.Namespace) -> int:
    _require_gate0(args.results_dir)
    run_id = run_id_from_path(args.run_path)
    csv_path = unified_csv_path(unified_artifacts_dir(args.results_dir), run_id)
    if not csv_path.is_file():
        raise SystemExit(f"Run build first — missing {csv_path}")

    stats = score_openai_and_update_unified_csv(
        args.run_path,
        results_dir=args.results_dir,
        fetch_openai_missing=not args.no_fetch,
        openai_model=args.openai_model,
        openai_request_delay_sec=args.delay,
        openai_failure_delay_step_sec=args.failure_step,
    )
    rows = rows_from_unified_csv(
        csv_path,
        unified_dir=unified_artifacts_dir(args.results_dir),
        load_embeddings=True,
    )
    manifest_path = args.results_dir / "phase1_manifest.json"
    prior: dict = {}
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = prior.get("artifacts") or {
        "csv": str(csv_path),
        "embeddings": str(unified_artifacts_dir(args.results_dir) / f"{run_id}_embeddings.npy"),
        "genome_ids": str(unified_artifacts_dir(args.results_dir) / f"{run_id}_genome_ids.json"),
        "stats": str(unified_artifacts_dir(args.results_dir) / f"{run_id}_stats.json"),
    }
    write_phase1_manifest(
        args.results_dir,
        run_id=run_id,
        run_path=args.run_path,
        rows=rows,
        artifact_paths=artifact_paths,
        stats=stats,
        phase="complete",
        fetch_openai_missing=not args.no_fetch,
    )
    n_g = len(rows)
    n_o = stats.get("n_with_openai_objectives", sum(1 for r in rows if "objective_vector_openai" in r))
    print(f"Updated CSV: {csv_path}")
    print(f"  OpenAI scores: {n_o:,} / {n_g:,}")
    if not args.no_fetch:
        print(f"  API fetches this run: {stats.get('n_openai_fetched', 0):,}")
    print(f"Manifest (complete): {manifest_path}")
    if n_o < n_g:
        print("WARNING: OpenAI incomplete — re-run score-openai to resume (uses cache)")
        return 1
    return 0


def _add_openai_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-fetch",
        action="store_true",
        help="Apply disk cache only; no API calls",
    )
    parser.add_argument("--openai-model", default="omni-moderation-latest")
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Sleep after each API attempt (s)",
    )
    parser.add_argument(
        "--failure-step",
        type=float,
        default=0.5,
        help="Add to sleep after each failed API attempt (s)",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the unified Google + OpenAI scoring table",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
commands:
  build         Google + embeddings → unified CSV (fast, no OpenAI)
  score-openai  OpenAI scores → update the same CSV (slow; uses cache)
  all           build, then score-openai

Typical first-time run:
  python build_unified_dataset.py build
  python build_unified_dataset.py score-openai

If OpenAI cache is already full:
  python build_unified_dataset.py score-openai --no-fetch
""",
    )
    parser.add_argument(
        "--run-path",
        type=Path,
        default=DEFAULT_PRIMARY_RUN,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
    )
    sub = parser.add_subparsers(
        dest="command",
        metavar="COMMAND",
        title="command (required)",
    )

    sub.add_parser("build", help="Google + embeddings → unified CSV (no OpenAI)")

    p_score = sub.add_parser("score-openai", help="OpenAI scoring → update unified CSV")
    _add_openai_flags(p_score)

    p_all = sub.add_parser("all", help="build then score-openai")
    _add_openai_flags(p_all)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        print(
            "\nerror: COMMAND is required — use build, score-openai, or all\n",
            file=sys.stderr,
        )
        return 2
    if args.command == "build":
        return cmd_build(args)
    if args.command == "score-openai":
        return cmd_score_openai(args)
    if args.command == "all":
        rc = cmd_build(args)
        if rc != 0:
            return rc
        return cmd_score_openai(args)
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
