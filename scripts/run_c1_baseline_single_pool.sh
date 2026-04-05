#!/usr/bin/env bash
# Approximate C1 "ToxSearch" baseline (single semantic pool): one species worth of radius.
#
# The codebase always runs the speciation pipeline; there is no --no-speciation flag. Setting
# theta_sim and theta_merge to 1.0 (max ensemble distance) makes leader–follower assignment
# extremely permissive so the population collapses toward a single-species-style pool, which
# is closer to non-island ToxSearch than default θ values. This is still not bit-identical to a
# hypothetical no-speciation branch — see experiments/EXPERIMENT_PLAN.md.
#
# Usage (from repo root, after venv activate):
#   bash scripts/run_c1_baseline_single_pool.sh --output-dir data/outputs/c1_pool_run01 --max-total-genomes 1000
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
exec python src/main.py \
  --theta-sim 1.0 \
  --theta-merge 1.0 \
  --min-island-size 1 \
  --species-stagnation 99999 \
  "$@"
