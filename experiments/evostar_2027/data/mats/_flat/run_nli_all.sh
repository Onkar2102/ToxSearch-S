#!/usr/bin/env bash
# Checkpointed NLI for B2000 then B3139; rematerialize per-batch slices after each stage.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src:.
PY="${ROOT}/venv/bin/python"
POOL=experiments/evostar_2027/data/pool
FLAT=experiments/evostar_2027/data/mats/_flat
MAT=experiments/evostar_2027/rq1/scripts/p1_materialize_data.py
P1C=experiments/evostar_2027/rq1/scripts/p1c_d_negE_and_d_N_mats.py

echo "=== p1c N=2000 (covers B1000 via slice) ==="
"$PY" "$P1C" --p0-dir "$POOL" --mats-dir "$FLAT" \
  --max-n 2000 --which both --checkpoint-every 10 --batch-size 256

echo "=== rematerialize after N=2000 ==="
"$PY" "$MAT"

echo "=== p1c N=3139 (full pool; --allow-large-n) ==="
"$PY" "$P1C" --p0-dir "$POOL" --mats-dir "$FLAT" \
  --max-n 3139 --allow-large-n --which both --checkpoint-every 10 --batch-size 256

echo "=== rematerialize after N=3139 ==="
"$PY" "$MAT"
echo "DONE"
