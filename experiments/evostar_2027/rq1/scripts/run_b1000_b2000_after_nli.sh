#!/usr/bin/env bash
# Wait for N=2000 NLI mats, then materialize + RQ1 grids for B1000/B2000 + LaTeX tables.
# Safe to re-run: skips completed stages.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src:${ROOT}"
export PYTHONUNBUFFERED=1
PY="${ROOT}/venv/bin/python"
FLAT="${ROOT}/experiments/evostar_2027/data/mats/_flat"
DATA="${ROOT}/experiments/evostar_2027/data"
RQ1="${ROOT}/experiments/evostar_2027/rq1"
NOTES="${ROOT}/notes/LaTeX2e+Proceedings+Templates+download"
LOG="${FLAT}/post_nli_b1k_b2k.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

nli_complete() {
  local label="$1"
  local prog="${FLAT}/${label}_2000.progress.json"
  [[ -f "$prog" ]] || return 1
  "$PY" -c "import json,sys; m=json.load(open(sys.argv[1])); sys.exit(0 if m.get('complete') else 1)" "$prog"
}

log "=== post-NLI orchestrator starting (pid $$) ==="
log "Waiting for d_N_2000 + d_negE_2000 complete…"

while true; do
  if nli_complete d_N && nli_complete d_negE; then
    log "NLI @2000 complete."
    break
  fi
  dn="?"
  de="?"
  [[ -f "${FLAT}/d_N_2000.progress.json" ]] && dn=$("$PY" -c "import json; print(json.load(open('${FLAT}/d_N_2000.progress.json')).get('next_i','?'))")
  [[ -f "${FLAT}/d_negE_2000.progress.json" ]] && de=$("$PY" -c "import json; print(json.load(open('${FLAT}/d_negE_2000.progress.json')).get('next_i','?'))")
  log "  still waiting… d_N next_i=${dn}  d_negE next_i=${de}"
  sleep 120
done

log "===== STAGE: materialize (attach NLI) ====="
"$PY" experiments/evostar_2027/rq1/scripts/p1_materialize_data.py --data-dir "$DATA"

run_batch() {
  local N="$1"
  local BID="B${N}"
  local MATS="${DATA}/mats/${BID}"
  local SW="${RQ1}/sweeps/${BID}"
  local P0="${DATA}/batches/${BID}/p0.pkl"
  mkdir -p "$SW"

  if [[ ! -f "${MATS}/d_N.npy" || ! -f "${MATS}/d_negE.npy" ]]; then
    log "ERROR: missing NLI mats under ${MATS}"
    exit 1
  fi

  log "===== STAGE: p1d hparam ${BID} ====="
  "$PY" experiments/evostar_2027/rq1/scripts/p1d_hparam_sweep.py \
    --mats-dir "$MATS" \
    --sweeps-dir "$SW" \
    --prefer-n "$N" \
    --p0 "$P0" \
    --reference-radius 0.1

  log "===== STAGE: threshold grid ${BID} ====="
  "$PY" experiments/evostar_2027/rq1/scripts/p1d_threshold_grid.py \
    --mats-dir "$MATS" \
    --p0 "$P0" \
    --sweeps-dir "$SW" \
    --t-min 0.1 --t-max 0.5 --t-step 0.05

  log "===== STAGE: coherence ${BID} ====="
  "$PY" experiments/evostar_2027/rq1/scripts/p1d_cluster_coherence.py \
    --species-dir "${SW}/species" \
    --d-e "${MATS}/d_E.npy" \
    --out "${SW}/coherence.csv"

  log "===== STAGE: LaTeX tables ${BID} ====="
  "$PY" experiments/evostar_2027/rq1/scripts/p1d_generate_rq1_tables.py \
    --batch "$BID" \
    --sweeps-dir "$SW" \
    --out "${NOTES}/rq1_${BID,,}_tables.tex" \
    --n "$N"
}

run_batch 1000
run_batch 2000

marker_path="${RQ1}/sweeps/CROSS_BATCH_READY.json"
"$PY" - <<PY | tee -a "$LOG"
import json
from pathlib import Path
marker = Path("${marker_path}")
marker.write_text(json.dumps({
    "batches": ["B500", "B1000", "B2000"],
    "ready": True,
}, indent=2))
print(f"Wrote {marker}")
PY

log "===== STAGE: finalize notebook + paper + PDF ====="
"$PY" experiments/evostar_2027/rq1/scripts/finalize_cross_batch_rq1.py | tee -a "$LOG"

log "=== ALL STAGES DONE ==="
log "DONE"
