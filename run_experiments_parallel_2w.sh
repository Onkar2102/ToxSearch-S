#!/bin/bash
# Local experiment runner: MPI parallel with exactly TWO worker ranks (plus rank-0 master).
#
# mpiexec -n 3  =>  rank 0 = master, ranks 1–2 = workers  (see src/parallel/master_worker.py)
#
# Usage (from project root):
#   bash run_experiments_parallel_2w.sh
#
# Environment overrides (same spirit as run_experiments_local.sh):
#   PYTHON=python3
#   THETA_VALUES="0.25 0.30 0.35"
#   MPI_EXTRA_OPTS="..."   optional extra args passed to mpiexec (e.g. --map-by socket)
#
# Prerequisites:
#   - Open MPI (or compatible) mpiexec on PATH
#   - mpi4py and project deps installed in the active venv
#   - .env with PERSPECTIVE_API_KEY (or keys) as required by parallel mode
#
# To run sequential + parallel from one place, either run this script after
# run_experiments_local.sh or uncomment parallel in run_experiments_local.sh and set
# MPI_RANKS=3, RUN_PARALLEL=1.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d ".spvenv" ]; then
    source .spvenv/bin/activate
fi

if [ -f ".env" ]; then
    set +u
    set -a
    source .env
    set +a
    set -u
fi

if ! command -v mpiexec >/dev/null 2>&1; then
    echo "ERROR: mpiexec not found. Install Open MPI or use the sequential script." >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/src"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
# Total MPI ranks = 1 master + 2 workers
MPI_RANKS="${MPI_RANKS:-3}"
MPI_EXTRA_OPTS="${MPI_EXTRA_OPTS:-}"

THETA_VALUES="${THETA_VALUES:-0.25 0.30 0.35}"

# Keep CLI aligned with run_experiments_local.sh for comparability.
ARGS_ARR=(
    --parallel
    --moderation-methods google
    --stagnation-limit 5
    --min-stability-gens 5
    --species-capacity 100
    --cluster0-max-capacity 1000
    --cluster0-min-cluster-size 1
    --min-island-size 3
    --species-stagnation 20
    --embedding-model all-MiniLM-L6-v2
    --embedding-dim 384
    --embedding-batch-size 64
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf
    --operators all
    --max-variants 1
    --seed-file data/prompt_100.csv
    --seed 42
    --max-total-genomes 1000
)

run_parallel() {
    local theta="${1:?run_parallel: theta required}"
    local out_tag
    out_tag="$(echo "$theta" | tr '.' 'p')"
    echo "=========================================="
    echo "Parallel (mpiexec -n ${MPI_RANKS} = 1 master + $((MPI_RANKS - 1)) workers)"
    echo "  theta_sim=${theta}  theta_merge=${theta}"
    echo "Output: data/outputs/local_${RUN_TS}_parallel2w_theta${out_tag}"
    echo "=========================================="
    # shellcheck disable=SC2086
    ( export PYTHONPATH="${SCRIPT_DIR}/src"
      exec mpiexec -n "${MPI_RANKS}" ${MPI_EXTRA_OPTS} \
        "$PYTHON" src/main.py "${ARGS_ARR[@]}" \
        --theta-sim "$theta" \
        --theta-merge "$theta" \
        --output-dir "data/outputs/local_${RUN_TS}_parallel2w_theta${out_tag}" )
}

for theta in $THETA_VALUES; do
    run_parallel "$theta"
    echo ""
done

echo "All parallel (2-worker) experiments completed!"
echo "RUN_TS=${RUN_TS}"
echo "Outputs under: data/outputs/local_${RUN_TS}_parallel2w_theta0p25|0p30|0p35"
