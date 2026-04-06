#!/bin/bash
# Local experiment runner: sequential sweeps over theta_sim (and matching theta_merge).
# Parallel (MPI) block is commented out below; uncomment to run mpiexec after sequential.
#
# Usage:
#   From project root: bash run_experiments_local.sh
#
# Options:
#   - RUN_SEQUENTIAL=0   Skip sequential runs (default: 1).
#   - PYTHON=python3     Python interpreter (default: python3).
#   - THETA_VALUES="0.25 0.30 0.35"  Override similarity sweep (space-separated).
#
# Parallel (disabled by default):
#   - RUN_PARALLEL=0     Skip parallel run (default: 0).
#   - MPI_RANKS=2        mpiexec -n value (default: 2 → 1 master + 1 worker).
#
# Environment:
#   - .env is loaded if present (PERSPECTIVE_API_KEY, etc.).
#   - PYTHONPATH is set to src for the run so imports and config resolve.
#
# Profiling (cProfile):
#   Add --profile to enable; profile is saved in the run's output directory as profile_main.prof.
#   Inspect with: python -m pstats <path> or snakeviz <path>.
#
# Parallel requires: Open MPI or compatible `mpiexec` on PATH.
#
# Research questions and full experiment matrix:
#   See experiments/EXPERIMENT_PLAN.md.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d ".spvenv" ]; then
    source .spvenv/bin/activate
fi

# Load .env for API keys (e.g. PERSPECTIVE_API_KEY)
if [ -f ".env" ]; then
    set +u
    set -a
    source .env
    set +a
    set -u
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/src"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MPI_RANKS="${MPI_RANKS:-2}"

# Shared ToxSearch-S CLI (sequential and parallel use the same knobs for comparability).
# theta_sim / theta_merge are passed per run (see THETA_VALUES).
# Termination: --max-total-genomes only.
ARGS_ARR=(
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
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
    --pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
    --operators all
    --max-variants 1
    --seed-file data/prompt_100.csv
    --seed 42
    --max-total-genomes 100
)

run_with_python() {
    ( export PYTHONPATH="${SCRIPT_DIR}/src"; exec "$PYTHON" src/main.py "$@" )
}

# theta_sim sweep (similarity); theta_merge set to the same value each run.
THETA_VALUES="${THETA_VALUES:-0.25 0.30 0.35}"

run_sequential() {
    local theta="${1:?run_sequential: theta required}"
    local out_tag
    out_tag="$(echo "$theta" | tr '.' 'p')"
    echo "=========================================="
    echo "Sequential (single process)  theta_sim=${theta}  theta_merge=${theta}"
    echo "Output: data/outputs/local_${RUN_TS}_sequential_theta${out_tag}"
    echo "=========================================="
    run_with_python "${ARGS_ARR[@]}" \
        --theta-sim "$theta" \
        --theta-merge "$theta" \
        --output-dir "data/outputs/local_${RUN_TS}_sequential_theta${out_tag}"
}

# Parallel (MPI): uncomment function body and the RUN_PARALLEL block below to enable.
# run_parallel() {
#     if ! command -v mpiexec >/dev/null 2>&1; then
#         echo "ERROR: mpiexec not found; install Open MPI or set RUN_PARALLEL=0." >&2
#         return 1
#     fi
#     local theta="${1:?run_parallel: theta required}"
#     local out_tag
#     out_tag="$(echo "$theta" | tr '.' 'p')"
#     echo "=========================================="
#     echo "Parallel (mpiexec -n ${MPI_RANKS}, --parallel)  theta_sim=${theta}"
#     echo "Output: data/outputs/local_${RUN_TS}_parallel_theta${out_tag}"
#     echo "=========================================="
#     ( export PYTHONPATH="${SCRIPT_DIR}/src"
#       exec mpiexec -n "${MPI_RANKS}" "$PYTHON" src/main.py --parallel "${ARGS_ARR[@]}" \
#         --theta-sim "$theta" --theta-merge "$theta" \
#         --output-dir "data/outputs/local_${RUN_TS}_parallel_theta${out_tag}" )
# }

RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-1}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"

if [ "$RUN_SEQUENTIAL" = "1" ]; then
    for theta in $THETA_VALUES; do
        run_sequential "$theta"
        echo ""
    done
fi

# if [ "$RUN_PARALLEL" = "1" ]; then
#     if ! command -v mpiexec >/dev/null 2>&1; then
#         echo "ERROR: mpiexec not found; set RUN_PARALLEL=0 or install Open MPI." >&2
#         exit 1
#     fi
#     for theta in $THETA_VALUES; do
#         [ "$RUN_SEQUENTIAL" = "1" ] && { echo "Waiting 5 seconds before parallel theta=${theta}..."; sleep 5; echo ""; }
#         run_parallel "$theta"
#         echo ""
#     done
# fi

echo "All requested experiments completed!"
echo "RUN_TS=${RUN_TS}"
echo "Sequential outputs: data/outputs/local_${RUN_TS}_sequential_theta0p25|0p30|0p35"
