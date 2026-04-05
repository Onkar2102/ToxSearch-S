#!/bin/bash
# Local experiment runner: sequential then MPI parallel (1 master + 4 workers = 5 ranks).
#
# Usage:
#   From project root: bash run_experiments_local.sh
#
# Options:
#   - RUN_SEQUENTIAL=0   Skip sequential run (default: 1).
#   - RUN_PARALLEL=0     Skip parallel run (default: 1).
#   - MPI_RANKS=5        mpiexec -n value (default: 5 → 1 master + 4 workers).
#   - PYTHON=python3     Python interpreter (default: python3).
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
# Termination: --max-total-genomes only.
ARGS_ARR=(
    --moderation-methods google
    --stagnation-limit 5
    --theta-sim 0.30
    --theta-merge 0.30
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
    --max-total-genomes 160
)

run_with_python() {
    ( export PYTHONPATH="${SCRIPT_DIR}/src"; exec "$PYTHON" src/main.py "$@" )
}

run_sequential() {
    echo "=========================================="
    echo "Sequential (single process)"
    echo "Output: data/outputs/local_${RUN_TS}_sequential"
    echo "=========================================="
    run_with_python "${ARGS_ARR[@]}" \
        --output-dir "data/outputs/local_${RUN_TS}_sequential"
}

run_parallel() {
    if ! command -v mpiexec >/dev/null 2>&1; then
        echo "ERROR: mpiexec not found; install Open MPI or set RUN_PARALLEL=0." >&2
        return 1
    fi
    echo "=========================================="
    echo "Parallel (mpiexec -n ${MPI_RANKS}, --parallel)"
    echo "Output: data/outputs/local_${RUN_TS}_parallel"
    echo "=========================================="
    ( export PYTHONPATH="${SCRIPT_DIR}/src"
      exec mpiexec -n "${MPI_RANKS}" "$PYTHON" src/main.py --parallel "${ARGS_ARR[@]}" \
        --output-dir "data/outputs/local_${RUN_TS}_parallel" )
}

RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-1}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"

if [ "$RUN_SEQUENTIAL" = "1" ]; then
    run_sequential
    echo ""
fi

if [ "$RUN_PARALLEL" = "1" ]; then
    [ "$RUN_SEQUENTIAL" = "1" ] && { echo "Waiting 5 seconds before parallel run..."; sleep 5; echo ""; }
    run_parallel
    echo ""
fi

echo "All requested experiments completed!"
echo "RUN_TS=${RUN_TS}  sequential: data/outputs/local_${RUN_TS}_sequential  parallel: data/outputs/local_${RUN_TS}_parallel"
