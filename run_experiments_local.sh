#!/bin/bash
# Multiple Experiment Runner
#
# Runs experiments from the project root. Supports:
#   - Sequential mode: single process (no MPI)
#   - Parallel mode: MPI master + workers (mpiexec)
#
# Usage:
#   1. Add experiments to SEQUENTIAL_EXPERIMENTS and/or PARALLEL_EXPERIMENTS
#   2. Run: bash run_experiments_local.sh
#
# For parallel runs:
#   - Requires MPI (e.g. openmpi/mpiexec). Install: brew install open-mpi
#   - PERSPECTIVE_API_KEY is read from .env if present, or set it in the script/env
#   - Logs: one file per rank (e.g. ..._master.log, ..._worker1.log)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Load .env for API keys (used by parallel runs)
# Temporarily relax nounset (-u) because .env may contain $vars in strings
if [ -f ".env" ]; then
    set +u
    set -a
    source .env
    set +a
    set -u
fi

PYTHON="${PYTHON:-python3}"

# ---- Sequential experiments (single process, no MPI) ----
SEQUENTIAL_EXPERIMENTS=(
    "$PYTHON src/main.py \
        --generations 50 \
        --threshold 0.99 \
        --moderation-methods google \
        --stagnation-limit 5 \
        --theta-sim 0.25 \
        --theta-merge 0.25 \
        --species-capacity 7 \
        --cluster0-max-capacity 20 \
        --cluster0-min-cluster-size 1 \
        --min-island-size 3 \
        --species-stagnation 4 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
        --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
        --operators all \
        --max-variants 1 \
        --seed-file data/prompt.csv"
)

# ---- Parallel experiments (MPI: 1 master + N workers) ----
# Uses PYTHONPATH=src so imports and config paths resolve from project root.
# Adjust -n to change number of processes (1 master + (n-1) workers).
PARALLEL_EXPERIMENTS=(
    "mpiexec -n 3 env PYTHONPATH=src $PYTHON src/main.py \
        --parallel \
        --batch-size 20 \
        --generations 10 \
        --moderation-methods google \
        --theta-sim 0.25 \
        --theta-merge 0.25 \
        --species-capacity 7 \
        --cluster0-max-capacity 20 \
        --cluster0-min-cluster-size 1 \
        --min-island-size 3 \
        --species-stagnation 4 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
        --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
        --operators all \
        --seed-file data/prompt.csv"
)

run_sequential() {
    local total="${#SEQUENTIAL_EXPERIMENTS[@]}"
    [ "$total" -eq 0 ] && return 0
    echo "Running $total sequential experiment(s)..."
    for i in "${!SEQUENTIAL_EXPERIMENTS[@]}"; do
        local num=$((i + 1))
        echo "=========================================="
        echo "Sequential experiment $num/$total"
        echo "=========================================="
        echo "Command: ${SEQUENTIAL_EXPERIMENTS[$i]}"
        echo ""
        bash -lc "${SEQUENTIAL_EXPERIMENTS[$i]}" || echo "Experiment $num failed"
        echo ""
        [ "$num" -lt "$total" ] && { echo "Waiting 5 seconds..."; sleep 5; echo ""; }
    done
}

run_parallel() {
    local total="${#PARALLEL_EXPERIMENTS[@]}"
    [ "$total" -eq 0 ] && return 0
    echo "Running $total parallel (MPI) experiment(s)..."
    for i in "${!PARALLEL_EXPERIMENTS[@]}"; do
        local num=$((i + 1))
        echo "=========================================="
        echo "Parallel experiment $num/$total"
        echo "=========================================="
        echo "Command: ${PARALLEL_EXPERIMENTS[$i]}"
        echo ""
        bash -lc "${PARALLEL_EXPERIMENTS[$i]}" || echo "Experiment $num failed"
        echo ""
        [ "$num" -lt "$total" ] && { echo "Waiting 5 seconds..."; sleep 5; echo ""; }
    done
}

# Toggle which experiment types to run (0 = skip, 1 = run)
RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-1}"

if [ "$RUN_SEQUENTIAL" = "1" ]; then
    run_sequential
fi
if [ "$RUN_PARALLEL" = "1" ]; then
    run_parallel
fi

echo "All experiments completed!"
