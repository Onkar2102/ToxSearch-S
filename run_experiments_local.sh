#!/bin/bash
# Local experiment runner: sequential mode only (no MPI / multiple GPUs).
#
# Usage:
#   From project root: bash run_experiments_local.sh
#
# Options:
#   - RUN_SEQUENTIAL=0    Skip running (default: 1).
#   - PYTHON=python3      Python interpreter (default: python3).
#
# Environment:
#   - .env is loaded if present (PERSPECTIVE_API_KEY, etc.).
#   - PYTHONPATH is set to src for the run so imports and config resolve.
#
# Profiling (cProfile):
#   Add --profile to enable; profile is saved in the run's output directory as profile_main.prof.
#   Inspect with: python -m pstats <path> or snakeviz <path>.
#
# Output directory is auto-generated (data/outputs/<timestamp>) per run.
#
# Termination (sequential only):
#   Always by --max-total-genomes (no generations cap). Required.
# Research questions and full experiment matrix:
#   See experiments/RESEARCH_QUESTIONS.txt.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment if present
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
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

# ---- Sequential experiments (single process) ----
# Termination: always by --max-total-genomes (sequential has no generations cap).
# Each entry is a full command; all project parameters set explicitly.
# Profiling: add --profile to enable; profile is saved in the run's (auto-generated) output directory.
# Aligned with rc_script.sh (minus parallel/Spack). Small max-total-genomes for quick local run.
SEQUENTIAL_EXPERIMENTS=(
    "$PYTHON src/main.py \
        --moderation-methods google \
        --stagnation-limit 5 \
        --theta-sim 0.35 \
        --theta-merge 0.35 \
        --min-stability-gens 5 \
        --species-capacity 100 \
        --cluster0-max-capacity 1000 \
        --cluster0-min-cluster-size 1 \
        --min-island-size 3 \
        --species-stagnation 20 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
        --operators all \
        --max-variants 1 \
        --seed-file data/prompt.csv \
        --seed 42 \
        --batch-size 25 \
        --max-total-genomes 80 \
        --output-dir data/outputs/local_run"
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
        ( export PYTHONPATH="${SCRIPT_DIR}/src"; eval "${SEQUENTIAL_EXPERIMENTS[$i]}" ) || echo "Experiment $num failed"
        echo ""
        [ "$num" -lt "$total" ] && { echo "Waiting 5 seconds..."; sleep 5; echo ""; }
    done
}

RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-1}"
if [ "$RUN_SEQUENTIAL" = "1" ]; then
    run_sequential
fi

echo "All experiments completed!"
