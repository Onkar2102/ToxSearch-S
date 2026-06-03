#!/bin/bash
# Local experiment runner: single sequential run (default 150 total genomes).
#
# Usage:
#   From project root: bash run_experiments_local.sh
#
# Environment overrides:
#   MAX_TOTAL_GENOMES=150   Termination cap (elites + reserves + archive).
#   EVALUATOR=google          Moderation backend: google | openai.
#   NORTH_STAR_METRIC=        Optional; profile default if unset (toxicity / violence).
#   OPENAI_MODEL=             When EVALUATOR=openai (default: omni-moderation-latest).
#   THETA_SIM=0.25            Species similarity threshold.
#   THETA_MERGE=              Defaults to THETA_SIM.
#   RUN_THETA_SWEEP=0         Set to 1 to run THETA_VALUES sweep (legacy).
#   THETA_VALUES="0.25 0.30 0.35"
#   PYTHON=python3
#   MAX_ATTEMPTS=2            Retry count for transient failures only.
#
# Parallel (disabled by default):
#   RUN_PARALLEL=0, MPI_RANKS=2 — see commented block at bottom.
#
# .env is loaded for PERSPECTIVE_API_KEY / OPENAI_API_KEY etc.
# PYTHONPATH is set to src for imports and config resolution.

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
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/src"

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-150}"
EVALUATOR="${EVALUATOR:-google}"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-}"
OPENAI_MODEL="${OPENAI_MODEL:-omni-moderation-latest}"
THETA_SIM="${THETA_SIM:-0.25}"
THETA_MERGE="${THETA_MERGE:-$THETA_SIM}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"
MPI_RANKS="${MPI_RANKS:-2}"

RG_MODEL="${RG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q5_K_M.gguf}"

preflight() {
    if [ ! -f "src/main.py" ]; then
        echo "ERROR: src/main.py not found (run from project root)." >&2
        exit 1
    fi

    case "$EVALUATOR" in
        google)
            if [ -z "${PERSPECTIVE_API_KEY:-}" ] && [ -z "${PERSPECTIVE_API_KEYS:-}" ]; then
                echo "ERROR: EVALUATOR=google requires PERSPECTIVE_API_KEY (or PERSPECTIVE_API_KEYS) in .env." >&2
                exit 1
            fi
            ;;
        openai)
            if [ -z "${OPENAI_API_KEY:-}" ]; then
                echo "ERROR: EVALUATOR=openai requires OPENAI_API_KEY in .env." >&2
                exit 1
            fi
            ;;
        *)
            echo "ERROR: EVALUATOR must be google or openai (got: ${EVALUATOR})." >&2
            exit 1
            ;;
    esac

    for gguf in "$RG_MODEL" "$PG_MODEL"; do
        if [ ! -f "$gguf" ]; then
            echo "WARNING: GGUF not found: $gguf" >&2
        fi
    done
}

build_base_args() {
    ARGS_ARR=(
        --evaluator "$EVALUATOR"
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
        --rg "$RG_MODEL"
        --pg "$PG_MODEL"
        --operators all
        --max-variants 1
        --seed-file data/prompt_100.csv
        --seed 42
        --max-total-genomes "$MAX_TOTAL_GENOMES"
    )
    if [ -n "$NORTH_STAR_METRIC" ]; then
        ARGS_ARR+=(--north-star-metric "$NORTH_STAR_METRIC")
    fi
    if [ "$EVALUATOR" = "openai" ]; then
        ARGS_ARR+=(--openai-model "$OPENAI_MODEL")
    fi
}

run_with_python() {
    ( export PYTHONPATH="${SCRIPT_DIR}/src"; exec "$PYTHON" src/main.py "$@" )
}

run_until_success() {
    local attempt=1
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "--- Attempt ${attempt}/${MAX_ATTEMPTS} ---"
        if run_with_python "$@"; then
            return 0
        fi
        echo "Run failed (attempt ${attempt})." >&2
        if [ "$attempt" -ge "$MAX_ATTEMPTS" ]; then
            echo "ERROR: All ${MAX_ATTEMPTS} attempt(s) failed." >&2
            return 1
        fi
        echo "Fix the issue above, then retrying in 3s..." >&2
        sleep 3
        attempt=$((attempt + 1))
    done
    return 1
}

run_single_sequential() {
    local out_dir="data/outputs/local_${RUN_TS}_sequential_g${MAX_TOTAL_GENOMES}"
    echo "=========================================="
    echo "Sequential (single process)"
    echo "  max_total_genomes=${MAX_TOTAL_GENOMES}"
    echo "  evaluator=${EVALUATOR}"
    echo "  north_star_metric=${NORTH_STAR_METRIC:-<profile default>}"
    echo "  theta_sim=${THETA_SIM}  theta_merge=${THETA_MERGE}"
    echo "Output: ${out_dir}"
    echo "=========================================="
    run_until_success "${ARGS_ARR[@]}" \
        --theta-sim "$THETA_SIM" \
        --theta-merge "$THETA_MERGE" \
        --output-dir "$out_dir"
}

run_sequential_theta() {
    local theta="${1:?run_sequential_theta: theta required}"
    local out_tag
    out_tag="$(echo "$theta" | tr '.' 'p')"
    local out_dir="data/outputs/local_${RUN_TS}_sequential_theta${out_tag}_g${MAX_TOTAL_GENOMES}"
    echo "=========================================="
    echo "Sequential (theta sweep)  theta_sim=${theta}  theta_merge=${theta}"
    echo "  max_total_genomes=${MAX_TOTAL_GENOMES}"
    echo "Output: ${out_dir}"
    echo "=========================================="
    run_until_success "${ARGS_ARR[@]}" \
        --theta-sim "$theta" \
        --theta-merge "$theta" \
        --output-dir "$out_dir"
}

preflight
build_base_args

RUN_SEQUENTIAL="${RUN_SEQUENTIAL:-1}"
RUN_THETA_SWEEP="${RUN_THETA_SWEEP:-0}"
RUN_PARALLEL="${RUN_PARALLEL:-0}"
THETA_VALUES="${THETA_VALUES:-0.25 0.30 0.35}"

if [ "$RUN_SEQUENTIAL" = "1" ]; then
    if [ "$RUN_THETA_SWEEP" = "1" ]; then
        for theta in $THETA_VALUES; do
            run_sequential_theta "$theta"
            echo ""
        done
    else
        run_single_sequential
        echo ""
    fi
fi

# Parallel (MPI): set RUN_PARALLEL=1 to enable.
# run_parallel() { ... mpiexec -n "${MPI_RANKS}" "$PYTHON" src/main.py --parallel ... }

echo "All requested experiments completed!"
echo "RUN_TS=${RUN_TS}"
if [ "$RUN_THETA_SWEEP" = "1" ]; then
    echo "Sequential outputs: data/outputs/local_${RUN_TS}_sequential_theta*_g${MAX_TOTAL_GENOMES}"
else
    echo "Sequential output: data/outputs/local_${RUN_TS}_sequential_g${MAX_TOTAL_GENOMES}"
fi
