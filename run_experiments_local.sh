#!/bin/bash
# Local experiment runner: leader–follower + Incremental DBSCAN (default 777 genomes).
#
# Usage (from project root):
#   bash run_experiments_local.sh
#
# Run one mode only:
#   CLUSTERING_METHODS=leader_follower bash run_experiments_local.sh
#   CLUSTERING_METHODS=dbscan bash run_experiments_local.sh
#
# Notes:
#   - Figures are written under <output-dir>/figures/ after Gen 0 and each later gen.
#   - PG/RG max_new_tokens are capped in config/PGConfig.yaml and config/RGConfig.yaml
#     (256 / 512). Restart the run after changing those — a live process keeps old values.
#   - operators=all × 8B is still slow per generation; use OPERATORS=cm for faster local smokes.
#
# Environment overrides (defaults match SpeciationConfig / CLI):
#   MAX_TOTAL_GENOMES=777
#   CLUSTERING_METHODS="leader_follower dbscan"
#   EVALUATOR=google|openai
#   NORTH_STAR_METRIC=          profile default if unset
#   OPENAI_MODEL=omni-moderation-latest
#   OPERATORS=all               ie | cm | all
#   MAX_VARIANTS=1
#   SEED_FILE=data/prompt_100.csv
#   SEED=42
#   THETA_SIM=0.25
#   THETA_MERGE=0.1             must be <= THETA_SIM
#   MIN_STABILITY_GENS=5
#   SPECIES_CAPACITY=100
#   MIN_ISLAND_SIZE=2
#   SPECIES_STAGNATION=20
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   EMBEDDING_DIM=384
#   EMBEDDING_BATCH_SIZE=64
#   DISTANCE_METHOD=embedding
#   DISTANCE_ALPHA=0.7
#   DBSCAN_EPS=                 defaults to THETA_SIM when empty
#   DBSCAN_MIN_SAMPLES=2
#   INC_DBSCAN_VALIDATE_EVERY=0
#   STAGNATION_LIMIT=5
#   RG_MODEL / PG_MODEL         GGUF paths
#   PYTHON=python3
#   MAX_ATTEMPTS=2
#
# .env is loaded for API keys. PYTHONPATH is set to src.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
elif [ -d ".venv" ]; then
    # shellcheck source=/dev/null
    source .venv/bin/activate
elif [ -d ".spvenv" ]; then
    # shellcheck source=/dev/null
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

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-210}"
CLUSTERING_METHODS="${CLUSTERING_METHODS:-leader_follower dbscan}"
EVALUATOR="${EVALUATOR:-google}"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-}"
OPENAI_MODEL="${OPENAI_MODEL:-omni-moderation-latest}"
OPERATORS="${OPERATORS:-all}"
MAX_VARIANTS="${MAX_VARIANTS:-1}"
SEED_FILE="${SEED_FILE:-data/prompt_100.csv}"
SEED="${SEED:-42}"
STAGNATION_LIMIT="${STAGNATION_LIMIT:-5}"

THETA_SIM="${THETA_SIM:-0.25}"
THETA_MERGE="${THETA_MERGE:-0.25}"
MIN_STABILITY_GENS="${MIN_STABILITY_GENS:-5}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-10}"
MIN_ISLAND_SIZE="${MIN_ISLAND_SIZE:-1}"
SPECIES_STAGNATION="${SPECIES_STAGNATION:-5}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

DISTANCE_METHOD="${DISTANCE_METHOD:-embedding}"
DISTANCE_ALPHA="${DISTANCE_ALPHA:-0.7}"

DBSCAN_EPS="${DBSCAN_EPS:-}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-1}"
INC_DBSCAN_VALIDATE_EVERY="${INC_DBSCAN_VALIDATE_EVERY:-0}"

RG_MODEL="${RG_MODEL:-models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-Q4_K_S.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-Q4_K_S.gguf}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"

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

    if [ ! -f "$SEED_FILE" ]; then
        echo "ERROR: seed file not found: $SEED_FILE" >&2
        exit 1
    fi
}

# Shared args for both speciation modes (all operators + full speciation CLI).
build_common_args() {
    ARGS_ARR=(
        --evaluator "$EVALUATOR"
        --operators "$OPERATORS"
        --max-variants "$MAX_VARIANTS"
        --stagnation-limit "$STAGNATION_LIMIT"
        --seed-file "$SEED_FILE"
        --seed "$SEED"
        --max-total-genomes "$MAX_TOTAL_GENOMES"
        --rg "$RG_MODEL"
        --pg "$PG_MODEL"
        --theta-sim "$THETA_SIM"
        --theta-merge "$THETA_MERGE"
        --min-stability-gens "$MIN_STABILITY_GENS"
        --species-capacity "$SPECIES_CAPACITY"
        --min-island-size "$MIN_ISLAND_SIZE"
        --species-stagnation "$SPECIES_STAGNATION"
        --embedding-model "$EMBEDDING_MODEL"
        --embedding-dim "$EMBEDDING_DIM"
        --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
        --distance-method "$DISTANCE_METHOD"
        --distance-alpha "$DISTANCE_ALPHA"
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

run_clustering_method() {
    local method="${1:?clustering method required}"
    local tag out_dir
    case "$method" in
        leader_follower|lf)
            method="leader_follower"
            tag="lf"
            ;;
        dbscan|inc_dbscan|incdbscan)
            method="dbscan"
            tag="inc_dbscan"
            ;;
        *)
            echo "ERROR: unknown clustering method: $method (use leader_follower or dbscan)" >&2
            return 1
            ;;
    esac

    out_dir="data/outputs/local_${RUN_TS}_${tag}_g${MAX_TOTAL_GENOMES}"

    echo "=========================================="
    echo "Speciation: ${method}"
    echo "  max_total_genomes=${MAX_TOTAL_GENOMES}"
    echo "  operators=${OPERATORS}"
    echo "  evaluator=${EVALUATOR}"
    echo "  north_star_metric=${NORTH_STAR_METRIC:-<profile default>}"
    echo "  theta_sim=${THETA_SIM}  theta_merge=${THETA_MERGE}"
    echo "  distance_method=${DISTANCE_METHOD}  distance_alpha=${DISTANCE_ALPHA}"
    if [ "$method" = "dbscan" ]; then
        echo "  dbscan_eps=${DBSCAN_EPS:-$THETA_SIM}  dbscan_min_samples=${DBSCAN_MIN_SAMPLES}"
        echo "  inc_dbscan_validate_every=${INC_DBSCAN_VALIDATE_EVERY}"
    fi
    echo "Output: ${out_dir}"
    echo "=========================================="

    local method_args=(
        "${ARGS_ARR[@]}"
        --clustering-method "$method"
        --output-dir "$out_dir"
    )

    if [ "$method" = "dbscan" ]; then
        method_args+=(--dbscan-min-samples "$DBSCAN_MIN_SAMPLES")
        method_args+=(--inc-dbscan-validate-every "$INC_DBSCAN_VALIDATE_EVERY")
        if [ -n "$DBSCAN_EPS" ]; then
            method_args+=(--dbscan-eps "$DBSCAN_EPS")
        else
            method_args+=(--dbscan-eps "$THETA_SIM")
        fi
    fi

    run_until_success "${method_args[@]}"
}

preflight
build_common_args

echo "RUN_TS=${RUN_TS}"
echo "Will run clustering methods: ${CLUSTERING_METHODS}"
echo ""

for method in $CLUSTERING_METHODS; do
    run_clustering_method "$method"
    echo ""
done

echo "All requested experiments completed!"
echo "RUN_TS=${RUN_TS}"
echo "Outputs under: data/outputs/local_${RUN_TS}_*_g${MAX_TOTAL_GENOMES}"
