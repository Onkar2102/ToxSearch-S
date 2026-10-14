#!/bin/bash
# Local experiment runner: MPI parallel with exactly TWO worker ranks (plus rank-0 master).
#
# mpiexec -n 3  =>  rank 0 = master, ranks 1–2 = workers  (see src/parallel/master_worker.py)
#
# Usage (from project root):
#   bash run_experiments_parallel_2w.sh
#
# Ablation examples (same knobs as run_experiments_local.sh):
#   DISTANCE_METHOD=nli CLUSTERING_METHODS=dbscan bash run_experiments_parallel_2w.sh
#   CLUSTERING_METHODS=leader_follower THETA_VALUES="0.25" bash run_experiments_parallel_2w.sh
#
# Environment overrides (defaults match SpeciationConfig / run_experiments_local.sh):
#   MAX_TOTAL_GENOMES=1000
#   CLUSTERING_METHODS="leader_follower dbscan"
#   EVALUATOR=google|openai
#   NORTH_STAR_METRIC=
#   OPENAI_MODEL=omni-moderation-latest
#   OPERATORS=all               ie | cm | all
#   MAX_VARIANTS=1
#   SEED_FILE=data/prompt.csv
#   SEED=42
#   THETA_SIM=0.25              used when THETA_VALUES is empty
#   THETA_MERGE=0.1             must be <= each theta_sim
#   THETA_VALUES=               optional sweep; if set, overrides THETA_SIM per run
#   MIN_STABILITY_GENS=5
#   SPECIES_CAPACITY=100
#   SPECIES_STAGNATION=20
#   EMBEDDING_MODEL=all-MiniLM-L6-v2
#   EMBEDDING_DIM=384
#   EMBEDDING_BATCH_SIZE=64
#   DISTANCE_METHOD=embedding
#   NLI_MODEL=cross-encoder/nli-deberta-v3-small
#   DISTANCE_ALPHA=0.7
#   DBSCAN_EPS=                 defaults to theta_sim when empty
#   DBSCAN_MIN_SAMPLES=2
#   INC_DBSCAN_VALIDATE_EVERY=0
#   STAGNATION_LIMIT=5
#   RG_MODEL / PG_MODEL         default Llama-3.2-1B IQ4_XS
#   MPI_RANKS=3
#   MPI_EXTRA_OPTS=
#   PYTHON=python3
#   MAX_ATTEMPTS=2
#
# Prerequisites:
#   - Open MPI (or compatible) mpiexec on PATH
#   - mpi4py and project deps installed in the active venv
#   - .env with API keys for EVALUATOR
#
# Setup mirrors cluster rc_script.sh (portable bits): venv → isolate Python →
# threading/ML quiet flags → .env keys → PYTHONPATH=src. For multi-GPU cluster
# jobs with Spack/CUDA/srun, use rc_script.sh instead.
#
# Optional: SKIP_CLI_CHECK=1 to skip the --help flag smoke check.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1) Activate venv
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

# 2) Python isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME || true

# 3) Threading + quiet ML stacks
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-4}}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-3}"
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="${KERAS_BACKEND:-torch}"

# 4) API keys from .env
if [ -f ".env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi

if [ -n "${PERSPECTIVE_API_KEY:-}" ]; then
    export PERSPECTIVE_API_KEY
fi

if ! command -v mpiexec >/dev/null 2>&1; then
    echo "ERROR: mpiexec not found. Install Open MPI or use run_experiments_local.sh." >&2
    exit 1
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/src"

# Total MPI ranks = 1 master + 2 workers
MPI_RANKS="${MPI_RANKS:-3}"
MPI_EXTRA_OPTS="${MPI_EXTRA_OPTS:-}"

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-1000}"
CLUSTERING_METHODS="${CLUSTERING_METHODS:-leader_follower dbscan}"
EVALUATOR="${EVALUATOR:-google}"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-}"
OPENAI_MODEL="${OPENAI_MODEL:-omni-moderation-latest}"
OPERATORS="${OPERATORS:-all}"
MAX_VARIANTS="${MAX_VARIANTS:-1}"
SEED_FILE="${SEED_FILE:-data/prompt.csv}"
SEED="${SEED:-42}"
STAGNATION_LIMIT="${STAGNATION_LIMIT:-5}"

THETA_SIM="${THETA_SIM:-0.25}"
THETA_MERGE="${THETA_MERGE:-0.1}"
# Optional sweep; empty => single run at THETA_SIM
THETA_VALUES="${THETA_VALUES:-}"
MIN_STABILITY_GENS="${MIN_STABILITY_GENS:-5}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-100}"
SPECIES_STAGNATION="${SPECIES_STAGNATION:-20}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

DISTANCE_METHOD="${DISTANCE_METHOD:-embedding}"
DISTANCE_ALPHA="${DISTANCE_ALPHA:-0.7}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"

DBSCAN_EPS="${DBSCAN_EPS:-}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-2}"
INC_DBSCAN_VALIDATE_EVERY="${INC_DBSCAN_VALIDATE_EVERY:-0}"

RG_MODEL="${RG_MODEL:-models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-IQ4_XS.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-IQ4_XS.gguf}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-2}"

preflight() {
    if [ ! -f "src/main.py" ]; then
        echo "ERROR: src/main.py not found (run from project root)." >&2
        exit 1
    fi

    if [ "${SKIP_CLI_CHECK:-0}" != "1" ]; then
        "$PYTHON" src/main.py --help | grep -E "theta-sim|embedding-model|species-capacity|distance-method|clustering-method|max-total-genomes|evaluator" >/dev/null \
            || { echo "ERROR: main.py CLI missing expected flags (set SKIP_CLI_CHECK=1 to bypass)." >&2; exit 1; }
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

    if [ "$MPI_RANKS" -lt 2 ]; then
        echo "ERROR: MPI_RANKS must be >= 2 (got ${MPI_RANKS})." >&2
        exit 1
    fi
}

build_common_args() {
    ARGS_ARR=(
        --parallel
        --evaluator "$EVALUATOR"
        --operators "$OPERATORS"
        --max-variants "$MAX_VARIANTS"
        --stagnation-limit "$STAGNATION_LIMIT"
        --seed-file "$SEED_FILE"
        --seed "$SEED"
        --max-total-genomes "$MAX_TOTAL_GENOMES"
        --rg "$RG_MODEL"
        --pg "$PG_MODEL"
        --min-stability-gens "$MIN_STABILITY_GENS"
        --species-capacity "$SPECIES_CAPACITY"
        --species-stagnation "$SPECIES_STAGNATION"
        --embedding-model "$EMBEDDING_MODEL"
        --embedding-dim "$EMBEDDING_DIM"
        --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
        --distance-method "$DISTANCE_METHOD"
        --distance-alpha "$DISTANCE_ALPHA"
        --nli-model "$NLI_MODEL"
    )
    if [ -n "$NORTH_STAR_METRIC" ]; then
        ARGS_ARR+=(--north-star-metric "$NORTH_STAR_METRIC")
    fi
    if [ "$EVALUATOR" = "openai" ]; then
        ARGS_ARR+=(--openai-model "$OPENAI_MODEL")
    fi
}

run_with_mpi() {
    # shellcheck disable=SC2086
    ( export PYTHONPATH="${SCRIPT_DIR}/src"
      exec mpiexec -n "${MPI_RANKS}" ${MPI_EXTRA_OPTS} \
        "$PYTHON" src/main.py "$@" )
}

run_until_success() {
    local attempt=1
    while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
        echo "--- Attempt ${attempt}/${MAX_ATTEMPTS} ---"
        if run_with_mpi "$@"; then
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

run_one() {
    local method="${1:?clustering method required}"
    local theta="${2:?theta_sim required}"
    local tag out_dir merge_theta out_tag

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

    merge_theta="$THETA_MERGE"
    awk -v m="$merge_theta" -v s="$theta" 'BEGIN { if (m > s) exit 1 }' || {
        echo "ERROR: THETA_MERGE (${merge_theta}) must be <= theta_sim (${theta})." >&2
        return 1
    }

    out_tag="$(echo "$theta" | tr '.' 'p')"
    out_dir="data/outputs/local_${RUN_TS}_parallel2w_${tag}_theta${out_tag}_g${MAX_TOTAL_GENOMES}"

    echo "=========================================="
    echo "Parallel speciation: ${method}"
    echo "  mpiexec -n ${MPI_RANKS} (1 master + $((MPI_RANKS - 1)) workers)"
    echo "  max_total_genomes=${MAX_TOTAL_GENOMES}"
    echo "  operators=${OPERATORS}"
    echo "  evaluator=${EVALUATOR}"
    echo "  north_star_metric=${NORTH_STAR_METRIC:-<profile default>}"
    echo "  theta_sim=${theta}  theta_merge=${merge_theta}"
    echo "  distance_method=${DISTANCE_METHOD}  distance_alpha=${DISTANCE_ALPHA}  nli_model=${NLI_MODEL}"
    if [ "$method" = "dbscan" ]; then
        echo "  dbscan_eps=${DBSCAN_EPS:-$theta}  dbscan_min_samples=${DBSCAN_MIN_SAMPLES}"
        echo "  inc_dbscan_validate_every=${INC_DBSCAN_VALIDATE_EVERY}"
    fi
    echo "Output: ${out_dir}"
    echo "=========================================="

    local method_args=(
        "${ARGS_ARR[@]}"
        --clustering-method "$method"
        --theta-sim "$theta"
        --theta-merge "$merge_theta"
        --output-dir "$out_dir"
    )

    if [ "$method" = "dbscan" ]; then
        method_args+=(--dbscan-min-samples "$DBSCAN_MIN_SAMPLES")
        method_args+=(--inc-dbscan-validate-every "$INC_DBSCAN_VALIDATE_EVERY")
        if [ -n "$DBSCAN_EPS" ]; then
            method_args+=(--dbscan-eps "$DBSCAN_EPS")
        else
            method_args+=(--dbscan-eps "$theta")
        fi
    fi

    run_until_success "${method_args[@]}"
}

preflight
build_common_args

if [ -z "$THETA_VALUES" ]; then
    THETA_VALUES="$THETA_SIM"
fi

echo "RUN_TS=${RUN_TS}"
echo "Will run clustering methods: ${CLUSTERING_METHODS}"
echo "Will run theta_sim values: ${THETA_VALUES}"
echo ""

for method in $CLUSTERING_METHODS; do
    for theta in $THETA_VALUES; do
        run_one "$method" "$theta"
        echo ""
    done
done

echo "All parallel (2-worker) experiments completed!"
echo "RUN_TS=${RUN_TS}"
echo "Outputs under: data/outputs/local_${RUN_TS}_parallel2w_*_g${MAX_TOTAL_GENOMES}"
