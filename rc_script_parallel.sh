#!/bin/bash -l
#
# SLURM MPI-PARALLEL run script for ToxSearch-S (srun + --parallel).
# Submit from project root: sbatch rc_script_parallel.sh
#
# Layout: 1 master + (ntasks-1) workers. Default: 4 nodes / 5 tasks / 4 GPUs.
# For sequential (single process), use: sbatch rc_script.sh
#
# One job = one Stage C cell. Override via env when submitting, e.g.:
#   DISTANCE_METHOD=nli CLUSTERING_METHOD=dbscan EVALUATOR=google \
#     SEED=42 OUTPUT_DIR=data/outputs/stagec/M0_google/dN_IncDBSCAN_s42 \
#     sbatch rc_script_parallel.sh
#
# Termination: --max-total-genomes (required). See docs/ToxSearch_S_14_Run_Manifest.csv.
#
# Project root: use the NEW tree (not ~/ToxSearch-S, which is an older checkout).
# Override if needed: PROJECT_ROOT=/path/to/ToxSearch-S sbatch rc_script_parallel.sh

#SBATCH --job-name=txsp_par
#SBATCH --time=02-11:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=4
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --mail-user=slack:@U05PK8K2HEE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:4
#SBATCH --chdir=/home/os9660/Downloads/ToxSearch-S

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}"
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)"

# ---------------------------------------------------------------------------
# 1) Spack env (same one you use interactively)
# ---------------------------------------------------------------------------
spack env activate default-nlp-x86_64-25111801

# ---------------------------------------------------------------------------
# 2) Find CUDA root from nvcc and expose libs
# ---------------------------------------------------------------------------
NVCC_PATH=$(command -v nvcc || true)
if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found in PATH after activating Spack env; CUDA toolkit not visible." >&2
    exit 1
fi

CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

echo "[DEBUG] NVCC_PATH=$NVCC_PATH"
echo "[DEBUG] CUDA_ROOT=$CUDA_ROOT"
echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# ---------------------------------------------------------------------------
# 3) Activate cluster venv
# ---------------------------------------------------------------------------
source .spvenv/bin/activate

# ---------------------------------------------------------------------------
# 4) Python isolation (avoid user-site / stale PYTHONPATH collisions)
# ---------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# ---------------------------------------------------------------------------
# 5) Threading + quiet ML stacks
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"

# ---------------------------------------------------------------------------
# 6) API keys from .env (after isolation so keys are not wiped)
# ---------------------------------------------------------------------------
if [ -f ".env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi

# Optional: pin one Perspective key for this job (multi-job scheduling)
#   PERSPECTIVE_API_KEY=... sbatch rc_script_parallel.sh
if [ -n "${PERSPECTIVE_API_KEY:-}" ]; then
    export PERSPECTIVE_API_KEY
    unset PERSPECTIVE_API_KEYS || true
fi

# ---------------------------------------------------------------------------
# 7) Experiment knobs (Stage C defaults; override at sbatch time)
# ---------------------------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/src"

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-1000}"
EVALUATOR="${EVALUATOR:-google}"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-}"
OPENAI_MODEL="${OPENAI_MODEL:-omni-moderation-latest}"
OPERATORS="${OPERATORS:-all}"
MAX_VARIANTS="${MAX_VARIANTS:-1}"
SEED_FILE="${SEED_FILE:-data/prompt.csv}"
SEED="${SEED:-42}"
STAGNATION_LIMIT="${STAGNATION_LIMIT:-5}"
BATCH_SIZE="${BATCH_SIZE:-25}"

THETA_SIM="${THETA_SIM:-0.25}"
THETA_MERGE="${THETA_MERGE:-0.1}"
MIN_STABILITY_GENS="${MIN_STABILITY_GENS:-5}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-100}"
SPECIES_STAGNATION="${SPECIES_STAGNATION:-20}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"

DISTANCE_METHOD="${DISTANCE_METHOD:-embedding}"
DISTANCE_ALPHA="${DISTANCE_ALPHA:-0.7}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"

CLUSTERING_METHOD="${CLUSTERING_METHOD:-leader_follower}"
DBSCAN_EPS="${DBSCAN_EPS:-}"
DBSCAN_MIN_SAMPLES="${DBSCAN_MIN_SAMPLES:-2}"
INC_DBSCAN_VALIDATE_EVERY="${INC_DBSCAN_VALIDATE_EVERY:-0}"

RG_MODEL="${RG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
SMOKE_GGUF="${SMOKE_GGUF:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
DIST_TAG="$(echo "$DISTANCE_METHOD" | tr '+/' '__')"
case "$CLUSTERING_METHOD" in
    leader_follower|lf) CLUSTERING_METHOD="leader_follower"; CLUS_TAG="lf" ;;
    dbscan|inc_dbscan|incdbscan) CLUSTERING_METHOD="dbscan"; CLUS_TAG="inc_dbscan" ;;
    *)
        echo "ERROR: CLUSTERING_METHOD must be leader_follower or dbscan (got: ${CLUSTERING_METHOD})" >&2
        exit 1
        ;;
esac
OUTPUT_DIR="${OUTPUT_DIR:-data/outputs/slurm_par_${RUN_TS}_${EVALUATOR}_${DIST_TAG}_${CLUS_TAG}_s${SEED}_g${MAX_TOTAL_GENOMES}}"

# ---------------------------------------------------------------------------
# 8) GPU sanity check
# ---------------------------------------------------------------------------
nvidia-smi || { echo "No GPU visible"; exit 1; }

# ---------------------------------------------------------------------------
# 9) llama-cpp smoke test
# ---------------------------------------------------------------------------
python - <<PY
import llama_cpp
from llama_cpp import Llama
print("llama-cpp-python:", llama_cpp.__version__)
llm = Llama(
    model_path="${SMOKE_GGUF}",
    n_ctx=1024,
    n_gpu_layers=-1,
    n_batch=1024,
    verbose=False,
)
print("LLAMA_GPU_SMOKETEST_OK")
PY

# ---------------------------------------------------------------------------
# 10) CLI / config preflight (modern flags; no removed cluster0 / min-island)
# ---------------------------------------------------------------------------
python src/main.py --help | grep -E "theta-sim|embedding-model|species-capacity|distance-method|clustering-method|max-total-genomes|evaluator" || exit 1

if [ ! -f "$SEED_FILE" ]; then
    echo "ERROR: seed file not found: $SEED_FILE" >&2
    exit 1
fi
if [ ! -f "$RG_MODEL" ] || [ ! -f "$PG_MODEL" ]; then
    echo "ERROR: GGUF missing (RG=${RG_MODEL}, PG=${PG_MODEL})" >&2
    exit 1
fi
awk -v m="$THETA_MERGE" -v s="$THETA_SIM" 'BEGIN { if (m > s) exit 1 }' || {
    echo "ERROR: THETA_MERGE (${THETA_MERGE}) must be <= THETA_SIM (${THETA_SIM})." >&2
    exit 1
}
case "$EVALUATOR" in
    google)
        if [ -z "${PERSPECTIVE_API_KEY:-}" ] && [ -z "${PERSPECTIVE_API_KEYS:-}" ]; then
            echo "ERROR: EVALUATOR=google requires PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEYS" >&2
            exit 1
        fi
        ;;
    openai)
        if [ -z "${OPENAI_API_KEY:-}" ]; then
            echo "ERROR: EVALUATOR=openai requires OPENAI_API_KEY" >&2
            exit 1
        fi
        ;;
    *)
        echo "ERROR: EVALUATOR must be google or openai (got: ${EVALUATOR})" >&2
        exit 1
        ;;
esac

echo "=========================================="
echo "SLURM ToxSearch-S parallel run"
echo "  evaluator=${EVALUATOR}  distance=${DISTANCE_METHOD}  clustering=${CLUSTERING_METHOD}"
echo "  max_total_genomes=${MAX_TOTAL_GENOMES}  seed=${SEED}  seed_file=${SEED_FILE}"
echo "  theta_sim=${THETA_SIM}  theta_merge=${THETA_MERGE}  batch_size=${BATCH_SIZE}"
echo "  ntasks=${SLURM_NTASKS:-?} (1 master + workers via --parallel)"
echo "  output_dir=${OUTPUT_DIR}"
echo "=========================================="

# ---------------------------------------------------------------------------
# 11) MPI launch with srun (1 master + ntasks-1 workers)
# ---------------------------------------------------------------------------
SRUN_ARGS=(
    --parallel
    --batch-size "$BATCH_SIZE"
    --max-total-genomes "$MAX_TOTAL_GENOMES"
    --evaluator "$EVALUATOR"
    --operators "$OPERATORS"
    --max-variants "$MAX_VARIANTS"
    --stagnation-limit "$STAGNATION_LIMIT"
    --theta-sim "$THETA_SIM"
    --theta-merge "$THETA_MERGE"
    --min-stability-gens "$MIN_STABILITY_GENS"
    --species-capacity "$SPECIES_CAPACITY"
    --species-stagnation "$SPECIES_STAGNATION"
    --embedding-model "$EMBEDDING_MODEL"
    --embedding-dim "$EMBEDDING_DIM"
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE"
    --distance-method "$DISTANCE_METHOD"
    --distance-alpha "$DISTANCE_ALPHA"
    --nli-model "$NLI_MODEL"
    --clustering-method "$CLUSTERING_METHOD"
    --rg "$RG_MODEL"
    --pg "$PG_MODEL"
    --seed-file "$SEED_FILE"
    --seed "$SEED"
    --output-dir "$OUTPUT_DIR"
)

if [ -n "$NORTH_STAR_METRIC" ]; then
    SRUN_ARGS+=(--north-star-metric "$NORTH_STAR_METRIC")
fi
if [ "$EVALUATOR" = "openai" ]; then
    SRUN_ARGS+=(--openai-model "$OPENAI_MODEL")
fi
if [ "$CLUSTERING_METHOD" = "dbscan" ]; then
    SRUN_ARGS+=(--dbscan-min-samples "$DBSCAN_MIN_SAMPLES")
    SRUN_ARGS+=(--inc-dbscan-validate-every "$INC_DBSCAN_VALIDATE_EVERY")
    if [ -n "$DBSCAN_EPS" ]; then
        SRUN_ARGS+=(--dbscan-eps "$DBSCAN_EPS")
    else
        SRUN_ARGS+=(--dbscan-eps "$THETA_SIM")
    fi
fi

srun python src/main.py "${SRUN_ARGS[@]}"

echo "All experiments completed!"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
