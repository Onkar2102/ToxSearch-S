#!/bin/bash -l
#
# SLURM run script: 10 independent L–F + embedding replicates
# (Google Perspective / toxicity, theta_sim = theta_merge = 0.1).
#
# RIT RC job arrays:
#   https://research-computing.git-pages.rit.edu/docs/slurm_tutorial_2.html
#
# Submit (preferred — one GPU task per replicate, seeds 42..51):
#   sbatch --array=0-9%2 rc_script.sh
#
# Or one allocation, sequential:
#   EXECUTION_MODE=loop sbatch rc_script.sh
#
# Override PROJECT_ROOT if your tree is elsewhere.
# Termination: --max-total-genomes (required).

#SBATCH --job-name=txsp_lf_r01
#SBATCH --time=00-13:59:00
# Use %A (array job id) and %a (task id) so array tasks do not clobber logs.
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --mail-user=slack:@U05PK8K2HEE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/os9660/Downloads/ToxSearch-S
# Array is OFF by default so EXECUTION_MODE=loop is one allocation.
# Prefer: sbatch --array=0-9%2 rc_script.sh
##SBATCH --array=0-9%2

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}"
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)"

# ---------------------------------------------------------------------------
# 1) Spack env
# ---------------------------------------------------------------------------
spack env activate default-nlp-x86_64-25111801

# ---------------------------------------------------------------------------
# 2) CUDA libs
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
# 3) Cluster venv
# ---------------------------------------------------------------------------
source .spvenv/bin/activate

# ---------------------------------------------------------------------------
# 4) Python isolation
# ---------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# ---------------------------------------------------------------------------
# 5) Threading + quiet ML stacks
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"

# ---------------------------------------------------------------------------
# 6) API keys from .env
# ---------------------------------------------------------------------------
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
    unset PERSPECTIVE_API_KEYS || true
fi

# ---------------------------------------------------------------------------
# 7) Fixed experiment: L–F + embedding + Perspective/toxicity @ r=0.1
# ---------------------------------------------------------------------------
export PYTHONPATH="${PROJECT_ROOT}/src"

N_REPLICATES="${N_REPLICATES:-10}"
SEED_BASE="${SEED_BASE:-2027}"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/lf_tox_r01}"

EVALUATOR="google"
NORTH_STAR_METRIC="toxicity"
DISTANCE_METHOD="embedding"
CLUSTERING_METHOD="leader_follower"
THETA_SIM="${THETA_SIM:-0.1}"
THETA_MERGE="${THETA_MERGE:-0.1}"

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-1000}"
OPERATORS="${OPERATORS:-all}"
MAX_VARIANTS="${MAX_VARIANTS:-1}"
SEED_FILE="${SEED_FILE:-data/prompt.csv}"
STAGNATION_LIMIT="${STAGNATION_LIMIT:-5}"

MIN_STABILITY_GENS="${MIN_STABILITY_GENS:-3}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-20}"
SPECIES_STAGNATION="${SPECIES_STAGNATION:-7}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
DISTANCE_ALPHA="${DISTANCE_ALPHA:-0.7}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"

RG_MODEL="${RG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
SMOKE_GGUF="${SMOKE_GGUF:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"

SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-auto}"  # auto | array | loop

# Build replicate list: seeds SEED_BASE .. SEED_BASE+N_REPLICATES-1
REPLICATE_RUNS=()
for ((i = 0; i < N_REPLICATES; i++)); do
    seed=$((SEED_BASE + i))
    rid="LF_tox_r01_rep${i}_s${seed}"
    out="${OUTPUT_ROOT}/rep${i}_s${seed}"
    REPLICATE_RUNS+=("${rid}|${out}|${seed}")
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
cell_already_done() {
    local out="$1"
    local tracker="${out}/EvolutionTracker.json"
    if [ "${SKIP_IF_DONE}" != "1" ]; then
        return 1
    fi
    if [ ! -f "$tracker" ]; then
        return 1
    fi
    python - "$tracker" <<'PY'
import json, sys
p = sys.argv[1]
try:
    with open(p, encoding="utf-8") as f:
        t = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if str(t.get("status", "")).lower() == "complete" else 1)
PY
}

preflight() {
    if [ ! -f "$SEED_FILE" ]; then
        echo "ERROR: seed file not found: $SEED_FILE" >&2
        exit 1
    fi
    if [ ! -f "$RG_MODEL" ] || [ ! -f "$PG_MODEL" ]; then
        echo "ERROR: GGUF missing (RG=${RG_MODEL}, PG=${PG_MODEL})" >&2
        exit 1
    fi
    if [ -z "${PERSPECTIVE_API_KEY:-}" ] && [ -z "${PERSPECTIVE_API_KEYS:-}" ]; then
        echo "ERROR: PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEYS required" >&2
        exit 1
    fi
    awk -v m="$THETA_MERGE" -v s="$THETA_SIM" 'BEGIN { if (m > s) exit 1 }' || {
        echo "ERROR: THETA_MERGE (${THETA_MERGE}) must be <= THETA_SIM (${THETA_SIM})." >&2
        exit 1
    }
}

run_one_execution() {
    local run_id="$1"
    local output_dir="$2"
    local run_seed="$3"

    if cell_already_done "$output_dir"; then
        echo "[SKIP] ${run_id}: already complete at ${output_dir}"
        return 0
    fi

    echo "=========================================="
    echo "SLURM ToxSearch-S: L–F embedding @ r=0.1"
    echo "  run_id=${run_id}"
    echo "  evaluator=${EVALUATOR}  north_star=${NORTH_STAR_METRIC}"
    echo "  distance=${DISTANCE_METHOD}  clustering=${CLUSTERING_METHOD}"
    echo "  max_total_genomes=${MAX_TOTAL_GENOMES}  seed=${run_seed}  seed_file=${SEED_FILE}"
    echo "  theta_sim=${THETA_SIM}  theta_merge=${THETA_MERGE}"
    echo "  output_dir=${output_dir}"
    echo "  array_task=${SLURM_ARRAY_TASK_ID:-n/a}  job=${SLURM_JOB_ID:-local}"
    echo "=========================================="

    mkdir -p "$output_dir"
    python src/main.py \
        --max-total-genomes "$MAX_TOTAL_GENOMES" \
        --evaluator "$EVALUATOR" \
        --north-star-metric "$NORTH_STAR_METRIC" \
        --operators "$OPERATORS" \
        --max-variants "$MAX_VARIANTS" \
        --stagnation-limit "$STAGNATION_LIMIT" \
        --theta-sim "$THETA_SIM" \
        --theta-merge "$THETA_MERGE" \
        --min-stability-gens "$MIN_STABILITY_GENS" \
        --species-capacity "$SPECIES_CAPACITY" \
        --species-stagnation "$SPECIES_STAGNATION" \
        --embedding-model "$EMBEDDING_MODEL" \
        --embedding-dim "$EMBEDDING_DIM" \
        --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
        --distance-method "$DISTANCE_METHOD" \
        --distance-alpha "$DISTANCE_ALPHA" \
        --nli-model "$NLI_MODEL" \
        --clustering-method "$CLUSTERING_METHOD" \
        --rg "$RG_MODEL" \
        --pg "$PG_MODEL" \
        --seed-file "$SEED_FILE" \
        --seed "$run_seed" \
        --output-dir "$output_dir"

    echo "[DONE] ${run_id} → ${output_dir}"
}

# ---------------------------------------------------------------------------
# 8) GPU sanity + smoke (once per allocation)
# ---------------------------------------------------------------------------
nvidia-smi || { echo "No GPU visible"; exit 1; }

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

python src/main.py --help | grep -E "theta-sim|embedding-model|distance-method|clustering-method|max-total-genomes|evaluator" || exit 1
preflight

# ---------------------------------------------------------------------------
# 9) Dispatch
# ---------------------------------------------------------------------------
if [ "$EXECUTION_MODE" = "auto" ]; then
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        EXECUTION_MODE="array"
    else
        EXECUTION_MODE="loop"
    fi
fi

echo "[DEBUG] EXECUTION_MODE=${EXECUTION_MODE}  N_REPLICATES=${#REPLICATE_RUNS[@]}"
echo "[DEBUG] THETA_SIM=${THETA_SIM}  THETA_MERGE=${THETA_MERGE}  OUTPUT_ROOT=${OUTPUT_ROOT}"

case "$EXECUTION_MODE" in
    array)
        idx="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID unset}"
        if [ "$idx" -lt 0 ] || [ "$idx" -ge "${#REPLICATE_RUNS[@]}" ]; then
            echo "ERROR: SLURM_ARRAY_TASK_ID=${idx} out of range [0, $((${#REPLICATE_RUNS[@]} - 1))]" >&2
            exit 1
        fi
        IFS='|' read -r RUN_ID OUTPUT_DIR RUN_SEED <<<"${REPLICATE_RUNS[$idx]}"
        run_one_execution "$RUN_ID" "$OUTPUT_DIR" "$RUN_SEED"
        ;;

    loop)
        if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ "${SLURM_ARRAY_TASK_ID}" -ne 0 ]; then
            echo "[SKIP] EXECUTION_MODE=loop: only array task 0 runs (this is task ${SLURM_ARRAY_TASK_ID})"
            exit 0
        fi
        n_ok=0
        n_skip=0
        for spec in "${REPLICATE_RUNS[@]}"; do
            IFS='|' read -r RUN_ID OUTPUT_DIR RUN_SEED <<<"$spec"
            if cell_already_done "$OUTPUT_DIR"; then
                echo "[SKIP] ${RUN_ID}"
                n_skip=$((n_skip + 1))
                continue
            fi
            run_one_execution "$RUN_ID" "$OUTPUT_DIR" "$RUN_SEED"
            n_ok=$((n_ok + 1))
        done
        echo "Loop finished: launched=${n_ok} skipped_done=${n_skip}"
        ;;

    *)
        echo "ERROR: EXECUTION_MODE must be auto|array|loop (got: ${EXECUTION_MODE})" >&2
        exit 1
        ;;
esac

echo "All experiments completed for this allocation."
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-n/a}"
