# Shared RQ2 live-cell runner. Sourced by the four cell scripts.
# Expects: DISTANCE_METHOD CLUSTERING_METHOD THETA_SIM DBSCAN_EPS
#          DBSCAN_MIN_SAMPLES OUTPUT_ROOT CELL_TAG
#
# OpenAI key: OPENAI_API_KEY_PIN in this process, else rc_rq2_openai_key.sh,
# else the environment / .env. The pin wins so a later resubmit uses the
# same edited key.

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}"
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)  cell=${CELL_TAG}"

_CELL_PIN="${OPENAI_API_KEY_PIN:-}"
if [ -f "rc_rq2_openai_key.sh" ]; then
    # shellcheck source=/dev/null
    source "rc_rq2_openai_key.sh"
fi
if [ -n "$_CELL_PIN" ]; then
    OPENAI_API_KEY_PIN="$_CELL_PIN"
fi
unset _CELL_PIN

spack env activate default-nlp-x86_64-25111801

NVCC_PATH=$(command -v nvcc || true)
if [ -z "$NVCC_PATH" ]; then
    echo "ERROR: nvcc not found in PATH after activating Spack env; CUDA toolkit not visible." >&2
    exit 1
fi
CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"

source .spvenv/bin/activate

export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
export PYTHONPATH="${PROJECT_ROOT}/src"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"

if [ -f ".env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi

# Pin overrides .env / leftover environment so all 10 seeds share one key.
if [ -n "${OPENAI_API_KEY_PIN:-}" ]; then
    export OPENAI_API_KEY="$OPENAI_API_KEY_PIN"
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
    export OPENAI_API_KEY
fi
if [ -n "${PERSPECTIVE_API_KEY:-}" ]; then
    export PERSPECTIVE_API_KEY
    unset PERSPECTIVE_API_KEYS || true
fi

N_REPLICATES="${N_REPLICATES:-10}"
SEED_BASE="${SEED_BASE:-0}"

EVALUATOR="openai"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-violence}"
OPENAI_MODEL="${OPENAI_MODEL:-omni-moderation-latest}"
THETA_MERGE="${THETA_MERGE:-$THETA_SIM}"

MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-2000}"
OPERATORS="${OPERATORS:-all}"
MAX_VARIANTS="${MAX_VARIANTS:-1}"
SEED_FILE="${SEED_FILE:-data/prompt.csv}"
STAGNATION_LIMIT="${STAGNATION_LIMIT:-5}"
MIN_STABILITY_GENS="${MIN_STABILITY_GENS:-3}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-25}"
SPECIES_STAGNATION="${SPECIES_STAGNATION:-7}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
EMBEDDING_DIM="${EMBEDDING_DIM:-384}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-64}"
DISTANCE_ALPHA="${DISTANCE_ALPHA:-0.7}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
NLI_CACHE_FILE="${NLI_CACHE_FILE:-data/cache/nli_pairs.sqlite}"

RG_MODEL="${RG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
SMOKE_GGUF="${SMOKE_GGUF:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"

SKIP_IF_DONE="${SKIP_IF_DONE:-1}"
EXECUTION_MODE="${EXECUTION_MODE:-auto}"

REPLICATE_RUNS=()
for ((i = 0; i < N_REPLICATES; i++)); do
    seed=$((SEED_BASE + i))
    rid="${CELL_TAG}_rep${i}_s${seed}"
    out="${OUTPUT_ROOT}/rep${i}_s${seed}"
    REPLICATE_RUNS+=("${rid}|${out}|${seed}")
done

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
    if [ -z "${OPENAI_API_KEY:-}" ]; then
        echo "ERROR: set OPENAI_API_KEY_PIN in rc_rq2_openai_key.sh (copy from .example)." >&2
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
    echo "SLURM ToxSearch-S: RQ2 ${CELL_TAG}  OpenAI  N=${MAX_TOTAL_GENOMES}"
    echo "  run_id=${run_id}"
    echo "  evaluator=${EVALUATOR}  north_star=${NORTH_STAR_METRIC}  openai_model=${OPENAI_MODEL}"
    echo "  distance=${DISTANCE_METHOD}  clustering=${CLUSTERING_METHOD}"
    echo "  theta=${THETA_SIM}  eps=${DBSCAN_EPS}  min_samples=${DBSCAN_MIN_SAMPLES}"
    echo "  capacity=${SPECIES_CAPACITY}  stag=${SPECIES_STAGNATION}  seed=${run_seed}"
    echo "  output_dir=${output_dir}"
    echo "  array_task=${SLURM_ARRAY_TASK_ID:-n/a}  job=${SLURM_JOB_ID:-local}"
    echo "=========================================="

    mkdir -p "$output_dir"
    set +e
    python src/main.py \
        --max-total-genomes "$MAX_TOTAL_GENOMES" \
        --evaluator "$EVALUATOR" \
        --north-star-metric "$NORTH_STAR_METRIC" \
        --openai-model "$OPENAI_MODEL" \
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
        --nli-cache-file "$NLI_CACHE_FILE" \
        --clustering-method "$CLUSTERING_METHOD" \
        --dbscan-eps "$DBSCAN_EPS" \
        --dbscan-min-samples "$DBSCAN_MIN_SAMPLES" \
        --rg "$RG_MODEL" \
        --pg "$PG_MODEL" \
        --seed-file "$SEED_FILE" \
        --seed "$run_seed" \
        --output-dir "$output_dir"
    local rc=$?
    set -e
    if [ "$rc" -ne 0 ]; then
        echo "[FAIL] ${run_id} exit=${rc} → ${output_dir}"
        return "$rc"
    fi
    echo "[DONE] ${run_id} → ${output_dir}"
    return 0
}

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

python src/main.py --help | grep -E "theta-sim|dbscan-eps|embedding-model|distance-method|clustering-method|max-total-genomes|evaluator" || exit 1
preflight

if [ "$EXECUTION_MODE" = "auto" ]; then
    if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
        EXECUTION_MODE="array"
    else
        EXECUTION_MODE="loop"
    fi
fi

echo "[DEBUG] EXECUTION_MODE=${EXECUTION_MODE}  N_REPLICATES=${#REPLICATE_RUNS[@]}"
echo "[DEBUG] CELL=${CELL_TAG}  DIST=${DISTANCE_METHOD}  CLUST=${CLUSTERING_METHOD}  t=${THETA_SIM}  ms=${DBSCAN_MIN_SAMPLES}  OUT=${OUTPUT_ROOT}"
echo "[DEBUG] OPENAI_API_KEY is $([ -n "${OPENAI_API_KEY:-}" ] && echo set || echo missing)  pin=$([ -n "${OPENAI_API_KEY_PIN:-}" ] && echo set || echo empty)"

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
        n_fail=0
        for spec in "${REPLICATE_RUNS[@]}"; do
            IFS='|' read -r RUN_ID OUTPUT_DIR RUN_SEED <<<"$spec"
            if cell_already_done "$OUTPUT_DIR"; then
                echo "[SKIP] ${RUN_ID}"
                n_skip=$((n_skip + 1))
                continue
            fi
            if run_one_execution "$RUN_ID" "$OUTPUT_DIR" "$RUN_SEED"; then
                n_ok=$((n_ok + 1))
            else
                n_fail=$((n_fail + 1))
                echo "[WARN] continuing loop after ${RUN_ID} failed"
            fi
        done
        echo "Loop finished: launched_ok=${n_ok} skipped_done=${n_skip} failed=${n_fail}"
        if [ "$n_fail" -gt 0 ]; then
            exit 1
        fi
        ;;
    *)
        echo "ERROR: EXECUTION_MODE must be auto|array|loop (got: ${EXECUTION_MODE})" >&2
        exit 1
        ;;
esac

echo "All experiments completed for this allocation."
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}  SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-n/a}"
