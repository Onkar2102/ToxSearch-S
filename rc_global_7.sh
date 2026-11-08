#!/bin/bash -l
#
# Global-EA baseline (no speciation): embedding distance tag, OpenAI illicit, N=2000.
# Single execution 7/10 — paste this job's OpenAI key in OPENAI_API_KEY_PIN.
# Seed/replicate: 7 → rep7_s7
#
# Shared output root (each script writes its own seed dir):
#   data/outputs/evostar2027/global_ea_embedding_openai_illicit_n2000/rep{i}_s{seed}
#
# sbatch rc_global_7.sh
#
# Paste the OpenAI key in OPENAI_API_KEY_PIN below. Only this job uses that pin.

#SBATCH --job-name=global_7
#SBATCH --time=1-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
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

OPENAI_API_KEY_PIN=""

CELL_TAG="GlobalEA_embedding_openai_illicit_n2000"
DISTANCE_METHOD="embedding"
CLUSTERING_METHOD="none"
THETA_SIM="0.1"
DBSCAN_EPS="0.1"
DBSCAN_MIN_SAMPLES="2"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/evostar2027/global_ea_embedding_openai_illicit_n2000}"
NLI_CACHE_FILE="${NLI_CACHE_FILE:-data/cache/nli_pairs_global_ea_embedding.sqlite}"

# This script runs exactly one replicate/seed.
REP_INDEX=7
RUN_SEED=7

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}"
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)"

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

# Pin wins over .env so this job uses its own key.
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

EVALUATOR="openai"
NORTH_STAR_METRIC="${NORTH_STAR_METRIC:-illicit}"
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

RG_MODEL="${RG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
PG_MODEL="${PG_MODEL:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"
SMOKE_GGUF="${SMOKE_GGUF:-models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf}"

SKIP_IF_DONE="${SKIP_IF_DONE:-1}"

RUN_ID="${CELL_TAG}_rep${REP_INDEX}_s${RUN_SEED}"
OUTPUT_DIR="${OUTPUT_ROOT}/rep${REP_INDEX}_s${RUN_SEED}"

cell_already_done() {
    local out="$1"
    local tracker="${out}/EvolutionTracker.json"
    if [ "${SKIP_IF_DONE}" != "1" ]; then
        return 1
    fi
    if [ ! -f "$tracker" ]; then
        return 1
    fi
    python - "$tracker" <<'DONEPY'
import json, sys
p = sys.argv[1]
try:
    with open(p, encoding="utf-8") as f:
        t = json.load(f)
except Exception:
    sys.exit(1)
sys.exit(0 if str(t.get("status", "")).lower() == "complete" else 1)
DONEPY
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
        echo "ERROR: set OPENAI_API_KEY_PIN at the top of this script." >&2
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
    echo "SLURM ToxSearch-S: Global-EA (no speciation) ${CELL_TAG}  OpenAI  N=${MAX_TOTAL_GENOMES}"
    echo "  run_id=${run_id}"
    echo "  evaluator=${EVALUATOR}  north_star=${NORTH_STAR_METRIC}  openai_model=${OPENAI_MODEL}"
    echo "  distance=${DISTANCE_METHOD}  clustering=${CLUSTERING_METHOD}"
    echo "  theta=${THETA_SIM}  eps=${DBSCAN_EPS}  min_samples=${DBSCAN_MIN_SAMPLES}"
    echo "  capacity=${SPECIES_CAPACITY}  stag=${SPECIES_STAGNATION}  seed=${run_seed}"
    echo "  output_dir=${output_dir}"
    echo "  job=${SLURM_JOB_ID:-local}"
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

python - <<SMOKE
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
SMOKE

python src/main.py --help | grep -E "theta-sim|dbscan-eps|embedding-model|distance-method|clustering-method|max-total-genomes|evaluator" || exit 1
preflight

echo "[DEBUG] JOB=7/10  CELL=${CELL_TAG}  DIST=${DISTANCE_METHOD}  CLUST=${CLUSTERING_METHOD}  REP=${REP_INDEX}  SEED=${RUN_SEED}  OUT=${OUTPUT_DIR}"
echo "[DEBUG] OPENAI_API_KEY is $([ -n "${OPENAI_API_KEY:-}" ] && echo set || echo missing)  pin=$([ -n "${OPENAI_API_KEY_PIN:-}" ] && echo set || echo empty)"

run_one_execution "$RUN_ID" "$OUTPUT_DIR" "$RUN_SEED"

echo "All experiments completed for this allocation."
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
