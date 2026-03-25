#!/bin/bash
#===============================================================================
# RainbowPlus — Slurm batch script (single node, vLLM + Google Perspective)
#
# Submit (from this repo root):
#   sbatch sbatch_rainbowplus.sh
#
# 10 replicates (one job each, distinct outputs via SLURM_JOB_ID):
#   for i in {1..10}; do sbatch sbatch_rainbowplus.sh; done
#
# Or a job array (set RUN_ID per task; uncomment #SBATCH --array):
#   #SBATCH --array=0-9
#   (script sets RUN_ID from array index if USE_ARRAY_RUN_ID=1)
#
# Override via environment (sbatch --export=ALL,MAX_GENOMES=10000,...) or #SBATCH.
# Required secrets: PERSPECTIVE_API_KEY (or PERSPECTIVE_API_KEYS) in .env or env.
# Optional: HF_TOKEN / huggingface-cli login for gated models.
#===============================================================================

#SBATCH --job-name=rainbowplus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00

# Slurm output (create logs-slurm before submit, or rely on mkdir below)
# For job arrays, %a is the array task id; %A is the array job id.
#SBATCH --output=logs-slurm/%x_%j.log
#SBATCH --error=logs-slurm/%x_%j.err

##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=gpu
##SBATCH --qos=YOUR_QOS
##SBATCH --constraint=a100
## Some sites use: #SBATCH --gres=gpu:1
## Job array (optional): #SBATCH --array=0-9

set -euo pipefail

# Repo root = directory containing this script (works no matter where sbatch was invoked)
RP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$RP_ROOT"

mkdir -p logs-slurm

# ------------------------------------------------------------------------------
# Environment (Python)
# ------------------------------------------------------------------------------
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"
export NUMEXPR_NUM_THREADS="${OMP_NUM_THREADS}"

# Optional: avoid user-site packages leaking into the job
# export PYTHONNOUSERSITE=1

# ------------------------------------------------------------------------------
# Activate your Python environment (EDIT ONE for your cluster)
# ------------------------------------------------------------------------------
# module purge
# module load cuda/12.1
# source ~/.bashrc && conda activate rainbowplus
# source "${RP_ROOT}/venv/bin/activate"

# ------------------------------------------------------------------------------
# Secrets and API limits (.env in repo root: PERSPECTIVE_API_KEY=...)
# ------------------------------------------------------------------------------
if [[ -f "${RP_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${RP_ROOT}/.env"
  set +a
fi

if [[ -z "${PERSPECTIVE_API_KEY:-}" && -z "${PERSPECTIVE_API_KEYS:-}" ]]; then
  echo "ERROR: Set PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEYS (e.g. in .env)." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# Preflight
# ------------------------------------------------------------------------------
command -v python3 >/dev/null 2>&1 || { echo "ERROR: python3 not in PATH" >&2; exit 1; }

CONFIG_FILE="${CONFIG_FILE:-configs/base.yml}"
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "ERROR: config not found: $CONFIG_FILE (cwd=$(pwd))" >&2
  exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || true
fi

# ------------------------------------------------------------------------------
# Experiment parameters (override with sbatch --export=ALL,VAR=value)
# ------------------------------------------------------------------------------
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_ITERS="${MAX_ITERS:-10000}"
MAX_GENOMES="${MAX_GENOMES:-1000}"
NUM_MUTATIONS="${NUM_MUTATIONS:-3}"
FITNESS_THRESHOLD="${FITNESS_THRESHOLD:-0.3}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LOG_DIR="${LOG_DIR:-${RP_ROOT}/logs}"

# Unique run folder: logs/<model>/<dataset>/<run_id>/all_genomes.jsonl
# Priority: RUN_ID env > SLURM_JOB_ID > timestamp (so interactive runs do not clobber)
if [[ -n "${RUN_ID:-}" ]]; then
  :
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
  if [[ -n "${SLURM_ARRAY_TASK_ID:-}" && "${USE_ARRAY_RUN_ID:-0}" == "1" ]]; then
    RUN_ID="${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
  else
    RUN_ID="${SLURM_JOB_ID}"
  fi
else
  RUN_ID="local_$(date +%Y%m%d_%H%M%S)_$$"
fi

# ------------------------------------------------------------------------------
# Build command
# ------------------------------------------------------------------------------
PY=(python3 -m rainbowplus.rainbowplus)
PY+=(
  --config_file "$CONFIG_FILE"
  --num_samples "$NUM_SAMPLES"
  --max_iters "$MAX_ITERS"
  --max_genomes "$MAX_GENOMES"
  --num_mutations "$NUM_MUTATIONS"
  --fitness_threshold "$FITNESS_THRESHOLD"
  --log_dir "$LOG_DIR"
  --log_interval "$LOG_INTERVAL"
  --run_id "$RUN_ID"
)

[[ -n "${DATASET:-}" ]] && PY+=(--dataset "$DATASET")
[[ -n "${TARGET_LLM:-}" ]] && PY+=(--target_llm "$TARGET_LLM")
[[ -n "${NO_SHUFFLE:-}" ]] && PY+=(--no-shuffle)
[[ -n "${RANDOM_SEED:-}" ]] && PY+=(--random_seed "$RANDOM_SEED")

# ------------------------------------------------------------------------------
# Launch
# ------------------------------------------------------------------------------
echo "============================================================================"
echo "Host:     $(hostname)"
echo "Job ID:   ${SLURM_JOB_ID:-N/A}  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "RUN_ID:   $RUN_ID"
echo "CWD:      $RP_ROOT"
echo "LOG_DIR:  $LOG_DIR  (experiment data under <model>/<dataset>/$RUN_ID/)"
echo "Command:  ${PY[*]}"
echo "============================================================================"

# Many clusters: plain exec is fine for single-task GPU jobs.
# If the GPU is not visible, try uncommenting srun below and comment out exec.
if [[ "${USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1; then
  exec srun --unbuffered --cpu-bind=none "${PY[@]}"
else
  exec "${PY[@]}"
fi
