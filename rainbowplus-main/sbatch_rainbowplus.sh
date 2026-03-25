#!/bin/bash -l
#
# SLURM run script for RainbowPlus (single-node vLLM + Google Perspective).
# Modeled on repo-root rc_script.sh (Spack, CUDA paths, .spvenv, threading env).
#
# Submit from anywhere:
#   sbatch /path/to/rainbowplus-main/sbatch_rainbowplus.sh
#
# Parent ToxSearch-S tree (optional): if this file lives in .../ToxSearch-S/rainbowplus-main/,
# we auto-use ../.spvenv when present (same as running rc_script.sh from ToxSearch-S root).

#SBATCH --job-name=rainbowplus
#SBATCH --time=2-23:59:00
#SBATCH --output=logs-slurm/%x_%j.out
#SBATCH --error=logs-slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
## Job array (optional): #SBATCH --array=0-9

set -euo pipefail

# ------------------------------------------------------------------------------
# 0) Repo root (directory containing this script)
# ------------------------------------------------------------------------------
RP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$RP_ROOT"
mkdir -p logs-slurm

# Parent project root (ToxSearch-S when layout is .../ToxSearch-S/rainbowplus-main/)
TOX_ROOT="${TOX_ROOT:-$(cd "${RP_ROOT}/.." && pwd)}"

# ------------------------------------------------------------------------------
# 1) Spack env (same pattern as rc_script.sh; set USE_SPACK=0 to skip)
# ------------------------------------------------------------------------------
if [[ "${USE_SPACK:-1}" == "1" ]] && command -v spack >/dev/null 2>&1; then
  SPACK_ENV_NAME="${SPACK_ENV_NAME:-default-nlp-x86_64-25111801}"
  # shellcheck disable=SC1090
  spack env activate "${SPACK_ENV_NAME}"
else
  echo "[INFO] Skipping Spack (USE_SPACK=${USE_SPACK:-1} or spack not in PATH)"
fi

# ------------------------------------------------------------------------------
# 2) CUDA: find nvcc and expose libs (same as rc_script.sh)
# ------------------------------------------------------------------------------
NVCC_PATH=$(command -v nvcc || true)
if [[ -z "${NVCC_PATH}" ]]; then
  if [[ "${SKIP_NVCC_CHECK:-0}" == "1" ]]; then
    echo "[WARN] nvcc not in PATH; SKIP_NVCC_CHECK=1 (set LD_LIBRARY_PATH yourself if needed)"
  else
    echo "ERROR: nvcc not found in PATH; CUDA toolkit not visible after env setup. Set SKIP_NVCC_CHECK=1 to bypass (conda-only)." >&2
    exit 1
  fi
else
  CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
  echo "[DEBUG] NVCC_PATH=$NVCC_PATH"
  echo "[DEBUG] CUDA_ROOT=$CUDA_ROOT"
  echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
fi

# ------------------------------------------------------------------------------
# 3) Python venv (prefer explicit VENV_ACTIVATE; else ToxSearch-S .spvenv; else local)
# ------------------------------------------------------------------------------
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
elif [[ -f "${TOX_ROOT}/.spvenv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${TOX_ROOT}/.spvenv/bin/activate"
elif [[ -f "${RP_ROOT}/.spvenv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${RP_ROOT}/.spvenv/bin/activate"
elif [[ -f "${RP_ROOT}/venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${RP_ROOT}/venv/bin/activate"
else
  echo "ERROR: No venv found. Set VENV_ACTIVATE=/path/to/venv/bin/activate or create .spvenv/venv under ${TOX_ROOT} or ${RP_ROOT}" >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# 4) Python isolation (same as rc_script.sh; RainbowPlus does not need PYTHONPATH=src)
# ------------------------------------------------------------------------------
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# ------------------------------------------------------------------------------
# 5) Threading / ML stack env (same as rc_script.sh)
# ------------------------------------------------------------------------------
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"

# ------------------------------------------------------------------------------
# 6) API secrets (.env in rainbowplus-main, then Perspective check)
# ------------------------------------------------------------------------------
if [[ -f "${RP_ROOT}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${RP_ROOT}/.env"
  set +a
fi

if [[ -z "${PERSPECTIVE_API_KEY:-}" && -z "${PERSPECTIVE_API_KEYS:-}" ]]; then
  echo "ERROR: Set PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEYS (e.g. in ${RP_ROOT}/.env)." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# 7) GPU sanity check (strict, like rc_script.sh)
# ------------------------------------------------------------------------------
nvidia-smi || { echo "ERROR: No GPU visible" >&2; exit 1; }

# ------------------------------------------------------------------------------
# 8) Smoke test: Torch CUDA (RainbowPlus uses vLLM/torch; not llama-cpp)
# ------------------------------------------------------------------------------
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA not available to PyTorch")
print("TORCH_CUDA_SMOKETEST_OK")
PY

# Optional: fail fast if vLLM import breaks in this env (SKIP_VLLM_SMOKE=1 to skip)
if [[ "${SKIP_VLLM_SMOKE:-0}" != "1" ]]; then
  python - <<'PY'
import vllm
print("vllm:", getattr(vllm, "__version__", "unknown"))
print("VLLM_IMPORT_OK")
PY
fi

# ------------------------------------------------------------------------------
# 9) Experiment parameters (override: sbatch --export=ALL,MAX_GENOMES=2000,...)
# ------------------------------------------------------------------------------
CONFIG_FILE="${CONFIG_FILE:-configs/base.yml}"
[[ -f "$CONFIG_FILE" ]] || { echo "ERROR: config not found: $CONFIG_FILE" >&2; exit 1; }

NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_ITERS="${MAX_ITERS:-10000}"
MAX_GENOMES="${MAX_GENOMES:-1000}"
NUM_MUTATIONS="${NUM_MUTATIONS:-3}"
FITNESS_THRESHOLD="${FITNESS_THRESHOLD:-0.3}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LOG_DIR="${LOG_DIR:-${RP_ROOT}/logs}"

# RUN_ID: explicit > Slurm job > local timestamp
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

PY_CMD=(
  python -m rainbowplus.rainbowplus
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

[[ -n "${DATASET:-}" ]] && PY_CMD+=(--dataset "$DATASET")
[[ -n "${TARGET_LLM:-}" ]] && PY_CMD+=(--target_llm "$TARGET_LLM")
[[ -n "${NO_SHUFFLE:-}" ]] && PY_CMD+=(--no-shuffle)
[[ -n "${RANDOM_SEED:-}" ]] && PY_CMD+=(--random_seed "$RANDOM_SEED")

# ------------------------------------------------------------------------------
# 10) Run (single-task GPU: plain python; set USE_SRUN=1 if your site requires srun)
# ------------------------------------------------------------------------------
echo "============================================================================"
echo "Host:       $(hostname)"
echo "RP_ROOT:    $RP_ROOT"
echo "TOX_ROOT:   $TOX_ROOT"
echo "Job ID:     ${SLURM_JOB_ID:-N/A}  Array task: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "RUN_ID:     $RUN_ID"
echo "LOG_DIR:    $LOG_DIR  -> <model>/<dataset>/$RUN_ID/"
echo "Command:    ${PY_CMD[*]}"
echo "============================================================================"

if [[ "${USE_SRUN:-0}" == "1" ]] && command -v srun >/dev/null 2>&1; then
  exec srun --unbuffered --cpu-bind=none "${PY_CMD[@]}"
fi
exec "${PY_CMD[@]}"
