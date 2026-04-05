#!/bin/bash -l
#
# SLURM run script for RainbowPlus (single-node vLLM + Google Perspective).
# Spack is OFF by default: activating the NLP Spack view can inject libcrypto.so that
# breaks system Python 3.9 in .venv-rainbow (ImportError: OPENSSL_* / _hashlib). Use USE_SPACK=1
# only if you use a Spack-built Python or accept fixing LD_LIBRARY_PATH.
#
# Slurm copies this script to /var/spool/slurmd/job*/ — do not trust BASH_SOURCE for the repo path.
# Submit from the repo root. We resolve RP_ROOT by checking (in order) RAINBOWPLUS_ROOT,
# SLURM_SUBMIT_DIR, then the job's initial $PWD (often same as submit dir when you cd first).
# If your site omits SLURM_SUBMIT_DIR, initial PWD still usually works. Last resort: set
#   sbatch --export=ALL,RAINBOWPLUS_ROOT=/absolute/path/to/rainbowplus-main
# Or add once:  #SBATCH --chdir=/absolute/path/to/rainbowplus-main
#
# Python env: activates ${RP_ROOT}/.venv-rainbow when present (override with VENV_ACTIVATE).
#
# Parent ToxSearch-S tree (optional): if this file lives in .../ToxSearch-S/rainbowplus-main/,
# we auto-use ../.spvenv when present (same as running rc_script.sh from ToxSearch-S root).

#SBATCH --job-name=rainbowplus
#SBATCH --time=0-03:00:00
# Slurm logs under ./logs/ (paths relative to submission dir — run sbatch from repo root).
#SBATCH --output=logs/slurm_%x_%j.out
#SBATCH --error=logs/slurm_%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=9G
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:1
## If Slurm never sets SLURM_SUBMIT_DIR / wrong cwd, set repo once (absolute path):
##SBATCH --chdir=/path/to/rainbowplus-main
## Job array (optional): #SBATCH --array=0-9

set -euo pipefail

# Capture before any cd (Slurm usually starts the job in your submit directory).
START_PWD=$(pwd)

_is_rainbow_repo() {
  [[ -n "$1" && -d "$1/rainbowplus" && -f "$1/configs/base.yml" ]]
}

# ------------------------------------------------------------------------------
# 0) Repo root (Slurm batch script lives in spool; BASH_SOURCE is not the git checkout)
# ------------------------------------------------------------------------------
RP_ROOT=""
if [[ -n "${RAINBOWPLUS_ROOT:-}" ]] && _is_rainbow_repo "${RAINBOWPLUS_ROOT}"; then
  RP_ROOT=$(cd "${RAINBOWPLUS_ROOT}" && pwd)
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]] && _is_rainbow_repo "${SLURM_SUBMIT_DIR}"; then
  RP_ROOT=$(cd "${SLURM_SUBMIT_DIR}" && pwd)
elif _is_rainbow_repo "${START_PWD}"; then
  RP_ROOT=$(cd "${START_PWD}" && pwd)
else
  _from_script=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  if _is_rainbow_repo "${_from_script}"; then
    RP_ROOT="${_from_script}"
  fi
fi

if [[ -z "${RP_ROOT}" ]]; then
  echo "ERROR: Could not find RainbowPlus repo (need rainbowplus/ + configs/base.yml)." >&2
  echo "  SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}  START_PWD=${START_PWD}  BASH_SOURCE=${BASH_SOURCE[0]}" >&2
  echo "  Fix: sbatch --export=ALL,RAINBOWPLUS_ROOT=/path/to/rainbowplus-main ..." >&2
  echo "  Or add: #SBATCH --chdir=/path/to/rainbowplus-main" >&2
  exit 1
fi

cd "$RP_ROOT" || { echo "ERROR: cannot cd to RP_ROOT=$RP_ROOT" >&2; exit 1; }

# Parent project root (ToxSearch-S when layout is .../ToxSearch-S/rainbowplus-main/)
TOX_ROOT="${TOX_ROOT:-$(cd "${RP_ROOT}/.." && pwd)}"

# ------------------------------------------------------------------------------
# 1) Spack env — activation commented out (OpenSSL / system-Python venv conflicts).
#    Uncomment the block below to re-enable; then USE_SPACK=1 will matter again.
# ------------------------------------------------------------------------------
echo "[INFO] Spack env activation disabled in this script (block commented out)."
# if [[ "${USE_SPACK:-0}" == "1" ]] && command -v spack >/dev/null 2>&1; then
#   SPACK_ENV_NAME="${SPACK_ENV_NAME:-default-nlp-x86_64-25111801}"
#   # shellcheck disable=SC1090
#   spack env activate "${SPACK_ENV_NAME}"
# else
#   echo "[INFO] Spack skipped (USE_SPACK=${USE_SPACK:-0}). Set USE_SPACK=1 to match rc_script.sh."
# fi

# ------------------------------------------------------------------------------
# 2) CUDA toolkit (optional at runtime: vLLM/torch use the driver + bundled runtimes)
# Default SKIP_NVCC_CHECK=1 — no nvcc required. Set SKIP_NVCC_CHECK=0 to fail if nvcc
# missing (rc_script.sh style). If libcudart errors appear, run: module load cuda/...
# ------------------------------------------------------------------------------
NVCC_PATH=$(command -v nvcc || true)
if [[ -n "${NVCC_PATH}" ]]; then
  CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
  export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
  echo "[DEBUG] NVCC_PATH=$NVCC_PATH"
  echo "[DEBUG] CUDA_ROOT=$CUDA_ROOT"
  echo "[DEBUG] LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
elif [[ "${SKIP_NVCC_CHECK:-1}" == "1" ]]; then
  echo "[INFO] nvcc not in PATH; continuing (SKIP_NVCC_CHECK=1 default). Add a cuda module or Spack if GPU libs fail."
else
  echo "ERROR: nvcc not in PATH. Set SKIP_NVCC_CHECK=1 or load CUDA (e.g. module load cuda)." >&2
  exit 1
fi

# ------------------------------------------------------------------------------
# 3) Python venv (VENV_ACTIVATE override, else repo .venv-rainbow, then fallbacks)
# ------------------------------------------------------------------------------
if [[ -n "${VENV_ACTIVATE:-}" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_ACTIVATE}"
elif [[ -f "${RP_ROOT}/.venv-rainbow/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${RP_ROOT}/.venv-rainbow/bin/activate"
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
  echo "ERROR: No venv found. Create ${RP_ROOT}/.venv-rainbow or set VENV_ACTIVATE=/path/to/venv/bin/activate" >&2
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
NUM_MUTATIONS="${NUM_MUTATIONS:-2}"
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