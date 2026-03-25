#!/bin/bash -l
#
# Slurm job for RainbowPlus (single-node vLLM + Perspective).
# Submit from this directory:
#   cd /path/to/rainbowplus-main && sbatch sbatch_rainbowplus.sh
#
# Override experiment knobs without editing the script:
#   sbatch --export=ALL,MAX_GENOMES=10000,DATASET=./data/toxsearch_seed.jsonl sbatch_rainbowplus.sh

#SBATCH --job-name=rainbowplus
#SBATCH --time=24:00:00
#SBATCH --output=logs-slurm/%x_%j.out
#SBATCH --error=logs-slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

##SBATCH --account=YOUR_ACCOUNT
##SBATCH --partition=gpu
##SBATCH --constraint=a100

set -euo pipefail

RP_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$RP_ROOT"

mkdir -p logs-slurm

if [ -f .env ]; then
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
fi

# Uncomment and adjust for your site:
# source ~/.bashrc && conda activate rainbowplus
# source "${RP_ROOT}/venv/bin/activate"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${OMP_NUM_THREADS}"

nvidia-smi || true

CONFIG_FILE="${CONFIG_FILE:-configs/base.yml}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
MAX_ITERS="${MAX_ITERS:-20000}"
MAX_GENOMES="${MAX_GENOMES:-5000}"
NUM_MUTATIONS="${NUM_MUTATIONS:-100}"
FITNESS_THRESHOLD="${FITNESS_THRESHOLD:-0.3}"
LOG_INTERVAL="${LOG_INTERVAL:-50}"
LOG_DIR="${LOG_DIR:-./logs}"

CMD=(
  python -m rainbowplus.rainbowplus
  --config_file "$CONFIG_FILE"
  --num_samples "$NUM_SAMPLES"
  --max_iters "$MAX_ITERS"
  --max_genomes "$MAX_GENOMES"
  --num_mutations "$NUM_MUTATIONS"
  --fitness_threshold "$FITNESS_THRESHOLD"
  --log_dir "$LOG_DIR"
  --log_interval "$LOG_INTERVAL"
)

if [ -n "${DATASET:-}" ]; then
  CMD+=(--dataset "$DATASET")
fi

if [ -n "${TARGET_LLM:-}" ]; then
  CMD+=(--target_llm "$TARGET_LLM")
fi

if [ -n "${NO_SHUFFLE:-}" ]; then
  CMD+=(--no-shuffle)
fi

echo "Working directory: $RP_ROOT"
echo "Command: ${CMD[*]}"
exec "${CMD[@]}"
