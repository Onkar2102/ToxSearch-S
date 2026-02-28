#!/bin/bash

#SBATCH --job-name=search1
#SBATCH --time=3-23:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=4
#SBATCH --mem=48g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --gres=gpu:a100:2

set -euo pipefail
cd /home/os9660/ToxSearch-S

module purge 2>/dev/null || true

# 1) Spack env (same one you use interactively)
spack env activate default-nlp-x86_64-25111801

# 2) Find CUDA root from nvcc and expose libs
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

# 3) Activate venv
source .spvenv/bin/activate

# 4) Python isolation
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME

# 5) Threading
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"

# 6) GPU sanity check
nvidia-smi || { echo "No GPU visible"; exit 1; }

# 7) llama-cpp smoke test
python - <<'PY'
import llama_cpp
from llama_cpp import Llama
print("llama-cpp-python:", llama_cpp.__version__)
llm = Llama(
    model_path="models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q3_K_S.gguf",
    n_ctx=1024,
    n_gpu_layers=-1,
    n_batch=256,
    verbose=False,
)
print("LLAMA_GPU_SMOKETEST_OK")
PY

cd /home/os9660/ToxSearch-S || exit 1
export PYTHONPATH=/home/os9660/ToxSearch-S/src
python /home/os9660/ToxSearch-S/src/main.py --help | grep -E "theta-sim|embedding-model|species-capacity|cluster0" || exit 1

# 8) Single MPI launch with mpiexec (from Spack env). -n 3 = 1 master + 2 workers (matches --ntasks=3).
# python -u for unbuffered output.
mpiexec -n 3 python -u src/main.py \
    --parallel \
    --batch-size 100 \
    --generations 250 \
    --threshold 0.99 \
    --moderation-methods google \
    --stagnation-limit 10 \
    --theta-sim 0.35 \
    --theta-merge 0.35 \
    --species-capacity 150 \
    --cluster0-max-capacity 1000 \
    --cluster0-min-cluster-size 1 \
    --min-island-size 5 \
    --species-stagnation 30 \
    --embedding-model all-MiniLM-L6-v2 \
    --embedding-dim 384 \
    --embedding-batch-size 64 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
    --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
    --operators all \
    --max-variants 1 \
    --seed-file data/prompt.csv

echo "All experiments completed!"
