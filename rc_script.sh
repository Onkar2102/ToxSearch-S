#!/bin/bash -l

#SBATCH --job-name=txs141
#SBATCH --time=02-11:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --mail-user=slack:@U05PK8K2HEE
#SBATCH --mail-type=ALL
# Sequential single process: one GPU (parallel/MPI would request more elsewhere).
#SBATCH --gres=gpu:a100:1

set -euo pipefail
cd /home/os9660/ToxSearch-S

# module purge 2>/dev/null || true

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
    model_path="models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf",
    n_ctx=1024,
    n_gpu_layers=-1,
    n_batch=1024,
    verbose=False,
)
print("LLAMA_GPU_SMOKETEST_OK")
PY


export PYTHONPATH=/home/os9660/ToxSearch-S/src
python /home/os9660/ToxSearch-S/src/main.py --help | grep -E "theta-sim|embedding-model|species-capacity|cluster0|min-stability" || exit 1

# 9) Sequential ToxSearch-S (single process, no --parallel). Termination: --max-total-genomes.
#    NUM_RUNS repetitions; same --seed 42 each run. Output dirs from main.py (timestamped).
NUM_RUNS="${NUM_RUNS:-10}"

for ((i = 1; i <= NUM_RUNS; i++)); do
    echo ""
    echo "=========================================="
    echo "Run ${i}/${NUM_RUNS}  (seed=42)"
    echo "=========================================="

    srun python src/main.py \
        --profile \
        --max-total-genomes 1000 \
        --moderation-methods google \
        --stagnation-limit 5 \
        --theta-sim 0.25 \
        --theta-merge 0.25 \
        --min-stability-gens 5 \
        --species-capacity 100 \
        --cluster0-max-capacity 1000 \
        --cluster0-min-cluster-size 1 \
        --min-island-size 3 \
        --species-stagnation 20 \
        --embedding-model all-MiniLM-L6-v2 \
        --embedding-dim 384 \
        --embedding-batch-size 64 \
        --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
        --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
        --operators all \
        --max-variants 1 \
        --seed-file data/prompt.csv \
        --seed 42
done

echo "All ${NUM_RUNS} experiment runs completed!"
