#!/bin/bash -l
#
# SLURM / local one-command RQ1 driver (Part 1 + optional Part 2).
# Mirrors rc_script.sh cluster bootstrap; runs run_rq1_pipeline.py.
#
# Sync experiments/evostar_2027/data/ (pool+batches+cheap mats) to the cluster first.
# Do NOT use SMOKE=1 on the cluster — that is a local wiring check only.
#
# Cluster (reuse synced pool/batches; run NLI @ 2000; 1d held)::
#   STAGES=nli,materialize,p1e sbatch rc_rq1.sh
#
# Or default stages: pool/materialize are no-ops if artifacts already exist::
#   sbatch rc_rq1.sh
#
# Full-pool NLI after 2000 is done::
#   NLI_MAX_N=3139 ALLOW_LARGE_N=1 STAGES=nli,materialize,p1e sbatch rc_rq1.sh
#
# Enable Part 1d + freeze::
#   RUN_P1D=1 STAGES=p1d,p1e sbatch rc_rq1.sh
#
# Part 1d⁺ independent threshold grid (0.1 step 0.05) on B500::
#   PYTHONPATH=src:. python experiments/evostar_2027/rq1/scripts/p1d_threshold_grid.py \
#     --mats-dir experiments/evostar_2027/data/mats/B500 \
#     --p0 experiments/evostar_2027/data/batches/B500/p0.pkl \
#     --sweeps-dir experiments/evostar_2027/rq1/sweeps/B500 \
#     --t-min 0.1 --t-max 0.5 --t-step 0.05
#
# Local B500 test (NLI @ 500; nested prefix of fitness-sorted pool)::
#   STAGES=nli,materialize NLI_MAX_N=500 bash rc_rq1.sh
#
# Local smoke only (minutes; never for campaign results)::
#   SMOKE=1 bash rc_rq1.sh
#
# Optional Part 2 pilots (needs GGUF + Perspective like rc_script.sh)::
#   RUN_PILOTS=1 PILOT_RUN=1 sbatch rc_rq1.sh

#SBATCH --job-name=rq1_evostar
#SBATCH --time=01-11:59:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --mail-user=slack:@U05PK8K2HEE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/os9660/Downloads/ToxSearch-S

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}"
# When invoked as `bash rc_rq1.sh` from a checkout, prefer that tree.
if [ -z "${SLURM_JOB_ID:-}" ] && [ -f "$(pwd)/experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py" ]; then
    PROJECT_ROOT="$(pwd)"
fi
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)  SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------
SMOKE="${SMOKE:-0}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CLUSTER_ENV="${SKIP_CLUSTER_ENV:-0}"
# Prefer NLI-focused stages when pool/batches already synced (avoids rebuild work).
if [ -f "experiments/evostar_2027/data/pool/p0_full.pkl" ] \
   && [ -f "experiments/evostar_2027/data/batches/B2000/p0.pkl" ] \
   && [ -f "experiments/evostar_2027/data/mats/B2000/d_E.npy" ]; then
    DEFAULT_STAGES="nli,materialize,p1e"
else
    DEFAULT_STAGES="all"
fi
STAGES="${STAGES:-$DEFAULT_STAGES}"
NLI_MAX_N="${NLI_MAX_N:-}"
ALLOW_LARGE_N="${ALLOW_LARGE_N:-0}"
RUN_P1D="${RUN_P1D:-0}"
RUN_PILOTS="${RUN_PILOTS:-0}"
PILOT_RUN="${PILOT_RUN:-0}"
NLI_WHICH="${NLI_WHICH:-both}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
FORCE="${FORCE:-0}"

if [ "$SMOKE" = "1" ]; then
    echo "WARNING: SMOKE=1 is for local wiring only — do not use for cluster campaign runs." >&2
    NLI_MAX_N="${NLI_MAX_N:-50}"
    echo "[SMOKE] NLI_MAX_N=${NLI_MAX_N}"
fi

# ---------------------------------------------------------------------------
# Cluster env (skip on local / SMOKE unless forced)
# ---------------------------------------------------------------------------
if [ -n "${SLURM_JOB_ID:-}" ] && [ "$SKIP_CLUSTER_ENV" != "1" ]; then
    spack env activate default-nlp-x86_64-25111801

    NVCC_PATH=$(command -v nvcc || true)
    if [ -z "$NVCC_PATH" ]; then
        echo "ERROR: nvcc not found after Spack activate" >&2
        exit 1
    fi
    CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
    echo "[DEBUG] CUDA_ROOT=$CUDA_ROOT"

    # Prefer cluster venv; fall back to .venv / venv
    if [ -f .spvenv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .spvenv/bin/activate
    elif [ -f .venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
    elif [ -f venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source venv/bin/activate
    else
        echo "ERROR: no .spvenv / .venv / venv found" >&2
        exit 1
    fi

    export PYTHONNOUSERSITE=1
    unset PYTHONPATH PYTHONHOME || true

    nvidia-smi || { echo "ERROR: no GPU visible"; exit 1; }
else
    echo "[LOCAL] Skipping Spack/CUDA bootstrap (SLURM_JOB_ID unset or SKIP_CLUSTER_ENV=1)"
    if [ -f venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source venv/bin/activate
    elif [ -f .venv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .venv/bin/activate
    elif [ -f .spvenv/bin/activate ]; then
        # shellcheck source=/dev/null
        source .spvenv/bin/activate
    fi
fi

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

if [ -f ".env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
PIPELINE="experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py"
if [ ! -f "$PIPELINE" ]; then
    echo "ERROR: missing $PIPELINE" >&2
    exit 1
fi
if [ ! -d "data/outputs/toxsearch_s_outputs copy" ] && [ ! -f "experiments/evostar_2027/data/pool/p0_full.pkl" ]; then
    echo "ERROR: need historic source or existing data/pool/p0_full.pkl" >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Build argv
# ---------------------------------------------------------------------------
ARGV=(python "$PIPELINE" --stages "$STAGES" --nli-which "$NLI_WHICH"
      --checkpoint-every "$CHECKPOINT_EVERY" --batch-size "$BATCH_SIZE"
      --nli-model "$NLI_MODEL")

if [ -n "$NLI_MAX_N" ]; then
    ARGV+=(--nli-max-n "$NLI_MAX_N")
fi
if [ "$SMOKE" = "1" ]; then
    ARGV+=(--smoke)
fi
if [ "$DRY_RUN" = "1" ]; then
    ARGV+=(--dry-run)
fi
if [ "$ALLOW_LARGE_N" = "1" ]; then
    ARGV+=(--allow-large-n)
fi
if [ "$RUN_P1D" = "1" ]; then
    ARGV+=(--run-p1d)
fi
if [ "$RUN_PILOTS" = "1" ]; then
    ARGV+=(--run-pilots)
fi
if [ "$PILOT_RUN" = "1" ]; then
    ARGV+=(--pilot-run)
fi
if [ "$FORCE" = "1" ]; then
    ARGV+=(--force)
fi

echo "=========================================="
echo "RQ1 pipeline"
echo "  stages=${STAGES}"
echo "  smoke=${SMOKE} dry_run=${DRY_RUN}"
echo "  nli_max_n=${NLI_MAX_N:-default} allow_large=${ALLOW_LARGE_N}"
echo "  run_p1d=${RUN_P1D} run_pilots=${RUN_PILOTS} pilot_run=${PILOT_RUN}"
echo "  provisional radii: theta_sim=eps=0.1 (until 1d)"
echo "  job=${SLURM_JOB_ID:-local}"
echo "=========================================="

"${ARGV[@]}"

echo "[DONE] RQ1 pipeline  SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
