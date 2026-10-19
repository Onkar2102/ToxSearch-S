#!/bin/bash -l
#
# SLURM / local one-command RQ1 driver (Part 1 + optional Part 2).
# Resume-safe: NLI checkpoints continue; finished batch grids/pilots are skipped.
#
# Sync experiments/evostar_2027/data/ (pool+batches+cheap mats + NLI checkpoints)
# to the cluster first. Do NOT use SMOKE=1 on the cluster.
#
# ---------------------------------------------------------------------------
# RQ1 Part 1 only (offline). Live 40-cell factorial is rc_rq2.sh.
# ---------------------------------------------------------------------------
#   sbatch rc_rq1.sh
#
# Explicit Part 1::
#   STAGES=nli,materialize,p1d,p1e,finalize RUN_PILOTS=0 \
#   NLI_MAX_N=2000 BATCH_SIZE=256 RUN_P1D=1 \
#   RQ1_BATCHES=500,1000,2000,3139 T_MIN=0.05 T_MAX=0.5 T_STEP=0.05 \
#   FREEZE_BATCH=500 \
#   FREEZE_CLI_METHODS=embedding,nli_entail \
#   FREEZE_CLUSTERING_METHODS=leader_follower,dbscan \
#   PILOT_MAX_GENOMES=2000 \
#   sbatch rc_rq1.sh
#
# Live RQ2 (do NOT use this script)::
#   sbatch rc_rq2.sh
#
# Local smoke::
#   SMOKE=1 bash rc_rq1.sh
#
# ---------------------------------------------------------------------------
# SLURM resource budget (Part 1 only)
# ---------------------------------------------------------------------------
#   NLI stays at N=2000 (B3139 has cheap mats only; no full-pool NLI unless
#     NLI_MAX_N=3139 ALLOW_LARGE_N=1 — multi-day GPU).
#   Materialize + B1000/B2000/B3139 grids (B500 already done): ~1–4 h
#   Finalize / freeze: ~1 h
#   Total ≈ 4–8 h → request 1d (buffer for queue/GPU hiccups).
#
#SBATCH --job-name=rq1_offline
#SBATCH --time=01-00:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32g
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

# Resume remaining Part 1: NLI → attach → grids → freeze → paper (no live EA)
if [ -f "experiments/evostar_2027/data/pool/p0_full.pkl" ] \
   && [ -f "experiments/evostar_2027/data/batches/B2000/p0.pkl" ] \
   && [ -f "experiments/evostar_2027/data/mats/B2000/d_E.npy" ]; then
    DEFAULT_STAGES="nli,materialize,p1d,p1e,finalize"
else
    DEFAULT_STAGES="all"
fi
STAGES="${STAGES:-$DEFAULT_STAGES}"

NLI_MAX_N="${NLI_MAX_N:-2000}"
ALLOW_LARGE_N="${ALLOW_LARGE_N:-0}"
RUN_P1D="${RUN_P1D:-1}"
# Nested prefixes of p0_full; 3139 = full pool (cheap d_E/d_P/d_EP; NLI only ≤2000)
RQ1_BATCHES="${RQ1_BATCHES:-500,1000,2000,3139}"
REF_RADIUS="${REF_RADIUS:-0.1}"
T_MIN="${T_MIN:-0.05}"
T_MAX="${T_MAX:-0.5}"
T_STEP="${T_STEP:-0.05}"
FREEZE_BATCH="${FREEZE_BATCH:-500}"
# Keep frozen_config aligned with RQ2 live shortlist when p1e runs
FREEZE_CLI_METHODS="${FREEZE_CLI_METHODS:-embedding,nli_entail}"
FREEZE_CLUSTERING_METHODS="${FREEZE_CLUSTERING_METHODS:-leader_follower,dbscan}"

# Live EA is rc_rq2.sh — keep pilots off here
RUN_PILOTS="${RUN_PILOTS:-0}"
PILOT_RUN="${PILOT_RUN:-0}"
PILOT_DISTANCES="${PILOT_DISTANCES:-auto}"
PILOT_CLUSTERINGS="${PILOT_CLUSTERINGS:-auto}"
PILOT_SEEDS="${PILOT_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
PILOT_MAX_GENOMES="${PILOT_MAX_GENOMES:-2000}"

NLI_WHICH="${NLI_WHICH:-both}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
FORCE="${FORCE:-0}"

if [ "$SMOKE" = "1" ]; then
    echo "WARNING: SMOKE=1 is for local wiring only — do not use for cluster campaign runs." >&2
    NLI_MAX_N="${NLI_MAX_N:-50}"
    RUN_P1D="${RUN_P1D:-0}"
    RUN_PILOTS="${RUN_PILOTS:-0}"
    PILOT_RUN="${PILOT_RUN:-0}"
    echo "[SMOKE] NLI_MAX_N=${NLI_MAX_N} RUN_P1D=${RUN_P1D}"
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

export RUN_P1D RQ1_BATCHES REF_RADIUS T_MIN T_MAX T_STEP NLI_MAX_N
export RUN_PILOTS PILOT_RUN PILOT_DISTANCES PILOT_CLUSTERINGS PILOT_SEEDS PILOT_MAX_GENOMES FREEZE_BATCH
export FREEZE_CLI_METHODS FREEZE_CLUSTERING_METHODS

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
      --nli-model "$NLI_MODEL"
      --nli-max-n "$NLI_MAX_N"
      --batches "$RQ1_BATCHES"
      --reference-radius "$REF_RADIUS"
      --t-min "$T_MIN" --t-max "$T_MAX" --t-step "$T_STEP"
      --freeze-batch "$FREEZE_BATCH"
      --freeze-cli-methods "$FREEZE_CLI_METHODS"
      --freeze-clustering-methods "$FREEZE_CLUSTERING_METHODS"
      --pilot-distances "$PILOT_DISTANCES"
      --pilot-clusterings "$PILOT_CLUSTERINGS"
      --pilot-seeds "$PILOT_SEEDS"
      --pilot-max-genomes "$PILOT_MAX_GENOMES")

if [ "$RUN_P1D" = "1" ]; then
    ARGV+=(--run-p1d)
else
    ARGV+=(--no-p1d)
fi
if [ "$RUN_PILOTS" = "1" ]; then
    ARGV+=(--run-pilots)
fi
if [ "$PILOT_RUN" = "1" ]; then
    ARGV+=(--pilot-run)
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
if [ "$FORCE" = "1" ]; then
    ARGV+=(--force)
fi

echo "=========================================="
echo "RQ1 pipeline (resume remaining Part 1 + Part 2)"
echo "  stages=${STAGES}"
echo "  smoke=${SMOKE} dry_run=${DRY_RUN} force=${FORCE}"
echo "  nli_max_n=${NLI_MAX_N} batch_size=${BATCH_SIZE}"
echo "  run_p1d=${RUN_P1D} batches=${RQ1_BATCHES} t=${T_MIN}:${T_STEP}:${T_MAX}"
echo "  freeze_batch=${FREEZE_BATCH} freeze_cli=${FREEZE_CLI_METHODS}"
echo "  run_pilots=${RUN_PILOTS} (live EA → rc_rq2.sh)"
echo "  job=${SLURM_JOB_ID:-local}"
echo "=========================================="

"${ARGV[@]}"

echo "[DONE] RQ1 pipeline  SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
