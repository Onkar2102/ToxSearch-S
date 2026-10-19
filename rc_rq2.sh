#!/bin/bash -l
#
# SLURM / local RQ2 live factorial driver.
#   2 distances (embedding=d_E, nli_entail=d_negE)
# × 2 speciation (leader_follower, dbscan)
# × 10 seeds (0..9)
# = 40 cells @ N_max=2000
#
# Resume-safe: finished cells are skipped.
#
# ---------------------------------------------------------------------------
# Recommended (cluster)
# ---------------------------------------------------------------------------
#   sbatch rc_rq2.sh
#
# Dry-run stubs only::
#   PILOT_RUN=0 sbatch rc_rq2.sh
#
# Local dry-run::
#   PILOT_RUN=0 bash rc_rq2.sh
#
# After runs complete, analyze::
#   PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
#
# Walltime: 40 × ~N_max=2000. Embedding cells are cheaper than nli_entail.
#   Rough: embedding ~ few–10+ h/cell; nli_entail much longer (NLI calls).
#   Request 7d; raise if queue policy allows / use array jobs later.
#
#SBATCH --job-name=rq2_factorial
#SBATCH --time=07-00:00:00
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
if [ -z "${SLURM_JOB_ID:-}" ] && [ -f "$(pwd)/experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py" ]; then
    PROJECT_ROOT="$(pwd)"
fi
cd "$PROJECT_ROOT"
echo "[DEBUG] PROJECT_ROOT=$PROJECT_ROOT  pwd=$(pwd)  SLURM_JOB_ID=${SLURM_JOB_ID:-local}"

SKIP_CLUSTER_ENV="${SKIP_CLUSTER_ENV:-0}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"

# RQ2 factorial (locked)
STAGES="${STAGES:-p1e,pilots}"
RUN_P1D="${RUN_P1D:-1}"
RUN_PILOTS="${RUN_PILOTS:-1}"
PILOT_RUN="${PILOT_RUN:-1}"
FREEZE_BATCH="${FREEZE_BATCH:-500}"
FREEZE_CLI_METHODS="${FREEZE_CLI_METHODS:-embedding,nli_entail}"
FREEZE_CLUSTERING_METHODS="${FREEZE_CLUSTERING_METHODS:-leader_follower,dbscan}"
PILOT_DISTANCES="${PILOT_DISTANCES:-embedding,nli_entail}"
PILOT_CLUSTERINGS="${PILOT_CLUSTERINGS:-leader_follower,dbscan}"
PILOT_SEEDS="${PILOT_SEEDS:-0,1,2,3,4,5,6,7,8,9}"
PILOT_MAX_GENOMES="${PILOT_MAX_GENOMES:-2000}"

# Unused by STAGES=p1e,pilots but required by pipeline argv
NLI_MAX_N="${NLI_MAX_N:-2000}"
RQ1_BATCHES="${RQ1_BATCHES:-500}"
REF_RADIUS="${REF_RADIUS:-0.1}"
T_MIN="${T_MIN:-0.05}"
T_MAX="${T_MAX:-0.5}"
T_STEP="${T_STEP:-0.05}"
NLI_WHICH="${NLI_WHICH:-both}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-10}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NLI_MODEL="${NLI_MODEL:-cross-encoder/nli-deberta-v3-small}"
ALLOW_LARGE_N="${ALLOW_LARGE_N:-0}"

if [ -n "${SLURM_JOB_ID:-}" ] && [ "$SKIP_CLUSTER_ENV" != "1" ]; then
    spack env activate default-nlp-x86_64-25111801
    NVCC_PATH=$(command -v nvcc || true)
    if [ -z "$NVCC_PATH" ]; then
        echo "ERROR: nvcc not found after Spack activate" >&2
        exit 1
    fi
    CUDA_ROOT=$(dirname "$(dirname "$NVCC_PATH")")
    export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:$CUDA_ROOT/lib:${LD_LIBRARY_PATH:-}"
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
    echo "[LOCAL] Skipping Spack/CUDA bootstrap"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export NUMEXPR_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
export TF_CPP_MIN_LOG_LEVEL=3
export TRANSFORMERS_NO_TF=1
export TRANSFORMERS_NO_FLAX=1
export KERAS_BACKEND="torch"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"

export RUN_P1D RUN_PILOTS PILOT_RUN PILOT_DISTANCES PILOT_CLUSTERINGS
export PILOT_SEEDS PILOT_MAX_GENOMES FREEZE_BATCH
export FREEZE_CLI_METHODS FREEZE_CLUSTERING_METHODS

if [ -f ".env" ]; then
    set +u
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
    set -u
fi
if [ -n "${PERSPECTIVE_API_KEY:-}" ]; then
    export PERSPECTIVE_API_KEY
    unset PERSPECTIVE_API_KEYS || true
fi
if [ -z "${PERSPECTIVE_API_KEY:-}" ] && [ -z "${PERSPECTIVE_API_KEYS:-}" ]; then
    echo "ERROR: PERSPECTIVE_API_KEY or PERSPECTIVE_API_KEYS required in .env" >&2
    exit 1
fi

PIPELINE="experiments/evostar_2027/rq1/scripts/run_rq1_pipeline.py"
FROZEN="experiments/evostar_2027/rq1/frozen_config.json"
if [ ! -f "$PIPELINE" ]; then
    echo "ERROR: missing $PIPELINE" >&2
    exit 1
fi
if [ ! -f "data/prompt.csv" ]; then
    echo "ERROR: missing data/prompt.csv" >&2
    exit 1
fi
# Stale dry-run stub from earlier wiring (wrong N/theta) — do not treat as done
STALE_PILOT="experiments/evostar_2027/rq1/pilots/embedding__leader_follower__seed0"
if [ -f "${STALE_PILOT}/cell.json" ] && [ ! -d "${STALE_PILOT}/run" ]; then
    mode=$(python -c "import json;print(json.load(open('${STALE_PILOT}/cell.json')).get('mode',''))" 2>/dev/null || true)
    if [ "$mode" = "dry-run" ]; then
        echo "[INFO] removing stale dry-run stub ${STALE_PILOT}"
        rm -rf "$STALE_PILOT"
    fi
fi

ARGV=(python "$PIPELINE" --stages "$STAGES"
      --nli-which "$NLI_WHICH"
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
if [ "$DRY_RUN" = "1" ]; then
    ARGV+=(--dry-run)
fi
if [ "$FORCE" = "1" ]; then
    ARGV+=(--force)
fi

echo "=========================================="
echo "RQ2 factorial live grid"
echo "  stages=${STAGES}"
echo "  distances=${PILOT_DISTANCES}"
echo "  clusterings=${PILOT_CLUSTERINGS}"
echo "  seeds=${PILOT_SEEDS}"
echo "  N_max=${PILOT_MAX_GENOMES}  (40 cells)"
echo "  freeze_cli=${FREEZE_CLI_METHODS}"
echo "  pilot_run=${PILOT_RUN} force=${FORCE} dry_run=${DRY_RUN}"
echo "  frozen=${FROZEN}"
echo "  job=${SLURM_JOB_ID:-local}"
echo "=========================================="

"${ARGV[@]}"

echo "[DONE] RQ2 factorial  SLURM_JOB_ID=${SLURM_JOB_ID:-n/a}"
echo "Analyze with: PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py"
