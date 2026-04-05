#!/bin/bash -l
# Submit a chain of dependent jobs. SLURM needs a partition (and often an account) on every sbatch.
#
# Usage:
#   sbatch dependency_parallel.sh          # wrong: this file is not a job script
#   bash dependency_parallel.sh            # correct
#
# Override defaults:
#   TXS_SLURM_PARTITION=compute TXS_SLURM_ACCOUNT=myacct bash dependency_parallel.sh

set -euo pipefail

JOB_SCRIPT="${JOB_SCRIPT:-script101.sh}"
PARTITION="${TXS_SLURM_PARTITION:-debug}"
ACCOUNT="${TXS_SLURM_ACCOUNT:-evostar}"

if [[ ! -f "$JOB_SCRIPT" ]]; then
  echo "ERROR: job script not found: $JOB_SCRIPT" >&2
  exit 1
fi

SBATCH_EXTRA=(--partition="$PARTITION" --account="$ACCOUNT")

echo
echo "Preparing to submit dependent jobs..."
echo "  script:    $JOB_SCRIPT"
echo "  partition: $PARTITION"
echo "  account:   $ACCOUNT"
echo

job1=$(sbatch --parsable "${SBATCH_EXTRA[@]}" "$JOB_SCRIPT")
echo "Submitted $JOB_SCRIPT as job $job1"

job2=$(sbatch --parsable "${SBATCH_EXTRA[@]}" --dependency=afterok:"$job1" "$JOB_SCRIPT")
echo "Submitted $JOB_SCRIPT as job $job2 (afterok:$job1)"

job3=$(sbatch --parsable "${SBATCH_EXTRA[@]}" --dependency=afterok:"$job2" "$JOB_SCRIPT")
echo "Submitted $JOB_SCRIPT as job $job3 (afterok:$job2)"

job4=$(sbatch --parsable "${SBATCH_EXTRA[@]}" --dependency=afterok:"$job3" "$JOB_SCRIPT")
echo "Submitted $JOB_SCRIPT as job $job4 (afterok:$job3)"

echo
echo "Done submitting dependent jobs!"
