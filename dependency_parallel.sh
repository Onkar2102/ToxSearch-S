#!/bin/bash -l
set -euo pipefail

echo
echo "Preparing to submit dependent jobs (script141.sh × 5)..."
echo

job1=$(sbatch --parsable script141.sh)
echo "Submitted job $job1"

job2=$(sbatch --parsable --dependency=afterany:"$job1" script141.sh)
echo "Submitted job $job2 (afterany:$job1)"

job3=$(sbatch --parsable --dependency=afterany:"$job2" script141.sh)
echo "Submitted job $job3 (afterany:$job2)"

job4=$(sbatch --parsable --dependency=afterany:"$job3" script141.sh)
echo "Submitted job $job4 (afterany:$job3)"

job5=$(sbatch --parsable --dependency=afterany:"$job4" script141.sh)
echo "Submitted job $job5 (afterany:$job4)"

echo
echo "Done submitting dependent jobs!"
