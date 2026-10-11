#!/usr/bin/env bash
set -eo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH=src
set +u
set -a
# shellcheck disable=SC1091
source .env
set +a

GGUF=models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-Q4_K_S.gguf
COMMON=(
  --parallel
  --max-total-genomes 300
  --batch-size 24
  --evaluator google
  --north-star-metric toxicity
  --seed-file data/prompt.csv
  --rg "$GGUF"
  --pg "$GGUF"
  --theta-sim 0.25
  --species-capacity 100
  --max-variants 1
  --operators all
)

rm -rf data/outputs/run_300_mpi_lf data/outputs/run_300_mpi_dbscan

echo "======== MPI leader_follower 300 ========"
/opt/homebrew/bin/mpiexec -n 3 ./venv/bin/python src/main.py "${COMMON[@]}" \
  --clustering-method leader_follower \
  --output-dir data/outputs/run_300_mpi_lf \
  2>&1 | tee data/outputs/run_300_mpi_lf_run.log

echo "======== MPI dbscan 300 ========"
/opt/homebrew/bin/mpiexec -n 3 ./venv/bin/python src/main.py "${COMMON[@]}" \
  --clustering-method dbscan \
  --output-dir data/outputs/run_300_mpi_dbscan \
  2>&1 | tee data/outputs/run_300_mpi_dbscan_run.log

echo "======== BOTH 300-GENOME RUNS DONE ========"
