#!/bin/bash -l
#
# RQ2 cell: d_E + incremental DBSCAN, 10 seeds, N=2000, OpenAI.
# Lock: ε=0.10, MinPts=2.  sbatch rc_rq2_de_incdbscan.sh
#
# Pin the OpenAI key in rc_rq2_openai_key.sh (all four cells share it).
# Or set OPENAI_API_KEY_PIN below; a non-empty value wins.

#SBATCH --job-name=rq2_de_inc
#SBATCH --time=05-12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30g
#SBATCH --account=evostar
#SBATCH --partition=tier3
#SBATCH --mail-user=slack:@U05PK8K2HEE
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:a100:1
#SBATCH --chdir=/home/os9660/Downloads/ToxSearch-S
##SBATCH --array=0-9%2

OPENAI_API_KEY_PIN="${OPENAI_API_KEY_PIN:-}"

CELL_TAG="INC_dE_oai_t010_ms2_n2000"
DISTANCE_METHOD="embedding"
CLUSTERING_METHOD="dbscan"
THETA_SIM="0.1"
DBSCAN_EPS="0.1"
DBSCAN_MIN_SAMPLES="2"
OUTPUT_ROOT="${OUTPUT_ROOT:-data/outputs/rq2_inc_embedding_openai_t010_ms2_n2000}"

# shellcheck source=rc_rq2_common.sh
source "${PROJECT_ROOT:-/home/os9660/Downloads/ToxSearch-S}/rc_rq2_common.sh"
