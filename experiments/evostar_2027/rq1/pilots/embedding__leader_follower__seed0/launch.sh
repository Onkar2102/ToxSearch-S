#!/usr/bin/env bash
set -euo pipefail
cd /Users/onkars/Documents/Projects/ToxSearch-S
export PYTHONPATH=src:.
/Users/onkars/Documents/Projects/ToxSearch-S/venv/bin/python -m cli --seed-file data/prompt.csv --seed 0 --max-total-genomes 250 --distance embedding --clustering-method leader_follower --theta-sim 0.2491595995024483 --output-dir experiments/evostar_2027/rq1/pilots/embedding__leader_follower__seed0/run
