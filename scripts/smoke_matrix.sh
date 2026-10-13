#!/bin/bash
# Smoke the full Stage C matrix (distance x clustering x evaluator) on a tiny budget,
# then validate every run directory.
#
# Usage (from project root):
#   bash scripts/smoke_matrix.sh
#   EVALUATORS=google DISTANCES="embedding nli" bash scripts/smoke_matrix.sh
#
# Defaults keep each cell to ~1 min with the 1B GGUF: 6 seeds, budget 14, cm operators.

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

if [ -d "venv" ]; then
    # shellcheck source=/dev/null
    source venv/bin/activate
fi
if [ -f ".env" ]; then
    set -a
    # shellcheck source=/dev/null
    source .env
    set +a
fi

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="${SCRIPT_DIR}/src"

EVALUATORS="${EVALUATORS:-google openai}"
DISTANCES="${DISTANCES:-embedding nli embedding+objective}"
CLUSTERINGS="${CLUSTERINGS:-leader_follower dbscan}"
MAX_TOTAL_GENOMES="${MAX_TOTAL_GENOMES:-14}"
SEED_FILE="${SEED_FILE:-data/smoke/prompt_6.csv}"
OPERATORS="${OPERATORS:-cm}"
SEED="${SEED:-42}"
SPECIES_CAPACITY="${SPECIES_CAPACITY:-4}"
CELL_TIMEOUT="${CELL_TIMEOUT:-600}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
ROOT="data/outputs/smoke_matrix_${RUN_TS}"
mkdir -p "$ROOT"

summary="${ROOT}/summary.tsv"
printf "evaluator\tclustering\tdistance\texit_code\tvalidate\tseconds\n" > "$summary"

for ev in $EVALUATORS; do
    for cl in $CLUSTERINGS; do
        for dm in $DISTANCES; do
            tag="${ev}_${cl}_${dm//+/-}"
            out="${ROOT}/${tag}"
            echo "=== ${tag}"
            start=$(date +%s)
            "$PYTHON" scripts/_run_with_timeout.py "$CELL_TIMEOUT" -- "$PYTHON" src/main.py \
                --evaluator "$ev" \
                --clustering-method "$cl" \
                --distance-method "$dm" \
                --max-total-genomes "$MAX_TOTAL_GENOMES" \
                --seed-file "$SEED_FILE" \
                --seed "$SEED" \
                --operators "$OPERATORS" \
                --max-variants 1 \
                --species-capacity "$SPECIES_CAPACITY" \
                --output-dir "$out" \
                > "${out}.log" 2>&1
            rc=$?
            secs=$(( $(date +%s) - start ))
            if "$PYTHON" scripts/validate_run_outputs.py "$out" > "${out}.validate.txt" 2>&1; then
                v=PASS
            else
                v=FAIL
            fi
            printf "%s\t%s\t%s\t%s\t%s\t%s\n" "$ev" "$cl" "$dm" "$rc" "$v" "$secs" >> "$summary"
            echo "    exit=${rc} validate=${v} ${secs}s"
        done
    done
done

echo ""
column -t -s $'\t' "$summary"
echo "Outputs: ${ROOT}"
