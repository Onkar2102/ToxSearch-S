#!/usr/bin/env bash
# Run MPI test suite under mpiexec.
# Heavy integration modules are executed as scripts (stable with Open MPI 5);
# lightweight pytest modules cover speciation config + serialization.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}/src"
export PYTHONUNBUFFERED=1
PYTHON="${PYTHON:-./venv/bin/python}"
MPIEXEC="${MPIEXEC:-mpiexec}"
TIMEOUT_SEC="${MPI_TEST_TIMEOUT:-180}"

if [ -x ./venv/bin/python ]; then
  PYTHON=./venv/bin/python
fi

FAIL=0
WRAP=("$PYTHON" "$ROOT/scripts/_run_with_timeout.py" "$TIMEOUT_SEC" --)
run_step() {
  local title="$1"; shift
  echo "=== ${title} ==="
  "${WRAP[@]}" "$@"
  local rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "OK: ${title}"
    return 0
  fi
  echo "FAIL: ${title} (exit=${rc})"
  FAIL=1
  # Best-effort cleanup of orphaned ranks after a failure/timeout.
  pkill -9 -f 'prterun|tests/test_.*mpi|tests/test_phase|tests/test_mpi' 2>/dev/null || true
  sleep 1
  return 0
}

run_step "MPI pytest: speciation config (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" -m pytest tests/test_mpi_speciation_config.py -q --tb=short -m mpi

run_step "MPI script: gen0 pull-based (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_mpi_gen0.py

run_step "MPI script: phase3 (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase3_mpi.py

run_step "MPI script: phase4 (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase4_mpi.py

run_step "MPI script: phase5 (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase5_mpi.py

run_step "MPI script: phase67 (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase67_mpi.py

run_step "MPI script: phase8 integration (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase8_integration.py

run_step "MPI script: phase3_run (n=3)" \
  $MPIEXEC -n 3 "$PYTHON" tests/test_phase3_run.py

run_step "MPI pytest: phase8 serialization (n=2)" \
  $MPIEXEC -n 2 "$PYTHON" -m pytest tests/test_phase8_serialization.py -q --tb=short -m mpi

run_step "MPI script: gen0 small-queue (n=4)" \
  $MPIEXEC -n 4 "$PYTHON" tests/_run_mpi_gen0_small_queue.py

if [ "$FAIL" -ne 0 ]; then
  echo "=== MPI TESTS FINISHED WITH FAILURES ==="
  exit 1
fi
echo "=== ALL MPI TESTS PASSED ==="
