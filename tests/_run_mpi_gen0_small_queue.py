"""Run gen0 small-queue MPI test: mpiexec -n 4 python tests/_run_mpi_gen0_small_queue.py"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from mpi4py import MPI

spec = importlib.util.spec_from_file_location(
    "test_mpi_gen0", ROOT / "tests" / "test_mpi_gen0.py"
)
mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mod)

ok, err = mod.run_mpi_gen0_small_queue_test()
rank = MPI.COMM_WORLD.Get_rank()
if not ok:
    if rank == 0:
        print(f"FAIL: {err}", file=sys.stderr)
    raise SystemExit(1)
if rank == 0:
    print("test_mpi_gen0_many_workers_small_queue PASSED")
