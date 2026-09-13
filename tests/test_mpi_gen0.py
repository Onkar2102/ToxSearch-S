from __future__ import annotations

import csv
import json
import logging
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from mpi4py import MPI  # noqa: E402


def _write_seed_csv(path: Path, prompts: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["questions"])
        for p in prompts:
            w.writerow([p])


class MockResponseGenerator:
    model_name = "mock-rg"

    def generate_response(self, prompt: str):
        return (f"resp-{len(prompt)}", 0.001)


class MockPromptGenerator:
    pass


class MockEvaluator:
    _last_evaluation_time = {
        "duration": 0.01,
        "retries": 0,
        "attempt_durations": [],
        "api_wait_seconds": 0.0,
    }

    def select_key(self, idx):
        pass

    def _evaluate_text_hybrid(self, text, genome_id, moderation_methods=None):
        return {"google": {"scores": {"toxicity": 0.42}}}


def run_mpi_gen0_pull_based_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 3:
        return False, f"need >= 3 MPI ranks, got {size}"

    from speciation.config import SpeciationConfig
    from utils.population_io import set_outputs_path
    from parallel.master_worker import run as run_parallel

    tmp = None
    if rank == 0:
        tmp = tempfile.mkdtemp(prefix="toxsearch_mpi_gen0_")
    tmp = comm.bcast(tmp, root=0)
    out_path = Path(tmp)
    set_outputs_path(out_path)

    seed_path = None
    if rank == 0:
        seed_path = str(out_path / "seed.csv")
        unique = [f"unique_mpi_prompt_{i}" for i in range(10)]
        _write_seed_csv(Path(seed_path), unique + unique + unique)
    seed_path = comm.bcast(seed_path, root=0)

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(f"test_mpi_gen0_r{rank}")
    logger.setLevel(logging.WARNING)

    kwargs = dict(
        logger=logger,
        K=5,
        outputs_path=str(out_path),
        max_total_genomes=1,
        run_speciation_fn=None,
        operators_mode="cm",
        seed_file=seed_path,
        speciation_config=SpeciationConfig(),
        log_file=None,
        perspective_api_keys=["dummy-key-for-test"],
        gen0_batch_size=4,
    )
    if rank != 0:
        kwargs["response_generator"] = MockResponseGenerator()
        kwargs["prompt_generator"] = MockPromptGenerator()
        kwargs["evaluator"] = MockEvaluator()

    run_parallel(**kwargs)
    comm.Barrier()

    if rank == 0:
        reserves_path = out_path / "reserves.json"
        if not reserves_path.exists():
            return False, "reserves.json missing"
        data = json.loads(reserves_path.read_text(encoding="utf-8"))
        if len(data) != 10:
            return False, f"expected 10 genomes in reserves, got {len(data)}"

    comm.Barrier()
    return True, None


def run_mpi_gen0_small_queue_test():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if size < 4:
        return False, f"need >= 4 MPI ranks, got {size}"

    from speciation.config import SpeciationConfig
    from utils.population_io import set_outputs_path
    from parallel.master_worker import run as run_parallel

    tmp = None
    if rank == 0:
        tmp = tempfile.mkdtemp(prefix="toxsearch_mpi_gen0_small_")
    tmp = comm.bcast(tmp, root=0)
    out_path = Path(tmp)
    set_outputs_path(out_path)

    seed_path = None
    if rank == 0:
        seed_path = str(out_path / "seed_small.csv")
        _write_seed_csv(Path(seed_path), [f"small_q_{i}" for i in range(5)])
    seed_path = comm.bcast(seed_path, root=0)

    logging.basicConfig(level=logging.WARNING)
    logger = logging.getLogger(f"test_mpi_gen0_small_r{rank}")
    logger.setLevel(logging.WARNING)

    kwargs = dict(
        logger=logger,
        K=5,
        outputs_path=str(out_path),
        max_total_genomes=1,
        run_speciation_fn=None,
        operators_mode="cm",
        seed_file=seed_path,
        speciation_config=SpeciationConfig(),
        log_file=None,
        perspective_api_keys=["dummy-key-for-test"],
        gen0_batch_size=10,
    )
    if rank != 0:
        kwargs["response_generator"] = MockResponseGenerator()
        kwargs["prompt_generator"] = MockPromptGenerator()
        kwargs["evaluator"] = MockEvaluator()

    run_parallel(**kwargs)
    comm.Barrier()

    if rank == 0:
        data = json.loads((out_path / "reserves.json").read_text(encoding="utf-8"))
        if len(data) != 5:
            return False, f"expected 5 genomes, got {len(data)}"

    comm.Barrier()
    return True, None


def test_mpi_gen0_pull_based():
    import pytest
    ok, err = run_mpi_gen0_pull_based_test()
    if not ok:
        if err and err.startswith("need >="):
            pytest.skip(err)
        assert ok, err


def test_mpi_gen0_many_workers_small_queue():
    import pytest
    if MPI.COMM_WORLD.Get_size() < 4:
        pytest.skip("need at least 4 MPI ranks (mpirun -np 4)")
    ok, err = run_mpi_gen0_small_queue_test()
    assert ok, err


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    r = comm.Get_rank()
    s = comm.Get_size()
    ok, err = run_mpi_gen0_pull_based_test()
    if not ok:
        if r == 0:
            print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(1)
    if r == 0:
        print("test_mpi_gen0_pull_based PASSED")

    if s >= 4:
        ok2, err2 = run_mpi_gen0_small_queue_test()
        if not ok2:
            if r == 0:
                print(f"FAIL (small queue): {err2}", file=sys.stderr)
            sys.exit(1)
        if r == 0:
            print("test_mpi_gen0_many_workers_small_queue PASSED")
    elif r == 0:
        print("SKIP small-queue test (use mpirun -np 4 to run it)")
