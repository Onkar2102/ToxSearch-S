import json
import sys
import os
import csv
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ea.evolution_engine as ee_module


def _make_logger():
    import logging
    rank = MPI.COMM_WORLD.Get_rank()
    logger = logging.getLogger(f"test_p67_rank{rank}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"[rank {rank}] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def _create_seed_csv(path, prompts):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["questions"])
        for p in prompts:
            writer.writerow([p])


def _make_mock_rg():
    rg = MagicMock()
    rg.generate_response.side_effect = lambda prompt, **kw: (f"response_{prompt[:15]}", 0.05)
    rg.model_name = "mock-model"
    rg.model_cfg = {"name": "mock-model"}
    return rg


def _make_mock_evaluator():
    ev = MagicMock()
    ev._evaluate_text_hybrid.return_value = {"google": {"scores": {"toxicity": 0.6}}}
    ev._last_evaluation_time = {"duration": 0.01}
    ev.select_key = MagicMock()
    ev._api_keys = ["key0"]
    ev._active_key_index = 0
    return ev


def _mock_generate_single_variant(parents, pg, **kwargs):
    p0 = parents[0] if parents else {}
    return [{
        "prompt": f"evolved_{p0.get('prompt', 'x')[:15]}",
        "operator": "MockMutation",
        "variant_type": "mutation",
        "parents": [{"id": p.get("id"), "score": 0.5} for p in parents],
        "status": "pending_generation",
        "creation_info": {"type": "mutation", "operator": "MockMutation"},
    }]


speciation_called_with = {}


def _mock_run_speciation(temp_path=None, current_generation=0,
                          config=None, log_file=None,
                          north_star_metric="toxicity"):
    speciation_called_with["temp_path"] = temp_path
    speciation_called_with["generation"] = current_generation
    speciation_called_with["config"] = config
    speciation_called_with["call_count"] = speciation_called_with.get("call_count", 0) + 1

    temp = Path(temp_path)
    reserves_path = temp.parent / "reserves.json"

    with open(temp, "r", encoding="utf-8") as f:
        genomes = json.load(f)

    existing = []
    if reserves_path.exists():
        with open(reserves_path, "r", encoding="utf-8") as f:
            existing = json.load(f)

    for g in genomes:
        g.setdefault("species_id", 0)
    existing.extend(genomes)

    with open(reserves_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    with open(temp, "w", encoding="utf-8") as f:
        json.dump([], f)

    return {
        "success": True,
        "species_count": 1,
        "reserves_size": len(existing),
        "elites_moved": 0,
    }


                                                                            

def test_gen0_partial_speciation(comm, rank, size, logger):

    tmpdir = None
    seed_csv = None

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p67_partial_")
        seed_csv = os.path.join(tmpdir, "seed.csv")
        _create_seed_csv(seed_csv, [
            "What is the meaning of life?",
            "Explain machine learning.",
        ])

    tmpdir = comm.bcast(tmpdir, root=0)
    seed_csv = comm.bcast(seed_csv, root=0)
    outputs_dir = os.path.join(tmpdir, "outputs")

    mock_rg = _make_mock_rg()
    mock_pg = MagicMock()
    mock_evaluator = _make_mock_evaluator()

    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    speciation_called_with.clear()

    try:
        from parallel.master_worker import run

        run(
            logger,
            K=100,
            outputs_path=outputs_dir,
            north_star_metric="toxicity",
            speciation_config=None,
            log_file=None,
            max_generations=2,
            run_speciation_fn=_mock_run_speciation,
            operators_mode="all",
            seed_file=seed_csv,
            moderation_methods=["google"],
            response_generator=mock_rg,
            prompt_generator=mock_pg,
            evaluator=mock_evaluator,
            perspective_api_keys=["key0"],
        )

        if rank == 0:
            out = Path(outputs_dir)

            assert speciation_called_with.get("call_count", 0) >= 1, \
                "Speciation was never called despite Gen 0 completion"
            logger.info("Speciation called %d times", speciation_called_with["call_count"])

            tracker_path = out / "EvolutionTracker.json"
            assert tracker_path.exists(), "EvolutionTracker.json missing"
            with open(tracker_path) as f:
                tracker = json.load(f)
            assert tracker["total_generations"] >= 1, \
                f"Expected >= 1 generation, got {tracker['total_generations']}"

            reserves_path = out / "reserves.json"
            with open(reserves_path) as f:
                reserves = json.load(f)
            assert len(reserves) >= 1, \
                f"Expected >= 1 genome in reserves, got {len(reserves)}"

            temp_path = out / "temp.json"
            with open(temp_path) as f:
                temp = json.load(f)
            assert len(temp) == 0, f"temp.json should be empty, got {len(temp)}"

            logger.info("PASS: Gen 0 partial speciation (2 prompts, K=100)")

        else:
            logger.info("Worker %d: PASS", rank)

    finally:
        ee_module.generate_single_variant = original_gsv

    comm.Barrier()

    if rank == 0 and tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("Cleaned up %s", tmpdir)


def test_cli_args_flow(comm, rank, size, logger):

    tmpdir = None
    seed_csv = None

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p67_cli_")
        seed_csv = os.path.join(tmpdir, "seed.csv")
        _create_seed_csv(seed_csv, [
            f"prompt_{i}" for i in range(6)
        ])

    tmpdir = comm.bcast(tmpdir, root=0)
    seed_csv = comm.bcast(seed_csv, root=0)
    outputs_dir = os.path.join(tmpdir, "outputs")

    mock_rg = _make_mock_rg()
    mock_pg = MagicMock()
    mock_evaluator = _make_mock_evaluator()

    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    speciation_called_with.clear()

    try:
        from parallel.master_worker import run

        run(
            logger,
            K=4,
            outputs_path=outputs_dir,
            north_star_metric="toxicity",
            speciation_config={"custom": True},
            log_file=None,
            max_generations=2,
            run_speciation_fn=_mock_run_speciation,
            operators_mode="all",
            seed_file=seed_csv,
            moderation_methods=["google"],
            response_generator=mock_rg,
            prompt_generator=mock_pg,
            evaluator=mock_evaluator,
            perspective_api_keys=["key0"],
        )

        if rank == 0:
            out = Path(outputs_dir)

            assert speciation_called_with.get("call_count", 0) >= 1, \
                "run_speciation_fn was never called"
            assert speciation_called_with.get("config") == {"custom": True}, \
                f"speciation_config not passed through: {speciation_called_with.get('config')}"

            tracker_path = out / "EvolutionTracker.json"
            assert tracker_path.exists()
            with open(tracker_path) as f:
                tracker = json.load(f)
            assert tracker["total_generations"] >= 1

            reserves_path = out / "reserves.json"
            with open(reserves_path) as f:
                reserves = json.load(f)
            assert len(reserves) >= 1

            logger.info("PASS: CLI args flow through correctly")

        else:
            logger.info("Worker %d: PASS", rank)

    finally:
        ee_module.generate_single_variant = original_gsv

    comm.Barrier()

    if rank == 0 and tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)
        logger.info("Cleaned up %s", tmpdir)


                                                                            

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = _make_logger()

    if size < 2:
        logger.error("Need >= 2 ranks. Run with: mpiexec -n 3 python tests/test_phase67_mpi.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting Phase 6+7 MPI integration test  (size=%d)", size)
    logger.info("=" * 60)

    test_gen0_partial_speciation(comm, rank, size, logger)

    comm.Barrier()

    test_cli_args_flow(comm, rank, size, logger)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL PHASE 6+7 TESTS PASSED")
        logger.info("=" * 60)
