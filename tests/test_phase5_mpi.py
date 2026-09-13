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
    logger = logging.getLogger(f"test_p5_rank{rank}")
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
    rg.generate_response.side_effect = lambda prompt, **kw: (f"response_for_{prompt[:20]}", 0.05)
    rg.model_name = "mock-model"
    rg.model_cfg = {"name": "mock-model"}
    return rg


_eval_call_count = 0


def _make_failing_evaluator():
    ev = MagicMock()

    def _eval_hybrid(text, genome_id, moderation_methods=None):
        global _eval_call_count
        _eval_call_count += 1
        if _eval_call_count % 3 == 0:
            raise RuntimeError("Simulated Perspective API 429 exhaustion")
        return {"google": {"scores": {"toxicity": 0.55}}}

    ev._evaluate_text_hybrid.side_effect = _eval_hybrid
    ev._last_evaluation_time = {"duration": 0.01}
    ev.select_key = MagicMock()
    ev._api_keys = ["key0", "key1"]
    ev._active_key_index = 0
    return ev


def _mock_generate_single_variant(parents, pg, **kwargs):
    p0 = parents[0] if parents else {}
    return [{
        "prompt": f"evolved_{p0.get('prompt', 'x')[:20]}",
        "operator": "MockMutation",
        "variant_type": "mutation",
        "parents": [{"id": p.get("id"), "score": 0.5} for p in parents],
        "status": "pending_generation",
        "creation_info": {"type": "mutation", "operator": "MockMutation"},
    }]


                                                                                

def test_phase5(comm, rank, size, logger):
    global _eval_call_count
    _eval_call_count = 0

    tmpdir = None
    seed_csv = None

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p5_")
        seed_csv = os.path.join(tmpdir, "seed.csv")
        _create_seed_csv(seed_csv, [
            "What is the meaning of life?",
            "Tell me about quantum computing.",
            "Explain machine learning.",
            "What are neural networks?",
            "How does gravity work?",
            "Describe photosynthesis.",
        ])

    tmpdir = comm.bcast(tmpdir, root=0)
    seed_csv = comm.bcast(seed_csv, root=0)
    outputs_dir = os.path.join(tmpdir, "outputs")

    mock_rg = _make_mock_rg()
    mock_pg = MagicMock()
    mock_evaluator = _make_failing_evaluator()

    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    try:
        from parallel.master_worker import run

        K = 4
        MAX_GENS = 2

        run(
            logger,
            K=K,
            outputs_path=outputs_dir,
            north_star_metric="toxicity",
            speciation_config=None,
            log_file=None,
            max_generations=MAX_GENS,
            run_speciation_fn=None,
            operators_mode="all",
            seed_file=seed_csv,
            moderation_methods=["google"],
            response_generator=mock_rg,
            prompt_generator=mock_pg,
            evaluator=mock_evaluator,
            perspective_api_keys=["key0", "key1"],
        )

        if rank == 0:
            out = Path(outputs_dir)

            tracker_path = out / "EvolutionTracker.json"
            assert tracker_path.exists(), "EvolutionTracker.json missing"
            with open(tracker_path) as f:
                tracker = json.load(f)

            reserves_path = out / "reserves.json"
            assert reserves_path.exists(), "reserves.json missing"
            with open(reserves_path) as f:
                reserves = json.load(f)

            temp_path = out / "temp.json"
            with open(temp_path) as f:
                temp = json.load(f)

            logger.info("Validation: reserves=%d  temp=%d  generations=%s",
                        len(reserves), len(temp), tracker.get("total_generations"))

            assert len(temp) == 0, f"temp.json should be empty after speciation, got {len(temp)}"
            assert tracker["total_generations"] >= 1, \
                f"Expected >= 1 generation, got {tracker['total_generations']}"

            for g in reserves:
                assert g.get("status") != "error", \
                    f"Error genome leaked into reserves: id={g.get('id')} error={g.get('error')}"

            ids = [g["id"] for g in reserves]
            assert len(ids) == len(set(ids)), f"Duplicate genome IDs: {ids}"

            total_discarded = sum(
                gen.get("total_discarded", 0) for gen in tracker.get("generations", []))
            logger.info("Total discarded across generations: %d", total_discarded)
            assert total_discarded > 0, \
                "Expected some discarded genomes (errors), but total_discarded=0"

            logger.info("PHASE 5: PASS")

        else:
            assert mock_evaluator.select_key.call_count > 0, \
                f"Worker {rank}: select_key was never called"
            logger.info("Worker %d: select_key called %d times",
                        rank, mock_evaluator.select_key.call_count)

            assert mock_rg.generate_response.call_count > 0, \
                f"Worker {rank}: ResponseGenerator was never called"
            logger.info("Worker %d: PHASE 5 PASS", rank)

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
        logger.error("Need >= 2 ranks. Run with: mpiexec -n 3 python tests/test_phase5_mpi.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting Phase 1-5 MPI integration test  (size=%d)", size)
    logger.info("=" * 60)

    test_phase5(comm, rank, size, logger)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL PHASE 1-5 TESTS PASSED")
        logger.info("=" * 60)
