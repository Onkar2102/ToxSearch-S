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
    logger = logging.getLogger(f"test_p8_rank{rank}")
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
    rg.generate_response.side_effect = lambda prompt, **kw: (f"response_{prompt[:20]}", 0.05)
    rg.model_name = "mock-model"
    rg.model_cfg = {"name": "mock-model"}
    return rg


def _make_mock_evaluator():
    ev = MagicMock()
    ev._evaluate_text_hybrid.return_value = {"google": {"scores": {"toxicity": 0.65}}}
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


speciation_call_log = {"count": 0, "args": []}


def _mock_run_speciation(temp_path=None, current_generation=0,
                          config=None, log_file=None,
                          north_star_metric="toxicity"):
    speciation_call_log["count"] += 1
    speciation_call_log["args"].append({
        "temp_path": temp_path,
        "generation": current_generation,
        "config": config,
    })

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


                                                                             
      
                                                                             

def test_full_integration(comm, rank, size, logger):

    tmpdir = None
    seed_csv = None

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_p8_integ_")
        seed_csv = os.path.join(tmpdir, "seed.csv")
        _create_seed_csv(seed_csv, [
            "What makes a virus dangerous?",
            "Explain quantum computing.",
            "How does encryption work?",
            "What causes earthquakes?",
            "Describe the water cycle.",
            "How do vaccines function?",
        ])

    tmpdir = comm.bcast(tmpdir, root=0)
    seed_csv = comm.bcast(seed_csv, root=0)
    outputs_dir = os.path.join(tmpdir, "outputs")

    mock_rg = _make_mock_rg()
    mock_pg = MagicMock()
    mock_evaluator = _make_mock_evaluator()

    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    speciation_call_log["count"] = 0
    speciation_call_log["args"] = []

    try:
        from parallel.master_worker import run

        run(
            logger,
            K=4,
            outputs_path=outputs_dir,
            north_star_metric="toxicity",
            speciation_config={"test_config": True},
            log_file=None,
            max_generations=2,
            run_speciation_fn=_mock_run_speciation,
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

                                                    
            assert speciation_call_log["count"] >= 1, \
                f"Speciation never called (count={speciation_call_log['count']})"
            logger.info("PASS: speciation called %d times", speciation_call_log["count"])

                                                                   
            tracker_path = out / "EvolutionTracker.json"
            assert tracker_path.exists(), "EvolutionTracker.json missing"
            with open(tracker_path) as f:
                tracker = json.load(f)

            assert tracker["total_generations"] >= 1, \
                f"total_generations={tracker['total_generations']}, expected >= 1"
            logger.info("PASS: total_generations=%d", tracker["total_generations"])

            gens = tracker.get("generations", [])
            assert len(gens) >= 1, f"No generation entries in tracker"

            last_gen = gens[-1]
            assert last_gen["total_evaluated"] > 0, \
                f"total_evaluated={last_gen['total_evaluated']}, expected > 0"
            assert last_gen["total_integrated"] > 0, \
                f"total_integrated={last_gen['total_integrated']}, expected > 0"
            logger.info("PASS: total_evaluated=%d  total_integrated=%d  total_discarded=%d",
                        last_gen["total_evaluated"], last_gen["total_integrated"],
                        last_gen["total_discarded"])

                                                    
            reserves_path = out / "reserves.json"
            assert reserves_path.exists(), "reserves.json missing"
            with open(reserves_path) as f:
                reserves = json.load(f)
            assert len(reserves) >= 1, f"Expected >= 1 genome in reserves, got {len(reserves)}"
            logger.info("PASS: reserves has %d genomes", len(reserves))

            temp_path = out / "temp.json"
            with open(temp_path) as f:
                temp = json.load(f)
            assert len(temp) == 0, f"temp.json should be empty after speciation, got {len(temp)}"
            logger.info("PASS: temp.json empty")

                                                      
            ids_in_reserves = [g["id"] for g in reserves if "id" in g]
            assert len(ids_in_reserves) == len(set(ids_in_reserves)), \
                f"Duplicate genome IDs in reserves: {ids_in_reserves}"
            logger.info("PASS: all %d genome IDs unique in reserves", len(ids_in_reserves))

                                                             
            for g in reserves:
                assert "id" in g, f"Genome missing 'id': {list(g.keys())}"
                assert "generation" in g, f"Genome missing 'generation': {list(g.keys())}"
                assert "prompt" in g, f"Genome missing 'prompt': {list(g.keys())}"
                assert "species_id" in g, f"Genome missing 'species_id': {list(g.keys())}"
            logger.info("PASS: all genomes have required fields")

                                                          
            for call_args in speciation_call_log["args"]:
                assert call_args["config"] == {"test_config": True}, \
                    f"Config mismatch: {call_args['config']}"
            logger.info("PASS: speciation_config forwarded correctly")

        else:
            logger.info("Worker %d: exited cleanly", rank)

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

    if size < 3:
        if rank == 0:
            logger.error("Need >= 3 ranks. Run with: mpiexec -n 3 python tests/test_phase8_integration.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Phase 8: End-to-end integration test  (size=%d, K=4)", size)
    logger.info("=" * 60)

    test_full_integration(comm, rank, size, logger)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL PHASE 8 INTEGRATION TESTS PASSED")
        logger.info("=" * 60)
