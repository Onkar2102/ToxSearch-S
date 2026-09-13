import json
import sys
import os
import csv
import tempfile
import shutil
from unittest.mock import MagicMock

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import ea.evolution_engine as ee_module


def _mock_generate_single_variant(parents, pg, **kwargs):
    p0 = parents[0] if parents else {}
    return [{
        "prompt": f"evolved_{p0.get('prompt', 'x')[:15]}",
        "operator": "MockOp", "variant_type": "mutation",
        "parents": [{"id": p.get("id"), "score": 0.5} for p in parents],
        "status": "pending_generation",
        "creation_info": {"type": "mutation", "operator": "MockOp"},
    }]


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    import logging
    logger = logging.getLogger(f"run_test_rank{rank}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"[rank {rank}] %(levelname)s %(message)s"))
        logger.addHandler(handler)

    if size < 2:
        logger.error("Need >= 2 ranks")
        sys.exit(1)

    tmpdir = None
    seed_csv = None
    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_run_test_")
        seed_csv = os.path.join(tmpdir, "seed.csv")
        with open(seed_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["questions"])
            for p in ["prompt_a", "prompt_b", "prompt_c", "prompt_d",
                       "prompt_e", "prompt_f"]:
                writer.writerow([p])
    tmpdir = comm.bcast(tmpdir, root=0)
    seed_csv = comm.bcast(seed_csv, root=0)

    mock_rg = MagicMock()
    mock_rg.generate_response.side_effect = lambda p, **kw: (f"resp_{p[:10]}", 0.02)
    mock_rg.model_name = "mock"

    mock_eval = MagicMock()
    mock_eval._evaluate_text_hybrid.return_value = {"google": {"scores": {"toxicity": 0.6}}}
    mock_eval._last_evaluation_time = {"duration": 0.01}

    original_gsv = ee_module.generate_single_variant
    ee_module.generate_single_variant = _mock_generate_single_variant

    try:
        from parallel.master_worker import run

        run(logger, K=4, outputs_path=os.path.join(tmpdir, "outputs"),
            north_star_metric="toxicity", max_generations=2,
            seed_file=seed_csv, moderation_methods=["google"],
            response_generator=mock_rg, prompt_generator=MagicMock(),
            evaluator=mock_eval)
    finally:
        ee_module.generate_single_variant = original_gsv

    if rank == 0:
        from pathlib import Path
        outputs = Path(tmpdir) / "outputs"

        with open(outputs / "reserves.json", "r") as f:
            reserves = json.load(f)
        with open(outputs / "temp.json", "r") as f:
            temp = json.load(f)
        with open(outputs / "EvolutionTracker.json", "r") as f:
            tracker = json.load(f)

        assert len(temp) == 0, f"temp should be empty, got {len(temp)}"
        assert len(reserves) >= 4, f"reserves should have >= 4, got {len(reserves)}"
        assert tracker["total_generations"] >= 1

        ids = [g["id"] for g in reserves]
        assert len(ids) == len(set(ids)), f"Duplicate IDs: {ids}"

        for g in reserves:
            assert "id" in g, f"Genome missing id: {g}"
            assert "generation" in g, f"Genome missing generation: {g}"
            assert "prompt" in g, f"Genome missing prompt: {g}"

        logger.info("=" * 60)
        logger.info("RUN TEST PASSED  reserves=%d  tracker_gens=%d",
                     len(reserves), tracker["total_generations"])
        logger.info("=" * 60)

        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
