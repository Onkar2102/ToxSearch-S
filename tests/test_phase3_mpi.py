import json
import sys
import os
import tempfile
import shutil

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallel.master_worker import (
    PARENTS_REQUEST, PARENTS, EVALUATED_VARIANT, GEN0_BATCH,
    send_payload, recv_payload,
    _merge_and_speciate, _stub_speciation, _update_tracker,
    _load_existing_prompts,
)


def _make_logger():
    import logging
    rank = MPI.COMM_WORLD.Get_rank()
    logger = logging.getLogger(f"test_rank{rank}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"[rank {rank}] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


                                                                                

def test_phase1(comm, rank, size, logger):
    assert size >= 2, "Need at least 2 ranks"
    if rank == 0:
        logger.info("PHASE 1: master sees %d workers", size - 1)
    else:
        logger.info("PHASE 1: worker %d alive", rank)
    comm.Barrier()
    logger.info("PHASE 1: barrier passed")


                                                                                

def test_phase2(comm, rank, size, logger):
    if rank == 0:
        n_workers = size - 1
        for _ in range(n_workers):
            data, tag, src = recv_payload(comm, logger=logger)
            assert tag == PARENTS_REQUEST
            assert data.get("request_id") is not None
            send_payload(comm, src, PARENTS,
                         {"request_id": data["request_id"],
                          "parents": [{"id": 1, "prompt": "p1"}]},
                         logger=logger)

        for _ in range(n_workers):
            data, tag, src = recv_payload(comm, logger=logger)
            assert tag == EVALUATED_VARIANT
            assert data.get("request_id") is not None
            assert data.get("local_variant_id") is not None

        for w in range(1, size):
            send_payload(comm, w, PARENTS, None, logger=logger)

        logger.info("PHASE 2: master protocol OK")
    else:
        req_id = f"{rank}_p2"
        send_payload(comm, 0, PARENTS_REQUEST, {"request_id": req_id}, logger=logger)
        data, tag, _ = recv_payload(comm, source=0, logger=logger)
        assert tag == PARENTS
        assert data["request_id"] == req_id

        variant = {"request_id": req_id, "local_variant_id": f"{rank}_0",
                    "prompt": "hello", "toxicity": 0.5}
        send_payload(comm, 0, EVALUATED_VARIANT, variant, logger=logger)

        data2, tag2, _ = recv_payload(comm, source=0, logger=logger)
        assert tag2 == PARENTS and data2 is None

        logger.info("PHASE 2: worker %d protocol OK", rank)

    comm.Barrier()


                                                                                

def test_dedup(comm, rank, size, logger):
    if rank != 0:
        comm.Barrier()
        return

    from collections import defaultdict
    from pathlib import Path

    tmpdir = tempfile.mkdtemp(prefix="toxsearch_dedup_")
    outputs = Path(tmpdir)

    for fn in ("temp.json", "elites.json", "reserves.json", "archive.json"):
        with open(outputs / fn, "w") as f:
            json.dump([], f)

    with open(outputs / "reserves.json", "w") as f:
        json.dump([{"id": 99, "prompt": "already_exists", "species_id": 0}], f)

    buffers = defaultdict(list)
    buffers[1] = [
        {"prompt": "unique_A", "toxicity": 0.5, "request_id": "1_0", "local_variant_id": "1_0"},
        {"prompt": "already_exists", "toxicity": 0.6, "request_id": "1_1", "local_variant_id": "1_1"},
        {"prompt": "unique_B", "toxicity": 0.7, "request_id": "1_2", "local_variant_id": "1_2"},
        {"prompt": "unique_A", "toxicity": 0.8, "request_id": "1_3", "local_variant_id": "1_3"},
    ]

    accepted, discarded, nid, _ = _merge_and_speciate(
        buffers, 10, outputs, 0, 1, "toxicity", None, None, logger)

    assert accepted == 2, f"Expected 2 accepted, got {accepted}"
    assert discarded == 2, f"Expected 2 discarded, got {discarded}"

    with open(outputs / "reserves.json", "r") as f:
        reserves = json.load(f)
    new_prompts = {g["prompt"] for g in reserves if g["id"] != 99}
    assert "unique_A" in new_prompts
    assert "unique_B" in new_prompts
    assert nid == 3, f"Expected next_genome_id=3, got {nid}"

    logger.info("DEDUP TEST: PASS  accepted=%d  discarded=%d", accepted, discarded)

    shutil.rmtree(tmpdir, ignore_errors=True)
    comm.Barrier()


                                                                                

def test_phase3(comm, rank, size, logger):
    tmpdir = None

    if rank == 0:
        tmpdir = tempfile.mkdtemp(prefix="toxsearch_test_")
    tmpdir = comm.bcast(tmpdir, root=0)

    K = 4
    MAX_GENERATIONS = 2
    n_workers = size - 1

    if rank == 0:
        from collections import defaultdict
        from pathlib import Path

        outputs = Path(tmpdir)
        for fn in ("temp.json", "elites.json", "reserves.json", "archive.json"):
            with open(outputs / fn, "w") as f:
                json.dump([], f)

        buffers = defaultdict(list)
        generation_id = 0
        next_genome_id = 1
        total_evaluated = 0
        total_integrated = 0
        total_discarded = 0
        gen0_complete = False
        shutdown = False
        finished = 0

        while finished < n_workers:
            data, tag, src = recv_payload(comm, logger=logger)

            if tag == PARENTS_REQUEST:
                req_id = data.get("request_id") if data else None

                if shutdown:
                    send_payload(comm, src, PARENTS, None, logger=logger)
                    finished += 1
                    logger.info("Sent shutdown to worker %d  (%d/%d)", src, finished, n_workers)

                elif not gen0_complete:
                    send_payload(comm, src, GEN0_BATCH,
                                 {"request_id": req_id, "prompt_start": 0, "prompt_end": 0},
                                 logger=logger)
                    logger.info("Sent GEN0_BATCH to worker %d", src)

                else:
                    parents_list = []
                    reserves_path = outputs / "reserves.json"
                    if reserves_path.exists():
                        with open(reserves_path, "r") as f:
                            reserves = json.load(f)
                        if reserves:
                            parents_list = reserves[:2]

                    if parents_list:
                        send_payload(comm, src, PARENTS,
                                     {"request_id": req_id, "parents": parents_list, "top_10": []},
                                     logger=logger)
                        logger.info("Sent PARENTS to worker %d  (%d parents)", src, len(parents_list))
                    else:
                        send_payload(comm, src, PARENTS, None, logger=logger)
                        finished += 1
                        logger.info("No parents available; shutdown worker %d", src)

            elif tag == EVALUATED_VARIANT:
                buffers[src].append(data)
                total_evaluated += 1

                total_buf = sum(len(b) for b in buffers.values())
                if total_buf >= K:
                    accepted, discarded, next_genome_id, spec = _merge_and_speciate(
                        buffers, K, outputs, generation_id, next_genome_id,
                        "toxicity", None, None, logger)
                    total_integrated += accepted
                    total_discarded += discarded
                    _update_tracker(outputs, generation_id, total_evaluated,
                                    total_integrated, total_discarded, spec, logger)
                    gen0_complete = True
                    generation_id += 1

                    if generation_id >= MAX_GENERATIONS:
                        shutdown = True
                        logger.info("Max generations %d reached. Shutdown flag set.", MAX_GENERATIONS)

                          
        reserves_path = outputs / "reserves.json"
        with open(reserves_path, "r") as f:
            reserves = json.load(f)

        temp_path = outputs / "temp.json"
        with open(temp_path, "r") as f:
            temp = json.load(f)

        tracker_path = outputs / "EvolutionTracker.json"
        with open(tracker_path, "r") as f:
            tracker = json.load(f)

        assert len(temp) == 0, f"temp should be empty, got {len(temp)}"
        assert len(reserves) >= K, f"reserves should have >= {K}, got {len(reserves)}"
        assert generation_id >= MAX_GENERATIONS, f"gen_id should be >= {MAX_GENERATIONS}, got {generation_id}"
        assert total_evaluated >= K, f"total_evaluated should be >= {K}, got {total_evaluated}"
        assert total_integrated >= K, f"total_integrated should be >= {K}, got {total_integrated}"
        assert tracker["total_generations"] >= 1

        ids = [g["id"] for g in reserves]
        assert len(ids) == len(set(ids)), f"Duplicate genome IDs: {ids}"

        logger.info("PHASE 3: PASS  reserves=%d  gen_id=%d  evaluated=%d  integrated=%d  discarded=%d",
                     len(reserves), generation_id, total_evaluated, total_integrated, total_discarded)

    else:
        seq = 0
        cycle = 0
        while True:
            req_id = f"{rank}_c{cycle}"
            send_payload(comm, 0, PARENTS_REQUEST, {"request_id": req_id}, logger=logger)
            data, tag, _ = recv_payload(comm, source=0, logger=logger)

            if tag == PARENTS and data is None:
                logger.info("Worker %d: shutdown received at cycle %d", rank, cycle)
                break

            if tag == GEN0_BATCH:
                logger.info("Worker %d: GEN0_BATCH at cycle %d", rank, cycle)
            elif tag == PARENTS:
                logger.info("Worker %d: PARENTS at cycle %d (%d parents)",
                            rank, cycle, len(data.get("parents", [])))

            for vi in range(2):
                lid = f"{rank}_{seq}"
                variant = {
                    "request_id": req_id,
                    "local_variant_id": lid,
                    "prompt": f"prompt_r{rank}_s{seq}",
                    "status": "complete",
                    "toxicity": round(0.1 + 0.03 * seq, 4),
                    "moderation_result": {"google": {"scores": {"toxicity": round(0.1 + 0.03 * seq, 4)}}},
                }
                send_payload(comm, 0, EVALUATED_VARIANT, variant, logger=logger)
                seq += 1

            cycle += 1

        logger.info("Worker %d: sent %d variants over %d cycles", rank, seq, cycle)

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
        logger.error("Need >= 2 ranks. Run with: mpiexec -n 3 python tests/test_phase3_mpi.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Starting Phase 1-3 MPI tests  (size=%d)", size)
    logger.info("=" * 60)

    test_phase1(comm, rank, size, logger)
    comm.Barrier()

    test_phase2(comm, rank, size, logger)
    comm.Barrier()

    test_dedup(comm, rank, size, logger)
    comm.Barrier()

    test_phase3(comm, rank, size, logger)
    comm.Barrier()

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED")
        logger.info("=" * 60)
