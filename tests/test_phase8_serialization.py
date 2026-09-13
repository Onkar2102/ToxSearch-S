import sys
import os

from mpi4py import MPI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from parallel.master_worker import (
    PARENTS_REQUEST, PARENTS, EVALUATED_VARIANT, GEN0_BATCH,
    send_payload, recv_payload,
)


COMPLEX_DICT = {
    "request_id": "1_0",
    "local_variant_id": "1_42",
    "prompt": "What is the meaning of life?",
    "generated_output": "The meaning of life is 42.",
    "status": "complete",
    "score": 0.75,
    "moderation_result": {
        "google": {
            "scores": {"toxicity": 0.55, "severe_toxicity": 0.01},
            "raw": None,
        }
    },
    "parents": [
        {"id": 1, "prompt": "parent prompt", "score": 0.6},
        {"id": 2, "prompt": "another parent", "score": 0.7},
    ],
    "tags_list": ["mutation", "crossover"],
    "nested": {"a": {"b": {"c": [1, 2.5, "three", None, True, False]}}},
    "empty_dict": {},
    "empty_list": [],
    "zero": 0,
    "negative": -3.14,
    "large_int": 2**40,
    "unicode": "toxicity \u2603 \u00e9\u00e8\u00ea",
}

TAGS = [PARENTS_REQUEST, PARENTS, EVALUATED_VARIANT, GEN0_BATCH]
TAG_NAMES = {
    PARENTS_REQUEST: "PARENTS_REQUEST",
    PARENTS: "PARENTS",
    EVALUATED_VARIANT: "EVALUATED_VARIANT",
    GEN0_BATCH: "GEN0_BATCH",
}


def _make_logger():
    import logging
    rank = MPI.COMM_WORLD.Get_rank()
    logger = logging.getLogger(f"test_serial_rank{rank}")
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            f"[rank {rank}] %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def test_round_trip(comm, rank, logger):
    if rank == 0:
        for tag in TAGS:
            send_payload(comm, dest=1, tag=tag, payload=COMPLEX_DICT, logger=logger)
            logger.info("Sent COMPLEX_DICT with tag %s (%d)", TAG_NAMES[tag], tag)

        send_payload(comm, dest=1, tag=PARENTS, payload=None, logger=logger)
        logger.info("Sent None (shutdown) with tag PARENTS")

    else:
        for tag in TAGS:
            data, tag_id, source = recv_payload(comm, source=0, tag=tag, logger=logger)
            assert tag_id == tag, f"Expected tag {tag}, got {tag_id}"
            assert source == 0, f"Expected source 0, got {source}"
            assert data == COMPLEX_DICT, (
                f"Dict mismatch for tag {TAG_NAMES[tag]}:\n"
                f"  sent keys: {sorted(COMPLEX_DICT.keys())}\n"
                f"  recv keys: {sorted(data.keys()) if isinstance(data, dict) else type(data)}"
            )
            assert data["moderation_result"]["google"]["scores"]["toxicity"] == 0.55
            assert data["nested"]["a"]["b"]["c"] == [1, 2.5, "three", None, True, False]
            assert data["large_int"] == 2**40
            assert data["unicode"] == "toxicity \u2603 \u00e9\u00e8\u00ea"
            logger.info("PASS: tag %s (%d) round-trip OK", TAG_NAMES[tag], tag)

        data, tag_id, source = recv_payload(comm, source=0, tag=PARENTS, logger=logger)
        assert data is None, f"Expected None, got {type(data)}"
        assert tag_id == PARENTS
        logger.info("PASS: None (shutdown) round-trip OK")

    comm.Barrier()
    if rank == 0:
        logger.info("ROUND-TRIP SERIALIZATION: ALL PASSED")


def test_empty_and_minimal(comm, rank, logger):
    payloads = [
        {},
        {"x": 1},
        {"data": list(range(1000))},
    ]

    if rank == 0:
        for i, payload in enumerate(payloads):
            send_payload(comm, dest=1, tag=EVALUATED_VARIANT, payload=payload, logger=logger)
    else:
        for i, expected in enumerate(payloads):
            data, _, _ = recv_payload(comm, source=0, tag=EVALUATED_VARIANT, logger=logger)
            assert data == expected, f"Payload {i} mismatch"
            logger.info("PASS: edge-case payload %d OK", i)

    comm.Barrier()
    if rank == 0:
        logger.info("EDGE CASES: ALL PASSED")


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    logger = _make_logger()

    if size != 2:
        if rank == 0:
            logger.error("Need exactly 2 ranks. Run with: mpiexec -n 2 python tests/test_phase8_serialization.py")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Phase 8: Round-trip serialization test  (size=%d)", size)
    logger.info("=" * 60)

    test_round_trip(comm, rank, logger)
    test_empty_and_minimal(comm, rank, logger)

    if rank == 0:
        logger.info("=" * 60)
        logger.info("ALL PHASE 8 SERIALIZATION TESTS PASSED")
        logger.info("=" * 60)
