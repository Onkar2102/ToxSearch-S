import importlib.util
import logging
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_REAL_MPI = importlib.util.find_spec("mpi4py") is not None
if not _REAL_MPI:
    _fake_mpi = types.ModuleType("mpi4py")
    _fake_mpi.MPI = MagicMock()
    sys.modules["mpi4py"] = _fake_mpi

root = Path(__file__).resolve().parent.parent
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    """Skip legacy MPI/spacy integration modules when optional deps are missing."""
    name = collection_path.name
    needs_mpi = "mpi" in name.lower() or name in (
        "test_phase3_run.py",
        "test_phase8_integration.py",
        "test_phase8_serialization.py",
    )
    if needs_mpi and not _REAL_MPI:
        return True
    if name.startswith("test_phase") and importlib.util.find_spec("spacy") is None:
        if "unit" not in name and name not in ("test_phase4_unit.py", "test_phase5_unit.py", "test_phase67_unit.py"):
            return True
        if name in ("test_phase3_run.py", "test_phase8_integration.py", "test_phase8_serialization.py"):
            return True
    return None


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        node = item.nodeid.lower()
        # phase8_* are MPI payload tests (need comm/rank fixtures under mpiexec).
        if (
            "mpi" in node
            or "phase3_mpi" in node
            or "phase4_mpi" in node
            or "phase5_mpi" in node
            or "phase67_mpi" in node
            or "phase8_integration" in node
            or "phase8_serialization" in node
            or "phase3_run" in node
        ):
            item.add_marker(pytest.mark.mpi)
    if not _REAL_MPI:
        skip_mpi = pytest.mark.skip(reason="mpi4py not installed")
        for item in items:
            node = item.nodeid.lower()
            if "mpi" in node or "phase8_integration" in node or "phase8_serialization" in node or "phase3_run" in node:
                item.add_marker(skip_mpi)


# ---------------------------------------------------------------------------
# MPI fixtures (used by tests/test_*mpi*.py under ``mpiexec -n N pytest -m mpi``)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def comm():
    if not _REAL_MPI:
        pytest.skip("mpi4py not installed")
    from mpi4py import MPI

    return MPI.COMM_WORLD


@pytest.fixture(scope="session")
def rank(comm):
    return int(comm.Get_rank())


@pytest.fixture(scope="session")
def size(comm):
    return int(comm.Get_size())


@pytest.fixture
def logger(rank):
    log = logging.getLogger(f"pytest_mpi_rank{rank}")
    log.setLevel(logging.DEBUG)
    if not log.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(f"[rank {rank}] %(levelname)s %(message)s")
        )
        log.addHandler(handler)
    return log


@pytest.fixture(autouse=True)
def _mpi_min_ranks_and_sync(request):
    """Skip MPI tests without enough ranks; barrier before/after to keep ranks aligned."""
    if "mpi" not in request.node.keywords:
        yield
        return
    if not _REAL_MPI:
        pytest.skip("mpi4py not installed")
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = int(comm.Get_size())
    node = request.node.nodeid.lower()
    # Serialization round-trip is written for exactly 2 ranks (0↔1).
    if "phase8_serialization" in node:
        if size != 2:
            pytest.skip(f"phase8_serialization needs exactly 2 ranks, got {size}")
        yield
        return
    if "mpi_gen0_many_workers" in node or "small_queue" in node:
        if size < 4:
            pytest.skip(f"need >= 4 MPI ranks, got {size}")
    elif size < 2:
        pytest.skip(
            f"MPI tests need >= 2 ranks (mpiexec -n 3 pytest -m mpi); got {size}"
        )

    # Align ranks around each MPI test so a skip/fail on one rank cannot
    # leave others blocked inside a later collective.
    if size >= 2:
        comm.Barrier()
    yield
    if size >= 2:
        try:
            comm.Barrier()
        except Exception:
            pass
