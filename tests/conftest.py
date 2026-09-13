import importlib.util
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
        if "mpi" in node or "phase3_mpi" in node or "phase4_mpi" in node:
            item.add_marker(pytest.mark.mpi)
    if not _REAL_MPI:
        skip_mpi = pytest.mark.skip(reason="mpi4py not installed")
        for item in items:
            node = item.nodeid.lower()
            if "mpi" in node or "phase8_integration" in node or "phase8_serialization" in node or "phase3_run" in node:
                item.add_marker(skip_mpi)
