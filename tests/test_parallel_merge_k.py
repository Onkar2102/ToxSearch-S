import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from parallel.master_worker import resolve_parallel_merge_k  # noqa: E402


@pytest.fixture
def outputs_dir():
    with tempfile.TemporaryDirectory(prefix="toxsearch_merge_k_") as tmp:
        yield Path(tmp)


def _write_tracker(outputs_dir: Path, selection_mode: str) -> None:
    p = outputs_dir / "EvolutionTracker.json"
    p.write_text(
        json.dumps({"selection_mode": selection_mode, "generations": []}),
        encoding="utf-8",
    )


def test_resolve_override_wins(outputs_dir):
    _write_tracker(outputs_dir, "default")
    assert resolve_parallel_merge_k(outputs_dir, "all", 1, 7) == 7


def test_resolve_all_default_mode_24(outputs_dir):
    _write_tracker(outputs_dir, "default")
    assert resolve_parallel_merge_k(outputs_dir, "all", 1, None) == 24


def test_resolve_all_explore_39(outputs_dir):
    for mode in ("explore", "exploration", "exploit", "exploitation"):
        _write_tracker(outputs_dir, mode)
        assert resolve_parallel_merge_k(outputs_dir, "all", 1, None) == 39


def test_resolve_scaled_by_max_variants(outputs_dir):
    _write_tracker(outputs_dir, "default")
    assert resolve_parallel_merge_k(outputs_dir, "all", 2, None) == 48
    _write_tracker(outputs_dir, "explore")
    assert resolve_parallel_merge_k(outputs_dir, "all", 2, None) == 78


def test_resolve_non_all_legacy_100(outputs_dir):
    _write_tracker(outputs_dir, "default")
    assert resolve_parallel_merge_k(outputs_dir, "cm", 1, None) == 100
    assert resolve_parallel_merge_k(outputs_dir, "ie", 1, None) == 100


def test_resolve_missing_tracker_defaults_default_mode(outputs_dir):
    assert resolve_parallel_merge_k(outputs_dir, "all", 1, None) == 24
