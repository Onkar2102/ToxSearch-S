import json
import sys
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from utils.population_io import (  # noqa: E402
    update_evolution_tracker_with_statistics,
    compute_run_summary,
)


@pytest.fixture
def tracker_path():
    with tempfile.TemporaryDirectory(prefix="toxsearch_tracker_") as tmp:
        p = Path(tmp) / "EvolutionTracker.json"
        p.write_text(
            json.dumps(
                {
                    "status": "not_complete",
                    "total_generations": 0,
                    "generations_since_improvement": 0,
                    "avg_fitness_history": [],
                    "slope_of_avg_fitness": 0.0,
                    "selection_mode": "default",
                    "run_metadata": {},
                    "generations": [],
                }
            ),
            encoding="utf-8",
        )
        yield p


def test_evaluated_this_generation_increments_cumulative(tracker_path):
    stats0 = {
        "elites_count": 2,
        "reserves_count": 0,
        "archived_count": 0,
        "total_population": 2,
        "avg_fitness_generation": 0.5,
        "population_max_toxicity": 0.6,
        "evaluated_this_generation": 10,
        "discarded_this_generation": 1,
        "api_calls": 10,
        "llm_calls": 10,
        "llm_calls_response_generation": 10,
        "llm_calls_variant_creation": 0,
    }
    assert update_evolution_tracker_with_statistics(
        evolution_tracker_path=str(tracker_path),
        current_generation=0,
        statistics=stats0,
        operator_statistics=None,
        logger=None,
    )
    stats1 = dict(stats0)
    stats1["evaluated_this_generation"] = 5
    stats1["discarded_this_generation"] = 0
    stats1["api_calls"] = 5
    stats1["llm_calls"] = 5
    assert update_evolution_tracker_with_statistics(
        evolution_tracker_path=str(tracker_path),
        current_generation=1,
        statistics=stats1,
        operator_statistics=None,
        logger=None,
    )
    with open(tracker_path, "r", encoding="utf-8") as f:
        tracker = json.load(f)
    assert tracker["cumulative_variants_evaluated"] == 15
    assert tracker["cumulative_variants_discarded"] == 1
    summary = compute_run_summary(tracker)
    assert summary["total_evaluated"] == 15


def test_compute_run_summary_prefers_cumulative(tracker_path):
    with open(tracker_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generations": [
                    {"generation_number": 0, "evaluated_this_generation": 3, "variants_integrated": 2},
                    {"generation_number": 1, "evaluated_this_generation": 4, "variants_integrated": 3},
                ],
                "cumulative_variants_evaluated": 7,
                "run_metadata": {"run_duration_seconds": 10.0},
            },
            f,
        )
    with open(tracker_path, "r", encoding="utf-8") as f:
        tracker = json.load(f)
    s = compute_run_summary(tracker)
    assert s["total_evaluated"] == 7
    assert s["total_integrated"] == 5
