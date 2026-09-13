# ToxSearch-S tests

Smoke, unit, and integration tests for refusal detection, config loading, speciation phases, parallel merge **K**, MPI generation 0, and tracker generation metrics.

## Running tests

From the **project root** (ToxSearch-S/), with `src` on the Python path:

```bash
# If using a virtualenv that has pytest and project deps:
PYTHONPATH=src python -m pytest tests/ -v

# Or activate venv first, then:
export PYTHONPATH=src
pytest tests/ -v
```

`conftest.py` adds `src` to `sys.path` when pytest discovers tests; if you run pytest from project root and still get import errors, set `PYTHONPATH=src` or install the package in editable mode.

## Test modules

| File | Focus |
|------|--------|
| `test_refusal_detector.py` | `utils.refusal_detector` — sentence count, short response, refusal pattern, `is_refusal` |
| `test_config_load.py` | Project root and config paths; `config/RGConfig.yaml` and `config/PGConfig.yaml` load as valid YAML |
| `test_phase3_run.py`, `test_phase3_mpi.py` | Speciation phase 3 (merge-related) |
| `test_phase4_unit.py`, `test_phase4_mpi.py` | Phase 4 (capacity) |
| `test_phase5_unit.py`, `test_phase5_mpi.py` | Phase 5 (stagnation / freeze) |
| `test_phase67_unit.py`, `test_phase67_mpi.py` | Phases 6–7 (redistribution / file sync) |
| `test_phase8_integration.py`, `test_phase8_serialization.py` | Phase 8 metrics and serialization |
| `test_parallel_merge_k.py` | Merge **K** resolution vs operators / `max_variants` |
| `test_mpi_gen0.py` | MPI generation-0 bootstrap behaviour |
| `test_tracker_generation_metrics.py` | `EvolutionTracker` / generation metric fields |
