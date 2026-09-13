# ToxSearch-S tests

Unit, smoke, and integration tests for the research framework: distances, config, RNG, CLI, refusal detection, speciation phases, parallel merge **K**, MPI generation 0, and tracker metrics.

## Running tests

From the **repository root**, with `src` on the Python path (or after `pip install -e .`):

```bash
# Recommended for local / CI-like runs (skip MPI suite)
PYTHONPATH=src python -m pytest tests/ -v -m "not mpi"

# Full suite including MPI (needs mpi4py + mpiexec)
PYTHONPATH=src python -m pytest tests/ -v

# MPI-only
PYTHONPATH=src python -m pytest tests/ -v -m mpi
```

`conftest.py` adds `src` to `sys.path`, stubs `mpi4py` when missing, and skips MPI-heavy modules so collection succeeds in lightweight environments.

Optional: ignore legacy phase-4 unit tests that need a full operator stack:

```bash
python -m pytest tests/ -m "not mpi" --ignore=tests/test_phase4_unit.py -q
```

## Markers

| Marker | Meaning |
|--------|---------|
| `mpi` | Requires MPI (`mpi4py` / `mpiexec`) |
| `integration` | Heavier fixtures or network (reserved) |

Defined in [`pyproject.toml`](../pyproject.toml) under `[tool.pytest.ini_options]`.

## Test modules

### Framework / reproducibility (preferred for CI)

| File | Focus |
|------|--------|
| `test_distance.py` | `speciation.distance` — norms, ensemble weights, batch vs scalar |
| `test_speciation_config.py` | `SpeciationConfig` validation and `to_dict` / `from_dict` |
| `test_run_config.py` | `run_config.json` serialization |
| `test_rng.py` | Seeded Python RNG determinism |
| `test_cli.py` | CLI defaults (`theta_sim` vs `SpeciationConfig`) |

### Evaluators and utilities

| File | Focus |
|------|--------|
| `test_refusal_detector.py` | Refusal heuristics |
| `test_config_load.py` | Project paths; `config/RGConfig.yaml` / `PGConfig.yaml` |
| `test_evaluator_profiles.py` | Evaluator / north-star profiles |
| `test_evaluator_openai.py` | OpenAI evaluator wiring (may need mocks/keys) |
| `test_tracker_generation_metrics.py` | `EvolutionTracker` generation fields |
| `test_parallel_merge_k.py` | Merge **K** vs operators / `max_variants` |

### Speciation phases and MPI

| File | Focus |
|------|--------|
| `test_phase3_run.py`, `test_phase3_mpi.py` | Phase 3 (merge-related) |
| `test_phase4_unit.py`, `test_phase4_mpi.py` | Phase 4 (capacity) |
| `test_phase5_unit.py`, `test_phase5_mpi.py` | Phase 5 (stagnation / freeze) |
| `test_phase67_unit.py`, `test_phase67_mpi.py` | Phases 6–7 (redistribution / file sync) |
| `test_phase8_integration.py`, `test_phase8_serialization.py` | Phase 8 metrics and MPI payloads |
| `test_mpi_gen0.py` | MPI generation-0 bootstrap |

## CI

[`.github/workflows/ci.yml`](../.github/workflows/ci.yml) installs a minimal dependency set and runs non-MPI tests (with `test_phase4_unit.py` ignored). No GPU or moderation API keys are required in CI.
