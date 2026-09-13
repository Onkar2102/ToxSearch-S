# ToxSearch-S architecture

This document describes the research framework layout after the production refactor. Scientific algorithms (distance formulas, speciation thresholds, moderation scoring) are unchanged unless explicitly noted in a changelog.

## Overview

ToxSearch-S runs a quality–diversity evolutionary loop over adversarial prompts:

1. A local **GGUF** model answers (and helps mutate) prompts.
2. An external **moderation API** (Perspective or OpenAI) scores responses → fitness / phenotype.
3. **Semantic speciation** clusters prompts in embedding space (genotype) and uses ensemble distance with phenotype for species membership, merge, and capacity.

```text
CLI (cli.py) --> run_config.json + RNG seed
       |
       +--> sequential: main.py generation loop
       |         |
       |         +--> gne/ (generate + evaluate)
       |         +--> ea/  (operators, selection)
       |         +--> speciation/ (species lifecycle)
       |         +--> utils/ (I/O, tracker, logging)
       |
       +--> parallel: parallel/master_worker.py (MPI)
                 same packages on master / workers
```

## Package layout (`src/`)

| Package / module | Role |
|------------------|------|
| `gne/` | GGUF response/prompt generators; moderation evaluators |
| `ea/` | Variation operators, evolution engine, parent selection |
| `speciation/` | Species assignment, merge/extinction, embeddings, distances, config |
| `parallel/` | MPI master–worker orchestration and serialization |
| `utils/` | Population I/O, logging, `run_config`, `rng`, model YAML patching, live analysis |
| `cli.py` | Argument parsing, evaluator setup, dispatch sequential vs `--parallel` |
| `main.py` | Sequential run orchestration (generation loop); entry re-exports CLI |

Console entry after `pip install -e .`: `toxsearch` → `cli:main`.

## Configuration layers

| Layer | Location | Purpose |
|-------|----------|---------|
| Speciation defaults | `SpeciationConfig` in `src/speciation/config.py` | Species radii, capacities, embeddings, ensemble weights (CLI overrides) |
| Inference YAML | `config/RGConfig.yaml`, `config/PGConfig.yaml` | GGUF model paths and generation args |
| CLI / env | flags, shell scripts, `.env` | Run knobs and API keys |

Speciation fields are validated in `SpeciationConfig.__post_init__`. There is a single `config/` directory for inference YAML—not a separate experiment-preset tree.

## Data flow (sequential)

1. CLI resolves evaluator / north-star, builds `SpeciationConfig`, writes **`run_config.json`**, calls `init_run_rng(seed)`.
2. **Gen 0:** seed CSV → responses → moderation → refusal penalties → speciation → `elites.json` / `reserves.json`.
3. **Later generations:** parent selection → operators → evaluate → speciation cadence → `EvolutionTracker.json`.
4. Optional live analysis / GDP projection under the run output directory.

## Data flow (parallel)

1. Rank 0 updates RG/PG YAML from `--rg` / `--pg`, broadcasts output path and log name.
2. Workers initialize generators with the shared LLM `seed` and process gen0 / variant jobs.
3. Master merges batches of size **K**, runs speciation, updates tracker; termination when total genomes hit `--max-total-genomes`.

## Distances (genotype / phenotype / ensemble)

Implemented in `speciation/distance.py` and `speciation/phenotype_distance.py`:

- **Genotype:** cosine distance on L2-normalized prompt embeddings (`semantic_distance`). Not a classical metric; see `verfier/` for the 2-relaxed triangle inequality.
- **Phenotype:** distance over moderation score vectors (missing phenotypes treated as maximal distance `1.0` in ensemble).
- **Ensemble:** `w_genotype * (d_genotype / 2) + w_phenotype * d_phenotype` with weights summing to 1 (defaults 0.7 / 0.3). Species radii (`theta_sim`, `theta_merge`) apply in this ensemble space.

Analysis code should reuse core distances (e.g. EMNLP `genotype_distance` wraps `semantic_distance / 2`) rather than reimplementing formulas.

## Experiments vs results vs live runs

| Path | In git? | Contents |
|------|---------|----------|
| `experiments/` | Drivers yes | Analysis / report Python only |
| `results/**` | No (except manifests) | Generated CSVs, figures, JSON |
| `results/manifests/` | Yes | SHA256 indexes + regen notes |
| `data/outputs/` | No | Live evolution run directories |

## Reproducibility surfaces

- **`--seed`** — LLM GGUF seed and Python/NumPy RNG (`utils/rng.py`).
- **`run_config.json`** — CLI + `SpeciationConfig.to_dict()` + optional git commit (`utils/run_config.py`).
- **`EvolutionTracker`** — generation metrics and `run_metadata`.
- **Manifests** — `python results/manifests/build_study_manifest.py`.

## Testing and CI

- Unit tests for distance, `SpeciationConfig`, run config, RNG, CLI defaults (`tests/`).
- MPI / heavy integration tests marked `mpi` or skipped when optional deps are missing (`tests/conftest.py`).
- CI: `.github/workflows/ci.yml` — non-MPI pytest + ruff on new framework modules.

## Formal methods

Distance semantics, config well-formedness, and related claims: [`verfier/`](../verfier/README.md) (directory name kept for historical links).
