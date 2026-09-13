# ToxSearch-S architecture

## Overview

ToxSearch-S runs a quality–diversity evolutionary loop over adversarial prompts. A local GGUF model generates responses; an external moderation API scores them; semantic speciation clusters prompts in embedding space.

## Package layout (`src/`)

| Package | Role |
|---------|------|
| `gne/` | GGUF response/prompt generators and moderation evaluators |
| `ea/` | Variation operators, evolution engine, parent selection |
| `speciation/` | Species assignment, merge/extinction, embeddings, distances |
| `parallel/` | MPI master–worker orchestration |
| `utils/` | Population I/O, logging, run config, paths, live analysis |
| `cli.py` | CLI entry (`python src/main.py` or `toxsearch` after editable install) |
| `main.py` | Sequential run orchestration (generation loop) |

## Data flow (sequential)

1. CLI writes `run_config.json` and initializes RNG + output directory.
2. Gen 0: seed CSV → responses → moderation → speciation → `elites.json` / `reserves.json`.
3. Later generations: `run_evolution` → speciation cadence → `EvolutionTracker.json`.
4. Optional live analysis and GDP projection plots under the run output dir.

## Experiments vs results

- `experiments/` — analysis drivers (Python only in git).
- `results/` — generated CSVs/figures (gitignored except `results/manifests/`).
- `data/outputs/` — live evolution runs (gitignored).

## Reproducibility

- `--seed` — LLM GGUF seed plus Python/NumPy RNG for operators.
- `run_config.json` — full CLI + `SpeciationConfig` snapshot at run start.
- Study manifests — `python results/manifests/build_study_manifest.py`.

## Formal methods

Distance semantics and metricity notes: [`verfier/`](../verfier/README.md).
