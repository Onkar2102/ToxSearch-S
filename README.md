# ToxSearch-S

Evolutionary search for **adversarial prompts** against local LLMs (GGUF). The loop is quality–diversity style: fitness comes from an external moderation API (Google [Perspective](https://developers.perspectiveapi.com/) or OpenAI omni-moderation) applied to model outputs, and **semantic speciation** keeps diversity by clustering prompts in embedding space.

**Index**

- [What you need](#what-you-need)
- [Installation](#installation)
- [Models (GGUF)](#models-gguf)
- [Project parameters](#project-parameters)
- [How to run](#how-to-run)
- [Repository layout](#repository-layout)
- [Where outputs go](#where-outputs-go)
- [Reproducibility](#reproducibility)
- [Development and CI](#development-and-ci)
- [Analysis and paper artifacts](#analysis-and-paper-artifacts)
- [Documentation](#documentation)
- [Dataset](#dataset)

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10 or newer |
| **Dependencies** | `pip install -r requirements.txt` (core); optional analysis/dev files below |
| **GPU** | Recommended for GGUF inference |
| **Perspective API** | Required for `--evaluator google` (at least one key in `.env`) |
| **OpenAI API** | Required for `--evaluator openai` (`OPENAI_API_KEY` in `.env`; optional org/project IDs) |
| **MPI (optional)** | Only for `--parallel`; Open MPI (or compatible) plus `mpi4py` built against that MPI |

---

## Installation

1. **Clone and enter the repo**

   ```bash
   git clone <repository-url>
   cd ToxSearch-S
   ```

2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate          # Windows: .venv\Scripts\activate
   ```

   `run_experiments_local.sh` also looks for `venv` or `.spvenv` if `.venv` is missing.

3. **Install Python packages**

   Core evolution + speciation:

   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

   Paper / cluster analysis (optional):

   ```bash
   pip install -r requirements-analysis.txt
   ```

   Development (tests, ruff, mypy):

   ```bash
   pip install -r requirements-dev.txt
   ```

   After editable install you can use `toxsearch …` instead of `python src/main.py …`.  
   Without editable install, use `export PYTHONPATH=src` (or prefix commands with `PYTHONPATH=src`).

4. **Configure API keys**

   ```bash
   cp env_example.txt .env
   ```

   Edit `.env` and set `PERSPECTIVE_API_KEY` for Google runs, or `OPENAI_API_KEY` (and optional `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID`) for OpenAI runs (see `env_example.txt`).

---

## Models (GGUF)

Download weights and place them under `models/`. The CLI flags **`--rg`** (response generator) and **`--pg`** (prompt generator) must point at `.gguf` files that exist on your machine, or match paths under `models/` the same way the YAML configs do.

Default paths in `src/cli.py` currently expect something like:

```text
models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf
```

If your files differ, pass explicit paths, for example:

```bash
--rg models/<your-folder>/<model>.gguf --pg models/<your-folder>/<model>.gguf
```

Inference hyperparameters live in [`config/RGConfig.yaml`](config/RGConfig.yaml) and [`config/PGConfig.yaml`](config/PGConfig.yaml) (updated from `--rg` / `--pg` at run start).

---

## Project parameters

Defaults follow [`SpeciationConfig`](src/speciation/config.py) and [`src/cli.py`](src/cli.py) unless you override them on the command line.

| Parameter | Meaning |
|-----------|---------|
| Max total genomes | Stop when elites + reserves + archive reach this count (required termination). Set with `--max-total-genomes`. |
| Theta similarity | Species assignment radius in **ensemble** (genotype + phenotype) distance; followers join a leader within this radius. Set with `--theta-sim` (default **0.25**). |
| Theta merge | Two species whose leaders are closer than this may merge; must be ≤ `--theta-sim`. Set with `--theta-merge`. |
| Min stability generations | Both species must be at least this old before they are allowed to merge. Set with `--min-stability-gens`. |
| Species capacity | Maximum individuals kept per species (excess archived by fitness). Set with `--species-capacity`. |
| Cluster-0 max capacity | Upper bound on individuals in cluster 0 / reserves before archiving. Set with `--cluster0-max-capacity`. |
| Cluster-0 min cluster size | Minimum cohesive cluster size in cluster 0 before a new species can form from it. Set with `--cluster0-min-cluster-size`. |
| Min island size | Islands smaller than this are treated as extinct. Set with `--min-island-size`. |
| Species stagnation | Generations without improvement after which a species can go extinct. Set with `--species-stagnation`. |
| Embedding model | Sentence-transformer name for prompt embeddings in speciation. Set with `--embedding-model`. |
| Embedding dimension | Vector size for embeddings (must match the model). Set with `--embedding-dim`. |
| Embedding batch size | Batch size when computing embeddings. Set with `--embedding-batch-size`. |
| Response generator | GGUF path for the model that answers the adversarial prompt. Set with `--rg`. |
| Prompt generator | GGUF path for the model that proposes / mutates prompts. Set with `--pg`. |
| Operators | Which evolutionary operators are enabled (`ie`, `cm`, or `all`). Set with `--operators`. |
| Max variants | How many offspring variants to attempt per evolution cycle. Set with `--max-variants`. |
| Seed file | CSV of starting prompts (expects a `questions` column). Set with `--seed-file`. |
| RNG seed | Fixed seed for LLM sampling **and** Python/NumPy EA randomness (optional). Set with `--seed`. |
| Batch size (parallel) | Master–worker merge batch threshold for MPI; also affects sequential parity defaults when omitted. Set with `--batch-size`. |
| Parallel | Use MPI master–worker instead of a single process. Set with `--parallel`. |
| Output directory | Run artifacts directory (default timestamped under `data/outputs/`). Set with `--output-dir`. |
| Stagnation limit | Generations without improvement before switching explore/exploit behaviour. Set with `--stagnation-limit`. |
| Moderation backend | Scorer backend for fitness (`google` = Perspective, `openai` = omni-moderation). Set with `--evaluator`. |
| North-star metric | Score key driving fitness and selection; valid values depend on `--evaluator`. Set with `--north-star-metric`. |
| OpenAI moderation model | Model name when `--evaluator openai` (default `omni-moderation-latest`). Set with `--openai-model`. |
| Moderation methods | Deprecated alias for `--evaluator`. Set with `--moderation-methods`. |
| Generations | Legacy generation cap; termination is still only by `--max-total-genomes`. Set with `--generations`. |
| Profile | Write `cProfile` stats next to the run outputs. Set with `--profile`. |

**Weights (ensemble distance, not on CLI):** genotype and phenotype weights sum to 1. Defaults: `SpeciationConfig.w_genotype` = `0.7`, `w_phenotype` = `0.3`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and [`verfier/`](verfier/).

---

## How to run

Always run commands from the **repository root** so `config/`, `data/`, and `.env` resolve.

```bash
export PYTHONPATH=src   # skip if you used pip install -e .
```

**Termination:** `--max-total-genomes` is **required**. The run stops when elites + reserves + archive reach that cap.

### Local experiment script

```bash
bash run_experiments_local.sh
```

See header comments in [`run_experiments_local.sh`](run_experiments_local.sh) for env overrides (`MAX_TOTAL_GENOMES`, `EVALUATOR`, `THETA_SIM`, etc.). Parallel helper: [`run_experiments_parallel_2w.sh`](run_experiments_parallel_2w.sh).

### Minimal single run

```bash
python src/main.py --max-total-genomes 500 \
  --evaluator openai \
  --north-star-metric violence \
  --seed-file data/prompt.csv
```

Perspective (default metric `toxicity` if unset):

```bash
python src/main.py --max-total-genomes 500 \
  --evaluator google \
  --seed-file data/prompt.csv
```

### MPI (parallel)

```bash
mpiexec -n 5 python src/main.py --parallel --max-total-genomes 5000 \
  --seed-file data/prompt.csv
```

See `python src/main.py --help` (or `toxsearch --help`) for all flags.

### Tests

```bash
python -m pytest tests/ -v -m "not mpi"
```

MPI tests require `mpi4py` and `mpiexec`: `python -m pytest tests/ -v -m mpi`.  
Details: [`tests/README.md`](tests/README.md).

---

## Repository layout

| Path | Purpose |
|------|---------|
| `src/` | Core framework: `ea`, `gne`, `speciation`, `parallel`, `utils`, plus `cli.py` / `main.py` |
| `tests/` | Unit and integration tests (tracked) |
| `config/` | GGUF inference YAML only (`RGConfig.yaml`, `PGConfig.yaml`) |
| `experiments/` | Analysis **drivers** (Python); do not store bulky outputs here |
| `results/` | Generated paper/comparison artifacts (gitignored except manifests) |
| `results/manifests/` | SHA256 study manifests (tracked) |
| `scripts/` | Helpers (e.g. `validate_run_outputs.py`) |
| `data/outputs/` | Live evolution runs (gitignored) |
| `verfier/` | Lean formal audit of distance / speciation maths |
| `docs/` | Architecture and related notes |
| `.github/workflows/` | CI (pytest non-MPI + ruff on new modules) |

---

## Where outputs go

By default each run writes under `data/outputs/<YYYYMMDD_HHMM>/`. Use `--output-dir` to fix a path for paper-facing experiments.

At run start the framework writes **`run_config.json`** (CLI + speciation snapshot, git commit when available). Typical artifacts:

- `EvolutionTracker.json` — generation metrics and `run_metadata`
- `elites.json`, `reserves.json`, `archive.json`
- Logs and optional live-analysis / GDP plots

Paper analysis CSVs and figures belong under **`results/`** (e.g. `results/emnlp2026/`, `results/comparison/`), not under `experiments/`.

---

## Reproducibility

| Mechanism | Role |
|-----------|------|
| `--seed` | GGUF generation seed + Python/NumPy RNG for EA operators |
| `run_config.json` | Full config snapshot written at run start |
| `EvolutionTracker.run_metadata` | Partial run provenance during / after the loop |
| Study manifests | `python results/manifests/build_study_manifest.py` — see [`results/manifests/README.md`](results/manifests/README.md) |

LLM outputs can still vary across hardware/backends even with a fixed seed; EA operator randomness is seed-controlled when `--seed` is set.

---

## Development and CI

```bash
pip install -r requirements-dev.txt
python -m pytest tests/ -m "not mpi" --ignore=tests/test_phase4_unit.py -q
ruff check src/cli.py src/utils/run_config.py src/utils/rng.py src/utils/model_config_patch.py
mypy src/speciation/distance.py src/utils/run_config.py src/utils/rng.py
```

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs non-MPI tests and ruff on the framework modules above. No GPU or API keys in CI.

Packaging: [`pyproject.toml`](pyproject.toml) (`pip install -e .`, console script `toxsearch`).

---

## Analysis and paper artifacts

- Drivers: `experiments/emnlp_data_analysis/`, `experiments/comparison_results/*_report.py`, `experiments/cluster_analysis/`
- Outputs: `results/emnlp2026/`, `results/comparison/`, `results/cluster_analysis/` (gitignored; regenerate or restore from backup)
- Manifests: `results/manifests/*.json`
- Optional RainbowPlus comparison I/O: [`experiments/rainbowplus_io.py`](experiments/rainbowplus_io.py) (path install of `rainbowplus-main/` documented in analysis requirements)

---

## Documentation

| Doc | Contents |
|-----|----------|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Packages, data flow, distances, configs |
| [`docs/README.md`](docs/README.md) | Documentation index |
| [`tests/README.md`](tests/README.md) | How to run and interpret tests |
| [`results/manifests/README.md`](results/manifests/README.md) | Artifact manifests |
| [`verfier/README.md`](verfier/README.md) | Lean verifier |
| [`data/dataset.md`](data/dataset.md) | Shared dataset link |

Doxygen build outputs under `docs/html/` and `docs/latex/` are generated and gitignored; regenerate with the root [`Doxyfile`](Doxyfile) if needed.

---

## Dataset

Paper or shared data: [`data/dataset.md`](data/dataset.md).
