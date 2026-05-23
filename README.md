# ToxSearch-S

Evolutionary search for **adversarial prompts** against local LLMs (GGUF). The loop is quality–diversity style: fitness comes from an external moderation API (Google [Perspective](https://developers.perspectiveapi.com/) or OpenAI omni-moderation) applied to model outputs, and **semantic speciation** keeps diversity by clustering prompts in embedding space.

**Index**

- [What you need](#what-you-need)
- [Installation](#installation)
- [Models (GGUF)](#models-gguf)
- [Project parameters](#project-parameters)
- [How to run](#how-to-run)
- [Where outputs go](#where-outputs-go)
- [Dataset](#dataset)

---

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10 or newer |
| **Dependencies** | `pip install -r requirements.txt` |
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
   python3 -m venv venv
   source venv/bin/activate          # Windows: venv\Scripts\activate
   ```

   `run_experiments_local.sh` also looks for `.venv` or `.spvenv` if `venv` is missing.

3. **Install Python packages**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API keys**

   ```bash
   cp env_example.txt .env
   ```

   Edit `.env` and set `PERSPECTIVE_API_KEY` for Google runs, or `OPENAI_API_KEY` (and optional `OPENAI_ORG_ID` / `OPENAI_PROJECT_ID`) for OpenAI runs (see `env_example.txt`).

---

## Models (GGUF)

Download weights and place them under `models/`. The CLI flags **`--rg`** (response generator) and **`--pg`** (prompt generator) must point at `.gguf` files that exist on your machine, or match paths under `models/` the same way the YAML configs do.

Default paths in `src/main.py` currently expect something like:

```text
models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf
```

If your files differ, pass explicit paths, for example:

```bash
--rg models/<your-folder>/<model>.gguf --pg models/<your-folder>/<model>.gguf
```

---

## Project parameters

Defaults follow `src/main.py` and `SpeciationConfig` unless you override them on the command line (`python src/main.py …`).

| Parameter | Meaning |
|-----------|---------|
| Max total genomes | Stop when elites + reserves + archive reach this count (required termination). Set with `--max-total-genomes`. |
| Theta similarity | Species assignment radius in **ensemble** (genotype + phenotype) distance; followers join a leader within this radius. Set with `--theta-sim`. |
| Theta merge | Two species whose leaders are closer than this may merge; must be ≤ the similarity threshold (`--theta-sim`). Set with `--theta-merge`. |
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
| RNG seed | Fixed seed for reproducible LLM sampling (optional). Set with `--seed`. |
| Batch size (parallel) | Master–worker merge batch threshold for MPI; also affects sequential parity defaults when omitted. Set with `--batch-size`. |
| Parallel | Use MPI master–worker instead of a single process. Set with `--parallel`. |
| Output directory | Run artifacts directory (default timestamped under `data/outputs/`). Set with `--output-dir`. |
| Stagnation limit | Generations without improvement before switching explore/exploit behaviour. Set with `--stagnation-limit`. |
| Moderation backend | Scorer backend for fitness (`google` = Perspective, `openai` = omni-moderation). Set with `--evaluator`. |
| North-star metric | Score key driving fitness and selection; valid values depend on `--evaluator`. Set with `--north-star-metric`. |
| OpenAI moderation model | Model name when `--evaluator openai` (default `omni-moderation-latest`). Set with `--openai-model`. |
| Moderation methods | Deprecated alias for `--evaluator` (`google`/`perspective`/`all` → google; `openai`/`omni` → openai). Set with `--moderation-methods`. |
| Generations | Legacy generation cap; termination is still only by the total genome cap (`--max-total-genomes`). Set with `--generations`. |
| Profile | Write `cProfile` stats next to the run outputs. Set with `--profile`. |

**Weights (ensemble distance, not on CLI):** genotype and phenotype weights sum to 1 and scale the two distance terms before applying `--theta-sim` / `--theta-merge`. Defaults: `SpeciationConfig.w_genotype` = `0.7`, `w_phenotype` = `0.3`.

---

## How to run

Always run commands from the **repository root** so `config/`, `data/`, and `.env` resolve. Set:

```bash
export PYTHONPATH=src
```

(or prefix individual commands with `PYTHONPATH=src`).

**Termination:** `--max-total-genomes` is **required**. The run stops when the total number of genomes (elites, reserves, and archive) reaches that cap.

### Local experiment script

Sequential sweeps over similarity thresholds (MPI block is optional and commented inside the script):

```bash
export PYTHONPATH=src
bash run_experiments_local.sh
```

See the header comments in `run_experiments_local.sh` for environment variables (`RUN_SEQUENTIAL`, `THETA_VALUES`, `RUN_PARALLEL`, etc.).

### Minimal single run

```bash
export PYTHONPATH=src
python src/main.py --max-total-genomes 500 \
  --evaluator openai \
  --north-star-metric violence \
  --seed-file data/prompt.csv
```

Perspective example:

```bash
export PYTHONPATH=src
python src/main.py --max-total-genomes 500 \
  --evaluator google \
  --north-star-metric threat \
  --seed-file data/prompt.csv
```

Default run (Perspective, toxicity metric):

```bash
export PYTHONPATH=src
python src/main.py --max-total-genomes 500 \
  --seed-file data/prompt.csv
```

Add `--rg` / `--pg` if your GGUF paths differ from the defaults in `src/main.py`.

### MPI (parallel)

Use one process per rank; rank 0 is the master. Everyone needs the same `.env` and the same `--max-total-genomes`. Example with five ranks (one master + four workers):

```bash
export PYTHONPATH=src
mpiexec -n 5 python src/main.py --parallel --max-total-genomes 5000 \
  --seed-file data/prompt.csv
```

See `python src/main.py --help` for `--batch-size`, speciation knobs, and other flags.

### Tests

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Where outputs go

By default each run writes under `data/outputs/<YYYYMMDD_HHMM>/`. Use `--output-dir` to fix a directory name (for reproducible experiments or paper artifacts).

Typical files include `EvolutionTracker.json`, `elites.json`, population-related JSON, logs, and plots from live analysis when that path runs successfully.

---

## Dataset

Paper or shared data details: [`data/dataset.md`](data/dataset.md).
