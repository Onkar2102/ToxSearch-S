# ToxSearch-S

Evolutionary search for adversarial prompts against a local GGUF model. Fitness comes from a moderation API. Speciation groups prompts by a chosen distance.

## What you need

| Requirement | Notes |
|-------------|--------|
| **Python** | 3.10 or newer |
| **Dependencies** | `pip install -r requirements.txt` |
| **GPU** | Recommended for GGUF inference |
| **Perspective API** | Required for `--evaluator google` (at least one key in `.env`) |
| **OpenAI API** | Required for `--evaluator openai` (`OPENAI_API_KEY` in `.env`; optional org/project IDs) |
| **MPI (optional)** | Only for `--parallel`; Open MPI (or compatible) plus `mpi4py` built against that MPI |

## Installation

From the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp env_example.txt .env
```

Edit `.env` and set `PERSPECTIVE_API_KEY` or `OPENAI_API_KEY`. Place GGUF weights under `models/`. The CLI flags `--rg` and `--pg` must point at those files. The defaults in `src/cli.py` expect:

```text
models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-IQ4_XS.gguf
```

If your files differ:

```bash
--rg models/<your-folder>/<model>.gguf --pg models/<your-folder>/<model>.gguf
```

Inference settings are in `config/RGConfig.yaml` and `config/PGConfig.yaml`.

After `pip install -e .` the `toxsearch` command is the same entry as `python src/main.py`. Without the editable install, set `PYTHONPATH=src`.

## Project parameters

Defaults follow `SpeciationConfig` in `src/speciation/config.py` and `src/cli.py` unless you override them on the command line.

| Parameter | Meaning |
|-----------|---------|
| Max total genomes | Stop when elites + archive reach this count (required termination). Set with `--max-total-genomes`. |
| Theta similarity | Species assignment radius under `--distance-method`; followers join a leader within this radius. Set with `--theta-sim` (default **0.25**). |
| Distance method | Clustering distance ablation: `embedding` (default), `nli`, `nli_entail`, `nli+embedding`, `objective`, `embedding+objective`. Set with `--distance-method`. |
| NLI model | HuggingFace MNLI model for `nli` / `nli_entail` / `nli+embedding` (default `cross-encoder/nli-deberta-v3-small`). Set with `--nli-model`. |
| Theta merge | Two species whose leaders are closer than this may merge; must be ≤ `--theta-sim`. Set with `--theta-merge`. |
| Min stability generations | Both species must be at least this old before they are allowed to merge. Set with `--min-stability-gens`. |
| Species capacity | Maximum individuals kept per species (excess archived by fitness). Set with `--species-capacity`. |
| Species stagnation | Generations without improvement after which a species can freeze. Set with `--species-stagnation`. |
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

**Weights (ensemble distance, not on CLI):** genotype and phenotype weights sum to 1. Defaults: `SpeciationConfig.w_genotype` = `0.7`, `w_phenotype` = `0.3`.

`python src/main.py --help` lists every flag.

## How to run

Run every step from the repository root so `config/`, `data/`, and `.env` resolve. After `pip install -e .` you can skip `PYTHONPATH`. Otherwise:

```bash
export PYTHONPATH=src
```

### 1. Load the seed data

The search starts from a CSV with one `questions` column. That file is already in the repo:

```text
data/prompt.csv
```

A run loads it through `--seed-file` (the default is this path). You do not need to rebuild it.

To build a different seed from CategoricalHarmfulQA, HarmfulQA, DoNotAnswer, and BeaverTails, write a new file. This downloads those datasets and does not replace `data/prompt.csv`:

```bash
PYTHONPATH=src python src/utils/data_loader.py \
  --n-total 50 --seed 48 --output data/my_prompts.csv
```

Pass that path as `--seed-file` in the next step.

### 2. Run the search

`--max-total-genomes` is required. The process loads the seed, scores prompts, and writes a population until that count is reached.

OpenAI:

```bash
python src/main.py --max-total-genomes 500 \
  --evaluator openai \
  --north-star-metric illicit \
  --seed-file data/prompt.csv
```

Perspective:

```bash
python src/main.py --max-total-genomes 500 \
  --evaluator google \
  --seed-file data/prompt.csv
```

MPI:

```bash
mpiexec -n 5 python src/main.py --parallel --max-total-genomes 500 \
  --seed-file data/prompt.csv
```

### 3. Read the run

Each run writes `data/outputs/<YYYYMMDD_HHMM>/` unless you set `--output-dir`. The directory contains `run_config.json`, `EvolutionTracker.json`, `elites.json`, and `archive.json`.

### 4. Analyze the EvoStar 2027 runs

The study code is `experiments/evostar_2027/`. It reads finished runs from `data/outputs/evostar2027/`. Load those runs first, then run the report scripts in the order in [experiments/evostar_2027/README.md](experiments/evostar_2027/README.md).

Tests, after the steps above:

```bash
python -m pytest tests/ -m "not mpi" --ignore=tests/test_phase4_unit.py -q
```

MPI tests: `bash scripts/run_mpi_tests.sh`.
