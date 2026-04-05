# ToxSearch-S

ToxSearch-S is a **research framework** for automated red-teaming of large language models (LLMs) using a **quality-diversity evolutionary algorithm** with **semantic speciation**. The goal is to discover prompts that elicit harmful or toxic model responses while maintaining a diverse set of failure modes (semantic niches) rather than converging to a single attack type. The method combines steady-state (μ + λ) evolution with leader-follower clustering in embedding space: the population is partitioned into species by semantic similarity, and selection and variation operate within and across these niches. Fitness is defined as the toxicity of the LLM’s response to a prompt, as scored by an external moderation API (e.g. Google Perspective API). This repository provides the implementation, configuration, and documentation needed to reproduce and extend the experiments.

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for embeddings and models)
- Google Perspective API key

### Setup

```bash
# Clone repository
git clone <repository-url>
cd ToxSearch-S

# Create virtual environment
python -m venv venv
source venv/bin/activate   # macOS/Linux
# or: .\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env_example.txt .env
# Edit .env: add PERSPECTIVE_API_KEY=your_api_key_here
```

### Model Setup

Place GGUF models in the `models/` directory. Example structure:

```
models/
└── llama3.2-3b-instruct-gguf/
    └── Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

---

## How to Start

### Option 1: Run experiments script

```bash
bash run_experiments_local.sh
```

Edit `PARALLEL_EXPERIMENTS` (MPI) and/or `SEQUENTIAL_EXPERIMENTS` in the script to configure your runs.

### Option 2: Run directly

```bash
python src/main.py \
    --max-total-genomes 5000 \
    --generations 50 \
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --seed-file data/prompt.csv
```

**Pre-execution checklist (sequential + MPI)**

- **`--max-total-genomes`**: Required in both modes (CLI enforces this). Termination is by total genomes in elites + reserves + archive, not by `--generations`.
- **`.env`**: `PERSPECTIVE_API_KEY` or `PERSPECTIVE_API_KEYS` (parallel needs at least one key before workers start).
- **`PYTHONPATH=src`** (or run from layouts that already set it); project root as cwd so `config/`, `data/`, and `.env` resolve.
- **Parallel**: `mpiexec` / `srun` with **≥2 ranks** (1 master + ≥1 worker); optional `--output-dir` per experiment; omit `--batch-size` for K=24/39 parity when `--operators all`.
- **Models**: GGUF paths exist; rank 0 updates `RGConfig.yaml` / `PGConfig.yaml` before workers load them (parallel).
- **Post-run (parallel)**: `scripts/aggregate_worker_metrics.py <run_dir>` for `worker_metrics.json` if you need load-balance / API-wait summaries.
- **Per-file `src/` review tracker**: [`docs/SRC_FINAL_REVIEW_CHECKLIST.md`](docs/SRC_FINAL_REVIEW_CHECKLIST.md) (line-chunk checklists for large modules).

### Metrics and outputs (EvolutionTracker, workers, C1–C3)

Canonical artifacts live under each run’s **`--output-dir`** (default: `data/outputs/...`):

| File | Role |
|------|------|
| `EvolutionTracker.json` | Per-generation population, speciation, budget, and timing fields; run-level metadata under `run_metadata`. |
| `master_metrics.json` | Parallel master only: merge batch history, timing aggregates. |
| `*_workerN.log` / `*_master.log` | Per-rank logs in parallel mode. |

**Per-generation fields (high level)**

- **`evaluated_this_generation`**: Count of moderation/evaluation completions attributed to that generation. In **parallel** mode the master sets this from evaluated variants **since the previous merge/speciation** (not a cumulative run total). In **sequential** mode it is aligned with `budget.api_calls` for that generation when available. Legacy rows may only have ambiguous `total_evaluated` (cumulative); analysis scripts should prefer `evaluated_this_generation`.
- **`discarded_this_generation`**: Variants rejected or dropped in that merge/speciation window (parallel); sequential uses the same field when provided.
- **`cumulative_variants_evaluated` / `cumulative_variants_discarded`**: Run-level counters on the tracker root; updated when generation rows are written.
- **`variants_integrated`**: Genomes accepted into the population from that generation’s variant set (after dedup/speciation rules).
- **`budget.*`**: `llm_calls`, `api_calls`, and timing splits from `calculate_budget_metrics` / generation statistics (`src/utils/population_io.py`). Variant-creation LLM calls are counted when `creation_info.operator` or `operator` matches the LLM operator set in code.

**`generation_duration_seconds` and `generation_duration_scope`**

Both modes now use the same **trailing edge**: `generation_duration_scope` is **`through_evolution_tracker_statistics_write`** — wall time from a generation-local start until the main `update_evolution_tracker_with_statistics` call **persists** that generation’s row (just before adaptive selection runs on the tracker).

- **Start anchor (still mode-specific):** **Sequential** — gen 0: before initial response generation; gen ≥ 1: start of the generation loop (evolution through pre-tracker work, including operator-effectiveness CSV, first live-analysis pass, and auxiliary tracker merge). **Parallel** — immediately after the previous generation’s full tracker update returned on the master (`gen_start` reset), through buffer fill, merge/dedup, speciation, and master-side `calculate_generation_statistics` / tracker prep inside `_update_tracker`.
- **Not included:** adaptive selection update (`update_adaptive_selection_logic`) and any visualization passes **after** that tracker write (sequential may run an additional `run_live_analysis` after adaptive selection).

Absolute seconds are still **not** directly comparable across modes (parallel overlaps worker GPU time; sequential is single-process), but the **definition of what the duration stops at** matches for C2-style accounting.

**Experiments C1–C3**

- Full study design, budgets, and gaps: [`experiments/EXPERIMENT_PLAN.md`](experiments/EXPERIMENT_PLAN.md).
- **C1** (quality/diversity, three-way): helpers in [`experiments/rainbowplus_io.py`](experiments/rainbowplus_io.py), omnibus-style entry [`experiments/rq1_three_way.py`](experiments/rq1_three_way.py), and an approximate **single-pool** sequential baseline launcher [`scripts/run_c1_baseline_single_pool.sh`](scripts/run_c1_baseline_single_pool.sh) (see script header for caveats). Extend [`experiments/rq1.py`](experiments/rq1.py) for publication figures.
- **C2** (sequential vs parallel): [`experiments/compare_sequential_vs_parallel.py`](experiments/compare_sequential_vs_parallel.py), [`scripts/experiment_metrics.py`](scripts/experiment_metrics.py), [`scripts/aggregate_worker_metrics.py`](scripts/aggregate_worker_metrics.py).
- **C3** (species vs archive cells): [`experiments/c3_species_bridge.py`](experiments/c3_species_bridge.py) bridges ToxSearch-S tracker species counts to RainbowPlus-style archive keys when JSONL is available.

**Parallel merge batching**

Sequential runs ignore `--batch-size`. In parallel, omitting `--batch-size` uses merge **K** in parity with sequential operator counts (see Parallel Mode section below).

---

## Parallel Mode (MPI)

ToxSearch-S supports distributed execution via MPI, using a master-worker architecture. Rank 0 runs the master (population management, speciation) on CPU, while ranks 1..N run as workers (evolution, response generation, evaluation) — ideally on separate GPUs.

### Prerequisites

- An MPI runtime: [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)
- `mpi4py` Python package (see below)

### Installing mpi4py (especially on clusters)

- **Local / generic:** `pip install mpi4py` (builds against whatever MPI is in `PATH`, or uses a stub).
- **Cluster (Spack + Slurm):** So that `mpi4py` uses the cluster’s MPI (and works with `srun`), install it **after** activating the Spack env that provides Open MPI, then your venv:
  ```bash
  spack env activate default-nlp-x86_64-25111801   # or your Spack env name
  source .spvenv/bin/activate                        # or your venv path
  pip install mpi4py
  ```
  This builds `mpi4py` against the Spack Open MPI. If your Spack Open MPI is built without `mpiexec`/`mpirun` (e.g. `~legacylaunchers`), use **srun** to launch (see `rc_script.sh` and “Running on HPC Clusters” below).

### Basic Usage

```bash
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --max-total-genomes 5000 --batch-size 100 --seed-file data/prompt.csv
```

This launches 1 master + 4 workers. **`--max-total-genomes` is required** for parallel mode (primary termination). **`--batch-size`** optionally overrides merge `K` (buffered genomes before speciation **after** generation 0). If you **omit** `--batch-size` with `--operators all`, `K` follows **sequential parity**: **24** (default selection mode) or **39** (explore/exploit), scaled by `--max-variants`. With `--operators cm` or `ie`, omitting `--batch-size` uses **100**. **Generation 0** waits until **all** seed prompts have been evaluated, then runs a **single** merge/speciation on the full buffered bootstrap set (not capped by `K`).

### Full Example

```bash
PYTHONPATH=src mpiexec -n 9 python src/main.py \
    --parallel \
    --max-total-genomes 10000 \
    --batch-size 200 \
    --generations 50 \
    --seed-file data/prompt.csv \
    --seed 42 \
    --operators all \
    --moderation-methods google \
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --pg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --theta-sim 0.2 \
    --species-capacity 100
```

### How It Works

- **Config:** Only rank 0 updates `RGConfig.yaml` and `PGConfig.yaml` from `--rg` and `--pg` before the run; workers load models from those YAMLs (paths broadcast in config). Config and seed file paths are resolved from the project root.
- Rank 0 is the **master** (CPU only): receives worker messages, keeps per-worker in-memory buffers, runs merge/dedup/speciation, updates population files, and writes tracker/statistics. Workers send **WORKER_READY** after loading models; master waits for all (or **WORKER_INIT_FAILED** / timeout) before starting the dispatch loop.
- Ranks 1..N are **workers** (typically one GPU each): request work, evolve prompts from parents, generate responses, evaluate with moderation APIs, and send evaluated genomes back immediately.
- **Generation 0** uses index ranges (`prompt_start`, `prompt_end`) so workers read seed prompts locally from the resolved seed file path.
- `temp.json` is **transient**: it is filled only during merge/speciation and cleared by speciation; evaluated genomes wait in in-memory buffers first.
- A generation increments when speciation runs. In parallel mode, **generation 0** runs once all seed evaluations are in the master buffer (full bootstrap). Each later generation runs when at least **merge K** evaluated genomes are buffered (`--batch-size` if set, else 24/39 parity for `--operators all` as above). **Termination** is by `--max-total-genomes` (total genomes in elites + reserves + archive).

### API key and .env

Parallel mode requires at least one Perspective API key. You can:

- Put it in **`.env`** in the project root: `PERSPECTIVE_API_KEY=your_key` (or `PERSPECTIVE_API_KEYS=key1,key2`). The parallel runtime loads `.env` from the project root at startup; the sequential path loads it when the evaluator is imported.
- Or export before running: `export PERSPECTIVE_API_KEY=your_key`

To distribute rate limits across workers, set a comma-separated list:

```bash
export PERSPECTIVE_API_KEYS="key1,key2,key3,key4"
```

The master assigns keys round-robin to workers. Alternatively, use indexed variables: `PERSPECTIVE_API_KEY_0`, `PERSPECTIVE_API_KEY_1`, etc. Storing multiple keys in `PERSPECTIVE_API_KEY` as comma-separated values is also supported.

### Logs

Parallel runs write one log file per rank:

- `..._master.log` for rank 0
- `..._worker1.log`, `..._worker2.log`, ... for worker ranks

This avoids interleaved concurrent writes to one file and makes debugging per-rank behavior much easier.

#### Worker log messages (what they mean)

Worker logs follow the same structure for every rank; the only difference is the worker index. In order of appearance:

| Message | When | Meaning |
|--------|------|--------|
| `MPI rank N of M starting` | Once at startup | This process is worker N in a world of M ranks. |
| `Worker N received config: [...]` | After master broadcasts | Worker received run configuration (keys: north_star_metric, seed_file, outputs_path, rg_config_path, pg_config_path, etc.). |
| `Worker N: ResponseGenerator initialised` | If not injected | LLM (response generator) loaded for this worker. |
| `Worker N: PromptGenerator initialised` | If not injected | Prompt-generation model loaded. |
| `Worker N: HybridModerationEvaluator initialised` | If not injected | Moderation (e.g. Perspective API) evaluator ready. |
| (Master) `Waiting for all N worker(s) to report ready (timeout=900s)...` | After broadcast | Master waits for each worker to send WORKER_READY (or WORKER_INIT_FAILED); then starts dispatch loop. |
| (Master) `All workers ready. Starting dispatch loop.` | When all workers ready | All workers finished init; master begins the main receive loop. |
| `Worker N ready. Entering request loop.` | Before first request | Worker sent WORKER_READY and is idle; will repeatedly request work from the master. |
| `Sent PARENTS_REQUEST request_id=... (cycle=C, total_sent=S)` | Every cycle | Worker asked master for work; `cycle` = request count, `total_sent` = total variants sent so far. |
| **Generation 0 (bootstrap)** | | |
| `GEN0_BATCH received: request_id=... prompts[S:E] (N prompts) key_idx=...` | When master sends seed batch | Worker received a slice of the seed file: prompts from index S to E (N prompts). `key_idx` is the Perspective API key index (if multiple keys). |
| `Worker N: using Perspective API key index K` | If master assigned a key | This batch will use the K-th API key for moderation. |
| `Worker N: received empty GEN0 batch ...` | If N=0 | **Warning:** This worker was assigned no seed prompts (e.g. more workers than prompts). |
| `Gen0 batch: loaded N prompts from seed file, processing...` | After loading CSV slice | Worker loaded N prompts and will generate + evaluate each. |
| `Gen0 progress: X/Y prompts processed (ok, errors)` | Every 5 prompts (DEBUG/INFO) | Progress within the current Gen0 batch. |
| `Gen0 batch complete: sent N variants (ok, errors) for request_id=... in X.XXs` | End of Gen0 batch | Gen0 batch finished; N variants sent, wall-clock time. |
| **Evolution cycles (generation ≥ 1)** | | |
| `Received shutdown (None) from master. Exiting request loop.` | When master signals stop | Master sent shutdown; worker exits the request loop. |
| `PARENTS received: request_id=... parents=P top_10=T key_idx=... (cycle=C)` | When master sends parents | Worker received P parents and T top-10 exemplars for this evolution cycle. |
| `Worker N: using Perspective API key index K` | If key assigned for this batch | This evolution batch uses the K-th API key. |
| `Evolution cycle C: generating variants from P parents...` | Before variant generation | Starting variant generation for cycle C with P parents. |
| `Generated N variant(s) in X.XXs. Processing pipeline...` | After operator, before LLM/eval | Variation operator produced N prompts; worker will now generate responses and evaluate. |
| `Evolution cycle C complete: sent N variants (ok, errors) in X.XXs (total_sent=S)` | End of cycle | Cycle C done: N variants sent this cycle, cumulative total_sent=S. |
| **Shutdown** | | |
| `Worker N done. total_variants_sent=S cycles=C total_errors=E uptime=X.Xs` | On exit | Summary: S variants sent over C cycles, E errors, total uptime. |

At **DEBUG** level you also see per-variant lines (e.g. which prompt is being generated or evaluated). **ERROR** lines indicate pipeline failures (e.g. LLM or API errors) for a specific variant; the worker continues and reports that variant as failed.

### Running tests

From the project root, with `PYTHONPATH=src`:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

See `tests/README.md` for details. Tests include refusal-detector unit tests and config-loading smoke tests.

### Running on HPC Clusters

You do **not** need to add code to detect GPUs or CUDA. The codebase already does that:

- **Device selection** is handled in `src/utils/device_utils.py` (DeviceManager). It chooses MPS (Apple), then CUDA (NVIDIA), then CPU. PyTorch and the embedding model use this. The LLM (llama.cpp) uses the same device via `config/RGConfig.yaml` (`device_config.cuda` / `gpu_layers`).
- **CUDA**: When `torch.cuda.is_available()` is true, the app uses CUDA. Each process uses “device 0” from its own point of view. On HPC, you make that “device 0” be the correct GPU by controlling what each process sees (see below).

**Recommended approach on HPC: one GPU per worker via the scheduler**

- Reserve **one GPU per MPI rank** (or one per worker; master can share a node with a worker or run on CPU-only).
- Let the **job scheduler** set `CUDA_VISIBLE_DEVICES` (or equivalent) so that each rank sees exactly one GPU. Then our code’s “use cuda device 0” is correct for every worker; no code changes are required.
- Run from the **project root** and set `PYTHONPATH=src` so config and imports resolve.

**Example (SLURM): one GPU per task, N tasks on one node**

```bash
#!/bin/bash
#SBATCH --job-name=toxsearch
#SBATCH --nodes=1
#SBATCH --ntasks=5
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8

cd /path/to/ToxSearch-S
source venv/bin/activate
export PYTHONPATH=src

# Optional: pass API key into the job
export PERSPECTIVE_API_KEY="your_key"

# Run 1 master + 4 workers; each task gets one GPU from the scheduler
# Parallel requires --max-total-genomes; optional: set PERSPECTIVE_API_KEY or use .env
srun --gpus-per-task=1 python src/main.py --parallel --max-total-genomes 5000 --batch-size 100 \
  --generations 50 --seed-file data/prompt.csv --seed 42 \
  --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf \
  --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf
```

If your cluster uses `mpiexec` instead of `srun` for launching MPI, request the same number of GPUs and run:

```bash
mpiexec -n 5 python src/main.py --parallel --batch-size 100 ...
```

Ensure the MPI launcher or scheduler is configured so that different ranks get different GPUs (many clusters do this when you request `--gpus-per-task=1` and multiple tasks).

**Optional: force CPU for master (rank 0)**

If the master should not use a GPU (e.g. to leave all GPUs for workers), set `CUDA_VISIBLE_DEVICES` only for worker ranks. That usually requires a small wrapper or launcher that sets the env per rank; the application code does not need to be changed for standard “one GPU per worker” setups.

---

## Experiment metrics (throughput and search performance)

You can report two kinds of results from a run:

1. **Throughput** — evaluated genomes per second (how many prompts got a response and moderation per second).
2. **Search performance** — whether the search finds more toxic responses faster (best toxicity over wall-clock time; time to reach a given toxicity threshold).

Both are computed from `EvolutionTracker.json`. Use the helper script (from project root):

```bash
PYTHONPATH=src python scripts/experiment_metrics.py [run_dir]
```

Example: `PYTHONPATH=src python scripts/experiment_metrics.py data/outputs/20260311_1742`

- **run_dir** is a run directory under `data/outputs/<timestamp>`. If omitted, the latest run is used.
- The script prints overall and per-generation throughput, final best fitness, and time to reach toxicity thresholds (e.g. 0.2, 0.3, 0.4, 0.5).
- It writes `run_dir/experiment_metrics.json` and, if matplotlib is available, `run_dir/figures/toxicity_vs_time.png` (best fitness vs cumulative time).

Data source: each generation already has `generation_duration_seconds` and either `total_evaluated` (parallel) or `budget.llm_calls` (sequential); best toxicity is in `best_fitness` / `max_score_variants`.

---

## Reproducibility

To reproduce or compare experimental results, the following should be fixed or recorded.

**Environment**
- Python version and dependency versions (see `requirements.txt`). For parallel runs, note the MPI implementation and that `mpi4py` must be built against it (see “Installing mpi4py” under Parallel Mode).
- Embedding model and dimension (e.g. `all-MiniLM-L6-v2`, 384). LLM: exact GGUF path and quantization.

**Randomness**
- For full reproducibility, set a fixed random seed (e.g. `random.seed`, `numpy.random.seed`) before the run; the codebase does not set a global seed by default.

**Inputs**
- Seed prompts: path to the CSV and that it has a `questions` column; document row count.
- Moderation: metric name (e.g. `toxicity`) and API (e.g. Google Perspective). Multiple API keys only distribute rate limits; they do not change the metric.

**Run configuration**
- Record all command-line arguments (or config): `--generations`, `--max-total-genomes`, `--batch-size`, `--theta-sim`, `--theta-merge`, `--species-capacity`, `--seed-file`, `--seed`, model paths (`--rg`, `--pg`). For parallel runs, record the number of MPI ranks and GPU assignment (e.g. one GPU per worker). Config and seed paths are resolved from the project root.

**Outputs**
- Each run writes to an output directory (e.g. under `data/outputs/`) containing: `EvolutionTracker.json` (per-generation and cumulative metrics), `elites.json`, `reserves.json`, `archive.json`, `speciation_state.json`, `genome_tracker.json`, and optionally figures in `figures/`. Parallel runs also produce one log file per rank. See [ARCHITECTURE.md](ARCHITECTURE.md) for a full list of artifacts and their role in the method.

**Example minimal reproducible run (sequential)**
```bash
export PYTHONPATH=src
python src/main.py --generations 10 --seed-file data/prompt.csv \
  --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```
Document the seed file, model path, and any non-default flags.

---

## Hyperparameters

### Evolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations` | None | Max evolution generations |
| `--max-total-genomes` | None | Total genomes cap (elites + reserves + archive). **Required for parallel.** |
| `--stagnation-limit` | 5 | Generations without improvement before EXPLORE mode |
| `--max-variants` | 1 | Max variants per evolution cycle |
| `--operators` | all | Operators: `ie` (InformedEvolution only), `cm` (all except InformedEvolution), `all` |
| `--seed-file` | data/prompt.csv | Seed prompts CSV (requires `questions` column) |
| `--seed` | None | Fixed seed for LLM generation (RG/PG); improves reproducibility when set |
| `--parallel` | off | Run in MPI master-worker mode (use with `mpiexec` or `srun`) |
| `--batch-size` | None | Parallel merge `K` after generation 0. Omit for auto: **24/39 × max-variants** (`operators all`, from tracker mode) or **100** (`cm`/`ie`). Generation 0 always uses all seed evaluations |
| `--output-dir` | None | Output directory (default: `data/outputs/<timestamp>`). Set for reproducible paths. |

### Speciation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--theta-sim` | 0.2 | Similarity threshold for species assignment |
| `--theta-merge` | 0.1 | Merge threshold (≤ theta-sim) |
| `--species-capacity` | 100 | Max individuals per species |
| `--cluster0-max-capacity` | 1000 | Max individuals in reserves (cluster 0) |
| `--cluster0-min-cluster-size` | 2 | Min cluster size for new species formation |
| `--min-island-size` | 2 | Min species size before dissolution |
| `--species-stagnation` | 20 | Generations without improvement before freezing |

### Embedding

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-model` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `--embedding-dim` | 384 | Embedding dimensionality |
| `--embedding-batch-size` | 64 | Embedding batch size |

### Models

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rg` | llama3.2-3b (Q4_K_M) | Response generator GGUF path |
| `--pg` | llama3.2-3b (Q4_K_M) | Prompt generator GGUF path |

### Moderation

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--moderation-methods` | google | Moderation API: `google` or `all` |

---

## Troubleshooting

### `Parallel mode requires at least one Perspective API key`

Set the key in the environment or in `.env` in the project root (see [API key and .env](#api-key-and-env)). Parallel runs load `.env` at startup so the key is available before workers are sent work.

### `llama_context: n_ctx_per_seq (1024) < n_ctx_train (131072)`

This is an **informational warning** from llama.cpp: the model was trained with a large context window (e.g. 131072 tokens), but inference is using a smaller context (`n_ctx`, e.g. 1024 or 4096) for memory and speed. You can ignore it unless you need longer context; to use a larger window, set `context_length` in your model’s `device_config` (e.g. in `config/RGConfig.yaml` or the config passed to the response generator).

### `AttributeError: 'LlamaModel' object has no attribute 'sampler'` (during exit)

This comes from a **known bug in llama-cpp-python**: the destructor `LlamaModel.__del__` calls `close()`, which accesses `self.sampler` without checking that it exists. It often appears when the process is shutting down (e.g. MPI workers after the job is stopped) and can be safely ignored; it does not affect results. Workers now call an explicit model cleanup before exit to reduce the chance of this message. To fix it in the library, patch `llama_cpp/_internals.py` so that `close()` uses `hasattr(self, "sampler") and self.sampler is not None` before using `self.sampler`, or upgrade to a version/fork that includes this fix (e.g. [llama-cpp-python#2104](https://github.com/abetlen/llama-cpp-python/issues/2104)).
