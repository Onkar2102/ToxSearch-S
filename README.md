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
    --generations 50 \
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --seed-file data/prompt.csv
```

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
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 --seed-file data/prompt.csv
```

This launches 1 master + 4 workers. The `--batch-size` flag sets `K`, the number of genomes collected before triggering speciation.

### Full Example

```bash
PYTHONPATH=src mpiexec -n 9 python src/main.py \
    --parallel \
    --batch-size 200 \
    --generations 50 \
    --seed-file data/prompt.csv \
    --operators all \
    --moderation-methods google \
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --theta-sim 0.2 \
    --species-capacity 100
```

### How It Works

- Rank 0 is the **single-threaded master**: receives worker messages, keeps per-worker in-memory buffers, runs merge/dedup/speciation, updates population files, and writes tracker/statistics.
- Ranks 1..N are **workers**: request work, evolve prompts from parents, generate responses, evaluate with moderation APIs, and send evaluated genomes back immediately.
- **Generation 0** uses index ranges (`prompt_start`, `prompt_end`) so workers read seed prompts locally.
- `temp.json` is **transient**: it is filled only during merge/speciation and cleared by speciation; evaluated genomes wait in in-memory buffers first.
- A generation increments when speciation runs. In parallel mode, this is driven by `K` (`--batch-size`): once `K` evaluated genomes are buffered, master runs speciation.

### Multiple API Keys

To distribute Perspective API rate limits across workers, set a comma-separated list:

```bash
export PERSPECTIVE_API_KEYS="key1,key2,key3,key4"
```

The master assigns keys round-robin to workers. Alternatively, use indexed variables: `PERSPECTIVE_API_KEY_0`, `PERSPECTIVE_API_KEY_1`, etc.

If you currently store multiple keys in `PERSPECTIVE_API_KEY` as comma-separated values, that is also supported.

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
| `Worker N received config: [...]` | After master broadcasts | Worker received run configuration (keys: north_star_metric, seed_file, outputs_path, etc.). |
| `Worker N: ResponseGenerator initialised` | If not injected | LLM (response generator) loaded for this worker. |
| `Worker N: PromptGenerator initialised` | If not injected | Prompt-generation model loaded. |
| `Worker N: HybridModerationEvaluator initialised` | If not injected | Moderation (e.g. Perspective API) evaluator ready. |
| `Worker N ready. Entering request loop.` | Before first request | Worker is idle and will now repeatedly request work from the master. |
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

### Running MPI Tests

```bash
# Serialization round-trip (2 ranks)
mpiexec -n 2 python tests/test_phase8_serialization.py

# Full integration test (3 ranks)
mpiexec -n 3 python tests/test_phase8_integration.py

# All MPI tests
mpiexec -n 3 python tests/test_phase3_mpi.py
mpiexec -n 3 python tests/test_phase4_mpi.py
mpiexec -n 3 python tests/test_phase5_mpi.py
mpiexec -n 3 python tests/test_phase67_mpi.py
```

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
srun --gpus-per-task=1 python src/main.py --parallel --batch-size 100 \
  --generations 50 --seed-file data/prompt.csv \
  --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
  --pg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

If your cluster uses `mpiexec` instead of `srun` for launching MPI, request the same number of GPUs and run:

```bash
mpiexec -n 5 python src/main.py --parallel --batch-size 100 ...
```

Ensure the MPI launcher or scheduler is configured so that different ranks get different GPUs (many clusters do this when you request `--gpus-per-task=1` and multiple tasks).

**Optional: force CPU for master (rank 0)**

If the master should not use a GPU (e.g. to leave all GPUs for workers), set `CUDA_VISIBLE_DEVICES` only for worker ranks. That usually requires a small wrapper or launcher that sets the env per rank; the application code does not need to be changed for standard “one GPU per worker” setups.

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
- Record all command-line arguments (or config): `--generations`, `--batch-size`, `--theta-sim`, `--theta-merge`, `--species-capacity`, `--seed-file`, model paths. For parallel runs, record the number of MPI ranks and GPU assignment (e.g. one GPU per worker).

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
| `--stagnation-limit` | 5 | Generations without improvement before EXPLORE mode |
| `--max-variants` | 1 | Max variants per evolution cycle |
| `--operators` | all | Operators: `ie`, `cm`, or `all` |
| `--seed-file` | data/prompt.csv | Seed prompts CSV (requires `questions` column) |
| `--parallel` | off | Run in MPI master-worker mode (use with `mpiexec`) |
| `--batch-size` | 100 | Genomes per generation batch (`K`) for parallel mode |

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
