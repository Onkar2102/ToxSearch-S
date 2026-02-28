# Speciated ToxSearch

A quality-diversity evolutionary framework for LLM red-teaming with semantic speciation. Evolves prompts to elicit toxic responses from target models while maintaining diverse semantic niches.

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
    --threshold 0.99 \
    --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf \
    --seed-file data/prompt.csv
```

---

## Parallel Mode (MPI)

ToxSearch-S supports distributed execution via MPI, using a master-worker architecture. Rank 0 runs the master (population management, speciation) on CPU, while ranks 1..N run as workers (evolution, response generation, evaluation) — ideally on separate GPUs.

### Prerequisites

- An MPI runtime: [OpenMPI](https://www.open-mpi.org/) or [MPICH](https://www.mpich.org/)
- `mpi4py` Python package (`pip install mpi4py`)

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
    --threshold 0.99 \
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

## Hyperparameters

### Evolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations` | None | Max evolution generations (runs until threshold if not set) |
| `--threshold` | 0.99 | North-star toxicity threshold for stopping |
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
