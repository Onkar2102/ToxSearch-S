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
cd eost-cam-llm

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

Edit the `EXPERIMENTS` array in the script to configure your runs.

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
mpiexec -n 5 python src/main.py --parallel --batch-size 100 --seed-file data/prompt.csv
```

This launches 1 master + 4 workers. The `--batch-size` flag sets `K`, the number of genomes collected before triggering speciation.

### Full Example

```bash
mpiexec -n 9 python src/main.py \
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

The master (rank 0, CPU) manages the population files, runs speciation, and coordinates parent selection. Workers (rank 1..N, GPU) request work, evolve new prompts from selected parents, generate LLM responses, evaluate them via moderation APIs, and return evaluated genomes to the master. Generation 0 distributes seed prompts across workers for initial evaluation.

### Multiple API Keys

To distribute Perspective API rate limits across workers, set a comma-separated list:

```bash
export PERSPECTIVE_API_KEYS="key1,key2,key3,key4"
```

The master assigns keys round-robin to workers. Alternatively, use indexed variables: `PERSPECTIVE_API_KEY_0`, `PERSPECTIVE_API_KEY_1`, etc.

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
