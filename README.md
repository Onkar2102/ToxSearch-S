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
