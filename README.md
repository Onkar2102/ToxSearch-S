# ToxSearch-S

Research code for **adversarial prompt search** against LLMs: a quality–diversity evolutionary loop with **semantic speciation** (embedding-space niches). Fitness uses an external moderation API (e.g. Google Perspective) on model responses.

---

## SETUP

**Prerequisites:** Python 3.10+ (see `requirements.txt`), GPU recommended, [Google Perspective API](https://perspectiveapi.com/) key.

**Clone and environment**

```bash
git clone <repository-url>
cd ToxSearch-S
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp env_example.txt .env             # add PERSPECTIVE_API_KEY=...
```

**Models:** Put GGUF weights under `models/` (paths must match `--rg` / `--pg`). Example:

```
models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

**Run from repo root** with `PYTHONPATH=src` so `config/`, `data/`, and `.env` resolve.

**Dataset:** Shared / external data notes — [`data/dataset.md`](data/dataset.md).

**Parallel (MPI, optional):** Install an MPI stack and `mpi4py` (build `mpi4py` against the same MPI you use for `mpiexec` / `srun`). Keys: same `.env`; `--max-total-genomes` required; see `src/main.py --help` for `--parallel`, `--batch-size`, etc.

---

## Quick start

```bash
export PYTHONPATH=src
bash run_experiments_local.sh
```

Minimal sequential run:

```bash
export PYTHONPATH=src
python src/main.py --max-total-genomes 500 --seed-file data/prompt.csv \
  --rg models/llama3.2-3b-instruct-gguf/Llama-3.2-3B-Instruct-Q4_K_M.gguf
```

MPI example (1 master + 4 workers):

```bash
export PYTHONPATH=src
mpiexec -n 5 python src/main.py --parallel --max-total-genomes 5000 \
  --seed-file data/prompt.csv
```

Tests:

```bash
PYTHONPATH=src python -m pytest tests/ -v
```

---

## Documentation

| Topic | Location |
|--------|----------|
| Study design, C1–C3, commands | [`experiments/EXPERIMENT_PLAN.md`](experiments/EXPERIMENT_PLAN.md) |
| CLI flags and defaults | `python src/main.py --help` |
| Post-run throughput / thresholds | `python scripts/experiment_metrics.py <run_dir>` |
| RainbowPlus fork (separate venv) | [`rainbowplus-main/README.md`](rainbowplus-main/README.md) |

Run artifacts (e.g. `EvolutionTracker.json`, `elites.json`, per-rank logs in parallel) are written under `--output-dir` (default `data/outputs/<timestamp>`).
