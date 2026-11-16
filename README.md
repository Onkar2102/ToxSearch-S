# ToxSearch-S

Evolutionary search for adversarial prompts. A local GGUF model writes each prompt and its response. Fitness is an OpenAI or Perspective moderation score. Speciation groups prompts so the search covers more than one niche.

The EvoStar 2027 study and its analysis scripts are in [`experiments/evostar_2027/`](experiments/evostar_2027/ANALYSIS.md).

## Setup

Python 3.10 or newer. A GPU is recommended for GGUF inference.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp env_example.txt .env
```

Set `OPENAI_API_KEY` for OpenAI moderation, or `PERSPECTIVE_API_KEY` for Google Perspective. Place the response and prompt GGUF files under `models/`. The default path is:

```text
models/llama3.2-1b-instruct-gguf/Llama-3.2-1B-Instruct-IQ4_XS.gguf
```

Pass `--rg` and `--pg` if the files live somewhere else. Inference settings are in `config/RGConfig.yaml` and `config/PGConfig.yaml`.

Run every command from the repository root. If you did not `pip install -e .`, set `PYTHONPATH=src` first.

## Run a search

`--max-total-genomes` is required. The run stops when the stored population reaches that count. Starting prompts are the `questions` column of `data/prompt.csv`.

```bash
python src/main.py \
  --max-total-genomes 2000 \
  --evaluator openai \
  --north-star-metric illicit \
  --seed-file data/prompt.csv \
  --seed 0 \
  --clustering-method leader_follower \
  --distance-method embedding \
  --theta-sim 0.1 \
  --operators all \
  --output-dir data/outputs/example_run
```

Other clustering choices are `dbscan` and `none`. Other distances include `nli_entail`. `python src/main.py --help` lists the flags. MPI runs add `--parallel` and are started with `mpiexec`.

Each run writes `run_config.json`, `EvolutionTracker.json`, `elites.json`, and `archive.json` under the output directory. The default directory is `data/outputs/<timestamp>/`. Those outputs are not committed.

## Tests

```bash
python -m pytest tests/ -m "not mpi" --ignore=tests/test_phase4_unit.py -q
```

The MPI suite is `bash scripts/run_mpi_tests.sh`. It needs `mpiexec`.
