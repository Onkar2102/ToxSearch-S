# EvoStar 2027 analysis

Run these commands from the repository root:

```bash
source venv/bin/activate
export PYTHONPATH=src:.
```

The scripts read finished evolution runs. They do not start a new search.

## Where the code is

| Path | Role |
|------|------|
| `rq1/` | Offline distance calibration, frozen radii, neighborhood reports |
| `rq2/scripts/` | Comparisons across the 50 runs |
| `rq2/reports/` | Tables and curves those scripts write |
| `rq3/scripts/` | Extra audits of the same runs |
| `validation/` | Checks and the anytime population-structure figure |
| `data/` | Local prompt pool and distance matrices (not in git) |
| `rq1/frozen_config.json` | Radii and budget used for the live runs |

## Runs the scripts expect

Fifty directories, ten seeds (0–9) for each configuration:

```text
data/outputs/evostar2027/lf_embedding/rep{i}_s{i}
data/outputs/evostar2027/inc_embedding/rep{i}_s{i}
data/outputs/evostar2027/lf_nli/rep{i}_s{i}
data/outputs/evostar2027/inc_nli/rep{i}_s{i}
data/outputs/evostar2027/no_speciation/rep{i}_s{i}
```

Each run used `data/prompt.csv`, a budget of 2000 evaluations, OpenAI `illicit` as fitness, and the radii in `rq1/frozen_config.json`. `embedding` is leader–follower or incremental DBSCAN with the embedding distance. `nli` is the same two clusterings with `nli_entail`. `no_speciation` is clustering method `none`.

`data/outputs/` is gitignored. Copy or generate those run directories before the commands below.

## Rebuild the reports

Factorial summary (fitness, coverage, run-level CSV):

```bash
python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
```

Species counts, high-fitness coverage, and the bin tables:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_species_best_fitness.py
python experiments/evostar_2027/rq2/scripts/rq2_species_dynamics.py
```

Combined quality–diversity figure. This also writes into `notes/rough_draft/figures/` when that folder exists:

```bash
python experiments/evostar_2027/rq2/scripts/compose_rq2_qd_mature_combined.py
```

Clustering table, from `rq2/reports/rq2_run_level.csv`. Writes `notes/rough_draft/rq2_clustering_table.tex` when the paper tree is present:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_clustering_table.py
```

Topic labels for every stored prompt. The first run embeds the prompts and caches them:

```bash
python experiments/evostar_2027/rq2/scripts/export_rq2_topic_vocabulary_xlsx.py
```

Response duration and prompt length versus species, for fitness above 0.6:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_text_diagnostics.py
```

Prompt perplexity on that same set. This loads a local Llama 3.1 8B GGUF and takes a while:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_prompt_perplexity.py
```

Checks and the anytime species-structure figure:

```bash
python experiments/evostar_2027/validation/mandatory_checks.py
python experiments/evostar_2027/validation/rq2_rq3_mechanism_exhibits.py
```

Reports land in `rq2/reports/`, `rq3/reports/`, and `validation/reports/`. Large `.npy` and `.pkl` files under `data/` stay on disk only.
