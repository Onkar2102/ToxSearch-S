# EvoStar 2027 analysis

Scripts that read finished runs and write the study reports. They do not start a new evolution.

Run every command from the repository root:

```bash
export PYTHONPATH=src:.
```

## Where things are

| Path | Contents |
|------|----------|
| `experiments/evostar_2027/rq1/` | Radius calibration, distance matrices, neighborhood reports |
| `experiments/evostar_2027/rq2/` | Factorial reports, species figures, text diagnostics |
| `experiments/evostar_2027/rq3/` | Coherence, coverage, and transfer-prep scripts |
| `experiments/evostar_2027/validation/` | Checks and the anytime structure figure |
| `experiments/evostar_2027/rq1/frozen_config.json` | Radii locked before the live runs |
| `data/outputs/evostar2027/` | The 50 finished runs (not in git). Loaded in step 1. |

Reports land in `experiments/evostar_2027/rq2/reports/` unless a script says otherwise. Distance matrices under `experiments/evostar_2027/data/` stay on disk and are gitignored.

## Steps

Run these in order. Later steps read files written by earlier ones.

### 1. Load the finished runs

The reports read genomes from disk. These folders have to be present before anything else:

```text
data/outputs/evostar2027/lf_embedding/rep{0-9}_s{0-9}
data/outputs/evostar2027/inc_embedding/rep{0-9}_s{0-9}
data/outputs/evostar2027/lf_nli/rep{0-9}_s{0-9}
data/outputs/evostar2027/inc_nli/rep{0-9}_s{0-9}
data/outputs/evostar2027/no_speciation/rep{0-9}_s{0-9}
```

Each folder is one finished run (`elites.json`, `archive.json`, `temp.json`, `run_config.json`). The analysis does not start a new search.

### 2. Run-level table

Fitness, diversity, and species counts for every seed. This writes `experiments/evostar_2027/rq2/reports/rq2_run_level.csv`.

```bash
python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
```

### 3. Clustering table

Reads `rq2_run_level.csv` from step 2 and writes the clustering table.

```bash
python experiments/evostar_2027/rq2/scripts/rq2_clustering_table.py
```

### 4. Top-50 fitness

```bash
python experiments/evostar_2027/rq2/scripts/rq2_topk50_primary.py
```

### 5. Species fitness and the combined figure

Species counts and fitness bins, then the quality–diversity figure that uses those reports.

```bash
python experiments/evostar_2027/rq2/scripts/rq2_species_best_fitness.py
python experiments/evostar_2027/rq2/scripts/compose_rq2_qd_mature_combined.py
```

### 6. Text of the high-fitness prompts

Response time and same-topic length, for prompts with fitness above 0.6. Perplexity uses a local Llama 3.1 8B GGUF and takes about 40 minutes.

```bash
python experiments/evostar_2027/rq2/scripts/rq2_text_diagnostics.py
python experiments/evostar_2027/rq2/scripts/rq2_prompt_perplexity.py
```

### 7. Topic labels

```bash
python experiments/evostar_2027/rq2/scripts/export_rq2_topic_vocabulary_xlsx.py
```

### 8. Anytime species-count figure

```bash
python experiments/evostar_2027/validation/rq2_rq3_mechanism_exhibits.py
```

### 9. RQ1 neighborhood graph

Needs the pool and distance matrices already under `experiments/evostar_2027/data/pool/p0_full.pkl` and `experiments/evostar_2027/data/mats/B3139/`.

```bash
python experiments/evostar_2027/rq1/scripts/rq1_neighborhood_graph.py
```

Most scripts accept `--help`. Other files under `rq1/scripts/`, `rq2/scripts/`, `rq3/scripts/`, and `validation/` are the same kind of one-shot report: they read the run folders from step 1 and write a CSV or JSON.
