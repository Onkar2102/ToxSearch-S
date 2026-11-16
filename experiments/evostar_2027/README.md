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
| `data/outputs/evostar2027/` | The 50 finished runs (not in git) |

Run folders:

```text
data/outputs/evostar2027/lf_embedding/rep{0-9}_s{0-9}
data/outputs/evostar2027/inc_embedding/rep{0-9}_s{0-9}
data/outputs/evostar2027/lf_nli/rep{0-9}_s{0-9}
data/outputs/evostar2027/inc_nli/rep{0-9}_s{0-9}
data/outputs/evostar2027/no_speciation/rep{0-9}_s{0-9}
```

Reports land in `experiments/evostar_2027/rq2/reports/` unless a script says otherwise. Distance matrices under `experiments/evostar_2027/data/` stay on disk and are gitignored.

## Reports from the finished runs

RQ2 factorial table (fitness, diversity, species counts):

```bash
python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
```

Top-50 fitness table:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_topk50_primary.py
```

Species counts, fitness bins, and the combined quality–diversity figure:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_species_best_fitness.py
python experiments/evostar_2027/rq2/scripts/compose_rq2_qd_mature_combined.py
```

Clustering table used in the manuscript:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_clustering_table.py
```

Response time and same-topic length, for prompts with fitness above 0.6:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_text_diagnostics.py
```

Prompt perplexity on that same set. This loads a local Llama 3.1 8B GGUF and takes about 40 minutes:

```bash
python experiments/evostar_2027/rq2/scripts/rq2_prompt_perplexity.py
```

Topic labels workbook:

```bash
python experiments/evostar_2027/rq2/scripts/export_rq2_topic_vocabulary_xlsx.py
```

Anytime species-count figure:

```bash
python experiments/evostar_2027/validation/rq2_rq3_mechanism_exhibits.py
```

RQ1 neighborhood graph (needs `experiments/evostar_2027/data/pool/p0_full.pkl` and `data/mats/B3139/`):

```bash
python experiments/evostar_2027/rq1/scripts/rq1_neighborhood_graph.py
```

Most scripts accept `--help`. Other files under `rq1/scripts/`, `rq2/scripts/`, `rq3/scripts/`, and `validation/` are the same kind of one-shot report: point them at the run directory above and read the CSV or JSON they write.
