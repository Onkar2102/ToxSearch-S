# Experiments for the 5 Research Questions

This document specifies **what changes (if any) are needed** and **which experiments to run** to answer each of the five A*-oriented research questions.

---

## Code / Data Readiness

The codebase already records the following; **no further code changes are required** to collect data for RQ1–RQ5:

| Data | Where | Used for |
|------|--------|----------|
| `run_metadata.num_workers` | EvolutionTracker.json | RQ1, RQ2, RQ3 – group runs by worker count |
| `run_metadata.batch_size`, `theta_sim`, `species_capacity`, `num_perspective_keys` | EvolutionTracker.json | RQ4 (keys), RQ5 (params) – facet by config |
| `generation_duration_seconds`, `genomes_per_second` | generations[] | RQ1, RQ3 – scaling and evolution speed |
| `speciation.species_count`, `active_species_count` | generations[].speciation | RQ2, RQ5 – diversity over time |
| `best_fitness`, `population_max_toxicity` | gen entry / tracker | RQ2, RQ3, RQ5 – quality over time |
| `budget.total_evaluation_api_wait_seconds` | generations[].budget | RQ1, RQ4 – time waiting on API |

**Optional but recommended:** use `--output-dir` so each run has a stable path (e.g. `data/outputs/rq1_w4`) for scripts and GDP.

---

## RQ1: Scaling Laws and Bottlenecks

**Question:** What are the scaling laws (linear vs sublinear) for throughput and discovery quality when parallelizing, and where are the bottlenecks (API, speciation, selection)?

### Experiments

- **Same config** for all runs: same seed file, same `--generations`, same `--batch-size` (K), same speciation params.
- **Vary workers:** 1 (sequential), 2, 4, 8 (e.g. `mpiexec -n 2`, `-n 5`, `-n 9` → 1, 4, 8 workers).
- **Repeats:** 3–5 runs per worker count for confidence intervals.

### Example commands

```bash
# Sequential (1 worker)
PYTHONPATH=src python src/main.py --generations 30 --seed-file data/prompt.csv \
  --output-dir data/outputs/rq1_w1_run1

# Parallel, 4 workers
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 \
  --generations 30 --seed-file data/prompt.csv \
  --output-dir data/outputs/rq1_w4_run1
```

### Analysis

- Plot **genomes_per_second** (mean over generations) vs **num_workers**; fit scaling curve (e.g. Amdahl).
- Plot **generation_duration_seconds** vs generation, by num_workers.
- Compare **speciation_duration_seconds** and **total_evaluation_api_wait_seconds** as fraction of generation time to attribute bottlenecks.
- Use `experiments/plot_scaling_curves.py` on the EvolutionTracker paths.

---

## RQ2: Parallelism–Diversity Trade-off

**Question:** Does increasing parallelism improve or harm behavioral diversity (species count, phenotypic spread) and discovery of distinct failure modes?

### Experiments

- **Same runs as RQ1** (same worker counts, same config, same output dirs).
- No extra runs needed if RQ1 is already done.

### Analysis

- Plot **species_count** (or **active_species_count**) vs generation, one curve per run, colored/faceted by **num_workers**.
- Compare **population_max_toxicity** vs generation by num_workers (do more workers reach higher toxicity faster or not?).
- Optionally: diversity metrics from elites (e.g. cluster count, semantic spread) using existing analysis scripts; compare by num_workers.

---

## RQ3: When Is Distributed Worth It?

**Question:** Under what conditions (budget, API key availability, population size) does distributed (MPI) outperform single-process, and at what cost?

### Experiments

- **Same runs as RQ1/RQ2:** 1 worker vs 2, 4, 8 workers with **same total budget** (e.g. same `--generations` or same total genomes).
- Optionally: repeat with different **--generations** (e.g. 10 vs 50) to see “when” MPI wins.

### Analysis

- For **same generations:** compare final **population_max_toxicity**, final **species_count**, and total wall-clock time by num_workers.
- Build a small table or phase diagram: (workers, keys, generations) → recommendation (use MPI when …).
- Compare cost: total time, API wait fraction, complexity (e.g. need for multiple keys).

---

## RQ4: Resource-Limited Evaluation (API Key Scarcity)

**Question:** When evaluation is the bottleneck (limited API keys), how do throughput, wait time, and discovery quality trade off, and what allocation strategies work?

### Experiments

- **Fix workers:** e.g. 4 workers (`mpiexec -n 5`).
- **Vary number of Perspective API keys:** 1 key, 2 keys, 4 keys (set `PERSPECTIVE_API_KEYS` or `PERSPECTIVE_API_KEY_0`, `_1`, …).
- **Same** seed, generations, batch-size, speciation params.
- **Repeats:** 2–3 per (workers, keys) cell.

### Example

```bash
# 4 workers, 1 key
export PERSPECTIVE_API_KEYS="key1"
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 \
  --generations 20 --output-dir data/outputs/rq4_4w_1key_run1

# 4 workers, 4 keys
export PERSPECTIVE_API_KEYS="key1,key2,key3,key4"
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 \
  --generations 20 --output-dir data/outputs/rq4_4w_4keys_run1
```

### Analysis

- Plot **genomes_per_second** and **total_evaluation_api_wait_seconds** (or fraction of generation time) vs **num_perspective_keys** (from `run_metadata`).
- Compare **population_max_toxicity** and **species_count** at same generation across key counts.
- Conclude: recommended key count or allocation strategy (e.g. “keys ≥ workers” or “batch when keys < workers”).

---

## RQ5: Parameter Robustness and Design Principles

**Question:** How robust are discovery quality and diversity to speciation/search parameters, and what design principles can we derive for tuning?

### Experiments

- **Fix workers** (e.g. 4) and **same** seed and generations.
- **Vary one parameter at a time:**
  - **theta_sim:** e.g. 0.1, 0.2, 0.3
  - **species_capacity:** e.g. 50, 100, 200
  - **batch_size (K):** e.g. 50, 100, 200
- **Repeats:** 2–3 per setting.

### Example

```bash
# theta_sim = 0.1
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 \
  --generations 25 --theta-sim 0.1 --species-capacity 100 \
  --output-dir data/outputs/rq5_theta0.1_run1

# species_capacity = 200
PYTHONPATH=src mpiexec -n 5 python src/main.py --parallel --batch-size 100 \
  --generations 25 --theta-sim 0.2 --species-capacity 200 \
  --output-dir data/outputs/rq5_cap200_run1
```

### Analysis

- Plot **species_count**, **population_max_toxicity**, and (if available) diversity metrics vs generation, faceted by **theta_sim**, **species_capacity**, or **batch_size** (from `run_metadata`).
- Summarize robustness: e.g. “quality stable for theta_sim in [0.15, 0.25]”; “diversity increases with species_capacity up to 100.”
- Write short design principles (e.g. “use K ≥ 100 when workers ≥ 4”).

---

## Summary: Minimal Experiment Matrix

| RQ | Vary | Fix | Min runs (single repeat) | Repeats for stats |
|----|------|-----|--------------------------|-------------------|
| RQ1 | num_workers ∈ {1, 2, 4, 8} | seed, gens, K, params | 4 | 3–5 per worker count |
| RQ2 | (use RQ1 runs) | — | 0 | — |
| RQ3 | (use RQ1 runs) | — | 0 | — |
| RQ4 | num_perspective_keys ∈ {1, 2, 4} | workers=4, seed, gens, K | 3 | 2–3 per key count |
| RQ5 | theta_sim, species_capacity, or K (one at a time) | workers=4, seed, gens | 3–9 (3 values × 1–3 params) | 2–3 per setting |

**Total new runs (if you do RQ1 + RQ4 + RQ5 with one repeat each):**  
4 (RQ1) + 3 (RQ4) + e.g. 9 (RQ5, 3 params × 3 values) = **16 runs**. With repeats, multiply accordingly.

---

## Running the Experiments

1. **By hand:** Use the example commands above; set `--output-dir` to a distinct path per run (e.g. `data/outputs/rq1_w4_run1`).
2. **Script:** Use or extend `run_experiments_local.sh`: add entries that set `OUTPUT_DIR` or pass `--output-dir` and vary workers, keys, or params as in the table.
3. **Analysis:** After runs, point `experiments/plot_scaling_curves.py` at the EvolutionTracker.json paths, or write a small notebook that loads multiple trackers, groups by `run_metadata` fields, and plots the metrics above.

No additional code changes are required to **answer** the five questions; the optional `--output-dir` and `run_metadata` (batch_size, theta_sim, species_capacity, num_perspective_keys) make analysis and scripting easier.
