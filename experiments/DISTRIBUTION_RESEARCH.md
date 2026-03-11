# Distribution & Scaling Research (ToxSearch-S)

This document describes the **code support** for research questions related to **scalability** and **performance** when varying the number of workers (parallel runs).

---

## 1. Research Questions

### Scalability
- **Genomes per second**: How much does adding more workers improve throughput (genomes per second)?
- **API wait time**: How much time is spent waiting on the API (e.g. when keys or quota are insufficient)?

### Performance
- **Evolution speed**: How much does adding more workers improve evolution speed (e.g. time per generation)?
- **Species over time**: Does the search find more/less species with more workers over time?
- **Max toxicity over time**: Do more workers get to higher toxicity quicker?
- **GDP plots**: How does the search look (e.g. Genetic Distance Projection) with different numbers of workers?

---

## 2. Where the Data Lives

### Run-level (one value per run)
- **`EvolutionTracker.json` → `run_metadata`**  
  - **`num_workers`**: Number of workers (1 = single-process; 2+ = MPI). Use to group runs for scaling.
  - **`batch_size`**: K (genomes per batch) in parallel mode.
  - **`theta_sim`**, **`species_capacity`**: Speciation params (for parameter-sensitivity / RQ5).
  - **`num_perspective_keys`**: Number of API keys used (for API-scarcity / RQ4).

### Per-generation (one value per generation)
In **`EvolutionTracker.json` → `generations[]`** each entry has:

| Field | Description |
|-------|-------------|
| `generation_duration_seconds` | Wall-clock time for that generation (evolution + response + moderation + speciation). |
| `genomes_per_second` | `variants_created / generation_duration_seconds` (throughput for that gen). |
| `speciation` → `species_count`, `active_species_count` | Species counts after speciation. |
| `speciation` → `speciation_duration_seconds` | Time spent in speciation for that gen. |
| `best_fitness` / tracker-level `population_max_toxicity` | Max toxicity (fitness) so far. |

### Per-generation budget (API wait)
- **`generations[].budget`**:
  - `total_evaluation_time`: Total time in Perspective API calls.
  - **`total_evaluation_api_wait_seconds`**: Time spent in **sleep** due to rate-limit/retry (waiting on API).
- **`cumulative_budget.total_evaluation_api_wait_seconds`**: Cumulative API wait over the run.

### Per-genome (for deeper analysis)
Genomes in `elites.json` / `reserves.json` / `temp.json` (before clear) can have:
- `response_duration`, `variant_creation_duration`, `worker_cycle_duration`, `batch_variant_creation_duration`, `worker_rank`
- `evaluation_retries`, `evaluation_attempt_durations`, **`evaluation_api_wait_seconds`**

---

## 3. How to Answer the Research Questions

### Scalability: Genomes per second
- **Metric**: `genomes_per_second` per generation (or mean over generations).
- **Compare**: Group runs by `run_metadata.num_workers`; plot mean/median `genomes_per_second` vs `num_workers` (and optionally vs generation).

### Scalability: Time waiting on API
- **Metric**: `budget.total_evaluation_api_wait_seconds` per generation; or `cumulative_budget.total_evaluation_api_wait_seconds` per run.
- **Compare**: By `num_workers` and by generation; e.g. fraction of generation time spent waiting = `total_evaluation_api_wait_seconds / generation_duration_seconds`.

### Performance: Evolution speed
- **Metric**: `generation_duration_seconds` per generation (and total run time).
- **Compare**: By `num_workers` (same max_generations or same total genomes).

### Performance: Species over time
- **Metric**: `speciation.species_count` (or `active_species_count`) per generation.
- **Compare**: Plot species count vs generation, one curve per run, colored/faceted by `num_workers`.

### Performance: Max toxicity over time
- **Metric**: `population_max_toxicity` (tracker-level cumulative) or per-gen `best_fitness`.
- **Compare**: Plot max toxicity vs generation (or vs wall-clock time), by `num_workers`.

### GDP plots for different worker counts
- Run experiments with **different worker counts** (e.g. `mpiexec -n 2`, `-n 4`, `-n 8`), each writing to a different output dir.
- Each run’s **`EvolutionTracker.json`** has `run_metadata.num_workers`.
- Use the **GDP pipeline** (see `genetic-distance-projection-main/GDP_FOR_SPECIES_JOURNEY.md`) on each output dir to produce projections; then compare plots side-by-side or overlay (e.g. by labeling with `num_workers`).
- The script `experiments/plot_scaling_curves.py` can aggregate multiple EvolutionTracker files and plot generation_duration_seconds, genomes_per_second, species_count, and population_max_toxicity vs generation, faceted by `num_workers`.

---

## 4. Running Experiments

1. **Single process** (1 worker):  
   `python -m src.main ...`  
   → `run_metadata.num_workers` = 1 (set when initializing EvolutionTracker).

2. **Parallel (N workers)**:  
   `mpiexec -n <N+1> python -m src.main --parallel ...`  
   → Master + N workers; `run_metadata.num_workers` = N (set when creating or updating the tracker).

3. For scaling studies, run the **same** config (seed, max_generations, K, etc.) with **different** `-n` values and different output dirs, then use the tracker fields above and the plotting script to compare.

4. Use **`--output-dir data/outputs/<label>`** so each run has a stable path (e.g. `data/outputs/rq1_w4_run1`) for analysis scripts and GDP.

**Full experiment plan for the 5 research questions:** see **`experiments/RQ_EXPERIMENTS.md`** (experiment matrix, commands, and analysis steps).
