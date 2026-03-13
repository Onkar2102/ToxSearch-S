# Research Questions: Distribution and Search Performance (ToxSearch-S)

This document defines **three research questions** for evaluating how different worker-count setups affect **throughput**, **search effectiveness**, and **combined performance** in parallel ToxSearch-S. The questions align with the following goals:

- **Throughput / efficiency:** Show that certain setups yield higher **evaluated genomes per second** (and where time is spent: response generation, evaluation, variant creation, API wait).
- **Search effectiveness:** Compare how the search **performs**—whether it finds **more toxic prompts** and **faster** under different configurations.
- **Scalability and performance:** How much adding workers improves genomes-per-second and evolution speed; how species count and max toxicity evolve over time; and how the search *looks* (e.g. GDP plots) with different worker counts.

---

## Experiment Constraint: One Run per Configuration

- **Configurations:** Sequential execution, and parallel with **1, 2, 3, and 4 workers** (five setups in total).
- **Runs:** Exactly **one execution per configuration**. No repeated runs for the same worker count.
- **Implication:** Conclusions are based on **comparative analysis across the five single runs** (tables, plots, and qualitative comparison). We do not use statistical tests or confidence intervals across multiple runs; we report observed differences in throughput, quality, and diversity across configurations.

Use a **fixed total budget** for all runs (e.g. same `max_total_genomes` or same number of generations) and the **same** seed file, batch size (K), and speciation parameters so that comparisons are meaningful.

---

## RQ1: Throughput and Bottlenecks (Scalability)

**Question:** How much does adding more workers improve **evaluated genomes per second**, and where is time spent (response generation, evaluation, variant creation, API wait)?

This addresses:
- Whether different setups give **more throughput** (evaluated genomes per second).
- **API wait time:** How much time is spent waiting on the API (e.g. when keys or quota are limiting).

### What to measure (one run per configuration)

- **Throughput:** Mean (or median) **genomes per second** over generations (`generation_duration_seconds`, `variants_created` → `genomes_per_second` per generation; aggregate to a single run-level throughput).
- **Generation duration:** `generation_duration_seconds` per generation, compared across configurations.
- **Time breakdown:** Per generation (or run-level aggregates):  
  - `budget.total_response_time`, `budget.total_evaluation_time`, `budget.total_variant_creation_time`  
  - `budget.total_evaluation_api_wait_seconds` (time waiting on API).  
  Express each as fraction of generation time to identify bottlenecks (e.g. API-bound vs compute-bound).
- **Speciation time:** `speciation_duration_seconds` per generation as fraction of `generation_duration_seconds`.

### How to answer with single runs

- **Table:** For each of the five configurations: run-level throughput (evaluated genomes per second), total wall-clock time, and mean fraction of time in response gen / evaluation / variant creation / API wait / speciation.
- **Plot (optional):** Throughput (y) vs worker count (x), with sequential and 1–4 workers; annotate with API-wait fraction if notable.
- **Conclusion:** Describe how throughput scales with worker count and which component (API, response gen, evaluation, variant creation, speciation) dominates; note whether API wait is a bottleneck when keys are limited.

### Data sources

- **Throughput (genomes_per_second):** For each generation, **variants_integrated** (genomes actually added to the population in that generation) divided by **generation_duration_seconds** (wall-clock from the previous population update—or run start—until this update). So it is “genomes added per second” for that generation.
- `EvolutionTracker.json` → `run_metadata.num_workers` (and run mode: sequential vs parallel).
- `EvolutionTracker.json` → `generations[]`: `generation_duration_seconds`, `genomes_per_second`, `variants_integrated`.
- `generations[].budget`: `total_response_time`, `total_evaluation_time`, `total_variant_creation_time`, `total_evaluation_api_wait_seconds`.
- `generations[].speciation`: `speciation_duration_seconds`.

---

## RQ2: Search Effectiveness (Quality and Diversity)

**Question:** Does the search **find more toxic prompts** and a **more diverse set of failure modes** (species) when we change the worker-count setup?

This addresses:
- **Quality:** Whether the search finds **more** or **higher** toxicity under different configurations.
- **Diversity:** Whether it finds **more/less species** with more workers over time.
- **Evolution speed (quality):** Do more workers reach **higher max toxicity quicker**?

### What to measure (one run per configuration)

- **Quality over time:** `population_max_toxicity` (or `best_fitness`) per generation; time (or generation index) to first reach a given toxicity threshold (e.g. 0.80, 0.90) if applicable.
- **Diversity over time:** `speciation.species_count` and/or `active_species_count` per generation.
- **Final quality:** Final `population_max_toxicity` and, if available, top-K mean toxicity (e.g. top 10/50) at end of run.
- **Final diversity:** Final species count and, if available, semantic spread or cluster metrics from elites/reserves.

### How to answer with single runs

- **Table:** For each configuration: final max toxicity, final species count, and (optionally) generation at which a fixed toxicity threshold was first reached.
- **Plots:**  
  - Max toxicity vs generation (one curve per configuration).  
  - Species count vs generation (one curve per configuration).  
- **Conclusion:** Describe whether adding workers (or sequential vs parallel) is associated with higher final toxicity, more species, or faster rise in toxicity; note that with one run per setup, differences are observational, not statistically tested.

### Data sources

- `EvolutionTracker.json` → `generations[]`: `best_fitness`, `population_max_toxicity` (and tracker-level cumulative max).
- `generations[].speciation`: `species_count`, `active_species_count`.
- Optional: elites/reserves/archive for cluster or semantic diversity metrics.

---

## RQ3: Combined Performance and Search Behavior (Speed to Quality + Visual Comparison)

**Question:** How does **evolution speed** (time per generation, time to reach a quality level) and the **visual structure of the search** (e.g. GDP) differ across worker-count setups? In other words: do some setups help us **find toxic things faster** and does the search *look* different (e.g. exploration vs convergence) with different worker counts?

This addresses:
- **Evolution speed:** How much adding workers improves **evolution speed** (e.g. time per generation, or time to reach a target toxicity).
- **“Finding toxic things faster”:** Combination of throughput (RQ1) and quality-over-time (RQ2)—e.g. wall-clock time to reach a fixed toxicity threshold, or number of toxic genomes found per minute.
- **GDP plots:** How the search **looks** with different numbers of workers (Genetic Distance Projection or similar visualizations).

### What to measure (one run per configuration)

- **Evolution speed:** Mean `generation_duration_seconds` per run; or wall-clock time to complete the same number of generations (or same total genomes).
- **Speed-to-quality:** For a chosen toxicity threshold (e.g. 0.85): wall-clock time (or generation index) to first reach it; optionally, number of genomes evaluated until that point.
- **Search structure:** GDP (or other projection) plots per run; qualitative comparison of spread, clustering, and coverage across configurations.

### How to answer with single runs

- **Table:** Per configuration: mean generation duration, total wall-clock time, and (if defined) time/generation to first reach a toxicity threshold.
- **Plots:**  
  - Generation duration vs generation (by configuration).  
  - Time to reach threshold (or toxicity vs wall-clock time) by configuration.  
  - **GDP (or equivalent) figures:** One (or a few) representative generations per configuration to compare how the search looks (e.g. 1 vs 2 vs 4 workers, and sequential).
- **Conclusion:** Summarize whether more workers reduce time per generation and time-to-threshold, and how the search behavior (e.g. diversity, convergence) appears to differ in the GDP plots across setups.

### Data sources

- Same as RQ1 and RQ2 for durations and toxicity.
- GDP/projection outputs (e.g. from `utils/live_analysis` or experiment scripts) for the visual comparison.

---

## Summary Table

| RQ | Focus | Key metrics | Single-run use |
|----|--------|-------------|----------------|
| **RQ1** | Throughput and bottlenecks | Genomes/sec, generation duration, time breakdown (response, evaluation, variant creation, API wait, speciation) | Compare the five configurations in a table/plot; identify bottleneck components. |
| **RQ2** | Search effectiveness | Max toxicity over time, species count over time, final quality and diversity | Plot quality and diversity curves; table of final outcomes per configuration. |
| **RQ3** | Evolution speed and search behavior | Time per generation, time-to-threshold, GDP (or similar) visuals | Compare evolution speed and “speed to quality”; show GDP (or equivalent) for each setup. |

---

## Running the Experiments

- **Sequential:**  
  `PYTHONPATH=src python src/main.py --seed-file data/prompt.csv --output-dir data/outputs/rq_sequential ...`  
  (no `--parallel`; single process = “sequential” in this document.)

- **Parallel (1–4 workers):**  
  - 1 worker: `mpiexec -n 2 ...` (1 master + 1 worker).  
  - 2 workers: `mpiexec -n 3 ...`.  
  - 3 workers: `mpiexec -n 4 ...`.  
  - 4 workers: `mpiexec -n 5 ...`.  

Use the **same** `--seed-file`, `--batch-size` (K), `max_total_genomes` (or generations), and speciation parameters for all five runs. Use distinct `--output-dir` per configuration (e.g. `data/outputs/rq_sequential`, `data/outputs/rq_w1`, `data/outputs/rq_w2`, `data/outputs/rq_w3`, `data/outputs/rq_w4`) so each run has a stable path for analysis and GDP.

After the runs, point analysis scripts (or a small notebook) at the five `EvolutionTracker.json` paths and the GDP outputs to produce the tables and figures described above.

---

## Measuring TPS, RPS, and Latency (Throughput vs Speed)

Definitions (aligned with common serving metrics):

- **Tokens per Second (TPS):** Total **output tokens** generated per second (across all workers / the run).
- **Requests per Second (RPS):** Number of **requests successfully completed** per second (one “request” = one genome through response generation + evaluation).
- **Throughput vs latency:** **Throughput** = capacity (how much work per second). **Latency** = speed of one request (time to complete one genome end-to-end).

What we can measure today vs what needs small additions:

| Metric | Possible? | How (current data) | Optional addition |
|--------|------------|--------------------|--------------------|
| **RPS** | Yes | **Per generation:** `budget.llm_calls` or `budget.api_calls` / `generation_duration_seconds`. **Run-level:** `cumulative_budget.total_llm_calls` / `run_metadata.run_duration_seconds`. So “completed response generations per second” or “completed evaluations per second” with no code change. | None. |
| **Throughput (capacity)** | Yes | Already have **genomes_per_second** (integrated) and can derive RPS (completed requests) as above. | None. |
| **Latency (per request)** | Yes | Per genome we have `response_duration` and `evaluation_duration`. **End-to-end latency** for one genome = response_duration + evaluation_duration. Compute **mean / median / p95** per generation or run by iterating genomes in elites/reserves (and temp if needed). No change to tracker required; analysis script only. | Optionally store per-generation **mean_latency** or **median_latency** in the tracker for quick reporting. |
| **TPS** | Partially | We do **not** store output token count per genome. The response generator uses **word count** as a proxy (`len(generated_text.split())`) and keeps a running total in memory only. So we cannot compute TPS from saved data after the run. | **To measure TPS:** (1) When setting `genome["generated_output"]`, also set e.g. `genome["output_tokens"] = len(generated_output.split())` (or use `model_interface._estimate_token_count(generated_output)` for a heuristic). (2) In budget or generation stats, sum `output_tokens` for the generation and store **total_output_tokens**. Then **TPS** = total_output_tokens / generation_duration_seconds (per gen) or cumulative total_output_tokens / run_duration_seconds (run-level). |

Summary:

- **RPS and latency** can be measured with current data (RPS from budget + duration; latency from per-genome response_duration + evaluation_duration).
- **TPS** is possible only if we persist an output token count (or proxy) per genome and aggregate it per generation/run.

---

## Metrics readiness: what you have vs what to add

Below is a concise checklist so you can see whether you need to add or improve **per-generation**, **run-level (global)**, or **post-hoc** metrics for the three RQs.

### RQ1: Throughput and bottlenecks

| Need | Status | Where / note |
|------|--------|---------------|
| **Per-generation** | | |
| `generation_duration_seconds` | Yes | `generations[]` |
| `genomes_per_second` | Yes | `generations[]`: **variants_integrated** / generation_duration_seconds. Uses *created/integrated* count, not “evaluated” count. |
| `variants_created` | Yes | `generations[]` |
| `budget.total_response_time` | Yes | `generations[].budget` |
| `budget.total_evaluation_time` | Yes | `generations[].budget` |
| `budget.total_variant_creation_time` | Yes | `generations[].budget` |
| `budget.total_evaluation_api_wait_seconds` | Yes (parallel); partial (sequential) | Parallel: master sums from genomes. Sequential: from `calculate_budget_metrics` (elites+reserves+temp by generation). |
| `speciation_duration_seconds` | Yes | `generations[].speciation` |
| **Evaluated count per generation** | **Missing** | Throughput is currently “created/integrated per second”. For strict “evaluated genomes per second”, store **cumulative `total_evaluated`** (or **genomes_evaluated_this_generation**) in each generation entry; then compute evaluated throughput. Optional but improves RQ1 accuracy. |
| **Run-level (global)** | | |
| `run_metadata.num_workers` | Yes | Set for both sequential (1) and parallel. |
| `run_metadata.run_mode` | Yes | Set at end of run (sequential vs parallel). Cannot distinguish “sequential” (no MPI) from “parallel with 1 worker” (MPI, 2 ranks). Add e.g. `run_mode: "sequential" \| "parallel"` when writing the tracker. |
| `run_metadata.run_duration_seconds` | Yes | Total run wall-clock (seconds), written at end of run for RQ3 and run-level tables. |
| **Post-hoc** | | |
| Mean throughput, fraction of time in each phase | No code change | Compute from `generations[]` and `budget`: mean(genomes_per_second), and (response_time, evaluation_time, variant_creation_time, api_wait, speciation) as fraction of generation_duration_seconds. |

### RQ2: Search effectiveness

| Need | Status | Where / note |
|------|--------|---------------|
| **Per-generation** | | |
| `best_fitness` / `population_max_toxicity` | Yes | `generations[]` and tracker-level cumulative max |
| `speciation.species_count`, `active_species_count` | Yes | `generations[].speciation` |
| **Run-level** | | |
| Final max toxicity, final species count | No code change | Last generation entry (or tracker-level population_max_toxicity). |
| **Post-hoc** | | |
| Time/generation to first reach toxicity threshold (e.g. 0.85) | No code change | Scan `generations[]` for first gen where `population_max_toxicity` ≥ threshold. |
| Top-K mean toxicity (e.g. top 10/50) | No code change | Compute from elites/reserves (and archive if needed) after the run. |
| Cluster/semantic diversity | Optional | Use existing cluster/diversity helpers (e.g. from rq1/rq2 scripts) on saved elites/reserves. |

### RQ3: Combined performance and search behavior

| Need | Status | Where / note |
|------|--------|---------------|
| **Per-generation** | | |
| Same as RQ1 + RQ2 | Yes | As above. |
| **Run-level** | | |
| Total run wall-clock | Yes | `run_metadata.run_duration_seconds` so you can report “time to reach threshold” in wall-clock and mean generation duration. |
| **Post-hoc** | | |
| Wall-clock time to reach toxicity threshold | No code change | Cumulative sum of `generation_duration_seconds` up to the first generation where `population_max_toxicity` ≥ threshold. |
| Mean generation duration per run | No code change | Mean of `generation_duration_seconds` over generations. |
| **GDP / visual** | Yes | GDP projection and figures are produced by `live_analysis` / `generate_gdp_projection_plot` (sequential and parallel). Use the saved outputs per run for RQ3 visual comparison. |

---

### Summary: do you need to add or improve metrics?

- **You can run the three RQs with current metrics.** Per-generation metrics (duration, budget breakdown, speciation, quality, species count) and GDP are in place. Post-hoc you can compute throughput averages, time-to-threshold, and top-K toxicity from the tracker and population files.

- **Recommended additions (small, high value):**
  1. **Run-level:** **run_metadata.run_duration_seconds** and **run_metadata.run_mode** are written at end of run. (Previously: Write **total run wall-clock** (e.g. `run_metadata.run_duration_seconds`) at end of run (sequential and parallel).
  2. **Run-level:** Add **`run_metadata.run_mode`** (`"sequential"` vs `"parallel"`) so the five configurations (sequential, 1–4 workers) are uniquely identifiable.
  3. **Per-generation (optional):** Persist **cumulative `total_evaluated`** (or **genomes_evaluated_this_generation**) in each generation entry so “evaluated genomes per second” is exact; otherwise continue using `genomes_per_second` (created/integrated per second) and state that clearly in the report.

- **No need to add:** Post-hoc metrics (time-to-threshold, top-K mean, mean throughput, fraction of time per phase) can all be computed from existing tracker and population data in an analysis script or notebook.
