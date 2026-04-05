# ToxSearch-S — Method and System Architecture

This document describes the design and implementation of Speciated ToxSearch: a **quality-diversity evolutionary algorithm** for automated red-teaming of large language models (LLMs). The method combines a steady-state (μ + λ) evolution with **semantic speciation** (leader-follower clustering in embedding space) to maintain diverse prompt niches while optimizing for a toxicity-based fitness. The following sections specify the algorithm, population structure, speciation phases, and parallel runtime so that experiments can be reproduced and extended.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Reproducibility and Experimental Setup](#2-reproducibility-and-experimental-setup)
3. [Module Layout](#3-module-layout)
4. [Core Components](#4-core-components)
5. [Evolution Flow](#5-evolution-flow)
6. [Speciation Framework](#6-speciation-framework)
7. [Parent Selection System](#7-parent-selection-system)
8. [Key Metrics](#8-key-metrics)
9. [Configuration Parameters](#9-configuration-parameters)
10. [MPI Parallel Runtime](#10-mpi-parallel-runtime)

---

## 1. System Overview

The system implements a single-objective evolutionary search over prompt space, with fitness defined as the toxicity of the LLM’s response to that prompt (as scored by an external moderation API). To avoid premature convergence and to encourage exploration of distinct failure modes, the population is partitioned into *species* via semantic (embedding-based) clustering; selection and variation are applied within and across these niches. The following subsections specify the algorithm type, fitness function, population structure, and distance measure used for speciation.

### 1.1 Algorithm Type

**Steady-State (μ + λ) Evolutionary Algorithm** with dynamic semantic speciation.

- **μ (mu)**: Parent population size (elites + reserves)
- **λ (lambda)**: Offspring generated per generation
- **Steady-state**: Population continuously updated, not replaced in generations

### 1.2 Fitness Function

```
f(x) = toxicity_score(LLM(x)) ∈ [0, 1]
```

- `x`: Prompt (genome)
- `LLM(x)`: Model's response to the prompt
- `toxicity_score`: Moderation API score (Google Perspective API)

### 1.3 Population Structure

```
Population P = E ∪ R

E (Elites):   Active species members (species_id > 0)
R (Reserves): Cluster 0 outliers (species_id = 0)
Archive:      Capacity-overflow genomes (species_id = -1)
```

### 1.4 Ensemble Distance

Species assignment uses ensemble distance combining genotype and phenotype:

```
d_ensemble = 0.7 × d_genotype + 0.3 × d_phenotype
```

- **d_genotype**: Cosine distance between prompt embeddings
- **d_phenotype**: Normalized difference in toxicity scores

---

## 2. Reproducibility and Experimental Setup

To reproduce or compare results, the following should be fixed or recorded.

**Software environment**
- Python version (e.g. 3.8+); dependencies as in `requirements.txt` with versions where applicable.
- For parallel runs: MPI implementation and version (e.g. Open MPI, MPICH) and `mpi4py` built against that MPI.
- Embedding model: sentence-transformers identifier (e.g. `all-MiniLM-L6-v2`) and embedding dimension (e.g. 384).
- LLM: exact GGUF model path and quantization (e.g. Q4_K_M); response and prompt generators may use the same or different checkpoints.

**Randomness**
- A fixed random seed is not currently set at process start; for full reproducibility, the codebase or launcher should set `random.seed` and `numpy.random.seed` (and any other RNGs) to a documented value before the first generation.

**Input data**
- Seed prompts: CSV with a `questions` column; each row is one initial prompt. Document the file path and row count.
- Moderation: Google Perspective API (or equivalent); document the metric name (e.g. `toxicity`) and that multiple keys are used only to distribute rate limits, not to change the metric.

**Output artifacts**
- Output directory contains: `EvolutionTracker.json` (per-generation and cumulative metrics), `elites.json`, `reserves.json`, `archive.json`, `speciation_state.json`, `genome_tracker.json`, `events_tracker.json`, and optionally `parents.json`, `top_10.json`. Figures under `figures/` are derived from the tracker and population files.
- For parallel runs: one log file per MPI rank (`*_master.log`, `*_worker1.log`, …). See README for interpretation of worker log messages.

**Run configuration**
- Record all command-line arguments (or equivalent config): `--generations`, `--batch-size` (K), **`--max-total-genomes` (required for sequential and parallel; primary termination)**, `--theta-sim`, `--theta-merge`, `--species-capacity`, `--cluster0-max-capacity`, `--seed-file`, `--seed`, model paths (`--rg`, `--pg`), and any overrides. For multi-node or multi-GPU runs, record the number of MPI ranks and how GPUs are assigned (e.g. one GPU per worker via the scheduler). Config and seed paths are resolved from the project root so runs are independent of working directory. See README **Pre-execution checklist** before final batch runs.

---

## 3. Module Layout

```
src/
├── main.py                    # Entry point and orchestration
│
├── ea/                        # Evolutionary Algorithm
│   ├── evolution_engine.py    # ID generation, variant creation
│   ├── run_evolution.py       # Evolution loop orchestration
│   ├── parent_selector.py     # Adaptive parent selection
│   ├── variation_operators.py # Operator base classes
│   ├── paraphrasing.py        # LLM-based paraphrasing
│   ├── concept_addition.py    # Concept addition operator
│   ├── back_translation.py    # Back translation operator
│   └── operator_statistics.py # Effectiveness tracking
│
├── gne/                       # Generate-Evaluate
│   ├── response_generator.py  # LLM response generation
│   ├── evaluator.py           # Moderation API integration
│   └── model_interface.py     # Model loading/management
│
├── parallel/
│   └── master_worker.py       # MPI master-worker runtime (rank 0 master, ranks 1..N workers)
│
├── speciation/                # Speciation Framework
│   ├── run_speciation.py      # 8-phase speciation process
│   ├── config.py              # SpeciationConfig dataclass
│   ├── leader_follower.py     # Leader-follower clustering
│   ├── species.py             # Species class and management
│   ├── reserves.py            # Cluster 0 management
│   ├── merging.py             # Species merging logic
│   ├── extinction.py          # Freezing and dissolution
│   ├── embeddings.py          # Embedding computation
│   ├── distance.py            # Distance calculations
│   ├── metrics.py             # Diversity metrics
│   ├── genome_tracker.py      # Authoritative species_id tracking
│   └── labeling.py            # c-TF-IDF label generation
│
└── utils/                     # Utilities
    ├── population_io.py       # File I/O and statistics
    ├── device_utils.py        # Device detection (CUDA/MPS/CPU) and MPI device assignment
    ├── refusal_detector.py    # Refusal detection
    ├── refusal_penalty.py     # Penalty application
    ├── cluster_quality.py     # Cluster quality metrics
    └── operator_effectiveness.py # Operator metrics
```

---

## 4. Core Components

### 4.1 Evolution Engine

**Purpose**: Variant creation and ID management

- `next_id()`: Generates globally unique genome IDs in sequential mode
- `create_child()`: Creates variants using operators with parent tracking

### 4.2 Parent Selector

**Purpose**: Adaptive parent selection based on population fitness trends

**Categories**:
- **Category 1**: Active species ∪ reserves (primary selection pool)
- **Category 2**: Frozen species (fallback when Category 1 empty)

**Selection Modes**:
- **DEFAULT**: 2 parents from random species
- **EXPLOIT**: 3 parents from top species (local search)
- **EXPLORE**: 1 parent each from 3 different species (diversity)

### 4.3 Speciation Engine

**Purpose**: 8-phase speciation process for each generation

1. **Existing Species Processing**: Assign variants to species or cluster 0 (radius enforcement)
2. **Cluster 0 Speciation**: Form new species from cohesive clusters
3. **Merging**: Combine similar species (θ_merge); radius enforcement after merging
4. **Capacity Enforcement**: Enforce species capacity only (radius in Phases 1 and 3)
5. **Freeze & Incubator**: Track stagnation, freeze/dissolve species
6. **Cluster 0 Capacity Enforcement**: Archive excess reserves
7. **Final Redistribution**: Update species_id from tracker for elites, reserves, temp; redistribute to files; archive append-only
8. **Metrics & Stats**: Calculate diversity and cluster quality

### 4.4 Genome Tracker

**Purpose**: Authoritative source of truth for species_id assignments

- Updated at every speciation event
- Used in Phase 7 to synchronize file-based species_id values
- Enables efficient deferred file updates
- Archive is a final destination: genomes in archive are not moved back to elites or reserves

---

## 5. Evolution Flow

### 5.1 Generation 0 (Initialization)

```
1. System Setup
   └── Initialize output directory, load models, load seed prompts

2. Initial Population
   └── Generate responses for seed prompts → temp.json

3. Evaluation
   └── Evaluate via moderation API, apply refusal penalty

4. Pre-Speciation Metrics
   └── Calculate avg_fitness = mean(temp.json)

5. Speciation
   └── Leader-follower clustering → elites.json, reserves.json

6. Statistics & Tracking
   └── Update EvolutionTracker.json, speciation_state.json
```

### 5.2 Generation N (Evolution Loop)

```
PHASE 1: Variant Generation
├── Load population (elites + reserves)
├── Adaptive parent selection (DEFAULT/EXPLOIT/EXPLORE)
└── Apply variation operators → temp.json

PHASE 2: Response Generation
└── Generate LLM responses for all variants

PHASE 3: Evaluation
├── Moderation API evaluation
├── Refusal penalty application
└── Pre-speciation metrics: avg_fitness = mean(elites + reserves + temp)

PHASE 4: Speciation (8 phases)
└── See Section 5

PHASE 5: Post-Processing
├── Operator effectiveness metrics
├── Update EvolutionTracker
├── Adaptive selection logic update
└── Generation statistics
```

### 5.3 Termination Conditions

- **Parallel:** Primary termination is by `--max-total-genomes` (total genomes in elites + reserves + archive). When the cap is reached, master signals shutdown and drains buffered genomes.
- Maximum generations reached (sequential or as a secondary limit)
- Threshold achieved (population_max_toxicity ≥ threshold)
- All species frozen and reserves empty
- Runtime error or user interruption

---

## 6. Speciation Framework

### 6.1 Phase 1: Existing Species Processing

Assign new variants to existing species or cluster 0 using leader-follower clustering.

```
For each variant v:
  For each species s with leader L:
    if d_ensemble(v, L) < θ_sim:
      assign v to s
      break
  else:
    assign v to cluster 0 (reserves)
```

### 6.2 Phase 2: Cluster 0 Speciation

Form new species when cluster 0 contains cohesive clusters.

```
When |cluster_0| ≥ cluster0_min_cluster_size:
  Select leader L = argmax(fitness) from cluster 0
  Create new species S with leader L
  For each genome g in cluster 0:
    if d_ensemble(g, L) < θ_sim:
      move g to S
```

### 6.3 Phase 3: Merging

Combine similar species to reduce redundancy.

```
For all species pairs (S_i, S_j):
  if d_ensemble(L_i, L_j) < θ_merge:
    Create merged species S_new
    S_new.members = S_i.members ∪ S_j.members
    S_new.leader = argmax(fitness) from S_new.members
    Mark S_i, S_j as extinct
```

### 6.4 Phase 4: Capacity Enforcement

Radius enforcement is done in Phase 1 and Phase 3; Phase 4 only enforces species capacity.

```
For each species S (species_id > 0):
  if |S| > species_capacity:
    Sort members by fitness (descending)
    Archive excess members (lowest fitness)
```

### 6.5 Phase 5: Freeze & Incubator

```
For each species S:
  Update stagnation counter:
    if max_fitness increased: stagnation = 0
    else if S was selected as parent: stagnation += 1

  if stagnation ≥ species_stagnation:
    S.state = "frozen"

  if |S| < min_island_size:
    Move S.members to cluster 0
    S.state = "incubator"
```

### 6.6 Phase 6: Cluster 0 Capacity Enforcement

```
if |cluster_0| > cluster0_max_capacity:
  Sort by fitness (descending)
  Archive excess (lowest fitness)
```

### 6.7 Phase 7: Final Redistribution

Update species_id from genome_tracker for genomes in elites, reserves, and temp only. Redistribute those to the correct files by species_id. Archive is not read for redistribution; genomes in archive are not updated or moved back.

```
For each genome g in (elites ∪ reserves ∪ temp):
  g.species_id = genome_tracker[g.id].species_id

Redistribute:
  species_id > 0  → elites.json
  species_id = 0  → reserves.json
  species_id = -1 → archive.json
```

Archive.json is append-only; genomes already in archive stay there.

### 6.8 Phase 8: Metrics & Stats

Calculate and record:
- Inter-species diversity (mean distance between leaders)
- Intra-species diversity (mean distance within species)
- Cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

---

## 7. Parent Selection System

### 7.1 Category System

| Category | Contents | Usage |
|----------|----------|-------|
| Category 1 | Active species ∪ reserves | Primary selection pool |
| Category 2 | Frozen species | Fallback only |

### 7.2 Selection Modes

| Mode | Trigger | Parents | Strategy |
|------|---------|---------|----------|
| DEFAULT | Normal | 2 from random species | Balanced exploration |
| EXPLOIT | slope_of_avg_fitness ≤ 0 | 3 from top species | Local search |
| EXPLORE | generations_since_improvement ≥ stagnation_limit | 1 from each of 3 species | Diversity |

### 7.3 Mode Determination

```python
if generations_since_improvement >= stagnation_limit:
    mode = EXPLORE
elif slope_of_avg_fitness <= 0:
    mode = EXPLOIT
else:
    mode = DEFAULT
```

---

## 8. Key Metrics

### 8.1 Fitness Metrics

| Metric | Definition | Timing |
|--------|------------|--------|
| `avg_fitness` | mean(elites + reserves + temp) | Before speciation |
| `avg_fitness_generation` | mean(elites + reserves) | After distribution |
| `max_score_variants` | max(temp.json fitness) | Before speciation |
| `population_max_toxicity` | Cumulative max across all generations | After each generation |

### 8.2 Adaptive Selection Metrics

| Metric | Definition |
|--------|------------|
| `generations_since_improvement` | Consecutive generations without population_max_toxicity increase |
| `slope_of_avg_fitness` | Linear regression slope over recent avg_fitness_history |
| `avg_fitness_history` | Sliding window of recent avg_fitness values |

### 8.3 Diversity Metrics

| Metric | Description |
|--------|-------------|
| `inter_species_diversity` | Mean pairwise distance between species leaders |
| `intra_species_diversity` | Mean pairwise distance within species |
| `separation_ratio` | inter / intra (higher = better separation) |

### 8.4 Cluster Quality Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Silhouette Score | [-1, 1] | Higher = better separation |
| Davies-Bouldin Index | ≥ 0 | Lower = better |
| Calinski-Harabasz Index | > 0 | Higher = better |

### 8.5 Run artifacts and tracker fields

- **`EvolutionTracker.json`** is updated each generation from `calculate_generation_statistics` and `update_evolution_tracker_with_statistics` in [`src/utils/population_io.py`](src/utils/population_io.py). Budget breakdowns come from `calculate_budget_metrics` in the same module (LLM variant-creation calls match operator name strings in `OPERATORS_USING_LLM`, plus `creation_info.operator` / `operator` on genomes).
- **Per-generation evaluation:** `evaluated_this_generation` and `discarded_this_generation` are the portable fields for throughput analysis. Run-level **cumulative** totals use `cumulative_variants_evaluated` and `cumulative_variants_discarded` on the tracker root. Older parallel rows may lack `evaluated_this_generation`; scripts fall back to `budget.api_calls` / `budget.llm_calls` with caution.
- **`generation_duration_seconds` + `generation_duration_scope`:** Both modes set scope **`through_evolution_tracker_statistics_write`** (trailing edge aligned at the main tracker row write). Start anchors still differ (sequential: generation loop / gen0 bootstrap; parallel: master clock after previous tracker update through merge+speciation+tracker prep). See [README.md](README.md#metrics-and-outputs-evolutiontracker-workers-c1c3).

---

## 9. Configuration Parameters

### 9.1 Speciation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta_sim` | 0.2 | Similarity threshold for species assignment |
| `theta_merge` | 0.1 | Merge threshold (must be ≤ theta_sim) |
| `species_capacity` | 100 | Maximum individuals per species |
| `cluster0_max_capacity` | 1000 | Maximum individuals in cluster 0 |
| `cluster0_min_cluster_size` | 2 | Minimum size for cluster 0 speciation |
| `min_island_size` | 2 | Minimum species size before dissolution |
| `species_stagnation` | 20 | Generations without improvement before freezing |

### 9.2 Embedding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `embedding_dim` | 384 | Embedding dimensionality |
| `embedding_batch_size` | 64 | Batch size for computation |

### 9.3 Ensemble Distance Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_genotype` | 0.7 | Weight for embedding distance |
| `w_phenotype` | 0.3 | Weight for toxicity difference |

**Constraint**: `w_genotype + w_phenotype = 1.0`

---

## 10. MPI Parallel Runtime

### 10.1 Process Roles

- **Master (rank 0)**: Owns shared persistent state (`temp.json`, `elites.json`, `reserves.json`, `archive.json`, `EvolutionTracker.json`), performs dedup + merge + speciation, parent selection, tracker/statistics updates.
- **Workers (rank 1..N)**: Request work, generate/evaluate variants, send evaluated genomes back. Workers do not mutate shared population files.

### 10.2 Data flow: per-variant streaming

Workers **do not** wait for all variants in a batch to be evaluated before sending. For each variant, the worker:

1. Receives **parents** and **top_10** from the master (message `PARENTS`).
2. Calls **generate_single_variant** (one operator application; typically returns 1 variant per cycle with `max_variants=1`).
3. For **each** variant in the returned list:
   - Generates LLM response (**process_single_genome**),
   - Evaluates with the moderation API (**evaluate_single_genome**),
   - Applies refusal penalty,
   - **Immediately** sends that single genome to the master as **EVALUATED_VARIANT**.
4. Then requests work again (sends **PARENTS_REQUEST** and blocks on **recv**).

The master runs a **single dispatch loop**: it blocks on **recv** from any worker. When it receives **EVALUATED_VARIANT**, it appends that genome to a **per-worker buffer** (`buffers[source]`) and increments `total_evaluated`. **After generation 0**, when the **total** number of genomes across all buffers reaches **merge K**, the master runs merge (drain up to K from buffers, round-robin), dedup, writes `temp.json`, runs speciation, updates the tracker, and increments the generation. So evaluated genomes from different workers are **interleaved** in the order they arrive; merge drains round-robin from worker buffers so no worker is starved.

**Merge K (sequential parity):** If `--batch-size` is **omitted** and `--operators all`, K is **`24 × --max-variants`** when `EvolutionTracker.selection_mode` is default (2 parents in sequential) and **`39 × --max-variants`** when selection mode is explore or exploit (3 parents), matching the sequential `EvolutionEngine` attempt counts. If `--batch-size` is set, it overrides. For `--operators cm` or `ie`, omitting `--batch-size` uses **100** (legacy default). Each merge logs `K_used`; `master_metrics.json` includes `merge_k_history`.

**Generation 0 (bootstrap):** the master sends each worker a **GEN0_BATCH** (a slice of seed prompt indices). The worker loads those prompts, and for **each** prompt generates a response, evaluates it, and **immediately** sends one **EVALUATED_VARIANT** back. The master buffers them and **does not** trigger speciation on `K` alone. It waits until **all** GEN0 batches have been dispatched (`gen0_assignments` empty) **and** the number of returned Gen0 variants reaches the expected seed count; then it runs **one** merge/speciation that drains the **entire** buffered bootstrap set (cap = full buffer, not `K`). **Generation 1+** then follow the usual `K`-batched rule.

**After shutdown** (e.g. max generations), the master may still have genomes left in buffers. It runs a **drain phase**: merge+speciation on the remaining buffered genomes, then final statistics.

### 10.3 Message Protocol

- `PARENTS_REQUEST (10)`: Worker -> Master asks for work (`request_id` included).
- `PARENTS (11)`: Master -> Worker sends parents + top_10 and key index, or `None` for shutdown.
- `EVALUATED_VARIANT (12)`: Worker -> Master sends one evaluated genome immediately after evaluation (`request_id`, `local_variant_id`).
- `GEN0_BATCH (13)`: Master -> Worker sends seed prompt index range (`prompt_start`, `prompt_end`) for generation 0 bootstrap.
- `STOP (14)`: Master -> Worker signals stop (workers may also receive `PARENTS` with payload `None` for shutdown).
- `WORKER_READY (20)`: Worker -> Master signals successful init (models loaded); master waits for all workers before starting the dispatch loop.
- `WORKER_INIT_FAILED (21)`: Worker -> Master signals init failure (e.g. model load error); payload `{rank, error}`; master aborts.

**Startup:** Master broadcasts config; workers load `.env` from project root (for `PERSPECTIVE_API_KEY`), then init RG, PG, and evaluator. Each worker sends `WORKER_READY` on success or `WORKER_INIT_FAILED` on exception. Master waits for all `WORKER_READY` with a timeout (e.g. 900 s); on timeout or any `WORKER_INIT_FAILED`, the run aborts. Parallel mode requires at least one Perspective API key (abort with a clear error if missing).

### 10.4 Generation Semantics in Parallel

- **Merge batch K** after generation 0: `--batch-size` if set; else sequential parity **24 / 39** (times `--max-variants`) for `--operators all` from tracker `selection_mode`; else **100** for `cm` / `ie`.
- Master keeps evaluated genomes in **per-worker in-memory buffers**.
- **Generation 0:** one speciation after all seed evaluations are buffered; merge drains the full buffer (all bootstrap genomes, round-robin), subject to dedup/error skips inside `_merge_and_speciate`.
- **Generation 1+:** when buffered genomes reach `K`, master drains up to `K`, deduplicates by exact prompt match, writes `temp.json`, and runs speciation.
- `generation_id` increments per speciation run.

### 10.5 `temp.json` in Parallel

- `temp.json` is **transient** in parallel mode.
- It is populated during merge/speciation and then cleared by speciation redistribution.
- Seeing `temp.json` as `[]` between speciation runs is expected.

### 10.6 Tracker, Metrics, and Figures

- After each parallel speciation, master computes generation statistics from population files and updates `EvolutionTracker.json`.
- Tracker stores per-generation population/speciation metrics (fitness, counts, diversity/cluster quality when available).
- Operator statistics are aggregated from accepted evaluated genomes (`operator`, `variant_type`).
- Master maintains **runtime** cumulative counters (`total_evaluated`, `total_integrated`, `total_discarded`) for logging; **per-generation** evaluation counts in the tracker use `evaluated_this_generation` / `discarded_this_generation` (written via `_update_tracker`). Run-level cumulative totals on the tracker use `cumulative_variants_evaluated` / `cumulative_variants_discarded`.
- `master_metrics.json` records `merge_k_history` (per merge: `generation_number`, `merge_k_used`, `buffered_before_merge`, optional `drain_phase`).
- Live analysis/visualizations and final statistics generation run in parallel mode as best-effort post-processing steps.
- If a full tracker update fails, the master sets `status="degraded"` and `last_tracker_error`, then raises **(fail-fast)** so the run does not silently continue with sparse generation rows.

### 10.7 Sequential vs parallel tracker parity (summary)

| Aspect | Sequential (`src/main.py`) | Parallel (`src/parallel/master_worker.py`) |
|--------|----------------------------|--------------------------------------------|
| Generation update pipeline | Evolution → moderation → `run_speciation` → `calculate_generation_statistics` → `update_evolution_tracker_with_statistics` → adaptive selection | Buffered variants → merge/dedup → `run_speciation` → same statistics + tracker update → adaptive selection |
| `evaluated_this_generation` | From `api_calls` / explicit field for that generation | Delta of evaluated variants since previous merge |
| `generation_duration_seconds` | Through main `update_evolution_tracker_with_statistics` (after evolution→speciation→pre-tracker viz/aux passes); excludes adaptive selection and any post-tracker viz | Through same tracker write inside `_update_tracker` after merge+speciation (includes master-side stats prep); excludes adaptive selection and `_run_live_analysis` after the update |
| Operator schedule | Full `EvolutionEngine` grid / schedule | Workers use `generate_single_variant` per request (not identical ordering to sequential) |

Single sources of truth for population and budget stats: `calculate_generation_statistics` and `calculate_budget_metrics` in [`src/utils/population_io.py`](src/utils/population_io.py).

### 10.8 Logging

- Parallel runtime writes per-rank logs (`*_master.log`, `*_workerN.log`) to avoid mixed concurrent output in one file. For a detailed interpretation of worker log messages (e.g. request cycles, Gen0 batches, evolution cycles, API key assignment), see the "Worker log messages" section in [README.md](README.md#worker-log-messages-what-they-mean).

**Logging and streaming:** The implementation logs the streaming behaviour correctly. On the **master**, every **EVALUATED_VARIANT** received is logged at INFO (worker, request_id, local_variant_id, status, prompt snippet); buffer state is logged periodically (every 5 variants or when total buffered ≥ K) and at merge/speciation (during Gen0 bootstrap, buffered may stay &lt; K until all seeds complete). On the **worker**, each **EVALUATED_VARIANT** send is logged at DEBUG to avoid log flood; INFO logs cover PARENTS received, number of variants generated, and evolution-cycle summary (variants sent, ok/errors, time, total_sent). So the flow (request work → receive parents → generate variants → evaluate and send each variant immediately) is observable in the logs.

### 10.9 Device and GPU usage (HPC)

- **No rank-based GPU selection in code.** Device choice is centralized in `utils/device_utils.py` (DeviceManager): it picks MPS (macOS), then CUDA if available, then CPU. The LLM (llama.cpp) and embeddings (sentence-transformers) use that device; CUDA is always used as “device 0” from the process’s view.
- **On HPC clusters**, assign one GPU per worker via the job scheduler (e.g. SLURM `--gpus-per-task=1`). The scheduler sets `CUDA_VISIBLE_DEVICES` (or equivalent) so each process sees exactly one GPU; the application then correctly uses that GPU without any code changes. See README “Running on HPC Clusters” for an example job script.

---

## References

- [ARCHITECTURE_DIAGRAM.md](ARCHITECTURE_DIAGRAM.md) — Architecture diagrams (system, components, sequential flow, master-worker MPI)
- [README.md](README.md) — Installation, setup, hyperparameters, reproducibility, and worker log interpretation
- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Output file field definitions
