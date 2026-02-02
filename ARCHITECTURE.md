# Speciated ToxSearch — System Architecture

This document describes the architecture of the Speciated ToxSearch framework, a quality-diversity evolutionary system for LLM red-teaming with leader-follower semantic speciation.

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Module Layout](#2-module-layout)
3. [Core Components](#3-core-components)
4. [Evolution Flow](#4-evolution-flow)
5. [Speciation Framework](#5-speciation-framework)
6. [Parent Selection System](#6-parent-selection-system)
7. [Key Metrics](#7-key-metrics)
8. [Configuration Parameters](#8-configuration-parameters)

---

## 1. System Overview

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

## 2. Module Layout

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
    ├── refusal_detector.py    # Refusal detection
    ├── refusal_penalty.py     # Penalty application
    ├── cluster_quality.py     # Cluster quality metrics
    └── operator_effectiveness.py # Operator metrics
```

---

## 3. Core Components

### 3.1 Evolution Engine

**Purpose**: Variant creation and ID management

- `next_id()`: Generates globally unique genome IDs
- `create_child()`: Creates variants using operators with parent tracking

### 3.2 Parent Selector

**Purpose**: Adaptive parent selection based on population fitness trends

**Categories**:
- **Category 1**: Active species ∪ reserves (primary selection pool)
- **Category 2**: Frozen species (fallback when Category 1 empty)

**Selection Modes**:
- **DEFAULT**: 2 parents from random species
- **EXPLOIT**: 3 parents from top species (local search)
- **EXPLORE**: 1 parent each from 3 different species (diversity)

### 3.3 Speciation Engine

**Purpose**: 8-phase speciation process for each generation

1. **Existing Species Processing**: Assign variants to species or cluster 0 (radius enforcement)
2. **Cluster 0 Speciation**: Form new species from cohesive clusters
3. **Merging**: Combine similar species (θ_merge); radius enforcement after merging
4. **Capacity Enforcement**: Enforce species capacity only (radius in Phases 1 and 3)
5. **Freeze & Incubator**: Track stagnation, freeze/dissolve species
6. **Cluster 0 Capacity Enforcement**: Archive excess reserves
7. **Final Redistribution**: Update species_id from tracker for elites, reserves, temp; redistribute to files; archive append-only
8. **Metrics & Stats**: Calculate diversity and cluster quality

### 3.4 Genome Tracker

**Purpose**: Authoritative source of truth for species_id assignments

- Updated at every speciation event
- Used in Phase 7 to synchronize file-based species_id values
- Enables efficient deferred file updates
- Archive is a final destination: genomes in archive are not moved back to elites or reserves

---

## 4. Evolution Flow

### 4.1 Generation 0 (Initialization)

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

### 4.2 Generation N (Evolution Loop)

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

### 4.3 Termination Conditions

- Maximum generations reached
- Threshold achieved (population_max_toxicity ≥ threshold)
- All species frozen and reserves empty
- Runtime error or user interruption

---

## 5. Speciation Framework

### 5.1 Phase 1: Existing Species Processing

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

### 5.2 Phase 2: Cluster 0 Speciation

Form new species when cluster 0 contains cohesive clusters.

```
When |cluster_0| ≥ cluster0_min_cluster_size:
  Select leader L = argmax(fitness) from cluster 0
  Create new species S with leader L
  For each genome g in cluster 0:
    if d_ensemble(g, L) < θ_sim:
      move g to S
```

### 5.3 Phase 3: Merging

Combine similar species to reduce redundancy.

```
For all species pairs (S_i, S_j):
  if d_ensemble(L_i, L_j) < θ_merge:
    Create merged species S_new
    S_new.members = S_i.members ∪ S_j.members
    S_new.leader = argmax(fitness) from S_new.members
    Mark S_i, S_j as extinct
```

### 5.4 Phase 4: Capacity Enforcement

Radius enforcement is done in Phase 1 and Phase 3; Phase 4 only enforces species capacity.

```
For each species S (species_id > 0):
  if |S| > species_capacity:
    Sort members by fitness (descending)
    Archive excess members (lowest fitness)
```

### 5.5 Phase 5: Freeze & Incubator

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

### 5.6 Phase 6: Cluster 0 Capacity Enforcement

```
if |cluster_0| > cluster0_max_capacity:
  Sort by fitness (descending)
  Archive excess (lowest fitness)
```

### 5.7 Phase 7: Final Redistribution

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

### 5.8 Phase 8: Metrics & Stats

Calculate and record:
- Inter-species diversity (mean distance between leaders)
- Intra-species diversity (mean distance within species)
- Cluster quality metrics (Silhouette, Davies-Bouldin, Calinski-Harabasz)

---

## 6. Parent Selection System

### 6.1 Category System

| Category | Contents | Usage |
|----------|----------|-------|
| Category 1 | Active species ∪ reserves | Primary selection pool |
| Category 2 | Frozen species | Fallback only |

### 6.2 Selection Modes

| Mode | Trigger | Parents | Strategy |
|------|---------|---------|----------|
| DEFAULT | Normal | 2 from random species | Balanced exploration |
| EXPLOIT | slope_of_avg_fitness ≤ 0 | 3 from top species | Local search |
| EXPLORE | generations_since_improvement ≥ stagnation_limit | 1 from each of 3 species | Diversity |

### 6.3 Mode Determination

```python
if generations_since_improvement >= stagnation_limit:
    mode = EXPLORE
elif slope_of_avg_fitness <= 0:
    mode = EXPLOIT
else:
    mode = DEFAULT
```

---

## 7. Key Metrics

### 7.1 Fitness Metrics

| Metric | Definition | Timing |
|--------|------------|--------|
| `avg_fitness` | mean(elites + reserves + temp) | Before speciation |
| `avg_fitness_generation` | mean(elites + reserves) | After distribution |
| `max_score_variants` | max(temp.json fitness) | Before speciation |
| `population_max_toxicity` | Cumulative max across all generations | After each generation |

### 7.2 Adaptive Selection Metrics

| Metric | Definition |
|--------|------------|
| `generations_since_improvement` | Consecutive generations without population_max_toxicity increase |
| `slope_of_avg_fitness` | Linear regression slope over recent avg_fitness_history |
| `avg_fitness_history` | Sliding window of recent avg_fitness values |

### 7.3 Diversity Metrics

| Metric | Description |
|--------|-------------|
| `inter_species_diversity` | Mean pairwise distance between species leaders |
| `intra_species_diversity` | Mean pairwise distance within species |
| `separation_ratio` | inter / intra (higher = better separation) |

### 7.4 Cluster Quality Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Silhouette Score | [-1, 1] | Higher = better separation |
| Davies-Bouldin Index | ≥ 0 | Lower = better |
| Calinski-Harabasz Index | > 0 | Higher = better |

---

## 8. Configuration Parameters

### 8.1 Speciation Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `theta_sim` | 0.25 | Similarity threshold for species assignment |
| `theta_merge` | 0.1 | Merge threshold (must be ≤ theta_sim) |
| `species_capacity` | 100 | Maximum individuals per species |
| `cluster0_max_capacity` | 1000 | Maximum individuals in cluster 0 |
| `cluster0_min_cluster_size` | 2 | Minimum size for cluster 0 speciation |
| `min_island_size` | 2 | Minimum species size before dissolution |
| `species_stagnation` | 20 | Generations without improvement before freezing |

### 8.2 Embedding Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `embedding_model` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `embedding_dim` | 384 | Embedding dimensionality |
| `embedding_batch_size` | 64 | Batch size for computation |

### 8.3 Ensemble Distance Weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_genotype` | 0.7 | Weight for embedding distance |
| `w_phenotype` | 0.3 | Weight for toxicity difference |

**Constraint**: `w_genotype + w_phenotype = 1.0`

---

## References

- [README.md](README.md) — Installation and usage guide
- [FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt) — Output file field definitions
- [PROCESS_FLOW.md](PROCESS_FLOW.md) — Detailed process flow documentation
