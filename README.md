# Speciated ToxSearch: Quality-Diversity Evolutionary Search for LLM Red-Teaming

A black-box evolutionary framework for systematic LLM safety testing through prompt evolution with semantic speciation. This system implements a steady-state evolutionary algorithm with leader-follower clustering to discover diverse prompts that elicit toxic responses from target language models.

---

## Abstract

Speciated ToxSearch is a quality-diversity approach to adversarial prompt discovery that partitions the solution space into behaviorally distinct species. The framework uses semantic embeddings and leader-follower clustering to maintain multiple distinct niches, each exploring different regions of the toxicity landscape. Our experiments demonstrate that speciated search achieves higher peak toxicity than baseline approaches while simultaneously exploring a broader semantic landscape.

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Algorithm](#algorithm)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Configuration](#configuration)
7. [Output Files](#output-files)
8. [Documentation](#documentation)
9. [Citation](#citation)

---

## Overview

### Research Contribution

This system implements a **steady-state (μ + λ) evolutionary algorithm** with **dynamic semantic speciation** to evolve prompts that maximize toxicity scores from target LLMs. Key contributions include:

- **Semantic Speciation**: Leader-follower clustering with ensemble distance combining genotype (prompt embeddings) and phenotype (toxicity scores)
- **Adaptive Parent Selection**: Three selection modes (DEFAULT, EXPLOIT, EXPLORE) based on population fitness trends
- **Species Management**: Dynamic species formation, merging, freezing, and capacity enforcement
- **Quality-Diversity Optimization**: Maintains multiple distinct niches with different toxicity distributions

### Use Cases

- **LLM Safety Testing**: Systematically discover prompts that trigger harmful responses
- **Red-Teaming**: Automated adversarial prompt generation across diverse attack vectors
- **Model Evaluation**: Assess model robustness to prompt variations
- **Research**: Study evolutionary dynamics in prompt space and speciation behavior

---

## Key Features

### Core Capabilities

- **Steady-State Evolution**: Continuous population updates without generational replacement
- **Leader-Follower Speciation**: Ensemble distance with configurable thresholds (θ_sim, θ_merge)
- **Adaptive Selection**: Dynamic parent selection based on fitness trends and stagnation
- **Refusal Detection**: Pattern-based detection with penalty application
- **Comprehensive Metrics**: Operator effectiveness, diversity, and cluster quality tracking

### Speciation Framework

- **Species Formation**: New species emerge when cluster 0 reaches cohesive clusters
- **Species Merging**: Similar species (d < θ_merge) are merged to reduce redundancy
- **Species Freezing**: Stagnant species are frozen but preserved for potential merging
- **Capacity Management**: Species and cluster 0 capacity limits with fitness-based archiving

---

## Algorithm

### Fitness Function

```
f(x) = toxicity_score(LLM(x)) ∈ [0, 1]
```

Where `x` is the prompt, `LLM(x)` is the model's response, and `toxicity_score` is obtained from Google Perspective API.

### Population Structure

- **Elites (E)**: Active species members (species_id > 0)
- **Reserves (R)**: Cluster 0 outliers (species_id = 0)
- **Archive**: Capacity-overflow genomes (species_id = -1). Archived genomes are not reintroduced into the active population.

### Ensemble Distance

Species assignment uses ensemble distance combining semantic and behavioral similarity:

```
d_ensemble = 0.7 × d_genotype + 0.3 × d_phenotype
```

- **d_genotype**: Cosine distance between prompt embeddings (all-MiniLM-L6-v2)
- **d_phenotype**: Normalized difference in toxicity scores

### Speciation Thresholds

- **Assignment (θ_sim = 0.25)**: Variants within this distance of a leader join the species
- **Merging (θ_merge = 0.1)**: Species with leader distance below this threshold are merged

### Parent Selection Modes

| Mode | Trigger | Strategy |
|------|---------|----------|
| DEFAULT | Normal operation | 2 parents from random species |
| EXPLOIT | slope_of_avg_fitness ≤ 0 | 3 parents from top species (local search) |
| EXPLORE | generations_since_improvement ≥ stagnation_limit | 1 parent each from 3 different species (diversity) |

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Perspective API key

### Setup

```bash
# Clone repository
git clone <repository-url>
cd eost-cam-llm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env_example.txt .env
# Edit .env: add PERSPECTIVE_API_KEY=your_api_key_here
```

### Model Setup

Place GGUF models in the `models/` directory:

```
models/
└── llama3.1-8b-instruct-gguf/
    └── Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf
```

---

## Usage

### Quick Start

```bash
bash run_experiments_local.sh
```

### Direct Execution

```bash
python src/main.py \
    --generations 50 \
    --threshold 0.99 \
    --moderation-methods google \
    --theta-sim 0.25 \
    --theta-merge 0.1 \
    --species-capacity 100 \
    --species-stagnation 20 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --seed-file data/prompt.csv
```

### Minimal Example

```bash
python src/main.py \
    --generations 10 \
    --rg models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q4_K_M.gguf \
    --operators paraphrasing \
    --seed-file data/prompt.csv
```

---

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--generations` | 50 | Maximum number of generations |
| `--threshold` | 0.99 | North-star threshold for termination |
| `--theta-sim` | 0.25 | Similarity threshold for species assignment |
| `--theta-merge` | 0.1 | Merge threshold for combining similar species |
| `--species-capacity` | 100 | Maximum individuals per species |
| `--cluster0-max-capacity` | 1000 | Maximum individuals in cluster 0 |
| `--species-stagnation` | 20 | Generations without improvement before freezing |
| `--stagnation-limit` | 5 | Generations without improvement before EXPLORE mode |

### Embedding Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embedding-model` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `--embedding-dim` | 384 | Embedding dimensionality |
| `--embedding-batch-size` | 64 | Batch size for embedding computation |

---

## Output Files

All outputs are saved to `data/outputs/YYYYMMDD_HHMM/`:

| File | Description |
|------|-------------|
| `elites.json` | Species members (species_id > 0) |
| `reserves.json` | Cluster 0 members (species_id = 0) |
| `archive.json` | Archived genomes (capacity overflow); append-only, not moved back to elites/reserves |
| `EvolutionTracker.json` | Per-generation statistics and speciation metrics |
| `speciation_state.json` | Species structure and state |
| `genome_tracker.json` | Genome lineage and species_id tracking |
| `operator_effectiveness_cumulative.csv` | Operator performance metrics |
| `figures/` | Visualization outputs |

### Species States

- **active**: Participates in evolution and parent selection
- **frozen**: Stagnated, excluded from primary selection, can still merge
- **incubator**: Dissolved due to small size, members moved to cluster 0
- **extinct**: Parent species after merging

---

## Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: System architecture and algorithmic details
- **[FIELD_DEFINITIONS.txt](FIELD_DEFINITIONS.txt)**: Field definitions for all output files
- **[PROCESS_FLOW.md](PROCESS_FLOW.md)**: Complete end-to-end process flow

---

## Citation

```bibtex
@inproceedings{speciated-toxsearch-2026,
  title     = {Speciated ToxSearch: Quality-Diversity Evolutionary Search for LLM Red-Teaming},
  author    = {[Authors]},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference (GECCO)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA}
}
```

---

## License

[License information]

---

## Acknowledgments

[Acknowledgments]
