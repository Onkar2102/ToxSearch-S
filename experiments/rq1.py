#!/usr/bin/env python3
"""RQ1: Quality + diversity comparison (baseline vs speciated ToxSearch). Outputs to experiments/comparison_results/rq1_quality/."""

# ===========================================================================
# # RQ1: Quality + Diversity Comparison (Baseline vs Speciated ToxSearch)
# 
# **Research Question**: Does Speciated ToxSearch find higher-quality toxic prompts faster, and does it discover a more diverse set of toxic behaviors?
# 
# ## How to run
# - Run all cells top-to-bottom.
# - Inputs:
#   - Baseline: `data/outputs/run01_comb` … `run10_comb`
#   - Speciated: `data/outputs/run01_speciated` … `run10_speciated`
# - Outputs: `experiments/comparison_results/rq1_quality/`.
# 
# ## Metrics
# 
# ### Quality Metrics
# - **Best-of-run toxicity**: max(toxicity) per run
# - **Time-to-threshold**: Eval index to first reach 0.80/0.90/0.95
# - **AUC (best-so-far)**: Area under best-so-far toxicity curve
# - **Top-10/Top-50 mean**: Mean toxicity of top K unique prompts
# 
# ### Diversity Metrics
# - **Cluster count**: # DBSCAN clusters in top-50 embeddings
# - **Semantic coverage**: Avg pairwise distance among top-50
# - **Novel cluster %**: % speciated top-50 outside baseline clusters
# 
# ### Statistical Analysis
# - Mann-Whitney U test
# - Cliff's delta effect size
# - Bootstrap 95% CI
# - Holm-Bonferroni correction
# 
# ## Notes / assumptions
# - Some baseline runs may not include embeddings; diversity metrics that require embeddings will be reported as undefined (NaN) when needed.
# ===========================================================================

# =============================================================================
# Section 1: Setup and Imports
# =============================================================================
import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances
from sklearn.manifold import MDS

# Topic modeling imports (optional - will handle gracefully if not available)
try:
    from bertopic import BERTopic
    from sentence_transformers import SentenceTransformer
    BERTOPIC_AVAILABLE = True
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False
    SENTENCE_TRANSFORMER_AVAILABLE = False
    print("Warning: BERTopic not available. Topic-based diversity analysis will be skipped.")
    print("Install with: pip install bertopic sentence-transformers")

# Embedding model for generating missing embeddings
_embedding_model = None
_embedding_generation_count = 0  # Track how many embeddings we've generated
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Same model used in speciation/embeddings.py

# Refusal detection
try:
    from src.utils.refusal_detector import is_refusal
    REFUSAL_DETECTOR_AVAILABLE = True
except ImportError:
    REFUSAL_DETECTOR_AVAILABLE = False
    print("Warning: Refusal detector not available. Will skip refusal filtering.")

# Set publication-grade style
sns.set_theme(style="whitegrid", context="paper", palette="deep", font_scale=1.2)
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Helvetica', 'sans-serif']
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['grid.linewidth'] = 0.8
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['lines.markersize'] = 6
plt.rcParams['patch.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.1

# Paths
PROJ = Path(os.getcwd()).resolve()
if not (PROJ / "src").exists():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "src").exists():
            PROJ = p
            break

BASE = PROJ / "data" / "outputs"
OUT = PROJ / "experiments" / "comparison_results" / "rq1_quality"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# Run directories (using runs 01-05 only)
RUNS_BASELINE = [BASE / f"run0{i}_comb" for i in range(1, 6)]  # run01..run05
RUNS_SPECIATION = [BASE / f"run0{i}_speciated" for i in range(1, 6)]  # run01..run05

# Constants
SEED = 42
np.random.seed(SEED)
DBSCAN_EPS = 0.3  # Cosine distance threshold
DBSCAN_MIN_SAMPLES = 2
THRESHOLDS = [0.80, 0.90, 0.95]  # Time-to-threshold levels

# Colors
COLOR_BASELINE = "#00B8D9"
COLOR_SPECIATION = "#FF5630"

print(f"Project: {PROJ}")
print(f"Output: {OUT}")
print(f"Baseline runs: {[p.name for p in RUNS_BASELINE]}")
print(f"Speciation runs: {[p.name for p in RUNS_SPECIATION]}")

# Verify embedding generation capability
print("\n" + "="*80)
print("EMBEDDING GENERATION CHECK")
print("="*80)
if SENTENCE_TRANSFORMER_AVAILABLE:
    print(f"✓ sentence-transformers available")
    print(f"  Model: {EMBEDDING_MODEL_NAME}")
    # Test model loading
    try:
        test_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        test_emb = test_model.encode("test prompt", normalize_embeddings=True)
        print(f"  ✓ Model loaded successfully (embedding dim: {len(test_emb)})")
        del test_model  # Free memory
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        print("  WARNING: Embedding generation will not work!")
else:
    print("✗ sentence-transformers NOT available")
    print("  Install with: pip install sentence-transformers")
    print("  WARNING: Cannot generate missing embeddings!")
print("="*80)

# =============================================================================
# Section 1b: Helper Functions
# =============================================================================

def load_json(path: Path) -> Any:
    """Load JSON file; return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def canonicalize_prompt(text: str) -> str:
    """Canonicalize prompt: strip + collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def get_toxicity(genome: Dict[str, Any]) -> Optional[float]:
    """Extract toxicity with strict precedence."""
    if not isinstance(genome, dict):
        return None
    for key in ("toxicity", "north_star_score", "fitness"):
        v = genome.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    try:
        v = genome.get("moderation_result", {}).get("google", {}).get("scores", {}).get("toxicity")
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    except Exception:
        pass
    return None


def _get_embedding_model():
    """Get or create singleton embedding model for generating missing embeddings."""
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMER_AVAILABLE:
        try:
            print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}' for generating missing embeddings...")
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            print("Embedding model loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            return None
    return _embedding_model


def get_embedding(genome: Dict[str, Any], generate_if_missing: bool = True) -> Optional[np.ndarray]:
    """
    Extract prompt embedding as numpy array.
    
    If embedding is missing and generate_if_missing=True, generates it using SentenceTransformer
    (all-MiniLM-L6-v2 model, same as used in speciation/embeddings.py).
    
    Args:
        genome: Genome dictionary
        generate_if_missing: If True, generate embedding when missing (default: True)
    
    Returns:
        Embedding as numpy array, or None if unavailable
    """
    if not isinstance(genome, dict):
        return None
    
    # Try to extract existing embedding
    emb = genome.get("prompt_embedding")
    if emb is not None and isinstance(emb, list) and len(emb) > 0:
        return np.array(emb, dtype=np.float32)
    
    # If missing and generation is enabled, generate it
    if generate_if_missing and SENTENCE_TRANSFORMER_AVAILABLE:
        prompt = genome.get("prompt")
        if prompt and isinstance(prompt, str) and prompt.strip():
            model = _get_embedding_model()
            if model is not None:
                try:
                    # Generate embedding (L2-normalized, same as in embeddings.py)
                    embedding = model.encode(prompt, normalize_embeddings=True, show_progress_bar=False)
                    global _embedding_generation_count
                    _embedding_generation_count += 1
                    if _embedding_generation_count == 1:
                        print(f"Generating missing embeddings using {EMBEDDING_MODEL_NAME}...")
                    return np.array(embedding, dtype=np.float32)
                except Exception as e:
                    # Silently fail if generation fails
                    pass
    
    return None


def load_run_genomes(run_dir: Path, condition: str) -> List[Dict[str, Any]]:
    """Load genomes for a run based on condition."""
    if condition == "baseline":
        fnames = ["elites.json", "non_elites.json"]
    else:  # speciation
        fnames = ["elites.json", "reserves.json"]
    
    genomes = []
    for fname in fnames:
        data = load_json(run_dir / fname)
        if isinstance(data, list):
            genomes.extend(data)
    return genomes


def dedup_genomes(genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate by prompt, keeping highest toxicity instance."""
    prompt_map = {}
    for g in genomes:
        if not isinstance(g, dict):
            continue
        prompt = canonicalize_prompt(g.get("prompt", ""))
        if not prompt:
            continue
        tox = get_toxicity(g)
        if tox is None:
            continue
        if prompt not in prompt_map or tox > get_toxicity(prompt_map[prompt]):
            prompt_map[prompt] = g
    return list(prompt_map.values())

# =============================================================================
# Section 2: Quality Metrics Computation
# =============================================================================

def compute_best_so_far_series(tracker: Dict) -> List[float]:
    """Compute best-so-far toxicity series from tracker."""
    generations = tracker.get("generations", [])
    if not generations:
        return []
    
    generations = sorted(generations, key=lambda g: g.get("generation_number", 0))
    best_so_far = []
    running_max = 0.0
    
    for gen in generations:
        m = gen.get("max_score_variants", 0.0)
        if m is None:
            m = 0.0
        running_max = max(running_max, float(m))
        best_so_far.append(running_max)
    
    return best_so_far


def compute_cumulative_max_avg_fitness(tracker: Dict) -> List[float]:
    """Compute cumulative max of avg_fitness_generation from tracker."""
    generations = tracker.get("generations", [])
    if not generations:
        return []
    
    generations = sorted(generations, key=lambda g: g.get("generation_number", 0))
    cumulative_max = []
    running_max = 0.0
    
    for gen in generations:
        avg_fit = gen.get("avg_fitness_generation", 0.0)
        if avg_fit is None:
            avg_fit = 0.0
        running_max = max(running_max, float(avg_fit))
        cumulative_max.append(running_max)
    
    return cumulative_max


def compute_cumulative_population_max_toxicity(run_dir: Path, condition: str, max_gen: int = 51) -> List[float]:
    """Compute cumulative max population toxicity from genomes.
    
    For each generation, finds the maximum toxicity among all genomes
    that existed up to that generation (cumulative maximum).
    """
    # Load all genomes
    genomes = load_run_genomes(run_dir, condition)
    
    if not genomes:
        return [0.0] * max_gen
    
    # Get toxicity for each genome with its generation
    genome_tox = []
    for g in genomes:
        tox = get_toxicity(g)
        gen = g.get("generation", 0)
        if tox is not None and gen is not None:
            genome_tox.append((int(gen), float(tox)))
    
    if not genome_tox:
        return [0.0] * max_gen
    
    # For each generation, find cumulative max toxicity among genomes up to that generation
    cumulative_max = []
    running_max = 0.0
    
    for gen_num in range(max_gen):
        # Find max toxicity among genomes with generation <= gen_num
        max_tox = max([tox for gen, tox in genome_tox if gen <= gen_num], default=0.0)
        # Maintain running maximum to ensure it's truly cumulative and non-decreasing
        running_max = max(running_max, max_tox)
        cumulative_max.append(running_max)
    
    return cumulative_max


def compute_time_to_threshold(best_so_far: List[float], thresholds: List[float]) -> Dict[float, Optional[int]]:
    """Compute generation index to first reach each threshold."""
    result = {t: None for t in thresholds}
    for t in thresholds:
        for i, val in enumerate(best_so_far):
            if val >= t:
                result[t] = i
                break
        if result[t] is None:
            result[t] = len(best_so_far)  # Never reached = max
    return result


def compute_auc(best_so_far: List[float]) -> float:
    """Compute AUC of best-so-far curve (trapezoidal rule)."""
    if not best_so_far:
        return 0.0
    return np.trapz(best_so_far) / len(best_so_far)  # Normalize by length


def compute_topk_metrics(genomes: List[Dict[str, Any]], k_values: List[int]) -> Dict[str, float]:
    """Compute top-K mean toxicity metrics."""
    # Get toxicity scores and sort
    scores = [get_toxicity(g) for g in genomes if get_toxicity(g) is not None]
    scores_sorted = sorted(scores, reverse=True)
    
    result = {}
    for k in k_values:
        topk = scores_sorted[:k]
        if topk:
            result[f"top{k}_mean"] = np.mean(topk)
            result[f"top{k}_n"] = len(topk)
        else:
            result[f"top{k}_mean"] = None
            result[f"top{k}_n"] = 0
    return result


def extract_quality_metrics(run_dir: Path, condition: str) -> Dict[str, Any]:
    """Extract all quality metrics for a single run."""
    result = {
        "run_id": run_dir.name,
        "condition": condition,
    }
    
    # Load tracker for trajectory
    tracker = load_json(run_dir / "EvolutionTracker.json")
    if tracker:
        best_so_far = compute_best_so_far_series(tracker)
        cumulative_max_avg_fit = compute_cumulative_max_avg_fitness(tracker)
        cumulative_pop_max_tox = compute_cumulative_population_max_toxicity(run_dir, condition)
        result["best_so_far_series"] = best_so_far
        result["cumulative_max_avg_fitness"] = cumulative_max_avg_fit
        result["cumulative_population_max_toxicity"] = cumulative_pop_max_tox
        result["Qmax"] = max(best_so_far) if best_so_far else None
        result["AUC"] = compute_auc(best_so_far)
        result["G_observed"] = len(best_so_far)
        
        # Time-to-threshold
        ttt = compute_time_to_threshold(best_so_far, THRESHOLDS)
        for t, val in ttt.items():
            result[f"ttt_{t}"] = val
    
    # Load genomes for top-K metrics
    genomes = load_run_genomes(run_dir, condition)
    genomes_dedup = dedup_genomes(genomes)
    result["n_unique_prompts"] = len(genomes_dedup)
    
    topk = compute_topk_metrics(genomes_dedup, [10, 50])
    result.update(topk)
    
    return result


# Compute quality metrics for all runs
quality_results = []
for run_dir in RUNS_BASELINE:
    quality_results.append(extract_quality_metrics(run_dir, "baseline"))
for run_dir in RUNS_SPECIATION:
    quality_results.append(extract_quality_metrics(run_dir, "speciation"))

# Store series separately for plotting
best_so_far_data = {r["run_id"]: r.pop("best_so_far_series", []) for r in quality_results}
cumulative_max_avg_fit_data = {r["run_id"]: r.pop("cumulative_max_avg_fitness", []) for r in quality_results}
cumulative_pop_max_tox_data = {r["run_id"]: r.pop("cumulative_population_max_toxicity", []) for r in quality_results}

df_quality = pd.DataFrame(quality_results)
print("Quality Metrics Summary:")

# =============================================================================
# Section 3: Diversity Metrics Computation
# =============================================================================

def get_all_embeddings(genomes: List[Dict[str, Any]], generate_if_missing: bool = True) -> Tuple[np.ndarray, List[float]]:
    """
    Get embeddings for all prompts with valid toxicity scores.
    
    If embeddings are missing and generate_if_missing=True, generates them using SentenceTransformer
    (all-MiniLM-L6-v2 model, same as used in speciation/embeddings.py).
    
    This function ensures ALL valid prompts get embeddings by:
    1. Filtering for genomes with valid toxicity
    2. Batch-generating missing embeddings for efficiency
    3. Ensuring no valid prompts are skipped
    
    Args:
        genomes: List of genome dictionaries
        generate_if_missing: If True, generate embeddings when missing (default: True)
    
    Returns:
        Tuple of (embeddings array, toxicities list)
    """
    # Reset generation counter at start of batch
    global _embedding_generation_count
    _embedding_generation_count = 0
    
    # First, filter for genomes with valid toxicity and prompts
    valid_genomes = []
    for g in genomes:
        tox = get_toxicity(g)
        prompt = g.get("prompt", "")
        if tox is not None and prompt and isinstance(prompt, str) and prompt.strip():
            valid_genomes.append((g, tox, prompt))
    
    if not valid_genomes:
        return np.array([]), []
    
    # Separate genomes into those with and without embeddings
    genomes_with_emb = []
    genomes_without_emb = []
    
    for g, tox, prompt in valid_genomes:
        emb = g.get("prompt_embedding")
        if emb is not None and isinstance(emb, list) and len(emb) > 0:
            genomes_with_emb.append((g, tox, np.array(emb, dtype=np.float32)))
        else:
            genomes_without_emb.append((g, tox, prompt))
    
    # Batch generate embeddings for those missing them
    if genomes_without_emb and generate_if_missing and SENTENCE_TRANSFORMER_AVAILABLE:
        model = _get_embedding_model()
        if model is not None:
            try:
                # Extract prompts for batch encoding
                prompts_to_encode = [prompt for _, _, prompt in genomes_without_emb]
                
                if prompts_to_encode:
                    if _embedding_generation_count == 0:
                        print(f"Generating {len(prompts_to_encode)} missing embeddings using {EMBEDDING_MODEL_NAME}...")
                    
                    # Batch encode (more efficient than one-by-one)
                    generated_embeddings = model.encode(
                        prompts_to_encode, 
                        normalize_embeddings=True, 
                        show_progress_bar=False,
                        batch_size=64
                    )
                    
                    # Add generated embeddings to the list
                    for i, (g, tox, _) in enumerate(genomes_without_emb):
                        emb_array = np.array(generated_embeddings[i], dtype=np.float32)
                        genomes_with_emb.append((g, tox, emb_array))
                        _embedding_generation_count += 1
                    
                    if _embedding_generation_count > 0:
                        print(f"  Generated {_embedding_generation_count} missing embeddings.")
            except Exception as e:
                print(f"Warning: Failed to generate embeddings: {e}")
                # Continue with only genomes that had embeddings
    
    if not genomes_with_emb:
        return np.array([]), []
    
    # Extract embeddings and toxicities
    embeddings = np.array([emb for _, _, emb in genomes_with_emb])
    toxicities = [tox for _, tox, _ in genomes_with_emb]
    
    return embeddings, toxicities


def compute_cluster_count(embeddings: np.ndarray, eps: float = DBSCAN_EPS, 
                          min_samples: int = DBSCAN_MIN_SAMPLES) -> int:
    """Compute number of DBSCAN clusters."""
    if len(embeddings) < min_samples:
        return 0
    
    # Use cosine distance
    distances = cosine_distances(embeddings)
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="precomputed")
    labels = clustering.fit_predict(distances)
    
    # Count unique clusters (excluding noise = -1)
    unique_labels = set(labels) - {-1}
    return len(unique_labels)


def compute_semantic_spread(embeddings: np.ndarray) -> float:
    """Compute average pairwise cosine distance."""
    if len(embeddings) < 2:
        return 0.0
    
    distances = cosine_distances(embeddings)
    # Get upper triangle (excluding diagonal)
    upper_tri = distances[np.triu_indices_from(distances, k=1)]
    return float(np.mean(upper_tri))


def extract_diversity_metrics(run_dir: Path, condition: str) -> Dict[str, Any]:
    """Extract diversity metrics for a single run."""
    result = {
        "run_id": run_dir.name,
        "condition": condition,
    }
    
    # Load and dedup genomes
    genomes = load_run_genomes(run_dir, condition)
    genomes_dedup = dedup_genomes(genomes)
    
    # Get all embeddings (no top-K filtering for diversity comparison)
    # This will generate embeddings if missing
    embeddings, toxicities = get_all_embeddings(genomes_dedup, generate_if_missing=True)
    result["n_embeddings"] = len(embeddings)
    result["embeddings"] = embeddings  # Store for later use
    
    # Diagnostic: check if we got embeddings for all genomes
    n_genomes_with_tox = sum(1 for g in genomes_dedup if get_toxicity(g) is not None)
    if n_genomes_with_tox > len(embeddings):
        missing = n_genomes_with_tox - len(embeddings)
        print(f"  WARNING [{run_dir.name}]: {missing} genomes with toxicity but no embeddings")
    
    if len(embeddings) > 0:
        result["cluster_count"] = compute_cluster_count(embeddings)
        result["semantic_spread"] = compute_semantic_spread(embeddings)
    else:
        result["cluster_count"] = 0
        result["semantic_spread"] = 0.0
    
    return result


# Compute diversity metrics for all runs
diversity_results = []
for run_dir in RUNS_BASELINE:
    diversity_results.append(extract_diversity_metrics(run_dir, "baseline"))
for run_dir in RUNS_SPECIATION:
    diversity_results.append(extract_diversity_metrics(run_dir, "speciation"))

# Store embeddings separately
embeddings_data = {r["run_id"]: r.pop("embeddings") for r in diversity_results}

df_diversity = pd.DataFrame(diversity_results)
print("Diversity Metrics Summary:")

# =============================================================================
# Section 3b: Novel Cluster Discovery
# =============================================================================

# NOTE:
# Baseline runs in this repo do not include `prompt_embedding` in `elites.json/non_elites.json`.
# That means the novel-cluster metric (defined in embedding space) is **undefined** for this dataset.
# We return NaN (and skip downstream plotting) rather than crashing.

def compute_novel_cluster_rate(
    baseline_embeddings_list: List[np.ndarray],
    speciated_embeddings: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute fraction of speciated top-K embeddings far from baseline clusters.

    Returns NaN if baseline embeddings are unavailable.
    """
    if speciated_embeddings is None or len(speciated_embeddings) == 0:
        return 0.0

    # Combine all baseline embeddings (if any)
    baseline_nonempty = [e for e in baseline_embeddings_list if isinstance(e, np.ndarray) and e.size > 0]
    if len(baseline_nonempty) == 0:
        return float("nan")

    all_baseline = np.vstack(baseline_nonempty)

    # Cluster baseline embeddings
    distances_base = cosine_distances(all_baseline)
    clustering = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = clustering.fit_predict(distances_base)

    # Compute centroids for each cluster
    unique_labels = set(labels) - {-1}
    if not unique_labels:
        centroids = np.mean(all_baseline, axis=0, keepdims=True)
    else:
        centroids = np.array([np.mean(all_baseline[labels == l], axis=0) for l in unique_labels])

    # For each speciated embedding, check if it's far from all centroids
    dists = cosine_distances(speciated_embeddings, centroids)
    novel = (np.min(dists, axis=1) > threshold)

    return float(np.mean(novel))


# Compute novel cluster rate for speciated runs (runs 01-05)
baseline_embs = [embeddings_data.get(f"run0{i}_comb", np.array([])) for i in range(1, 6)]

novel_rates = []
for run_id in [f"run0{i}_speciated" for i in range(1, 6)]:
    spec_emb = embeddings_data.get(run_id, np.array([]))
    rate = compute_novel_cluster_rate(baseline_embs, spec_emb)
    novel_rates.append({"run_id": run_id, "novel_cluster_rate": rate})

df_novel = pd.DataFrame(novel_rates)
print("Novel Cluster Discovery Rates (NaN = baseline embeddings unavailable):")

# =============================================================================
# Section 4: Statistical Analysis
# =============================================================================

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Cliff's delta effect size."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return np.nan
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def interpret_cliffs_delta(d: float) -> str:
    """Interpret Cliff's delta magnitude."""
    d_abs = abs(d)
    if d_abs < 0.147:
        return "negligible"
    elif d_abs < 0.33:
        return "small"
    elif d_abs < 0.474:
        return "medium"
    else:
        return "large"


def bootstrap_ci(data: np.ndarray, stat_func=np.median, n_bootstrap: int = 10000, 
                 ci: float = 0.95) -> Tuple[float, float]:
    """Bootstrap confidence interval."""
    if len(data) == 0:
        return (np.nan, np.nan)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(stat_func(sample))
    alpha = (1 - ci) / 2
    return (np.percentile(boot_stats, alpha * 100), 
            np.percentile(boot_stats, (1 - alpha) * 100))


def compare_conditions(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """Compare baseline vs speciation for a metric."""
    baseline_vals = df[df["condition"] == "baseline"][metric].dropna().values
    spec_vals = df[df["condition"] == "speciation"][metric].dropna().values
    
    if len(baseline_vals) == 0 or len(spec_vals) == 0:
        return {"metric": metric, "error": "insufficient data"}
    
    # Mann-Whitney U test
    stat, p = mannwhitneyu(baseline_vals, spec_vals, alternative='two-sided')
    
    # Cliff's delta
    delta = cliffs_delta(spec_vals, baseline_vals)
    
    # Bootstrap CI for median difference
    diff = np.median(spec_vals) - np.median(baseline_vals)
    
    return {
        "metric": metric,
        "baseline_median": np.median(baseline_vals),
        "speciation_median": np.median(spec_vals),
        "baseline_iqr": (np.percentile(baseline_vals, 25), np.percentile(baseline_vals, 75)),
        "speciation_iqr": (np.percentile(spec_vals, 25), np.percentile(spec_vals, 75)),
        "median_diff": diff,
        "U_statistic": stat,
        "p_value": p,
        "cliffs_delta": delta,
        "effect_interpretation": interpret_cliffs_delta(delta),
    }


# Merge quality and diversity dataframes
df_merged = df_quality.merge(df_diversity[["run_id", "cluster_count", "semantic_spread"]], on="run_id")

# Metrics to compare
metrics_to_compare = ["Qmax", "AUC", "top10_mean", "top50_mean", 
                      "cluster_count", "semantic_spread"]

# Add time-to-threshold metrics
for t in THRESHOLDS:
    metrics_to_compare.append(f"ttt_{t}")

# Run comparisons
stats_results = []
for metric in metrics_to_compare:
    if metric in df_merged.columns:
        result = compare_conditions(df_merged, metric)
        stats_results.append(result)

df_stats = pd.DataFrame(stats_results)

# Apply Holm-Bonferroni correction
if "p_value" in df_stats.columns:
    p_values = df_stats["p_value"].values
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    corrected = np.zeros(n)
    for i, idx in enumerate(sorted_idx):
        corrected[idx] = min(p_values[idx] * (n - i), 1.0)
    df_stats["p_corrected"] = corrected
    df_stats["significant"] = df_stats["p_corrected"] < 0.05

print("Statistical Analysis Results:")

# Save
df_stats.to_json(OUT / "rq1_stats.json", orient="records", indent=2)
print(f"\nSaved: {OUT / 'rq1_stats.json'}")

# =============================================================================
# Section 5: Figure 1 - Cumulative Population Max Toxicity and Cumulative Max Avg Fitness
# =============================================================================

# GECCO/ACM two-column friendly sizing
# Single-column width is ~3.33 in; use ~3.35 for safety.
FIG_W_IN = 3.35
FIG_H_IN = 2.10

fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))

# Collect series by condition (runs 01-05)
baseline_pop_max_series = [cumulative_pop_max_tox_data[f"run0{i}_comb"] for i in range(1, 6)]
speciated_pop_max_series = [cumulative_pop_max_tox_data[f"run0{i}_speciated"] for i in range(1, 6)]

baseline_avg_fit_series = [cumulative_max_avg_fit_data[f"run0{i}_comb"] for i in range(1, 6)]
speciated_avg_fit_series = [cumulative_max_avg_fit_data[f"run0{i}_speciated"] for i in range(1, 6)]

# Pad to same length
max_len = max(
    max(len(s) for s in baseline_pop_max_series) if baseline_pop_max_series else 0,
    max(len(s) for s in speciated_pop_max_series) if speciated_pop_max_series else 0,
)

def pad_series(series_list, max_len):
    padded = []
    for s in series_list:
        if len(s) < max_len:
            padded.append(s + ([s[-1]] * (max_len - len(s))) if s else ([0.0] * max_len))
        else:
            padded.append(s[:max_len])
    return np.array(padded, dtype=np.float32)

baseline_pop_max_arr = pad_series(baseline_pop_max_series, max_len)
speciated_pop_max_arr = pad_series(speciated_pop_max_series, max_len)
baseline_avg_fit_arr = pad_series(baseline_avg_fit_series, max_len)
speciated_avg_fit_arr = pad_series(speciated_avg_fit_series, max_len)

generations = np.arange(max_len)

# Compute maximum across runs (to show the best performance) for cumulative max toxicity
base_pop_max_max = np.max(baseline_pop_max_arr, axis=0)
spec_pop_max_max = np.max(speciated_pop_max_arr, axis=0)

# Median for avg fitness
base_avg_fit_median = np.median(baseline_avg_fit_arr, axis=0)
spec_avg_fit_median = np.median(speciated_avg_fit_arr, axis=0)

# Plot cumulative max toxicity (maximum across all runs for each generation) - solid lines
l1, = ax.plot(generations, base_pop_max_max, color=COLOR_BASELINE, linewidth=1.8, linestyle='-', label='Baseline')
l2, = ax.plot(generations, spec_pop_max_max, color=COLOR_SPECIATION, linewidth=1.8, linestyle='-', label='Speciated')

# Plot dotted lines (cumulative max avg_fitness_generation)
ax.plot(generations, base_avg_fit_median, color=COLOR_BASELINE, linewidth=1.4, linestyle='--', alpha=0.8)
ax.plot(generations, spec_avg_fit_median, color=COLOR_SPECIATION, linewidth=1.4, linestyle='--', alpha=0.8)

# Labels (GECCO-friendly sizes)
ax.set_xlabel("Generation", fontsize=8, fontweight='bold')
ax.set_ylabel("Toxicity Score", fontsize=8, fontweight='bold')

# Ticks
ax.tick_params(axis='both', labelsize=7, width=0.8)

# Limits
ax.set_xlim(0, max_len - 1)
ax.set_ylim(0.0, 1.0)

# Grid + spines
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
ax.set_axisbelow(True)
for side in ['bottom', 'left']:
    ax.spines[side].set_linewidth(0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend: colour only for Baseline / Speciated (solid = toxicity, dotted = avg fitness)
legend_elements = [
    Patch(facecolor=COLOR_BASELINE, edgecolor='none', label='Baseline'),
    Patch(facecolor=COLOR_SPECIATION, edgecolor='none', label='Speciated'),
]
ax.legend(handles=legend_elements, loc='upper left', frameon=False, fontsize=7)

plt.tight_layout(pad=0.2)
plt.savefig(OUT / "figures" / "fig1_trajectory.png", dpi=300, bbox_inches='tight')
plt.savefig(OUT / "figures" / "fig1_trajectory.pdf", bbox_inches='tight')
plt.close()
print("Saved: fig1_trajectory.png/pdf")

# =============================================================================
# Section 5b: Topic-as-Species Diversity Analysis (Alternative Figure 2)
# =============================================================================

def filter_refusals(genomes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out refusals from genomes."""
    if not REFUSAL_DETECTOR_AVAILABLE:
        return genomes  # Skip filtering if detector not available
    
    filtered = []
    for g in genomes:
        response = g.get("generated_output", "")
        if response and not is_refusal(response):
            filtered.append(g)
    return filtered


def build_comparable_analysis_set(run_dirs: List[Path], condition: str, 
                                   slice_method: str = "none", k: int = 50, 
                                   toxicity_threshold: float = 0.30) -> List[Dict[str, Any]]:
    """
    Build a comparable analysis set per run.
    
    Args:
        run_dirs: List of run directories
        condition: "baseline" or "speciation"
        slice_method: "none", "top_k", or "toxicity_threshold"
        k: Number of top prompts to keep (for top_k method)
        toxicity_threshold: Minimum toxicity (for toxicity_threshold method)
    
    Returns:
        List of genomes (deduped, filtered, optionally sliced)
    """
    all_genomes = []
    
    for run_dir in run_dirs:
        # Load genomes based on condition
        if condition == "baseline":
            # Baseline: only elites.json
            genomes = load_json(run_dir / "elites.json") or []
        else:  # speciation
            # Speciated: elites.json + reserves.json
            elites = load_json(run_dir / "elites.json") or []
            reserves = load_json(run_dir / "reserves.json") or []
            genomes = elites + reserves
        
        # Dedup by canonical prompt text, keep max toxicity per prompt
        genomes_dedup = dedup_genomes(genomes)
        
        # Filter refusals
        genomes_dedup = filter_refusals(genomes_dedup)
        
        # Apply slice (if requested)
        if slice_method == "top_k":
            # Sort by toxicity descending, take top K
            genomes_with_tox = [(g, get_toxicity(g)) for g in genomes_dedup 
                               if get_toxicity(g) is not None]
            genomes_with_tox.sort(key=lambda x: x[1], reverse=True)
            k_eff = min(k, len(genomes_with_tox))
            sliced = [g for g, _ in genomes_with_tox[:k_eff]]
        elif slice_method == "toxicity_threshold":
            # Keep only prompts with toxicity >= threshold
            sliced = [g for g in genomes_dedup 
                     if get_toxicity(g) is not None and get_toxicity(g) >= toxicity_threshold]
        else:  # slice_method == "none"
            # Use all genomes (no slicing)
            sliced = genomes_dedup
        
        all_genomes.extend(sliced)
    
    return all_genomes


def fit_global_topic_model(all_prompts: List[str], embedding_model_name: str = "all-MiniLM-L6-v2"):
    """
    Fit a single topic model on the union of all prompts.
    
    Returns:
        BERTopic model, topic assignments (list of topic IDs)
    """
    if not BERTOPIC_AVAILABLE:
        raise ImportError("BERTopic not available. Install with: pip install bertopic sentence-transformers")
    
    # Initialize embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    # Fit BERTopic
    topic_model = BERTopic(
        embedding_model=embedding_model,
        verbose=True,
        calculate_probabilities=False,
        nr_topics="auto"  # Let BERTopic determine optimal number
    )
    
    topics, probs = topic_model.fit_transform(all_prompts)
    
    return topic_model, topics


def compute_topic_diversity_metrics(topic_assignments: List[int]) -> Dict[str, float]:
    """
    Compute diversity metrics from topic assignments.
    
    Metrics:
    - N_1: Effective number of topics (Hill number, order 1)
    - K_topics: Raw count of unique topics
    - Evenness: J = H / log(K_topics)
    
    Returns:
        Dictionary with metrics
    """
    if len(topic_assignments) == 0:
        return {"N_1": 0.0, "K_topics": 0, "evenness": 0.0, "shannon_H": 0.0}
    
    # Count topics (exclude noise = -1)
    topic_counts = {}
    for t in topic_assignments:
        if t != -1:  # Exclude noise
            topic_counts[t] = topic_counts.get(t, 0) + 1
    
    if len(topic_counts) == 0:
        return {"N_1": 0.0, "K_topics": 0, "evenness": 0.0, "shannon_H": 0.0}
    
    # Total count (excluding noise)
    N = sum(topic_counts.values())
    if N == 0:
        return {"N_1": 0.0, "K_topics": 0, "evenness": 0.0, "shannon_H": 0.0}
    
    # Topic proportions
    p_t = {t: count / N for t, count in topic_counts.items()}
    
    # Shannon entropy
    H = -sum(p * np.log(p) for p in p_t.values() if p > 0)
    
    # Effective number of topics (Hill number, order 1)
    N_1 = np.exp(H) if H > 0 else 1.0
    
    # Raw count of unique topics
    K_topics = len(topic_counts)
    
    # Evenness
    evenness = H / np.log(K_topics) if K_topics > 1 and np.log(K_topics) > 0 else 0.0
    
    return {
        "N_1": float(N_1),
        "K_topics": int(K_topics),
        "evenness": float(evenness),
        "shannon_H": float(H)
    }


def compute_novel_topic_rate(baseline_topics: set, speciated_topics: set) -> float:
    """Compute fraction of speciated topics not in baseline."""
    if len(speciated_topics) == 0:
        return 0.0
    novel = speciated_topics - baseline_topics
    return len(novel) / len(speciated_topics)


# Run topic-based diversity analysis if BERTopic is available
if BERTOPIC_AVAILABLE:
    print("\n" + "="*80)
    print("TOPIC-AS-SPECIES DIVERSITY ANALYSIS")
    print("="*80)
    
    # Step 1: Build analysis sets (all genomes, deduped, refusals filtered)
    print("\nStep 1: Building analysis sets (all genomes: baseline=elites.json, speciated=elites.json+reserves.json)...")
    baseline_genomes = build_comparable_analysis_set(RUNS_BASELINE, "baseline", slice_method="none")
    speciated_genomes = build_comparable_analysis_set(RUNS_SPECIATION, "speciation", slice_method="none")
    
    print(f"  Baseline: {len(baseline_genomes)} prompts")
    print(f"  Speciated: {len(speciated_genomes)} prompts")
    
    # Extract prompts
    baseline_prompts = [canonicalize_prompt(g.get("prompt", "")) for g in baseline_genomes]
    speciated_prompts = [canonicalize_prompt(g.get("prompt", "")) for g in speciated_genomes]
    
    # Remove empty prompts
    baseline_prompts = [p for p in baseline_prompts if p]
    speciated_prompts = [p for p in speciated_prompts if p]
    
    # Step 2: Fit global topic model on union
    print("\nStep 2: Fitting global topic model on union of all prompts...")
    all_prompts = baseline_prompts + speciated_prompts
    topic_model, all_topics = fit_global_topic_model(all_prompts)
    
    # Split topics back
    n_baseline = len(baseline_prompts)
    baseline_topics = all_topics[:n_baseline]
    speciated_topics = all_topics[n_baseline:]
    
    print(f"  Total topics discovered: {len(set(t for t in all_topics if t != -1))}")
    
    # Step 3: Compute diversity metrics per run
    print("\nStep 3: Computing topic diversity metrics per run...")
    
    topic_diversity_results = []
    
    # Baseline runs
    for i, run_dir in enumerate(RUNS_BASELINE):
        # Get prompts for this run (all genomes from elites.json)
        run_genomes = build_comparable_analysis_set([run_dir], "baseline", slice_method="none")
        run_prompts = [canonicalize_prompt(g.get("prompt", "")) for g in run_genomes]
        run_prompts = [p for p in run_prompts if p]
        
        if len(run_prompts) > 0:
            # Get topic assignments for this run's prompts
            run_topics, _ = topic_model.transform(run_prompts)
            metrics = compute_topic_diversity_metrics(run_topics)
            metrics["run_id"] = run_dir.name
            metrics["condition"] = "baseline"
            topic_diversity_results.append(metrics)
        else:
            topic_diversity_results.append({
                "run_id": run_dir.name,
                "condition": "baseline",
                "N_1": 0.0,
                "K_topics": 0,
                "evenness": 0.0,
                "shannon_H": 0.0
            })
    
    # Speciated runs
    for i, run_dir in enumerate(RUNS_SPECIATION):
        # Get prompts for this run (all genomes from elites.json + reserves.json)
        run_genomes = build_comparable_analysis_set([run_dir], "speciation", slice_method="none")
        run_prompts = [canonicalize_prompt(g.get("prompt", "")) for g in run_genomes]
        run_prompts = [p for p in run_prompts if p]
        
        if len(run_prompts) > 0:
            # Get topic assignments for this run's prompts
            run_topics, _ = topic_model.transform(run_prompts)
            metrics = compute_topic_diversity_metrics(run_topics)
            metrics["run_id"] = run_dir.name
            metrics["condition"] = "speciation"
            topic_diversity_results.append(metrics)
        else:
            topic_diversity_results.append({
                "run_id": run_dir.name,
                "condition": "speciation",
                "N_1": 0.0,
                "K_topics": 0,
                "evenness": 0.0,
                "shannon_H": 0.0
            })
    
    df_topic_diversity = pd.DataFrame(topic_diversity_results)
    
    # Compute novel topic rate for speciated runs
    baseline_topic_set = set(t for t in baseline_topics if t != -1)
    novel_rates = []
    for i, run_dir in enumerate(RUNS_SPECIATION):
        run_genomes = build_comparable_analysis_set([run_dir], "speciation", slice_method="none")
        run_prompts = [canonicalize_prompt(g.get("prompt", "")) for g in run_genomes]
        run_prompts = [p for p in run_prompts if p]
        if len(run_prompts) > 0:
            run_topics, _ = topic_model.transform(run_prompts)
            run_topic_set = set(t for t in run_topics if t != -1)
            novel_rate = compute_novel_topic_rate(baseline_topic_set, run_topic_set)
            novel_rates.append(novel_rate)
        else:
            novel_rates.append(0.0)
    
    # Add novel topic rate to dataframe
    df_topic_diversity["novel_topic_rate"] = [0.0] * len(RUNS_BASELINE) + novel_rates
    
    print("\nTopic Diversity Summary:")
    print(df_topic_diversity.groupby("condition")[["N_1", "K_topics", "evenness"]].agg(["mean", "std"]))
    
    print(f"\nNovel Topic Rate (Speciated): {np.mean(novel_rates):.3f} ± {np.std(novel_rates):.3f}")
    print(f"  (Fraction of speciated topics not found in baseline)")
    
    # Create Figure 2 Alternative: Topic Diversity (formatting aligned with Fig 1 / Fig 2 diversity)
    print("\nCreating Figure 2 Alternative: Topic-as-Species Diversity...")
    
    # Two-panel width ~2× single-column; height consistent with other figures
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6))
    
    # Panel A: Effective number of topics (N_1)
    ax = axes[0]
    baseline_N1 = df_topic_diversity[df_topic_diversity["condition"] == "baseline"]["N_1"].values
    spec_N1 = df_topic_diversity[df_topic_diversity["condition"] == "speciation"]["N_1"].values
    
    data_N1 = pd.DataFrame({
        "Condition": ["Baseline"] * len(baseline_N1) + ["Speciated"] * len(spec_N1),
        "N_1": list(baseline_N1) + list(spec_N1)
    })
    
    # Same colours as Fig 1 / Fig 2: Baseline #00B8D9, Speciated #FF5630
    sns.boxplot(data=data_N1, x="Condition", y="N_1", ax=ax,
                order=["Baseline", "Speciated"],
                palette={"Baseline": COLOR_BASELINE, "Speciated": COLOR_SPECIATION},
                linewidth=0.7, width=0.6)
    sns.stripplot(data=data_N1, x="Condition", y="N_1", ax=ax, color="black", alpha=0.5, size=2.5, linewidth=0.3)
    ax.set_ylabel(r"Effective Number of Topics ($N_1$)", fontsize=8, fontweight='bold')
    ax.set_xlabel("Condition", fontsize=8, fontweight='bold')
    # Statistical test and title with p, d
    if len(baseline_N1) > 0 and len(spec_N1) > 0:
        _, p_n1 = mannwhitneyu(baseline_N1, spec_N1, alternative='two-sided')
        delta_n1 = cliffs_delta(spec_N1, baseline_N1)
        ax.set_title(f"(A) Topic Diversity: Effective Topics (p={p_n1:.3f}, d={delta_n1:.2f})", fontsize=8, fontweight='bold', pad=6)
    else:
        ax.set_title("(A) Topic Diversity: Effective Topics", fontsize=8, fontweight='bold', pad=6)
    ax.tick_params(axis='both', labelsize=7, width=0.8)
    for side in ['bottom', 'left']:
        ax.spines[side].set_linewidth(0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Panel B: Raw topic count (K_topics)
    ax = axes[1]
    baseline_K = df_topic_diversity[df_topic_diversity["condition"] == "baseline"]["K_topics"].values
    spec_K = df_topic_diversity[df_topic_diversity["condition"] == "speciation"]["K_topics"].values
    
    data_K = pd.DataFrame({
        "Condition": ["Baseline"] * len(baseline_K) + ["Speciated"] * len(spec_K),
        "K_topics": list(baseline_K) + list(spec_K)
    })
    
    # Same colours as Fig 1 / Fig 2: Baseline #00B8D9, Speciated #FF5630
    sns.boxplot(data=data_K, x="Condition", y="K_topics", ax=ax,
                order=["Baseline", "Speciated"],
                palette={"Baseline": COLOR_BASELINE, "Speciated": COLOR_SPECIATION},
                linewidth=0.7, width=0.6)
    sns.stripplot(data=data_K, x="Condition", y="K_topics", ax=ax, color="black", alpha=0.5, size=2.5, linewidth=0.3)
    ax.set_ylabel("Number of Unique Topics", fontsize=8, fontweight='bold')
    ax.set_xlabel("Condition", fontsize=8, fontweight='bold')
    # Statistical test and title with p, d
    if len(baseline_K) > 0 and len(spec_K) > 0:
        _, p_k = mannwhitneyu(baseline_K, spec_K, alternative='two-sided')
        delta_k = cliffs_delta(spec_K, baseline_K)
        ax.set_title(f"(B) Topic Coverage: Unique Topics (p={p_k:.3f}, d={delta_k:.2f})", fontsize=8, fontweight='bold', pad=6)
    else:
        ax.set_title("(B) Topic Coverage: Unique Topics", fontsize=8, fontweight='bold', pad=6)
    ax.tick_params(axis='both', labelsize=7, width=0.8)
    for side in ['bottom', 'left']:
        ax.spines[side].set_linewidth(0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "figures" / "fig2_topic_diversity.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "figures" / "fig2_topic_diversity.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: fig2_topic_diversity.png/pdf")
    
    # Save topic diversity data
    df_topic_diversity.to_csv(OUT / "rq1_topic_diversity.csv", index=False)
    print(f"Saved: {OUT / 'rq1_topic_diversity.csv'}")
    
    # =============================================================================
    # Topic Visualization with MDS
    # =============================================================================
    print("\n" + "="*80)
    print("CREATING TOPIC VISUALIZATIONS (MDS)")
    print("="*80)
    
    # Collect all genomes with embeddings, topics, and toxicities
    print("\nCollecting data for visualization...")
    
    # Get embeddings for all genomes
    baseline_embeddings, baseline_toxicities = get_all_embeddings(baseline_genomes, generate_if_missing=True)
    speciated_embeddings, speciated_toxicities = get_all_embeddings(speciated_genomes, generate_if_missing=True)
    
    # Get topic assignments for all prompts
    baseline_topic_assignments, _ = topic_model.transform(baseline_prompts)
    speciated_topic_assignments, _ = topic_model.transform(speciated_prompts)
    
    print(f"  Baseline: {len(baseline_embeddings)} prompts with embeddings")
    print(f"  Speciated: {len(speciated_embeddings)} prompts with embeddings")
    
    if len(baseline_embeddings) > 0 and len(speciated_embeddings) > 0:
        # Combine for dimensionality reduction
        all_embeddings = np.vstack([baseline_embeddings, speciated_embeddings])
        all_topics_vis = list(baseline_topic_assignments) + list(speciated_topic_assignments)
        all_toxicities_vis = baseline_toxicities + speciated_toxicities
        all_conditions = ["baseline"] * len(baseline_embeddings) + ["speciation"] * len(speciated_embeddings)
        
        # Filter out noise topics (-1)
        valid_mask = np.array([t != -1 for t in all_topics_vis])
        all_embeddings_valid = all_embeddings[valid_mask]
        all_topics_valid = np.array(all_topics_vis)[valid_mask]
        all_toxicities_valid = np.array(all_toxicities_vis)[valid_mask]
        all_conditions_valid = np.array(all_conditions)[valid_mask]
        
        print(f"  Valid prompts (excluding noise): {len(all_embeddings_valid)}")
        
        # Get unique topics and create color map
        unique_topics = sorted(set(all_topics_valid))
        n_topics = len(unique_topics)
        print(f"  Number of unique topics: {n_topics}")
        
        # Create color map for topics
        cmap = plt.cm.get_cmap('tab20')
        if n_topics > 20:
            cmap = plt.cm.get_cmap('tab20')
            # Extend colormap if needed
            colors = [cmap(i % 20) for i in range(n_topics)]
        else:
            colors = [cmap(i) for i in range(n_topics)]
        topic_color_map = {topic: colors[i] for i, topic in enumerate(unique_topics)}
        
        # ===== MDS Visualization =====
        print("\nComputing MDS reduction...")
        mds = MDS(n_components=2, random_state=SEED, dissimilarity='precomputed', normalized_stress='auto')
        # Compute pairwise distances
        distances = cosine_distances(all_embeddings_valid)
        embedding_2d_mds = mds.fit_transform(distances)
        
        # Split by condition
        baseline_mask = all_conditions_valid == "baseline"
        spec_mask = all_conditions_valid == "speciation"
        baseline_2d_mds = embedding_2d_mds[baseline_mask]
        spec_2d_mds = embedding_2d_mds[spec_mask]
        baseline_topics_plot = all_topics_valid[baseline_mask]
        spec_topics_plot = all_topics_valid[spec_mask]
        baseline_tox_plot = all_toxicities_valid[baseline_mask]
        spec_tox_plot = all_toxicities_valid[spec_mask]
        
        # Compute max toxicity per topic for each condition
        print("\nComputing max toxicity per topic...")
        baseline_max_tox_by_topic = {}
        for topic, tox in zip(baseline_topics_plot, baseline_tox_plot):
            if topic not in baseline_max_tox_by_topic:
                baseline_max_tox_by_topic[topic] = tox
            else:
                baseline_max_tox_by_topic[topic] = max(baseline_max_tox_by_topic[topic], tox)
        
        spec_max_tox_by_topic = {}
        for topic, tox in zip(spec_topics_plot, spec_tox_plot):
            if topic not in spec_max_tox_by_topic:
                spec_max_tox_by_topic[topic] = tox
            else:
                spec_max_tox_by_topic[topic] = max(spec_max_tox_by_topic[topic], tox)
        
        # Normalize toxicity values for color intensity (0-1 range)
        all_max_tox = list(baseline_max_tox_by_topic.values()) + list(spec_max_tox_by_topic.values())
        if all_max_tox:
            min_tox = min(all_max_tox)
            max_tox = max(all_max_tox)
            tox_range = max_tox - min_tox if max_tox > min_tox else 1.0
        else:
            min_tox, max_tox, tox_range = 0.0, 1.0, 1.0
        
        # Create color function: base color (baseline=blue, speciated=red) with intensity based on max toxicity
        def get_color_with_intensity(topic, condition, max_tox):
            """Get color with intensity based on max toxicity for the topic.
            
            Args:
                topic: Topic ID
                condition: "baseline" or "speciation"
                max_tox: Maximum toxicity for this topic in this condition
            
            Returns:
                RGB tuple with color intensity based on toxicity
            """
            # Normalize toxicity to 0-1 for intensity (higher toxicity = brighter/darker shade)
            intensity = (max_tox - min_tox) / tox_range if tox_range > 0 else 0.5
            intensity = max(0.2, min(1.0, intensity))  # Clamp between 0.2 and 1.0 for visibility
            
            # Base color: baseline = blue, speciated = red/orange
            if condition == "baseline":
                # Convert hex COLOR_BASELINE to RGB
                base_rgb = np.array([int(COLOR_BASELINE[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
            else:
                # Convert hex COLOR_SPECIATION to RGB
                base_rgb = np.array([int(COLOR_SPECIATION[i:i+2], 16) for i in (1, 3, 5)]) / 255.0
            
            # Adjust brightness: higher toxicity = brighter (more intense), lower toxicity = darker
            # Limit shade range so it never goes too dark: floor at 0.70, ceiling at 0.95
            brightness_floor = 0.70   # do not go darker than this (avoids very dark shades)
            brightness_ceiling = 0.95
            raw = brightness_floor + (brightness_ceiling - brightness_floor) * intensity
            brightness = np.clip(raw, brightness_floor, brightness_ceiling)
            color = base_rgb * brightness
            
            return tuple(np.clip(color, 0, 1))
        
        # Create 2D figure with MDS (X, Y) and Toxicity as color shade (formatting aligned with Fig 1/2)
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5.0))
        
        # Plot baseline points in 2D (same base colours as Fig 1/2: Baseline #00B8D9, Speciated #FF5630)
        baseline_colors = [get_color_with_intensity(t, "baseline", baseline_max_tox_by_topic.get(t, 0.0))
                          for t in baseline_topics_plot]
        ax.scatter(baseline_2d_mds[:, 0], baseline_2d_mds[:, 1],
                   c=baseline_colors, s=14, alpha=0.75,
                   edgecolors='black', linewidths=0.2, marker='o')
        
        # Plot speciated points in 2D
        spec_colors = [get_color_with_intensity(t, "speciation", spec_max_tox_by_topic.get(t, 0.0))
                      for t in spec_topics_plot]
        ax.scatter(spec_2d_mds[:, 0], spec_2d_mds[:, 1],
                   c=spec_colors, s=14, alpha=0.75,
                   edgecolors='black', linewidths=0.2, marker='o')
        
        # Labels and ticks (match Fig 1/2)
        ax.set_xlabel("MDS Dimension 1", fontsize=8, fontweight='bold')
        ax.set_ylabel("MDS Dimension 2", fontsize=8, fontweight='bold')
        ax.tick_params(axis='both', labelsize=7, width=0.8)
        
        # Legend: colour only for Baseline / Speciated (match Fig 1/2)
        legend_elements = [
            Patch(facecolor=COLOR_BASELINE, edgecolor='none', label='Baseline'),
            Patch(facecolor=COLOR_SPECIATION, edgecolor='none', label='Speciated'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=7)
        
        # Grid + spines (match Fig 1/2)
        ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ['bottom', 'left']:
            ax.spines[side].set_linewidth(0.9)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
            
        # Add toxicity range information as text annotation
        # Compute toxicity stats per topic for each condition
        print("\nComputing toxicity statistics per topic...")
        
        # Baseline toxicity by topic
        baseline_tox_by_topic = {}
        for topic, tox in zip(baseline_topics_plot, baseline_tox_plot):
            if topic not in baseline_tox_by_topic:
                baseline_tox_by_topic[topic] = []
            baseline_tox_by_topic[topic].append(tox)
        
        # Speciated toxicity by topic
        spec_tox_by_topic = {}
        for topic, tox in zip(spec_topics_plot, spec_tox_plot):
            if topic not in spec_tox_by_topic:
                spec_tox_by_topic[topic] = []
            spec_tox_by_topic[topic].append(tox)
        
        # Create summary text
        summary_text = "Toxicity Ranges by Topic:\n\n"
        summary_text += "Baseline:\n"
        for topic in sorted(baseline_tox_by_topic.keys()):
            tox_values = baseline_tox_by_topic[topic]
            summary_text += f"  Topic {topic}: min={min(tox_values):.3f}, max={max(tox_values):.3f}, mean={np.mean(tox_values):.3f} (n={len(tox_values)})\n"
        
        summary_text += "\nSpeciated:\n"
        for topic in sorted(spec_tox_by_topic.keys()):
            tox_values = spec_tox_by_topic[topic]
            summary_text += f"  Topic {topic}: min={min(tox_values):.3f}, max={max(tox_values):.3f}, mean={np.mean(tox_values):.3f} (n={len(tox_values)})\n"
        
        # Save toxicity summary
        with open(OUT / "rq1_topic_toxicity_summary.txt", 'w') as f:
            f.write(summary_text)
        print(f"Saved toxicity summary to {OUT / 'rq1_topic_toxicity_summary.txt'}")
        
        plt.tight_layout(pad=0.2)
        plt.savefig(OUT / "figures" / "fig3_topic_visualization_mds.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUT / "figures" / "fig3_topic_visualization_mds.pdf", bbox_inches='tight')
        plt.close()
        print("Saved: fig3_topic_visualization_mds.png/pdf")
        
    else:
        print("  Skipping visualization: insufficient data")
    
else:
    print("\nSkipping topic-based diversity analysis (BERTopic not available)")

# =============================================================================
# Section 5c: Figure 2 - Diversity Comparison (Baseline: elites.json vs Speciated: elites.json + reserves.json)
# =============================================================================

# Compute diversity metrics for baseline (elites only) and speciated (elites + reserves)
def compute_diversity_for_condition(run_dirs, condition):
    """Compute diversity metrics for a condition."""
    diversity_metrics = []
    
    for run_dir in run_dirs:
        if condition == "baseline":
            # Baseline: only elites.json
            genomes = load_json(run_dir / "elites.json") or []
        else:
            # Speciated: elites.json + reserves.json
            elites = load_json(run_dir / "elites.json") or []
            reserves = load_json(run_dir / "reserves.json") or []
            genomes = elites + reserves
        
        genomes_dedup = dedup_genomes(genomes)
        
        # Get all embeddings (no top-K filtering)
        embeddings, toxicities = get_all_embeddings(genomes_dedup)
        
        metrics = {
            "run_id": run_dir.name,
            "condition": condition,
            "n_unique_prompts": len(genomes_dedup),
            "n_with_embeddings": len(embeddings),
        }
        
        if len(embeddings) > 0:
            metrics["cluster_count"] = compute_cluster_count(embeddings)
            metrics["semantic_spread"] = compute_semantic_spread(embeddings)
        else:
            metrics["cluster_count"] = 0
            metrics["semantic_spread"] = 0.0
        
        diversity_metrics.append(metrics)
    
    return diversity_metrics

def compute_prompt_toxicities(run_dirs, condition):
    """Collect all prompt toxicity scores for ECDF."""
    all_toxicities = []
    
    for run_dir in run_dirs:
        if condition == "baseline":
            # Baseline: elites.json + non_elites.json
            elites = load_json(run_dir / "elites.json") or []
            non_elites = load_json(run_dir / "non_elites.json") or []
            genomes = elites + non_elites
        else:
            # Speciated: elites.json + reserves.json
            elites = load_json(run_dir / "elites.json") or []
            reserves = load_json(run_dir / "reserves.json") or []
            genomes = elites + reserves
        
        genomes_dedup = dedup_genomes(genomes)
        
        # Extract toxicity scores
        for g in genomes_dedup:
            tox = get_toxicity(g)
            if tox is not None:
                all_toxicities.append(float(tox))
    
    return np.array(all_toxicities)

def plot_ecdf(data, ax, label, color):
    """Plot empirical cumulative distribution function."""
    if len(data) == 0:
        return np.array([]), np.array([])
    
    # Sort data
    sorted_data = np.sort(data)
    n = len(sorted_data)
    
    # ECDF: y = (number of values <= x) / n
    y = np.arange(1, n + 1) / n
    
    # Plot step function (linewidth matches Fig 1 main lines)
    ax.plot(sorted_data, y, label=label, color=color, linewidth=1.8, drawstyle='steps-post')
    
    return sorted_data, y

baseline_diversity = compute_diversity_for_condition(RUNS_BASELINE, "baseline")
speciated_diversity = compute_diversity_for_condition(RUNS_SPECIATION, "speciation")

df_diversity_comparison = pd.DataFrame(baseline_diversity + speciated_diversity)

# Get prompt toxicities for ECDF
baseline_toxicities = compute_prompt_toxicities(RUNS_BASELINE, "baseline")
speciated_toxicities = compute_prompt_toxicities(RUNS_SPECIATION, "speciation")

# Create figure with single ECDF plot (wider than Fig 1 for readability)
fig, ax = plt.subplots(1, 1, figsize=(3.9, 2.6))

# Plot ECDF curves
baseline_sorted, baseline_y = plot_ecdf(baseline_toxicities, ax, "Baseline", COLOR_BASELINE)
spec_sorted, spec_y = plot_ecdf(speciated_toxicities, ax, "Speciated", COLOR_SPECIATION)

# Compute statistics for markers
if len(baseline_toxicities) > 0:
    baseline_q95 = np.percentile(baseline_toxicities, 95)
    baseline_qmax = np.max(baseline_toxicities)
    baseline_top10 = np.percentile(baseline_toxicities, 100 - (10/len(baseline_toxicities)*100)) if len(baseline_toxicities) >= 10 else baseline_qmax
    baseline_top10_median = np.median(np.sort(baseline_toxicities)[-10:]) if len(baseline_toxicities) >= 10 else baseline_qmax
    
    # Add vertical markers (thinner, subtler so ECDF curves stand out)
    ax.axvline(baseline_q95, color=COLOR_BASELINE, linestyle='--', linewidth=1.2, alpha=0.55)
    ax.axvline(baseline_top10_median, color=COLOR_BASELINE, linestyle=':', linewidth=1.2, alpha=0.55)
    ax.axvline(baseline_qmax, color=COLOR_BASELINE, linestyle='-', linewidth=1.2, alpha=0.55)

if len(speciated_toxicities) > 0:
    spec_q95 = np.percentile(speciated_toxicities, 95)
    spec_qmax = np.max(speciated_toxicities)
    spec_top10_median = np.median(np.sort(speciated_toxicities)[-10:]) if len(speciated_toxicities) >= 10 else spec_qmax
    
    # Add vertical markers (thinner, subtler)
    ax.axvline(spec_q95, color=COLOR_SPECIATION, linestyle='--', linewidth=1.2, alpha=0.55)
    ax.axvline(spec_top10_median, color=COLOR_SPECIATION, linestyle=':', linewidth=1.2, alpha=0.55)
    ax.axvline(spec_qmax, color=COLOR_SPECIATION, linestyle='-', linewidth=1.2, alpha=0.55)

# Print ECDF statistics for validation
print("\n" + "="*80)
print("ECDF STATISTICS VALIDATION")
print("="*80)
if len(baseline_toxicities) > 0:
    print(f"Baseline:")
    print(f"  Q95 (95th percentile): {baseline_q95:.4f}")
    print(f"  Top-10 median: {baseline_top10_median:.4f}")
    print(f"  Qmax (maximum): {baseline_qmax:.4f}")
    print(f"  Total prompts: {len(baseline_toxicities)}")
if len(speciated_toxicities) > 0:
    print(f"\nSpeciated:")
    print(f"  Q95 (95th percentile): {spec_q95:.4f}")
    print(f"  Top-10 median: {spec_top10_median:.4f}")
    print(f"  Qmax (maximum): {spec_qmax:.4f}")
    print(f"  Total prompts: {len(speciated_toxicities)}")
print("="*80 + "\n")

# Labels (GECCO-friendly, bold, same as Fig 1)
ax.set_xlabel("Toxicity Score", fontsize=8, fontweight='bold')
ax.set_ylabel("ECDF", fontsize=8, fontweight='bold')

# Ticks
ax.tick_params(axis='both', labelsize=7, width=0.8)

# Limits
ax.set_xlim(left=0, right=1.0)
ax.set_ylim(bottom=0, top=1.0)

# Grid + spines (match Fig 1)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
ax.set_axisbelow(True)
for side in ['bottom', 'left']:
    ax.spines[side].set_linewidth(0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Legend: colour only for Baseline/Speciated (match Fig 1; vertical markers are same colours)
legend_elements = [
    Patch(facecolor=COLOR_BASELINE, edgecolor='none', label='Baseline'),
    Patch(facecolor=COLOR_SPECIATION, edgecolor='none', label='Speciated'),
]
ax.legend(handles=legend_elements, loc='upper right', frameon=False, fontsize=7)
plt.tight_layout(pad=0.2)
plt.savefig(OUT / "figures" / "fig2_diversity_comparison.png", dpi=300, bbox_inches='tight')
plt.savefig(OUT / "figures" / "fig2_diversity_comparison.pdf", bbox_inches='tight')
plt.close()  # Close figure to free memory
print("Saved: fig2_diversity_comparison.png/pdf")

# =============================================================================
# Section 6: Summary Table
# =============================================================================

def format_metric(median, iqr):
    """Format median +/- IQR."""
    if isinstance(iqr, tuple):
        return f"{median:.3f} [{iqr[0]:.3f}, {iqr[1]:.3f}]"
    return f"{median:.3f}"

summary_rows = []
for _, row in df_stats.iterrows():
    metric = row["metric"]
    if "error" in row and row.get("error"):
        continue
    
    summary_rows.append({
        "Metric": metric,
        "Baseline (median [IQR])": format_metric(row["baseline_median"], row["baseline_iqr"]),
        "Speciated (median [IQR])": format_metric(row["speciation_median"], row["speciation_iqr"]),
        "Cliff's d": f"{row['cliffs_delta']:.3f} ({row['effect_interpretation']})",
        "p (corrected)": f"{row.get('p_corrected', row['p_value']):.4f}",
        "Sig.": "*" if row.get('significant', row['p_value'] < 0.05) else ""
    })

df_summary = pd.DataFrame(summary_rows)
print("\n" + "="*80)
print("RQ1 SUMMARY TABLE")
print("="*80)

# Save
df_summary.to_csv(OUT / "rq1_summary_table.csv", index=False)
print(f"\nSaved: {OUT / 'rq1_summary_table.csv'}")

# =============================================================================
# Save All Data
# =============================================================================

# Merge all metrics into one dataframe
df_all = df_merged.copy()
# `df_novel` only contains speciated runs; merge safely
if "df_novel" in globals() and not df_novel.empty:
    df_all = df_all.merge(df_novel[["run_id", "novel_cluster_rate"]], on="run_id", how="left")
else:
    df_all["novel_cluster_rate"] = np.nan

df_all.to_csv(OUT / "rq1_metrics_runlevel.csv", index=False)
print(f"Saved: {OUT / 'rq1_metrics_runlevel.csv'}")

# =============================================================================
# Section 7: Collect All Genomes and Ensure Embeddings
# =============================================================================

def collect_all_genomes_with_embeddings():
    """
    Collect all genomes from baseline and speciated runs, ensure all have embeddings,
    and save to rq1_data.json.
    """
    print("\n" + "="*80)
    print("COLLECTING ALL GENOMES AND ENSURING EMBEDDINGS")
    print("="*80)
    
    all_genomes = []
    
    # Collect baseline genomes (elites.json only)
    print("\nCollecting baseline genomes (elites.json)...")
    baseline_count = 0
    for run_dir in RUNS_BASELINE:
        elites = load_json(run_dir / "elites.json") or []
        for g in elites:
            if isinstance(g, dict) and g.get("prompt"):
                g_copy = g.copy()
                g_copy["source_run"] = run_dir.name
                g_copy["condition"] = "baseline"
                all_genomes.append(g_copy)
                baseline_count += 1
    print(f"  Collected {baseline_count} baseline genomes")
    
    # Collect speciated genomes (elites.json + reserves.json)
    print("\nCollecting speciated genomes (elites.json + reserves.json)...")
    speciated_count = 0
    for run_dir in RUNS_SPECIATION:
        elites = load_json(run_dir / "elites.json") or []
        reserves = load_json(run_dir / "reserves.json") or []
        for g in elites + reserves:
            if isinstance(g, dict) and g.get("prompt"):
                g_copy = g.copy()
                g_copy["source_run"] = run_dir.name
                g_copy["condition"] = "speciation"
                all_genomes.append(g_copy)
                speciated_count += 1
    print(f"  Collected {speciated_count} speciated genomes")
    
    print(f"\nTotal genomes collected: {len(all_genomes)}")
    
    # Deduplicate by canonical prompt (keep max toxicity)
    print("\nDeduplicating genomes (keep max toxicity per prompt)...")
    prompt_map = {}
    for g in all_genomes:
        prompt = canonicalize_prompt(g.get("prompt", ""))
        if not prompt:
            continue
        tox = get_toxicity(g)
        if tox is None:
            continue
        if prompt not in prompt_map or tox > get_toxicity(prompt_map[prompt]):
            prompt_map[prompt] = g
    
    deduped_genomes = list(prompt_map.values())
    print(f"  After deduplication: {len(deduped_genomes)} unique prompts")
    
    # Check and generate embeddings
    print("\nChecking embeddings...")
    genomes_with_emb = 0
    genomes_without_emb = 0
    genomes_without_prompt = 0
    
    for g in deduped_genomes:
        prompt = g.get("prompt", "")
        if not prompt or not isinstance(prompt, str) or not prompt.strip():
            genomes_without_prompt += 1
            continue
        
        emb = g.get("prompt_embedding")
        if emb is not None and isinstance(emb, list) and len(emb) > 0:
            genomes_with_emb += 1
        else:
            genomes_without_emb += 1
    
    print(f"  Genomes with embeddings: {genomes_with_emb}")
    print(f"  Genomes without embeddings: {genomes_without_emb}")
    print(f"  Genomes without valid prompts: {genomes_without_prompt}")
    
    # Generate missing embeddings
    if genomes_without_emb > 0:
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            print("\nERROR: sentence-transformers not available. Cannot generate embeddings.")
            print("Please install: pip install sentence-transformers")
            return None
        
        print(f"\nGenerating {genomes_without_emb} missing embeddings...")
        model = _get_embedding_model()
        if model is None:
            print("ERROR: Failed to load embedding model.")
            return None
        
        # Collect prompts that need embeddings
        prompts_to_encode = []
        genomes_needing_emb = []
        for g in deduped_genomes:
            prompt = g.get("prompt", "")
            if not prompt or not isinstance(prompt, str) or not prompt.strip():
                continue
            
            emb = g.get("prompt_embedding")
            if emb is None or not isinstance(emb, list) or len(emb) == 0:
                prompts_to_encode.append(prompt)
                genomes_needing_emb.append(g)
        
        if prompts_to_encode:
            try:
                # Batch encode
                print(f"  Encoding {len(prompts_to_encode)} prompts in batches...")
                generated_embeddings = model.encode(
                    prompts_to_encode,
                    normalize_embeddings=True,
                    show_progress_bar=True,
                    batch_size=64
                )
                
                # Add embeddings to genomes
                for i, g in enumerate(genomes_needing_emb):
                    g["prompt_embedding"] = generated_embeddings[i].tolist()
                
                print(f"  Successfully generated {len(generated_embeddings)} embeddings.")
            except Exception as e:
                print(f"ERROR: Failed to generate embeddings: {e}")
                import traceback
                traceback.print_exc()
                return None
    
    # Final verification
    print("\nFinal verification...")
    final_with_emb = sum(1 for g in deduped_genomes 
                         if g.get("prompt_embedding") is not None 
                         and isinstance(g.get("prompt_embedding"), list) 
                         and len(g.get("prompt_embedding", [])) > 0)
    final_without_emb = len(deduped_genomes) - final_with_emb
    
    print(f"  Genomes with embeddings: {final_with_emb}")
    print(f"  Genomes without embeddings: {final_without_emb}")
    
    if final_without_emb > 0:
        print(f"\nWARNING: {final_without_emb} genomes still missing embeddings!")
        # Show examples
        missing_examples = [g for g in deduped_genomes[:5] 
                           if not (g.get("prompt_embedding") is not None 
                                  and isinstance(g.get("prompt_embedding"), list) 
                                  and len(g.get("prompt_embedding", [])) > 0)]
        for g in missing_examples[:3]:
            prompt = g.get("prompt", "")[:50]
            print(f"    Example: prompt='{prompt}...', has_emb={g.get('prompt_embedding') is not None}")
    else:
        print("  ✓ All genomes have embeddings!")
    
    # Save to rq1_data.json
    output_file = OUT / "rq1_data.json"
    print(f"\nSaving to {output_file}...")
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(deduped_genomes, f, indent=2, ensure_ascii=False)
        print(f"  Saved {len(deduped_genomes)} genomes to {output_file}")
        return deduped_genomes
    except Exception as e:
        print(f"ERROR: Failed to save data: {e}")
        return None

# Run data collection
collected_genomes = collect_all_genomes_with_embeddings()

print("\n" + "="*80)
print("RQ1 ANALYSIS COMPLETE")
print("="*80)
