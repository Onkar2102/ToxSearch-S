#!/usr/bin/env python3
"""
# RQ2: Species Quality and Toxicity Analysis

**Research Question**: How do individual species in Speciated ToxSearch differ in their
toxicity performance, and what are the toxicity scores achieved by different species?

## How to run
- Run script: `python experiments/rq2.py`
- Inputs: `data/outputs/run01_speciated` through `run05_speciated`
- Outputs: `experiments/comparison_results/rq2_speciation/`

## What this analysis covers
- Species cluster separation metrics (intra vs inter distances)
- Species-level toxicity distributions and correlations
- Semantic space visualizations (2D projections, word clouds)
- Network-based prompt similarity analysis

## Outputs (4 total)
1. Table 1: Cluster Metrics Summary (rq2_cluster_metrics.csv)
   - Per-run: species count, intra/inter distances, separation ratio
2. Figure 1: Toxicity Distribution by Species (boxplot)
3. Figure 2: Species Semantic Map + Word Clouds (MDS projection)
4. Figure 3: Force-Directed Prompt Similarity Graph

## Notes / assumptions
- Refusals (`is_refusal == 1`) are excluded from analysis
- Species IDs `0` and `-1` are excluded as invalid
- Uses run01_speciated through run05_speciated only
"""

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
from collections import defaultdict, Counter
import colorsys

warnings.filterwarnings('ignore')

import matplotlib

try:
    from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts
except ImportError:
    from utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts

configure_matplotlib_embedded_fonts()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_distances, cosine_similarity
from sklearn.manifold import MDS
from scipy import stats
from scipy.spatial.distance import pdist

# Publication-grade plot settings
plt.rcParams.update({
    'figure.figsize': (12, 8),
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

# Paths
PROJ = Path(os.getcwd()).resolve()
if not (PROJ / "src").exists():
    for p in [Path.cwd(), *Path.cwd().parents]:
        if (p / "src").exists():
            PROJ = p
            break

# Override with: export TOXSEARCH_DATA_OUTPUTS="/path/to/.../toxsearch_s_outputs copy"
_outputs_override = os.environ.get("TOXSEARCH_DATA_OUTPUTS", "").strip()
BASE = Path(_outputs_override).resolve() if _outputs_override else (PROJ / "data" / "outputs")
OUT = PROJ / "experiments" / "comparison_results" / "rq2_speciation"
(OUT / "figures").mkdir(parents=True, exist_ok=True)

# Run directories (speciated only) - run01 to run05
RUNS_SPECIATION = [BASE / f"run0{i}_speciated" for i in range(1, 6)]

# Representative run for detailed analysis
REPRESENTATIVE_RUN = BASE / "run02_speciated"

# Constants
SEED = 42
np.random.seed(SEED)

print(f"Project: {PROJ}")
print(f"Output: {OUT}")
print(f"Speciation runs: {[p.name for p in RUNS_SPECIATION if p.exists()]}")

# Optional imports with graceful fallback
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: NetworkX not installed. Force-directed graph will be skipped.")

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print("Warning: WordCloud not installed. Word cloud will be skipped.")


# =============================================================================
# Section 2: Helper Functions
# =============================================================================

# Global species color mapping (performance-based: high toxicity = dark/visible colors)
def build_global_species_color_map() -> Dict[Tuple[str, str], Tuple[float, float, float]]:
    """
    Build a color mapping for all species (run_id, species_id).
    Colors are assigned based on max toxicity: high-performing species get dark/visible colors.
    Returns: {(run_id, species_id): (R, G, B)} mapping
    """
    # Collect all species with their max toxicity
    species_data = []  # List of (run_id, species_id, max_toxicity)
    
    for run_dir in RUNS_SPECIATION:
        if not run_dir.exists():
            continue
        
        run_id = run_dir.name
        spec_state = load_json(run_dir / "speciation_state.json")
        
        if not spec_state or "species" not in spec_state:
            continue
        
        # Load genomes to calculate max toxicity per species
        spec_data = load_species_data(run_dir)
        species_tox = defaultdict(list)
        
        for genome in spec_data["elites"] + (spec_data["reserves"] or []):
            sid = genome.get("species_id")
            if sid is not None and sid != 0 and sid != -1:
                tox = get_toxicity(genome)
                if tox is not None:
                    species_tox[str(sid)].append(float(tox))
        
        # Collect mature species with their max toxicity
        for sid, sdata in spec_state["species"].items():
            species_state = sdata.get("species_state", "")
            if species_state in ("active", "frozen"):
                tox_values = species_tox.get(str(sid), [])
                max_tox = float(np.max(tox_values)) if tox_values else 0.0
                species_data.append((run_id, str(sid), max_tox))
    
    # Sort by max toxicity (descending) - high performers first
    species_data.sort(key=lambda x: x[2], reverse=True)
    
    # Assign colors: high toxicity = dark/visible, low toxicity = light/muted
    species_color_map = {}
    n_species = len(species_data)
    
    if n_species == 0:
        return species_color_map
    
    # Normalize toxicity range for color assignment
    max_tox_all = max(x[2] for x in species_data) if species_data else 1.0
    min_tox_all = min(x[2] for x in species_data) if species_data else 0.0
    tox_range = max_tox_all - min_tox_all if max_tox_all > min_tox_all else 1.0
    
    # Use a vibrant colormap for high performers, muted for low performers
    # High toxicity: dark, saturated colors (tab20, Set3)
    # Low toxicity: light, pastel colors (Pastel1, light versions)
    cmap_dark = plt.cm.get_cmap('tab20')  # Dark, visible colors
    cmap_medium = plt.cm.get_cmap('Set3')  # Medium saturation
    cmap_light = plt.cm.get_cmap('Pastel1')  # Light, muted colors
    
    for idx, (run_id, sid, max_tox) in enumerate(species_data):
        key = (run_id, sid)
        
        # Assign colors based on rank:
        # - Top 10 species: dark, vibrant colors
        # - Middle species (between top 10 and bottom 50%): vibrant colors
        # - Bottom 50%: light colors
        
        bottom_50_threshold = int(n_species * 0.5)  # Index threshold for bottom 50%
        
        if idx < 10:  # Top 10 species - dark, vibrant colors (highest toxicity)
            # Use tab20: distribute top 10 across the colormap for variety
            color_idx = idx % 20
            base_color = cmap_dark(color_idx / 19.0 if 19 > 0 else 0)[:3]
            # Darken for high performers - make it more visible and darker
            r, g, b = base_color
            # Darken: multiply by 0.7-0.8 to make it darker, but keep saturation
            color = tuple(max(0.2, min(0.7, c * 0.75)) for c in (r, g, b))
        
        elif idx < bottom_50_threshold:  # Middle species - vibrant colors (moderate saturation)
            # Generate vibrant colors using HSV with moderate saturation
            # Use a variety of hues to ensure visual distinction
            hue = ((idx - 10) * 0.618) % 1.0  # Golden ratio for good distribution
            # Moderate saturation (0.5-0.7) for vibrant colors, medium-high brightness (0.6-0.9)
            saturation = 0.5 + ((idx - 10) % 3) * 0.1  # Vary between 0.5-0.7
            brightness = 0.6 + ((idx - 10) % 4) * 0.1  # Vary between 0.6-0.9
            color = colorsys.hsv_to_rgb(hue, saturation, brightness)
        
        else:  # Bottom 50% - light colors (low saturation, high brightness)
            # Generate light colors using HSV with low saturation and high brightness
            hue = ((idx - bottom_50_threshold) * 0.618) % 1.0
            # Low saturation (0.2-0.4) and high brightness (0.85-0.95) for light colors
            saturation = 0.2 + ((idx - bottom_50_threshold) % 3) * 0.1  # Vary between 0.2-0.4
            brightness = 0.85 + ((idx - bottom_50_threshold) % 2) * 0.1  # Vary between 0.85-0.95
            color = colorsys.hsv_to_rgb(hue, saturation, brightness)
        
        species_color_map[key] = color
    
    return species_color_map

def load_json(path: Path) -> Any:
    """Load JSON file; return None if missing or invalid."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def get_toxicity(genome: Dict[str, Any]) -> Optional[float]:
    """Extract toxicity from genome with strict precedence."""
    if not isinstance(genome, dict):
        return None
    if genome.get("is_refusal") == 1:
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


def load_all_genomes(run_dir: Path) -> List[Dict[str, Any]]:
    """Load all genomes from elites and reserves."""
    genomes = []
    for fname in ["elites.json", "reserves.json"]:
        data = load_json(run_dir / fname)
        if isinstance(data, list):
            genomes.extend(data)
    return genomes


def load_species_data(run_dir: Path) -> Dict[str, Any]:
    """Load speciation state and genomes for a run."""
    return {
        "speciation_state": load_json(run_dir / "speciation_state.json"),
        "elites": load_json(run_dir / "elites.json") or [],
        "reserves": load_json(run_dir / "reserves.json") or [],
    }


def get_embedding(genome: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract prompt embedding."""
    if not isinstance(genome, dict):
        return None
    emb = genome.get("prompt_embedding")
    if emb is not None and isinstance(emb, list) and len(emb) > 0:
        return np.array(emb, dtype=np.float32)
    return None


def is_valid_species_id(sid: Any) -> bool:
    """Check if species_id is valid (not 0, -1, None, 'unknown')."""
    if sid is None:
        return False
    sid_str = str(sid)
    return sid_str not in ("", "0", "-1", "None", "unknown")


def get_species_labels(run_dir: Path, species_id: str) -> List[str]:
    """Get c-TF-IDF labels for a species from speciation_state."""
    spec_state = load_json(run_dir / "speciation_state.json")
    if not spec_state or "species" not in spec_state:
        return []
    species_info = spec_state["species"].get(species_id, {})
    return species_info.get("labels", [])


# Build global color map after all helper functions are defined
GLOBAL_SPECIES_COLOR_MAP = build_global_species_color_map()

def get_species_color(run_id: str, species_id: str) -> Tuple[float, float, float]:
    """Get consistent color for a species (run_id, species_id)."""
    key = (run_id, str(species_id))
    return GLOBAL_SPECIES_COLOR_MAP.get(key, (0.5, 0.5, 0.5))  # Default gray if not found


# =============================================================================
# Section 3: Compute Cluster/Species Separation Metrics
# =============================================================================
def compute_species_separation(spec_state: Dict, elites: List[Dict]) -> Dict[str, Any]:
    """Compute intra and inter species distances for cluster metrics."""
    if not spec_state or "species" not in spec_state:
        return {"error": "No species data"}
    
    species_info = spec_state["species"]
    
    # Get centroids (leader embeddings)
    centroids = {}
    for sid, sdata in species_info.items():
        emb = sdata.get("leader_embedding")
        if emb and isinstance(emb, list):
            centroids[sid] = np.array(emb, dtype=np.float32)
    
    if len(centroids) < 2:
        return {"error": "Not enough species with centroids", "n_species": len(centroids)}
    
    # Compute inter-species distances (between centroids)
    species_ids = list(centroids.keys())
    centroid_matrix = np.array([centroids[sid] for sid in species_ids])
    inter_distances = cosine_distances(centroid_matrix)
    
    # Get upper triangle (excluding diagonal)
    upper_tri = inter_distances[np.triu_indices_from(inter_distances, k=1)]
    
    # Compute intra-species distances (within species)
    species_elites = defaultdict(list)
    for elite in elites:
        sid = str(elite.get("species_id", ""))
        emb = get_embedding(elite)
        if sid and emb is not None:
            species_elites[sid].append(emb)
    
    intra_distances = []
    for sid, embs in species_elites.items():
        if len(embs) >= 2:
            embs_arr = np.array(embs)
            dists = cosine_distances(embs_arr)
            upper = dists[np.triu_indices_from(dists, k=1)]
            intra_distances.extend(upper.tolist())
    
    mean_inter = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0.0
    mean_intra = float(np.mean(intra_distances)) if intra_distances else 0.0
    
    return {
        "n_species": len(species_ids),
        "species_ids": species_ids,
        "inter_distance_matrix": inter_distances,
        "mean_inter_distance": mean_inter,
        "mean_intra_distance": mean_intra,
        "separation_ratio": mean_inter / mean_intra if mean_intra > 0 else float('inf'),
    }


# =============================================================================
# Section 4: Compute Per-Species Toxicity Statistics
# =============================================================================
def compute_species_toxicity_stats(elites: List[Dict], reserves: List[Dict] = None) -> pd.DataFrame:
    """Compute toxicity statistics per species."""
    all_genomes = elites + (reserves or [])
    species_toxicities = defaultdict(list)
    
    for genome in all_genomes:
        sid = str(genome.get("species_id", "unknown"))
        if not is_valid_species_id(sid):
            continue
        tox = get_toxicity(genome)
        if tox is not None:
            species_toxicities[sid].append(tox)
    
    stats_list = []
    for sid, toxicities in species_toxicities.items():
        if toxicities:
            stats_list.append({
                "species_id": sid,
                "count": len(toxicities),
                "max_toxicity": np.max(toxicities),
                "median_toxicity": np.median(toxicities),
                "mean_toxicity": np.mean(toxicities),
                "std_toxicity": np.std(toxicities),
                "q95_toxicity": np.percentile(toxicities, 95),
            })
    
    df = pd.DataFrame(stats_list)
    if not df.empty:
        df = df.sort_values("max_toxicity", ascending=False)
    return df


# =============================================================================
# Section 5: Load Data and Compute Metrics for All Runs
# =============================================================================
print("\n" + "=" * 80)
print("LOADING DATA AND COMPUTING METRICS")
print("=" * 80)

# Store results for all runs
all_cluster_metrics = []
all_species_toxicity = {}

for run_dir in RUNS_SPECIATION:
    if not run_dir.exists():
        print(f"  {run_dir.name}: NOT FOUND - skipping")
        continue
    
    run_id = run_dir.name
    spec_data = load_species_data(run_dir)
    
    if not spec_data["speciation_state"]:
        print(f"  {run_id}: No speciation_state.json - skipping")
        continue
    
    # Compute separation metrics
    sep = compute_species_separation(spec_data["speciation_state"], spec_data["elites"])
    
    # Compute per-species toxicity stats
    df_tox = compute_species_toxicity_stats(spec_data["elites"], spec_data["reserves"])
    all_species_toxicity[run_id] = df_tox
    
    # Store cluster metrics
    all_cluster_metrics.append({
        "run_id": run_id,
        "n_species": sep.get("n_species", 0),
        "n_elites": len(spec_data["elites"]),
        "n_reserves": len(spec_data["reserves"]),
        "mean_inter_dist": sep.get("mean_inter_distance", np.nan),
        "mean_intra_dist": sep.get("mean_intra_distance", np.nan),
        "separation_ratio": sep.get("separation_ratio", np.nan),
    })
    
    print(f"  {run_id}: {sep.get('n_species', 0)} species, "
          f"separation ratio = {sep.get('separation_ratio', 0):.3f}")

df_cluster_metrics = pd.DataFrame(all_cluster_metrics)


# =============================================================================
# TABLE 1: Cluster Metrics Summary
# =============================================================================
print("\n" + "=" * 80)
print("TABLE 1: CLUSTER METRICS SUMMARY")
print("=" * 80)

print("\nCluster Metrics by Run:")
print(df_cluster_metrics.to_string(index=False))

# Aggregate statistics
print("\n--- Aggregate Statistics (across runs) ---")
print(f"Mean # Species: {df_cluster_metrics['n_species'].mean():.1f} ± {df_cluster_metrics['n_species'].std():.1f}")
print(f"Mean Separation Ratio: {df_cluster_metrics['separation_ratio'].mean():.3f}")
print(f"Mean Inter-species Distance: {df_cluster_metrics['mean_inter_dist'].mean():.4f}")
print(f"Mean Intra-species Distance: {df_cluster_metrics['mean_intra_dist'].mean():.4f}")

# Round numeric columns to 4 decimal places before saving
df_cluster_metrics_rounded = df_cluster_metrics.copy()
numeric_cols = ['mean_inter_dist', 'mean_intra_dist', 'separation_ratio']
for col in numeric_cols:
    if col in df_cluster_metrics_rounded.columns:
        df_cluster_metrics_rounded[col] = df_cluster_metrics_rounded[col].round(4)

# Save Table 1 with 4 decimal places formatting
df_cluster_metrics_rounded.to_csv(OUT / "rq2_cluster_metrics.csv", index=False, float_format='%.4f')
print(f"\nSaved: {OUT / 'rq2_cluster_metrics.csv'}")


# =============================================================================
# FIGURE 1: Toxicity Distribution by Species (Boxplot)
# =============================================================================
print("\n" + "=" * 80)
print("FIGURE 1: TOXICITY DISTRIBUTION BY SPECIES")
print("=" * 80)

# Build unified dataframe from all 5 runs, filtering for mature species only
print("Building unified dataset from all runs (mature species only)...")
unified_data = []
species_labels_full = {}  # (run_id, species_id) -> all 10 labels

for run_dir in RUNS_SPECIATION:
    if not run_dir.exists():
        continue
    
    run_id = run_dir.name
    spec_state = load_json(run_dir / "speciation_state.json")
    
    if not spec_state or "species" not in spec_state:
        continue
    
    # Get mature species (active + frozen) for this run
    mature_species_ids = set()
    species_labels_key_map = {}  # species_id -> tuple of all labels (sorted, for exact matching)
    species_labels_display_map = {}  # species_id -> list of all labels (for display)
    
    for sid, sdata in spec_state["species"].items():
        species_state = sdata.get("species_state", "")
        if species_state in ("active", "frozen"):
            # Get all labels (exactly 10 from c-TF-IDF)
            labels = sdata.get("labels", [])
            if labels and len(labels) >= 10:  # Only include species with all 10 labels
                # Normalize labels: strip and filter empty
                all_labels = [l.strip() for l in labels[:10] if l and l.strip()]
                if len(all_labels) == 10:  # Must have exactly 10 labels
                    mature_species_ids.add(str(sid))
                    # Use sorted tuple as key for exact matching
                    labels_key = tuple(sorted(all_labels))
                    species_labels_key_map[str(sid)] = labels_key
                    species_labels_display_map[str(sid)] = all_labels
                    species_labels_full[(run_id, str(sid))] = all_labels
    
    # Load genomes and filter for mature species
    elites = load_json(run_dir / "elites.json") or []
    reserves = load_json(run_dir / "reserves.json") or []
    
    for genome in elites + reserves:
        sid = str(genome.get("species_id", ""))
        if sid in mature_species_ids and sid in species_labels_key_map:
            tox = get_toxicity(genome)
            if tox is not None:
                # Use the exact labels tuple as the grouping key
                labels_key = species_labels_key_map[sid]
                unified_data.append({
                    "species_labels_key": labels_key,  # Tuple for exact matching
                    "species_id": sid,
                    "run_id": run_id,
                    "toxicity": tox
                })

if unified_data:
    unified_df = pd.DataFrame(unified_data)
    
    # Group by exact labels key (species with identical all 10 labels are combined)
    # Compute max toxicity per labels key to identify top species
    label_stats = unified_df.groupby("species_labels_key")["toxicity"].agg([
        ("max_toxicity", "max"),
        ("median_toxicity", "median"),
        ("mean_toxicity", "mean"),
        ("count", "count")
    ]).reset_index()
    label_stats = label_stats.sort_values("max_toxicity", ascending=False)
    
    # Get top 10 species groups by max toxicity
    top_10_labels_keys = label_stats.head(10)["species_labels_key"].tolist()
    
    print(f"  - Found {len(label_stats)} unique species (by exact label match) across all runs")
    print(f"  - Top 10 species by max toxicity")
    
    # Filter unified_df to only top 10 label keys
    plot_df = unified_df[unified_df["species_labels_key"].isin(top_10_labels_keys)].copy()
    
    # Create display label map: labels_key -> comma-separated labels string
    label_display_map = {}  # labels_key (tuple) -> display string
    for labels_key in top_10_labels_keys:
        # Get the labels from any species with this key (they're all the same)
        # Find first occurrence to get the labels
        sample_species = plot_df[plot_df["species_labels_key"] == labels_key].iloc[0]
        run_id = sample_species["run_id"]
        sid = str(sample_species["species_id"])
        key = (run_id, sid)
        if key in species_labels_full:
            all_labels = species_labels_full[key]
            # Display as comma-separated (exactly 10 labels)
            display_label = ', '.join(all_labels)
            label_display_map[labels_key] = display_label
        else:
            # Fallback: convert tuple back to list
            display_label = ', '.join(sorted(labels_key))
            label_display_map[labels_key] = display_label
    
    # Convert tuple keys to strings for plotting (seaborn can't handle tuples directly)
    plot_df["species_labels_key_str"] = plot_df["species_labels_key"].apply(lambda x: label_display_map.get(x, str(x)))
    
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Order by max toxicity (descending) - use string version for ordering
        order_str = plot_df.groupby("species_labels_key_str")["toxicity"].max().sort_values(ascending=False).index.tolist()
        
        # Create horizontal boxplot with jittered strip overlay
        # Boxplot shows median, quartiles, IQR explicitly (reviewer-safe, no KDE artifacts)
        # Stripplot shows individual genomes (exposes sample size and outliers)
        # Use consistent species colors: map each labels_key to a representative species color
        labels_key_to_color = {}
        for labels_key in top_10_labels_keys:
            # Get first (run_id, species_id) for this labels_key to get consistent color
            sample_species = plot_df[plot_df["species_labels_key"] == labels_key].iloc[0]
            run_id = sample_species["run_id"]
            sid = str(sample_species["species_id"])
            labels_key_to_color[labels_key] = get_species_color(run_id, sid)
        
        # Create palette from consistent colors - use original species colors for boxplots
        palette = [labels_key_to_color.get(lk, (0.5, 0.5, 0.5)) for lk in top_10_labels_keys]
        sns.boxplot(data=plot_df, y="species_labels_key_str", x="toxicity", order=order_str,
                    hue="species_labels_key_str", palette=palette, ax=ax, orient='h', 
                    linewidth=1.5, width=0.6, legend=False)
        sns.stripplot(data=plot_df, y="species_labels_key_str", x="toxicity", order=order_str,
                      color="black", alpha=0.5, size=3, ax=ax, jitter=0.3, orient='h', linewidth=0.5)
        
        # Set y-axis labels to show all 10 labels for each species group
        # Labels are black (default color)
        ytick_labels = order_str  # Already contains the comma-separated labels
        ax.set_yticks(range(len(order_str)))
        ax.set_yticklabels(ytick_labels, fontsize=10, color='black')
        
        # Fix axis formatting
        ax.set_xlabel("Toxicity Score", fontsize=14, fontweight='bold')
        ax.set_ylabel("Species (by semantic label)", fontsize=14, fontweight='bold')
        ax.set_title("Toxicity Distribution by Species", fontsize=16)
        
        # X-axis: set limits and format ticks
        ax.set_xlim(0, 1)
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=12)
        ax.tick_params(axis='x', labelsize=12)
        
        # Y-axis: format ticks
        ax.tick_params(axis='y', labelsize=10)
        
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig(OUT / "figures" / "fig1_toxicity_by_species.png", dpi=300, bbox_inches='tight')
        plt.savefig(OUT / "figures" / "fig1_toxicity_by_species.pdf", bbox_inches='tight')
        plt.close()
        print(f"Saved: fig1_toxicity_by_species.png/pdf")
        print(f"  - Top 10 species by max toxicity")
        print(f"  - {len(plot_df)} genomes from {unified_df['run_id'].nunique()} runs")
        print(f"  - Mature species only (active + frozen)")
    else:
        print("No data available for plotting after filtering")
else:
    print("No mature species data available for Figure 1")



# =============================================================================
# FIGURE 2: Mega Word Cloud (All Species Labels, Toxicity-Shaded)
# =============================================================================
print("\n" + "=" * 80)
print("FIGURE 2: MEGA WORD CLOUD (ALL SPECIES)")
print("=" * 80)

# Collect ALL species data from all runs (not grouped by labels_key)
print("Collecting all species data from all runs...")

# Track each individual species (run_id, species_id) with its labels and toxicity
all_species_data = []  # List of dicts: {run_id, species_id, labels, max_toxicity}

for run_dir in RUNS_SPECIATION:
    if not run_dir.exists():
        continue

    run_id = run_dir.name
    spec_state = load_json(run_dir / "speciation_state.json")

    if not spec_state or "species" not in spec_state:
        continue

    # Load genomes for this run to get toxicity
    spec_data = load_species_data(run_dir)
    
    # Collect toxicity by species_id
    species_tox = defaultdict(list)
    for genome in spec_data["elites"]:
        sid = genome.get("species_id")
        if sid is not None and sid != 0 and sid != -1:
            tox = get_toxicity(genome)
            if tox is not None:
                species_tox[str(sid)].append(float(tox))

    # Process each mature species
    for sid, sdata in spec_state["species"].items():
        species_state = sdata.get("species_state", "")
        if species_state in ("active", "frozen"):
            labels = sdata.get("labels", [])
            if labels and len(labels) > 0:
                # Get all valid labels
                all_labels = [l.strip() for l in labels if l and l.strip()]
                if len(all_labels) > 0:
                    # Get max toxicity for this species
                    tox_values = species_tox.get(str(sid), [])
                    max_tox = float(np.max(tox_values)) if tox_values else 0.0
                    
                    all_species_data.append({
                        "run_id": run_id,
                        "species_id": str(sid),
                        "labels": all_labels,
                        "max_toxicity": max_tox
                    })

print(f"\nCollected {len(all_species_data)} species from all runs")

# Build word cloud: include ALL labels from ALL species
# Current weighting: term_freq[word] = number of species whose top-10 labels include that word.
# Color = species that "owns" the term (if multiple, the one with higher max_toxicity).
# Shade = toxicity of that species (darker = higher).
#
# Alternatives for word size (you can replace how term_freq is built):
#   (1) Current: count of species containing the term (term_freq[term] += 1 per species).
#   (2) Uniform: term_freq[term] = 1 for all terms → all words same size (use relative_scaling=0).
#   (3) Toxicity-weighted: term_freq[term] = sum of max_toxicity over species that have it.
#   (4) Rank-weighted: weight by inverse rank within each species (e.g. 1/rank), then sum.
#   (5) Binary + toxicity: term_freq[term] = max(max_toxicity) over species that have it.
# Export: term frequencies are written to fig2_term_frequencies.csv below.

if HAS_WORDCLOUD and len(all_species_data) > 0:
    # Count frequency of each term: how many species include it in their top-10 labels
    term_freq = defaultdict(int)
    term_to_species = {}  # term -> (run_id, species_id, max_toxicity) - tracks which species owns it
    
    for sp_data in all_species_data:
        labels = sp_data["labels"]
        max_tox = sp_data["max_toxicity"]
        run_id = sp_data["run_id"]
        sid = sp_data["species_id"]
        
        for term in labels:
            term_freq[term] += 1
            # If term appears in multiple species, keep the one with higher toxicity
            if term not in term_to_species or max_tox > term_to_species[term][2]:
                term_to_species[term] = (run_id, sid, max_tox)
    
    print(f"  Total unique terms: {len(term_freq)}")
    print(f"  Total term occurrences: {sum(term_freq.values())}")
    
    # Save term frequencies and owner species for inspection
    # term_freq[word] = number of species whose top-10 labels include that word
    freq_path = OUT / "fig2_term_frequencies.csv"
    with open(freq_path, "w") as f:
        f.write("term,count,run_id,species_id,max_toxicity\n")
        for term in sorted(term_freq.keys(), key=lambda t: (-term_freq[t], t)):
            cnt = term_freq[term]
            run_id, sid, tox = term_to_species.get(term, ("", "", 0.0))
            f.write(f"{repr(term)},{cnt},{run_id},{sid},{tox}\n")
    print(f"  Saved term frequencies to {freq_path}")
    
    # Use global species color mapping for consistency across all figures
    species_to_color = {}  # (run_id, species_id) -> base RGB color
    for sp_data in all_species_data:
        key = (sp_data["run_id"], sp_data["species_id"])
        if key not in species_to_color:
            # Use global color mapping (deterministic, consistent across figures)
            species_to_color[key] = get_species_color(sp_data["run_id"], sp_data["species_id"])
    
    # Normalize toxicity range for shading
    all_tox_values = [sp_data["max_toxicity"] for sp_data in all_species_data]
    vmin = float(np.min(all_tox_values)) if all_tox_values else 0.0
    vmax = float(np.max(all_tox_values)) if all_tox_values else 1.0
    vrng = (vmax - vmin) if (vmax > vmin) else 1.0
    
    def shade_rgb(base_rgb, tox_value):
        """Shade RGB color based on toxicity (darker = higher toxicity)."""
        z = (float(tox_value) - vmin) / vrng
        z = max(0.0, min(1.0, z))
        # Brightness: higher toxicity = darker (lower brightness)
        # Range: 0.3 (dark) to 1.0 (bright)
        brightness = 1.0 - 0.7 * z  # Invert so high toxicity = dark
        r, g, b = base_rgb
        return (min(1.0, r * brightness), min(1.0, g * brightness), min(1.0, b * brightness))
    
    # Color function: assign color based on species (run_id, species_id), shade based on toxicity
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        if word not in term_to_species:
            return "rgb(0,0,0)"
        
        run_id, sid, max_tox = term_to_species[word]
        key = (run_id, sid)
        base_color = species_to_color.get(key, (0.2, 0.2, 0.2))
        r, g, b = shade_rgb(base_color, max_tox)
        return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
    
    # Create word cloud with term frequencies (formatting aligned with rq1 / rq2 fig4)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.axis('off')
    
    # Word cloud dimensions scaled to figure aspect
    wc = WordCloud(
        width=1100,
        height=700,
        background_color='white',
        max_words=None,
        prefer_horizontal=0.9,
        random_state=SEED,
        collocations=False,
        min_font_size=5,
        max_font_size=100
    ).generate_from_frequencies(term_freq)
    
    wc = wc.recolor(color_func=color_func, random_state=SEED)
    
    ax.imshow(wc, interpolation='bilinear')
    ax.set_title("Species labels (shade = toxicity)", fontsize=8, fontweight='bold', pad=6)
    
    total_labels_in_wc = len([w for w in wc.words_.keys()])
    ax.text(0.02, 0.02,
            f"{total_labels_in_wc} terms from {len(all_species_data)} species",
            transform=ax.transAxes, ha='left', fontsize=7,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, linewidth=0.5))
    
    plt.tight_layout(pad=0.2)
    plt.savefig(OUT / "figures" / "fig2_semantic_map_wordcloud.png", dpi=300, bbox_inches='tight')
    plt.savefig(OUT / "figures" / "fig2_semantic_map_wordcloud.pdf", bbox_inches='tight')
    plt.close()
    print("Saved: fig2_semantic_map_wordcloud.png/pdf")
    print(f"  - {len(all_species_data)} species included (verified)")
    print(f"  - {len(term_freq)} unique terms in vocabulary")
    print(f"  - {total_labels_in_wc} terms displayed in word cloud (ALL labels)")
    print(f"  - Total label occurrences: {sum(term_freq.values())}")
    print(f"  - Color = species, Shade = toxicity (darker = higher toxicity)")
    
    # Additional verification: show species breakdown
    species_by_run = defaultdict(int)
    for sp_data in all_species_data:
        species_by_run[sp_data["run_id"]] += 1
    print(f"\n  Species count by run:")
    for run_id in sorted(species_by_run.keys()):
        print(f"    {run_id}: {species_by_run[run_id]} species")
else:
    print("WordCloud not available or no species data")


# =============================================================================
# FIGURE 3: Force-Directed Graph (REMOVED)
# =============================================================================
# Figure 3 has been removed as requested


# =============================================================================
# FIGURE 4: Label-Based Force-Directed Graph (Leader-Based Approach)
# =============================================================================
print("\n" + "=" * 80)
print("FIGURE 4: LABEL-BASED SPECIES SIMILARITY GRAPH (LEADERS ONLY)")
print("=" * 80)

if HAS_NETWORKX:
    print("Creating embedding-based similarity graph with all genomes...")
    print("  - All genomes shown as dots (leaders, followers, outliers)")
    print("  - Leaders: normal visibility")
    print("  - Followers and outliers: very very faint")
    print("  - Species radius circles: very faint dotted lines")
    
    # Collect ALL genomes from all runs
    all_genome_data = []  # List of all genomes with metadata
    species_info = {}  # species_id -> {leader_id, leader_embedding, radius, labels, run_id}
    
    for run_dir in RUNS_SPECIATION:
        if not run_dir.exists():
            continue
        
        run_id = run_dir.name
        spec_state = load_json(run_dir / "speciation_state.json")
        
        if not spec_state or "species" not in spec_state:
            continue
        
        # Load all genomes
        elites = load_json(run_dir / "elites.json") or []
        reserves = load_json(run_dir / "reserves.json") or []
        all_genomes = elites + reserves
        
        # Create mapping: genome_id -> genome
        genome_map = {g.get("id"): g for g in all_genomes}
        
        # Process only active and frozen species
        # Leader selection: highest toxicity genome in each species
        for sid, sdata in spec_state["species"].items():
            species_state = sdata.get("species_state", "")
            if species_state not in ("active", "frozen"):
                continue  # Skip incubator, extinct, etc.
            
            sid_int = int(sid)
            member_ids = sdata.get("member_ids", [])
            
            # Find leader: highest toxicity genome in this species
            # Fix type mismatch: normalize both to string for comparison
            species_genomes = []
            for genome in all_genomes:
                gid = genome.get("id")
                g_sid = genome.get("species_id")
                # Fix: compare as strings to handle type mismatch (int vs str)
                if str(g_sid) == str(sid_int):
                    emb = get_embedding(genome)
                    tox = get_toxicity(genome)
                    if emb is not None and tox is not None:
                        species_genomes.append({
                            "genome_id": gid,
                            "embedding": emb,
                            "toxicity": float(tox),
                            "genome": genome
                        })
            
            if not species_genomes:
                continue  # Skip species with no valid genomes
            
            # Select leader: highest toxicity
            leader_genome_data = max(species_genomes, key=lambda g: g["toxicity"])
            leader_id = leader_genome_data["genome_id"]
            leader_embedding = leader_genome_data["embedding"]
            leader_toxicity = leader_genome_data["toxicity"]
            
            labels = sdata.get("labels", [])
            valid_labels = [l.strip() for l in labels if l and l.strip()] if labels else []
            
            # Store species info (all leaders have same radius)
            species_info[f"{run_id}_species_{sid}"] = {
                "run_id": run_id,
                "species_id": sid_int,
                "leader_id": leader_id,
                "leader_embedding": leader_embedding,
                "leader_toxicity": leader_toxicity,
                "radius": 0.2,  # Same radius for all species
                "member_ids": set(str(mid) for mid in member_ids) if member_ids else set(),
                "labels": valid_labels,
                "species_state": species_state
            }
        
        # Build leader IDs set for O(1) lookup (optimization)
        leader_ids_this_run = set()
        for sp_key, sp_data in species_info.items():
            if sp_data["run_id"] == run_id:
                leader_ids_this_run.add(str(sp_data["leader_id"]))
        
        # Collect ONLY leaders and outliers (reserves) - NO followers/members
        for genome in all_genomes:
            gid = genome.get("id")
            sid = genome.get("species_id")
            emb = get_embedding(genome)
            tox = get_toxicity(genome)
            
            if emb is not None and tox is not None:
                # Optimized: O(1) lookup instead of O(N_species) loop
                is_leader = (str(gid) in leader_ids_this_run)
                is_outlier = (sid == 0 or str(sid) == "0")
                
                # Only include leaders and outliers (reserves)
                if is_leader or is_outlier:
                    all_genome_data.append({
                        "genome_id": gid,
                        "run_id": run_id,
                        "species_id": sid if sid is not None else 0,
                        "embedding": emb,
                        "toxicity": float(tox),
                        "is_leader": is_leader,
                        "is_follower": False,  # Never include followers
                        "is_outlier": is_outlier
                    })
    
    print(f"  - Found {len(all_genome_data)} total genomes")
    print(f"  - Found {len(species_info)} species with radius info")
    
    leaders_count = sum(1 for g in all_genome_data if g["is_leader"])
    outliers_count = sum(1 for g in all_genome_data if g["is_outlier"])
    print(f"  - Leaders: {leaders_count} (active+frozen species, highest toxicity), Outliers (reserves): {outliers_count}")
    
    if len(all_genome_data) > 0 and len(species_info) > 0:
        # Project all embeddings to 2D using MDS
        print("  - Projecting embeddings to 2D using MDS...")
        all_embeddings = np.array([g["embedding"] for g in all_genome_data])
        
        # Use MDS for 2D projection
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=SEED, normalized_stress='auto')
        distances = cosine_distances(all_embeddings)
        positions_2d = mds.fit_transform(distances)
        
        # Store 2D positions
        for i, g in enumerate(all_genome_data):
            g["pos_2d"] = positions_2d[i]
        
        # For species leaders not found in all_genome_data, project their embeddings directly
        # This handles cases where leaders are archived or in other states
        for sp_key, sp_data in species_info.items():
            # Check if we already have this leader in all_genome_data
            leader_found = False
            for g in all_genome_data:
                if (g["run_id"] == sp_data["run_id"] and 
                    str(g["genome_id"]) == str(sp_data["leader_id"])):
                    leader_found = True
                    break
            
            if not leader_found and sp_data["leader_embedding"] is not None:
                # Project leader embedding to 2D using transform (if available) or fit_transform on extended set
                # For simplicity, compute distance to all existing points and use weighted average position
                leader_emb = sp_data["leader_embedding"]
                leader_distances = cosine_distances(leader_emb.reshape(1, -1), all_embeddings)[0]
                
                # Use inverse distance weighting to estimate position
                # Closer genomes contribute more to the position estimate
                weights = 1.0 / (leader_distances + 1e-6)  # Add small epsilon to avoid division by zero
                weights = weights / weights.sum()  # Normalize
                estimated_pos = np.average(positions_2d, axis=0, weights=weights)
                
                # Store estimated position for this species leader
                sp_data["estimated_pos_2d"] = estimated_pos
                sp_data["leader_not_in_data"] = True
        
        # Get leader positions for radius circles
        leader_positions = {}  # species_key -> (pos_2d, radius)
        for sp_key, sp_data in species_info.items():
            # Find leader genome in all_genome_data
            leader_genome = None
            for g in all_genome_data:
                if (g["run_id"] == sp_data["run_id"] and 
                    str(g["genome_id"]) == str(sp_data["leader_id"]) and 
                    g["is_leader"]):
                    leader_genome = g
                    break
            
            if leader_genome:
                leader_positions[sp_key] = {
                    "pos_2d": leader_genome["pos_2d"],
                    "radius": 0.2,  # Same radius for all species
                    "labels": sp_data["labels"],
                    "species_id": sp_data["species_id"]
                }
        
        print(f"  - Found {len(leader_positions)} species with leader positions for radius circles")
        
        if len(all_genome_data) > 0:
            # Formatting aligned with rq1 figures: compact size, spines, grid, labels
            fig, ax = plt.subplots(figsize=(5.5, 5.0))
            
            # Separate genomes by type (only leaders and outliers, no followers)
            leader_genomes = [g for g in all_genome_data if g["is_leader"]]
            outlier_genomes = [g for g in all_genome_data if g["is_outlier"]]
            
            # Get toxicity range for colormap
            all_toxicities = [g["toxicity"] for g in all_genome_data]
            min_tox = min(all_toxicities) if all_toxicities else 0.0
            max_tox = max(all_toxicities) if all_toxicities else 1.0
            tox_range = max_tox - min_tox if max_tox > min_tox else 1.0
            
            # Draw species radius circles first (so they appear behind dots)
            from matplotlib.patches import Circle
            
            # Compute scaling factor: radius is in cosine distance space (0-2), need to scale to 2D MDS space
            # Estimate scale by comparing average distances in both spaces
            if len(all_genome_data) > 1:
                # Sample some distances in embedding space
                sample_size = min(100, len(all_genome_data))
                sample_indices = np.random.choice(len(all_genome_data), sample_size, replace=False)
                sample_embeddings = all_embeddings[sample_indices]
                sample_distances_emb = cosine_distances(sample_embeddings)
                mean_dist_emb = np.mean(sample_distances_emb[np.triu_indices_from(sample_distances_emb, k=1)])
                
                # Corresponding distances in 2D space
                sample_positions_2d = positions_2d[sample_indices]
                from scipy.spatial.distance import pdist
                sample_distances_2d = pdist(sample_positions_2d)
                mean_dist_2d = np.mean(sample_distances_2d) if len(sample_distances_2d) > 0 else 1.0
                
                # Scale factor: how much MDS compressed/expanded distances
                scale_factor = mean_dist_2d / mean_dist_emb if mean_dist_emb > 0 else 0.5
            else:
                scale_factor = 0.5  # Default fallback
            
            for sp_key, sp_data in leader_positions.items():
                pos = sp_data["pos_2d"]
                radius_emb = 0.2  # Same radius for all species (as requested)
                
                # Convert radius from embedding space to 2D MDS space
                circle_radius = radius_emb * scale_factor
                
                circle = Circle(
                    pos, circle_radius,
                    fill=False,
                    edgecolor='gray',
                    linestyle='--',
                    linewidth=0.3,
                    alpha=0.15
                )
                ax.add_patch(circle)
            
            # Draw all genomes as dots
            # 1. Outliers (reserves): very very faint (small dots, color by species)
            if outlier_genomes:
                outlier_positions = np.array([g["pos_2d"] for g in outlier_genomes])
                # Color: based on species (consistent across all figures)
                outlier_colors = [get_species_color(g["run_id"], str(g["species_id"])) for g in outlier_genomes]
                scatter_outliers = ax.scatter(
                    outlier_positions[:, 0], outlier_positions[:, 1],
                    c=outlier_colors,
                    s=6,  # Small dots (scaled for compact figure)
                    alpha=0.05,  # Very very faint
                    edgecolors='none'
                )
            
            # 3. Leaders: normal visibility (size based on toxicity, color based on species)
            if leader_genomes:
                leader_positions_arr = np.array([g["pos_2d"] for g in leader_genomes])
                leader_toxicities = [g["toxicity"] for g in leader_genomes]
                # Size: based on toxicity (scaled for compact figure)
                leader_sizes = [18 + 72 * ((tox - min_tox) / tox_range) if tox_range > 0 else 36 
                               for tox in leader_toxicities]
                
                # Color: based on species (consistent across all figures)
                leader_colors = [get_species_color(g["run_id"], str(g["species_id"])) for g in leader_genomes]
                
                scatter_leaders = ax.scatter(
                    leader_positions_arr[:, 0], leader_positions_arr[:, 1],
                    c=leader_colors,
                    s=leader_sizes,
                    alpha=0.8,
                    edgecolors='black',
                    linewidths=0.15,
                )
            
            # No labels for now (as requested)
            
            # =====================================================================
            # Add Force-Directed Graph Layer: Label-Based Species Connections
            # ANCHORED TO MDS POSITIONS (no spring layout movement)
            # =====================================================================
            print("  - Creating anchored label-based graph overlay...")
            
            # Create networkx graph for species leaders based on common labels
            G_labels = nx.Graph()
            
            # Add nodes: one per species leader (anchored to MDS positions)
            leader_species_nodes = {}  # species_key -> node_id
            pos_anchored = {}  # node_id -> MDS position (fixed, no spring layout)
            
            for sp_key, sp_data in species_info.items():
                # Find leader genome position
                leader_genome = None
                for g in leader_genomes:
                    if (g["run_id"] == sp_data["run_id"] and 
                        str(g["genome_id"]) == str(sp_data["leader_id"])):
                        leader_genome = g
                        break
                
                # Include ALL species (even without labels or if leader not found in leader_genomes)
                # If leader not found in leader_genomes, use estimated position from embedding
                if leader_genome:
                    pos_2d = leader_genome["pos_2d"]
                    toxicity = leader_genome["toxicity"]
                elif sp_data.get("estimated_pos_2d") is not None:
                    # Leader not in data, but we estimated position from embedding
                    pos_2d = sp_data["estimated_pos_2d"]
                    # Get toxicity from stored leader_toxicity
                    toxicity = sp_data.get("leader_toxicity", 0.0)
                else:
                    # Skip if we can't find or estimate position
                    continue
                
                # Include species even if no labels (empty set)
                node_id = sp_key
                leader_species_nodes[sp_key] = node_id
                
                # Anchor position to MDS (no spring layout movement)
                pos_anchored[node_id] = pos_2d
                
                G_labels.add_node(
                    node_id,
                    species_id=sp_data["species_id"],
                    run_id=sp_data["run_id"],
                    labels=set(sp_data["labels"]) if sp_data["labels"] else set(),  # Use set for intersection, empty if no labels
                    labels_list=sp_data["labels"] if sp_data["labels"] else [],
                    leader_pos_2d=pos_2d,
                    toxicity=toxicity,
                    species_state=sp_data.get("species_state", "unknown")
                )
            
            # Add edges based on common labels (include ALL edges with at least 1 common label)
            # All species will be included in the diagram
            min_common_labels = 1  # Include edges with at least 1 common label
            species_list = list(leader_species_nodes.keys())
            edges_added = 0
            
            for i, sp_key1 in enumerate(species_list):
                for sp_key2 in species_list[i + 1:]:
                    node1 = G_labels.nodes[leader_species_nodes[sp_key1]]
                    node2 = G_labels.nodes[leader_species_nodes[sp_key2]]
                    
                    labels1 = node1["labels"]
                    labels2 = node2["labels"]
                    common_labels = labels1.intersection(labels2)
                    n_common = len(common_labels)
                    
                    if n_common >= min_common_labels:
                        # Cap at 5 common labels for visualization strength
                        weight = min(n_common, 5)
                        G_labels.add_edge(
                            leader_species_nodes[sp_key1],
                            leader_species_nodes[sp_key2],
                            weight=weight,
                            n_common=n_common,  # Store actual count for reference
                            common_labels=list(common_labels)
                        )
                        edges_added += 1
            
            print(f"  - Added {edges_added} edges (all species included, min {min_common_labels} common label, max strength at 5 labels)")
            
            # Draw graph using ANCHORED positions (no spring layout)
            if len(G_labels.nodes()) > 0:
                # Draw edges (label-based connections) - anchored to MDS positions
                # Edge width and opacity scale with number of common labels (capped at 5)
                if edges_added > 0:
                    edges = G_labels.edges(data=True)
                    
                    # Edge width and alpha: scale based on number of common labels (weight, capped at 5)
                    # Linear scaling from 1 to 5 common labels
                    # 1 common label = very thin (0.3 width, 0.15 alpha)
                    # 5 common labels = thick (2.5 width, 0.7 alpha)
                    min_weight = 1
                    max_weight = 5  # Cap at 5
                    
                    # Group edges by weight for batch drawing (thinnest first, thickest last)
                    edges_by_weight = {}
                    for (u, v, data) in edges:
                        w = data.get('weight', 1)  # Already capped at 5
                        if w not in edges_by_weight:
                            edges_by_weight[w] = []
                        edges_by_weight[w].append((u, v))
                    
                    # Draw edges with constant width, only opacity scales
                    for weight in sorted(edges_by_weight.keys()):
                        if edges_by_weight[weight]:
                            # Constant width for all edges (reduced thickness)
                            # width = 0.4  # Fixed thin thickness
                            # Alpha: scales from 0.15 (1 label) to 0.7 (5 labels)
                            alpha = 0.15 + (0.7 - 0.15) * ((weight - min_weight) / (max_weight - min_weight))
                            
                            # Create subgraph for this weight class
                            subgraph = G_labels.edge_subgraph(edges_by_weight[weight])
                            nx.draw_networkx_edges(
                                subgraph, pos_anchored,
                                width=0.25,
                                alpha=alpha,
                                edge_color='blue',
                                style='solid',
                                ax=ax
                            )
                
                # Draw nodes (species leaders) - anchored to MDS positions
                # Note: Leaders are already drawn in scatter plot, so we draw graph nodes
                # with a distinct style (blue border) to show they're part of the graph
                node_toxicities = [G_labels.nodes[n]["toxicity"] for n in G_labels.nodes()]
                node_sizes_fd = [45 + 95 * ((tox - min_tox) / tox_range) if tox_range > 0 else 70
                                for tox in node_toxicities]
                
                # Color: based on species (consistent across all figures)
                node_colors_fd = []
                for n in G_labels.nodes():
                    run_id = G_labels.nodes[n]["run_id"]
                    sid = str(G_labels.nodes[n]["species_id"])
                    node_colors_fd.append(get_species_color(run_id, sid))
                
                # Draw graph nodes with blue borders (overlay on existing scatter)
                nx.draw_networkx_nodes(
                    G_labels, pos_anchored,
                    node_color=node_colors_fd,
                    node_size=node_sizes_fd,
                    alpha=0.7,
                    edgecolors='blue',
                    linewidths=0.8,
                    ax=ax
                )
                
                # No labels for now (as requested)
            
            # Labels and ticks (match rq1 figures)
            ax.set_xlabel("MDS Dimension 1", fontsize=8, fontweight='bold')
            ax.set_ylabel("MDS Dimension 2", fontsize=8, fontweight='bold')
            ax.tick_params(axis='both', labelsize=7, width=0.8)
            ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.6)
            ax.set_axisbelow(True)
            for side in ['bottom', 'left']:
                ax.spines[side].set_linewidth(0.9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            plt.tight_layout(pad=0.2)
            plt.savefig(OUT / "figures" / "fig4_label_similarity_graph.png", dpi=300, bbox_inches='tight')
            plt.savefig(OUT / "figures" / "fig4_label_similarity_graph.pdf", bbox_inches='tight')
            plt.close()
            print("Saved: fig4_label_similarity_graph.png/pdf")
            print(f"  Base Layer (Embedding View):")
            print(f"    - {leaders_count} leader genomes (active+frozen, highest toxicity, normal visibility)")
            print(f"    - {outliers_count} reserve genomes (very faint, alpha=0.05)")
            print(f"    - {len(leader_positions)} species radius circles (same radius=0.2 for all, dotted)")
            print(f"  Overlay Layer (Label View - Anchored Graph):")
            print(f"    - {len(G_labels.nodes()) if len(G_labels.nodes()) > 0 else 0} species nodes (active+frozen, anchored to MDS)")
            print(f"    - {edges_added} label-based connections (all edges with ≥1 common label)")
            print(f"    - Edge opacity: scales linearly 1-5 labels (capped at 5, width fixed at 0.4)")
            print(f"      • 1 label: width=0.4, alpha=0.15 (very faint)")
            print(f"      • 5 labels: width=0.4, alpha=0.7 (opaque)")
            print(f"    - Nodes fixed to MDS positions (no spring layout)")
            print(f"    - No labels shown (as requested)")
        else:
            print("Could not create graph - insufficient connections")
    else:
        print("Could not create graph - insufficient species")
else:
    print("Skipping Figure 4 - NetworkX not available")


# =============================================================================
# Section 6: Save Summary Statistics
# =============================================================================
print("\n" + "=" * 80)
print("SAVING SUMMARY STATISTICS")
print("=" * 80)

# Combine all species toxicity data
all_species_tox_combined = pd.concat(
    [df.assign(run_id=run_id) for run_id, df in all_species_toxicity.items()],
    ignore_index=True
)
all_species_tox_combined.to_csv(OUT / "rq2_species_toxicity.csv", index=False)
print(f"Saved: {OUT / 'rq2_species_toxicity.csv'}")

# Summary statistics JSON
summary_stats = {
    "n_runs": len(df_cluster_metrics),
    "total_species": int(all_species_tox_combined['species_id'].nunique()),
    "mean_species_per_run": float(df_cluster_metrics['n_species'].mean()),
    "std_species_per_run": float(df_cluster_metrics['n_species'].std()),
    "mean_separation_ratio": float(df_cluster_metrics['separation_ratio'].mean()),
    "mean_inter_distance": float(df_cluster_metrics['mean_inter_dist'].mean()),
    "mean_intra_distance": float(df_cluster_metrics['mean_intra_dist'].mean()),
    "max_toxicity_overall": float(all_species_tox_combined['max_toxicity'].max()),
    "median_max_toxicity": float(all_species_tox_combined['max_toxicity'].median()),
}

with open(OUT / "rq2_stats.json", "w") as f:
    json.dump(summary_stats, f, indent=2)
print(f"Saved: {OUT / 'rq2_stats.json'}")


# =============================================================================
# Section 7: Final Summary
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("RQ2: SPECIES QUALITY AND TOXICITY ANALYSIS - COMPLETE")
    print("=" * 80)
    
    print(f"\nOutputs generated:")
    print(f"  1. Table: rq2_cluster_metrics.csv")
    print(f"  2. Figure 1: fig1_toxicity_by_species.png/pdf")
    print(f"  3. Figure 2: fig2_semantic_map_wordcloud.png/pdf")
    print(f"  4. Figure 4: fig4_label_similarity_graph.png/pdf (label-based)")
    
    print(f"\nKey Findings:")
    print(f"  - Analyzed {len(df_cluster_metrics)} speciated runs")
    print(f"  - Mean species per run: {df_cluster_metrics['n_species'].mean():.1f}")
    print(f"  - Mean separation ratio: {df_cluster_metrics['separation_ratio'].mean():.3f}")
    print(f"    (> 1 indicates species are well-separated in semantic space)")
    print(f"  - Max toxicity achieved: {all_species_tox_combined['max_toxicity'].max():.4f}")
    
    print("\n" + "=" * 80)
