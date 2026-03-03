"""
Speciation module for Dynamic Islands framework.

Implements Leader-Follower clustering with semantic embeddings for maintaining
diverse species (islands) that evolve independently.

Key features:
- Species limited to top 100 genomes by fitness
- Cluster 0 (reserves) holds all non-elite genomes (max 1000)
- Constant radius (theta_sim) for all species - no dynamic adjustment
- Cluster origin tracking (merge/natural) in speciation_state.json
- reserves.json stores Cluster 0 individuals (replaces legacy limbo.json)
"""

# Configuration
from .config import SpeciationConfig

# Data structures
from .species import Individual, Species, generate_species_id, SpeciesIdGenerator

# Embeddings
from .embeddings import (
    EmbeddingModel, compute_and_save_embeddings, remove_embeddings_from_temp, get_embedding_model,
    backfill_embeddings_for_genomes,
)

# Distance functions
from .distance import (
    semantic_distance, semantic_distances_batch,
    ensemble_distance, ensemble_distances_batch
)
from .phenotype_distance import (
    extract_phenotype_vector,
    phenotype_distance,
    phenotype_distances_batch,
    PHENOTYPE_SCORE_ORDER
)

# Clustering
from .leader_follower import (
    leader_follower_clustering
)
from .gen0_clustering import Gen0Clustering

# Cluster 0 (reserves)
from .reserves import (
    Cluster0, Cluster0Individual, CLUSTER_0_ID
)



# Merging
from .merging import process_merges

# Extinction (process_extinctions not used by main flow; Phase 5 in run_speciation is in-line)
# Metrics
from .metrics import (
    GenerationMetrics, SpeciationMetricsTracker, compute_diversity_metrics,
    get_species_statistics, log_generation_summary
)

# Labeling (c-TF-IDF based species characterization)
from .labeling import (
    extract_species_labels, update_species_labels
)

# Main entry point (similar to run_evolution)
from .run_speciation import (
    run_speciation,
    reset_speciation_module,
    get_speciation_statistics,
    update_evolution_tracker_with_speciation,
    process_generation,
    phase8_redistribute_genomes,
)

__all__ = [
    # Main classes
    "SpeciationConfig",
    "Individual", "Species", "generate_species_id", "SpeciesIdGenerator",
    
    # Embeddings
    "EmbeddingModel", "compute_and_save_embeddings", "remove_embeddings_from_temp", "get_embedding_model",
    "backfill_embeddings_for_genomes",
    
    # Distance
    "semantic_distance", "semantic_distances_batch",
    "ensemble_distance", "ensemble_distances_batch",
    "extract_phenotype_vector", "phenotype_distance", "phenotype_distances_batch", "PHENOTYPE_SCORE_ORDER",
    
    # Clustering
    "leader_follower_clustering",
    "Gen0Clustering",
    
    # Cluster 0
    "Cluster0", "Cluster0Individual", "CLUSTER_0_ID",
    
    # Merging
    "process_merges",
    
    # Metrics
    "GenerationMetrics", "SpeciationMetricsTracker", "compute_diversity_metrics",
    "get_species_statistics", "log_generation_summary",
    
    # Labeling
    "extract_species_labels", "update_species_labels",
    
    # Main entry point
    "run_speciation",
    "reset_speciation_module",
    "get_speciation_statistics",
    "update_evolution_tracker_with_speciation",
    "process_generation",
    "phase8_redistribute_genomes",
]
