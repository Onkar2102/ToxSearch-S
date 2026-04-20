

from dataclasses import dataclass


@dataclass
class SpeciationConfig:
    """Configuration parameters for speciation framework. Attributes: # Clustering Parameters theta_sim: Similarity threshold for species assignment (ensemble distance). Individuals within this distance of a leader become followers. Range: [0, 1] where 0 = identical, 1 = maximally different. Default: 0.25 (species radius - moderate similarity) Also used as the constant radius for all species. theta_merge: Merge threshold for combining similar species. Species with leader distance < theta_merge are candidates for merging. Must be <= theta_sim (typically < theta_sim for effective merging). Default: 0.1 (stricter threshold - only very similar species merge) min_stability_gens: Minimum age (generations) for a species to be mergeable. Both species must satisfy (current_gen - created_at) >= min_stability_gens. Default: 5 (species must exist at least 5 generations before merging) # Cluster 0 Parameters cluster0_min_cluster_size: Minimum cluster size required for cluster 0 speciation. When cluster 0 individuals form a cohesive cluster of this size, they can create a new species. Default: 2 (minimum viable species) cluster0_max_capacity: Maximum individuals in cluster 0. When exceeded, excess genomes are archived. Default: 1000 individuals cluster0_selection: Selection strategy when cluster 0 exceeds capacity. "nsga2" — NSGA-II with diversity first, then toxicity (Pareto fronts + crowding; tie-break: diversity desc, toxicity desc). Preserves diverse outliers. "toxicity_only" — legacy: sort by toxicity desc, keep top N. Default: "nsga2" # Species Management Parameters species_capacity: Maximum individuals per species (keeps top by fitness). When exceeded, lowest-fitness members are archived. Default: 100 individuals min_island_size: Minimum island size before extinction. Islands smaller than this are considered extinct. Default: 2 (minimum viable population) species_stagnation: Maximum generations without improvement before species extinction. Species that stagnate beyond this threshold are extinguished. Default: 20 generations # Embedding Parameters embedding_model: Sentence-transformer model name for prompt embeddings. Model must be compatible with sentence-transformers library. Default: "all-MiniLM-L6-v2" (384-dim, fast, high quality) embedding_dim: Expected embedding dimensionality. Should match the chosen model's output dimension. Default: 384 (for all-MiniLM-L6-v2) embedding_batch_size: Batch size for embedding computation. Larger batches are faster but use more memory. Default: 64 prompts per batch""" 
    theta_sim: float = 0.25
    theta_merge: float = 0.1
    min_stability_gens: int = 5

    cluster0_min_cluster_size: int = 2
    cluster0_max_capacity: int = 1000
    cluster0_selection: str = "nsga2"
    
    species_capacity: int = 100
    min_island_size: int = 2
    species_stagnation: int = 20
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64
    
    w_genotype: float = 0.7
    w_phenotype: float = 0.3
    
    def __post_init__(self):
        
        assert 0 <= self.theta_sim <= 1, f"theta_sim must be in [0, 1] for ensemble distance"
        assert 0 <= self.theta_merge <= 1, f"theta_merge must be in [0, 1] for ensemble distance"
        assert self.theta_merge <= self.theta_sim, \
            f"theta_merge ({self.theta_merge}) must be <= theta_sim ({self.theta_sim})"
        assert self.min_stability_gens >= 0, "min_stability_gens must be non-negative"

        assert abs(self.w_genotype + self.w_phenotype - 1.0) < 1e-6, \
            f"Ensemble weights must sum to 1.0, got w_genotype={self.w_genotype}, w_phenotype={self.w_phenotype}"
        
        assert self.cluster0_max_capacity > 0, "cluster0_max_capacity must be positive"
        assert self.cluster0_selection in ("nsga2", "toxicity_only"), \
            f"cluster0_selection must be 'nsga2' or 'toxicity_only', got '{self.cluster0_selection}'"
        
        assert self.species_capacity > 0, "species_capacity must be positive"
    
    def to_dict(self) -> dict:
        
        return {
            "theta_sim": self.theta_sim,
            "theta_merge": self.theta_merge,
            "min_stability_gens": self.min_stability_gens,
            "cluster0_min_cluster_size": self.cluster0_min_cluster_size,
            "cluster0_max_capacity": self.cluster0_max_capacity,
            "cluster0_selection": self.cluster0_selection,
            "species_capacity": self.species_capacity,
            "min_island_size": self.min_island_size,
            "species_stagnation": self.species_stagnation,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "w_genotype": self.w_genotype,
            "w_phenotype": self.w_phenotype,
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "SpeciationConfig":
        
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})

DEFAULT_CONFIG = SpeciationConfig()
