from dataclasses import dataclass
from typing import Optional

from .distance import DISTANCE_METHODS, normalize_distance_method


@dataclass
class SpeciationConfig:
    """Configuration parameters for the speciation framework."""

    theta_sim: float = 0.25
    theta_merge: float = 0.1
    min_stability_gens: int = 5

    species_capacity: int = 100
    min_island_size: int = 2
    species_stagnation: int = 20

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # Weights for embedding+objective blend
    w_genotype: float = 0.7
    w_phenotype: float = 0.3

    # Weight for nli+embedding hybrid: d_H = alpha * d_E + (1-alpha) * d_N
    distance_alpha: float = 0.7

    # Clustering: leader_follower (default) or dbscan (= incremental DBSCAN + temp.json D_t)
    clustering_method: str = "leader_follower"
    dbscan_eps: Optional[float] = None  # defaults to theta_sim when None
    dbscan_min_samples: int = 2
    # Periodic batch-DBSCAN partition check for IncDBSCAN (0 = disabled)
    inc_dbscan_validate_every: int = 0

    # Distance used for membership / merge / DBSCAN matrix
    distance_method: str = "embedding"

    def __post_init__(self):
        assert 0 <= self.theta_sim <= 1, f"theta_sim must be in [0, 1]"
        assert 0 <= self.theta_merge <= 1, f"theta_merge must be in [0, 1]"
        assert self.theta_merge <= self.theta_sim, (
            f"theta_merge ({self.theta_merge}) must be <= theta_sim ({self.theta_sim})"
        )
        assert self.min_stability_gens >= 0, "min_stability_gens must be non-negative"

        assert abs(self.w_genotype + self.w_phenotype - 1.0) < 1e-6, (
            f"Ensemble weights must sum to 1.0, got w_genotype={self.w_genotype}, "
            f"w_phenotype={self.w_phenotype}"
        )
        assert 0.0 <= self.distance_alpha <= 1.0, (
            f"distance_alpha must be in [0, 1], got {self.distance_alpha}"
        )

        assert self.species_capacity > 0, "species_capacity must be positive"

        method = (self.clustering_method or "leader_follower").strip().lower().replace("-", "_")
        if method in ("lf",):
            method = "leader_follower"
        assert method in ("leader_follower", "dbscan"), (
            f"clustering_method must be 'leader_follower' or 'dbscan', got '{self.clustering_method}'"
        )
        self.clustering_method = method

        if self.dbscan_eps is not None:
            assert 0 <= self.dbscan_eps <= 1, f"dbscan_eps must be in [0, 1], got {self.dbscan_eps}"
        assert self.dbscan_min_samples >= 1, "dbscan_min_samples must be >= 1"
        assert self.inc_dbscan_validate_every >= 0, "inc_dbscan_validate_every must be >= 0"

        self.distance_method = normalize_distance_method(self.distance_method)
        assert self.distance_method in DISTANCE_METHODS

    def to_dict(self) -> dict:
        return {
            "theta_sim": self.theta_sim,
            "theta_merge": self.theta_merge,
            "min_stability_gens": self.min_stability_gens,
            "species_capacity": self.species_capacity,
            "min_island_size": self.min_island_size,
            "species_stagnation": self.species_stagnation,
            "embedding_model": self.embedding_model,
            "embedding_dim": self.embedding_dim,
            "embedding_batch_size": self.embedding_batch_size,
            "w_genotype": self.w_genotype,
            "w_phenotype": self.w_phenotype,
            "distance_alpha": self.distance_alpha,
            "clustering_method": self.clustering_method,
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "inc_dbscan_validate_every": self.inc_dbscan_validate_every,
            "distance_method": self.distance_method,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SpeciationConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


DEFAULT_CONFIG = SpeciationConfig()
