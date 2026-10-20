from dataclasses import dataclass
from typing import Optional

from .distance import (
    DISTANCE_METHODS,
    apply_distance_runtime_config,
    normalize_distance_method,
)


@dataclass
class SpeciationConfig:
    """Configuration parameters for the speciation framework."""

    theta_sim: float = 0.25
    theta_merge: float = 0.1
    min_stability_gens: int = 5

    species_capacity: int = 100
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

    # Distance used for membership / merge / DBSCAN matrix
    distance_method: str = "embedding"
    # HuggingFace MNLI model for distance_method=nli / nli+embedding
    nli_model: str = "cross-encoder/nli-deberta-v3-small"
    # SQLite file for shared directional NLI pair scores (LF and IncDBSCAN).
    nli_cache_path: Optional[str] = None

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

        self.distance_method = normalize_distance_method(self.distance_method)
        assert self.distance_method in DISTANCE_METHODS
        if not (self.nli_model or "").strip():
            self.nli_model = "cross-encoder/nli-deberta-v3-small"
        # Ensure MNLI scorer picks up --nli-model for nli / nli+embedding ablations.
        apply_distance_runtime_config(
            distance_method=self.distance_method,
            nli_model=self.nli_model,
            nli_cache_path=self.nli_cache_path,
        )

    def to_dict(self) -> dict:
        return {
            "theta_sim": self.theta_sim,
            "theta_merge": self.theta_merge,
            "min_stability_gens": self.min_stability_gens,
            "species_capacity": self.species_capacity,
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
            "distance_method": self.distance_method,
            "nli_model": self.nli_model,
            "nli_cache_path": self.nli_cache_path,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "SpeciationConfig":
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


DEFAULT_CONFIG = SpeciationConfig()
