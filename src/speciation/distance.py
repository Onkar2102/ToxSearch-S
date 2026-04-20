

import numpy as np
from typing import Union, Optional, List

from .phenotype_distance import phenotype_distance


def semantic_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    if not (np.isclose(norm_e1, 1.0) and np.isclose(norm_e2, 1.0)):
        raise ValueError(f"Embeddings must be L2-normalized. Got norms: {norm_e1}, {norm_e2}")
    
    cosine_similarity = np.dot(e1, e2)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def semantic_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    
    cosine_similarities = embeddings @ query_embedding
    cosine_similarities = np.clip(cosine_similarities, -1.0, 1.0)
    return 1.0 - cosine_similarities


def ensemble_distance(
    e1: np.ndarray,
    e2: np.ndarray,
    p1: Optional[np.ndarray] = None,
    p2: Optional[np.ndarray] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3
) -> float:
    
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}")
    
    d_genotype = semantic_distance(e1, e2)
    
    d_genotype_norm = d_genotype / 2.0
    
    if p1 is not None and p2 is not None:
        d_phenotype = phenotype_distance(p1, p2)
    else:
        d_phenotype = 0.0
    
    d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
    
    return float(d_ensemble)


def ensemble_distances_batch(
    query_embedding: np.ndarray,
    embeddings: np.ndarray,
    query_phenotype: Optional[np.ndarray] = None,
    phenotypes: Optional[Union[np.ndarray, List[Optional[np.ndarray]]]] = None,
    w_genotype: float = 0.7,
    w_phenotype: float = 0.3
) -> np.ndarray:
    
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}")
    
    if embeddings.ndim == 1:
        num_targets = 1
        embeddings_2d = embeddings.reshape(1, -1)
    else:
        num_targets = len(embeddings)
        embeddings_2d = embeddings
    
    d_genotype = semantic_distances_batch(query_embedding, embeddings_2d)
    
    d_genotype_norm = d_genotype / 2.0
    
    if query_phenotype is not None and phenotypes is not None:
        if isinstance(phenotypes, np.ndarray):
            if phenotypes.ndim == 1:
                phenotypes_array = phenotypes.reshape(1, -1)
                d_phenotype = phenotype_distances_batch(query_phenotype, phenotypes_array)
            else:
                d_phenotype = phenotype_distances_batch(query_phenotype, phenotypes)
        else:
            valid_phenotypes = []
            valid_indices = []
            for i, p in enumerate(phenotypes):
                if p is not None:
                    valid_phenotypes.append(p)
                    valid_indices.append(i)
            
            if valid_phenotypes:
                from .phenotype_distance import phenotype_distances_batch
                phenotypes_array = np.array(valid_phenotypes)
                d_phenotype_valid = phenotype_distances_batch(query_phenotype, phenotypes_array)
                d_phenotype = np.full(num_targets, 0.0)
                for idx, orig_idx in enumerate(valid_indices):
                    d_phenotype[orig_idx] = d_phenotype_valid[idx]
            else:
                d_phenotype = np.full(num_targets, 0.0)
    else:
        d_phenotype = np.full(num_targets, 0.0)
    
    d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
    return d_ensemble
