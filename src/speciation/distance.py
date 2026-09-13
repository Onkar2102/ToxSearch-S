
import numpy as np
from typing import Union, Optional, List

from .phenotype_distance import phenotype_distance, phenotype_distances_batch


def semantic_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    """Cosine distance on L2-normalized embeddings: ``1 - clip(dot)``.

    Not a classical metric (triangle inequality fails); see ``verfier/AUDIT.md``
    for the proved 2-relaxed triangle inequality.
    """
    norm_e1 = np.linalg.norm(e1)
    norm_e2 = np.linalg.norm(e2)
    if not (np.isclose(norm_e1, 1.0) and np.isclose(norm_e2, 1.0)):
        raise ValueError(f"Embeddings must be L2-normalized. Got norms: {norm_e1}, {norm_e2}")
    
    cosine_similarity = np.dot(e1, e2)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return float(1.0 - cosine_similarity)


def semantic_distances_batch(query_embedding: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    """Batch cosine distances; same unit-norm precondition as ``semantic_distance``."""
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)

    query_norm = np.linalg.norm(query_embedding)
    if not np.isclose(query_norm, 1.0):
        raise ValueError(f"Query embedding must be L2-normalized. Got norm: {query_norm}")
    embedding_norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(embedding_norms, 1.0):
        bad = np.where(~np.isclose(embedding_norms, 1.0))[0]
        raise ValueError(
            f"Embeddings must be L2-normalized. Non-unit norms at indices {bad.tolist()}: "
            f"{embedding_norms[bad].tolist()}"
        )
    
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
    """Weighted genotype + phenotype distance in ``[0, 1]``.

    Missing phenotypes are treated as maximal phenotype distance ``1.0``
    (aligned with ``phenotype_distance(None, ...)`` and ``verfier`` formal model).
    """
    if abs(w_genotype + w_phenotype - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got w_genotype={w_genotype}, w_phenotype={w_phenotype}")
    
    d_genotype = semantic_distance(e1, e2)
    
    d_genotype_norm = d_genotype / 2.0
    
    if p1 is not None and p2 is not None:
        d_phenotype = phenotype_distance(p1, p2)
    else:
        d_phenotype = 1.0
    
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
    """Batch ensemble distances; missing phenotypes contribute distance ``1.0``."""
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
            # Missing entries → maximal phenotype distance 1.0
            d_phenotype = np.full(num_targets, 1.0)
            valid_phenotypes = []
            valid_indices = []
            for i, p in enumerate(phenotypes):
                if p is not None:
                    valid_phenotypes.append(p)
                    valid_indices.append(i)
            
            if valid_phenotypes:
                phenotypes_array = np.array(valid_phenotypes)
                d_phenotype_valid = phenotype_distances_batch(query_phenotype, phenotypes_array)
                for idx, orig_idx in enumerate(valid_indices):
                    d_phenotype[orig_idx] = d_phenotype_valid[idx]
    else:
        d_phenotype = np.full(num_targets, 1.0)
    
    d_ensemble = w_genotype * d_genotype_norm + w_phenotype * d_phenotype
    return d_ensemble
