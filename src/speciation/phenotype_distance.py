

import numpy as np
from typing import Optional

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()

PHENOTYPE_SCORE_ORDER = [
    'toxicity',
    'threat',
    'profanity',
    'sexually_explicit',
    'identity_attack',
    'flirtation',
    'insult',
    'severe_toxicity'
]



def extract_phenotype_vector(genome: dict, logger=None) -> Optional[np.ndarray]:
    

    if logger is None:
        logger = get_logger("PhenotypeDistance")

    if not genome or "moderation_result" not in genome:
        return None
    
    moderation_result = genome.get("moderation_result")
    if not moderation_result or "google" not in moderation_result:
        return None
    
    google_scores = moderation_result.get("google", {})
    if not isinstance(google_scores, dict) or "scores" not in google_scores:
        return None
    
    scores = google_scores.get("scores", {})
    if not isinstance(scores, dict):
        return None
    
    phenotype = np.array([
        float(scores.get(score_name, 0.0))
        for score_name in PHENOTYPE_SCORE_ORDER
    ], dtype=np.float32)
    
    if not np.all((phenotype >= 0.0) & (phenotype <= 1.0)):
        invalid_indices = np.where((phenotype < 0.0) | (phenotype > 1.0))[0]
        logger.warning(f"Phenotype scores out of [0,1] range: indices {invalid_indices}")
        phenotype = np.clip(phenotype, 0.0, 1.0)
    
    return phenotype


def phenotype_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    
    if p1 is None or p2 is None:
        return 1.0
    
    p1 = np.array(p1, dtype=np.float32)
    p2 = np.array(p2, dtype=np.float32)
    
    diff = p1 - p2
    euclidean_dist = np.linalg.norm(diff)
    
    max_distance = np.sqrt(len(p1))
    normalized_dist = min(euclidean_dist / max_distance, 1.0)
    
    return float(normalized_dist)


def phenotype_distances_batch(
    query_phenotype: np.ndarray,
    phenotypes: np.ndarray
) -> np.ndarray:
    
    if query_phenotype is None:
        if phenotypes.ndim == 1:
            return np.array([1.0])
        return np.ones(len(phenotypes))
    
    if phenotypes.ndim == 1:
        phenotypes = phenotypes.reshape(1, -1)
    
    diff = phenotypes - query_phenotype
    euclidean_dists = np.linalg.norm(diff, axis=1)
    
    max_distance = np.sqrt(query_phenotype.shape[0])
    normalized_dists = np.clip(euclidean_dists / max_distance, 0.0, 1.0)
    
    return normalized_dists


__all__ = [
    "extract_phenotype_vector",
    "phenotype_distance",
    "phenotype_distances_batch",
    "PHENOTYPE_SCORE_ORDER"
]
