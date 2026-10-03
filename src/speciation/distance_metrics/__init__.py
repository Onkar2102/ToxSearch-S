"""Atomic distance metrics: embedding, objective, and NLI.

Selected and combined by ``speciation.distance`` (the controller).
"""

from .embedding import (
    embedding_distance,
    embedding_distances_batch,
    pairwise_embedding_matrix,
    semantic_distance,
    semantic_distances_batch,
    unit_normalize_rows,
)
from .objective import (
    PHENOTYPE_SCORE_ORDER,
    extract_objective_vector,
    extract_phenotype_vector,
    objective_distance,
    objective_distances_batch,
    pairwise_objective_matrix,
    phenotype_distance,
    phenotype_distances_batch,
)
from .nli import (
    clear_nli_cache,
    entailment_probability,
    nli_distance,
    nli_distances_batch,
    pairwise_nli_matrix,
    set_entailment_scorer,
)

__all__ = [
    "semantic_distance",
    "semantic_distances_batch",
    "embedding_distance",
    "embedding_distances_batch",
    "pairwise_embedding_matrix",
    "unit_normalize_rows",
    "PHENOTYPE_SCORE_ORDER",
    "extract_phenotype_vector",
    "extract_objective_vector",
    "objective_distance",
    "objective_distances_batch",
    "pairwise_objective_matrix",
    "phenotype_distance",
    "phenotype_distances_batch",
    "clear_nli_cache",
    "entailment_probability",
    "nli_distance",
    "nli_distances_batch",
    "pairwise_nli_matrix",
    "set_entailment_scorer",
]
