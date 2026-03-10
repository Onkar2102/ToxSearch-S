"""
reserve_selection.py

NSGA-II based selection for Cluster 0 (reserves) capacity enforcement.

When cluster 0 exceeds max_capacity, this module selects which genomes to keep
using NSGA-II with two objectives (both maximized):
  1. Diversity (mean ensemble distance to species leaders) — primary priority
  2. Toxicity (fitness / north-star score) — secondary priority

Tie-break when trimming the last Pareto front:
  crowding distance (desc) → diversity (desc) → toxicity (desc)
"""

import numpy as np
from typing import List, Dict, Any, Tuple


def _dominates(a_div: float, a_tox: float, b_div: float, b_tox: float) -> bool:
    """Return True if (a_div, a_tox) Pareto-dominates (b_div, b_tox) (both maximised)."""
    return (a_div >= b_div and a_tox >= b_tox) and (a_div > b_div or a_tox > b_tox)


def _non_dominated_sort(diversity: np.ndarray, toxicity: np.ndarray) -> List[List[int]]:
    """Partition indices into successive Pareto fronts (front 0 = non-dominated)."""
    n = len(diversity)
    fronts: List[List[int]] = []
    remaining = list(range(n))
    while remaining:
        front: List[int] = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                if _dominates(diversity[j], toxicity[j], diversity[i], toxicity[i]):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        fronts.append(front)
        remaining_set = set(remaining) - set(front)
        remaining = [r for r in remaining if r in remaining_set]
    return fronts


def _crowding_distance(obj1: np.ndarray, obj2: np.ndarray) -> np.ndarray:
    """Compute crowding distance for a set of points in 2-objective space."""
    n = len(obj1)
    if n <= 2:
        return np.full(n, np.inf)
    cd = np.zeros(n)
    for arr in (obj1, obj2):
        order = np.argsort(arr)
        cd[order[0]] = np.inf
        cd[order[-1]] = np.inf
        r = float(np.max(arr) - np.min(arr))
        if r < 1e-12:
            r = 1.0
        for k in range(1, n - 1):
            cd[order[k]] += (arr[order[k + 1]] - arr[order[k - 1]]) / r
    return cd


def select_reserves_nsga2(
    genomes: List[Dict[str, Any]],
    toxicity_vals: np.ndarray,
    diversity_vals: np.ndarray,
    capacity: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Select which cluster-0 genomes to keep using NSGA-II.

    Objectives (both maximised): diversity (primary), toxicity (secondary).
    Tie-break when trimming the last front:
        crowding (desc) → diversity (desc) → toxicity (desc)

    Args:
        genomes: List of genome dicts (length N, N > capacity).
        toxicity_vals: 1-D array of toxicity scores aligned with *genomes*.
        diversity_vals: 1-D array of diversity scores aligned with *genomes*.
                        NaN values are treated as 0.0.
        capacity: Number of genomes to keep.

    Returns:
        (keep_genomes, excess_genomes) — two disjoint lists of genome dicts.
    """
    n = len(genomes)
    if n <= capacity:
        return list(genomes), []

    div = np.array(diversity_vals, dtype=np.float64)
    tox = np.array(toxicity_vals, dtype=np.float64)
    np.nan_to_num(div, copy=False, nan=0.0)

    fronts = _non_dominated_sort(div, tox)

    selected_indices: List[int] = []
    for front in fronts:
        if len(selected_indices) + len(front) <= capacity:
            selected_indices.extend(front)
        else:
            need = capacity - len(selected_indices)
            front_arr = np.array(front)
            cd = _crowding_distance(div[front_arr], tox[front_arr])
            # Tie-break: diversity first, then toxicity (lexsort keys are last-to-first)
            order = np.lexsort((-tox[front_arr], -div[front_arr], -cd))
            for k in order[:need]:
                selected_indices.append(front_arr[k])
            break

    keep_set = set(selected_indices)
    keep_genomes = [genomes[i] for i in selected_indices]
    excess_genomes = [genomes[i] for i in range(n) if i not in keep_set]
    return keep_genomes, excess_genomes
