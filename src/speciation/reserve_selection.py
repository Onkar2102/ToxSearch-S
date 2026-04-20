

import numpy as np
from typing import List, Dict, Any, Tuple


def _dominates(a_div: float, a_tox: float, b_div: float, b_tox: float) -> bool:
    
    return (a_div >= b_div and a_tox >= b_tox) and (a_div > b_div or a_tox > b_tox)


def _non_dominated_sort(diversity: np.ndarray, toxicity: np.ndarray) -> List[List[int]]:
    
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
        remaining = list(set(remaining) - set(front))
    return fronts


def _crowding_distance(obj1: np.ndarray, obj2: np.ndarray) -> np.ndarray:
    
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
            order = np.lexsort((-tox[front_arr], -div[front_arr], -cd))
            for k in order[:need]:
                selected_indices.append(front_arr[k])
            break

    keep_set = set(selected_indices)
    keep_genomes = [genomes[i] for i in selected_indices]
    excess_genomes = [genomes[i] for i in range(n) if i not in keep_set]
    return keep_genomes, excess_genomes
