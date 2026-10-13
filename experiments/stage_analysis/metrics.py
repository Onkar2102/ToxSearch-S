"""Per-run outcomes for Stage C.

Everything here is computed after a run from files on disk:
  - population files (elites.json + archive.json for LF, temp.json = D_t for IncDBSCAN)
  - species_snapshots.jsonl (per-generation partition, speciation time, NLI counters)
  - EvolutionTracker.json (wall-clock)

Niche geometry is reported on the shared embedding canvas d_E for every cell, and
optionally under the forming distance d (``forming=True``; costly for d_N).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from utils.population_io import _extract_north_star_score, get_moderation_scores
from utils.evaluator_profiles import resolve_evaluator

RESERVE_ID = 0
ARCHIVE_ID = -1


def _load_list(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return [g for g in data if g] if isinstance(data, list) else []


def load_final_population(run_dir: Path, clustering_method: str) -> List[Dict[str, Any]]:
    """Retained evaluated individuals (the budget population)."""
    run_dir = Path(run_dir)
    if clustering_method == "dbscan":
        pop = _load_list(run_dir / "temp.json")
        for g in pop:
            g["_species"] = int(g.get("density_label", g.get("species_id", RESERVE_ID)) or 0)
        return pop
    pop = []
    for g in _load_list(run_dir / "elites.json"):
        g["_species"] = int(g.get("species_id") or 0)
        pop.append(g)
    for g in _load_list(run_dir / "archive.json"):
        g["_species"] = ARCHIVE_ID
        pop.append(g)
    return pop


def load_snapshots(run_dir: Path) -> List[Dict[str, Any]]:
    path = Path(run_dir) / "species_snapshots.jsonl"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _unit(emb: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.where(n == 0, 1.0, n)


def d_e_matrix(emb: np.ndarray) -> np.ndarray:
    e = _unit(np.asarray(emb, dtype=np.float64))
    return (1.0 - np.clip(e @ e.T, -1.0, 1.0)) / 2.0


def objective_matrix(genomes: List[Dict[str, Any]], evaluator: str) -> np.ndarray:
    order = resolve_evaluator(evaluator).phenotype_score_order
    rows = []
    for g in genomes:
        s = get_moderation_scores(g) or {}
        rows.append([float(s.get(k, 0.0) or 0.0) for k in order])
    x = np.asarray(rows, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)) / np.sqrt(x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def niche_geometry(dist: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Spread and separation for mature species (label > 0) under one dissimilarity."""
    mature = sorted(int(s) for s in set(labels.tolist()) if s > 0)
    spreads_max, spreads_med, intra = [], [], []
    for s in mature:
        idx = np.where(labels == s)[0]
        if len(idx) < 2:
            continue
        sub = dist[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)]
        spreads_max.append(float(sub.max()))
        spreads_med.append(float(np.median(sub)))
        intra.append(sub)
    in_mature = np.isin(labels, mature)
    m_idx = np.where(in_mature)[0]
    inter = []
    if len(m_idx) > 1:
        lab = labels[m_idx]
        sub = dist[np.ix_(m_idx, m_idx)]
        iu = np.triu_indices(len(m_idx), 1)
        diff = lab[iu[0]] != lab[iu[1]]
        inter = sub[iu][diff]
    intra_all = np.concatenate(intra) if intra else np.array([])
    mean_intra = float(intra_all.mean()) if intra_all.size else float("nan")
    mean_inter = float(np.mean(inter)) if len(inter) else float("nan")
    return {
        "spread_max": float(max(spreads_max)) if spreads_max else float("nan"),
        "spread_max_median_over_species": float(np.median(spreads_max)) if spreads_max else float("nan"),
        "spread_median_median_over_species": float(np.median(spreads_med)) if spreads_med else float("nan"),
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "separation_ratio": mean_inter / mean_intra if mean_intra and mean_intra > 0 else float("nan"),
    }


def forming_distance_matrix(genomes, distance_method: str, evaluator: str, emb: np.ndarray) -> Optional[np.ndarray]:
    if distance_method == "embedding":
        return d_e_matrix(emb)
    if distance_method == "embedding+objective":
        return 0.7 * d_e_matrix(emb) + 0.3 * objective_matrix(genomes, evaluator)
    if distance_method == "nli":
        from speciation.distance_metrics.nli import pairwise_nli_matrix
        return pairwise_nli_matrix([g.get("prompt") or "" for g in genomes])
    return None


def run_metrics(row: Dict[str, Any], *, forming: bool = False, top_k: tuple = (10, 50)) -> Dict[str, Any]:
    """Outcome dict for one indexed run (``row`` from ``runs.index_runs``)."""
    run_dir = Path(row["run_dir"])
    metric = row["north_star_metric"]
    pop = load_final_population(run_dir, row["clustering_method"])
    out: Dict[str, Any] = {"run_dir": str(run_dir), "n_retained": len(pop)}
    if not pop:
        return out

    fit = np.array([_extract_north_star_score(g, metric) for g in pop], dtype=np.float64)
    order = np.sort(fit)[::-1]
    out["Q_max"] = float(order[0])
    for k in top_k:
        out[f"top{k}_mean"] = float(order[:k].mean())
    out["refusal_rate"] = float(np.mean([int(g.get("is_refusal") or 0) for g in pop]))

    labels = np.array([g["_species"] for g in pop], dtype=int)
    mature = labels[labels > 0]
    out["n_species"] = int(len(set(mature.tolist())))
    out["reserve_size"] = int((labels == RESERVE_ID).sum())
    out["archive_size"] = int((labels == ARCHIVE_ID).sum())
    out["breedable_size"] = int((labels >= 0).sum())
    sizes = np.bincount(mature) if mature.size else np.array([0])
    out["largest_species_size"] = int(sizes.max())

    has_emb = [i for i, g in enumerate(pop) if g.get("prompt_embedding") is not None]
    out["n_missing_embedding"] = len(pop) - len(has_emb)
    if len(has_emb) > 1:
        sub = [pop[i] for i in has_emb]
        sub_labels = labels[has_emb]
        emb = np.asarray([g["prompt_embedding"] for g in sub], dtype=np.float64)
        for k, v in niche_geometry(d_e_matrix(emb), sub_labels).items():
            out[f"canvas_{k}"] = v
        if forming:
            dm = forming_distance_matrix(sub, row["distance_method"], row["evaluator"], emb)
            if dm is not None:
                for k, v in niche_geometry(dm, sub_labels).items():
                    out[f"forming_{k}"] = v

    snaps = load_snapshots(run_dir)
    if snaps:
        durs = [s.get("speciation_duration_seconds") for s in snaps if s.get("speciation_duration_seconds") is not None]
        out["speciation_seconds"] = float(np.sum(durs)) if durs else float("nan")
        nli = snaps[-1].get("nli") or {}
        out["nli_model_passes"] = nli.get("model_passes")
        out["nli_distance_calls"] = nli.get("distance_calls")
    wall = row.get("run_duration_seconds")
    if wall:
        out["wall_seconds"] = float(wall)
        spec = out.get("speciation_seconds")
        if spec is not None and spec == spec:
            out["speciation_share"] = out["speciation_seconds"] / float(wall)
    return out


def topic_coverage(prompt_sets: Dict[str, List[str]], embeddings: Dict[str, np.ndarray], seed: int = 0) -> Dict[str, Dict[str, float]]:
    """Effective topic count N1=exp(H) and K_topic per run, from ONE pooled topic model.

    Fitting a single BERTopic model on the union of all runs' prompts keeps the
    topic vocabulary identical across cells (per-run models are not comparable).
    """
    from bertopic import BERTopic
    from umap import UMAP

    keys, docs, embs = [], [], []
    for k, prompts in prompt_sets.items():
        keys += [k] * len(prompts)
        docs += prompts
        embs.append(np.asarray(embeddings[k]))
    model = BERTopic(umap_model=UMAP(random_state=seed), calculate_probabilities=False)
    topics, _ = model.fit_transform(docs, np.vstack(embs))
    keys_arr, topics_arr = np.array(keys), np.array(topics)
    out = {}
    for k in prompt_sets:
        t = topics_arr[(keys_arr == k) & (topics_arr >= 0)]
        if t.size == 0:
            out[k] = {"N1": 0.0, "K_topic": 0}
            continue
        p = np.bincount(t) / t.size
        p = p[p > 0]
        out[k] = {"N1": float(np.exp(-(p * np.log(p)).sum())), "K_topic": int(p.size)}
    return out
