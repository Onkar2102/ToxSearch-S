"""Shared loaders and helpers for data analysis."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import logging
import re
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
except ImportError:
    pass

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.evaluator_profiles import (  # noqa: E402
    GOOGLE_NORTH_STAR_CHOICES,
    OPENAI_NORTH_STAR_CHOICES,
)

PERSPECTIVE_AXIS_ORDER = list(GOOGLE_NORTH_STAR_CHOICES)
OPENAI_AXIS_ORDER = list(OPENAI_NORTH_STAR_CHOICES)
POP_FILES = ("elites.json", "reserves.json", "archive.json", "temp.json")
_logger = logging.getLogger("emnlp_4pager.analysis")
DEFAULT_PRIMARY_RUN = REPO_ROOT / "data/outputs/20260211_2122"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

__all__ = [
    "DEFAULT_PRIMARY_RUN",
    "PERSPECTIVE_AXIS_ORDER",
    "POP_FILES",
    "RESULTS_DIR",
    "annotate_pareto_ranks",
    "cohort_label_viz",
    "dominates",
    "embedding_matrix",
    "fast_non_dominated_sort",
    "OPENAI_AXIS_ORDER",
    "get_axis_order",
    "load_unified_genomes",
    "score_column_name",
    "run_id_from_path",
    "rows_for_pymoo_viz",
    "save_phase2_artifacts",
    "save_unified_artifacts",
    "smoke_validate_run",
]


def get_axis_order(evaluator: str = "google") -> List[str]:
    if evaluator == "google":
        return list(GOOGLE_NORTH_STAR_CHOICES)
    if evaluator == "openai":
        return list(OPENAI_NORTH_STAR_CHOICES)
    raise ValueError(f"Unsupported evaluator for analysis: {evaluator!r}")


def score_column_name(backend: str, axis: str) -> str:
    """CSV column for a moderation axis (`f_*` = Google, `oai_*` = OpenAI)."""
    safe = axis.replace("/", "_")
    if backend == "google":
        return f"f_{safe}"
    if backend == "openai":
        return f"oai_{safe}"
    raise ValueError(f"Unknown backend: {backend!r}")


def run_id_from_path(run_path: Path) -> str:
    return run_path.name.replace(" ", "_")


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Pareto dominance for maximization (8-D toxicity objectives)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return bool(np.all(a >= b) and np.any(a > b))


def fast_non_dominated_sort(F: np.ndarray) -> List[List[int]]:
    """NSGA-II non-dominated sorting; front 0 = local F₀ (vectorized dominance)."""
    F = np.asarray(F, dtype=np.float64)
    n = int(F.shape[0])
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    ge = np.all(F[:, None, :] >= F[None, :, :], axis=2)
    gt = np.any(F[:, None, :] > F[None, :, :], axis=2)
    dom = ge & gt
    np.fill_diagonal(dom, False)

    n_dom = dom.sum(axis=0, dtype=np.int32)
    S: List[List[int]] = [np.where(dom[p])[0].tolist() for p in range(n)]

    fronts: List[List[int]] = [np.where(n_dom == 0)[0].tolist()]
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)
    return [f for f in fronts if f]


def _stamp_pareto_ranks(
    F: np.ndarray,
    local_indices: Sequence[int],
    rank_out: np.ndarray,
    front_out: np.ndarray,
) -> None:
    """Write rank (0 = F₀) and on_f0 into rank_out/front_out at local_indices."""
    if not local_indices:
        return
    idx = np.asarray(local_indices, dtype=np.int64)
    sub = F[idx]
    fronts = fast_non_dominated_sort(sub)
    for k, front in enumerate(fronts):
        for li in front:
            j = int(idx[li])
            rank_out[j] = k
            if k == 0:
                front_out[j] = True


def cohort_label_viz(row: Dict[str, Any]) -> str:
    """Viz cohort (cluster_analysis style): reserves | species_<id> | archive | other."""
    source = str(row.get("source_file") or "")
    if "reserves.json" in source:
        return "reserves"
    sid = int(row.get("species_id") or 0)
    if sid > 0:
        return f"species_{sid}"
    if "archive.json" in source:
        return "archive"
    return "other"


def annotate_pareto_ranks(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assign global + local (species / reserves / archive) Pareto ranks to genomes."""
    valid_idx = [i for i, r in enumerate(rows) if "objective_vector" in r]
    n = len(valid_idx)
    if n == 0:
        return {"error": "no objective vectors"}

    F = np.vstack([rows[i]["objective_vector"] for i in valid_idx])
    rank_global = np.full(n, -1, dtype=np.int32)
    front_global = np.zeros(n, dtype=bool)
    rank_species = np.full(n, -1, dtype=np.int32)
    front_species = np.zeros(n, dtype=bool)
    rank_reserves = np.full(n, -1, dtype=np.int32)
    front_reserves = np.zeros(n, dtype=bool)
    rank_archive = np.full(n, -1, dtype=np.int32)
    front_archive = np.zeros(n, dtype=bool)
    rank_cohort = np.full(n, -1, dtype=np.int32)
    front_cohort = np.zeros(n, dtype=bool)

    _stamp_pareto_ranks(F, list(range(n)), rank_global, front_global)

    by_species: Dict[int, List[int]] = defaultdict(list)
    reserves_local: List[int] = []
    archive_local: List[int] = []
    for j, row_i in enumerate(valid_idx):
        r = rows[row_i]
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_species[sid].append(j)
        source = str(r.get("source_file") or "")
        if "reserves.json" in source:
            reserves_local.append(j)
        if "archive.json" in source:
            archive_local.append(j)

    for idxs in by_species.values():
        _stamp_pareto_ranks(F, idxs, rank_species, front_species)
    _stamp_pareto_ranks(F, reserves_local, rank_reserves, front_reserves)
    _stamp_pareto_ranks(F, archive_local, rank_archive, front_archive)

    by_viz_cohort: Dict[str, List[int]] = defaultdict(list)
    for j, row_i in enumerate(valid_idx):
        by_viz_cohort[cohort_label_viz(rows[row_i])].append(j)
    for idxs in by_viz_cohort.values():
        _stamp_pareto_ranks(F, idxs, rank_cohort, front_cohort)

    for j, row_i in enumerate(valid_idx):
        r = rows[row_i]
        r["global_pareto_rank"] = int(rank_global[j])
        r["on_global_f0"] = bool(front_global[j])
        r["species_pareto_rank"] = int(rank_species[j])
        r["on_species_f0"] = bool(front_species[j])
        r["reserves_pareto_rank"] = int(rank_reserves[j])
        r["on_reserves_f0"] = bool(front_reserves[j])
        r["archive_pareto_rank"] = int(rank_archive[j])
        r["on_archive_f0"] = bool(front_archive[j])
        r["cohort"] = cohort_label_viz(r)
        r["pareto_rank_cohort"] = int(rank_cohort[j])
        r["on_cohort_f0"] = bool(front_cohort[j])

    for r in rows:
        if "global_pareto_rank" not in r:
            r["global_pareto_rank"] = -1
            r["on_global_f0"] = False
            r["species_pareto_rank"] = -1
            r["on_species_f0"] = False
            r["reserves_pareto_rank"] = -1
            r["on_reserves_f0"] = False
            r["archive_pareto_rank"] = -1
            r["on_archive_f0"] = False
            r["cohort"] = cohort_label_viz(r)
            r["pareto_rank_cohort"] = -1
            r["on_cohort_f0"] = False

    global_fronts = fast_non_dominated_sort(F)
    f0_max = F[front_global].max(axis=0) if np.any(front_global) else F.max(axis=0)

    cohort_stats: List[Dict[str, Any]] = []
    by_cohort: Dict[str, List[int]] = defaultdict(list)
    for j in range(n):
        by_cohort[cohort_label_viz(rows[valid_idx[j]])].append(j)
    for lab, idxs in sorted(by_cohort.items()):
        cohort_stats.append({
            "cohort": lab,
            "n_genomes": len(idxs),
            "n_cohort_f0": int(sum(front_cohort[i] for i in idxs)),
            "n_global_f0": int(sum(front_global[i] for i in idxs)),
        })

    return {
        "n_genomes": n,
        "n_f0": int(front_global.sum()),
        "f0_fraction": float(front_global.mean()),
        "n_fronts": len(global_fronts),
        "f0_max": [float(x) for x in f0_max],
        "n_species_groups": len(by_species),
        "n_reserves": len(reserves_local),
        "n_archive": len(archive_local),
        "cohort_summary": cohort_stats,
    }


VECTOR_KEY_BY_EVALUATOR = {
    "google": "objective_vector",
    "openai": "objective_vector_openai",
}


def rows_for_pymoo_viz(
    rows: Sequence[Dict[str, Any]],
    *,
    evaluator: str = "google",
) -> List[Dict[str, Any]]:
    """Rows with ``objectives`` + ``cohort`` for pymoo PCP (cluster_analysis convention)."""
    vec_key = VECTOR_KEY_BY_EVALUATOR.get(evaluator)
    if vec_key is None:
        raise ValueError(f"Unsupported evaluator for viz: {evaluator!r}")
    out: List[Dict[str, Any]] = []
    for r in rows:
        vec = r.get(vec_key)
        if vec is None:
            continue
        out.append({
            "id": r.get("genome_id"),
            "cohort": r.get("cohort") or cohort_label_viz(r),
            "generation": r.get("generation"),
            "source_file": r.get("source_file"),
            "objectives": np.asarray(vec, dtype=np.float64),
            "species_id": r.get("species_id"),
            "evaluator": evaluator,
        })
    return out


def _evaluator_vector_key(evaluator: str) -> str:
    key = VECTOR_KEY_BY_EVALUATOR.get(evaluator)
    if key is None:
        raise ValueError(f"Unsupported evaluator: {evaluator!r}")
    return key


def global_pareto_annotate(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Return (global_rank, on_f0_mask, fronts) for one objective matrix."""
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=bool), []
    fronts = fast_non_dominated_sort(F)
    ranks = np.full(n, -1, dtype=np.int32)
    on_f0 = np.zeros(n, dtype=bool)
    for k, front in enumerate(fronts):
        for i in front:
            ranks[i] = k
            if k == 0:
                on_f0[i] = True
    return ranks, on_f0, fronts


def member_dominated_mask(F: np.ndarray) -> np.ndarray:
    """True if the row is dominated by at least one other row in F."""
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    if n == 1:
        return np.array([False], dtype=bool)
    ge = np.all(F[:, None, :] >= F[None, :, :], axis=2)
    gt = np.any(F[:, None, :] > F[None, :, :], axis=2)
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return dom.any(axis=0)


def genotype_distance(e1: np.ndarray, e2: np.ndarray) -> float:
    e1 = np.asarray(e1, dtype=np.float64).reshape(-1)
    e2 = np.asarray(e2, dtype=np.float64).reshape(-1)
    cos = float(np.clip(np.dot(e1, e2), -1.0, 1.0))
    return 0.5 * (1.0 - cos)


def topic_centroids(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
) -> Dict[int, Dict[str, Any]]:
    vec_key = _evaluator_vector_key(evaluator)
    by_topic: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = int(r.get("species_id") or 0)
        if sid > 0 and r.get(vec_key) is not None:
            by_topic[sid].append(r)

    out: Dict[int, Dict[str, Any]] = {}
    for sid, members in by_topic.items():
        F = np.vstack([m[vec_key] for m in members])
        embs = [m["_embedding"] for m in members if m.get("_embedding") is not None]
        emb_cent = np.mean(np.vstack(embs), axis=0) if embs else None
        if emb_cent is not None:
            nrm = np.linalg.norm(emb_cent)
            if nrm > 1e-12:
                emb_cent = emb_cent / nrm
        out[sid] = {
            "max_vector": F.max(axis=0),
            "mean_vector": F.mean(axis=0),
            "embedding_centroid": emb_cent,
            "members": members,
            "F": F,
        }
    return out


def distinct_topic_label(
    sid: int,
    centroids: Dict[int, Dict[str, Any]],
    *,
    tau_intra: float = 0.85,
    tau_inter: float = 0.35,
) -> Tuple[bool, float, float]:
    info = centroids.get(sid)
    if info is None:
        return False, 0.0, 0.0
    embs = [m["_embedding"] for m in info["members"] if m.get("_embedding") is not None]
    if len(embs) < 2:
        return False, 0.0, 0.0
    embs_arr = np.vstack(embs)
    norms = np.linalg.norm(embs_arr, axis=1, keepdims=True)
    embs_n = embs_arr / np.where(norms < 1e-12, 1.0, norms)
    cos_mat = embs_n @ embs_n.T
    triu = cos_mat[np.triu_indices(len(embs), k=1)]
    intra = float(np.mean(triu)) if triu.size else 0.0

    c_emb = info.get("embedding_centroid")
    if c_emb is None:
        return False, intra, 0.0
    inter_dists = []
    for other_sid, other in centroids.items():
        if other_sid == sid:
            continue
        o_emb = other.get("embedding_centroid")
        if o_emb is None:
            continue
        inter_dists.append(genotype_distance(c_emb, o_emb))
    inter_min = float(min(inter_dists)) if inter_dists else 0.0
    distinct = intra >= tau_intra and inter_min >= tau_inter
    return distinct, intra, inter_min


def compute_topic_summaries(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
    *,
    min_topic_size: int = 5,
    tau_intra: float = 0.85,
    tau_inter: float = 0.35,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-species TDI and coverage stats in one evaluator's objective space."""
    vec_key = _evaluator_vector_key(evaluator)
    axis_order = list(get_axis_order(evaluator))
    valid = [r for r in rows if r.get(vec_key) is not None]
    if not valid:
        return [], {"evaluator": evaluator, "error": "no objective vectors"}

    F = np.vstack([r[vec_key] for r in valid])
    ranks, on_f0, fronts = global_pareto_annotate(F)
    dominated = member_dominated_mask(F)
    f0_max = F[on_f0].max(axis=0) if np.any(on_f0) else F.max(axis=0)

    by_topic: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic[sid].append(i)

    centroids = topic_centroids(valid, evaluator)
    summaries: List[Dict[str, Any]] = []

    for sid, idxs in sorted(by_topic.items()):
        n = len(idxs)
        if n < min_topic_size:
            continue
        tdi = float(np.mean(dominated[idxs]))
        n_on_f0 = int(np.sum(on_f0[idxs]))
        max_vec = F[idxs].max(axis=0)
        axis_exclusive = [
            axis_order[i]
            for i in range(len(axis_order))
            if max_vec[i] > f0_max[i] + 1e-9
        ]
        distinct, intra, inter_min = distinct_topic_label(
            sid, centroids, tau_intra=tau_intra, tau_inter=tau_inter
        )
        dominating: List[int] = []
        for other_sid, other in centroids.items():
            if other_sid == sid:
                continue
            if dominates(other["max_vector"], max_vec):
                dominating.append(other_sid)

        summaries.append({
            "evaluator": evaluator,
            "species_id": sid,
            "n_members": n,
            "n_on_f0": n_on_f0,
            "frac_on_f0": float(n_on_f0 / n),
            "tdi": round(tdi, 4),
            "fully_dominated": n_on_f0 == 0,
            "distinct_topic": distinct,
            "axis_exclusive_axes": ";".join(axis_exclusive),
            "n_axis_exclusive": len(axis_exclusive),
            "intra_cosine_mean": round(intra, 4),
            "inter_dg_min": round(inter_min, 4),
            "dominating_topics": ";".join(str(x) for x in dominating),
            "max_axis_0": float(max_vec[0]),
        })

    global_stats: Dict[str, Any] = {
        "evaluator": evaluator,
        "n_genomes": len(valid),
        "n_objectives": len(axis_order),
        "n_f0": int(np.sum(on_f0)),
        "f0_fraction": float(np.mean(on_f0)),
        "n_fronts": len(fronts),
        "n_topics_summarized": len(summaries),
        "n_fully_dominated_topics": sum(1 for s in summaries if s["fully_dominated"]),
        "n_distinct_topics": sum(1 for s in summaries if s["distinct_topic"]),
        "n_distinct_fully_dominated": sum(
            1 for s in summaries if s["fully_dominated"] and s["distinct_topic"]
        ),
        "n_axis_exclusive_fully_dominated": sum(
            1 for s in summaries
            if s["fully_dominated"] and s["n_axis_exclusive"] > 0
        ),
        "tau_intra_cosine": tau_intra,
        "tau_inter_dg": tau_inter,
        "min_topic_size": min_topic_size,
    }
    return summaries, global_stats


def topic_domination_edges(
    summaries: Sequence[Dict[str, Any]],
    centroids: Dict[int, Dict[str, Any]],
) -> List[Dict[str, int]]:
    edges: List[Dict[str, int]] = []
    ids = [s["species_id"] for s in summaries]
    for s in summaries:
        c = s["species_id"]
        max_c = centroids[c]["max_vector"]
        for c2 in ids:
            if c2 == c:
                continue
            if dominates(centroids[c2]["max_vector"], max_c):
                edges.append({"from": c2, "to": c})
    return edges


def compare_evaluator_global_ranks(
    rows: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Compare global Pareto ranks between Google and OpenAI on genomes with both vectors."""
    g_key, o_key = "objective_vector", "objective_vector_openai"
    valid = [r for r in rows if r.get(g_key) is not None and r.get(o_key) is not None]
    if not valid:
        return [], {"error": "no genomes with both evaluators"}

    F_g = np.vstack([r[g_key] for r in valid])
    F_o = np.vstack([r[o_key] for r in valid])
    rank_g, f0_g, fronts_g = global_pareto_annotate(F_g)
    rank_o, f0_o, fronts_o = global_pareto_annotate(F_o)

    per_genome: List[Dict[str, Any]] = []
    for i, r in enumerate(valid):
        rg, ro = int(rank_g[i]), int(rank_o[i])
        per_genome.append({
            "genome_id": r.get("genome_id"),
            "species_id": int(r.get("species_id") or 0),
            "cohort": cohort_label_viz(r),
            "google_rank": rg,
            "openai_rank": ro,
            "rank_diff": ro - rg,
            "google_on_f0": bool(f0_g[i]),
            "openai_on_f0": bool(f0_o[i]),
            "same_rank": rg == ro,
            "both_on_f0": bool(f0_g[i] and f0_o[i]),
            "google_f0_only": bool(f0_g[i] and not f0_o[i]),
            "openai_f0_only": bool(f0_o[i] and not f0_g[i]),
        })

    n = len(per_genome)
    same_rank = sum(1 for g in per_genome if g["same_rank"])
    both_f0 = sum(1 for g in per_genome if g["both_on_f0"])
    g_f0_only = sum(1 for g in per_genome if g["google_f0_only"])
    o_f0_only = sum(1 for g in per_genome if g["openai_f0_only"])
    neither_f0 = sum(1 for g in per_genome if not g["google_on_f0"] and not g["openai_on_f0"])

    rank_diffs = np.array([g["rank_diff"] for g in per_genome], dtype=np.int32)
    rg_arr = np.array([g["google_rank"] for g in per_genome], dtype=np.int32)
    ro_arr = np.array([g["openai_rank"] for g in per_genome], dtype=np.int32)

    spearman_r = float("nan")
    try:
        from scipy.stats import spearmanr

        res = spearmanr(rg_arr, ro_arr)
        spearman_r = float(res.correlation) if res.correlation is not None else float("nan")
    except Exception:
        pass

    max_r = int(max(rank_g.max(), rank_o.max(), 0))
    cap = min(max_r, 12)
    contingency: Dict[str, int] = {}
    for g in per_genome:
        key = f"g{g['google_rank']}_o{g['openai_rank']}"
        if g["google_rank"] <= cap and g["openai_rank"] <= cap:
            contingency[key] = contingency.get(key, 0) + 1

    summary: Dict[str, Any] = {
        "n_genomes_both_evaluators": n,
        "n_google_f0": int(np.sum(f0_g)),
        "n_openai_f0": int(np.sum(f0_o)),
        "n_google_fronts": len(fronts_g),
        "n_openai_fronts": len(fronts_o),
        "n_same_rank": same_rank,
        "frac_same_rank": round(same_rank / n, 4),
        "n_both_on_f0": both_f0,
        "frac_both_on_f0": round(both_f0 / n, 4),
        "n_google_f0_only": g_f0_only,
        "n_openai_f0_only": o_f0_only,
        "n_neither_f0": neither_f0,
        "jaccard_f0_sets": round(
            both_f0 / max(int(np.sum(f0_g)) + int(np.sum(f0_o)) - both_f0, 1),
            4,
        ),
        "rank_diff_mean": float(rank_diffs.mean()),
        "rank_diff_median": float(np.median(rank_diffs)),
        "rank_diff_std": float(rank_diffs.std()),
        "rank_diff_min": int(rank_diffs.min()),
        "rank_diff_max": int(rank_diffs.max()),
        "n_rank_diff_zero": int(np.sum(rank_diffs == 0)),
        "n_openai_rank_higher": int(np.sum(rank_diffs > 0)),
        "n_google_rank_higher": int(np.sum(rank_diffs < 0)),
        "spearman_rank_correlation": spearman_r,
        "contingency_rank_pairs_lte_{}".format(cap): contingency,
        "among_google_f0_n": int(np.sum(f0_g)),
        "among_google_f0_same_rank_frac": round(
            sum(1 for i, g in enumerate(per_genome) if f0_g[i] and g["same_rank"])
            / max(int(np.sum(f0_g)), 1),
            4,
        ),
        "among_openai_f0_n": int(np.sum(f0_o)),
        "among_openai_f0_same_rank_frac": round(
            sum(1 for i, g in enumerate(per_genome) if f0_o[i] and g["same_rank"])
            / max(int(np.sum(f0_o)), 1),
            4,
        ),
    }
    return per_genome, summary


def save_phase3_artifacts(
    rows: List[Dict[str, Any]],
    out_dir: Path,
    run_id: str,
    *,
    min_topic_size: int = 5,
    tau_intra: float = 0.85,
    tau_inter: float = 0.35,
) -> Dict[str, Any]:
    """Phase 3: per-evaluator TDI tables + Google vs OpenAI global rank comparison."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}
    phase3_meta: Dict[str, Any] = {"run_id": run_id, "evaluators": {}}
    summaries_by_ev: Dict[str, List[Dict[str, Any]]] = {}
    edges_by_ev: Dict[str, List[Dict[str, int]]] = {}

    for evaluator in ("google", "openai"):
        ev_dir = out_dir / evaluator
        ev_dir.mkdir(parents=True, exist_ok=True)
        summaries, gstats = compute_topic_summaries(
            rows,
            evaluator,
            min_topic_size=min_topic_size,
            tau_intra=tau_intra,
            tau_inter=tau_inter,
        )
        summaries_by_ev[evaluator] = summaries
        vec_key = _evaluator_vector_key(evaluator)
        valid = [r for r in rows if r.get(vec_key) is not None]
        centroids = topic_centroids(valid, evaluator)
        edges = topic_domination_edges(summaries, centroids) if summaries else []
        edges_by_ev[evaluator] = edges

        csv_path = ev_dir / "topic_domination_summary.csv"
        if summaries:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
                w.writeheader()
                w.writerows(summaries)
        (ev_dir / "topic_domination_global_stats.json").write_text(
            json.dumps(gstats, indent=2), encoding="utf-8"
        )
        (ev_dir / "topic_domination_edges.json").write_text(
            json.dumps(edges, indent=2), encoding="utf-8"
        )
        artifacts[f"{evaluator}_topic_summary_csv"] = str(csv_path)
        artifacts[f"{evaluator}_global_stats_json"] = str(ev_dir / "topic_domination_global_stats.json")
        artifacts[f"{evaluator}_edges_json"] = str(ev_dir / "topic_domination_edges.json")
        phase3_meta["evaluators"][evaluator] = gstats

    # --- Phase 3b: topic Google vs OpenAI comparison + domination visuals ---
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    try:
        import importlib

        import phase3_viz

        importlib.reload(phase3_viz)

        cmp_rows, cmp_stats = phase3_viz.build_topic_evaluator_comparison(
            summaries_by_ev.get("google", []),
            summaries_by_ev.get("openai", []),
        )
        cmp_csv = out_dir / "topic_evaluator_comparison.csv"
        if cmp_rows:
            with cmp_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(cmp_rows[0].keys()))
                w.writeheader()
                w.writerows(cmp_rows)
        (out_dir / "topic_evaluator_comparison_stats.json").write_text(
            json.dumps(cmp_stats, indent=2), encoding="utf-8"
        )
        artifacts["topic_evaluator_comparison_csv"] = str(cmp_csv)
        artifacts["topic_evaluator_comparison_stats_json"] = str(
            out_dir / "topic_evaluator_comparison_stats.json"
        )
        phase3_meta["topic_evaluator_comparison"] = cmp_stats

        p = phase3_viz.plot_topic_evaluator_comparison(
            cmp_rows,
            fig_dir / "topic_tdi_google_vs_openai.png",
            title=f"{run_id}: topic TDI by evaluator",
        )
        if p:
            artifacts["fig_topic_tdi_comparison"] = p

        for evaluator in ("google", "openai"):
            summaries = summaries_by_ev.get(evaluator, [])
            edges = edges_by_ev.get(evaluator, [])
            mat, sids = phase3_viz.build_domination_matrix(summaries, edges)
            if mat.size == 0:
                continue
            labels = [str(s) for s in sids]
            mat_csv = out_dir / evaluator / "topic_domination_matrix.csv"
            try:
                import pandas as pd

                pd.DataFrame(mat, index=labels, columns=labels).to_csv(mat_csv)
            except ImportError:
                with mat_csv.open("w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([""] + labels)
                    for i, lab in enumerate(labels):
                        w.writerow([lab] + mat[i].tolist())
            artifacts[f"{evaluator}_domination_matrix_csv"] = str(mat_csv)

            hp = phase3_viz.plot_domination_heatmap(
                mat,
                sids,
                fig_dir / f"topic_domination_heatmap_{evaluator}.png",
                title=f"{run_id}: max-vector domination ({evaluator})",
            )
            if hp:
                artifacts[f"fig_domination_heatmap_{evaluator}"] = hp

            gp = phase3_viz.plot_domination_graph(
                edges,
                sids,
                fig_dir / f"topic_domination_graph_{evaluator}.png",
                title=f"{run_id}: topic domination graph ({evaluator})",
            )
            if gp:
                artifacts[f"fig_domination_graph_{evaluator}"] = gp

        phase3_meta["figures_dir"] = str(fig_dir)
        artifacts["figures_dir"] = str(fig_dir)
    except Exception as exc:
        _logger.error("Phase 3 comparison/figures failed: %s", exc)
        print(f"WARNING: Phase 3 figures failed: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    rank_rows, rank_summary = compare_evaluator_global_ranks(rows)
    rank_csv = out_dir / "evaluator_rank_agreement.csv"
    if rank_rows:
        with rank_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rank_rows[0].keys()))
            w.writeheader()
            w.writerows(rank_rows)
    rank_json = out_dir / "evaluator_rank_agreement_summary.json"
    rank_json.write_text(json.dumps(rank_summary, indent=2), encoding="utf-8")
    artifacts["rank_agreement_csv"] = str(rank_csv)
    artifacts["rank_agreement_summary_json"] = str(rank_json)
    phase3_meta["rank_agreement"] = rank_summary

    manifest_path = out_dir / "phase3_manifest.json"
    manifest_path.write_text(
        json.dumps({"artifacts": artifacts, "meta": phase3_meta}, indent=2),
        encoding="utf-8",
    )
    artifacts["phase3_manifest"] = str(manifest_path)
    return {"artifacts": artifacts, "meta": phase3_meta}


def save_phase4_artifacts(
    run_path: Path,
    out_dir: Path,
    run_id: str,
    *,
    reference_species: int = 9,
) -> Dict[str, Any]:
    """Phase 4: EvolutionTracker time series + temporal figure (Fig 1A)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: Dict[str, str] = {}

    try:
        import importlib

        import phase4_temporal

        importlib.reload(phase4_temporal)

        rows, stats = phase4_temporal.load_evolution_tracker_timeseries(run_path)
        if not rows:
            return {"artifacts": artifacts, "meta": stats}

        csv_path = out_dir / f"timeseries_{run_id}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        artifacts["timeseries_csv"] = str(csv_path)

        stats_path = out_dir / f"timeseries_{run_id}_stats.json"
        stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        artifacts["timeseries_stats_json"] = str(stats_path)

        fig_dir = out_dir / "figures"
        fig_path = phase4_temporal.plot_temporal_species(
            rows,
            fig_dir / "fig1_temporal_species_count.png",
            run_id=run_id,
            reference_species=reference_species,
        )
        if fig_path:
            artifacts["fig_temporal_species"] = fig_path
            artifacts["figures_dir"] = str(fig_dir)

        manifest_path = out_dir / "phase4_manifest.json"
        meta = {"run_id": run_id, "run_path": str(run_path), "stats": stats}
        manifest_path.write_text(
            json.dumps({"artifacts": artifacts, "meta": meta}, indent=2),
            encoding="utf-8",
        )
        artifacts["phase4_manifest"] = str(manifest_path)
        return {"artifacts": artifacts, "meta": meta}
    except Exception as exc:
        _logger.error("Phase 4 failed: %s", exc)
        print(f"WARNING: Phase 4 failed: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return {"artifacts": artifacts, "meta": {"error": str(exc)}}


def save_phase2_artifacts(
    rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    out_dir: Path,
    run_id: str,
) -> Dict[str, str]:
    """Write Phase 2 CSVs, per-cohort front exports, and optional pymoo figures."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_order = list(get_axis_order())

    export_rows = []
    for r in rows:
        er = {k: v for k, v in r.items() if k not in ("_embedding", "objective_vector")}
        export_rows.append(er)

    csv_path = out_dir / f"{run_id}_genomes_pareto.csv"
    try:
        import pandas as pd

        pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    except ImportError:
        if export_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(export_rows[0].keys()))
                w.writeheader()
                w.writerows(export_rows)

    summary_path = out_dir / "phase2_global_pareto.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    cohort_csv = out_dir / "pareto_cohort_summary.csv"
    cohort_rows = stats.get("cohort_summary") or []
    if cohort_rows:
        with cohort_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cohort_rows[0].keys()))
            w.writeheader()
            w.writerows(cohort_rows)

    by_cohort_dir = out_dir / "pareto_fronts_by_cohort"
    by_cohort_dir.mkdir(parents=True, exist_ok=True)
    by_cohort: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if "objective_vector" not in r:
            continue
        key = re.sub(r"[^A-Za-z0-9._-]+", "_", str(r.get("cohort") or "other"))
        by_cohort[key].append(r)

    for key, members in sorted(by_cohort.items()):
        path = by_cohort_dir / f"{key}.csv"
        fieldnames = [
            "genome_id",
            "species_id",
            "generation",
            "source_file",
            "species_pareto_rank",
            "global_pareto_rank",
            "reserves_pareto_rank",
            "archive_pareto_rank",
            "on_global_f0",
            "on_species_f0",
            "on_reserves_f0",
            "on_archive_f0",
            *axis_order,
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for r in members:
                vec = r["objective_vector"]
                w.writerow({
                    "genome_id": r.get("genome_id"),
                    "species_id": r.get("species_id"),
                    "generation": r.get("generation"),
                    "source_file": r.get("source_file"),
                    "species_pareto_rank": r.get("species_pareto_rank", -1),
                    "global_pareto_rank": r.get("global_pareto_rank", -1),
                    "reserves_pareto_rank": r.get("reserves_pareto_rank", -1),
                    "archive_pareto_rank": r.get("archive_pareto_rank", -1),
                    "on_global_f0": r.get("on_global_f0", False),
                    "on_species_f0": r.get("on_species_f0", False),
                    "on_reserves_f0": r.get("on_reserves_f0", False),
                    "on_archive_f0": r.get("on_archive_f0", False),
                    **{ax: float(vec[i]) for i, ax in enumerate(axis_order)},
                })

    fig_root = out_dir / "figures"
    pymoo_paths: Dict[str, str] = {}
    figure_dirs: Dict[str, str] = {}
    try:
        import pymoo_pcp_viz

        importlib.reload(pymoo_pcp_viz)
        generate_pymoo_viz = pymoo_pcp_viz.generate_pymoo_viz

        for evaluator in ("google", "openai"):
            viz_rows = rows_for_pymoo_viz(rows, evaluator=evaluator)
            if not viz_rows:
                _logger.warning("No rows with %s objectives; skipping figures.", evaluator)
                continue
            fig_dir = fig_root / evaluator
            fig_dir.mkdir(parents=True, exist_ok=True)
            figure_dirs[f"figures_{evaluator}_dir"] = str(fig_dir)
            paths = generate_pymoo_viz(
                viz_rows,
                fig_dir,
                evaluator=evaluator,
                label_style=getattr(
                    pymoo_pcp_viz, "PCP_LABEL_STYLE", "abbrev"
                ),
                title_suffix=f"{run_id} ({evaluator})",
            )
            if not paths:
                _logger.warning(
                    "pymoo produced no figures for %s (install pymoo + matplotlib?)",
                    evaluator,
                )
            for k, v in paths.items():
                if v:
                    pymoo_paths[f"pymoo_{evaluator}_{k}"] = v
    except Exception as exc:
        _logger.error("Phase 2 figure generation failed: %s", exc)
        _logger.debug(traceback.format_exc())
        print(f"WARNING: Phase 2 figures failed: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)

    artifacts = {
        "genomes_pareto_csv": str(csv_path),
        "phase2_summary": str(summary_path),
        "cohort_summary_csv": str(cohort_csv),
        "pareto_fronts_by_cohort_dir": str(by_cohort_dir),
        **figure_dirs,
        **{k: v for k, v in pymoo_paths.items() if v},
    }
    return artifacts


def _load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _extract_embedding(genome: Dict[str, Any]) -> Optional[np.ndarray]:
    emb = genome.get("prompt_embedding")
    if emb is None:
        return None
    arr = np.asarray(emb, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return None
    norm = np.linalg.norm(arr)
    if norm > 1e-12:
        arr = arr / norm
    return arr


def _scores_row_from_genome(
    genome: Dict[str, Any],
    axis_order: Sequence[str],
    backend: str,
) -> Optional[np.ndarray]:
    """Read moderation scores for one backend (google or openai)."""
    mr = genome.get("moderation_result") or {}
    scores = None
    if isinstance(mr, dict):
        block = mr.get(backend)
        if isinstance(block, dict) and isinstance(block.get("scores"), dict):
            scores = block["scores"]
    if not isinstance(scores, dict):
        if backend == "google" and genome.get("objective_vector") is not None:
            v = np.asarray(genome["objective_vector"], dtype=np.float64).reshape(-1)
            if v.size == len(axis_order):
                return np.clip(v, 0.0, 1.0)
        return None
    row = []
    for axis in axis_order:
        val = _score_lookup(scores, axis, backend)
        if val is None:
            return None
        try:
            row.append(float(np.clip(float(val), 0.0, 1.0)))
        except (TypeError, ValueError):
            return None
    return np.asarray(row, dtype=np.float64)


def _score_lookup(scores: Dict[str, Any], axis: str, backend: str) -> Any:
    """Resolve a score; OpenAI omni may use slashes or underscores."""
    val = scores.get(axis)
    if val is not None:
        return val
    if backend == "openai":
        val = scores.get(axis.replace("/", "_"))
        if val is not None:
            return val
        val = scores.get(axis.replace("-", "_"))
    return val


def openai_cache_path(run_id: str) -> Path:
    return RESULTS_DIR / "unified" / f"{run_id}_openai_cache.json"


def _load_openai_cache(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_openai_cache(path: Path, cache: Dict[str, Dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _attach_score_vector(
    record: Dict[str, Any],
    vec: np.ndarray,
    axis_order: Sequence[str],
    backend: str,
    *,
    vector_key: str,
) -> None:
    for i, axis in enumerate(axis_order):
        record[score_column_name(backend, axis)] = float(vec[i])
    record[vector_key] = vec


def _apply_openai_scores_to_record(
    record: Dict[str, Any],
    scores: Dict[str, float],
    openai_axes: Sequence[str],
) -> bool:
    raw = {"moderation_result": {"openai": {"scores": scores}}}
    vec = _scores_row_from_genome(raw, openai_axes, "openai")
    if vec is None:
        return False
    _attach_score_vector(
        record, vec, openai_axes, "openai", vector_key="objective_vector_openai"
    )
    return True


def _fetch_openai_moderation(
    raw_by_id: Dict[Any, Dict[str, Any]],
    rows: List[Dict[str, Any]],
    *,
    run_id: str,
    openai_model: str = "omni-moderation-latest",
    request_delay_sec: float = 1.0,
    failure_delay_step_sec: float = 0.5,
    cache_save_every: int = 50,
) -> Dict[str, int]:
    """Fetch OpenAI moderation; disk cache by genome_id + in-run dedup by text hash."""
    import os

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to .env or set fetch_openai_missing=False."
        )

    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    from gne.evaluator import HybridModerationEvaluator  # noqa: E402
    from utils import get_custom_logging  # noqa: E402

    get_logger, _, _, _ = get_custom_logging()
    log = get_logger("emnlp_openai_fetch")

    config_path = REPO_ROOT / "config" / "RGConfig.yaml"
    evaluator = HybridModerationEvaluator(
        config_path=str(config_path),
        evaluator="openai",
        openai_model=openai_model,
    )
    if not evaluator.openai_available:
        raise RuntimeError("OpenAI moderation client is not available (check API key).")

    openai_axes = list(get_axis_order("openai"))
    cache_path = openai_cache_path(run_id)
    cache = _load_openai_cache(cache_path)
    text_cache: Dict[str, Dict[str, float]] = {}

    n_from_disk = 0
    n_from_api = 0
    n_ok = 0
    n_skipped_no_text = 0
    n_failed = 0
    pending: List[Dict[str, Any]] = []

    for record in rows:
        if record.get("objective_vector_openai") is not None:
            continue
        gid = str(record["genome_id"])
        if gid in cache:
            if _apply_openai_scores_to_record(record, cache[gid], openai_axes):
                n_from_disk += 1
            continue
        pending.append(record)

    total = len(pending)
    current_delay_sec = request_delay_sec
    print(
        f"OpenAI moderation: {n_from_disk} from cache, {total} to fetch via API "
        f"(sleep {request_delay_sec}s after each API attempt; "
        f"+{failure_delay_step_sec}s on failure)"
    )

    for i, record in enumerate(pending, 1):
        gid = str(record["genome_id"])
        raw = raw_by_id.get(record["genome_id"]) or raw_by_id.get(gid)
        if raw is None:
            n_failed += 1
            continue
        text = str(raw.get("generated_output") or "").strip()
        if not text:
            n_skipped_no_text += 1
            continue

        th = _text_hash(text)
        from_api = False
        if th in text_cache:
            scores = text_cache[th]
        else:
            result, _info = evaluator._evaluate_with_openai(text, gid)
            if current_delay_sec > 0:
                time.sleep(current_delay_sec)
            if not isinstance(result, dict) or not isinstance(result.get("scores"), dict):
                n_failed += 1
                current_delay_sec += failure_delay_step_sec
                log.warning(
                    "OpenAI moderation: no scores returned for genome %s "
                    "(next sleep %.1fs)",
                    gid,
                    current_delay_sec,
                )
                continue
            scores = {k: float(v) for k, v in result["scores"].items()}
            text_cache[th] = scores
            n_from_api += 1
            from_api = True

        cache[gid] = scores
        if not _apply_openai_scores_to_record(record, scores, openai_axes):
            n_failed += 1
            if from_api:
                current_delay_sec += failure_delay_step_sec
            if from_api:
                log.warning(
                    "OpenAI moderation: failed to apply scores for genome %s "
                    "(next sleep %.1fs)",
                    gid,
                    current_delay_sec,
                )
            else:
                log.warning(
                    "OpenAI moderation: failed to apply scores for genome %s",
                    gid,
                )
            continue

        if from_api:
            current_delay_sec = request_delay_sec
        n_ok += 1
        log.info(
            "OpenAI moderation ok genome %s (%s, %d/%d ok)",
            gid,
            "api" if from_api else "text_dedup",
            n_ok,
            total,
        )

        if i % cache_save_every == 0 or i == total:
            _save_openai_cache(cache_path, cache)
            if total > 0:
                print(
                    f"  OpenAI fetch {i}/{total} "
                    f"(ok={n_ok}, api={n_from_api}, unique_texts={len(text_cache)}, "
                    f"failed={n_failed}, sleep={current_delay_sec:.1f}s)"
                )

    _save_openai_cache(cache_path, cache)
    return {
        "n_openai_from_cache": n_from_disk,
        "n_openai_fetched": n_from_api,
        "n_openai_ok": n_ok,
        "n_openai_failed": n_failed,
        "n_openai_no_text": n_skipped_no_text,
        "n_openai_unique_texts": len(text_cache),
    }


def load_unified_genomes(
    run_path: Path,
    *,
    include_google: bool = True,
    include_openai: bool = True,
    fetch_openai_missing: bool = True,
    openai_model: str = "omni-moderation-latest",
    openai_request_delay_sec: float = 1.0,
    openai_failure_delay_step_sec: float = 0.5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load deduplicated genomes with Google + OpenAI moderation columns and embeddings."""
    google_axes = list(get_axis_order("google"))
    openai_axes = list(get_axis_order("openai")) if include_openai else []
    run_id = run_id_from_path(run_path)
    seen: Dict[Any, Dict[str, Any]] = {}
    raw_by_id: Dict[Any, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "run_id": run_id,
        "run_path": str(run_path),
        "files": {},
        "files_present": {},
        "n_with_objectives": 0,
        "n_with_google_objectives": 0,
        "n_with_openai_objectives": 0,
        "n_openai_fetched": 0,
        "n_with_embedding": 0,
        "n_duplicates_dropped": 0,
        "google_axis_order": google_axes,
        "openai_axis_order": openai_axes,
        "fetch_openai_missing": fetch_openai_missing,
    }

    for fname in POP_FILES:
        fp = run_path / fname
        stats["files_present"][fname] = fp.is_file()
        genomes = _load_json_list(fp)
        stats["files"][fname] = len(genomes)
        for g in genomes:
            gid = g.get("id")
            if gid is None:
                continue
            emb = _extract_embedding(g)
            record: Dict[str, Any] = {
                "run_id": run_id,
                "genome_id": int(gid) if str(gid).isdigit() else gid,
                "species_id": int(g.get("species_id") or 0),
                "generation": int(g.get("generation") or 0),
                "source_file": fname,
                "prompt": str(g.get("prompt") or "")[:500],
                "has_embedding": emb is not None,
            }
            if include_google:
                g_vec = _scores_row_from_genome(g, google_axes, "google")
                if g_vec is not None:
                    _attach_score_vector(
                        record, g_vec, google_axes, "google", vector_key="objective_vector"
                    )
            if include_openai:
                o_vec = _scores_row_from_genome(g, openai_axes, "openai")
                if o_vec is not None:
                    _attach_score_vector(
                        record, o_vec, openai_axes, "openai", vector_key="objective_vector_openai"
                    )
            if emb is not None:
                record["_embedding"] = emb
            if gid in seen:
                stats["n_duplicates_dropped"] += 1
            seen[gid] = record
            raw_by_id[record["genome_id"]] = g
            raw_by_id[gid] = g

    rows = list(seen.values())
    if include_openai:
        cache_path = openai_cache_path(run_id)
        stats["openai_cache_path"] = str(cache_path)
        n_from_cache = 0
        openai_axes_list = list(openai_axes)
        cache = _load_openai_cache(cache_path)
        for record in rows:
            if record.get("objective_vector_openai") is not None:
                continue
            gid = str(record["genome_id"])
            if gid in cache and _apply_openai_scores_to_record(
                record, cache[gid], openai_axes_list
            ):
                n_from_cache += 1
        stats["n_openai_from_cache"] = n_from_cache

        if fetch_openai_missing:
            fetch_stats = _fetch_openai_moderation(
                raw_by_id,
                rows,
                run_id=run_id,
                openai_model=openai_model,
                request_delay_sec=openai_request_delay_sec,
                failure_delay_step_sec=openai_failure_delay_step_sec,
            )
            stats.update(fetch_stats)

    stats["n_with_google_objectives"] = sum(1 for r in rows if "objective_vector" in r)
    stats["n_with_openai_objectives"] = sum(
        1 for r in rows if "objective_vector_openai" in r
    )
    stats["n_with_objectives"] = stats["n_with_google_objectives"]
    stats["n_with_embedding"] = sum(1 for r in rows if r.get("_embedding") is not None)
    stats["n_genomes"] = len(rows)
    stats["n_species"] = len({r["species_id"] for r in rows if r["species_id"] > 0})
    return rows, stats


def embedding_matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
    """Stack L2-normalized prompt embeddings; row order matches returned genome_ids."""
    ids: List[int] = []
    vecs: List[np.ndarray] = []
    for r in rows:
        emb = r.get("_embedding")
        if emb is not None:
            ids.append(int(r["genome_id"]))
            vecs.append(emb)
    if not vecs:
        return np.zeros((0, 384), dtype=np.float64), []
    return np.vstack(vecs), ids


def save_unified_artifacts(
    rows: List[Dict[str, Any]],
    stats: Dict[str, Any],
    out_dir: Path,
    run_id: Optional[str] = None,
) -> Dict[str, str]:
    """Export unified genome table (CSV), embeddings (.npy), ids (JSON), and load stats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = run_id or stats.get("run_id") or "run"

    skip_keys = ("_embedding", "objective_vector", "objective_vector_openai")
    export_rows = [
        {k: v for k, v in r.items() if k not in skip_keys}
        for r in rows
    ]

    csv_path = out_dir / f"{run_id}_genomes.csv"
    try:
        import pandas as pd

        pd.DataFrame(export_rows).to_csv(csv_path, index=False)
    except ImportError:
        import csv

        if export_rows:
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(export_rows[0].keys()))
                w.writeheader()
                w.writerows(export_rows)

    emb, gids = embedding_matrix(rows)
    emb_path = out_dir / f"{run_id}_embeddings.npy"
    ids_path = out_dir / f"{run_id}_genome_ids.json"
    np.save(emb_path, emb)
    ids_path.write_text(json.dumps(gids), encoding="utf-8")

    stats_path = out_dir / f"{run_id}_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    return {
        "csv": str(csv_path),
        "embeddings": str(emb_path),
        "genome_ids": str(ids_path),
        "stats": str(stats_path),
    }


def smoke_validate_run(
    run_path: Path,
    *,
    min_genomes: int = 5000,
    min_topics: int = 5,
    min_topic_size: int = 5,
    min_objective_frac: float = 0.95,
) -> Dict[str, Any]:
    """Phase 0 smoke gate: dataset size, topic coverage, and Google score completeness.

    Does not call the OpenAI API (that is Phase 1 only).
    """
    required_files = ("elites.json", "reserves.json", "archive.json")
    missing_files = [f for f in required_files if not (run_path / f).is_file()]

    rows, stats = load_unified_genomes(
        run_path,
        include_openai=False,
        fetch_openai_missing=False,
    )
    topic_sizes: Dict[int, int] = {}
    for r in rows:
        sid = int(r["species_id"])
        if sid > 0:
            topic_sizes[sid] = topic_sizes.get(sid, 0) + 1

    n_large_topics = sum(1 for n in topic_sizes.values() if n >= min_topic_size)
    n_obj = sum(1 for r in rows if "objective_vector" in r)
    frac_obj = n_obj / max(len(rows), 1)

    checks = {
        "run_exists": run_path.is_dir(),
        "required_files_present": len(missing_files) == 0,
        "min_genomes": len(rows) >= min_genomes,
        "min_large_topics": n_large_topics >= min_topics,
        "min_objective_frac": frac_obj >= min_objective_frac,
    }

    return {
        "pass": all(checks.values()),
        "checks": checks,
        "missing_files": missing_files,
        "n_genomes": len(rows),
        "n_species": stats["n_species"],
        "n_large_topics": n_large_topics,
        "frac_with_objectives": frac_obj,
        "n_with_embedding": stats["n_with_embedding"],
        "n_duplicates_dropped": stats["n_duplicates_dropped"],
        "files": stats["files"],
        "files_present": stats["files_present"],
        "topic_sizes": dict(sorted(topic_sizes.items(), key=lambda x: -x[1])),
        "thresholds": {
            "min_genomes": min_genomes,
            "min_topics": min_topics,
            "min_topic_size": min_topic_size,
            "min_objective_frac": min_objective_frac,
        },
        "run_id": stats["run_id"],
        "run_path": str(run_path),
    }
