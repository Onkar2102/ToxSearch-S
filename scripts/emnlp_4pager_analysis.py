#!/usr/bin/env python3
"""EMNLP 4-pager analytics (single module). See report/notes/EMNLP_4PAGER_PLAN.txt."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.evaluator_profiles import PERSPECTIVE_AXIS_ORDER
from utils.objectives import (
    crowding_distance,
    dominates,
    extract_objective_vector,
    fast_non_dominated_sort,
    get_axis_order,
    normalize_per_front,
)

POP_FILES = ("elites.json", "reserves.json", "archive.json", "temp.json")


# src/analytics/unified_table.py


@dataclass
class Manifest:
    primary: str
    replication: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    collapse_contrast: List[str] = field(default_factory=list)
    excluded: List[str] = field(default_factory=list)
    thresholds: Dict[str, Any] = field(default_factory=dict)
    output_root: str = "report/analytics/topic_domination"

    def all_analysis_runs(self) -> List[str]:
        return [self.primary]

    def counterfactual_runs(self) -> List[str]:
        return [self.primary]


def default_manifest() -> Manifest:
    return Manifest(
        primary="data/outputs/20260211_2122",
        excluded=[
            "data/outputs/toxsearch_s_outputs copy/run01_speciated",
            "data/outputs/toxsearch_s_outputs copy/run02_speciated",
            "data/outputs/toxsearch_s_outputs copy/run03_speciated",
            "data/outputs/toxsearch_s_outputs copy/run04_speciated",
            "data/outputs/toxsearch_s_outputs copy/run05_speciated",
            "data/outputs/20260322_1641",
            "data/outputs/20260323_0306-4",
            "validate_175_google_s42",
            "validate_175_openai_s42",
            "moea_*_20260520_*",
            "emnlp2026",
        ],
        thresholds={
            "tau_intra_cosine": 0.85,
            "tau_inter_dg": 0.35,
            "min_topic_size": 5,
            "bootstrap_n": 2000,
            "perm_n": 10000,
            "epsilon_grid": [0.0, 0.01, 0.02, 0.05, 0.10],
        },
        output_root="report/analytics/topic_domination",
    )


def load_manifest(path: Path) -> Manifest:
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Manifest(
        primary=raw["primary"],
        replication=list(raw.get("replication") or []),
        optional=list(raw.get("optional") or []),
        collapse_contrast=list(raw.get("collapse_contrast") or []),
        excluded=list(raw.get("excluded") or []),
        thresholds=dict(raw.get("thresholds") or {}),
        output_root=str(raw.get("output_root") or "report/analytics/topic_domination"),
    )


def resolve_run_path(run_ref: str, repo_root: Optional[Path] = None) -> Path:
    root = repo_root or REPO_ROOT
    p = Path(run_ref)
    if not p.is_absolute():
        p = root / p
    return p.resolve()


def run_id_from_path(run_path: Path) -> str:
    return run_path.name.replace(" ", "_")


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


def _objective_row(genome: Dict[str, Any], axis_order: Sequence[str]) -> Optional[np.ndarray]:
    if genome.get("objective_vector") is not None:
        v = np.asarray(genome["objective_vector"], dtype=np.float64).reshape(-1)
        if v.size == len(axis_order):
            return np.clip(v, 0.0, 1.0)
    try:
        v = extract_objective_vector(genome, evaluator="google")
        return np.clip(v, 0.0, 1.0)
    except (KeyError, TypeError, ValueError):
        pass
    mr = genome.get("moderation_result") or {}
    scores = None
    if isinstance(mr, dict):
        ggl = mr.get("google")
        if isinstance(ggl, dict) and isinstance(ggl.get("scores"), dict):
            scores = ggl["scores"]
    if not isinstance(scores, dict):
        return None
    row = []
    for axis in axis_order:
        val = scores.get(axis)
        if val is None:
            return None
        try:
            row.append(float(np.clip(float(val), 0.0, 1.0)))
        except (TypeError, ValueError):
            return None
    return np.asarray(row, dtype=np.float64)


def load_unified_genomes(
    run_path: Path,
    *,
    evaluator: str = "google",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load deduplicated genomes with objective vectors and embeddings."""
    axis_order = list(get_axis_order(evaluator))
    run_id = run_id_from_path(run_path)
    seen: Dict[Any, Dict[str, Any]] = {}
    stats: Dict[str, Any] = {
        "run_id": run_id,
        "run_path": str(run_path),
        "files": {},
        "n_with_objectives": 0,
        "n_with_embedding": 0,
        "n_duplicates_dropped": 0,
    }

    for fname in POP_FILES:
        fp = run_path / fname
        genomes = _load_json_list(fp)
        stats["files"][fname] = len(genomes)
        for g in genomes:
            gid = g.get("id")
            if gid is None:
                continue
            row_vec = _objective_row(g, axis_order)
            emb = _extract_embedding(g)
            record = {
                "run_id": run_id,
                "genome_id": int(gid) if str(gid).isdigit() else gid,
                "species_id": int(g.get("species_id") or 0),
                "generation": int(g.get("generation") or 0),
                "source_file": fname,
                "prompt": str(g.get("prompt") or "")[:500],
                "has_embedding": emb is not None,
            }
            if row_vec is not None:
                for i, axis in enumerate(axis_order):
                    record[f"f_{axis}"] = float(row_vec[i])
                record["objective_vector"] = row_vec
            if emb is not None:
                record["_embedding"] = emb
            if gid in seen:
                stats["n_duplicates_dropped"] += 1
            seen[gid] = record

    rows = list(seen.values())
    stats["n_with_objectives"] = sum(1 for r in rows if "objective_vector" in r)
    stats["n_with_embedding"] = sum(1 for r in rows if r.get("_embedding") is not None)
    stats["n_genomes"] = len(rows)
    stats["n_species"] = len({r["species_id"] for r in rows if r["species_id"] > 0})
    stats["axis_order"] = axis_order
    stats["evaluator"] = evaluator
    return rows, stats


def objective_matrix(rows: Sequence[Dict[str, Any]], axis_order: Sequence[str]) -> np.ndarray:
    valid = [r for r in rows if "objective_vector" in r]
    if not valid:
        return np.zeros((0, len(axis_order)), dtype=np.float64)
    return np.vstack([r["objective_vector"] for r in valid])


def embedding_matrix(rows: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, List[int]]:
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
    run_id: str,
) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    axis_order = stats.get("axis_order") or list(PERSPECTIVE_AXIS_ORDER)
    export_rows = []
    for r in rows:
        er = {k: v for k, v in r.items() if k != "_embedding" and k != "objective_vector"}
        export_rows.append(er)

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


def smoke_validate_run(run_path: Path, min_genomes: int = 5000, min_topics: int = 5, min_topic_size: int = 5) -> Dict[str, Any]:
    rows, stats = load_unified_genomes(run_path)
    topic_sizes: Dict[int, int] = {}
    for r in rows:
        sid = int(r["species_id"])
        if sid > 0:
            topic_sizes[sid] = topic_sizes.get(sid, 0) + 1
    n_large_topics = sum(1 for n in topic_sizes.values() if n >= min_topic_size)
    n_obj = sum(1 for r in rows if "objective_vector" in r)
    frac_obj = n_obj / max(len(rows), 1)
    return {
        "pass": len(rows) >= min_genomes and n_large_topics >= min_topics and frac_obj >= 0.95,
        "n_genomes": len(rows),
        "n_large_topics": n_large_topics,
        "frac_with_objectives": frac_obj,
        "topic_sizes": dict(sorted(topic_sizes.items(), key=lambda x: -x[1])[:15]),
    }



# src/analytics/topic_domination.py


def global_pareto_annotate(F: np.ndarray) -> Tuple[np.ndarray, np.ndarray, List[List[int]]]:
    """Return (global_rank, on_f0_mask, fronts)."""
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
    """True if row is dominated by at least one other row in F."""
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


def phenotype_distance(f1: np.ndarray, f2: np.ndarray, m: int) -> float:
    return float(np.linalg.norm(f1 - f2) / np.sqrt(max(m, 1)))


def topic_centroids(
    rows: Sequence[Dict[str, Any]],
    axis_order: Sequence[str],
) -> Dict[int, Dict[str, Any]]:
    by_topic: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        sid = int(r.get("species_id") or 0)
        if sid > 0 and "objective_vector" in r:
            by_topic[sid].append(r)

    out: Dict[int, Dict[str, Any]] = {}
    for sid, members in by_topic.items():
        F = np.vstack([m["objective_vector"] for m in members])
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
    *,
    min_topic_size: int = 5,
    tau_intra: float = 0.85,
    tau_inter: float = 0.35,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    axis_order = list(get_axis_order("google"))
    valid = [r for r in rows if "objective_vector" in r]
    if not valid:
        return [], {"error": "no objective vectors"}

    F = np.vstack([r["objective_vector"] for r in valid])
    ranks, on_f0, fronts = global_pareto_annotate(F)
    dominated = member_dominated_mask(F)

    for i, r in enumerate(valid):
        r["pareto_rank_global"] = int(ranks[i])
        r["on_global_f0"] = bool(on_f0[i])
        r["tdi_member"] = bool(dominated[i])

    f0_max = F[on_f0].max(axis=0) if np.any(on_f0) else F.max(axis=0)

    by_topic: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic[sid].append(i)

    centroids = topic_centroids(valid, axis_order)
    summaries: List[Dict[str, Any]] = []

    for sid, idxs in sorted(by_topic.items()):
        n = len(idxs)
        if n < min_topic_size:
            continue
        sub_dom = dominated[idxs]
        tdi = float(np.mean(sub_dom))
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
            "species_id": sid,
            "n_members": n,
            "n_on_f0": n_on_f0,
            "tdi": tdi,
            "fully_dominated": n_on_f0 == 0,
            "distinct_topic": distinct,
            "axis_exclusive_axes": ";".join(axis_exclusive),
            "n_axis_exclusive": len(axis_exclusive),
            "intra_cosine_mean": intra,
            "inter_dg_min": inter_min,
            "dominating_topics": ";".join(str(x) for x in dominating),
            "max_toxicity": float(max_vec[0]),
        })

    global_stats = {
        "n_genomes": len(valid),
        "n_f0": int(np.sum(on_f0)),
        "f0_fraction": float(np.mean(on_f0)),
        "n_fronts": len(fronts),
        "n_topics_summarized": len(summaries),
        "n_fully_dominated": sum(1 for s in summaries if s["fully_dominated"]),
        "n_distinct_dominated": sum(
            1 for s in summaries if s["fully_dominated"] and s["distinct_topic"]
        ),
        "n_axis_exclusive_dominated": sum(
            1 for s in summaries
            if s["fully_dominated"] and s["n_axis_exclusive"] > 0
        ),
    }
    return summaries, global_stats


def topic_domination_digraph(summaries: Sequence[Dict[str, Any]], centroids: Dict[int, Dict[str, Any]]) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    ids = [s["species_id"] for s in summaries]
    for s in summaries:
        c = s["species_id"]
        max_c = centroids[c]["max_vector"]
        for c2 in ids:
            if c2 == c:
                continue
            if dominates(centroids[c2]["max_vector"], max_c):
                edges.append((c2, c))
    return edges


def run_topic_domination_analysis(
    rows: Sequence[Dict[str, Any]],
    *,
    min_topic_size: int = 5,
    tau_intra: float = 0.85,
    tau_inter: float = 0.35,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Tuple[int, int]]]:
    summaries, stats = compute_topic_summaries(
        rows,
        min_topic_size=min_topic_size,
        tau_intra=tau_intra,
        tau_inter=tau_inter,
    )
    axis_order = list(get_axis_order("google"))
    valid = [r for r in rows if "objective_vector" in r]
    centroids = topic_centroids(valid, axis_order)
    edges = topic_domination_digraph(summaries, centroids) if summaries else []
    return summaries, stats, edges


def write_topic_outputs(
    out_dir: Path,
    run_id: str,
    summaries: List[Dict[str, Any]],
    stats: Dict[str, Any],
    edges: List[Tuple[int, int]],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    import csv

    csv_path = out_dir / "topic_domination_summary.csv"
    if summaries:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summaries[0].keys()))
            w.writeheader()
            w.writerows(summaries)
    (out_dir / "global_pareto_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    edge_rows = [{"from": a, "to": b} for a, b in edges]
    (out_dir / "topic_domination_edges.json").write_text(json.dumps(edge_rows, indent=2), encoding="utf-8")



# src/analytics/counterfactual.py






def _indices_by_topic(rows: Sequence[Dict[str, Any]], min_topic_size: int) -> Dict[int, List[int]]:
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(rows):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)
    return {sid: idxs for sid, idxs in by_topic.items() if len(idxs) >= min_topic_size}


def _trim_front(F: np.ndarray, idxs: List[int], cap: int) -> List[int]:
    if len(idxs) <= cap:
        return list(idxs)
    sub = F[idxs]
    fronts = fast_non_dominated_sort(sub)
    kept: List[int] = []
    for front in fronts:
        if len(kept) + len(front) <= cap:
            kept.extend(idxs[i] for i in front)
            continue
        remaining = cap - len(kept)
        if remaining <= 0:
            break
        cd = crowding_distance(sub, front)
        order = sorted(range(len(front)), key=lambda k: (-cd[k], idxs[front[k]]))
        kept.extend(idxs[front[order[i]]] for i in range(remaining))
        break
    return kept


def policy_scalar(rows: Sequence[Dict[str, Any]], k: int) -> Set[int]:
    scored = [(i, float(r["objective_vector"][0])) for i, r in enumerate(rows) if "objective_vector" in r]
    scored.sort(key=lambda x: (-x[1], x[0]))
    return {i for i, _ in scored[:k]}


def policy_global_mo(rows: Sequence[Dict[str, Any]], cap: int) -> Set[int]:
    valid_idx = [i for i, r in enumerate(rows) if "objective_vector" in r]
    if not valid_idx:
        return set()
    F = np.vstack([rows[i]["objective_vector"] for i in valid_idx])
    fronts = fast_non_dominated_sort(F)
    f0_local = fronts[0] if fronts else list(range(len(valid_idx)))
    f0_global = [valid_idx[i] for i in f0_local]
    if len(f0_global) <= cap:
        return set(f0_global)
    F0 = F[f0_local]
    Fn = normalize_per_front(F0, list(range(len(f0_local))))
    cd = crowding_distance(Fn, list(range(len(f0_local))))
    order = sorted(range(len(f0_local)), key=lambda k: (-cd[k], f0_global[k]))
    return {f0_global[i] for i in order[:cap]}


def policy_speciated_oracle(rows: Sequence[Dict[str, Any]], min_topic_size: int) -> Set[int]:
    by_topic = _indices_by_topic(rows, min_topic_size)
    kept: Set[int] = set()
    for idxs in by_topic.values():
        F = np.vstack([rows[i]["objective_vector"] for i in idxs])
        fronts = fast_non_dominated_sort(F)
        if fronts:
            kept.update(idxs[i] for i in fronts[0])
    return kept


def topic_survival(
    kept: Set[int],
    rows: Sequence[Dict[str, Any]],
    min_topic_size: int,
) -> Dict[str, Any]:
    by_topic = _indices_by_topic(rows, min_topic_size)
    total = len(by_topic)
    survived = sum(1 for sid, idxs in by_topic.items() if any(i in kept for i in idxs))
    rate = survived / total if total else 0.0
    return {
        "topics_total": total,
        "topics_survived": survived,
        "survival_rate": rate,
        "extinction_rate": 1.0 - rate,
    }


def run_counterfactual(
    rows: Sequence[Dict[str, Any]],
    *,
    min_topic_size: int = 5,
) -> List[Dict[str, Any]]:
    valid = [r for r in rows if "objective_vector" in r]
    if not valid:
        return []
    # Map back to indices in rows
    row_list = list(rows)
    mo_kept = policy_global_mo(row_list, cap=len(valid))
    k = len(mo_kept) if mo_kept else len(valid)
    results = []
    for policy_name, kept in [
        ("scalar", policy_scalar(row_list, k)),
        ("global_mo", mo_kept),
        ("speciated_oracle", policy_speciated_oracle(row_list, min_topic_size)),
    ]:
        stats = topic_survival(kept, row_list, min_topic_size)
        stats["policy"] = policy_name
        stats["kept_size"] = len(kept)
        results.append(stats)
    return results



# src/analytics/temporal.py



matplotlib.use("Agg")


def load_tracker_series(run_path: Path) -> List[Dict[str, Any]]:
    tracker_path = run_path / "EvolutionTracker.json"
    if not tracker_path.is_file():
        return []
    with tracker_path.open(encoding="utf-8") as f:
        tracker = json.load(f)
    rows: List[Dict[str, Any]] = []
    for g in tracker.get("generations") or []:
        sp = g.get("speciation") or {}
        sc = sp.get("species_count")
        if sc in (0, None) and sp.get("active_species_count") is not None:
            sc = sp.get("active_species_count")
        rows.append({
            "generation": int(g.get("generation_number") or 0),
            "species_count": int(sc or 0),
            "inter_species_diversity": float(sp.get("inter_species_diversity") or 0.0),
            "max_score_variants": float(g.get("max_score_variants") or 0.0),
            "selection_mode": str(g.get("selection_mode") or ""),
            "variants_created": int(g.get("variants_created") or 0),
        })
    return rows


def write_timeseries_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_temporal(rows: List[Dict[str, Any]], out_path: Path, title: str = "") -> Optional[str]:
    if not rows:
        return None
    gens = [r["generation"] for r in rows]
    sc = [r["species_count"] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(gens, sc, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Species count")
    ax.set_title(title or "Topic count over generations")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)



# src/analytics/behavioural_viz.py



matplotlib.use("Agg")



def _umap_coords(embeddings: np.ndarray) -> np.ndarray:
    try:
        import umap

        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        return reducer.fit_transform(embeddings)
    except ImportError:
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2, random_state=42)
        return pca.fit_transform(embeddings)


def plot_umap_species(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    *,
    max_points: int = 3000,
) -> Optional[str]:
    pts = [(r, r["_embedding"]) for r in rows if r.get("_embedding") is not None and "objective_vector" in r]
    if len(pts) < 10:
        return None
    if len(pts) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = [pts[i] for i in sorted(idx)]

    embs = np.vstack([p[1] for p in pts])
    coords = _umap_coords(embs)
    fig, ax = plt.subplots(figsize=(7, 5))
    for (r, _), (x, y) in zip(pts, coords):
        sid = int(r.get("species_id") or 0)
        on_f0 = bool(r.get("on_global_f0"))
        color = plt.cm.tab20(sid % 20) if sid > 0 else "#999999"
        marker = "o" if on_f0 else "x"
        ax.scatter(x, y, c=[color], marker=marker, s=18, alpha=0.7)
    ax.set_title("UMAP: species colour; circle=on F0, x=dominated")
    ax.set_xticks([])
    ax.set_yticks([])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def plot_centroid_dg_dp(
    rows: Sequence[Dict[str, Any]],
    summaries: Sequence[Dict[str, Any]],
    out_path: Path,
) -> Optional[str]:
    axis_order = list(get_axis_order("google"))
    m = len(axis_order)
    centroids = topic_centroids([r for r in rows if "objective_vector" in r], axis_order)
    sids = [s["species_id"] for s in summaries if s.get("distinct_topic")]
    if len(sids) < 2:
        sids = [s["species_id"] for s in summaries]
    if len(sids) < 2:
        return None

    xs, ys, cols = [], [], []
    for i, a in enumerate(sids):
        for b in sids[i + 1:]:
            ca, cb = centroids[a], centroids[b]
            if ca.get("embedding_centroid") is None or cb.get("embedding_centroid") is None:
                continue
            dg = genotype_distance(ca["embedding_centroid"], cb["embedding_centroid"])
            dp = phenotype_distance(ca["mean_vector"], cb["mean_vector"], m)
            xs.append(dg)
            ys.append(dp)
            fa = next((s for s in summaries if s["species_id"] == a), {})
            fb = next((s for s in summaries if s["species_id"] == b), {})
            dominated = fa.get("fully_dominated") or fb.get("fully_dominated")
            cols.append("#c0392b" if dominated else "#2980b9")

    if not xs:
        return None
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(xs, ys, c=cols, alpha=0.75)
    ax.set_xlabel("Genotype distance d_g (centroids)")
    ax.set_ylabel("Phenotype distance d_p (centroids)")
    ax.set_title("Topic centroid separation")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out_path)


def compute_cluster_quality(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Silhouette / DB / CH on (embeddings, species_id) for rows with both."""
    embs: List[np.ndarray] = []
    labels: List[int] = []
    for r in rows:
        emb = r.get("_embedding")
        sid = int(r.get("species_id") or 0)
        if emb is not None and sid > 0:
            embs.append(np.asarray(emb, dtype=np.float64).reshape(-1))
            labels.append(sid)
    if len(set(labels)) < 2 or len(labels) < 10:
        return {"num_samples": len(labels), "num_clusters": len(set(labels)), "skipped": True}
    X = np.vstack(embs)
    y = np.asarray(labels)
    from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

    return {
        "silhouette_score": float(silhouette_score(X, y)),
        "davies_bouldin_index": float(davies_bouldin_score(X, y)),
        "calinski_harabasz_index": float(calinski_harabasz_score(X, y)),
        "num_samples": int(len(labels)),
        "num_clusters": int(len(set(labels))),
        "skipped": False,
    }



# src/analytics/inference.py






def bootstrap_dominated_distinct_rate(
    run_rates: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    if not run_rates:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0}
    rng = np.random.default_rng(seed)
    arr = np.asarray(run_rates, dtype=np.float64)
    boots = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        boots.append(float(np.mean(sample)))
    boots.sort()
    lo = boots[int(0.025 * n_boot)]
    hi = boots[int(0.975 * n_boot)]
    return {"mean": float(np.mean(arr)), "ci_lo": lo, "ci_hi": hi, "n_runs": len(arr)}


def permutation_embedding_separation(
    rows: Sequence[Dict[str, Any]],
    *,
    n_perm: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Test if mean inter-group genotype distance exceeds label-shuffled null."""
    labeled = []
    for r in rows:
        emb = r.get("_embedding")
        if emb is None:
            continue
        if r.get("on_global_f0"):
            labeled.append((emb, 1))
        elif r.get("tdi_member") and int(r.get("species_id") or 0) > 0:
            labeled.append((emb, 0))

    if len(labeled) < 20:
        return {"p_value": 1.0, "obs_stat": 0.0, "n": len(labeled), "skipped": True}

    def stat(pairs):
        g0 = [e for e, y in pairs if y == 0]
        g1 = [e for e, y in pairs if y == 1]
        if not g0 or not g1:
            return 0.0
        dists = []
        for a in g0[:50]:
            for b in g1[:50]:
                dists.append(genotype_distance(a, b))
        return float(np.mean(dists)) if dists else 0.0

    obs = stat(labeled)
    rng = np.random.default_rng(seed)
    count = 0
    labels = [y for _, y in labeled]
    for _ in range(n_perm):
        perm_labels = rng.permutation(labels)
        pairs = [(labeled[i][0], int(perm_labels[i])) for i in range(len(labeled))]
        if stat(pairs) >= obs:
            count += 1
    p = (count + 1) / (n_perm + 1)
    return {"p_value": p, "obs_stat": obs, "n": len(labeled), "skipped": False}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def write_inference_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")



# src/analytics/sensitivity.py


def epsilon_dominates(a: np.ndarray, b: np.ndarray, eps: float) -> bool:
    return bool(np.all(a >= b + eps) and np.any(a > b + eps))


def epsilon_dominated_count(F: np.ndarray, eps: float) -> int:
    n = F.shape[0]
    cnt = 0
    for p in range(n):
        for q in range(n):
            if p != q and epsilon_dominates(F[q], F[p], eps):
                cnt += 1
                break
    return cnt


def epsilon_dominated_f0_mask(F: np.ndarray, eps: float) -> np.ndarray:
    """First front under epsilon-dominance (maximization)."""
    n = F.shape[0]
    if n == 0:
        return np.array([], dtype=bool)
    if eps <= 0:
        _, on_f0, _ = global_pareto_annotate(F)
        return on_f0
    ge = np.all(F[:, None, :] >= F[None, :, :] + eps, axis=2)
    gt = np.any(F[:, None, :] > F[None, :, :] + eps, axis=2)
    dom = ge & gt
    np.fill_diagonal(dom, False)
    return ~dom.any(axis=0)


def epsilon_sweep(rows: Sequence[Dict[str, Any]], grid: Sequence[float], min_topic_size: int) -> List[Dict[str, Any]]:
    valid = [r for r in rows if "objective_vector" in r]
    if not valid:
        return []
    F = np.vstack([r["objective_vector"] for r in valid])
    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)

    out = []
    for eps in grid:
        f0_mask = epsilon_dominated_f0_mask(F, eps)
        n_fully_dom_topics = 0
        for sid, idxs in by_topic.items():
            if len(idxs) < min_topic_size:
                continue
            if not any(f0_mask[i] for i in idxs):
                n_fully_dom_topics += 1
        out.append({"epsilon": eps, "n_f0": int(np.sum(f0_mask)), "n_fully_dominated_topics": n_fully_dom_topics})
    return out


def threshold_sweep(
    rows: Sequence[Dict[str, Any]],
    tau_intra_grid: Sequence[float],
    tau_inter_grid: Sequence[float],
    min_topic_size: int,
) -> List[Dict[str, Any]]:
    axis_order = list(get_axis_order("google"))
    valid = [r for r in rows if "objective_vector" in r]
    if not valid:
        return []
    F = np.vstack([r["objective_vector"] for r in valid])
    _, on_f0, _ = global_pareto_annotate(F)
    centroids = topic_centroids(valid, axis_order)

    by_topic: Dict[int, List[int]] = {}
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic.setdefault(sid, []).append(i)

    base_topics = []
    for sid, idxs in sorted(by_topic.items()):
        if len(idxs) < min_topic_size:
            continue
        n_on_f0 = int(np.sum(on_f0[idxs]))
        max_vec = F[idxs].max(axis=0)
        f0_max = F[on_f0].max(axis=0) if np.any(on_f0) else F.max(axis=0)
        axis_exclusive = sum(1 for i in range(len(axis_order)) if max_vec[i] > f0_max[i] + 1e-9)
        base_topics.append({
            "species_id": sid,
            "fully_dominated": n_on_f0 == 0,
            "n_axis_exclusive": axis_exclusive,
        })

    out = []
    for ti in tau_intra_grid:
        for te in tau_inter_grid:
            n_distinct_dominated = 0
            n_axis_exclusive_dominated = 0
            for topic in base_topics:
                sid = topic["species_id"]
                distinct, _, _ = distinct_topic_label(
                    sid, centroids, tau_intra=ti, tau_inter=te
                )
                if topic["fully_dominated"] and distinct:
                    n_distinct_dominated += 1
                if topic["fully_dominated"] and topic["n_axis_exclusive"] > 0:
                    n_axis_exclusive_dominated += 1
            out.append({
                "tau_intra": ti,
                "tau_inter": te,
                "n_distinct_dominated": n_distinct_dominated,
                "n_axis_exclusive_dominated": n_axis_exclusive_dominated,
            })
    return out


def population_ablation(rows: Sequence[Dict[str, Any]], source: str, min_topic_size: int) -> Dict[str, Any]:
    if source == "elites":
        sub = [r for r in rows if r.get("source_file") == "elites.json"]
    elif source == "no_reserves":
        sub = [r for r in rows if r.get("source_file") != "reserves.json"]
    else:
        sub = list(rows)
    _, stats = compute_topic_summaries(sub, min_topic_size=min_topic_size)
    stats["population"] = source
    return stats


def write_sensitivity_outputs(out_dir: Path, payload: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sensitivity.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")



# export figures

#!/usr/bin/env python3
"""Copy analytics figures to paper-ready paths."""




def _plot_counterfactual_bar(analytics_root: Path, paper_dir: Path) -> None:
    cf = analytics_root / "counterfactual_survival.csv"
    if not cf.is_file():
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    by_policy: dict[str, float] = {}
    with cf.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            by_policy[row["policy"]] = float(row["survival_rate"])

    order = ["scalar", "global_mo", "speciated_oracle"]
    labels = ["Scalar", "Global MO", "Speciated oracle"]
    means = [by_policy.get(p, 0.0) for p in order]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.bar(labels, means, color=["#888888", "#4477AA", "#228833"])
    ax.set_ylabel("Topic survival rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Counterfactual retention (20260211_2122)")
    fig.tight_layout()
    fig.savefig(paper_dir / "fig3_counterfactual.png", dpi=300)
    plt.close(fig)


def export_figures(analytics_root: Path) -> None:
    fig_dir = analytics_root / "figures"
    paper_dir = analytics_root / "paper_figures"
    paper_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "fig1_temporal_species_count.png": "fig1_temporal.png",
        "fig1_umap.png": "fig1_umap.png",
    }
    post_hoc = fig_dir / "post_hoc"
    if post_hoc.is_dir():
        for name in ("pymoo_pcp_pareto.png", "pareto_global_front_drivers.png"):
            src = post_hoc / name
            if src.is_file():
                shutil.copy2(src, paper_dir / name)

    for src_name, dst_name in mapping.items():
        src = fig_dir / src_name
        if src.is_file():
            shutil.copy2(src, paper_dir / dst_name)

    cf = analytics_root / "counterfactual_survival.csv"
    if cf.is_file():
        shutil.copy2(cf, paper_dir / "counterfactual_survival.csv")

    _plot_counterfactual_bar(analytics_root, paper_dir)
    print(f"Paper figures staged in {paper_dir}")


def run_cross_evaluator(run_dir: Path, out_path: Path) -> None:
    rows, _ = load_unified_genomes(run_dir.resolve())
    summaries_g, _, _ = run_topic_domination_analysis(rows)
    dom_g = {s["species_id"] for s in summaries_g if s["fully_dominated"] and s["distinct_topic"]}
    openai_path = out_path.parent / "openai_rescores.jsonl"
    n_openai = 0
    if openai_path.is_file():
        n_openai = sum(1 for ln in openai_path.read_text().splitlines() if ln.strip())
    result = {
        "google_n_distinct_dominated": len(dom_g),
        "openai_rescores_available": n_openai,
        "jaccard": None,
        "note": "No openai_rescores.jsonl" if not n_openai else "Partial merge pending",
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(result.keys()))
        w.writeheader()
        w.writerow(result)



# orchestrator


def _meta_row(run_id: str, summaries: List[Dict], stats: Dict) -> Dict[str, Any]:
    n_topics = len(summaries)
    n_dist_dom = sum(1 for s in summaries if s["fully_dominated"] and s["distinct_topic"])
    return {
        "run_id": run_id,
        "n_genomes": stats.get("n_genomes", 0),
        "n_topics": n_topics,
        "n_fully_dominated": stats.get("n_fully_dominated", 0),
        "n_distinct_dominated": n_dist_dom,
        "n_axis_exclusive_dominated": stats.get("n_axis_exclusive_dominated", 0),
        "f0_fraction": stats.get("f0_fraction", 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional YAML manifest override (default: built-in 2122-only config)",
    )
    parser.add_argument("--paper-figures", action="store_true")
    parser.add_argument("--smoke-primary", action="store_true")
    args = parser.parse_args()

    manifest = load_manifest(args.manifest) if args.manifest else default_manifest()
    out_root = REPO_ROOT / manifest.output_root
    thresholds = manifest.thresholds
    min_topic = int(thresholds.get("min_topic_size", 5))
    tau_intra = float(thresholds.get("tau_intra_cosine", 0.85))
    tau_inter = float(thresholds.get("tau_inter_dg", 0.35))

    primary_path = resolve_run_path(manifest.primary, REPO_ROOT)
    gate0 = smoke_validate_run(primary_path)
    print("Gate 0:", "PASS" if gate0["pass"] else "FAIL", gate0)
    if args.smoke_primary:
        print(json.dumps(gate0, indent=2))
        sys.exit(0 if gate0["pass"] else 1)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "gate0_smoke.json").write_text(json.dumps(gate0, indent=2), encoding="utf-8")
    unified_dir = out_root / "unified"
    meta_rows: List[Dict[str, Any]] = []
    primary_rows = None
    primary_summaries = None
    primary_stats = None

    # Phase 1–3 per run
    for run_ref in manifest.all_analysis_runs():
        run_path = resolve_run_path(run_ref, REPO_ROOT)
        if not run_path.is_dir():
            print(f"SKIP {run_ref}")
            continue
        run_id = run_id_from_path(run_path)
        rows, stats = load_unified_genomes(run_path)
        save_unified_artifacts(rows, stats, unified_dir, run_id)

        summaries, gstats, edges = run_topic_domination_analysis(
            rows, min_topic_size=min_topic, tau_intra=tau_intra, tau_inter=tau_inter
        )
        run_out = out_root / run_id
        write_topic_outputs(run_out, run_id, summaries, gstats, edges)
        meta_rows.append(_meta_row(run_id, summaries, gstats))

        if run_id == run_id_from_path(primary_path):
            primary_rows = rows
            primary_summaries = summaries
            primary_stats = gstats

    # Phase 4 temporal
    figures_dir = out_root / "figures"
    ts = load_tracker_series(primary_path)
    write_timeseries_csv(ts, out_root / f"timeseries_{run_id_from_path(primary_path)}.csv")
    plot_temporal(ts, figures_dir / "fig1_temporal_species_count.png", title="2122: species over generations")

    if primary_rows and primary_summaries:
        plot_umap_species(primary_rows, figures_dir / "fig1_umap.png")
        plot_centroid_dg_dp(primary_rows, primary_summaries, figures_dir / "appendix_dg_dp.png")
        cq = compute_cluster_quality(primary_rows)
        (out_root / "cluster_quality_primary.json").write_text(
            json.dumps(cq, indent=2), encoding="utf-8"
        )

    # Phase 5 counterfactual
    cf_rows = []
    for run_ref in manifest.counterfactual_runs():
        run_path = resolve_run_path(run_ref, REPO_ROOT)
        if not run_path.is_dir():
            continue
        rows, _ = load_unified_genomes(run_path)
        for r in run_counterfactual(rows, min_topic_size=min_topic):
            r["run_id"] = run_id_from_path(run_path)
            cf_rows.append(r)
    cf_path = out_root / "counterfactual_survival.csv"
    if cf_rows:
        with cf_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(cf_rows[0].keys()))
            w.writeheader()
            w.writerows(cf_rows)

    # Phase 7 inference (within-run permutation only; single run, no cross-run bootstrap)
    perm = permutation_embedding_separation(
        primary_rows or [], n_perm=int(thresholds.get("perm_n", 10000))
    )
    write_inference_json(out_root / "inference_permutation.json", perm)

    meta_path = out_root / "meta_analysis_by_run.csv"
    if meta_rows:
        with meta_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(meta_rows[0].keys()))
            w.writeheader()
            w.writerows(meta_rows)

    # Phase 8 sensitivity (primary)
    if primary_rows:
        eps_grid = thresholds.get("epsilon_grid") or [0.0, 0.01, 0.02, 0.05, 0.10]
        print("Phase 8: sensitivity...")
        sens = {
            "epsilon_sweep": epsilon_sweep(primary_rows, eps_grid, min_topic),
            "threshold_sweep": threshold_sweep(
                primary_rows,
                [0.80, 0.85, 0.90],
                [0.25, 0.35, 0.45],
                min_topic,
            ),
            "population_ablation": [
                population_ablation(primary_rows, s, min_topic)
                for s in ("full", "elites", "no_reserves")
            ],
        }
        write_sensitivity_outputs(out_root / "sensitivity", sens)
        print("Phase 8: done")

    # Phase 6 cross-evaluator (optional if rescores absent)
    try:
        run_cross_evaluator(primary_path, out_root / "cross_evaluator_agreement.csv")
    except Exception as e:
        print(f"cross_eval skipped: {e}")

    # Gate summaries
    gates = {
        "gate0": gate0,
        "gate2": {
            "pass": bool(primary_stats and primary_stats.get("n_axis_exclusive_dominated", 0) >= 1),
            "stats": primary_stats,
        },
        "gate5": _gate5(cf_rows, run_id_from_path(primary_path)),
        "gate7": {"permutation": perm},
    }
    (out_root / "gates.json").write_text(json.dumps(gates, indent=2), encoding="utf-8")

    # Phase 10 snippet
    _write_results_snippet(out_root, gates, primary_stats, perm)

    if args.paper_figures:
        try:
            from utils.post_hoc.runner import run_post_hoc_analysis

            run_post_hoc_analysis(primary_path, out_dir=figures_dir / "post_hoc")
        except Exception as e:
            print(f"post_hoc skipped: {e}")

        export_figures(out_root)

    print(f"Done. Outputs under {out_root}")


def _gate5(cf_rows: List[Dict], primary_id: str) -> Dict[str, Any]:
    by_policy = {r["policy"]: r for r in cf_rows if r.get("run_id") == primary_id}
    mo = by_policy.get("global_mo", {})
    ora = by_policy.get("speciated_oracle", {})
    pass_ = (
        float(mo.get("extinction_rate", 0)) > float(ora.get("extinction_rate", 0))
        if mo and ora
        else False
    )
    return {"pass": pass_, "global_mo": mo, "speciated_oracle": ora}


def _write_results_snippet(out_root, gates, stats, perm) -> None:
    s = stats or {}
    g2 = gates.get("gate2", {})
    g5 = gates.get("gate5", {})
    mo = g5.get("global_mo") or {}
    ora = g5.get("speciated_oracle") or {}
    snippet = f"""# Results snippet (auto-generated)

Run 20260211_2122 ({s.get('n_genomes', '?')} genomes): {s.get('n_fully_dominated', 0)} fully dominated topics, {s.get('n_f0', '?')} genomes on global F0 ({100 * s.get('f0_fraction', 0):.1f}%). Distinct + fully dominated topics: {s.get('n_distinct_dominated', 0)}; axis-exclusive dominated: {s.get('n_axis_exclusive_dominated', 0)} (Gate 2: {g2.get('pass')}).

Counterfactual retention (2122 only): global MO topic extinction {float(mo.get('extinction_rate', 0)):.0%} vs speciated oracle {float(ora.get('extinction_rate', 0)):.0%} (Gate 5: {g5.get('pass')}).

Embedding-separation permutation p={perm.get('p_value', 1):.4f}.
"""
    (out_root / "RESULTS_SNIPPET.md").write_text(snippet, encoding="utf-8")


if __name__ == "__main__":
    main()
