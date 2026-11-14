"""Shared RQ2 run-level metrics: top-10 anytime, MiniLM silhouette, genomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

NOISE_SIDS = {None, 0, -1, "0", "-1"}
EMBEDDING_CACHE_NAME = "prompt_embeddings.npz"
EMBEDDING_DIM = 384


def embedding_cache_path(run_dir: Path) -> Path:
    return Path(run_dir) / EMBEDDING_CACHE_NAME


def load_embedding_cache(run_dir: Path) -> Dict[str, np.ndarray]:
    """Load id -> (384,) vectors from ``prompt_embeddings.npz``, if present."""
    path = embedding_cache_path(run_dir)
    if not path.is_file():
        return {}
    try:
        data = np.load(path, allow_pickle=False)
        ids = data["ids"]
        embs = np.asarray(data["embeddings"], dtype=np.float32)
    except Exception:
        return {}
    if embs.ndim != 2 or embs.shape[1] != EMBEDDING_DIM:
        return {}
    out: Dict[str, np.ndarray] = {}
    for i, rid in enumerate(ids.tolist()):
        out[str(rid)] = embs[i]
    return out


def save_embedding_cache(
    run_dir: Path,
    ids: Sequence[str],
    embeddings: np.ndarray,
) -> Path:
    """Write L2-ready MiniLM cache; ``embeddings`` must be shape (n, 384)."""
    path = embedding_cache_path(run_dir)
    arr = np.asarray(embeddings, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"expected (n, {EMBEDDING_DIM}) embeddings, got {arr.shape}")
    if len(ids) != arr.shape[0]:
        raise ValueError("ids and embeddings length mismatch")
    id_arr = np.asarray([str(x) for x in ids])
    np.savez_compressed(path, ids=id_arr, embeddings=arr)
    return path


def attach_cached_embeddings(
    genomes: Sequence[Dict[str, Any]],
    run_dir: Path,
    cache: Optional[Mapping[str, np.ndarray]] = None,
) -> List[Dict[str, Any]]:
    """Return shallow copies with ``prompt_embedding`` filled from the NPZ cache."""
    if cache is None:
        cache = load_embedding_cache(run_dir)
    if not cache:
        return list(genomes)
    out: List[Dict[str, Any]] = []
    for rec in genomes:
        if not isinstance(rec, dict):
            continue
        e = rec.get("prompt_embedding")
        if isinstance(e, list) and len(e) == EMBEDDING_DIM:
            out.append(rec)
            continue
        rid = str(rec.get("id"))
        vec = cache.get(rid)
        if vec is None:
            out.append(rec)
            continue
        g = dict(rec)
        g["prompt_embedding"] = np.asarray(vec, dtype=np.float32).tolist()
        out.append(g)
    return out


def load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def illicit_of(rec: Any) -> Optional[float]:
    if not isinstance(rec, dict):
        return None
    mr = rec.get("moderation_result") or {}
    oa = mr.get("openai") if isinstance(mr, dict) else None
    scores = (oa or {}).get("scores") if isinstance(oa, dict) else {}
    for v in (
        scores.get("illicit") if isinstance(scores, dict) else None,
        rec.get("illicit"),
        rec.get("fitness"),
        rec.get("score"),
    ):
        if v is None:
            continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if x == x:
            return x
    return None


def load_genomes(
    run_dir: Path,
    clustering: str,
    *,
    attach_embeddings: bool = True,
) -> List[Dict[str, Any]]:
    """Load unique genomes for a run.

    IncDBSCAN uses ``temp.json``; LF and Global EA (``none``) use
    ``elites.json`` + ``archive.json``. When ``attach_embeddings`` is true,
    missing ``prompt_embedding`` fields are filled from ``prompt_embeddings.npz``.
    """
    recs: List[Dict[str, Any]] = []
    if clustering == "dbscan":
        names = ("temp.json",)
    else:
        # leader_follower and none (Global EA)
        names = ("elites.json", "archive.json")
    seen = set()
    for name in names:
        data = load_json(run_dir / name)
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("id")
            if rid in seen:
                continue
            seen.add(rid)
            recs.append(rec)
    recs.sort(key=lambda r: (int(r["id"]) if str(r.get("id", "")).isdigit() else 10**9, str(r.get("id"))))
    if attach_embeddings:
        recs = attach_cached_embeddings(recs, run_dir)
    return recs


def topk_anytime(genomes: Sequence[Dict[str, Any]], k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of the ``k`` highest illicit scores among the first b unique genomes."""
    if k < 1:
        raise ValueError("k must be >= 1")
    xs: List[int] = []
    ys: List[float] = []
    best: List[float] = []
    n = 0
    for rec in genomes:
        s = illicit_of(rec)
        if s is None:
            continue
        n += 1
        best.append(s)
        best.sort(reverse=True)
        if len(best) > k:
            best = best[:k]
        if n >= k:
            xs.append(n)
            ys.append(float(np.mean(best)))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def top10_anytime(genomes: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Mean of the ten highest illicit scores among the first b unique genomes."""
    return topk_anytime(genomes, k=10)


def best_anytime(genomes: Sequence[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
    """Running maximum illicit among the first b unique genomes."""
    xs: List[int] = []
    ys: List[float] = []
    best = float("-inf")
    n = 0
    for rec in genomes:
        s = illicit_of(rec)
        if s is None:
            continue
        n += 1
        if s > best:
            best = s
        xs.append(n)
        ys.append(best)
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def common_silhouette(genomes: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Cosine silhouette on MiniLM embeddings of assigned species members.

    Noise / unassigned (species_id 0) and clusters of size 1 are excluded.
    Undefined when fewer than two remaining clusters.
    """
    embs: List[List[float]] = []
    labels: List[int] = []
    for rec in genomes:
        sid = rec.get("species_id")
        if sid in NOISE_SIDS:
            continue
        try:
            sid_i = int(sid)
        except (TypeError, ValueError):
            continue
        e = rec.get("prompt_embedding")
        if not isinstance(e, list) or len(e) < 8:
            continue
        embs.append(e)
        labels.append(sid_i)
    if len(embs) < 4:
        return None
    y = np.asarray(labels)
    keep = np.zeros(len(y), dtype=bool)
    for lab in np.unique(y):
        m = y == lab
        if int(m.sum()) >= 2:
            keep |= m
    if keep.sum() < 4 or len(np.unique(y[keep])) < 2:
        return None
    try:
        from sklearn.metrics import silhouette_score
    except Exception:
        return None
    x = np.asarray(embs, dtype=float)[keep]
    score = float(silhouette_score(x, y[keep], metric="cosine"))
    return score if score == score else None


def assigned_fraction(genomes: Sequence[Dict[str, Any]]) -> Optional[float]:
    if not genomes:
        return None
    n_ass = sum(1 for r in genomes if r.get("species_id") not in NOISE_SIDS)
    return n_ass / len(genomes)


def moderated_genomes(genomes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unique genomes that carry a moderation result (true evaluation budget)."""
    out: List[Dict[str, Any]] = []
    for rec in genomes:
        if illicit_of(rec) is None and not rec.get("moderation_result"):
            continue
        out.append(rec)
    return out


def n_evaluated(genomes: Sequence[Dict[str, Any]]) -> int:
    """Count of unique moderated genomes (honest unique-evaluation budget)."""
    return len(moderated_genomes(genomes))


def llm_calls_from_genomes(genomes: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    """Recount LLM / evaluation calls from unique genome records.

    The EvolutionTracker cumulative counters double-count IncDBSCAN runs
    (generation-0 logs ~150 evaluations for 50 seed genomes). Prefer these
    per-genome recounts for reporting and tests.
    """
    OPERATORS_USING_LLM = {
        "LLMParaphrasingOperator",
        "LLMSynonymReplacement",
        "LLMBackTranslation_DE",
        "LLMBackTranslation_FR",
        "LLMBackTranslation_HI",
        "LLMBackTranslation_JA",
        "LLMBackTranslation_ZH",
        "NegationOperator",
        "TypographicalErrorsOperator",
        "ConceptAdditionOperator",
        "SemanticFusionCrossover",
    }
    n_resp = 0
    n_var = 0
    n_eval = 0
    for rec in moderated_genomes(genomes):
        if rec.get("response_duration") is not None or rec.get("generated_output"):
            n_resp += 1
        ci = rec.get("creation_info") or {}
        op = ci.get("operator") or rec.get("operator")
        if op and op in OPERATORS_USING_LLM:
            n_var += 1
        if illicit_of(rec) is not None or rec.get("moderation_result"):
            n_eval += 1
    return {
        "n_evaluated": n_eval,
        "llm_calls_response": n_resp,
        "llm_calls_variant": n_var,
        "llm_calls_total": n_resp + n_var,
    }


def topk_mean_truncated(
    genomes: Sequence[Dict[str, Any]],
    k: int = 10,
    budget: int = 2000,
) -> Optional[float]:
    """Mean of the ``k`` highest scores among the first ``budget`` moderated genomes."""
    scored = moderated_genomes(genomes)[:budget]
    vals = [s for s in (illicit_of(r) for r in scored) if s is not None]
    if len(vals) < k:
        return None
    vals.sort(reverse=True)
    return float(np.mean(vals[:k]))
