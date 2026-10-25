r"""NLI dissimilarity \(d_N\) (expected squared semantic distance).

Qiu & Miikkulainen, Semantic Density (NeurIPS 2024, Eq. 9), symmetrized:

\[
d_N(u,v) = \bar{p}_C + \tfrac12\bar{p}_N
= 1 - \bar{p}_E - \tfrac12\bar{p}_N
\quad(u \neq v),\qquad d_N(u,u) = 0,
\]

where \(\bar{p}_r = \tfrac12\big(p_r(u\!\to\!v)+p_r(v\!\to\!u)\big)\) for
\(r\in\{E,N,C\}\).

Class probabilities come from a pluggable scorer (default: HuggingFace MNLI
pipeline). Unit tests can inject ``set_nli_scorer`` or the legacy
``set_entailment_scorer`` (remainder mass treated as contradiction).
"""

from __future__ import annotations

import atexit
import hashlib
import sqlite3
import threading
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()

# (text_a, text_b) -> (P_E, P_N, P_C) on the probability simplex
NLIProbs = Tuple[float, float, float]
NLIScorer = Callable[[str, str], NLIProbs]
# Legacy: (text_a, text_b) -> P(entailment) in [0, 1]
EntailmentScorer = Callable[[str, str], float]

_scorer_lock = threading.RLock()
_nli_scorer: Optional[NLIScorer] = None
# Cross-encoder DeBERTa-v3-small (MNLI+SNLI): ~142M params, local-friendly, already
# commonly cached. bart-large-mnli (~407M) remains usable via --nli-model.
_default_model_name = "cross-encoder/nli-deberta-v3-small"
_model_name: str = _default_model_name
_pair_cache: Dict[Tuple[str, str], NLIProbs] = {}
# True when set_nli_scorer / set_entailment_scorer injected a test/custom scorer.
_nli_scorer_injected = False
# Cumulative per process: directional model passes, cache hits, symmetric d_N evaluations.
_stats: Dict[str, int] = {"model_passes": 0, "cache_hits": 0, "distance_calls": 0, "disk_hits": 0}
_cache_db_path: Optional[Path] = None
_cache_db: Optional[sqlite3.Connection] = None
_cache_db_pending = 0
_logged_torch_batch = False


def get_nli_stats() -> Dict[str, int]:
    with _scorer_lock:
        return dict(_stats)


def set_nli_model_name(model_name: Optional[str]) -> None:
    """Set HuggingFace MNLI model id used by the default scorer (clears cache/scorer)."""
    global _model_name, _nli_scorer, _pair_cache
    name = (model_name or _default_model_name).strip()
    with _scorer_lock:
        if name != _model_name:
            _model_name = name
            _nli_scorer = None
            _pair_cache = {}
            _reset_torch_nli_locked()


def get_nli_model_name() -> str:
    with _scorer_lock:
        return _model_name


def set_nli_scorer(scorer: Optional[NLIScorer]) -> None:
    """Inject or clear the directional (P_E, P_N, P_C) scorer."""
    global _nli_scorer, _pair_cache, _nli_scorer_injected
    with _scorer_lock:
        _nli_scorer = scorer
        _nli_scorer_injected = scorer is not None
        _pair_cache = {}


def set_entailment_scorer(scorer: Optional[EntailmentScorer]) -> None:
    """Legacy inject: map P_E to (P_E, 0, 1-P_E) so remainder is contradiction.

    Prefer ``set_nli_scorer`` when all three class probabilities are available.
    """
    if scorer is None:
        set_nli_scorer(None)
        return

    def _wrapped(premise: str, hypothesis: str) -> NLIProbs:
        pe = float(np.clip(scorer(premise, hypothesis), 0.0, 1.0))
        return (pe, 0.0, float(1.0 - pe))

    set_nli_scorer(_wrapped)


def clear_nli_cache() -> None:
    """Clear the in-memory pair cache. The on-disk SQLite file is kept."""
    with _scorer_lock:
        _pair_cache.clear()


def get_nli_pair_cache_path() -> Optional[str]:
    with _scorer_lock:
        return str(_cache_db_path) if _cache_db_path is not None else None


def _flush_cache_db_locked() -> None:
    global _cache_db_pending
    if _cache_db is None:
        return
    _cache_db.commit()
    _cache_db_pending = 0


def close_nli_pair_cache() -> None:
    global _cache_db, _cache_db_path, _cache_db_pending
    with _scorer_lock:
        if _cache_db is not None:
            try:
                _cache_db.commit()
            except sqlite3.Error:
                pass
            try:
                _cache_db.close()
            except sqlite3.Error:
                pass
        _cache_db = None
        _cache_db_path = None
        _cache_db_pending = 0


def set_nli_pair_cache_path(path: Optional[str]) -> None:
    """Persist directional MNLI triples so LF and IncDBSCAN reuse the same pairs.

    ``path`` is a SQLite file. Rows are keyed by ``(model, cache_key_a, cache_key_b)``.
    Pass ``None`` or an empty string to disable disk persistence. Both clusterers share this
    file across processes (WAL). In-memory hits still win first.
    """
    global _cache_db, _cache_db_path, _cache_db_pending
    raw = (path or "").strip()
    with _scorer_lock:
        if _cache_db is not None:
            try:
                _cache_db.commit()
                _cache_db.close()
            except sqlite3.Error:
                pass
            _cache_db = None
            _cache_db_path = None
            _cache_db_pending = 0
        if not raw:
            return
        p = Path(raw).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(p), timeout=30.0, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS nli_pairs (
                model TEXT NOT NULL,
                ka TEXT NOT NULL,
                kb TEXT NOT NULL,
                pe REAL NOT NULL,
                pn REAL NOT NULL,
                pc REAL NOT NULL,
                PRIMARY KEY (model, ka, kb)
            )
            """
        )
        conn.commit()
        _cache_db = conn
        _cache_db_path = p
        _cache_db_pending = 0


def _disk_get_locked(ka: str, kb: str) -> Optional[NLIProbs]:
    if _cache_db is None:
        return None
    cur = _cache_db.execute(
        "SELECT pe, pn, pc FROM nli_pairs WHERE model=? AND ka=? AND kb=?",
        (_model_name, ka, kb),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return (float(row[0]), float(row[1]), float(row[2]))


def _disk_put_locked(ka: str, kb: str, probs: NLIProbs) -> None:
    global _cache_db_pending
    if _cache_db is None:
        return
    _cache_db.execute(
        "INSERT OR REPLACE INTO nli_pairs (model, ka, kb, pe, pn, pc) VALUES (?,?,?,?,?,?)",
        (_model_name, ka, kb, float(probs[0]), float(probs[1]), float(probs[2])),
    )
    _cache_db.commit()
    _cache_db_pending = 0


atexit.register(close_nli_pair_cache)


def _cache_key(a: str, b: str) -> Tuple[str, str]:
    # Hash long prompts to keep the cache dict small
    def _h(t: str) -> str:
        if len(t) <= 128:
            return t
        return hashlib.sha1(t.encode("utf-8")).hexdigest()

    return (_h(a), _h(b))


def _normalize_probs(pe: float, pn: float, pc: float) -> NLIProbs:
    pe = float(max(pe, 0.0))
    pn = float(max(pn, 0.0))
    pc = float(max(pc, 0.0))
    s = pe + pn + pc
    if s <= 0.0:
        raise ValueError("NLI probabilities sum to zero")
    return (pe / s, pn / s, pc / s)


def _parse_nli_scores(scores: Dict[str, float]) -> NLIProbs:
    """Map classifier label dict to (P_E, P_N, P_C)."""
    pe = pn = pc = None
    for key, val in scores.items():
        k = str(key).lower()
        if "entail" in k:
            pe = float(val)
        elif "neutral" in k:
            pn = float(val)
        elif "contradict" in k:
            pc = float(val)

    if pe is not None and pn is not None and pc is not None:
        return _normalize_probs(pe, pn, pc)

    # LABEL_i schemes: DeBERTa MNLI / cross-encoder convention is
    # 0=contradiction, 1=entailment, 2=neutral.
    if all(str(k).lower().startswith("label_") for k in scores):
        by_idx = {int(str(k).lower().split("_", 1)[1]): float(v) for k, v in scores.items()}
        if {0, 1, 2}.issubset(by_idx):
            return _normalize_probs(by_idx[1], by_idx[2], by_idx[0])

    raise RuntimeError(
        f"Unrecognized NLI label scheme (need entailment/neutral/contradiction): "
        f"{sorted(scores)}"
    )


def _load_default_scorer(logger=None) -> NLIScorer:
    log = logger or get_logger("NLIDistance")
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "distance_method=nli requires the 'transformers' package "
            "(and a torch backend). Install transformers or pass set_nli_scorer()."
        ) from e

    model_name = get_nli_model_name()
    log.info("Loading NLI model %s (MNLI 3-class for d_N = P_C + 0.5 P_N)", model_name)
    clf = pipeline("text-classification", model=model_name, truncation=True)

    def _score(premise: str, hypothesis: str) -> NLIProbs:
        out = clf({"text": premise, "text_pair": hypothesis}, top_k=None)
        if isinstance(out, dict):
            out = [out]
        scores = {str(item["label"]).lower(): float(item["score"]) for item in out}
        return _parse_nli_scores(scores)

    return _score


def get_nli_scorer(logger=None) -> NLIScorer:
    """Return the 3-class NLI scorer, loading the HF model on first use.

    Model load runs *outside* the lock so a slow ``pipeline()`` / download
    cannot deadlock the main thread.
    """
    global _nli_scorer
    with _scorer_lock:
        if _nli_scorer is not None:
            return _nli_scorer
    scorer = _load_default_scorer(logger=logger)
    with _scorer_lock:
        if _nli_scorer is None:
            _nli_scorer = scorer
        return _nli_scorer


# Backward-compatible alias used by older call sites / docs.
def get_entailment_scorer(logger=None) -> EntailmentScorer:
    """Return a directional entailment-only view of the 3-class scorer."""
    scorer = get_nli_scorer(logger=logger)

    def _entail_only(premise: str, hypothesis: str) -> float:
        pe, _, _ = scorer(premise, hypothesis)
        return float(pe)

    return _entail_only


def nli_class_probabilities(text_a: str, text_b: str, *, logger=None) -> NLIProbs:
    """Directional (P_E, P_N, P_C) with caching; premise=a, hypothesis=b."""
    if text_a is None or text_b is None:
        raise ValueError("nli_class_probabilities requires non-None texts on both sides")
    a = str(text_a)
    b = str(text_b)
    key = _cache_key(a, b)
    with _scorer_lock:
        if key in _pair_cache:
            _stats["cache_hits"] += 1
            return _pair_cache[key]
        disk = _disk_get_locked(key[0], key[1])
        if disk is not None:
            _pair_cache[key] = disk
            _stats["cache_hits"] += 1
            _stats["disk_hits"] += 1
            return disk
    scorer = get_nli_scorer(logger=logger)
    pe, pn, pc = scorer(a, b)
    probs = _normalize_probs(pe, pn, pc)
    with _scorer_lock:
        _pair_cache[key] = probs
        _stats["model_passes"] += 1
        _disk_put_locked(key[0], key[1], probs)
    return probs


def entailment_probability(text_a: str, text_b: str, *, logger=None) -> float:
    """Directional P(entailment | premise=a, hypothesis=b) with caching."""
    pe, _, _ = nli_class_probabilities(text_a, text_b, logger=logger)
    return float(pe)


def _directional_expected_sq(pe: float, pn: float, pc: float) -> float:
    """Qiu Eq. 9: E[||v-v'||^2] = P_C + 0.5 P_N (= 1 - P_E - 0.5 P_N)."""
    return float(pc + 0.5 * pn)


def nli_distance(text_a: str, text_b: str, *, logger=None) -> float:
    """Symmetrized expected-squared NLI dissimilarity in ``[0, 1]``; ``d(u,u)=0``."""
    if text_a is None or text_b is None:
        raise ValueError("nli_distance requires non-None texts on both sides")
    a = str(text_a)
    b = str(text_b)
    with _scorer_lock:
        _stats["distance_calls"] += 1
    if a == b:
        return 0.0
    pe_ab, pn_ab, pc_ab = nli_class_probabilities(a, b, logger=logger)
    pe_ba, pn_ba, pc_ba = nli_class_probabilities(b, a, logger=logger)
    d_ab = _directional_expected_sq(pe_ab, pn_ab, pc_ab)
    d_ba = _directional_expected_sq(pe_ba, pn_ba, pc_ba)
    return float(0.5 * (d_ab + d_ba))


def nli_entail_distance(text_a: str, text_b: str, *, logger=None) -> float:
    r"""Symmetrized non-entailment distance \(d_{\neg E}=1-\bar{P}_E\) in ``[0, 1]``.

    ``d(u,u)=0``. Shares the same MNLI scorer / cache as ``nli_distance``.
    """
    if text_a is None or text_b is None:
        raise ValueError("nli_entail_distance requires non-None texts on both sides")
    a = str(text_a)
    b = str(text_b)
    with _scorer_lock:
        _stats["distance_calls"] += 1
    if a == b:
        return 0.0
    pe_ab, _, _ = nli_class_probabilities(a, b, logger=logger)
    pe_ba, _, _ = nli_class_probabilities(b, a, logger=logger)
    return float(1.0 - 0.5 * (pe_ab + pe_ba))


def _nli_distances_batch_sequential(
    query_text: str,
    texts: Sequence[str],
    *,
    entail: bool,
    logger=None,
) -> np.ndarray:
    fn = nli_entail_distance if entail else nli_distance
    return np.asarray([fn(query_text, t, logger=logger) for t in texts], dtype=np.float64)


def nli_distances_batch(
    query_text: str,
    texts: Sequence[str],
    *,
    logger=None,
) -> np.ndarray:
    return _nli_query_distances(query_text, texts, entail=False, logger=logger)


def nli_entail_distances_batch(
    query_text: str,
    texts: Sequence[str],
    *,
    logger=None,
) -> np.ndarray:
    r"""Query-to-many \(d_{\neg E}\). Uses batched torch (RQ1 path) unless a test scorer is injected."""
    return _nli_query_distances(query_text, texts, entail=True, logger=logger)


def _nli_query_distances(
    query_text: str,
    texts: Sequence[str],
    *,
    entail: bool,
    logger=None,
) -> np.ndarray:
    if not texts:
        return np.zeros(0, dtype=np.float64)
    with _scorer_lock:
        injected = _nli_scorer_injected
    if injected:
        return _nli_distances_batch_sequential(query_text, texts, entail=entail, logger=logger)
    global _logged_torch_batch
    torch_fn = nli_entail_distances_from_query_torch if entail else nli_distances_from_query_torch
    log = logger or get_logger("NLIDistance")
    try:
        out = torch_fn(query_text, texts, batch_size=256)
    except (ImportError, RuntimeError, OSError) as exc:
        log.warning("Batched torch NLI failed (%s); falling back to sequential pairs", exc)
        return _nli_distances_batch_sequential(query_text, texts, entail=entail, logger=logger)
    if not _logged_torch_batch:
        log.info("NLI query distances: batched torch (batch_size=256, entail=%s)", entail)
        _logged_torch_batch = True
    return out


def _pipeline_from_default_scorer(scorer: NLIScorer):
    """Return the HF TextClassificationPipeline closed over by the default scorer, if any."""
    closure = getattr(scorer, "__closure__", None)
    if not closure:
        return None
    for cell in closure:
        obj = cell.cell_contents
        if type(obj).__name__ == "TextClassificationPipeline":
            return obj
        if callable(obj) and hasattr(obj, "model") and hasattr(obj, "tokenizer"):
            return obj
    return None


def _score_pairs_batched(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    batch_size: int = 64,
    logger=None,
) -> list:
    """Directional (P_E, P_N, P_C) for aligned premise/hypothesis lists."""
    if len(premises) != len(hypotheses):
        raise ValueError("premises and hypotheses must have equal length")
    if not premises:
        return []

    scorer = get_nli_scorer(logger=logger)
    clf = _pipeline_from_default_scorer(scorer)
    out = []
    if clf is None:
        for a, b in zip(premises, hypotheses):
            out.append(nli_class_probabilities(a, b, logger=logger))
        return out

    for start in range(0, len(premises), batch_size):
        chunk_p = premises[start : start + batch_size]
        chunk_h = hypotheses[start : start + batch_size]
        inputs = [{"text": str(a), "text_pair": str(b)} for a, b in zip(chunk_p, chunk_h)]
        raw = clf(inputs, top_k=None)
        # Single-item edge: pipeline may flatten to list[dict]
        if raw and isinstance(raw[0], dict) and "label" in raw[0]:
            raw = [raw]
        for item in raw:
            scores = {str(x["label"]).lower(): float(x["score"]) for x in item}
            pe, pn, pc = _parse_nli_scores(scores)
            probs = _normalize_probs(pe, pn, pc)
            out.append(probs)
            with _scorer_lock:
                _stats["model_passes"] += 1
    return out


def nli_distances_from_query(
    query_text: str,
    texts: Sequence[str],
    *,
    batch_size: int = 64,
    logger=None,
) -> np.ndarray:
    """Symmetrized d_N from one query to many texts (batched HF when available)."""
    texts = [str(t) for t in texts]
    q = str(query_text)
    n = len(texts)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    with _scorer_lock:
        _stats["distance_calls"] += n
    # Split identical vs distinct to keep d(u,u)=0
    idxs = [i for i, t in enumerate(texts) if t != q]
    if not idxs:
        return out
    others = [texts[i] for i in idxs]
    ab = _score_pairs_batched([q] * len(others), others, batch_size=batch_size, logger=logger)
    ba = _score_pairs_batched(others, [q] * len(others), batch_size=batch_size, logger=logger)
    for k, i in enumerate(idxs):
        pe_ab, pn_ab, pc_ab = ab[k]
        pe_ba, pn_ba, pc_ba = ba[k]
        out[i] = 0.5 * (
            _directional_expected_sq(pe_ab, pn_ab, pc_ab)
            + _directional_expected_sq(pe_ba, pn_ba, pc_ba)
        )
    return out


_torch_tok = None
_torch_model = None
_torch_device = None
_torch_label_idx = None  # (entail, neutral, contradict)


def _reset_torch_nli_locked() -> None:
    global _torch_tok, _torch_model, _torch_device, _torch_label_idx
    _torch_tok = None
    _torch_model = None
    _torch_device = None
    _torch_label_idx = None


def _ensure_torch_nli(logger=None):
    """Lazy-load tokenizer+model for high-throughput MPS/CUDA batching."""
    global _torch_tok, _torch_model, _torch_device, _torch_label_idx
    if _torch_model is not None:
        return
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    log = logger or get_logger("NLIDistance")
    name = get_nli_model_name()
    with _scorer_lock:
        pipe = _pipeline_from_default_scorer(_nli_scorer) if _nli_scorer is not None else None
    if pipe is not None and hasattr(pipe, "model") and hasattr(pipe, "tokenizer"):
        log.info("Reusing loaded HF pipeline weights for batched torch NLI (%s)", name)
        _torch_tok = pipe.tokenizer
        _torch_model = pipe.model
    else:
        log.info("Loading torch NLI model %s for batched pairwise d_N", name)
        _torch_tok = AutoTokenizer.from_pretrained(name)
        _torch_model = AutoModelForSequenceClassification.from_pretrained(name)
    if torch.backends.mps.is_available():
        _torch_device = torch.device("mps")
    elif torch.cuda.is_available():
        _torch_device = torch.device("cuda")
    else:
        _torch_device = torch.device("cpu")
    _torch_model.to(_torch_device).eval()
    id2label = {int(k): str(v).lower() for k, v in _torch_model.config.id2label.items()}
    # Prefer name matching; fall back to DeBERTa cross-encoder convention 0/1/2.
    ent = neu = con = None
    for i, lab in id2label.items():
        if "entail" in lab:
            ent = i
        elif "neutral" in lab:
            neu = i
        elif "contradict" in lab:
            con = i
    if ent is None or neu is None or con is None:
        ent, neu, con = 1, 2, 0
    _torch_label_idx = (ent, neu, con)


def _torch_score_pairs(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    *,
    batch_size: int = 256,
    max_length: int = 256,
) -> np.ndarray:
    """Return array (n, 3) of (P_E, P_N, P_C)."""
    import torch

    _ensure_torch_nli()
    assert _torch_tok is not None and _torch_model is not None and _torch_device is not None
    ent_i, neu_i, con_i = _torch_label_idx  # type: ignore[misc]
    rows = []
    bs = max(1, int(batch_size))
    with torch.no_grad():
        start = 0
        while start < len(premises):
            chunk_p = list(premises[start : start + bs])
            chunk_h = list(hypotheses[start : start + bs])
            try:
                enc = _torch_tok(
                    chunk_p,
                    chunk_h,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(_torch_device) for k, v in enc.items()}
                probs = torch.softmax(_torch_model(**enc).logits, dim=-1).detach().cpu().numpy()
                rows.append(np.stack([probs[:, ent_i], probs[:, neu_i], probs[:, con_i]], axis=1))
                with _scorer_lock:
                    _stats["model_passes"] += len(chunk_p)
                start += bs
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" in msg or "mps backend out of memory" in msg:
                    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                        torch.mps.empty_cache()
                    if bs <= 1:
                        raise
                    bs = max(1, bs // 2)
                    continue
                raise
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3), dtype=np.float64)


def nli_distances_from_query_torch(
    query_text: str,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    max_length: int = 256,
) -> np.ndarray:
    """Fast symmetrized d_N via direct torch batching (preferred for large matrices)."""
    texts = [str(t) for t in texts]
    q = str(query_text)
    n = len(texts)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    with _scorer_lock:
        _stats["distance_calls"] += n
    idxs = [i for i, t in enumerate(texts) if t != q]
    if not idxs:
        return out
    others = [texts[i] for i in idxs]
    ab = _torch_score_pairs([q] * len(others), others, batch_size=batch_size, max_length=max_length)
    ba = _torch_score_pairs(others, [q] * len(others), batch_size=batch_size, max_length=max_length)
    d_ab = ab[:, 2] + 0.5 * ab[:, 1]  # P_C + 0.5 P_N
    d_ba = ba[:, 2] + 0.5 * ba[:, 1]
    vals = 0.5 * (d_ab + d_ba)
    for k, i in enumerate(idxs):
        out[i] = float(vals[k])
    return out


def nli_entail_distances_from_query_torch(
    query_text: str,
    texts: Sequence[str],
    *,
    batch_size: int = 256,
    max_length: int = 256,
) -> np.ndarray:
    r"""Fast symmetrized \(d_{\neg E}=1-\bar{P}_E\) via torch batching."""
    texts = [str(t) for t in texts]
    q = str(query_text)
    n = len(texts)
    out = np.zeros(n, dtype=np.float64)
    if n == 0:
        return out
    with _scorer_lock:
        _stats["distance_calls"] += n
    idxs = [i for i, t in enumerate(texts) if t != q]
    if not idxs:
        return out
    others = [texts[i] for i in idxs]
    ab = _torch_score_pairs([q] * len(others), others, batch_size=batch_size, max_length=max_length)
    ba = _torch_score_pairs(others, [q] * len(others), batch_size=batch_size, max_length=max_length)
    vals = 1.0 - 0.5 * (ab[:, 0] + ba[:, 0])
    for k, i in enumerate(idxs):
        out[i] = float(vals[k])
    return out


def pairwise_nli_matrix(texts: Sequence[str], *, logger=None) -> np.ndarray:
    n = len(texts)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = nli_distance(texts[i], texts[j], logger=logger)
            dist[i, j] = d
            dist[j, i] = d
    return dist


def pairwise_nli_entail_matrix(texts: Sequence[str], *, logger=None) -> np.ndarray:
    n = len(texts)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = nli_entail_distance(texts[i], texts[j], logger=logger)
            dist[i, j] = d
            dist[j, i] = d
    return dist


__all__ = [
    "NLIProbs",
    "NLIScorer",
    "EntailmentScorer",
    "set_nli_scorer",
    "set_entailment_scorer",
    "set_nli_model_name",
    "get_nli_model_name",
    "get_nli_scorer",
    "get_entailment_scorer",
    "clear_nli_cache",
    "set_nli_pair_cache_path",
    "get_nli_pair_cache_path",
    "close_nli_pair_cache",
    "get_nli_stats",
    "nli_class_probabilities",
    "entailment_probability",
    "nli_distance",
    "nli_distances_batch",
    "nli_distances_from_query",
    "nli_distances_from_query_torch",
    "nli_entail_distance",
    "nli_entail_distances_batch",
    "nli_entail_distances_from_query_torch",
    "pairwise_nli_matrix",
    "pairwise_nli_entail_matrix",
]
