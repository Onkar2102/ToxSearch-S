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

import hashlib
import threading
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
# Cumulative per process: directional model passes, cache hits, symmetric d_N evaluations.
_stats: Dict[str, int] = {"model_passes": 0, "cache_hits": 0, "distance_calls": 0}


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


def get_nli_model_name() -> str:
    with _scorer_lock:
        return _model_name


def set_nli_scorer(scorer: Optional[NLIScorer]) -> None:
    """Inject or clear the directional (P_E, P_N, P_C) scorer."""
    global _nli_scorer, _pair_cache
    with _scorer_lock:
        _nli_scorer = scorer
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
    with _scorer_lock:
        _pair_cache.clear()


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
    scorer = get_nli_scorer(logger=logger)
    pe, pn, pc = scorer(a, b)
    probs = _normalize_probs(pe, pn, pc)
    with _scorer_lock:
        _pair_cache[key] = probs
        _stats["model_passes"] += 1
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


def nli_distances_batch(
    query_text: str,
    texts: Sequence[str],
    *,
    logger=None,
) -> np.ndarray:
    return np.asarray(
        [nli_distance(query_text, t, logger=logger) for t in texts],
        dtype=np.float64,
    )


def pairwise_nli_matrix(texts: Sequence[str], *, logger=None) -> np.ndarray:
    n = len(texts)
    dist = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            d = nli_distance(texts[i], texts[j], logger=logger)
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
    "get_nli_stats",
    "nli_class_probabilities",
    "entailment_probability",
    "nli_distance",
    "nli_distances_batch",
    "pairwise_nli_matrix",
]
