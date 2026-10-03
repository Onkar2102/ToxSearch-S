r"""NLI dissimilarity \(d_N\) (Option B: symmetrized entailment).

\[
d_N(u,v) = 1 - \\frac{p_{\\mathrm{ent}}(u,v) + p_{\\mathrm{ent}}(v,u)}{2}
\\quad(u \\neq v),\\qquad d_N(u,u) = 0.
\]

Entailment probabilities come from a pluggable scorer (default: HuggingFace
MNLI pipeline when available). Unit tests can inject ``set_entailment_scorer``.
"""

from __future__ import annotations

import hashlib
import threading
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from utils import get_custom_logging

get_logger, _, _, _ = get_custom_logging()

# (text_a, text_b) -> P(entailment) in [0, 1]
EntailmentScorer = Callable[[str, str], float]

_scorer_lock = threading.Lock()
_entailment_scorer: Optional[EntailmentScorer] = None
_default_model_name = "facebook/bart-large-mnli"
_pair_cache: Dict[Tuple[str, str], float] = {}


def set_entailment_scorer(scorer: Optional[EntailmentScorer]) -> None:
    """Inject or clear the directional entailment probability function."""
    global _entailment_scorer, _pair_cache
    with _scorer_lock:
        _entailment_scorer = scorer
        _pair_cache = {}


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


def _load_default_scorer(logger=None) -> EntailmentScorer:
    log = logger or get_logger("NLIDistance")
    try:
        from transformers import pipeline  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "distance_method=nli requires the 'transformers' package "
            "(and a torch backend). Install transformers or pass set_entailment_scorer()."
        ) from e

    log.info("Loading NLI model %s (zero-shot / MNLI entailment)", _default_model_name)
    clf = pipeline("text-classification", model=_default_model_name, truncation=True)

    def _score(premise: str, hypothesis: str) -> float:
        # MNLI-style: premise=text_a, hypothesis=text_b
        out = clf({"text": premise, "text_pair": hypothesis}, top_k=None)
        if isinstance(out, dict):
            out = [out]
        # Normalize labels
        scores = {str(item["label"]).lower(): float(item["score"]) for item in out}
        for key, val in scores.items():
            if "entail" in key:
                return float(np.clip(val, 0.0, 1.0))
        # Fallback: if labels are LABEL_0/1/2, bart-large-mnli uses
        # contradiction / neutral / entailment — already handled above.
        # Last resort: max score among non-contradiction labels.
        non_contra = [v for k, v in scores.items() if "contradict" not in k]
        if non_contra:
            return float(np.clip(max(non_contra), 0.0, 1.0))
        return 0.0

    return _score


def get_entailment_scorer(logger=None) -> EntailmentScorer:
    global _entailment_scorer
    with _scorer_lock:
        if _entailment_scorer is not None:
            return _entailment_scorer
        _entailment_scorer = _load_default_scorer(logger=logger)
        return _entailment_scorer


def entailment_probability(text_a: str, text_b: str, *, logger=None) -> float:
    """Directional P(entailment | premise=a, hypothesis=b) with caching."""
    if text_a is None or text_b is None:
        return 0.0
    a = str(text_a)
    b = str(text_b)
    key = _cache_key(a, b)
    with _scorer_lock:
        if key in _pair_cache:
            return _pair_cache[key]
    scorer = get_entailment_scorer(logger=logger)
    val = float(np.clip(scorer(a, b), 0.0, 1.0))
    with _scorer_lock:
        _pair_cache[key] = val
    return val


def nli_distance(text_a: str, text_b: str, *, logger=None) -> float:
    """Symmetrized entailment dissimilarity in ``[0, 1]``; ``d(u,u)=0``."""
    if text_a is None or text_b is None:
        return 1.0
    a = str(text_a)
    b = str(text_b)
    if a == b:
        return 0.0
    p_ab = entailment_probability(a, b, logger=logger)
    p_ba = entailment_probability(b, a, logger=logger)
    return float(1.0 - 0.5 * (p_ab + p_ba))


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
    "EntailmentScorer",
    "set_entailment_scorer",
    "clear_nli_cache",
    "entailment_probability",
    "nli_distance",
    "nli_distances_batch",
    "pairwise_nli_matrix",
]
