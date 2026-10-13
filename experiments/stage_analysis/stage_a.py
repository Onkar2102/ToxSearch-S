"""Stage A: offline intrinsic agreement, probes, and threshold calibration on P0.

Input ``P0`` is a table with one row per held-out prompt:
    prompt, response, emb (list[float]), scores_google (8 floats), scores_openai (13 floats)
in the axis order of ``utils.evaluator_profiles``. Building it requires one RG call
per prompt and one call to each live oracle (see ``build_p0``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import d_e_matrix

W_G, W_P = 0.7, 0.3


def build_p0(prompts: Sequence[str], out_path: str) -> None:
    """TODO: sample one response per prompt with the Stage C RG config, score with both oracles,
    embed with all-MiniLM-L6-v2, and write the P0 table. Use held-out prompts from the seed corpora,
    disjoint from data/prompt_100.csv."""
    raise NotImplementedError


def d_p_matrix(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)) / np.sqrt(x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def distance_matrices(p0: pd.DataFrame, nli: bool = True) -> Dict[str, np.ndarray]:
    emb = np.vstack(p0["emb"].map(np.asarray).to_list())
    mats = {"d_E": d_e_matrix(emb)}
    for ev in ("google", "openai"):
        dp = d_p_matrix(np.vstack(p0[f"scores_{ev}"].map(np.asarray).to_list()))
        mats[f"d_P[{ev}]"] = dp
        mats[f"d_EP[{ev}]"] = W_G * mats["d_E"] + W_P * dp
    if nli:
        from speciation.distance_metrics.nli import pairwise_nli_matrix
        mats["d_N"] = pairwise_nli_matrix(p0["prompt"].tolist())
    return mats


def _upper(m: np.ndarray) -> np.ndarray:
    return m[np.triu_indices(m.shape[0], 1)]


def spearman_table(mats: Dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(mats)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = spearmanr(_upper(mats[a]), _upper(mats[b])).correlation
    return out


def knn_jaccard(a: np.ndarray, b: np.ndarray, k: int = 10) -> float:
    """Mean Jaccard overlap of each point's k nearest neighbours under two dissimilarities."""
    n = a.shape[0]
    big = np.eye(n) * 1e9
    na = np.argsort(a + big, axis=1)[:, :k]
    nb = np.argsort(b + big, axis=1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / len(set(x) | set(y)) for x, y in zip(na, nb)]))


def knn_table(mats: Dict[str, np.ndarray], k: int = 10) -> pd.DataFrame:
    names = list(mats)
    out = pd.DataFrame(index=names, columns=names, dtype=float)
    for a in names:
        for b in names:
            out.loc[a, b] = knn_jaccard(mats[a], mats[b], k)
    return out


def output_term_share(mats: Dict[str, np.ndarray], evaluator: str) -> Dict[str, float]:
    """Share w_p*d_P / d_EP over pairs (how much the oracle term moves d_EP)."""
    dep = _upper(mats[f"d_EP[{evaluator}]"])
    dp = _upper(mats[f"d_P[{evaluator}]"])
    share = np.where(dep > 0, W_P * dp / np.where(dep > 0, dep, 1.0), 0.0)
    return {"mean": float(share.mean()), "median": float(np.median(share)), "p90": float(np.quantile(share, 0.9))}


def quantile_table(mats: Dict[str, np.ndarray], qs: Iterable[float] = (0.01, 0.05, 0.1, 0.25, 0.5)) -> pd.DataFrame:
    return pd.DataFrame({name: np.quantile(_upper(m), list(qs)) for name, m in mats.items()}, index=list(qs)).T


def neighbor_fraction(m: np.ndarray, radius: float, strict: bool) -> float:
    """Fraction of points with at least one other point within ``radius`` (the Stage B niche statistic)."""
    off = m + np.eye(m.shape[0]) * 1e9
    hit = (off < radius) if strict else (off <= radius)
    return float(hit.any(axis=1).mean())


def calibrate_radius(m: np.ndarray, target_fraction: float, strict: bool = True) -> float:
    """Smallest radius whose neighbour fraction reaches ``target_fraction``.

    ``strict=True`` matches LF (d < theta_sim); ``False`` matches IncDBSCAN (d <= eps).
    """
    off = m + np.eye(m.shape[0]) * 1e9
    nn = np.sort(off.min(axis=1))
    idx = int(np.ceil(target_fraction * len(nn))) - 1
    r = float(nn[max(0, min(idx, len(nn) - 1))])
    return float(np.nextafter(r, 1.0)) if strict else r


def calibration_table(mats: Dict[str, np.ndarray], reference: str = "d_E", reference_radius: float = 0.25) -> pd.DataFrame:
    """Match every dissimilarity to the neighbour fraction d_E has at its default radius."""
    target = neighbor_fraction(mats[reference], reference_radius, strict=True)
    rows = [{"distance": name, "target_fraction": target,
             "theta_sim": calibrate_radius(m, target, strict=True),
             "eps": calibrate_radius(m, target, strict=False)} for name, m in mats.items()]
    return pd.DataFrame(rows)


def probe_buckets(pairs: pd.DataFrame, p0_lookup: Dict[str, Dict], nli: bool = True) -> pd.DataFrame:
    """Per-bucket distance summaries (Table 'probes').

    ``pairs`` columns: bucket, prompt_a, prompt_b. ``p0_lookup`` maps prompt -> P0 row.
    TODO: author the labeled pairs (near-paraphrase, opposed intent, one-sided support,
    lexical-overlap trap, interrogative pairs, same topic / different harm) before Stage A.
    """
    from speciation.distance_metrics.nli import nli_distance

    recs: List[Dict] = []
    for r in pairs.itertuples(index=False):
        a, b = p0_lookup[r.prompt_a], p0_lookup[r.prompt_b]
        ea, eb = np.asarray(a["emb"]), np.asarray(b["emb"])
        de = float((1 - np.clip(ea @ eb / (np.linalg.norm(ea) * np.linalg.norm(eb)), -1, 1)) / 2)
        rec = {"bucket": r.bucket, "d_E": de}
        for ev in ("google", "openai"):
            sa, sb = np.asarray(a[f"scores_{ev}"]), np.asarray(b[f"scores_{ev}"])
            rec[f"d_EP[{ev}]"] = W_G * de + W_P * float(np.linalg.norm(sa - sb) / np.sqrt(len(sa)))
        if nli:
            rec["d_N"] = nli_distance(r.prompt_a, r.prompt_b)
        recs.append(rec)
    return pd.DataFrame(recs).groupby("bucket").agg(["mean", "std", "count"])


def oracle_agreement(p0: pd.DataFrame) -> Dict[str, float]:
    """Spearman between the two oracles' north stars on the shared P0 responses."""
    from utils.evaluator_profiles import GOOGLE_PHENOTYPE_SCORE_ORDER, OPENAI_PHENOTYPE_SCORE_ORDER

    g = np.vstack(p0["scores_google"].map(np.asarray).to_list())[:, GOOGLE_PHENOTYPE_SCORE_ORDER.index("toxicity")]
    o = np.vstack(p0["scores_openai"].map(np.asarray).to_list())[:, OPENAI_PHENOTYPE_SCORE_ORDER.index("violence")]
    return {"spearman_toxicity_vs_violence": float(spearmanr(g, o).correlation),
            "spearman_toxicity_vs_max_openai": float(spearmanr(
                g, np.vstack(p0["scores_openai"].map(np.asarray).to_list()).max(axis=1)).correlation)}
