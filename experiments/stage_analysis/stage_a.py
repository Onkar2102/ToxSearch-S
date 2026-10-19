"""Stage A: offline intrinsic agreement, probes, and threshold calibration on P0.

Input ``P0`` is a table with one row per held-out prompt:
    prompt, response, emb (list[float]), scores_google (8 floats), scores_openai (13 floats)
in the axis order of ``utils.evaluator_profiles``. Building it requires one RG call
per prompt and one call to each live oracle (see ``build_p0``).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .metrics import d_e_matrix

W_G, W_P = 0.7, 0.3


def _score_vector(scores: Dict[str, float], order: Sequence[str]) -> List[float]:
    missing = [k for k in order if k not in scores or scores[k] is None]
    if missing:
        raise ValueError(f"Missing phenotype scores: {missing}")
    return [float(scores[k]) for k in order]


def build_p0(
    prompts: Sequence[str],
    out_path: str,
    *,
    rg_config: str = "config/RGConfig.yaml",
    seed: int = 42,
    embedding_model: str = "all-MiniLM-L6-v2",
    openai_model: str = "omni-moderation-latest",
) -> pd.DataFrame:
    """Sample one response per prompt, score with both live oracles, embed, write ``out_path``.

    Output columns: ``prompt``, ``response``, ``emb``, ``scores_google``, ``scores_openai``.
    Prefer ``.parquet`` or ``.pkl``; ``.json``/``.jsonl`` also supported (lists for vectors).
    """
    from pathlib import Path

    from gne.evaluator import HybridModerationEvaluator, evaluate_single_genome
    from gne.response_generator import ResponseGenerator
    from speciation.embeddings import get_embedding_model
    from utils.evaluator_profiles import GOOGLE_PHENOTYPE_SCORE_ORDER, OPENAI_PHENOTYPE_SCORE_ORDER

    prompts = [str(p).strip() for p in prompts if str(p).strip()]
    if not prompts:
        raise ValueError("build_p0: empty prompt list")

    rg = ResponseGenerator(config_path=rg_config, seed=seed)
    ev_g = HybridModerationEvaluator(evaluator="google")
    ev_o = HybridModerationEvaluator(evaluator="openai", openai_model=openai_model)
    emb_model = get_embedding_model(model_name=embedding_model)

    rows: List[Dict] = []
    for i, prompt in enumerate(prompts):
        response, _ = rg.generate_response(prompt)
        if not (response or "").strip():
            raise RuntimeError(f"build_p0: empty response for prompt index {i}")

        g_g = {"id": f"p0_{i}", "generated_output": response}
        g_o = {"id": f"p0_{i}", "generated_output": response}
        evaluate_single_genome(ev_g, g_g, moderation_methods=["google"])
        evaluate_single_genome(ev_o, g_o, moderation_methods=["openai"])
        if g_g.get("status") != "complete" or g_o.get("status") != "complete":
            raise RuntimeError(
                f"build_p0: moderation failed for prompt index {i}: "
                f"google={g_g.get('error')}, openai={g_o.get('error')}"
            )

        sg = (g_g.get("moderation_result") or {}).get("google") or {}
        so = (g_o.get("moderation_result") or {}).get("openai") or {}
        scores_g = sg.get("scores") if isinstance(sg, dict) else None
        scores_o = so.get("scores") if isinstance(so, dict) else None
        if not isinstance(scores_g, dict) or not isinstance(scores_o, dict):
            raise RuntimeError(f"build_p0: missing score dicts for prompt index {i}")

        rows.append({
            "prompt": prompt,
            "response": response,
            "emb": emb_model.encode(prompt).tolist(),
            "scores_google": _score_vector(scores_g, GOOGLE_PHENOTYPE_SCORE_ORDER),
            "scores_openai": _score_vector(scores_o, OPENAI_PHENOTYPE_SCORE_ORDER),
        })

    df = pd.DataFrame(rows)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    elif path.suffix in (".pkl", ".pickle"):
        df.to_pickle(path)
    elif path.suffix == ".jsonl":
        df.to_json(path, orient="records", lines=True)
    else:
        df.to_json(path, orient="records", indent=2)
    return df


def load_p0(path: str, *, require_openai: bool = True) -> pd.DataFrame:
    """Load a P0 table written by ``build_p0`` or the held-out batch builder.

    Set ``require_openai=False`` for google-only RQ1 batches that reuse run artifacts.
    """
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".pkl", ".pickle"):
        df = pd.read_pickle(p)
    elif p.suffix == ".jsonl":
        df = pd.read_json(p, lines=True)
    else:
        df = pd.read_json(p)
    required = {"prompt", "response", "emb", "scores_google"}
    if require_openai:
        required.add("scores_openai")
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"P0 table missing columns: {sorted(missing)}")
    return df


def load_prompts_csv(path: str, column: str = "questions") -> List[str]:
    """Load prompts from a seed-style CSV (``questions`` column by default)."""
    df = pd.read_csv(path)
    if column not in df.columns:
        raise ValueError(f"CSV {path} has no column {column!r}; columns={list(df.columns)}")
    return [str(x).strip() for x in df[column].tolist() if str(x).strip()]


def d_p_matrix(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float64)
    sq = np.sum(x * x, axis=1, keepdims=True)
    d = np.sqrt(np.maximum(sq + sq.T - 2 * x @ x.T, 0.0)) / np.sqrt(x.shape[1])
    np.fill_diagonal(d, 0.0)
    return d


def distance_matrices(
    p0: pd.DataFrame,
    nli: bool = True,
    *,
    evaluators: Sequence[str] = ("google", "openai"),
) -> Dict[str, np.ndarray]:
    emb = np.vstack(p0["emb"].map(np.asarray).to_list())
    mats = {"d_E": d_e_matrix(emb)}
    for ev in evaluators:
        col = f"scores_{ev}"
        if col not in p0.columns:
            raise ValueError(f"distance_matrices: missing column {col}")
        dp = d_p_matrix(np.vstack(p0[col].map(np.asarray).to_list()))
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

    if "scores_openai" not in p0.columns:
        raise ValueError("oracle_agreement requires scores_openai")
    g = np.vstack(p0["scores_google"].map(np.asarray).to_list())[:, GOOGLE_PHENOTYPE_SCORE_ORDER.index("toxicity")]
    o = np.vstack(p0["scores_openai"].map(np.asarray).to_list())[:, OPENAI_PHENOTYPE_SCORE_ORDER.index("violence")]
    return {"spearman_toxicity_vs_violence": float(spearmanr(g, o).correlation),
            "spearman_toxicity_vs_max_openai": float(spearmanr(
                g, np.vstack(p0["scores_openai"].map(np.asarray).to_list()).max(axis=1)).correlation)}


def prompt_resample_cis(
    mats: Dict[str, np.ndarray],
    *,
    pairs: Sequence[Tuple[str, str]] = (
        ("d_E", "d_N"),
        ("d_E", "d_EP[google]"),
        ("d_N", "d_EP[google]"),
    ),
    knn_k: int = 10,
    n_boot: int = 200,
    seed: int = 0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Prompt-level bootstrap CIs for Spearman and mean k-NN Jaccard.

    Resamples matrix *rows* (prompts), not raw pairs as i.i.d. observations.
    Only pairs whose keys exist in ``mats`` are reported.
    """
    rng = np.random.default_rng(seed)
    n = next(iter(mats.values())).shape[0]
    out: Dict[str, Dict[str, float]] = {}
    usable = [(a, b) for a, b in pairs if a in mats and b in mats]
    if not usable or n < 3:
        return out

    for a, b in usable:
        spearmans, jaccards = [], []
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            ma = mats[a][np.ix_(idx, idx)]
            mb = mats[b][np.ix_(idx, idx)]
            ua, ub = _upper(ma), _upper(mb)
            if ua.size < 2 or np.all(ua == ua[0]) or np.all(ub == ub[0]):
                continue
            spearmans.append(float(spearmanr(ua, ub).correlation))
            jaccards.append(knn_jaccard(ma, mb, knn_k))
        for name, vals in (("spearman", spearmans), ("knn_jaccard", jaccards)):
            arr = np.asarray([v for v in vals if v == v], dtype=float)
            key = f"{a}__{b}__{name}"
            if arr.size == 0:
                out[key] = {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "n_boot": 0}
            else:
                out[key] = {
                    "mean": float(arr.mean()),
                    "ci_lo": float(np.quantile(arr, alpha / 2)),
                    "ci_hi": float(np.quantile(arr, 1 - alpha / 2)),
                    "n_boot": int(arr.size),
                }
    return out
