#!/usr/bin/env python3
"""RQ3 species coherence under independent representations.

Recomputes within-species / between-centroid dispersion for each run under:
  1. all-mpnet-base-v2 (different transformer encoder from MiniLM)
  2. TF-IDF character n-gram cosine

A size-preserving label-permutation null (B=1000) yields a z-score so that
coherence is judged relative to chance under the same sizes — removing the
circularity of scoring MiniLM-formed niches with MiniLM distance.

Example::

    MPLBACKEND=Agg PYTHONPATH=src:. venv/bin/python \\
      experiments/evostar_2027/rq3/scripts/rq3_coherence_independent.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
CACHE = REPORT / "coherence_cache"
NOTES = ROOT / "notes" / "rough_draft"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}
DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}
NOISE = {None, 0, -1, "0", "-1"}

N_PERM = 500
PERM_SEED = 2027
MPNET = "sentence-transformers/all-mpnet-base-v2"


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text()).get("status")) == "complete"
    except Exception:
        return False


def sid_int(rec: Dict[str, Any]) -> Optional[int]:
    s = rec.get("species_id")
    if s in NOISE:
        return None
    try:
        i = int(s)
    except Exception:
        return None
    return None if i <= 0 else i


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def mean_pairwise_cos_diss(embs: np.ndarray, max_n: int = 256, rng: Optional[np.random.Generator] = None) -> float:
    """Mean pairwise cosine dissimilarity; subsample large clusters."""
    if embs.shape[0] < 2:
        return float("nan")
    e = l2(embs)
    if e.shape[0] > max_n:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(e.shape[0], size=max_n, replace=False)
        e = e[idx]
    cos = np.clip(e @ e.T, -1.0, 1.0)
    D = (1.0 - cos) / 2.0
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.mean(D[iu]))


def coherence_from_embs(
    labels: Sequence[Optional[int]],
    embs: np.ndarray,
) -> Dict[str, float]:
    by: Dict[int, List[int]] = defaultdict(list)
    for i, lab in enumerate(labels):
        if lab is not None:
            by[int(lab)].append(i)
    within_vals: List[float] = []
    cents: List[np.ndarray] = []
    for sid, idxs in by.items():
        if len(idxs) < 2:
            continue
        sub = embs[np.asarray(idxs)]
        within_vals.append(mean_pairwise_cos_diss(sub))
        cents.append(l2(sub.mean(axis=0)))
    within = float(np.mean(within_vals)) if within_vals else float("nan")
    between = mean_pairwise_cos_diss(np.stack(cents, axis=0)) if len(cents) >= 2 else float("nan")
    sep = (between / within) if (within == within and within > 1e-12 and between == between) else float("nan")
    return {"within": within, "between": between, "sep_ratio": sep, "n_mature": float(len(by))}


def label_perm_null(
    labels: Sequence[Optional[int]],
    embs: np.ndarray,
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    obs = coherence_from_embs(labels, embs)
    # Only permute among assigned positions; keep unassigned slots fixed as None
    assigned_idx = [i for i, lab in enumerate(labels) if lab is not None]
    assigned_labs = [labels[i] for i in assigned_idx]
    rng = np.random.default_rng(seed)
    null_sep = np.empty(n_perm, dtype=float)
    null_within = np.empty(n_perm, dtype=float)
    for b in range(n_perm):
        perm = rng.permutation(len(assigned_labs))
        labs = list(labels)
        for j, i in enumerate(assigned_idx):
            labs[i] = assigned_labs[perm[j]]
        c = coherence_from_embs(labs, embs)
        null_sep[b] = c["sep_ratio"]
        null_within[b] = c["within"]
    def _z(obs_v: float, null: np.ndarray, higher_better: bool) -> Tuple[float, float]:
        finite = null[np.isfinite(null)]
        if not finite.size or not (obs_v == obs_v):
            return float("nan"), float("nan")
        mu = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        z = (obs_v - mu) / sd if sd and sd > 0 else float("nan")
        if higher_better:
            p = float((int(np.sum(finite >= obs_v)) + 1) / (finite.size + 1))
        else:
            p = float((int(np.sum(finite <= obs_v)) + 1) / (finite.size + 1))
        return z, p

    z_sep, p_sep = _z(obs["sep_ratio"], null_sep, higher_better=True)
    z_w, p_w = _z(obs["within"], null_within, higher_better=False)
    return {
        **{f"obs_{k}": v for k, v in obs.items()},
        "null_sep_mean": float(np.nanmean(null_sep)),
        "null_within_mean": float(np.nanmean(null_within)),
        "z_sep": z_sep,
        "p_sep": p_sep,
        "z_within": z_w,
        "p_within": p_w,
    }


def collect_prompts() -> Tuple[List[Tuple[str, str, int, List[str], List[Optional[int]]]], List[str]]:
    """Return per-run (distance, clustering, seed, prompts, labels) and unique prompt list."""
    runs = []
    all_prompts: List[str] = []
    seen = {}
    for distance, clustering in LIVE_DIR:
        for seed in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
            if not complete(run_dir):
                continue
            genomes = load_genomes(run_dir, clustering)
            prompts: List[str] = []
            labels: List[Optional[int]] = []
            for g in genomes:
                if illicit_of(g) is None:
                    continue
                text = g.get("prompt") or ""
                if not isinstance(text, str):
                    text = str(text)
                prompts.append(text)
                labels.append(sid_int(g))
                if text not in seen:
                    seen[text] = len(all_prompts)
                    all_prompts.append(text)
            runs.append((distance, clustering, seed, prompts, labels))
    return runs, all_prompts


def embed_mpnet(texts: Sequence[str], cache_path: Path) -> np.ndarray:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.is_file():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            print(f"Loaded mpnet cache {cache_path} {arr.shape}")
            return arr
    from sentence_transformers import SentenceTransformer

    print(f"Encoding {len(texts)} prompts with {MPNET} ...")
    model = SentenceTransformer(MPNET)
    embs = model.encode(
        list(texts),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    embs = np.asarray(embs, dtype=np.float64)
    np.save(cache_path, embs)
    print(f"Saved {cache_path} {embs.shape}")
    return embs


def embed_tfidf_run(texts: Sequence[str]) -> np.ndarray:
    """Per-run TF-IDF (avoids a dense 77k×vocab matrix)."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=20000,
    )
    X = vec.fit_transform(list(texts))
    return normalize(X, norm="l2", axis=1).toarray().astype(np.float64)


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x: Any, nd: int = 2) -> str:
    try:
        v = float(x)
    except Exception:
        return "---"
    if not math.isfinite(v):
        return "---"
    return f"{v:.{nd}f}"


def write_latex(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by rq3_coherence_independent.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Independent-representation species coherence (median over "
        r"$n{=}10$ runs). Within/between use cosine dissimilarity $(1-\cos)/2$ "
        r"under \texttt{all-mpnet-base-v2} or TF--IDF character $n$-grams. "
        r"$z_{\mathrm{sep}}$ is the separation-ratio $z$-score versus a "
        r"size-preserving label permutation. Positive $z$ means "
        r"niches are more separated than chance under the same sizes.}",
        r"\label{tab:rq3-coherence-indep}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Dissim. & Method & mpnet $z_{\mathrm{sep}}$ & mpnet within & "
        r"TF--IDF $z_{\mathrm{sep}}$ & TF--IDF within \\",
        r"\midrule",
    ]
    by = {(s["distance"], s["clustering"]): s for s in summaries}
    for distance, clustering in LIVE_DIR:
        s = by.get((distance, clustering))
        if not s:
            continue
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_fmt(s['mpnet_z_sep'], 2)} & {_fmt(s['mpnet_within'], 3)} & "
            f"{_fmt(s['tfidf_z_sep'], 2)} & {_fmt(s['tfidf_within'], 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Independent-representation coherence")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--skip-mpnet", action="store_true")
    p.add_argument("--skip-tfidf", action="store_true")
    args = p.parse_args()

    print("Collecting prompts ...")
    runs, all_prompts = collect_prompts()
    print(f"{len(runs)} runs, {len(all_prompts)} unique prompts")
    # Hash of prompt list for cache key
    h = hashlib.sha1("\n".join(all_prompts).encode("utf-8")).hexdigest()[:12]
    prompt_index = {t: i for i, t in enumerate(all_prompts)}

    emb_mpnet = None
    if not args.skip_mpnet:
        emb_mpnet = embed_mpnet(all_prompts, CACHE / f"mpnet_{h}.npy")

    rows: List[Dict[str, Any]] = []
    for distance, clustering, seed, prompts, labels in runs:
        idxs = np.asarray([prompt_index[t] for t in prompts], dtype=int)
        row: Dict[str, Any] = {
            "distance": distance,
            "clustering": clustering,
            "seed": seed,
            "n": len(prompts),
            "n_assigned": sum(1 for lab in labels if lab is not None),
        }
        if emb_mpnet is not None:
            print(f"mpnet null {distance} {clustering} seed{seed}", flush=True)
            stats = label_perm_null(
                labels,
                emb_mpnet[idxs],
                n_perm=args.n_perm,
                seed=PERM_SEED + seed,
            )
            row.update({f"mpnet_{k}": v for k, v in stats.items()})
        if not args.skip_tfidf:
            print(f"tfidf null {distance} {clustering} seed{seed}", flush=True)
            emb_tf = embed_tfidf_run(prompts)
            stats = label_perm_null(
                labels,
                emb_tf,
                n_perm=args.n_perm,
                seed=PERM_SEED + 1000 + seed,
            )
            row.update({f"tfidf_{k}": v for k, v in stats.items()})
        rows.append(row)

    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "rq3_coherence_independent_runs.csv", rows)

    summaries: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        summaries.append(
            {
                "distance": distance,
                "clustering": clustering,
                "mpnet_z_sep": float(np.nanmedian([r.get("mpnet_z_sep", float("nan")) for r in sub])),
                "mpnet_within": float(np.nanmedian([r.get("mpnet_obs_within", float("nan")) for r in sub])),
                "mpnet_sep": float(np.nanmedian([r.get("mpnet_obs_sep_ratio", float("nan")) for r in sub])),
                "tfidf_z_sep": float(np.nanmedian([r.get("tfidf_z_sep", float("nan")) for r in sub])),
                "tfidf_within": float(np.nanmedian([r.get("tfidf_obs_within", float("nan")) for r in sub])),
                "tfidf_sep": float(np.nanmedian([r.get("tfidf_obs_sep_ratio", float("nan")) for r in sub])),
                "n_sig_mpnet": sum(1 for r in sub if float(r.get("mpnet_p_sep", 1) or 1) < 0.05),
                "n_sig_tfidf": sum(1 for r in sub if float(r.get("tfidf_p_sep", 1) or 1) < 0.05),
            }
        )
    write_csv(REPORT / "rq3_coherence_independent_summary.csv", summaries)
    write_latex(NOTES / "rq3_coherence_independent_table.tex", summaries)

    meta = {
        "mpnet_model": MPNET,
        "tfidf": "char_wb ngram 3-5 max_features=50000",
        "n_perm": args.n_perm,
        "n_unique_prompts": len(all_prompts),
        "n_runs": len(rows),
        "prompt_hash": h,
        "summaries": summaries,
    }
    (REPORT / "rq3_coherence_independent_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )
    for s in summaries:
        print(
            DIST_TEX[s["distance"]],
            CLUS_TEX[s["clustering"]],
            f"mpnet_z={s['mpnet_z_sep']:.2f}",
            f"tfidf_z={s['tfidf_z_sep']:.2f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
