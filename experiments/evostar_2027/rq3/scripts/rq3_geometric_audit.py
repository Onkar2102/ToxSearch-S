#!/usr/bin/env python3
"""RQ3 geometric search-space audit (no human annotation).

Primary representation: MiniLM d_E = (1-cos)/2 on L2-normalized embeddings.
Reference set A: frozen B3139 calibration archive embeddings (descriptive;
used for threshold selection in RQ1 — not an independent hold-out).

Metrics per run:
  - high-fitness yield |H_r(tau)|
  - top-k pairwise dispersion D_r,k
  - fitness-conditioned k-NN novelty
  - reference coverage C_r(delta, tau) and rarefied coverage at matched m

Optional robustness: --second-model (default all-mpnet-base-v2).

  PYTHONPATH=src:. MPLBACKEND=Agg venv/bin/python \\
    experiments/evostar_2027/rq3/scripts/rq3_geometric_audit.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}
POOL = ROOT / "experiments" / "evostar_2027" / "data" / "pool" / "p0_full.pkl"
DYN = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "rq2_species_dynamics.csv"
REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
FIG = ROOT / "experiments" / "evostar_2027" / "rq3" / "figures"
NOTES = ROOT / "notes" / "rough_draft"
CACHE = REPORT / "embed_cache"

DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-12)
    return x / n


def d_E_matrix(emb: np.ndarray) -> np.ndarray:
    """Pairwise (1-cos)/2 for L2-normalized rows."""
    e = l2_normalize(emb)
    cos = np.clip(e @ e.T, -1.0, 1.0)
    return (1.0 - cos) / 2.0


def d_E_cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aa = l2_normalize(a)
    bb = l2_normalize(b)
    cos = np.clip(aa @ bb.T, -1.0, 1.0)
    return (1.0 - cos) / 2.0


def mean_pairwise(D: np.ndarray) -> float:
    n = D.shape[0]
    if n < 2:
        return float("nan")
    iu = np.triu_indices(n, k=1)
    return float(np.mean(D[iu]))


def novelty_mean(D: np.ndarray, k: int) -> float:
    n = D.shape[0]
    if n < 2:
        return float("nan")
    kk = min(k, n - 1)
    vals = []
    for i in range(n):
        row = D[i].copy()
        row[i] = np.inf
        nn = np.partition(row, kk - 1)[:kk]
        vals.append(float(np.mean(nn)))
    return float(np.mean(vals))


def coverage_fraction(ref: np.ndarray, disc: np.ndarray, delta: float) -> float:
    if disc.shape[0] == 0:
        return 0.0
    cross = d_E_cross(ref, disc)  # (|A|, |H|)
    nearest = cross.min(axis=1)
    return float(np.mean(nearest <= delta))


def rarefied_coverage(
    ref: np.ndarray,
    disc: np.ndarray,
    delta: float,
    m: int,
    draws: int,
    rng: np.random.Generator,
) -> float:
    n = disc.shape[0]
    if n < m:
        return float("nan")
    vals = []
    for _ in range(draws):
        idx = rng.choice(n, size=m, replace=False)
        vals.append(coverage_fraction(ref, disc[idx], delta))
    return float(np.mean(vals))


def load_calibration_emb() -> np.ndarray:
    df = pickle.load(POOL.open("rb"))
    embs = np.stack([np.asarray(e, dtype=np.float64) for e in df["emb"].tolist()], axis=0)
    return l2_normalize(embs)


def load_study_pool_emb(
    max_ref: int = 5000,
    seed: int = 0,
) -> Tuple[np.ndarray, int, int]:
    """Union of unique prompts across all 40 runs, then freeze a subsample.

    Full union can exceed ~70k prompts; coverage would be O(|A|·|H|) × rarefaction
    draws and stall interactively. We therefore subsample a fixed reference of
    size ``max_ref`` with a frozen RNG seed (identical for every run).
    """
    seen = {}
    for distance, clustering in LIVE_DIR:
        for s in range(10):
            run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{s}_s{s}"
            if not complete(run_dir):
                continue
            for rec in load_genomes(run_dir, clustering):
                prompt = str(rec.get("prompt") or "").strip()
                if not prompt:
                    continue
                emb = genome_emb(rec)
                if emb is None:
                    continue
                if prompt not in seen:
                    seen[prompt] = emb
    if not seen:
        raise RuntimeError("empty study pool reference")
    keys = sorted(seen.keys())  # deterministic order before subsample
    n_full = len(keys)
    rng = np.random.default_rng(seed)
    if n_full > max_ref:
        pick = rng.choice(n_full, size=max_ref, replace=False)
        keys = [keys[i] for i in sorted(int(i) for i in pick)]
    embs = l2_normalize(np.stack([seen[k] for k in keys], axis=0))
    return embs, embs.shape[0], n_full


def load_end_dynamics() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    if not DYN.is_file():
        return out
    with DYN.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                b = int(float(row.get("budget") or 0))
            except Exception:
                continue
            if b != 2000:
                continue
            key = (row["distance"], row["clustering"], int(row["seed"]))

            def _f(name: str) -> float:
                try:
                    return float(row[name])
                except Exception:
                    return float("nan")

            out[key] = {"n_mature": _f("n_mature"), "d_assigned": _f("d_assigned")}
    return out


def genome_emb(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    emb = rec.get("prompt_embedding")
    if isinstance(emb, list) and len(emb) == 384:
        return l2_normalize(np.asarray(emb, dtype=np.float64))
    return None


def encode_prompts(prompts: Sequence[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    CACHE.mkdir(parents=True, exist_ok=True)
    key = hashlib_prompts(prompts, model_name)
    path = CACHE / f"{key}.npy"
    if path.is_file():
        return np.load(path)
    model = SentenceTransformer(model_name)
    emb = np.asarray(model.encode(list(prompts), normalize_embeddings=True), dtype=np.float64)
    np.save(path, emb)
    return emb


def hashlib_prompts(prompts: Sequence[str], model_name: str) -> str:
    import hashlib

    h = hashlib.sha256()
    h.update(model_name.encode())
    for p in prompts:
        h.update(b"\0")
        h.update(p.encode("utf-8", errors="replace"))
    return h.hexdigest()[:24]


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text()).get("status")) == "complete"
    except Exception:
        return False


def analyze_run(
    distance: str,
    clustering: str,
    seed: int,
    ref_cal: np.ndarray,
    ref_study: np.ndarray,
    tau: float,
    deltas: Sequence[float],
    ks: Sequence[int],
    novelty_k: int,
    rare_ms: Sequence[int],
    rare_draws: int,
    rng: np.random.Generator,
    dyn: Dict[Tuple[str, str, int], Dict[str, float]],
) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    scored: List[Tuple[float, np.ndarray, str]] = []
    for rec in genomes:
        f = illicit_of(rec)
        if f is None:
            continue
        emb = genome_emb(rec)
        prompt = str(rec.get("prompt") or "")
        if emb is None:
            continue
        scored.append((float(f), emb, prompt))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    fits = np.asarray([s[0] for s in scored], dtype=np.float64)
    embs = np.stack([s[1] for s in scored], axis=0)

    high_mask = fits >= tau
    H = embs[high_mask]
    n_high = int(H.shape[0])

    row: Dict[str, Any] = {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "tau": tau,
        "n_unique_scored": int(len(scored)),
        "n_high": n_high,
        "n_mature_b2000": dyn.get((distance, clustering, seed), {}).get("n_mature", float("nan")),
        "d_assigned_b2000": dyn.get((distance, clustering, seed), {}).get("d_assigned", float("nan")),
    }

    for k in ks:
        top = embs[: min(k, embs.shape[0])]
        row[f"disp_top{k}"] = mean_pairwise(d_E_matrix(top)) if top.shape[0] >= 2 else float("nan")
        # Primary coverage estimand: fixed-count fitness-selected elites E_{r,k}
        # (avoids tau-filtered rarefaction when |H_r|<m).
        for prefix, ref in (("cal", ref_cal), ("study", ref_study)):
            for delta in deltas:
                row[f"cov_{prefix}_top{k}_d{delta:.2f}"] = (
                    coverage_fraction(ref, top, delta) if top.shape[0] else 0.0
                )
        if n_high >= 2:
            row[f"disp_H_top{k}"] = mean_pairwise(d_E_matrix(H[: min(k, n_high)]))
        else:
            row[f"disp_H_top{k}"] = float("nan")

    if n_high >= 2:
        DH = d_E_matrix(H)
        row["novelty_k"] = novelty_mean(DH, novelty_k)
        row["disp_H_all"] = mean_pairwise(DH)
    else:
        row["novelty_k"] = float("nan")
        row["disp_H_all"] = float("nan")

    # Absolute-quality sensitivity: coverage by H_r(tau) (not primary; may have |H|<10).
    for prefix, ref in (("cal", ref_cal), ("study", ref_study)):
        for delta in deltas:
            row[f"cov_{prefix}_d{delta:.2f}"] = (
                coverage_fraction(ref, H, delta) if n_high else 0.0
            )

    # Rarefaction of H_r: returns NaN when |H|<m (do not silently shrink m).
    rare_deltas = [d for d in deltas if d in (0.20, 0.30)] or list(deltas[:1])
    for delta in rare_deltas:
        for m in rare_ms:
            row[f"cov_study_raf_m{m}_d{delta:.2f}"] = (
                rarefied_coverage(ref_study, H, delta, m, rare_draws, rng) if n_high else float("nan")
            )

    return row


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    mask = np.isfinite(xa) & np.isfinite(ya)
    xa, ya = xa[mask], ya[mask]
    if xa.size < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        return float(spearmanr(xa, ya).statistic)
    except Exception:
        rx = np.argsort(np.argsort(xa)).astype(float)
        ry = np.argsort(np.argsort(ya)).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])


def summarize(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    out = []
    for distance, clustering in LIVE_DIR:
        vals = [
            float(r[key])
            for r in rows
            if r["distance"] == distance
            and r["clustering"] == clustering
            and r.get(key) == r.get(key)
            and math.isfinite(float(r.get(key, float("nan"))))
        ]
        if not vals:
            continue
        out.append(
            {
                "distance": distance,
                "clustering": clustering,
                "metric": key,
                "n": len(vals),
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
            }
        )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_latex(path: Path, rows: List[Dict[str, Any]], delta: float, m: int) -> None:
    elite_k = int(m)
    cov = f"cov_study_top{elite_k}_d{delta:.2f}"
    cov_cal = f"cov_cal_top{elite_k}_d{delta:.2f}"
    disp = f"disp_top{elite_k}"
    lines = [
        "% Auto-generated by rq3_geometric_audit.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{RQ3 elite exploration in MiniLM space. For each run, "
        rf"$E_{{r,{elite_k}}}$ is the {elite_k} fittest unique evaluated prompts "
        rf"(fixed count; not a $\tau{{=}}0.80$ filter). $D_{{r,{elite_k}}}$: mean "
        rf"pairwise $d_E$ on $E_{{r,{elite_k}}}$. "
        rf"$C_r^{{\mathrm{{top{elite_k}}}}}(\delta)$: coverage of the frozen pooled-study "
        rf"reference subsample by those {elite_k} elites. "
        r"$C_r^{\mathrm{cal}}$ uses the RQ1 calibration archive as a descriptive "
        rf"contrast at the same $\delta={delta:.2f}$ (not an independent hold-out). "
        r"$|H_r(\tau{=}0.80)|$ is reported only as absolute-quality context. "
        r"Representation-space geometry only---not validated behavioral diversity.}",
        r"\label{tab:rq3-geometric}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        rf"Dissim. & Method & $n$ & $|H_r|$ & $D_{{r,{elite_k}}}$ & "
        rf"$C_r^{{\mathrm{{top{elite_k}}}}}({delta:.2f})$ & $C_r^{{\mathrm{{cal}}}}$ \\",
        r"\midrule",
    ]
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue

        def med(key: str) -> str:
            vals = [float(r[key]) for r in sub if math.isfinite(float(r.get(key, float("nan"))))]
            return "---" if not vals else f"{np.median(vals):.3f}"

        def med_i(key: str) -> str:
            vals = [float(r[key]) for r in sub if math.isfinite(float(r.get(key, float("nan"))))]
            return "---" if not vals else f"{np.median(vals):.0f}"

        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & {len(sub)} & "
            f"{med_i('n_high')} & {med(disp)} & {med(cov)} & {med(cov_cal)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    # Pooled associations intentionally omitted from main-text TeX (Simpson risk; report §P0.4).
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_elite_panels(
    rows: List[Dict[str, Any]],
    delta: float,
    out: Path,
    *,
    elite_k: int = 100,
) -> None:
    """R3-F: run-level top-k dispersion and study coverage (two panels)."""
    cov_key = f"cov_study_top{elite_k}_d{delta:.2f}"
    disp_key = f"disp_top{elite_k}"
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.7), layout="constrained")
    colors = {"leader_follower": "#0F4D92", "dbscan": "#B64342"}
    xpos = 0
    tick_pos = []
    tick_lab = []
    for distance in ("embedding", "nli_entail"):
        for clustering in ("leader_follower", "dbscan"):
            xpos += 1
            tick_pos.append(xpos)
            tick_lab.append(f"{DIST_TEX[distance]}\n{CLUS_TEX[clustering]}")
            sub = [
                r
                for r in rows
                if r["distance"] == distance and r["clustering"] == clustering
            ]
            dvals = [
                float(r[disp_key])
                for r in sub
                if math.isfinite(float(r.get(disp_key, float("nan"))))
            ]
            cvals = [float(r[cov_key]) for r in sub if math.isfinite(float(r.get(cov_key, float("nan"))))]
            col = colors[clustering]
            ax0.scatter(np.full(len(dvals), xpos), dvals, color=col, alpha=0.7, s=26, zorder=3)
            ax1.scatter(np.full(len(cvals), xpos), cvals, color=col, alpha=0.7, s=26, zorder=3)
            if dvals:
                ax0.hlines(np.median(dvals), xpos - 0.28, xpos + 0.28, colors="black", lw=1.4, zorder=4)
            if cvals:
                ax1.hlines(np.median(cvals), xpos - 0.28, xpos + 0.28, colors="black", lw=1.4, zorder=4)
    for ax, ylab, title in (
        (ax0, rf"$D_{{r,{elite_k}}}$", "(a)"),
        (ax1, rf"$C_r^{{\mathrm{{top{elite_k}}}}}(\delta={delta:.2f})$", "(b)"),
    ):
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab, fontsize=7.5)
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=9, fontweight="bold", pad=2)
        ax.grid(True, axis="y", alpha=0.25)
    ax0.set_ylim(bottom=0.0)
    ax1.set_ylim(bottom=0.0)
    out.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FIG = NOTES / "figures"
    NOTES_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200)
    fig.savefig(NOTES_FIG / out.name, dpi=200)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    fig.savefig(NOTES_FIG / out.with_suffix(".png").name, dpi=200)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--deltas", type=str, default="0.10,0.20,0.30,0.40")
    ap.add_argument("--ks", type=str, default="10,50,100")
    ap.add_argument("--novelty-k", type=int, default=5)
    ap.add_argument("--rare-ms", type=str, default="5,10,25")
    ap.add_argument("--rare-draws", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--primary-delta", type=float, default=0.20)
    ap.add_argument("--max-ref", type=int, default=5000, help="subsample size for study-pool reference")
    args = ap.parse_args()

    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    rare_ms = [int(x) for x in args.rare_ms.split(",") if x.strip()]

    print("loading calibration + study-pool references...", flush=True)
    ref_cal = load_calibration_emb()
    ref_study, n_study, n_study_full = load_study_pool_emb(max_ref=args.max_ref, seed=args.seed)
    print(
        f"|A_cal|={ref_cal.shape[0]}  |A_study|={n_study} (from {n_study_full} unique)",
        flush=True,
    )
    dyn = load_end_dynamics()
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        for seed in range(10):
            print(f"geo {distance} {clustering} seed{seed}", flush=True)
            row = analyze_run(
                distance,
                clustering,
                seed,
                ref_cal,
                ref_study,
                args.tau,
                deltas,
                ks,
                args.novelty_k,
                rare_ms,
                args.rare_draws,
                rng,
                dyn,
            )
            if row:
                rows.append(row)

    REPORT.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    head = [
        "distance",
        "clustering",
        "seed",
        "tau",
        "n_unique_scored",
        "n_high",
        "n_mature_b2000",
        "d_assigned_b2000",
        "disp_top10",
        "disp_top50",
        "disp_top100",
        "disp_H_all",
        "novelty_k",
    ]
    fields = [f for f in head if f in fields] + [f for f in fields if f not in head]
    write_csv(REPORT / "rq3_geometric_runs.csv", rows, fields)

    elite_k = 100
    summ = []
    for key in [
        f"disp_top{elite_k}",
        f"cov_study_top{elite_k}_d{args.primary_delta:.2f}",
        f"cov_cal_top{elite_k}_d{args.primary_delta:.2f}",
        "n_high",
        f"cov_study_d{args.primary_delta:.2f}",
        f"cov_study_raf_m10_d{args.primary_delta:.2f}",
    ]:
        summ.extend(summarize(rows, key))
    write_csv(
        REPORT / "rq3_geometric_summary.csv",
        summ,
        ["distance", "clustering", "metric", "n", "median", "q25", "q75"],
    )

    write_latex(NOTES / "rq3_geometric_tables.tex", rows, args.primary_delta, m=elite_k)
    plot_elite_panels(
        rows, args.primary_delta, FIG / "fig_rq3_elite_geo.pdf", elite_k=elite_k
    )
    # Keep legacy coverage plot filename as tau-sensitivity diagnostic (supplement-oriented).
    plot_elite_panels(
        rows, args.primary_delta, FIG / "fig_rq3_coverage_geo.pdf", elite_k=elite_k
    )

    meta = {
        "n_runs": len(rows),
        "primary_estimand": f"fixed top-{elite_k} fitness-selected elites E_r,{elite_k} (dispersion + study/cal coverage)",
        "tau_sensitivity": args.tau,
        "deltas": deltas,
        "primary_delta": args.primary_delta,
        "ks": ks,
        "novelty_k": args.novelty_k,
        "rare_ms": rare_ms,
        "reference_cal": str(POOL),
        "reference_study": "subsampled union of unique prompts across 40 runs (empirical reference, not hold-out)",
        "n_study_ref": n_study,
        "n_study_full": n_study_full,
        "max_ref": args.max_ref,
        "reference_note": "Primary=study pool; cal=B3139 descriptive (also RQ1). Not behavioral diversity.",
        "claim_boundary": "semantic representation-space exploration, not behavioral diversity",
    }
    (REPORT / "rq3_geometric_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} runs -> {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
