#!/usr/bin/env python3
"""RQ3 species-size-preserving null model for top-k elite occupancy.

Keeps final (or discovery-time) species membership fixed and permutes fitness
among evaluated individuals. Recomputes K, U, and pi under the null to
test whether elite concentration exceeds what species sizes alone predict.
Primary elite set size is ``ELITE_K`` (currently 100).

Variants
--------
- ``all``: permute fitness across every moderated genome
- ``assigned``: permute only among mature-species members (unassigned keep
  their observed fitness ranking relative to the assigned pool)
- ``discovery``: use species labels at discovery time from
  ``species_snapshots.jsonl`` instead of the final snapshot

Example::

    MPLBACKEND=Agg PYTHONPATH=src:. venv/bin/python \\
      experiments/evostar_2027/rq3/scripts/rq3_occupancy_null.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
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
REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
FIG = ROOT / "experiments" / "evostar_2027" / "rq3" / "figures"
NOTES = ROOT / "notes" / "rough_draft"
NOTES_FIG = NOTES / "figures"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}
DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}
NOISE = {None, 0, -1, "0", "-1"}

N_PERM = 10_000
PERM_SEED = 2027
TAU = 0.80
ELITE_K = 100  # primary elite set size (K_{100}, pi_{100})

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "dejavusans",
        "savefig.dpi": 400,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


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


def discovery_labels(run_dir: Path, genome_ids: Sequence[Any]) -> Dict[Any, Optional[int]]:
    """First mature species label seen for each genome in species_snapshots.jsonl."""
    snap = run_dir / "species_snapshots.jsonl"
    first: Dict[Any, Optional[int]] = {gid: None for gid in genome_ids}
    pending = set(str(g) for g in genome_ids)
    if not snap.is_file():
        return first
    with snap.open(encoding="utf-8") as f:
        for line in f:
            if not pending:
                break
            try:
                obj = json.loads(line)
            except Exception:
                continue
            memb = obj.get("membership") or {}
            if not isinstance(memb, dict):
                continue
            for gid_s in list(pending):
                if gid_s not in memb:
                    continue
                lab = memb[gid_s]
                try:
                    li = int(lab)
                except Exception:
                    continue
                if li > 0:
                    # resolve original key type
                    for gid in genome_ids:
                        if str(gid) == gid_s and first[gid] is None:
                            first[gid] = li
                    pending.discard(gid_s)
    return first


def occupancy(
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    k: int = ELITE_K,
) -> Dict[str, float]:
    order = np.argsort(-np.asarray(fitness, dtype=float))
    top = order[:k]
    top_labs = [labels[i] for i in top]
    assigned = [lab for lab in top_labs if lab is not None]
    u10 = float(sum(1 for lab in top_labs if lab is None))
    k10 = float(len(set(assigned)))
    if assigned:
        pi10 = float(max(Counter(assigned).values()) / k)
    else:
        pi10 = 0.0
    # Keys retain the historical K10/pi10 names; ELITE_K sets the elite count.
    return {"K10": k10, "U10": u10, "pi10": pi10}


def high_yield(
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    tau: float = TAU,
) -> Dict[str, float]:
    H_idx = [i for i, f in enumerate(fitness) if f >= tau]
    n_high = len(H_idx)
    H_labs = [labels[i] for i in H_idx if labels[i] is not None]
    n_high_unass = sum(1 for i in H_idx if labels[i] is None)
    n_species_with_high = len(set(H_labs))
    share = float(max(Counter(H_labs).values()) / n_high) if H_labs and n_high else float("nan")
    mature = {lab for lab in labels if lab is not None}
    n_mat = len(mature)
    frac = (n_species_with_high / n_mat) if n_mat else float("nan")
    return {
        "n_high": float(n_high),
        "n_high_unass": float(n_high_unass),
        "n_species_with_high": float(n_species_with_high),
        "share_H_maxsp": share,
        "frac_species_with_high": frac,
        "n_mature": float(n_mat),
    }


def permute_null(
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    *,
    n_perm: int,
    seed: int,
    mode: str,
) -> Dict[str, Any]:
    fit = np.asarray(fitness, dtype=float)
    labs = list(labels)
    obs = occupancy(labs, fit, k=ELITE_K)
    obs_h = high_yield(labs, fit)

    rng = np.random.default_rng(seed)
    null_k = np.empty(n_perm, dtype=float)
    null_u = np.empty(n_perm, dtype=float)
    null_pi = np.empty(n_perm, dtype=float)
    null_share_h = np.empty(n_perm, dtype=float)
    null_frac_h = np.empty(n_perm, dtype=float)

    if mode == "assigned":
        idx = np.asarray([i for i, lab in enumerate(labs) if lab is not None], dtype=int)
    else:
        idx = np.arange(len(fit))

    for b in range(n_perm):
        shuffled = fit.copy()
        if idx.size > 1:
            perm = rng.permutation(idx.size)
            shuffled[idx] = fit[idx][perm]
        o = occupancy(labs, shuffled, k=ELITE_K)
        h = high_yield(labs, shuffled)
        null_k[b] = o["K10"]
        null_u[b] = o["U10"]
        null_pi[b] = o["pi10"]
        null_share_h[b] = h["share_H_maxsp"] if h["share_H_maxsp"] == h["share_H_maxsp"] else np.nan
        null_frac_h[b] = (
            h["frac_species_with_high"]
            if h["frac_species_with_high"] == h["frac_species_with_high"]
            else np.nan
        )

    def _stats(obs_v: float, null: np.ndarray, *, concentration: bool) -> Dict[str, float]:
        finite = null[np.isfinite(null)]
        if finite.size == 0 or not math.isfinite(obs_v):
            return {
                "obs": obs_v,
                "null_mean": float("nan"),
                "null_sd": float("nan"),
                "z": float("nan"),
                "p": float("nan"),
            }
        mu = float(np.mean(finite))
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan")
        z = (obs_v - mu) / sd if sd and sd == sd and sd > 0 else float("nan")
        # concentration: fewer species / higher pi than null
        if concentration:
            if obs_v <= mu:  # K10: smaller is more concentrated
                extreme = int(np.sum(finite <= obs_v))
            else:  # pi10: larger is more concentrated — handled by caller flag
                extreme = int(np.sum(finite >= obs_v))
        else:
            extreme = int(np.sum(finite <= obs_v))
        p = (extreme + 1) / (finite.size + 1)
        return {
            "obs": float(obs_v),
            "null_mean": mu,
            "null_sd": sd,
            "z": float(z) if z == z else float("nan"),
            "p": float(p),
        }

    # K: concentration = fewer species among the top-ELITE_K
    sk = _stats(obs["K10"], null_k, concentration=True)
    # override extreme count for K (lower = more concentrated)
    finite_k = null_k[np.isfinite(null_k)]
    sk["p"] = float((int(np.sum(finite_k <= obs["K10"])) + 1) / (finite_k.size + 1)) if finite_k.size else float("nan")
    # pi: higher share = more concentrated
    sp = _stats(obs["pi10"], null_pi, concentration=True)
    finite_pi = null_pi[np.isfinite(null_pi)]
    sp["p"] = float((int(np.sum(finite_pi >= obs["pi10"])) + 1) / (finite_pi.size + 1)) if finite_pi.size else float("nan")
    su = _stats(obs["U10"], null_u, concentration=False)

    finite_sh = null_share_h[np.isfinite(null_share_h)]
    sh_obs = obs_h["share_H_maxsp"]
    if finite_sh.size and sh_obs == sh_obs:
        sh_p = float((int(np.sum(finite_sh >= sh_obs)) + 1) / (finite_sh.size + 1))
        sh_mu = float(np.mean(finite_sh))
        sh_sd = float(np.std(finite_sh, ddof=1)) if finite_sh.size > 1 else float("nan")
        sh_z = (sh_obs - sh_mu) / sh_sd if sh_sd and sh_sd > 0 else float("nan")
    else:
        sh_p = sh_mu = sh_sd = sh_z = float("nan")

    finite_fr = null_frac_h[np.isfinite(null_frac_h)]
    fr_obs = obs_h["frac_species_with_high"]
    if finite_fr.size and fr_obs == fr_obs:
        fr_p = float((int(np.sum(finite_fr <= fr_obs)) + 1) / (finite_fr.size + 1))
        fr_mu = float(np.mean(finite_fr))
        fr_sd = float(np.std(finite_fr, ddof=1)) if finite_fr.size > 1 else float("nan")
        fr_z = (fr_obs - fr_mu) / fr_sd if fr_sd and fr_sd > 0 else float("nan")
    else:
        fr_p = fr_mu = fr_sd = fr_z = float("nan")

    return {
        "K10_obs": sk["obs"],
        "K10_null_mean": sk["null_mean"],
        "K10_null_sd": sk["null_sd"],
        "K10_z": sk["z"],
        "K10_p": sk["p"],
        "U10_obs": su["obs"],
        "U10_null_mean": su["null_mean"],
        "U10_null_sd": su["null_sd"],
        "U10_z": su["z"],
        "U10_p": su["p"],
        "pi10_obs": sp["obs"],
        "pi10_null_mean": sp["null_mean"],
        "pi10_null_sd": sp["null_sd"],
        "pi10_z": sp["z"],
        "pi10_p": sp["p"],
        "share_H_obs": sh_obs,
        "share_H_null_mean": sh_mu,
        "share_H_z": sh_z,
        "share_H_p": sh_p,
        "frac_high_obs": fr_obs,
        "frac_high_null_mean": fr_mu,
        "frac_high_z": fr_z,
        "frac_high_p": fr_p,
        "n_high": obs_h["n_high"],
        "n_mature": obs_h["n_mature"],
        "null_k_q05": float(np.percentile(finite_k, 5)) if finite_k.size else float("nan"),
        "null_k_q95": float(np.percentile(finite_k, 95)) if finite_k.size else float("nan"),
        "null_pi_q05": float(np.percentile(finite_pi, 5)) if finite_pi.size else float("nan"),
        "null_pi_q95": float(np.percentile(finite_pi, 95)) if finite_pi.size else float("nan"),
    }


def analyze_run(
    distance: str,
    clustering: str,
    seed: int,
    *,
    n_perm: int,
    mode: str,
    label_source: str,
) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    ids = [g.get("id") for g in genomes]
    fitness: List[float] = []
    kept: List[Dict[str, Any]] = []
    for g in genomes:
        f = illicit_of(g)
        if f is None:
            continue
        fitness.append(float(f))
        kept.append(g)
    if len(kept) < ELITE_K:
        return None

    if label_source == "discovery":
        disc = discovery_labels(run_dir, [g.get("id") for g in kept])
        labels = [disc.get(g.get("id")) for g in kept]
    else:
        labels = [sid_int(g) for g in kept]

    stats = permute_null(
        labels,
        fitness,
        n_perm=n_perm,
        seed=PERM_SEED + seed * 17 + hash((distance, clustering, mode, label_source)) % 10_000,
        mode=mode,
    )
    return {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "mode": mode,
        "label_source": label_source,
        "n_unique": len(kept),
        **stats,
    }


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [1.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        factor = m - rank
        val = min(1.0, float(pvals[i]) * factor)
        running = max(running, val)
        adj[i] = running
    return adj


def stouffer_combine(pvals: Sequence[float]) -> float:
    from scipy.stats import norm

    zs = []
    for p in pvals:
        if not (p == p) or p <= 0 or p >= 1:
            # clamp extremes for numerical stability
            if p == p and p <= 0:
                p = 1e-12
            elif p == p and p >= 1:
                p = 1.0 - 1e-12
            else:
                continue
        zs.append(float(norm.ppf(1.0 - p)))
    if not zs:
        return float("nan")
    z = float(np.sum(zs) / math.sqrt(len(zs)))
    return float(1.0 - norm.cdf(z))


def cell_summaries(rows: List[Dict[str, Any]], metric_p: str = "K10_p") -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    scratch: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        sub = [
            r
            for r in rows
            if r["distance"] == distance and r["clustering"] == clustering
        ]
        if not sub:
            continue
        pvals = [float(r[metric_p]) for r in sub if r.get(metric_p) == r.get(metric_p)]
        n_sig = sum(1 for p in pvals if p < 0.05)
        combined = stouffer_combine(pvals) if pvals else float("nan")
        rec = {
            "distance": distance,
            "clustering": clustering,
            "n_runs": len(sub),
            "metric_p": metric_p,
            "median_obs_K10": float(np.median([r["K10_obs"] for r in sub])),
            "median_null_K10": float(np.median([r["K10_null_mean"] for r in sub])),
            "median_obs_pi10": float(np.median([r["pi10_obs"] for r in sub])),
            "median_null_pi10": float(np.median([r["pi10_null_mean"] for r in sub])),
            "median_K10_z": float(np.nanmedian([r["K10_z"] for r in sub])),
            "median_pi10_z": float(np.nanmedian([r["pi10_z"] for r in sub])),
            "n_sig_p05": n_sig,
            "stouffer_p": combined,
            "median_share_H_obs": float(np.nanmedian([r["share_H_obs"] for r in sub])),
            "median_share_H_null": float(np.nanmedian([r["share_H_null_mean"] for r in sub])),
            "median_frac_high_obs": float(np.nanmedian([r["frac_high_obs"] for r in sub])),
            "median_frac_high_null": float(np.nanmedian([r["frac_high_null_mean"] for r in sub])),
            "mode": sub[0]["mode"],
            "label_source": sub[0]["label_source"],
        }
        scratch.append(rec)
        raw_p.append(combined if combined == combined else float("nan"))
    finite_idx = [i for i, p in enumerate(raw_p) if p == p]
    adj = [float("nan")] * len(raw_p)
    if finite_idx:
        adj_vals = holm_adjust([raw_p[i] for i in finite_idx])
        for ii, vv in zip(finite_idx, adj_vals):
            adj[ii] = vv
    for i, rec in enumerate(scratch):
        rec["stouffer_holm_p"] = adj[i] if adj[i] == adj[i] else ""
        out.append(rec)
    return out


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
    if 0 < abs(v) < 5e-3 and nd >= 3:
        return f"{v:.1e}"
    return f"{v:.{nd}f}"


def write_variants_latex(path: Path, summaries: List[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated from rq3_occupancy_null_variants.csv",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{RQ3 occupancy-null sensitivity: median $K_{{{ELITE_K}}}$ "
        r"(observed vs.\ null) under fitness permutation of all genomes or "
        r"assigned-only, with final or discovery-time labels. Primary main-text "
        r"analysis is all/final.}",
        r"\label{tab:app-rq3-null-variants}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llllcc@{}}",
        r"\toprule",
        rf"Mode & Labels & Dissim. & Method & $K_{{{ELITE_K}}}$ & "
        rf"$K_{{{ELITE_K}}}^{{\mathrm{{null}}}}$ \\",
        r"\midrule",
    ]
    for s in summaries:
        lines.append(
            f"{s['mode']} & {s['label_source']} & "
            f"{DIST_TEX[s['distance']]} & {CLUS_TEX[s['clustering']]} & "
            f"{_fmt(s['median_obs_K10'], 1)} & {_fmt(s['median_null_K10'], 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex(path: Path, summaries: List[Dict[str, Any]], run_rows: List[Dict[str, Any]]) -> None:
    """Main-text RQ3 table: observed vs size-preserving null."""
    del run_rows
    lines = [
        "% Auto-generated by rq3_occupancy_null.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{RQ3 elite occupancy versus a species-size-preserving null "
        rf"(median over $n{{=}}10$ runs; $B{{=}}10{{,}}000$ fitness permutations with "
        rf"labels fixed). $K_{{{ELITE_K}}}$: distinct mature species among the "
        rf"{ELITE_K} fittest individuals; $\pi_{{{ELITE_K}}}$: share of the top "
        rf"{ELITE_K} from one species. "
        r"Null means are the expected values under size alone. Stouffer-combined "
        r"one-sided $p$ (Holm over four cells) tests concentration beyond sizes. "
        r"Lineage-induced fitness autocorrelation remains a caveat.}",
        r"\label{tab:rq3-species}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccccc@{}}",
        r"\toprule",
        rf"Dissim. & Method & $K_{{{ELITE_K}}}$ & $K_{{{ELITE_K}}}^{{\mathrm{{null}}}}$ & "
        rf"$\pi_{{{ELITE_K}}}$ & $\pi_{{{ELITE_K}}}^{{\mathrm{{null}}}}$ & $z_{{K}}$ & Holm $p$ \\",
        r"\midrule",
    ]
    by = {(s["distance"], s["clustering"]): s for s in summaries}
    for distance, clustering in LIVE_DIR:
        s = by.get((distance, clustering))
        if not s:
            continue
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_fmt(s['median_obs_K10'], 1)} & {_fmt(s['median_null_K10'], 1)} & "
            f"{_fmt(s['median_obs_pi10'], 2)} & {_fmt(s['median_null_pi10'], 2)} & "
            f"{_fmt(s['median_K10_z'], 2)} & {_fmt(s.get('stouffer_holm_p'), 3)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Secondary yield table (q_i / h_i proxies)
    lines += [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{High-fitness yield diagnostics at $\tau{=}0.80$ (median). "
        r"$\mathrm{share}_H$: fraction of qualifying individuals in the single "
        r"largest contributing mature species; $\mathrm{frac}_{\mathrm{sp}}$: "
        r"fraction of mature species with at least one qualifier. Null columns "
        r"are size-preserving expectations.}",
        r"\label{tab:rq3-yield}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Dissim. & Method & $\mathrm{share}_H$ & $\mathrm{share}_H^{\mathrm{null}}$ & "
        r"$\mathrm{frac}_{\mathrm{sp}}$ & $\mathrm{frac}_{\mathrm{sp}}^{\mathrm{null}}$ \\",
        r"\midrule",
    ]
    for distance, clustering in LIVE_DIR:
        s = by.get((distance, clustering))
        if not s:
            continue
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_fmt(s['median_share_H_obs'], 2)} & {_fmt(s['median_share_H_null'], 2)} & "
            f"{_fmt(s['median_frac_high_obs'], 2)} & {_fmt(s['median_frac_high_null'], 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_with_null(run_rows: List[Dict[str, Any]], out: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.55), layout="constrained")
    colors = {"leader_follower": "#0F4D92", "dbscan": "#B64342"}
    xpos = 0
    ticks = []
    labs = []
    for distance in ("embedding", "nli_entail"):
        for clustering in ("leader_follower", "dbscan"):
            xpos += 1
            ticks.append(xpos)
            labs.append(f"{DIST_TEX[distance]}\n{CLUS_TEX[clustering]}")
            sub = [
                r
                for r in run_rows
                if r["distance"] == distance and r["clustering"] == clustering
            ]
            col = colors[clustering]
            kvals = [float(r["K10_obs"]) for r in sub]
            pvals = [float(r["pi10_obs"]) for r in sub]
            knull = [float(r["K10_null_mean"]) for r in sub]
            pnull = [float(r["pi10_null_mean"]) for r in sub]
            k_lo = [float(r["null_k_q05"]) for r in sub]
            k_hi = [float(r["null_k_q95"]) for r in sub]
            p_lo = [float(r["null_pi_q05"]) for r in sub]
            p_hi = [float(r["null_pi_q95"]) for r in sub]
            ax0.scatter(np.full(len(kvals), xpos - 0.12), kvals, color=col, alpha=0.75, s=28, zorder=3)
            ax1.scatter(np.full(len(pvals), xpos - 0.12), pvals, color=col, alpha=0.75, s=28, zorder=3)
            if knull:
                ax0.scatter(
                    [xpos + 0.12],
                    [np.median(knull)],
                    marker="D",
                    color="0.35",
                    s=36,
                    zorder=4,
                    label="null median" if xpos == 1 else None,
                )
                # null band from median of per-run 5–95% intervals
                ax0.vlines(xpos + 0.12, np.median(k_lo), np.median(k_hi), colors="0.5", lw=1.2, zorder=2)
            if pnull:
                ax1.scatter([xpos + 0.12], [np.median(pnull)], marker="D", color="0.35", s=36, zorder=4)
                ax1.vlines(xpos + 0.12, np.median(p_lo), np.median(p_hi), colors="0.5", lw=1.2, zorder=2)
            if kvals:
                ax0.hlines(np.median(kvals), xpos - 0.28, xpos + 0.02, colors="black", lw=1.4, zorder=5)
            if pvals:
                ax1.hlines(np.median(pvals), xpos - 0.28, xpos + 0.02, colors="black", lw=1.4, zorder=5)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(labs, fontsize=7.5)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labs, fontsize=7.5)
    ax0.set_ylabel(rf"Distinct species in top-{ELITE_K} ($K_{{{ELITE_K}}}$)")
    ax1.set_ylabel(rf"Share of top-{ELITE_K} from one species ($\pi_{{{ELITE_K}}}$)")
    ax0.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax1.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax0.set_ylim(0, max(32.0, ELITE_K * 0.35))
    ax1.set_ylim(0, 1.05)
    ax0.grid(True, axis="y", alpha=0.25)
    ax1.grid(True, axis="y", alpha=0.25)
    handles, labels = ax0.get_legend_handles_labels()
    if handles:
        ax0.legend(handles, labels, fontsize=7, loc="upper right", frameon=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (
        out,
        NOTES_FIG / out.name,
        out.with_suffix(".png"),
        NOTES_FIG / out.with_suffix(".png").name,
    ):
        fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="RQ3 species-size occupancy null")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument(
        "--modes",
        type=str,
        default="all,assigned",
        help="Comma-separated: all, assigned",
    )
    p.add_argument(
        "--label-sources",
        type=str,
        default="final,discovery",
        help="Comma-separated: final, discovery",
    )
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    label_sources = [x.strip() for x in args.label_sources.split(",") if x.strip()]

    all_rows: List[Dict[str, Any]] = []
    for mode in modes:
        for label_source in label_sources:
            for distance, clustering in LIVE_DIR:
                for seed in seeds:
                    print(
                        f"null {mode}/{label_source} {distance} {clustering} seed{seed}",
                        flush=True,
                    )
                    row = analyze_run(
                        distance,
                        clustering,
                        seed,
                        n_perm=args.n_perm,
                        mode=mode,
                        label_source=label_source,
                    )
                    if row:
                        all_rows.append(row)

    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "rq3_occupancy_null_runs.csv", all_rows)

    # Primary: all genomes, final labels
    primary = [
        r for r in all_rows if r["mode"] == "all" and r["label_source"] == "final"
    ]
    summaries = cell_summaries(primary, metric_p="K10_p")
    write_csv(REPORT / "rq3_occupancy_null_summary.csv", summaries)

    # Also summarize other variants
    variant_summaries: List[Dict[str, Any]] = []
    for mode in modes:
        for label_source in label_sources:
            sub = [
                r
                for r in all_rows
                if r["mode"] == mode and r["label_source"] == label_source
            ]
            for s in cell_summaries(sub, metric_p="K10_p"):
                variant_summaries.append(s)
    write_csv(REPORT / "rq3_occupancy_null_variants.csv", variant_summaries)

    write_latex(NOTES / "rq3_species_tables.tex", summaries, primary)
    write_variants_latex(NOTES / "rq3_null_variants_table.tex", variant_summaries)
    plot_with_null(primary, FIG / "fig_rq3_species_contrib.pdf")

    meta = {
        "n_perm": args.n_perm,
        "elite_k": ELITE_K,
        "perm_seed_base": PERM_SEED,
        "tau": TAU,
        "modes": modes,
        "label_sources": label_sources,
        "n_primary_runs": len(primary),
        "primary_summaries": summaries,
        "caveat": (
            "Within-species fitness autocorrelation from shared lineage remains; "
            "the null tests concentration beyond species sizes, not causation."
        ),
    }
    (REPORT / "rq3_occupancy_null_meta.json").write_text(
        json.dumps(meta, indent=2, default=str), encoding="utf-8"
    )

    for s in summaries:
        print(
            DIST_TEX[s["distance"]],
            CLUS_TEX[s["clustering"]],
            f"K10={s['median_obs_K10']:.1f} vs null {s['median_null_K10']:.1f}",
            f"pi10={s['median_obs_pi10']:.2f} vs null {s['median_null_pi10']:.2f}",
            f"z_K={s['median_K10_z']:.2f}",
            f"Stouffer={s['stouffer_p']:.4g} Holm={s.get('stouffer_holm_p')}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
