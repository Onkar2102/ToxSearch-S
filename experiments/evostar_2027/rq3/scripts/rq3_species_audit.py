#!/usr/bin/env python3
"""RQ3 species-level audit: coherence and high-fitness contribution.

For each completed RQ2 run (final snapshot labels):
  - within-species mean MiniLM d_E among assigned mature members
  - between-species mean MiniLM d_E among species centroids
  - size / assigned-dominance diagnostics
  - among the ten fittest unique individuals: #distinct mature species (K10),
    #unassigned (U10), share from the single most-represented species
  - absolute high-fitness yield |H|(tau=0.80) and fraction of H in the
    largest contributing mature species

  MPLBACKEND=Agg PYTHONPATH=src:. venv/bin/python \\
    experiments/evostar_2027/rq3/scripts/rq3_species_audit.py
"""

from __future__ import annotations

import csv
import json
import math
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


def l2(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x / n if n > 0 else x
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def emb_of(rec: Dict[str, Any]) -> Optional[np.ndarray]:
    e = rec.get("prompt_embedding")
    if isinstance(e, list) and len(e) == 384:
        return l2(np.asarray(e, dtype=np.float64))
    return None


def sid_of(rec: Dict[str, Any]) -> Optional[int]:
    s = rec.get("species_id")
    if s in NOISE:
        return None
    try:
        i = int(s)
    except Exception:
        return None
    return None if i <= 0 else i


def mean_pairwise(embs: np.ndarray) -> float:
    if embs.shape[0] < 2:
        return float("nan")
    e = l2(embs)
    cos = np.clip(e @ e.T, -1.0, 1.0)
    D = (1.0 - cos) / 2.0
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.mean(D[iu]))


def analyze_run(distance: str, clustering: str, seed: int, tau: float = 0.80) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    scored: List[Tuple[float, Optional[int], Optional[np.ndarray]]] = []
    by_sp: Dict[int, List[Tuple[float, np.ndarray]]] = {}
    for rec in genomes:
        f = illicit_of(rec)
        if f is None:
            continue
        emb = emb_of(rec)
        sid = sid_of(rec)
        scored.append((float(f), sid, emb))
        if sid is not None and emb is not None:
            by_sp.setdefault(sid, []).append((float(f), emb))
    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)

    # --- coherence (MiniLM) ---
    within_vals = []
    cents: List[np.ndarray] = []
    sizes = []
    for sid, members in by_sp.items():
        if len(members) < 2:
            continue
        embs = np.stack([m[1] for m in members], axis=0)
        within_vals.append(mean_pairwise(embs))
        cents.append(l2(embs.mean(axis=0)))
        sizes.append(len(members))
    n_mature = len(by_sp)
    n_assigned = sum(len(v) for v in by_sp.values())
    largest = max((len(v) for v in by_sp.values()), default=0)
    d_assigned = (largest / n_assigned) if n_assigned else float("nan")
    within = float(np.mean(within_vals)) if within_vals else float("nan")
    between = mean_pairwise(np.stack(cents, axis=0)) if len(cents) >= 2 else float("nan")

    # --- top-10 contribution ---
    top10 = scored[:10]
    top_sids = [s for _, s, _ in top10 if s is not None]
    u10 = sum(1 for _, s, _ in top10 if s is None)
    k10 = len(set(top_sids))
    if top_sids:
        from collections import Counter

        share_top_sp = max(Counter(top_sids).values()) / 10.0
    else:
        share_top_sp = 0.0
    top10_mean = float(np.mean([f for f, _, _ in top10]))

    # --- absolute high-fitness yield across species ---
    H = [(f, s) for f, s, _ in scored if f >= tau]
    n_high = len(H)
    H_sids = [s for _, s in H if s is not None]
    n_high_unass = sum(1 for _, s in H if s is None)
    n_species_with_high = len(set(H_sids))
    if H_sids:
        from collections import Counter

        share_H_top_sp = max(Counter(H_sids).values()) / max(n_high, 1)
    else:
        share_H_top_sp = float("nan")

    # species elite fitness (max in species); not a QD-score sum
    q_max = [max(f for f, _ in members) for members in by_sp.values()] if by_sp else []
    h_counts = [
        sum(1 for f, _ in members if f >= tau) for members in by_sp.values()
    ] if by_sp else []

    return {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "n_unique": len(scored),
        "n_mature": n_mature,
        "n_assigned": n_assigned,
        "largest": largest,
        "d_assigned": d_assigned,
        "within_dE": within,
        "between_dE": between,
        "sep_ratio": (between / within) if (within == within and within > 1e-12 and between == between) else float("nan"),
        "top10_mean": top10_mean,
        "K10": k10,
        "U10": u10,
        "share_top10_maxsp": share_top_sp,
        "n_high": n_high,
        "n_high_unass": n_high_unass,
        "n_species_with_high": n_species_with_high,
        "share_H_maxsp": share_H_top_sp,
        "median_q_max": float(np.median(q_max)) if q_max else float("nan"),
        "frac_species_with_high": (n_species_with_high / n_mature) if n_mature else float("nan"),
        "mean_h_per_species": float(np.mean(h_counts)) if h_counts else float("nan"),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _med(vals: Sequence[float]) -> str:
    v = [float(x) for x in vals if x == x and math.isfinite(float(x))]
    return "---" if not v else f"{np.median(v):.3f}"


def _med_i(vals: Sequence[float]) -> str:
    v = [float(x) for x in vals if x == x and math.isfinite(float(x))]
    return "---" if not v else f"{np.median(v):.1f}"


def write_latex(path: Path, rows: List[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by rq3_species_audit.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RQ3 species characteristics at the final snapshot (median over "
        r"$n{=}10$ runs). Within/between use MiniLM $d_E$ on assigned mature members "
        r"and species centroids. $K_{10}$: distinct mature species among the ten "
        r"fittest unique individuals; $U_{10}$: how many of those ten are unassigned; "
        r"$\pi_{10}$: share of the top ten from a single mature species. "
        r"These are algorithm-dependent internal diagnostics, not a QD-score and not "
        r"validated behavioral distinctness.}",
        r"\label{tab:rq3-species}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccccccc@{}}",
        r"\toprule",
        r"Dissim. & Method & $n_{\mathrm{mat}}$ & Within & Between & $K_{10}$ & $U_{10}$ & $\pi_{10}$ \\",
        r"\midrule",
    ]
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_med_i([r['n_mature'] for r in sub])} & "
            f"{_med([r['within_dE'] for r in sub])} & "
            f"{_med([r['between_dE'] for r in sub])} & "
            f"{_med_i([r['K10'] for r in sub])} & "
            f"{_med_i([r['U10'] for r in sub])} & "
            f"{_med([r['share_top10_maxsp'] for r in sub])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_contribution(rows: List[Dict[str, Any]], out: Path) -> None:
    """R3-F: how top-10 elites are distributed across species."""
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
            sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
            col = colors[clustering]
            kvals = [float(r["K10"]) for r in sub]
            pvals = [float(r["share_top10_maxsp"]) for r in sub]
            ax0.scatter(np.full(len(kvals), xpos), kvals, color=col, alpha=0.7, s=28, zorder=3)
            ax1.scatter(np.full(len(pvals), xpos), pvals, color=col, alpha=0.7, s=28, zorder=3)
            if kvals:
                ax0.hlines(np.median(kvals), xpos - 0.28, xpos + 0.28, colors="black", lw=1.4, zorder=4)
            if pvals:
                ax1.hlines(np.median(pvals), xpos - 0.28, xpos + 0.28, colors="black", lw=1.4, zorder=4)
    ax0.set_xticks(ticks)
    ax0.set_xticklabels(labs, fontsize=7.5)
    ax1.set_xticks(ticks)
    ax1.set_xticklabels(labs, fontsize=7.5)
    ax0.set_ylabel(r"Distinct species in top-10 ($K_{10}$)")
    ax1.set_ylabel(r"Share of top-10 from one species ($\pi_{10}$)")
    ax0.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax1.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax0.set_ylim(0, 10.5)
    ax1.set_ylim(0, 1.05)
    ax0.grid(True, axis="y", alpha=0.25)
    ax1.grid(True, axis="y", alpha=0.25)
    out.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (out, NOTES_FIG / out.name, out.with_suffix(".png"), NOTES_FIG / out.with_suffix(".png").name):
        fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        for seed in range(10):
            print(f"sp {distance} {clustering} seed{seed}", flush=True)
            row = analyze_run(distance, clustering, seed)
            if row:
                rows.append(row)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "rq3_species_runs.csv", rows)
    write_latex(NOTES / "rq3_species_tables.tex", rows)
    plot_contribution(rows, FIG / "fig_rq3_species_contrib.pdf")
    # quick print
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        print(
            DIST_TEX[distance],
            CLUS_TEX[clustering],
            f"K10={np.median([r['K10'] for r in sub]):.1f}",
            f"pi10={np.median([r['share_top10_maxsp'] for r in sub]):.2f}",
            f"within={np.nanmedian([r['within_dE'] for r in sub]):.3f}",
            f"between={np.nanmedian([r['between_dE'] for r in sub]):.3f}",
        )
    print(f"wrote {len(rows)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
