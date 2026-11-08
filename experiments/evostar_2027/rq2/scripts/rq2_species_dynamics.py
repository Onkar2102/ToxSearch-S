#!/usr/bin/env python3
"""RQ2 Improvement B: online species dynamics at fixed unique-individual budgets.

Reads EvolutionTracker.json (+ species_snapshots.jsonl for LF unassigned/archive
membership) from live cells under data/outputs/evostar2027.

Budgets: 250, 500, 1000, 1500, 2000.

Metrics (per run, first generation with N >= budget):
  n_mature          number of mature species
  d_assigned        |S_max| / #individuals in mature species
  largest_frac      |S_max| / N
  unassigned_frac   singletons (LF label 0) or IncDBSCAN noise / N
  n_eligible        active (non-frozen) mature species
  archived_frac     LF archive size / N (0 for IncDBSCAN)
  frozen_species_frac  frozen mature species / max(n_mature, 1)

Example::

    PYTHONPATH=src:. venv/bin/python experiments/evostar_2027/rq2/scripts/rq2_species_dynamics.py
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
LIVE = ROOT / "data" / "outputs" / "evostar2027"
OUT = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
FIG = ROOT / "experiments" / "evostar_2027" / "rq2" / "figures"
NOTES = ROOT / "notes" / "rough_draft"
PAPER_FIG = NOTES / "figures"

BUDGETS = [250, 500, 1000, 1500, 2000]

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}

CELLS: List[Tuple[str, str, str, str, str]] = [
    ("embedding", "leader_follower", r"LF $d_E$", "#0F4D92", "-"),
    ("embedding", "dbscan", r"Inc $d_E$", "#0F4D92", "--"),
    ("nli_entail", "leader_follower", r"LF $d_{\neg E}$", "#B64342", "-"),
    ("nli_entail", "dbscan", r"Inc $d_{\neg E}$", "#B64342", "--"),
]

DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.5,
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
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


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _snapshot_by_n(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """Map generation -> snapshot row."""
    path = run_dir / "species_snapshots.jsonl"
    out: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["generation"])] = row
    return out


def _membership_counts(snap: Optional[Dict[str, Any]]) -> Tuple[int, int, int, int, int]:
    """Return (N, n_unassigned, n_archived, n_mature_members, n_eligible_from_leaders).

    LF: label 0 = singleton pool; label -1 = archive; other labels with leaders = mature.
    Inc: label 0 = noise; leaders map mature (+ frozen) species.
    """
    if not snap:
        return 0, 0, 0, 0, 0
    mem = snap.get("membership") or {}
    leaders = snap.get("leaders") or {}
    counts = Counter(int(v) for v in mem.values())
    n = sum(counts.values())
    leader_ids = {int(k) for k in leaders}
    n_mature = sum(counts[i] for i in leader_ids if i in counts)
    n_eligible = sum(
        1 for meta in leaders.values() if str(meta.get("species_state", "")).lower() != "frozen"
    )
    # Heuristic: if any leader exists, treat 0 as unassigned/noise and -1 as archive (LF).
    n_unassigned = counts.get(0, 0)
    n_archived = counts.get(-1, 0)
    return n, n_unassigned, n_archived, n_mature, n_eligible


def _metrics_at_generation(
    gen: Dict[str, Any],
    snap: Optional[Dict[str, Any]],
    clustering: str,
) -> Dict[str, float]:
    sp = gen.get("speciation") or {}
    n = int(gen.get("total_population") or sp.get("total_population") or 0)
    if clustering == "dbscan":
        n = int(sp.get("density_size") or gen.get("density_size") or n)

    n_mature = float(sp.get("species_count") or 0)
    largest = float(sp.get("largest_species_size") or 0)
    n_active = float(sp.get("active_species_count") or 0)
    n_frozen = float(sp.get("frozen_species_count") or 0)

    n_snap, n_unassigned, n_archived, n_mature_mem, n_elig_snap = _membership_counts(snap)

    if clustering == "dbscan":
        clustered = float(sp.get("clustered_count") or n_mature_mem or 0)
        unassigned = float(sp.get("noise_count") or sp.get("singleton_species_count") or n_unassigned)
        archived = 0.0
        if n_active <= 0 and n_elig_snap > 0:
            n_active = float(n_elig_snap)
    else:
        # LF: prefer membership for unassigned/archive; ET archive_size as fallback.
        unassigned = float(n_unassigned if n_snap else 0)
        archived = float(
            n_archived
            if n_snap
            else (sp.get("archive_size") or gen.get("archived_count") or 0)
        )
        clustered = float(n_mature_mem) if n_mature_mem else max(n - unassigned - archived, 0.0)
        if n_active <= 0 and n_elig_snap > 0:
            n_active = float(n_elig_snap)

    d_assigned = (largest / clustered) if clustered > 0 else float("nan")
    largest_frac = (largest / n) if n > 0 else float("nan")
    unassigned_frac = (unassigned / n) if n > 0 else float("nan")
    archived_frac = (archived / n) if n > 0 else float("nan")
    frozen_species_frac = (n_frozen / n_mature) if n_mature > 0 else float("nan")

    return {
        "n": float(n),
        "n_mature": n_mature,
        "n_eligible": n_active,
        "largest": largest,
        "d_assigned": d_assigned,
        "largest_frac": largest_frac,
        "unassigned_frac": unassigned_frac,
        "archived_frac": archived_frac,
        "frozen_species_frac": frozen_species_frac,
        "generation": float(gen.get("generation_number") or 0),
    }


def extract_run(distance: str, clustering: str, seed: int) -> Optional[Dict[int, Dict[str, float]]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    et = _load_json(run_dir / "EvolutionTracker.json")
    if not et or str(et.get("status")) != "complete":
        return None
    snaps = _snapshot_by_n(run_dir)
    gens = et.get("generations") or []
    by_budget: Dict[int, Dict[str, float]] = {}
    for B in BUDGETS:
        hit = None
        for g in gens:
            sp = g.get("speciation") or {}
            n = int(g.get("total_population") or sp.get("total_population") or 0)
            if clustering == "dbscan":
                n = int(sp.get("density_size") or g.get("density_size") or n)
            if n >= B:
                hit = g
                break
        if hit is None:
            continue
        gi = int(hit.get("generation_number") or 0)
        by_budget[B] = _metrics_at_generation(hit, snaps.get(gi), clustering)
    return by_budget if by_budget else None


def collect_all() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for distance, clustering, label, color, ls in CELLS:
        for seed in range(10):
            by_b = extract_run(distance, clustering, seed)
            if not by_b:
                continue
            for B, m in by_b.items():
                rows.append(
                    {
                        "distance": distance,
                        "clustering": clustering,
                        "label": label,
                        "color": color,
                        "ls": ls,
                        "seed": seed,
                        "budget": B,
                        **m,
                    }
                )
    return rows


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "distance",
        "clustering",
        "seed",
        "budget",
        "generation",
        "n",
        "n_mature",
        "n_eligible",
        "largest",
        "d_assigned",
        "largest_frac",
        "unassigned_frac",
        "archived_frac",
        "frozen_species_frac",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _cell_median(rows: List[Dict[str, Any]], distance: str, clustering: str, B: int, key: str) -> Tuple[float, int]:
    vals = [
        r[key]
        for r in rows
        if r["distance"] == distance
        and r["clustering"] == clustering
        and r["budget"] == B
        and r[key] == r[key]  # not NaN
    ]
    if not vals:
        return float("nan"), 0
    return float(np.median(vals)), len(vals)


def write_latex_table(rows: List[Dict[str, Any]], path: Path) -> None:
    """Median dynamics at each budget; one block per metric (compact)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    lines.append("% Auto-generated by rq2_species_dynamics.py — do not hand-edit.")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(
        r"\caption{Online species dynamics at unique-individual budgets "
        r"$b\in\{250,500,1000,1500,2000\}$ (median over currently completed runs). "
        r"$n_{\mathrm{mat}}$: mature species. $D_{\mathrm{assigned}}$: largest mature "
        r"species over assigned mass. Unass.: singleton pool (LF) or noise (IncDBSCAN). "
        r"$n_{\mathrm{elig}}$: active non-frozen mature species. Arch./frz.: archive "
        r"fraction (LF) or frozen-species fraction (IncDBSCAN). Interim until $n{=}10$ "
        r"per cell.}"
    )
    lines.append(r"\label{tab:rq2-dynamics}")
    lines.append(r"\scriptsize")
    lines.append(r"\setlength{\tabcolsep}{2.6pt}")
    lines.append(r"\begin{tabular}{@{}llrrrrrr@{}}")
    lines.append(r"\toprule")
    lines.append(
        r"Dist. & Method & $b$ & $n_{\mathrm{mat}}$ & $D_{\mathrm{assigned}}$ & "
        r"Unass. & $n_{\mathrm{elig}}$ & Arch./frz. \\"
    )
    lines.append(r"\midrule")
    for distance, clustering, *_ in CELLS:
        first = True
        for B in BUDGETS:
            nm, n = _cell_median(rows, distance, clustering, B, "n_mature")
            da, _ = _cell_median(rows, distance, clustering, B, "d_assigned")
            ua, _ = _cell_median(rows, distance, clustering, B, "unassigned_frac")
            ne, _ = _cell_median(rows, distance, clustering, B, "n_eligible")
            if clustering == "dbscan":
                af, _ = _cell_median(rows, distance, clustering, B, "frozen_species_frac")
            else:
                af, _ = _cell_median(rows, distance, clustering, B, "archived_frac")
            if n == 0:
                continue
            dtex = DIST_TEX[distance] if first else ""
            ctex = CLUS_TEX[clustering] if first else ""
            first = False
            nm_s = f"{nm:.0f}" if abs(nm - round(nm)) < 1e-9 else f"{nm:.1f}"
            ne_s = f"{ne:.0f}" if abs(ne - round(ne)) < 1e-9 else f"{ne:.1f}"
            lines.append(
                f"{dtex} & {ctex} & ${B}$ & ${nm_s}$ & ${da:.2f}$ & "
                f"${100*ua:.0f}\\%$ & ${ne_s}$ & ${100*af:.0f}\\%$ \\\\"
            )
        lines.append(r"\midrule")
    if lines[-1] == r"\midrule":
        lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_dynamics(rows: List[Dict[str, Any]]) -> None:
    # Omit n_mature here: it is already the main-text Fig. rq2 panel (b).
    panels = [
        ("d_assigned", r"$D_{\mathrm{assigned}}$", True),
        ("unassigned_frac", r"Unassigned fraction", True),
        ("n_eligible", r"Eligible species $n_{\mathrm{elig}}$", False),
        ("mgmt_frac", r"Archived / frozen fraction", True),
    ]
    # synthesize mgmt_frac
    for r in rows:
        if r["clustering"] == "dbscan":
            r["mgmt_frac"] = r["frozen_species_frac"]
        else:
            r["mgmt_frac"] = r["archived_frac"]

    fig, axes = plt.subplots(2, 2, figsize=(6.4, 3.8), sharex=True)
    axes_flat = list(axes.ravel())

    for ax, (key, ylabel, is_frac) in zip(axes_flat, panels):
        for distance, clustering, label, color, ls in CELLS:
            ys = []
            ns = []
            for B in BUDGETS:
                med, n = _cell_median(rows, distance, clustering, B, key)
                ys.append(med)
                ns.append(n)
            if max(ns) == 0:
                continue
            y = np.array(ys, dtype=float)
            ax.plot(BUDGETS, y, color=color, ls=ls, lw=1.6, marker="o", ms=3.2, label=f"{label} ($n$={ns[-1]})")
        ax.set_ylabel(ylabel)
        ax.set_xticks(BUDGETS)
        if is_frac:
            ax.set_ylim(0.0, 1.05)
        ax.tick_params(length=3, width=0.7)

    axes_flat[2].set_xlabel(r"Unique evaluated individuals")
    axes_flat[3].set_xlabel(r"Unique evaluated individuals")
    handles, labels = axes_flat[0].get_legend_handles_labels()
    axes_flat[0].legend(handles, labels, loc="best", fontsize=6.5, handlelength=1.6)

    fig.tight_layout(w_pad=0.8, h_pad=0.9)
    FIG.mkdir(parents=True, exist_ok=True)
    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (FIG, PAPER_FIG):
        fig.savefig(dest / "fig_rq2_dynamics.pdf")
        fig.savefig(dest / "fig_rq2_dynamics.png")
    plt.close(fig)
    print(f"wrote {PAPER_FIG / 'fig_rq2_dynamics.pdf'}")


def main() -> None:
    rows = collect_all()
    print(f"rows={len(rows)}  runs≈{len(rows)//len(BUDGETS)}")
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(rows, OUT / "rq2_species_dynamics.csv")
    write_latex_table(rows, NOTES / "rq2_dynamics_table.tex")
    print(f"wrote {OUT / 'rq2_species_dynamics.csv'}")
    print(f"wrote {NOTES / 'rq2_dynamics_table.tex'}")
    plot_dynamics(rows)
    # quick summary at b=2000
    print("\n=== medians at b=2000 ===")
    for distance, clustering, label, *_ in CELLS:
        parts = []
        for key in ("n_mature", "d_assigned", "unassigned_frac", "n_eligible"):
            med, n = _cell_median(rows, distance, clustering, 2000, key)
            parts.append(f"{key}={med:.3f}")
        print(f"{label}: n={n}  " + "  ".join(parts))


if __name__ == "__main__":
    main()
