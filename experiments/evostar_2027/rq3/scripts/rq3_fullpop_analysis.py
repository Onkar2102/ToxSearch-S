#!/usr/bin/env python3
"""RQ3 full-population species characterization (Experiment B).

Primary analyses over *all* uniquely evaluated individuals in each of the
frozen 40 runs (not top-k elites):

1. Population structure: mature count, effective species count N_eff,
   unassigned / archived fractions.
2. Size-adjusted fitness organization: high-fitness counts h_s(τ) versus a
   species-size-preserving fitness-permutation null; continuous between /
   within fitness variation; effective productive-species diversity N_eff^H.
3. Budget trajectories for (2) at unique-evaluation checkpoints.

Independent-representation coherence (mpnet / TF--IDF) is loaded from the
existing rq3_coherence_independent outputs and joined into the main table.

Example::

    MPLBACKEND=Agg PYTHONPATH=src:. venv/bin/python \\
      experiments/evostar_2027/rq3/scripts/rq3_fullpop_analysis.py
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
CELLS = [
    ("embedding", "leader_follower", r"LF $d_E$", "#0F4D92", "-"),
    ("embedding", "dbscan", r"Inc $d_E$", "#0F4D92", "--"),
    ("nli_entail", "leader_follower", r"LF $d_{\neg E}$", "#B64342", "-"),
    ("nli_entail", "dbscan", r"Inc $d_{\neg E}$", "#B64342", "--"),
]

BUDGETS = [250, 500, 1000, 1500, 2000]
TAUS = (0.50, 0.70, 0.80, 0.90)
PRIMARY_TAU = 0.80
N_PERM = 2000
PERM_SEED = 2027

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


def load_tracker(run_dir: Path) -> Dict[str, Any]:
    return json.loads((run_dir / "EvolutionTracker.json").read_text(encoding="utf-8"))


def load_snapshots(run_dir: Path) -> Dict[int, Dict[str, Any]]:
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


def categorize_label(lab: Any, leaders: set) -> str:
    try:
        i = int(lab)
    except Exception:
        return "unassigned"
    if i == -1:
        return "archived"
    if i == 0 or i not in leaders:
        # IncDBSCAN noise is 0; LF singletons are 0; non-leader positive labels
        # should not occur if leaders are complete, but treat as unassigned.
        if i > 0 and i not in leaders:
            return "unassigned"
        return "unassigned" if i <= 0 else "mature"
    return "mature"


def effective_n(counts: Sequence[float]) -> float:
    arr = np.asarray([c for c in counts if c > 0], dtype=float)
    if arr.size == 0:
        return 0.0
    p = arr / arr.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def structure_from_membership(
    membership: Dict[str, Any],
    leaders: Dict[str, Any],
) -> Dict[str, float]:
    leader_ids = {int(k) for k in leaders}
    sizes: Dict[int, int] = defaultdict(int)
    n_unass = n_arch = n_mat_mem = 0
    for lab in membership.values():
        try:
            i = int(lab)
        except Exception:
            n_unass += 1
            continue
        if i == -1:
            n_arch += 1
        elif i == 0 or i not in leader_ids:
            n_unass += 1
        else:
            sizes[i] += 1
            n_mat_mem += 1
    n = len(membership)
    n_mat = float(len(sizes))
    return {
        "n": float(n),
        "n_mature": n_mat,
        "n_mature_members": float(n_mat_mem),
        "n_unassigned": float(n_unass),
        "n_archived": float(n_arch),
        "unassigned_frac": (n_unass / n) if n else float("nan"),
        "archived_frac": (n_arch / n) if n else float("nan"),
        "mature_frac": (n_mat_mem / n) if n else float("nan"),
        "N_eff": effective_n(list(sizes.values())),
        "largest_frac_assigned": (
            (max(sizes.values()) / n_mat_mem) if n_mat_mem else float("nan")
        ),
    }


def fitness_organization(
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    *,
    tau: float,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    """Size-preserving fitness null among mature-labelled individuals.

    Unassigned / archived keep their observed fitness (not permuted into niches).
    """
    fit = np.asarray(fitness, dtype=float)
    labs = list(labels)
    mature_idx = [i for i, lab in enumerate(labs) if lab is not None]
    if not mature_idx:
        return {
            "n_high": 0.0,
            "n_high_assigned": 0.0,
            "share_H": float("nan"),
            "N_eff_H": 0.0,
            "frac_species_with_high": float("nan"),
            "frac_enriched": float("nan"),
            "max_enrich_z": float("nan"),
            "share_H_null": float("nan"),
            "N_eff_H_null": float("nan"),
            "share_H_z": float("nan"),
            "N_eff_H_z": float("nan"),
            "eta_sq": float("nan"),
            "eta_sq_null": float("nan"),
            "eta_sq_z": float("nan"),
            "n_mature": 0.0,
        }

    def _stats(labs_local: Sequence[Optional[int]], fit_local: np.ndarray) -> Dict[str, float]:
        by: Dict[int, List[float]] = defaultdict(list)
        for i in mature_idx:
            by[int(labs_local[i])].append(float(fit_local[i]))
        n_mat = len(by)
        h = {s: sum(1 for f in vals if f >= tau) for s, vals in by.items()}
        H_ass = float(sum(h.values()))
        sizes = {s: len(vals) for s, vals in by.items()}
        N_ass = float(sum(sizes.values()))
        share = (max(h.values()) / H_ass) if H_ass > 0 else float("nan")
        neff_h = effective_n([h[s] for s in h if h[s] > 0])
        frac_high = (sum(1 for s in h if h[s] > 0) / n_mat) if n_mat else float("nan")
        # enrichment vs size expectation
        enriched = 0
        max_z = float("nan")
        if H_ass > 0 and N_ass > 0:
            zs = []
            for s, n_s in sizes.items():
                exp = n_s / N_ass * H_ass
                # Poisson-binomial variance under hypergeometric-like draw is
                # approx; use standardized residual (obs-exp)/sqrt(exp) when exp>0
                if exp > 0:
                    z = (h[s] - exp) / math.sqrt(exp)
                    zs.append(z)
                    if h[s] > exp:
                        enriched += 1
            max_z = float(max(zs)) if zs else float("nan")
        # continuous η² = SS_between / SS_total on mature members
        all_f = np.asarray([f for vals in by.values() for f in vals], dtype=float)
        if all_f.size >= 2 and np.unique(all_f).size >= 2:
            grand = float(all_f.mean())
            ss_tot = float(((all_f - grand) ** 2).sum())
            ss_bet = 0.0
            for vals in by.values():
                arr = np.asarray(vals, dtype=float)
                ss_bet += arr.size * float((arr.mean() - grand) ** 2)
            eta = (ss_bet / ss_tot) if ss_tot > 0 else float("nan")
        else:
            eta = float("nan")
        return {
            "n_high_assigned": H_ass,
            "share_H": share,
            "N_eff_H": neff_h,
            "frac_species_with_high": frac_high,
            "frac_enriched": (enriched / n_mat) if n_mat else float("nan"),
            "max_enrich_z": max_z,
            "eta_sq": eta,
            "n_mature": float(n_mat),
        }

    obs = _stats(labs, fit)
    n_high_all = float(sum(1 for f in fit if f >= tau))
    rng = np.random.default_rng(seed)
    null_share = np.empty(n_perm)
    null_neff = np.empty(n_perm)
    null_eta = np.empty(n_perm)
    idx = np.asarray(mature_idx, dtype=int)
    for b in range(n_perm):
        shuffled = fit.copy()
        shuffled[idx] = fit[idx][rng.permutation(idx.size)]
        o = _stats(labs, shuffled)
        null_share[b] = o["share_H"]
        null_neff[b] = o["N_eff_H"]
        null_eta[b] = o["eta_sq"]

    def zscore(obs_v: float, null: np.ndarray) -> float:
        finite = null[np.isfinite(null)]
        if finite.size < 2 or not math.isfinite(obs_v):
            return float("nan")
        sd = float(finite.std(ddof=1))
        return (obs_v - float(finite.mean())) / sd if sd > 0 else float("nan")

    return {
        "n_high": n_high_all,
        **obs,
        "share_H_null": float(np.nanmean(null_share)),
        "N_eff_H_null": float(np.nanmean(null_neff)),
        "eta_sq_null": float(np.nanmean(null_eta)),
        "share_H_z": zscore(obs["share_H"], null_share),
        "N_eff_H_z": zscore(obs["N_eff_H"], null_neff),
        "eta_sq_z": zscore(obs["eta_sq"], null_eta),
    }


def genome_tables(
    genomes: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    fitness: Dict[str, float] = {}
    for g in genomes:
        gid = str(g.get("id"))
        by_id[gid] = g
        f = illicit_of(g)
        if f is not None:
            fitness[gid] = float(f)
    return by_id, fitness


def labels_from_snapshot(
    snap: Dict[str, Any],
) -> Tuple[Dict[str, Optional[int]], set]:
    leaders = {int(k) for k in (snap.get("leaders") or {})}
    mem = snap.get("membership") or {}
    out: Dict[str, Optional[int]] = {}
    for gid, lab in mem.items():
        try:
            i = int(lab)
        except Exception:
            out[str(gid)] = None
            continue
        if i in leaders:
            out[str(gid)] = i
        else:
            out[str(gid)] = None  # unassigned / archived / noise
    return out, leaders


def final_population_metrics(
    distance: str,
    clustering: str,
    seed: int,
    *,
    n_perm: int,
) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    snaps = load_snapshots(run_dir)
    if not snaps:
        return None
    last_gen = max(snaps)
    snap = snaps[last_gen]
    struct = structure_from_membership(snap.get("membership") or {}, snap.get("leaders") or {})
    lab_map, _ = labels_from_snapshot(snap)
    _, fit_map = genome_tables(genomes)
    # Align on membership IDs that have fitness
    ids = [gid for gid in lab_map if gid in fit_map]
    labels = [lab_map[gid] for gid in ids]
    fitness = [fit_map[gid] for gid in ids]
    # category fractions already from snapshot; fitness org on aligned set
    org = fitness_organization(
        labels,
        fitness,
        tau=PRIMARY_TAU,
        n_perm=n_perm,
        seed=PERM_SEED + seed * 17 + hash((distance, clustering)) % 10000,
    )
    # tau sensitivity: share_H and N_eff_H only (lighter)
    tau_stats: Dict[str, float] = {}
    for tau in TAUS:
        o = fitness_organization(
            labels,
            fitness,
            tau=tau,
            n_perm=max(200, n_perm // 5),
            seed=PERM_SEED + int(tau * 100) + seed,
        )
        tau_stats[f"share_H_t{tau:.2f}"] = o["share_H"]
        tau_stats[f"share_H_z_t{tau:.2f}"] = o["share_H_z"]
        tau_stats[f"N_eff_H_t{tau:.2f}"] = o["N_eff_H"]
        tau_stats[f"N_eff_H_z_t{tau:.2f}"] = o["N_eff_H_z"]

    # archived/unassigned among LF from structure; Inc archived=0
    if clustering == "dbscan":
        struct["archived_frac"] = 0.0

    return {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        **{f"struct_{k}": v for k, v in struct.items()},
        **{f"fit_{k}": v for k, v in org.items()},
        **tau_stats,
    }


def budget_population_n(gen: Dict[str, Any], clustering: str) -> int:
    sp = gen.get("speciation") or {}
    n = int(gen.get("total_population") or sp.get("total_population") or 0)
    if clustering == "dbscan":
        n = int(sp.get("density_size") or gen.get("density_size") or n)
    return n


def trajectory_metrics(
    distance: str,
    clustering: str,
    seed: int,
    *,
    n_perm: int,
) -> List[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return []
    tracker = load_tracker(run_dir)
    snaps = load_snapshots(run_dir)
    genomes = load_genomes(run_dir, clustering)
    _, fit_map = genome_tables(genomes)
    gens = tracker.get("generations") or []
    rows: List[Dict[str, Any]] = []
    for B in BUDGETS:
        hit = None
        for g in gens:
            if budget_population_n(g, clustering) >= B:
                hit = g
                break
        if hit is None:
            continue
        gi = int(hit.get("generation_number") or 0)
        snap = snaps.get(gi)
        if not snap:
            # nearest snapshot at or before gi
            cands = [g for g in snaps if g <= gi]
            snap = snaps[max(cands)] if cands else None
        if not snap:
            continue
        struct = structure_from_membership(snap.get("membership") or {}, snap.get("leaders") or {})
        if clustering == "dbscan":
            struct["archived_frac"] = 0.0
        lab_map, _ = labels_from_snapshot(snap)
        ids = [gid for gid in lab_map if gid in fit_map]
        labels = [lab_map[gid] for gid in ids]
        fitness = [fit_map[gid] for gid in ids]
        org = fitness_organization(
            labels,
            fitness,
            tau=PRIMARY_TAU,
            n_perm=n_perm,
            seed=PERM_SEED + B + seed * 31 + hash((distance, clustering)) % 10000,
        )
        rows.append(
            {
                "distance": distance,
                "clustering": clustering,
                "seed": seed,
                "budget": B,
                **{f"struct_{k}": v for k, v in struct.items()},
                **{f"fit_{k}": v for k, v in org.items()},
            }
        )
    return rows


def load_coherence_join() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    path = REPORT / "rq3_coherence_independent_runs.csv"
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    if not path.is_file():
        return out
    with path.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["distance"], r["clustering"], int(r["seed"]))
            out[key] = {
                "mpnet_z_sep": float(r.get("mpnet_z_sep") or "nan"),
                "mpnet_within": float(r.get("mpnet_obs_within") or "nan"),
                "tfidf_z_sep": float(r.get("tfidf_z_sep") or "nan"),
                "tfidf_within": float(r.get("tfidf_obs_within") or "nan"),
            }
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
    return f"{v:.{nd}f}"


def write_main_table(
    path: Path,
    rows: List[Dict[str, Any]],
    coh: Dict[Tuple[str, str, int], Dict[str, float]],
) -> None:
    lines = [
        "% Auto-generated by rq3_fullpop_analysis.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Full-population species characterization (median over $n{=}10$ "
        r"runs). $n_{\mathrm{mat}}$: mature species; $N_{\mathrm{eff}}$: effective "
        r"species count among mature members; Unass./Arch.: unassigned (LF "
        r"singletons or IncDBSCAN noise) and archived fractions. "
        r"$z_{\mathrm{sep}}$: independent mpnet separation versus a size-preserving "
        r"label null. At $\tau{=}0.80$, $\mathrm{share}_H$: fraction of "
        r"high-fitness assigned individuals in the largest contributing species; "
        r"$N_{\mathrm{eff}}^{H}$: effective number of high-fitness species; "
        r"both compared with size-preserving fitness-permutation nulls "
        r"($B{=}2{,}000$).}",
        r"\label{tab:rq3-fullpop}",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        r"Dissim. & Method & $n_{\mathrm{mat}}$ & $N_{\mathrm{eff}}$ & Unass. & "
        r"Arch. & $z_{\mathrm{sep}}$ & $\mathrm{share}_H$ & "
        r"$\mathrm{share}_H^{\mathrm{null}}$ & $N_{\mathrm{eff}}^{H}$ \\",
        r"\midrule",
    ]
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        zsep = []
        for r in sub:
            c = coh.get((distance, clustering, int(r["seed"])))
            if c:
                zsep.append(c["mpnet_z_sep"])
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_fmt(np.median([r['struct_n_mature'] for r in sub]), 0)} & "
            f"{_fmt(np.median([r['struct_N_eff'] for r in sub]), 1)} & "
            f"{_fmt(100 * np.median([r['struct_unassigned_frac'] for r in sub]), 0)}"
            r"\% & "
            f"{_fmt(100 * np.median([r['struct_archived_frac'] for r in sub]), 0)}"
            r"\% & "
            f"{_fmt(np.nanmedian(zsep) if zsep else float('nan'), 1)} & "
            f"{_fmt(np.nanmedian([r['fit_share_H'] for r in sub]), 2)} & "
            f"{_fmt(np.nanmedian([r['fit_share_H_null'] for r in sub]), 2)} & "
            f"{_fmt(np.nanmedian([r['fit_N_eff_H'] for r in sub]), 1)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_productivity(traj: List[Dict[str, Any]], out: Path) -> None:
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(6.8, 2.55), layout="constrained")
    for distance, clustering, label, color, ls in CELLS:
        xs, share, share_null, neff_h, neff_null = [], [], [], [], []
        for B in BUDGETS:
            sub = [
                r
                for r in traj
                if r["distance"] == distance
                and r["clustering"] == clustering
                and int(r["budget"]) == B
            ]
            if not sub:
                continue
            xs.append(B)
            share.append(float(np.nanmedian([r["fit_share_H"] for r in sub])))
            share_null.append(float(np.nanmedian([r["fit_share_H_null"] for r in sub])))
            neff_h.append(float(np.nanmedian([r["fit_N_eff_H"] for r in sub])))
            neff_null.append(float(np.nanmedian([r["fit_N_eff_H_null"] for r in sub])))
        if not xs:
            continue
        ax0.plot(xs, share, color=color, ls=ls, lw=1.6, marker="o", ms=3.2, label=label)
        ax0.plot(xs, share_null, color=color, ls=ls, lw=1.0, alpha=0.35)
        ax1.plot(xs, neff_h, color=color, ls=ls, lw=1.6, marker="o", ms=3.2, label=label)
        ax1.plot(xs, neff_null, color=color, ls=ls, lw=1.0, alpha=0.35)

    ax0.set_xlabel("Unique evaluated individuals")
    ax1.set_xlabel("Unique evaluated individuals")
    ax0.set_ylabel(r"High-fitness share $\mathrm{share}_H$")
    ax1.set_ylabel(r"Effective productive species $N_{\mathrm{eff}}^{H}$")
    ax0.set_title("(a)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax1.set_title("(b)", loc="left", fontsize=9, fontweight="bold", pad=2)
    ax0.set_ylim(0, 1.05)
    ax0.set_xticks(BUDGETS)
    ax1.set_xticks(BUDGETS)
    ax0.grid(True, axis="y", alpha=0.25)
    ax1.grid(True, axis="y", alpha=0.25)
    ax0.legend(fontsize=6.2, loc="best", frameon=False)
    # faint null lines explained in caption
    out.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FIG.mkdir(parents=True, exist_ok=True)
    for dest in (out, NOTES_FIG / out.name, out.with_suffix(".png"), NOTES_FIG / out.with_suffix(".png").name):
        fig.savefig(dest)
    plt.close(fig)
    print(f"wrote {out}", flush=True)


def main() -> int:
    p = argparse.ArgumentParser(description="RQ3 full-population characterization")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    final_rows: List[Dict[str, Any]] = []
    traj_rows: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        for seed in seeds:
            print(f"fullpop final {distance} {clustering} seed{seed}", flush=True)
            row = final_population_metrics(
                distance, clustering, seed, n_perm=args.n_perm
            )
            if row:
                final_rows.append(row)
            print(f"fullpop traj  {distance} {clustering} seed{seed}", flush=True)
            traj_rows.extend(
                trajectory_metrics(distance, clustering, seed, n_perm=max(500, args.n_perm // 2))
            )

    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "rq3_fullpop_runs.csv", final_rows)
    write_csv(REPORT / "rq3_fullpop_trajectories.csv", traj_rows)

    coh = load_coherence_join()
    write_main_table(NOTES / "rq3_fullpop_table.tex", final_rows, coh)
    plot_productivity(traj_rows, FIG / "fig_rq3_productivity.pdf")

    meta = {
        "n_perm_final": args.n_perm,
        "primary_tau": PRIMARY_TAU,
        "taus": list(TAUS),
        "budgets": BUDGETS,
        "n_final_runs": len(final_rows),
        "n_traj_rows": len(traj_rows),
        "note": (
            "Species IDs are run-local and never pooled across executions. "
            "Fitness null permutes among mature-labelled individuals only."
        ),
    }
    (REPORT / "rq3_fullpop_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for distance, clustering in LIVE_DIR:
        sub = [r for r in final_rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        print(
            DIST_TEX[distance],
            CLUS_TEX[clustering],
            f"nmat={np.median([r['struct_n_mature'] for r in sub]):.0f}",
            f"Neff={np.median([r['struct_N_eff'] for r in sub]):.1f}",
            f"shareH={np.nanmedian([r['fit_share_H'] for r in sub]):.2f}",
            f"null={np.nanmedian([r['fit_share_H_null'] for r in sub]):.2f}",
            f"NeffH={np.nanmedian([r['fit_N_eff_H'] for r in sub]):.1f}",
            f"eta_z={np.nanmedian([r['fit_eta_sq_z'] for r in sub]):.2f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
