#!/usr/bin/env python3
"""RQ3 offline audit: species stats vs high-fitness descriptor coverage.

Uses existing 40 RQ2 runs only. Writes rarefaction, association, and purity
tables under experiments/evostar_2027/rq3/reports/ plus LaTeX fragments under
notes/rough_draft/. Does not print prompt/response text.

  PYTHONPATH=src:. MPLBACKEND=Agg venv/bin/python \\
    experiments/evostar_2027/rq3/scripts/rq3_diversity_audit.py
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
sys.path.insert(0, str(ROOT / "notes" / "rough_draft"))

from generate_rq2_coverage import (  # noqa: E402
    LIVE_DIR,
    _assign,
    _complete,
    _prototype_matrix,
    _run_dir,
)
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
FIG = ROOT / "experiments" / "evostar_2027" / "rq3" / "figures"
NOTES = ROOT / "notes" / "rough_draft"
RUN_LEVEL = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "rq2_run_level.csv"
DYN = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "rq2_species_dynamics.csv"

DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}
NOISE = {None, 0, -1, "0", "-1"}


def _sid(rec: Dict[str, Any]) -> Any:
    return rec.get("species_id", rec.get("density_label", rec.get("label")))


def _prompt(rec: Dict[str, Any]) -> Optional[str]:
    for k in ("prompt", "text", "genome", "content"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return None


def rarefaction_curve(
    labels: Sequence[Any],
    draws: int,
    rng: np.random.Generator,
    ms: Sequence[int] = (1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50),
) -> Dict[int, float]:
    n = len(labels)
    if n == 0:
        return {}
    # Keep as a Python list of hashable labels (e.g. (risk, style) tuples).
    # np.asarray(..., dtype=object) can promote tuple sequences to a 2-D array.
    lab = list(labels)
    out: Dict[int, float] = {}
    for m in ms:
        if m > n:
            continue
        vals = []
        for _ in range(draws):
            idx = rng.choice(n, size=m, replace=False)
            vals.append(len({lab[int(i)] for i in idx}))
        out[m] = float(np.mean(vals))
    return out


def spearman(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    xa = np.asarray(x, dtype=float)
    ya = np.asarray(y, dtype=float)
    if xa.size < 3:
        return float("nan"), float("nan")
    try:
        from scipy.stats import spearmanr

        r = spearmanr(xa, ya)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        rx = np.argsort(np.argsort(xa)).astype(float)
        ry = np.argsort(np.argsort(ya)).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan"), float("nan")
        return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def load_end_dynamics() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    if not DYN.is_file():
        return out
    with DYN.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                b = int(float(row.get("budget") or row.get("b") or 0))
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

            out[key] = {
                "n_mature": _f("n_mature") if "n_mature" in row else _f("n_mat"),
                "d_assigned": _f("d_assigned") if "d_assigned" in row else _f("D_assigned"),
                "unassigned_frac": _f("unassigned_frac"),
            }
    return out


def load_run_level() -> Dict[Tuple[str, str, int], Dict[str, float]]:
    out: Dict[Tuple[str, str, int], Dict[str, float]] = {}
    with RUN_LEVEL.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("status") not in ("complete", "ok", ""):
                # status field is "complete" in tracker-derived rows
                if row.get("status") == "missing_run":
                    continue
            key = (row["distance"], row["clustering"], int(row["seed"]))
            def _f(name: str) -> float:
                try:
                    return float(row[name])
                except Exception:
                    return float("nan")

            out[key] = {
                "top10_mean": _f("top10_mean"),
                "n_species": _f("n_species"),
                "largest_frac": _f("largest_frac"),
            }
    return out


def analyze_run(
    distance: str,
    clustering: str,
    seed: int,
    tau: float,
    encoder,
    risk_m,
    style_m,
    risk_names,
    style_names,
    cache,
    dyn,
    run_level,
    rng: np.random.Generator,
    draws: int,
) -> Optional[Dict[str, Any]]:
    run_dir = _run_dir(distance, clustering, seed)
    if not _complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    high_cells: List[Tuple[str, str]] = []
    by_species: Dict[Any, Counter] = defaultdict(Counter)
    n_high = 0
    for rec in genomes:
        f = illicit_of(rec)
        if f is None or f < tau:
            continue
        n_high += 1
        z = _assign(rec, risk_m, style_m, risk_names, style_names, encoder, cache)
        if z is None:
            continue
        high_cells.append(z)
        sid = _sid(rec)
        if sid not in NOISE:
            by_species[sid][z] += 1

    c_final = len(set(high_cells))
    # rarefaction expected coverage at common m grid
    curve = rarefaction_curve(high_cells, draws=draws, rng=rng)
    # within-species purity among high-fitness assigned members
    purities = []
    for cnt in by_species.values():
        tot = sum(cnt.values())
        if tot <= 0:
            continue
        purities.append(max(cnt.values()) / tot)
    mean_purity = float(np.mean(purities)) if purities else float("nan")
    n_species_with_high = len(by_species)

    key = (distance, clustering, seed)
    drow = dyn.get(key, {})
    rrow = run_level.get(key, {})
    row = {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "tau": tau,
        "n_high": n_high,
        "c_final": c_final,
        "n_species_end": rrow.get("n_species", float("nan")),
        "n_mature_b2000": drow.get("n_mature", float("nan")),
        "d_assigned_b2000": drow.get("d_assigned", float("nan")),
        "mean_species_purity_high": mean_purity,
        "n_species_with_high": n_species_with_high,
        "top10_mean": rrow.get("top10_mean", float("nan")),
    }
    for m in (1, 2, 3, 4, 5, 8, 10, 15, 20, 30, 40, 50):
        row[f"c_hat_m{m}"] = curve.get(m, float("nan")) if n_high >= m else float("nan")
    return row


def write_csv(path: Path, rows: List[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def association_table(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    pairs = [
        ("n_mature_b2000", "c_final"),
        ("n_mature_b2000", "n_high"),
        ("d_assigned_b2000", "c_final"),
        ("n_high", "c_final"),
        ("mean_species_purity_high", "c_final"),
        ("c_hat_m5", "c_final"),
    ]
    scopes = [("all", rows)]
    for d in ("embedding", "nli_entail"):
        scopes.append((d, [r for r in rows if r["distance"] == d]))
    for scope, subset in scopes:
        for xname, yname in pairs:
            xs = [float(r[xname]) for r in subset if r.get(xname) == r.get(xname)]
            ys = []
            xs2 = []
            for r in subset:
                try:
                    x = float(r[xname])
                    y = float(r[yname])
                except Exception:
                    continue
                if math.isfinite(x) and math.isfinite(y):
                    xs2.append(x)
                    ys.append(y)
            rho, p = spearman(xs2, ys)
            out.append(
                {
                    "scope": scope,
                    "x": xname,
                    "y": yname,
                    "n": len(xs2),
                    "spearman_rho": rho,
                    "p": p,
                }
            )
    return out


def rarefaction_summary(rows: List[Dict[str, Any]], m: int = 5) -> List[Dict[str, Any]]:
    out = []
    key = f"c_hat_m{m}"
    for distance, clustering in LIVE_DIR:
        vals = []
        nh = []
        cf = []
        for r in rows:
            if r["distance"] != distance or r["clustering"] != clustering:
                continue
            nh.append(float(r["n_high"]))
            cf.append(float(r["c_final"]))
            try:
                v = float(r[key])
            except Exception:
                continue
            if math.isfinite(v):
                vals.append(v)
        if not nh:
            continue
        out.append(
            {
                "distance": distance,
                "clustering": clustering,
                "m": m,
                "n_runs": len(nh),
                "n_rarefied": len(vals),
                "median_c_hat": float(np.median(vals)) if vals else float("nan"),
                "iqr_lo": float(np.percentile(vals, 25)) if vals else float("nan"),
                "iqr_hi": float(np.percentile(vals, 75)) if vals else float("nan"),
                "median_n_high": float(np.median(nh)),
                "median_c_final": float(np.median(cf)),
            }
        )
    return out


def write_latex(path: Path, rare: List[Dict[str, Any]], assoc: List[Dict[str, Any]]) -> None:
    lines = [
        "% Auto-generated by rq3_diversity_audit.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{RQ3 exploratory descriptor coverage (MiniLM prototype grid, "
        r"$\tau{=}0.80$). $C_r$: occupied high-fitness cells. $|H_r|$: qualifying "
        r"genomes. $\widehat C_r(5)$: expected occupied cells after sampling five "
        r"qualifying genomes (rarefaction; $n_{\mathrm{raf}}$ runs with $|H_r|\ge 5$). "
        r"Prototype labels are not human-validated behaviors; annotation is pending.}",
        r"\label{tab:rq3-coverage-audit}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccccc@{}}",
        r"\toprule",
        r"Dissim. & Method & $n$ & med.\ $C_r$ & med.\ $|H_r|$ & med.\ $\widehat C_r(5)$ & $n_{\mathrm{raf}}$ \\",
        r"\midrule",
    ]
    for row in rare:
        chat = row["median_c_hat"]
        chat_s = "---" if chat != chat else f"{chat:.2f}"
        lines.append(
            f"{DIST_TEX[row['distance']]} & {CLUS_TEX[row['clustering']]} & "
            f"{row['n_runs']} & {row['median_c_final']:.0f} & "
            f"{row['median_n_high']:.0f} & {chat_s} & {row['n_rarefied']} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Run-level Spearman associations for RQ3 (exploratory). "
        r"$n_{\mathrm{mat}}$ and $D_{\mathrm{assigned}}$ are taken at budget $2000$. "
        r"Not Holm-adjusted with RQ2 fitness tests.}",
        r"\label{tab:rq3-assoc}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrr@{}}",
        r"\toprule",
        r"Scope & Pair & $n$ & $\rho$ \\",
        r"\midrule",
    ]
    name = {
        "all": "all",
        "embedding": r"$d_E$",
        "nli_entail": r"$d_{\neg E}$",
        "n_mature_b2000": r"$n_{\mathrm{mat}}$",
        "d_assigned_b2000": r"$D_{\mathrm{assigned}}$",
        "n_high": r"$|H_r|$",
        "c_final": r"$C_r$",
    }
    for row in assoc:
        if row["scope"] not in ("all", "embedding", "nli_entail"):
            continue
        if row["x"] not in ("n_mature_b2000", "n_high", "d_assigned_b2000"):
            continue
        if row["y"] not in ("c_final", "n_high"):
            continue
        rho = row["spearman_rho"]
        rho_s = "---" if rho != rho else f"{rho:.2f}"
        lines.append(
            f"{name[row['scope']]} & {name[row['x']]} vs {name[row['y']]} & "
            f"{row['n']} & {rho_s} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_rarefaction(rows: List[Dict[str, Any]], out: Path) -> None:
    ms = [1, 2, 3, 4, 5, 8, 10, 15, 20]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), sharey=True)
    styles = {
        ("embedding", "leader_follower"): ("#0F4D92", "-"),
        ("embedding", "dbscan"): ("#0F4D92", "--"),
        ("nli_entail", "leader_follower"): ("#B64342", "-"),
        ("nli_entail", "dbscan"): ("#B64342", "--"),
    }
    for ax, distance in zip(axes, ("embedding", "nli_entail")):
        for clustering in ("leader_follower", "dbscan"):
            series = []
            for m in ms:
                vals = [
                    float(r[f"c_hat_m{m}"])
                    for r in rows
                    if r["distance"] == distance
                    and r["clustering"] == clustering
                    and math.isfinite(float(r.get(f"c_hat_m{m}", float("nan"))))
                ]
                series.append(float(np.median(vals)) if vals else float("nan"))
            color, ls = styles[(distance, clustering)]
            ax.plot(ms, series, color=color, linestyle=ls, marker="o", ms=3.5, label=CLUS_TEX[clustering])
        ax.set_title(DIST_TEX[distance])
        ax.set_xlabel(r"matched $|H|$ sample size $m$")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"median $\widehat C_r(m)$")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    fig.savefig(NOTES / "figures" / out.name, dpi=160)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--draws", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    risk_m, style_m, risk_names, style_names = _prototype_matrix(encoder)
    cache: Dict[str, Tuple[str, str]] = {}
    dyn = load_end_dynamics()
    run_level = load_run_level()
    rng = np.random.default_rng(args.seed)

    rows: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        for seed in range(10):
            print(f"audit {distance} {clustering} seed{seed}", flush=True)
            row = analyze_run(
                distance,
                clustering,
                seed,
                args.tau,
                encoder,
                risk_m,
                style_m,
                risk_names,
                style_names,
                cache,
                dyn,
                run_level,
                rng,
                args.draws,
            )
            if row:
                rows.append(row)

    REPORT.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r.keys()})
    # stable column order
    head = [
        "distance",
        "clustering",
        "seed",
        "tau",
        "n_high",
        "c_final",
        "n_species_end",
        "n_mature_b2000",
        "d_assigned_b2000",
        "mean_species_purity_high",
        "n_species_with_high",
        "top10_mean",
    ]
    fields = head + [f for f in fields if f not in head]
    write_csv(REPORT / "rq3_run_diversity.csv", rows, fields)
    assoc = association_table(rows)
    write_csv(
        REPORT / "rq3_associations.csv",
        assoc,
        ["scope", "x", "y", "n", "spearman_rho", "p"],
    )
    rare = rarefaction_summary(rows, m=5)
    write_csv(
        REPORT / "rq3_rarefaction_summary.csv",
        rare,
        [
            "distance",
            "clustering",
            "m",
            "n_runs",
            "n_rarefied",
            "median_c_hat",
            "iqr_lo",
            "iqr_hi",
            "median_n_high",
            "median_c_final",
        ],
    )
    write_latex(NOTES / "rq3_diversity_tables.tex", rare, assoc)
    plot_rarefaction(rows, FIG / "fig_rq3_rarefaction.pdf")
    meta = {
        "n_runs": len(rows),
        "tau": args.tau,
        "draws": args.draws,
        "seed": args.seed,
        "note": "Prototype MiniLM labels; not human-validated behaviors.",
    }
    (REPORT / "rq3_diversity_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} run rows to {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
