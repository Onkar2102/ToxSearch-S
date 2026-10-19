#!/usr/bin/env python3
"""RQ2 factorial report: d_E vs d_¬E × LF vs IncDBSCAN (10 seeds).

Scans ``experiments/evostar_2027/rq1/pilots/`` for completed cells, extracts
run-level fitness / coverage / cost / quality / diversity, runs Mann--Whitney U
pairwise tests, and writes CSV + LaTeX + optional figures.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PILOTS = Path("experiments/evostar_2027/rq1/pilots")
OUT = Path("experiments/evostar_2027/rq2/reports")
FIG = Path("experiments/evostar_2027/rq2/figures")
NOTES = Path("notes/LaTeX2e+Proceedings+Templates+download")

DIST_ORDER = ["embedding", "nli_entail"]
CLUS_ORDER = ["leader_follower", "dbscan"]
DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}

METRIC_COLS = [
    "best_fitness",
    "top10_mean",
    "top50_mean",
    "n_species",
    "inter_diversity",
    "intra_diversity",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "qd_score",
    "total_llm_calls",
    "total_api_calls",
    "total_response_time",
    "total_evaluation_time",
    "evaluated_proxy",
]


def _load_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _cell_id(distance: str, clustering: str, seed: int) -> str:
    d = distance.replace("+", "_").replace("/", "_")
    c = clustering.replace("-", "_")
    return f"{d}__{c}__seed{seed}"


def _find_tracker(run_dir: Path) -> Optional[Path]:
    for name in ("EvolutionTracker.json", "generation_history.json"):
        p = run_dir / name
        if p.is_file():
            return p
    # nested
    for p in run_dir.rglob("EvolutionTracker.json"):
        return p
    return None


def _toxicity_of(rec: Any) -> Optional[float]:
    if not isinstance(rec, dict):
        return None
    for k in ("toxicity", "score", "fitness", "perspective_toxicity"):
        if k in rec and rec[k] is not None:
            try:
                return float(rec[k])
            except Exception:
                continue
    return None


def _top_k_mean(path: Path, k: int) -> Optional[float]:
    data = _load_json(path)
    if not isinstance(data, list) or not data:
        return None
    vals = []
    for rec in data[:k]:
        v = _toxicity_of(rec)
        if v is not None:
            vals.append(v)
    return float(np.mean(vals)) if vals else None


def _best_so_far_curve(tracker: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    gens = tracker.get("generations") or []
    gens = sorted(gens, key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return np.asarray([0.0]), np.asarray([0.0])
    cum = 0.0
    best = 0.0
    xs: List[float] = [0.0]
    ys: List[float] = [0.0]
    for g in gens:
        step = g.get("evaluated_this_generation")
        if not isinstance(step, (int, float)):
            step = g.get("variants_created")
        step = float(step) if isinstance(step, (int, float)) else 0.0
        step = max(0.0, step)
        m = g.get("max_score_variants")
        if not isinstance(m, (int, float)):
            m = g.get("population_max_fitness") or g.get("best_fitness") or 0.0
        m = float(m) if isinstance(m, (int, float)) else 0.0
        best = max(best, m)
        cum += step
        xs.append(cum)
        ys.append(best)
    # fallback x: generation index if budget never advanced
    if max(xs) <= 0.0:
        xs = [float(i) for i in range(len(ys))]
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def extract_cell_metrics(cell_dir: Path) -> Dict[str, Any]:
    meta = _load_json(cell_dir / "cell.json") or {}
    distance = str(meta.get("distance_method") or "")
    clustering = str(meta.get("clustering_method") or "")
    seed = int(meta.get("seed") or 0)
    run_dir = cell_dir / "run"
    row: Dict[str, Any] = {
        "cell_id": cell_dir.name,
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "status": "missing_run",
        "mode": meta.get("mode"),
        "exit_code": meta.get("exit_code"),
    }
    for k in METRIC_COLS:
        row[k] = ""

    tracker_path = _find_tracker(run_dir) if run_dir.is_dir() else None
    if tracker_path is None:
        if meta.get("mode") == "run" and meta.get("exit_code") == 0:
            row["status"] = "run_ok_no_tracker"
        elif meta.get("mode") == "dry-run":
            row["status"] = "dry_run"
        return row

    tracker = _load_json(tracker_path)
    if not isinstance(tracker, dict):
        row["status"] = "bad_tracker"
        return row

    row["status"] = str(tracker.get("status") or "ok")
    xs, ys = _best_so_far_curve(tracker)
    row["best_fitness"] = float(ys[-1]) if len(ys) else float(tracker.get("population_max_toxicity") or 0.0)
    row["evaluated_proxy"] = float(xs[-1]) if len(xs) else ""

    top10 = _top_k_mean(run_dir / "top_10.json", 10)
    if top10 is None and tracker.get("generations"):
        last = tracker["generations"][-1]
        tlist = last.get("top_10") or []
        vals = [_toxicity_of(r) for r in tlist]
        vals = [v for v in vals if v is not None]
        top10 = float(np.mean(vals)) if vals else None
    if top10 is not None:
        row["top10_mean"] = top10

    # top-50: prefer elites/archive if present
    elites = _load_json(run_dir / "elites.json")
    if isinstance(elites, list) and elites:
        vals = [_toxicity_of(r) for r in elites[:50]]
        vals = [v for v in vals if v is not None]
        if vals:
            row["top50_mean"] = float(np.mean(vals))

    gens = tracker.get("generations") or []
    last_sp: Dict[str, Any] = {}
    if gens:
        last_sp = gens[-1].get("speciation") or {}
    summary = tracker.get("speciation_summary") or {}
    row["n_species"] = last_sp.get("species_count") or summary.get("current_species_count") or ""
    row["inter_diversity"] = last_sp.get("inter_species_diversity") or ""
    row["intra_diversity"] = last_sp.get("intra_species_diversity") or ""
    cq = last_sp.get("cluster_quality") or {}
    if isinstance(cq, dict):
        row["silhouette"] = cq.get("silhouette_score", "")
        row["davies_bouldin"] = cq.get("davies_bouldin_index", "")
        row["calinski_harabasz"] = cq.get("calinski_harabasz_index", "")
        row["qd_score"] = cq.get("qd_score", "")

    budget = tracker.get("cumulative_budget") or {}
    if isinstance(budget, dict):
        row["total_llm_calls"] = budget.get("total_llm_calls", "")
        row["total_api_calls"] = budget.get("total_api_calls", "")
        row["total_response_time"] = budget.get("total_response_time", "")
        row["total_evaluation_time"] = budget.get("total_evaluation_time", "")

    return row


def expected_grid(seeds: Sequence[int]) -> List[Tuple[str, str, int]]:
    out = []
    for d in DIST_ORDER:
        for c in CLUS_ORDER:
            for s in seeds:
                out.append((d, c, int(s)))
    return out


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return float("nan")
    gt = lt = 0
    for x in a:
        for y in b:
            if x > y:
                gt += 1
            elif x < y:
                lt += 1
    n = len(a) * len(b)
    return (gt - lt) / n if n else float("nan")


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


def _finite(vals: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for v in vals:
        if v in ("", None):
            continue
        try:
            x = float(v)
        except Exception:
            continue
        if math.isfinite(x):
            out.append(x)
    return out


def mannwhitney_table(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    try:
        from scipy.stats import mannwhitneyu
    except Exception:
        return []

    # cells = distance × clustering
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        if r.get("status") in ("dry_run", "missing_run", "bad_tracker"):
            continue
        key = (str(r["distance"]), str(r["clustering"]))
        groups.setdefault(key, []).append(r)

    contrasts: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = []
    # LF vs IncDBSCAN within distance
    for d in DIST_ORDER:
        contrasts.append((f"{d}: LF vs IncDBSCAN", (d, "leader_follower"), (d, "dbscan")))
    # d_E vs d_negE within speciation
    for c in CLUS_ORDER:
        contrasts.append((f"{c}: d_E vs d_negE", ("embedding", c), ("nli_entail", c)))

    out_rows: List[Dict[str, Any]] = []
    for metric in metrics:
        raw_p: List[float] = []
        scratch: List[Dict[str, Any]] = []
        for name, ka, kb in contrasts:
            xa = _finite(r.get(metric) for r in groups.get(ka, []))
            xb = _finite(r.get(metric) for r in groups.get(kb, []))
            rec = {
                "metric": metric,
                "contrast": name,
                "n_a": len(xa),
                "n_b": len(xb),
                "median_a": float(np.median(xa)) if xa else "",
                "median_b": float(np.median(xb)) if xb else "",
                "U": "",
                "p": "",
                "holm_p": "",
                "cliffs_delta": "",
            }
            if len(xa) >= 2 and len(xb) >= 2:
                u = mannwhitneyu(xa, xb, alternative="two-sided")
                rec["U"] = float(u.statistic)
                rec["p"] = float(u.pvalue)
                rec["cliffs_delta"] = float(cliffs_delta(xa, xb))
                raw_p.append(float(u.pvalue))
            else:
                raw_p.append(float("nan"))
            scratch.append(rec)
        finite_idx = [i for i, p in enumerate(raw_p) if p == p]
        adj = [float("nan")] * len(raw_p)
        if finite_idx:
            adj_vals = holm_adjust([raw_p[i] for i in finite_idx])
            for ii, vv in zip(finite_idx, adj_vals):
                adj[ii] = vv
        for i, rec in enumerate(scratch):
            if adj[i] == adj[i]:
                rec["holm_p"] = float(adj[i])
            out_rows.append(rec)
    return out_rows


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_latex_summary(path: Path, rows: List[Dict[str, Any]], stats: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "% Auto-generated by rq2_factorial_report.py — do not hand-edit.",
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{RQ2 live factorial status ($d_E$ / $d_{\neg E}$ $\times$ LF / IncDBSCAN $\times$ 10 seeds; "
        r"$N_{\max}=2000$). Medians over completed runs; empty cells mean the grid is not finished.}",
        r"\label{tab:rq2-factorial-summary}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llrrrrrr@{}}",
        r"\toprule",
        r"Distance & Speciation & $n$ & Best fitness & Top-10 & Species & Inter-div. & LLM calls \\",
        r"\midrule",
    ]
    for d in DIST_ORDER:
        for c in CLUS_ORDER:
            cell_rows = [
                r
                for r in rows
                if r.get("distance") == d
                and r.get("clustering") == c
                and r.get("status") not in ("dry_run", "missing_run", "bad_tracker")
            ]
            n = len(cell_rows)

            def med(key: str) -> str:
                vals = _finite(r.get(key) for r in cell_rows)
                return f"{np.median(vals):.3f}" if vals else "---"

            lines.append(
                f"{DIST_TEX[d]} & {CLUS_TEX[c]} & {n} & {med('best_fitness')} & {med('top10_mean')} & "
                f"{med('n_species')} & {med('inter_diversity')} & {med('total_llm_calls')} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]

    # Stats snippet for primary metric
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{RQ2 Mann--Whitney U tests on run-level best fitness "
        r"(two-sided; Holm-adjusted within the best-fitness family).}",
        r"\label{tab:rq2-mw-best}",
        r"\scriptsize",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Contrast & $n_a$/$n_b$ & Median$_a$/$_b$ & $p$ (Holm) & Cliff's $\delta$ \\",
        r"\midrule",
    ]
    for rec in stats:
        if rec.get("metric") != "best_fitness":
            continue
        if rec.get("p") in ("", None):
            continue
        med = f"{rec['median_a']:.3f}/{rec['median_b']:.3f}" if rec["median_a"] != "" else "---"
        p = f"{float(rec['p']):.3g}"
        hp = f"{float(rec['holm_p']):.3g}" if rec.get("holm_p") not in ("", None) else "---"
        dlt = f"{float(rec['cliffs_delta']):+.2f}" if rec.get("cliffs_delta") not in ("", None) else "---"
        lines.append(
            f"{rec['contrast']} & {rec['n_a']}/{rec['n_b']} & {med} & {p} ({hp}) & {dlt} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def maybe_plot(rows: List[Dict[str, Any]], fig_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("[warn] matplotlib unavailable; skip figures")
        return
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Boxplot best fitness by cell type
    labels = []
    data = []
    for d in DIST_ORDER:
        for c in CLUS_ORDER:
            vals = _finite(
                r.get("best_fitness")
                for r in rows
                if r.get("distance") == d and r.get("clustering") == c
            )
            labels.append(f"{DIST_TEX.get(d,d)}\n{CLUS_TEX.get(c,c)}")
            data.append(vals)
    if not any(data):
        return
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_ylabel("Best fitness (toxicity)")
    ax.set_title("RQ2 run-level best fitness")
    fig.tight_layout()
    out = fig_dir / "rq2_best_fitness_box.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Wrote {out}")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ2 factorial report")
    p.add_argument("--pilots-dir", type=Path, default=PILOTS)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--fig-dir", type=Path, default=FIG)
    p.add_argument("--notes-dir", type=Path, default=NOTES)
    p.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4,5,6,7,8,9",
        help="Expected seed list",
    )
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    rows: List[Dict[str, Any]] = []
    for d, c, s in expected_grid(seeds):
        cell = args.pilots_dir / _cell_id(d, c, s)
        if cell.is_dir():
            rows.append(extract_cell_metrics(cell))
        else:
            rows.append(
                {
                    "cell_id": _cell_id(d, c, s),
                    "distance": d,
                    "clustering": c,
                    "seed": s,
                    "status": "missing_run",
                    **{k: "" for k in METRIC_COLS},
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "distance", "clustering", "seed", "status", "mode", "exit_code"] + METRIC_COLS
    run_csv = args.out_dir / "rq2_run_level.csv"
    write_csv(run_csv, rows, fields)
    print(f"Wrote {run_csv} ({len(rows)} rows)")

    done = sum(1 for r in rows if r.get("status") not in ("missing_run", "dry_run", "bad_tracker"))
    print(f"Completed-ish cells: {done}/{len(rows)}")

    test_metrics = [
        "best_fitness",
        "top10_mean",
        "n_species",
        "inter_diversity",
        "intra_diversity",
        "silhouette",
        "total_llm_calls",
        "total_response_time",
    ]
    stats = mannwhitney_table(rows, test_metrics)
    stats_csv = args.out_dir / "rq2_mannwhitney.csv"
    write_csv(
        stats_csv,
        stats,
        ["metric", "contrast", "n_a", "n_b", "median_a", "median_b", "U", "p", "holm_p", "cliffs_delta"],
    )
    print(f"Wrote {stats_csv} ({len(stats)} rows)")

    tex = args.notes_dir / "rq2_factorial_tables.tex"
    write_latex_summary(tex, rows, stats)
    print(f"Wrote {tex}")

    maybe_plot(rows, args.fig_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
