#!/usr/bin/env python3
"""RQ2 factorial report: d_E vs d_¬E × LF vs IncDBSCAN (10 seeds).

Reads live cells from ``data/outputs/evostar2027/{lf,inc}_{embedding,nli}/repK_sK``.
Writes CSV + LaTeX + optional figures. Missing seeds stay in the grid as
``missing_run`` so the remaining-execution list is exact.

Example::

    PYTHONPATH=src:. python experiments/evostar_2027/rq2/scripts/rq2_factorial_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import (  # noqa: E402
    common_silhouette,
    llm_calls_from_genomes,
    load_genomes,
    n_evaluated,
    topk_mean_truncated,
)

LIVE = Path("data/outputs/evostar2027")
OUT = Path("experiments/evostar_2027/rq2/reports")
FIG = Path("experiments/evostar_2027/rq2/figures")
NOTES = Path("notes/rough_draft")

DIST_ORDER = ["embedding", "nli_entail"]
CLUS_ORDER = ["leader_follower", "dbscan"]
DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}

# Folder names under data/outputs/evostar2027
LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}

ELITE_K = 100

METRIC_COLS = [
    "best_fitness",
    "top10_mean",
    "top10_mean_b2000",
    "top50_mean",
    "top100_mean",
    "top100_mean_b2000",
    "n_species",
    "inter_diversity",
    "intra_diversity",
    "silhouette",
    "davies_bouldin",
    "calinski_harabasz",
    "qd_score",
    "n_unique",
    "n_evaluated",
    "tracker_api_calls",
    "tracker_llm_calls",
    "llm_calls_recount",
    "total_response_time",
    "total_evaluation_time",
    "wall_hours",
    "min_samples",
    "largest_frac",
    "silhouette_common",
    "assigned_frac",
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


def _run_dir(live_root: Path, distance: str, clustering: str, seed: int) -> Path:
    folder = LIVE_DIR[(distance, clustering)]
    return live_root / folder / f"rep{seed}_s{seed}"


def _illicit_of(rec: Any) -> Optional[float]:
    if not isinstance(rec, dict):
        return None
    mr = rec.get("moderation_result") or {}
    oa = mr.get("openai") if isinstance(mr, dict) else None
    scores = (oa or {}).get("scores") if isinstance(oa, dict) else {}
    for v in (scores.get("illicit") if isinstance(scores, dict) else None, rec.get("illicit"), rec.get("toxicity"), rec.get("fitness"), rec.get("score")):
        if v is None:
            continue
        try:
            return float(v)
        except Exception:
            continue
    return None


def _scored_records(run_dir: Path, clustering: str) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    if clustering == "dbscan":
        data = _load_json(run_dir / "temp.json")
        if isinstance(data, list):
            recs.extend(x for x in data if isinstance(x, dict))
    else:
        for name in ("elites.json", "archive.json"):
            data = _load_json(run_dir / name)
            if isinstance(data, list):
                recs.extend(x for x in data if isinstance(x, dict))
    return recs


def _unique_n(run_dir: Path, clustering: str, tracker: Dict[str, Any], recs: List[Dict[str, Any]]) -> int:
    ids = {r.get("id") for r in recs if r.get("id") is not None}
    if ids:
        return len(ids)
    gens = tracker.get("generations") or []
    if gens:
        last = gens[-1]
        n = last.get("total_population") or last.get("density_size")
        if isinstance(n, (int, float)) and n > 0:
            return int(n)
    n = tracker.get("cumulative_variants_evaluated")
    return int(n) if isinstance(n, (int, float)) else 0


def extract_run(run_dir: Path, distance: str, clustering: str, seed: int) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "cell_id": _cell_id(distance, clustering, seed),
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "status": "missing_run",
        "run_dir": str(run_dir),
    }
    for k in METRIC_COLS:
        row[k] = ""

    tracker = _load_json(run_dir / "EvolutionTracker.json")
    if not isinstance(tracker, dict):
        return row

    cfg = _load_json(run_dir / "run_config.json") or {}
    spec = cfg.get("speciation") or {}
    md = tracker.get("run_metadata") or {}
    row["status"] = str(tracker.get("status") or "ok")
    row["min_samples"] = spec.get("dbscan_min_samples") or md.get("dbscan_min_samples") or ""

    recs = load_genomes(run_dir, clustering) or _scored_records(run_dir, clustering)
    scores = sorted((v for v in (_illicit_of(r) for r in recs) if v is not None), reverse=True)
    row["n_unique"] = _unique_n(run_dir, clustering, tracker, recs)
    row["n_evaluated"] = n_evaluated(recs)
    recount = llm_calls_from_genomes(recs)
    row["llm_calls_recount"] = recount["llm_calls_total"]
    sil = common_silhouette(recs)
    row["silhouette_common"] = sil if sil is not None else ""
    if recs:
        n_ass = sum(1 for r in recs if r.get("species_id") not in (None, 0, -1, "0", "-1"))
        row["assigned_frac"] = n_ass / len(recs)
    if scores:
        row["best_fitness"] = float(scores[0])
        row["top10_mean"] = float(np.mean(scores[:10]))
        row["top50_mean"] = float(np.mean(scores[:50]))
        row["top100_mean"] = float(np.mean(scores[:ELITE_K]))
        trunc10 = topk_mean_truncated(recs, k=10, budget=2000)
        row["top10_mean_b2000"] = trunc10 if trunc10 is not None else row["top10_mean"]
        trunc100 = topk_mean_truncated(recs, k=ELITE_K, budget=2000)
        row["top100_mean_b2000"] = trunc100 if trunc100 is not None else row["top100_mean"]
    else:
        row["best_fitness"] = float(tracker.get("population_max_fitness") or 0.0)

    gens = tracker.get("generations") or []
    last_sp: Dict[str, Any] = {}
    if gens:
        last_sp = gens[-1].get("speciation") or {}
        wall = tracker.get("execution_time_seconds") or md.get("run_duration_seconds")
        if isinstance(wall, (int, float)):
            row["wall_hours"] = round(float(wall) / 3600.0, 2)
    summary = tracker.get("speciation_summary") or {}
    row["n_species"] = last_sp.get("species_count") or summary.get("current_species_count") or ""
    largest = last_sp.get("largest_species_size")
    try:
        nu = float(row.get("n_unique") or 0)
        if largest not in ("", None) and nu > 0:
            row["largest_frac"] = float(largest) / nu
    except (TypeError, ValueError):
        pass
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
        # Tracker counters are inflated for IncDBSCAN (gen-0 double-count).
        # Keep them for audit; use n_evaluated / llm_calls_recount for tests.
        row["tracker_api_calls"] = budget.get("total_api_calls", "")
        row["tracker_llm_calls"] = budget.get("total_llm_calls", "")
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
    """δ = P(X>Y) − P(X<Y). Positive values mean the first sample ranks higher."""
    if len(a) == 0 or len(b) == 0:
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


def cliffs_delta_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10000,
    seed: int = 0,
) -> Tuple[float, float]:
    xa = np.asarray(list(a), dtype=float)
    xb = np.asarray(list(b), dtype=float)
    if xa.size == 0 or xb.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sa = rng.choice(xa, size=xa.size, replace=True)
        sb = rng.choice(xb, size=xb.size, replace=True)
        vals[i] = cliffs_delta(sa, sb)
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


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


def _complete_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [r for r in rows if str(r.get("status")) == "complete"]


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


MWU_N_PERM = 10_000
MWU_PERM_SEED = 2027


def _u_stat(a: Sequence[float], b: Sequence[float]) -> float:
    """Mann--Whitney U for sample a (average ranks; ties allowed)."""
    xa, xb = list(a), list(b)
    n1 = len(xa)
    comb = xa + xb
    order = sorted(range(len(comb)), key=lambda i: comb[i])
    ranks = [0.0] * len(comb)
    i = 0
    while i < len(comb):
        j = i
        while j + 1 < len(comb) and comb[order[j + 1]] == comb[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    r1 = sum(ranks[:n1])
    return r1 - n1 * (n1 + 1) / 2.0


def _mwu(
    a: Sequence[float],
    b: Sequence[float],
    n_perm: int = MWU_N_PERM,
    seed: int = MWU_PERM_SEED,
) -> Tuple[float, float]:
    """Two-sided Mann--Whitney U with a permutation p-value.

    SciPy's exact null does not correct for ties. Permuting group labels
    remains valid for small samples that contain ties.
    """
    xa, xb = list(a), list(b)
    n1, n2 = len(xa), len(xb)
    if n1 == 0 or n2 == 0:
        return float("nan"), float("nan")
    u_obs = _u_stat(xa, xb)
    mu = n1 * n2 / 2.0
    obs = abs(u_obs - mu)
    pooled = xa + xb
    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        up = _u_stat(pooled[:n1], pooled[n1:])
        if abs(up - mu) >= obs - 1e-12:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return float(u_obs), float(p)


PRIMARY_CONTRASTS: List[Tuple[str, str, Tuple[str, str], Tuple[str, str]]] = [
    ("embedding: LF vs IncDBSCAN", r"$d_E$: LF vs IncDBSCAN", ("embedding", "leader_follower"), ("embedding", "dbscan")),
    ("nli_entail: LF vs IncDBSCAN", r"$d_{\neg E}$: LF vs IncDBSCAN", ("nli_entail", "leader_follower"), ("nli_entail", "dbscan")),
]


def mannwhitney_table(rows: List[Dict[str, Any]], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in _complete_rows(rows):
        key = (str(r["distance"]), str(r["clustering"]))
        groups.setdefault(key, []).append(r)

    contrasts: List[Tuple[str, Tuple[str, str], Tuple[str, str]]] = [
        (name, ka, kb) for name, _, ka, kb in PRIMARY_CONTRASTS
    ]

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
                "cliffs_ci_lo": "",
                "cliffs_ci_hi": "",
            }
            if len(xa) >= 2 and len(xb) >= 2:
                u, p = _mwu(xa, xb)
                rec["U"] = u
                rec["p"] = p
                rec["cliffs_delta"] = float(cliffs_delta(xa, xb))
                lo, hi = cliffs_delta_ci(xa, xb)
                rec["cliffs_ci_lo"] = lo
                rec["cliffs_ci_hi"] = hi
                raw_p.append(p)
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


def primary_top10_mwu(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unpaired MWU + Cliff's δ on final mean top-ELITE_K illicit (LF vs Inc, two distances)."""
    groups: Dict[Tuple[str, str], List[float]] = {}
    for r in _complete_rows(rows):
        vals = _finite([r.get("top100_mean")])
        if not vals:
            continue
        key = (str(r["distance"]), str(r["clustering"]))
        groups.setdefault(key, []).append(vals[0])

    raw: List[Dict[str, Any]] = []
    pvals: List[float] = []
    for csv_name, tex_name, ka, kb in PRIMARY_CONTRASTS:
        xa = groups.get(ka, [])
        xb = groups.get(kb, [])
        rec: Dict[str, Any] = {
            "contrast": tex_name,
            "contrast_id": csv_name,
            "n_a": len(xa),
            "n_b": len(xb),
            "median_a": float(np.median(xa)) if xa else "",
            "median_b": float(np.median(xb)) if xb else "",
            "U": "",
            "p": "",
            "holm_p": "",
            "cliffs_delta": "",
            "cliffs_ci_lo": "",
            "cliffs_ci_hi": "",
        }
        if len(xa) >= 2 and len(xb) >= 2:
            u, p = _mwu(xa, xb)
            rec["U"] = u
            rec["p"] = p
            rec["cliffs_delta"] = float(cliffs_delta(xa, xb))
            lo, hi = cliffs_delta_ci(xa, xb)
            rec["cliffs_ci_lo"] = lo
            rec["cliffs_ci_hi"] = hi
            pvals.append(p)
        else:
            pvals.append(float("nan"))
        raw.append(rec)
    finite_idx = [i for i, p in enumerate(pvals) if p == p]
    if finite_idx:
        adj = holm_adjust([pvals[i] for i in finite_idx])
        for ii, vv in zip(finite_idx, adj):
            raw[ii]["holm_p"] = float(vv)
    return raw


def remaining_jobs(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        if r.get("status") in ("missing_run", "dry_run", "bad_tracker", "run_ok_no_tracker"):
            out.append(
                {
                    "cell_id": r["cell_id"],
                    "distance": r["distance"],
                    "clustering": r["clustering"],
                    "seed": r["seed"],
                    "status": r.get("status"),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt(x: Any, nd: int = 3) -> str:
    if x in ("", None):
        return "---"
    try:
        v = float(x)
    except Exception:
        return "---"
    if not math.isfinite(v):
        return "---"
    return f"{v:.{nd}f}"


def _iqr(vals: Sequence[float], nd: int = 3) -> str:
    if not vals:
        return "---"
    s = sorted(float(v) for v in vals)
    med = float(np.median(s))
    if len(s) < 4:
        lo, hi = s[0], s[-1]
    else:
        lo, hi = (float(x) for x in np.percentile(s, [25, 75]))
    return f"{med:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def write_latex_summary(
    path: Path,
    rows: List[Dict[str, Any]],
    tests: List[Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    done = _complete_rows(rows)
    lines = [
        "% Auto-generated by rq2_factorial_report.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{Summary of the $2\times 2\times 10$ factorial (median [IQR]) "
        rf"from the same initial population of $50$ individuals. Each run is one "
        rf"observation; its score is the mean fitness of that run's {ELITE_K} fittest "
        r"unique individuals (not species elites). Wall time is end-to-end clock "
        r"hours at evaluation budget $N_{\max}\approx 2000$ on one NVIDIA A100. "
        r"Mann--Whitney contrasts on the same scores are reported in the text.}",
        r"\label{tab:rq2-factorial-summary}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        rf"Dissim. & Method & $n$ & Top-{ELITE_K} fitness & Wall (h) \\",
        r"\midrule",
    ]
    del tests  # statistical contrasts are narrated in the main text
    for d in DIST_ORDER:
        for c in CLUS_ORDER:
            cell_rows = [
                r for r in done if r.get("distance") == d and r.get("clustering") == c
            ]
            n = len(cell_rows)
            lines.append(
                f"{DIST_TEX[d]} & {CLUS_TEX[c]} & {n} & "
                f"{_iqr(_finite(r.get('top100_mean') for r in cell_rows))} & "
                f"{_iqr(_finite(r.get('wall_hours') for r in cell_rows), 1)} \\\\"
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
    done = _complete_rows(rows)
    if not done:
        return

    def _box(metric: str, ylabel: str, fname: str) -> None:
        labels, data = [], []
        for d in DIST_ORDER:
            for c in CLUS_ORDER:
                vals = _finite(
                    r.get(metric)
                    for r in done
                    if r.get("distance") == d and r.get("clustering") == c
                )
                labels.append(f"{DIST_TEX.get(d, d)}\n{CLUS_TEX.get(c, c)}")
                data.append(vals if vals else [np.nan])
        fig, ax = plt.subplots(figsize=(6.5, 3.2))
        ax.boxplot(data, tick_labels=labels, showmeans=True)
        ax.set_ylabel(ylabel)
        fig.tight_layout()
        out = fig_dir / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Wrote {out}")

    _box("best_fitness", "Best illicit", "rq2_best_fitness_box.png")
    _box("silhouette", "Silhouette", "rq2_silhouette_box.png")
    _box("intra_diversity", "Intra-species diversity", "rq2_intra_box.png")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ2 factorial report")
    p.add_argument("--live-dir", type=Path, default=LIVE)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--fig-dir", type=Path, default=FIG)
    p.add_argument("--notes-dir", type=Path, default=NOTES)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]

    rows: List[Dict[str, Any]] = []
    for d, c, s in expected_grid(seeds):
        run_dir = _run_dir(args.live_dir, d, c, s)
        if (run_dir / "EvolutionTracker.json").is_file():
            rows.append(extract_run(run_dir, d, c, s))
        else:
            rec = {
                "cell_id": _cell_id(d, c, s),
                "distance": d,
                "clustering": c,
                "seed": s,
                "status": "missing_run",
                "run_dir": str(run_dir),
            }
            rec.update({k: "" for k in METRIC_COLS})
            rows.append(rec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["cell_id", "distance", "clustering", "seed", "status", "run_dir"] + METRIC_COLS
    run_csv = args.out_dir / "rq2_run_level.csv"
    write_csv(run_csv, rows, fields)
    print(f"Wrote {run_csv} ({len(rows)} rows)")

    leftover = remaining_jobs(rows)
    done = len(rows) - len(leftover)
    print(f"Completed cells: {done}/{len(rows)}")
    print(f"Remaining: {len(leftover)}")
    for r in leftover:
        print(f"  {r['cell_id']}")

    test_metrics = [
        "best_fitness",
        "top10_mean",
        "top10_mean_b2000",
        "top100_mean",
        "top100_mean_b2000",
        "n_species",
        "inter_diversity",
        "intra_diversity",
        "silhouette",
        "n_evaluated",
        "llm_calls_recount",
        "wall_hours",
    ]
    stats = mannwhitney_table(rows, test_metrics)
    stats_csv = args.out_dir / "rq2_mannwhitney.csv"
    write_csv(
        stats_csv,
        stats,
        [
            "metric",
            "contrast",
            "n_a",
            "n_b",
            "median_a",
            "median_b",
            "U",
            "p",
            "holm_p",
            "cliffs_delta",
            "cliffs_ci_lo",
            "cliffs_ci_hi",
        ],
    )
    print(f"Wrote {stats_csv} ({len(stats)} rows)")

    tests = primary_top10_mwu(rows)
    tests_csv = args.out_dir / "rq2_mwu_top100.csv"
    write_csv(
        tests_csv,
        tests,
        [
            "contrast",
            "contrast_id",
            "n_a",
            "n_b",
            "median_a",
            "median_b",
            "U",
            "p",
            "holm_p",
            "cliffs_delta",
            "cliffs_ci_lo",
            "cliffs_ci_hi",
        ],
    )
    print(f"Wrote {tests_csv}")

    remaining_csv = args.out_dir / "rq2_remaining.csv"
    write_csv(remaining_csv, leftover, ["cell_id", "distance", "clustering", "seed", "status"])
    print(f"Wrote {remaining_csv}")

    status = {
        "n_expected": len(rows),
        "n_complete": done,
        "n_remaining": len(leftover),
        "remaining": leftover,
        "live_dir": str(args.live_dir),
        "inference": "unpaired_mannwhitney_permutation_all_complete",
        "upon_completion": [
            "Recompute Table tab:rq2-factorial-summary at n=10 independent runs per cell.",
            "Redraw Fig. fig:rq2 top-10 trajectories.",
        ],
    }
    status_path = args.out_dir / "rq2_status.json"
    status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")
    print(f"Wrote {status_path}")

    tex = args.notes_dir / "rq2_factorial_tables.tex"
    write_latex_summary(tex, rows, tests)
    print(f"Wrote {tex}")

    maybe_plot(rows, args.fig_dir)
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
