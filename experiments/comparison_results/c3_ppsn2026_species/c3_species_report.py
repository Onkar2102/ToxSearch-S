#!/usr/bin/env python3
"""
Species/speciation post-hoc analysis (no new runs).

Reads C1 toxsearch_s manifest and C2 sequential + MPI cohorts under data/outputs/ppsn2026/.

Writes under experiments/comparison_results/c3_ppsn2026_species/:
  - run_manifest.csv
  - species_metrics_per_run.csv
  - speciation_species_count_milestones_long.csv
  - c1_speciation_summary_by_milestone.csv
  - c2_speciation_summary_by_milestone.csv (seq vs 2w vs 4w)
  - species_stats_summary.json + species_stats_table.csv (3-group tests)
  - figures/speciation_summary_table.pdf (mean ± SD by condition)
  - figures/species_count_trajectory.pdf (median + IQR at milestones)
  - figures/speciation_outcomes_raincloud.pdf (final-generation structure proxies)

Run (from repo root):
  python experiments/comparison_results/c3_ppsn2026_species/c3_species_report.py
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kruskal, mannwhitneyu

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

from src.utils.matplotlib_embed_fonts import configure_matplotlib_embedded_fonts  # noqa: E402

configure_matplotlib_embedded_fonts()

DATA = PROJ / "data" / "outputs" / "ppsn2026"
SEQ_ROOT = DATA / "toxsearch_s"
PAR_ROOT = DATA / "toxsearch_s_2w"
PAR4_ROOT = DATA / "toxsearch_s_4w"
# Prefer the canonical 3-way manifest; fallback kept for older branches.
C1_MANIFEST = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_three_way" / "run_manifest.csv"

OUT = PROJ / "experiments" / "comparison_results" / "c3_ppsn2026_species"
FIG = OUT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["savefig.bbox"] = "tight"

MILESTONES = tuple(range(0, 1100, 100))


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def population_integrated_count(gen: Dict[str, Any]) -> int:
    vi = gen.get("variants_integrated")
    if vi is not None:
        return int(vi)
    if gen.get("total_evaluated") is not None:
        return int(gen["total_evaluated"])
    b = gen.get("budget") or {}
    if b.get("llm_calls") is not None:
        return int(b["llm_calls"])
    return 0


def stepwise_at(xs: Sequence[float], ys: Sequence[float], xq: float) -> float:
    cur = float(ys[0]) if ys else 0.0
    for x, y in zip(xs, ys):
        if float(x) <= float(xq):
            cur = float(y)
        else:
            break
    return cur


def discover_c2_runs(
    root: Path,
    expected_mode: str,
    expected_workers: int,
) -> List[Tuple[str, Path]]:
    out: List[Tuple[str, Path]] = []
    if not root.exists():
        return out
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        tpath = p / "EvolutionTracker.json"
        if not tpath.exists():
            continue
        t = load_json(tpath)
        st = (t.get("status") or "").lower()
        if st and st != "complete":
            continue
        rm = t.get("run_metadata") or {}
        mode = str(rm.get("run_mode") or "")
        nw = int(rm.get("num_workers") or 0)
        if mode != expected_mode or nw != expected_workers:
            continue
        out.append((p.name, p))
    return out


def load_c1_toxsearch_s_paths() -> List[Tuple[str, Path]]:
    if not C1_MANIFEST.exists():
        return []
    rows: List[Tuple[str, Path]] = []
    with C1_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("method") or "").strip() != "toxsearch_s":
                continue
            rid = (row.get("run_id") or "").strip()
            rp = (row.get("run_path") or "").strip()
            if not rid or not rp:
                continue
            p = Path(rp)
            if (p / "EvolutionTracker.json").exists():
                rows.append((rid, p))
    return rows


def speciation_state_summary(run_dir: Path) -> Dict[str, Any]:
    p = run_dir / "speciation_state.json"
    if not p.exists():
        return {}
    try:
        s = load_json(p)
    except Exception:
        return {}
    m = s.get("metrics") or {}
    summ = m.get("summary") or {}
    return {
        "final_species_count_state": summ.get("final_species_count"),
        "total_speciation_events_state": summ.get("total_speciation_events"),
        "total_merge_events_state": summ.get("total_merge_events"),
        "total_extinction_events_state": summ.get("total_extinction_events"),
    }


def species_tracker_final(tracker: Dict[str, Any]) -> Tuple[int, int, int, int]:
    """Last generation: species_count, active_species_count, n_generations, total_generations field."""
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return 0, 0, 0, int(tracker.get("total_generations") or 0)
    last = gens[-1]
    sp = last.get("speciation") or {}
    sc = int(sp.get("species_count") or 0)
    ac = int(sp.get("active_species_count") or 0)
    return sc, ac, len(gens), int(tracker.get("total_generations") or len(gens))


def cumulative_events_from_tracker(tracker: Dict[str, Any]) -> Tuple[int, int, int]:
    """Sum per-generation speciation block event counts."""
    sm = me = xe = 0
    for g in tracker.get("generations") or []:
        sp = g.get("speciation") or {}
        sm += int(sp.get("speciation_events") or 0)
        me += int(sp.get("merge_events") or 0)
        xe += int(sp.get("extinction_events") or 0)
    return sm, me, xe


def last_generation_speciation_extras(tracker: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """Last generation: cluster_quality + species diversity from speciation block."""
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return {
            "silhouette": None,
            "davies_bouldin": None,
            "calinski_harabasz": None,
            "inter_species_diversity": None,
            "intra_species_diversity": None,
        }
    sp = gens[-1].get("speciation") or {}
    cq = sp.get("cluster_quality") or {}

    def to_f(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)) and not math.isnan(float(v)):
            return float(v)
        return None

    return {
        "silhouette": to_f(cq.get("silhouette_score")),
        "davies_bouldin": to_f(cq.get("davies_bouldin_index")),
        "calinski_harabasz": to_f(cq.get("calinski_harabasz_index")),
        "inter_species_diversity": to_f(sp.get("inter_species_diversity")),
        "intra_species_diversity": to_f(sp.get("intra_species_diversity")),
    }


def collect_per_run_speciation_stats(run_dir: Path) -> Dict[str, Any]:
    """Fields used by speciation_summary_table.pdf only."""
    t = load_json(run_dir / "EvolutionTracker.json")
    sc, ac, _, _ = species_tracker_final(t)
    ex = last_generation_speciation_extras(t)
    return {
        "final_species_count": float(sc),
        "final_active_species": float(ac),
        "silhouette": ex.get("silhouette"),
        "davies_bouldin": ex.get("davies_bouldin"),
        "calinski_harabasz": ex.get("calinski_harabasz"),
        "inter_species_diversity": ex.get("inter_species_diversity"),
        "intra_species_diversity": ex.get("intra_species_diversity"),
    }


def mean_pm_sd(vals: Sequence[Optional[float]], nd: int = 2) -> str:
    arr = [
        float(v)
        for v in vals
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not arr:
        return "—"
    m = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return f"{m:.{nd}f} ± {sd:.{nd}f}"


def write_speciation_summary_table_pdf(
    seq_runs: List[Tuple[str, Path]],
    par2_runs: List[Tuple[str, Path]],
    par4_runs: List[Tuple[str, Path]],
    out_path: Path,
) -> None:
    """
    Single publication table: metrics as rows, sequential vs MPI columns (mean ± SD).
    Sequential runs are the shared toxsearch-s corpus (C1 / C2 sequential).
    """
    seq_stats = [collect_per_run_speciation_stats(p) for _, p in seq_runs]
    par2_stats = [collect_per_run_speciation_stats(p) for _, p in par2_runs]
    par4_stats = [collect_per_run_speciation_stats(p) for _, p in par4_runs]
    if not seq_stats and not par2_stats and not par4_stats:
        return

    def col_seq(key: str, nd: int = 2) -> str:
        if not seq_stats:
            return "—"
        return mean_pm_sd([s.get(key) for s in seq_stats], nd=nd)

    def col_par2(key: str, nd: int = 2) -> str:
        if not par2_stats:
            return "—"
        return mean_pm_sd([s.get(key) for s in par2_stats], nd=nd)

    def col_par4(key: str, nd: int = 2) -> str:
        if not par4_stats:
            return "—"
        return mean_pm_sd([s.get(key) for s in par4_stats], nd=nd)

    rows = [
        ("Final species count", "final_species_count", 1),
        ("Active species", "final_active_species", 1),
        ("Silhouette", "silhouette", 3),
        ("Davies–Bouldin index", "davies_bouldin", 2),
        ("Calinski–Harabasz index", "calinski_harabasz", 1),
        ("Inter-species diversity", "inter_species_diversity", 3),
    ]

    cell_text = []
    for label, key, nd in rows:
        cell_text.append(
            [
                label,
                col_seq(key, nd=nd),
                col_par2(key, nd=nd),
                col_par4(key, nd=nd),
            ]
        )

    col_labels = [
        "Metric",
        "Sequential (toxsearch-s)",
        "Parallel (2 workers)",
        "Parallel (4 workers)",
    ]

    # Keep the figure tight: compact canvas + minimal padding on save.
    fig_h = 0.38 * (len(rows) + 1) + 0.35
    fig, ax = plt.subplots(figsize=(9.2, min(fig_h, 10.0)))
    ax.axis("off")
    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="left",
        loc="center",
        bbox=[0.01, 0.01, 0.98, 0.98],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_text_props(weight="bold")
            cell.set_height(0.055)
        else:
            cell.set_height(0.045)
        if c > 0:
            cell.get_text().set_horizontalalignment("center")
    tbl.scale(1, 1.15)
    plt.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()


def get_toxicity(genome: Dict[str, Any]) -> Optional[float]:
    if not isinstance(genome, dict):
        return None
    if genome.get("is_refusal") == 1:
        return None
    for key in ("toxicity", "north_star_score", "fitness"):
        v = genome.get(key)
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    try:
        v = genome.get("moderation_result", {}).get("google", {}).get("scores", {}).get("toxicity")
        if isinstance(v, (int, float)) and not math.isnan(v):
            return float(v)
    except Exception:
        pass
    return None


def is_valid_species_id(sid: Any) -> bool:
    if sid is None:
        return False
    sid_str = str(sid)
    return sid_str not in ("", "0", "-1", "None", "unknown")


def load_all_genomes(run_dir: Path) -> List[Dict[str, Any]]:
    genomes: List[Dict[str, Any]] = []
    for fname in ("elites.json", "reserves.json"):
        p = run_dir / fname
        if not p.exists():
            continue
        data = load_json(p)
        if isinstance(data, list):
            genomes.extend(data)
    return genomes


def per_species_toxicity_table(run_dir: Path) -> List[Dict[str, Any]]:
    """
    Build per-species toxicity summaries from elites+reserves.
    Returns rows with species_id, n_genomes, max_toxicity, mean_toxicity.
    """
    genomes = load_all_genomes(run_dir)
    by_sp: Dict[str, List[float]] = defaultdict(list)
    for g in genomes:
        if not is_valid_species_id(g.get("species_id")):
            continue
        t = get_toxicity(g)
        if t is None:
            continue
        by_sp[str(g["species_id"])].append(float(t))
    rows: List[Dict[str, Any]] = []
    for sid, vals in by_sp.items():
        if not vals:
            continue
        rows.append(
            {
                "species_id": sid,
                "n_genomes": int(len(vals)),
                "max_toxicity": float(max(vals)),
                "mean_toxicity": float(np.mean(vals)),
            }
        )
    rows.sort(key=lambda r: float(r["max_toxicity"]), reverse=True)
    return rows


def write_top_species_outputs(
    manifest_rows: List[Dict[str, Any]],
    out_dir: Path,
    top_k: int = 5,
) -> None:
    """
    Highest-performing species per run (by max_toxicity within species).
    Writes:
      - top_species_per_run.csv (top_k species per run)
      - top_species_summary.csv (per-run top1 and topK mean of max_toxicity)
    """
    top_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    for row in manifest_rows:
        run_id = str(row["run_id"])
        run_path = Path(str(row["run_path"]))
        genomes_path = run_path / "elites.json"
        if not genomes_path.exists():
            continue

        per_sp = per_species_toxicity_table(run_path)
        if not per_sp:
            continue

        top1 = per_sp[0]
        topk = per_sp[: max(1, int(top_k))]
        summary_rows.append(
            {
                "cohort": row["cohort"],
                "method": row["method"],
                "run_id": run_id,
                "run_mode": row["run_mode"],
                "num_workers": row["num_workers"],
                "n_species_with_toxicity": len(per_sp),
                "top1_species_id": top1["species_id"],
                "top1_max_toxicity": top1["max_toxicity"],
                "topk": int(top_k),
                "topk_mean_max_toxicity": float(np.mean([r["max_toxicity"] for r in topk])),
            }
        )

        for rank, r in enumerate(topk, start=1):
            top_rows.append(
                {
                    "cohort": row["cohort"],
                    "method": row["method"],
                    "run_id": run_id,
                    "run_mode": row["run_mode"],
                    "num_workers": row["num_workers"],
                    "rank_within_run": rank,
                    "species_id": r["species_id"],
                    "n_genomes": r["n_genomes"],
                    "max_toxicity": r["max_toxicity"],
                    "mean_toxicity": r["mean_toxicity"],
                }
            )

    if top_rows:
        with (out_dir / "top_species_per_run.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "cohort",
                    "method",
                    "run_id",
                    "run_mode",
                    "num_workers",
                    "rank_within_run",
                    "species_id",
                    "n_genomes",
                    "max_toxicity",
                    "mean_toxicity",
                ],
            )
            w.writeheader()
            for r in top_rows:
                w.writerow(r)

    if summary_rows:
        with (out_dir / "top_species_summary.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "cohort",
                    "method",
                    "run_id",
                    "run_mode",
                    "num_workers",
                    "n_species_with_toxicity",
                    "top1_species_id",
                    "top1_max_toxicity",
                    "topk",
                    "topk_mean_max_toxicity",
                ],
            )
            w.writeheader()
            for r in summary_rows:
                w.writerow(r)


def toxicity_species_summary(run_dir: Path) -> Dict[str, Optional[float]]:
    genomes = load_all_genomes(run_dir)
    by_sp: Dict[str, List[float]] = defaultdict(list)
    for g in genomes:
        if not is_valid_species_id(g.get("species_id")):
            continue
        t = get_toxicity(g)
        if t is None:
            continue
        by_sp[str(g["species_id"])].append(float(t))
    if not by_sp:
        return {
            "n_species_with_toxicity": 0,
            "mean_max_toxicity_per_species": None,
            "std_max_toxicity_per_species": None,
            "global_max_toxicity": None,
        }
    maxes = [max(v) for v in by_sp.values()]
    return {
        "n_species_with_toxicity": len(by_sp),
        "mean_max_toxicity_per_species": float(np.mean(maxes)),
        "std_max_toxicity_per_species": float(np.std(maxes, ddof=1)) if len(maxes) > 1 else 0.0,
        "global_max_toxicity": float(max(maxes)),
    }


def eval_vs_species_step_series(
    tracker: Dict[str, Any],
    milestones: Sequence[int],
) -> List[float]:
    gens = sorted(tracker.get("generations") or [], key=lambda g: int(g.get("generation_number", 0) or 0))
    if not gens:
        return [float("nan")] * len(milestones)
    xs: List[float] = []
    ys: List[float] = []
    cum = 0.0
    for g in gens:
        cum += float(population_integrated_count(g))
        sp = (g.get("speciation") or {}).get("species_count")
        if sp is None:
            continue
        xs.append(cum)
        ys.append(float(sp))
    if not xs:
        return [float("nan")] * len(milestones)
    pairs = sorted(zip(xs, ys), key=lambda z: float(z[0]))
    xs2 = [float(p[0]) for p in pairs]
    ys2 = [float(p[1]) for p in pairs]
    return [stepwise_at(xs2, ys2, float(m)) for m in milestones]


def _species_count_cell(sc: float) -> Any:
    v = float(sc)
    return "" if math.isnan(v) else v


def _series_matrix(
    runs: List[Tuple[str, Path]],
    milestones: Sequence[int],
) -> List[List[float]]:
    out: List[List[float]] = []
    for _, p in runs:
        t = load_json(p / "EvolutionTracker.json")
        out.append(eval_vs_species_step_series(t, milestones))
    return out


def write_speciation_milestone_long(
    c1_runs: List[Tuple[str, Path]],
    seq_runs: List[Tuple[str, Path]],
    par2_runs: List[Tuple[str, Path]],
    par4_runs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    rows_out: List[Dict[str, Any]] = []
    groups: List[Tuple[str, List[Tuple[str, Path]]]] = [
        ("c1_toxsearch_s", c1_runs),
        ("c2_sequential", seq_runs),
        ("c2_parallel_2w", par2_runs),
        ("c2_parallel_4w", par4_runs),
    ]
    for analysis_group, runs in groups:
        for run_id, p in runs:
            t = load_json(p / "EvolutionTracker.json")
            series = eval_vs_species_step_series(t, milestones)
            for m, sc in zip(milestones, series):
                rows_out.append(
                    {
                        "analysis_group": analysis_group,
                        "run_id": run_id,
                        "evaluated_genomes_milestone": m,
                        "species_count": _species_count_cell(sc),
                    }
                )
    if not rows_out:
        return
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["analysis_group", "run_id", "evaluated_genomes_milestone", "species_count"],
        )
        w.writeheader()
        for row in rows_out:
            w.writerow(row)


def write_summary_by_milestone(
    run_pairs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    """One group: median + IQR of species_count at each milestone."""
    mat = _series_matrix(run_pairs, milestones)
    if not mat:
        return
    R = np.asarray(mat, dtype=float)
    rows_out: List[Dict[str, Any]] = []
    for j, m in enumerate(milestones):
        col = R[:, j]
        finite = col[~np.isnan(col)]
        if finite.size == 0:
            rows_out.append(
                {
                    "evaluated_genomes_milestone": m,
                    "n_runs": 0,
                    "median_species_count": "",
                    "q25_species_count": "",
                    "q75_species_count": "",
                }
            )
        else:
            rows_out.append(
                {
                    "evaluated_genomes_milestone": m,
                    "n_runs": int(finite.size),
                    "median_species_count": float(np.median(finite)),
                    "q25_species_count": float(np.quantile(finite, 0.25)),
                    "q75_species_count": float(np.quantile(finite, 0.75)),
                }
            )
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "evaluated_genomes_milestone",
                "n_runs",
                "median_species_count",
                "q25_species_count",
                "q75_species_count",
            ],
        )
        w.writeheader()
        for row in rows_out:
            w.writerow(row)


def _finite_col_stats(mat: np.ndarray, col_idx: int) -> Tuple[int, str, str, str]:
    if mat.size == 0:
        return 0, "", "", ""
    col = mat[:, col_idx]
    finite = col[~np.isnan(col)]
    if finite.size == 0:
        return 0, "", "", ""
    return (
        int(finite.size),
        str(float(np.median(finite))),
        str(float(np.quantile(finite, 0.25))),
        str(float(np.quantile(finite, 0.75))),
    )


def write_c2_two_group_summary_by_milestone(
    seq_runs: List[Tuple[str, Path]],
    par2_runs: List[Tuple[str, Path]],
    par4_runs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    s_mat = _series_matrix(seq_runs, milestones)
    p2_mat = _series_matrix(par2_runs, milestones)
    p4_mat = _series_matrix(par4_runs, milestones)
    if not s_mat and not p2_mat and not p4_mat:
        return
    S = np.asarray(s_mat, dtype=float) if s_mat else np.empty((0, len(milestones)))
    P2 = np.asarray(p2_mat, dtype=float) if p2_mat else np.empty((0, len(milestones)))
    P4 = np.asarray(p4_mat, dtype=float) if p4_mat else np.empty((0, len(milestones)))
    rows_out: List[Dict[str, Any]] = []
    for j, m in enumerate(milestones):
        ns, ms, q25s, q75s = _finite_col_stats(S, j)
        n2, m2, q252, q752 = _finite_col_stats(P2, j)
        n4, m4, q254, q754 = _finite_col_stats(P4, j)
        rows_out.append(
            {
                "evaluated_genomes_milestone": m,
                "n_sequential": ns,
                "median_sequential": ms,
                "q25_sequential": q25s,
                "q75_sequential": q75s,
                "n_parallel_2w": n2,
                "median_parallel_2w": m2,
                "q25_parallel_2w": q252,
                "q75_parallel_2w": q752,
                "n_parallel_4w": n4,
                "median_parallel_4w": m4,
                "q25_parallel_4w": q254,
                "q75_parallel_4w": q754,
            }
        )
    fields = [
        "evaluated_genomes_milestone",
        "n_sequential",
        "median_sequential",
        "q25_sequential",
        "q75_sequential",
        "n_parallel_2w",
        "median_parallel_2w",
        "q25_parallel_2w",
        "q75_parallel_2w",
        "n_parallel_4w",
        "median_parallel_4w",
        "q25_parallel_4w",
        "q75_parallel_4w",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)


def plot_species_count_trajectory(
    seq_runs: List[Tuple[str, Path]],
    par2_runs: List[Tuple[str, Path]],
    par4_runs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    """Median + IQR of species_count vs evaluated genomes milestones for 3 cohorts."""

    def series_rows(runs: List[Tuple[str, Path]]) -> np.ndarray:
        mat = _series_matrix(runs, milestones)
        return np.asarray(mat, dtype=float) if mat else np.empty((0, len(milestones)))

    S = series_rows(seq_runs)
    P2 = series_rows(par2_runs)
    P4 = series_rows(par4_runs)
    if S.size == 0 or P2.size == 0 or P4.size == 0:
        return

    x = np.asarray(list(milestones), dtype=float)

    def med_iqr(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        med = np.nanmedian(A, axis=0)
        q1 = np.nanquantile(A, 0.25, axis=0)
        q3 = np.nanquantile(A, 0.75, axis=0)
        return med, q1, q3

    med_s, q1_s, q3_s = med_iqr(S)
    med_p2, q1_p2, q3_p2 = med_iqr(P2)
    med_p4, q1_p4, q3_p4 = med_iqr(P4)

    plt.figure(figsize=(7.2, 4.2))
    ax = plt.gca()
    ax.plot(x, med_s, linewidth=2.5, label="Sequential", color="#1f77b4")
    ax.fill_between(x, q1_s, q3_s, alpha=0.2, color="#1f77b4")
    ax.plot(x, med_p2, linewidth=2.5, label="Parallel (2w)", color="#ff7f0e")
    ax.fill_between(x, q1_p2, q3_p2, alpha=0.2, color="#ff7f0e")
    ax.plot(x, med_p4, linewidth=2.5, label="Parallel (4w)", color="#2ca02c")
    ax.fill_between(x, q1_p4, q3_p4, alpha=0.2, color="#2ca02c")
    ax.set_xlabel("Evaluated genomes")
    ax.set_ylabel("Species count")
    ax.set_xlim(0, 1000)
    ax.set_xticks(list(range(0, 1100, 100)))
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which="major", alpha=0.35)
    ax.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


def _raincloud_half_violin(ax: Any, data: List[List[float]], positions: List[float], colors: List[str]) -> None:
    vp = ax.violinplot(
        data,
        positions=positions,
        widths=0.86,
        showmeans=False,
        showextrema=False,
        showmedians=False,
    )
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_edgecolor("none")
        body.set_alpha(0.28)
        path = body.get_paths()[0]
        verts = path.vertices
        c = positions[i]
        verts[:, 0] = np.minimum(verts[:, 0], c)


def _raincloud_box(ax: Any, data: List[List[float]], positions: List[float]) -> None:
    bp = ax.boxplot(
        data,
        positions=positions,
        widths=0.18,
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "black", "linewidth": 1.3},
        boxprops={"linewidth": 0.9, "edgecolor": "black"},
        whiskerprops={"linewidth": 0.9, "color": "black"},
        capprops={"linewidth": 0.9, "color": "black"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.9)


def _raincloud_points(ax: Any, data: List[List[float]], positions: List[float], colors: List[str]) -> None:
    rng = np.random.default_rng(12345)
    for i, ys in enumerate(data):
        if not ys:
            continue
        x0 = positions[i]
        xs = x0 + 0.12 + rng.uniform(0.0, 0.22, size=len(ys))
        ax.scatter(
            xs,
            ys,
            s=22,
            c=colors[i],
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )


def plot_speciation_outcomes_raincloud(
    metrics: Sequence["SpeciesMetricRow"],
    seq_paths: set[str],
    par2_paths: set[str],
    par4_paths: set[str],
    out_path: Path,
) -> None:
    """Raincloud summary of final-generation structure proxies across 3 cohorts."""

    def select(metric_name: str, paths: set[str]) -> List[float]:
        vals: List[float] = []
        for r in metrics:
            if str(Path(r.run_path).resolve()) not in paths:
                continue
            v = getattr(r, metric_name)
            try:
                vals.append(float(v))
            except Exception:
                continue
        return vals

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    pos = [1.0, 2.0, 3.0]

    panels = [
        ("final_species_count", "Final species\ncount"),
        ("inter_species_diversity", "Inter-species\ndiversity"),
        ("intra_species_diversity", "Intra-species\ndiversity"),
    ]

    plt.figure(figsize=(8.4, 4.2))
    for j, (field, title) in enumerate(panels, start=1):
        ax = plt.subplot(1, 3, j)
        data = [
            select(field, seq_paths),
            select(field, par2_paths),
            select(field, par4_paths),
        ]
        _raincloud_half_violin(ax, data, pos, colors)
        _raincloud_box(ax, data, pos)
        _raincloud_points(ax, data, pos, colors)
        ax.set_title(title)
        ax.set_xticks(pos)
        ax.set_xticklabels(["S", "2w", "4w"])
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_xlim(0.5, 3.5)
        ax.set_ylim(bottom=0.0)

    plt.tight_layout()
    plt.savefig(out_path, format="pdf")
    plt.close()


@dataclass
class SpeciesMetricRow:
    cohorts: str
    run_id: str
    run_path: str
    run_mode: str
    num_workers: int
    n_generations: int
    final_species_count: int
    final_active_species: int
    silhouette: str
    davies_bouldin: str
    calinski_harabasz: str
    inter_species_diversity: str
    intra_species_diversity: str
    total_speciation_events_tracker: int
    total_merge_events_tracker: int
    total_extinction_events_tracker: int
    final_species_count_state: str
    total_speciation_events_state: str
    total_merge_events_state: str
    total_extinction_events_state: str
    n_species_with_toxicity: int
    mean_max_toxicity_per_species: str
    std_max_toxicity_per_species: str
    global_max_toxicity: str


def build_row_c1(run_id: str, run_dir: Path) -> SpeciesMetricRow:
    t = load_json(run_dir / "EvolutionTracker.json")
    rm = t.get("run_metadata") or {}
    mode = str(rm.get("run_mode") or "sequential")
    nw = int(rm.get("num_workers") or 1)
    sc, ac, ngen, _ = species_tracker_final(t)
    ev = cumulative_events_from_tracker(t)
    ex = last_generation_speciation_extras(t)
    st = speciation_state_summary(run_dir)
    tox = toxicity_species_summary(run_dir)

    def fmt(v: Any) -> str:
        return "" if v is None else str(v)

    return SpeciesMetricRow(
        cohorts="c1",
        run_id=run_id,
        run_path=str(run_dir),
        run_mode=mode,
        num_workers=nw,
        n_generations=ngen,
        final_species_count=sc,
        final_active_species=ac,
        silhouette=fmt(ex.get("silhouette")),
        davies_bouldin=fmt(ex.get("davies_bouldin")),
        calinski_harabasz=fmt(ex.get("calinski_harabasz")),
        inter_species_diversity=fmt(ex.get("inter_species_diversity")),
        intra_species_diversity=fmt(ex.get("intra_species_diversity")),
        total_speciation_events_tracker=ev[0],
        total_merge_events_tracker=ev[1],
        total_extinction_events_tracker=ev[2],
        final_species_count_state=fmt(st.get("final_species_count_state")),
        total_speciation_events_state=fmt(st.get("total_speciation_events_state")),
        total_merge_events_state=fmt(st.get("total_merge_events_state")),
        total_extinction_events_state=fmt(st.get("total_extinction_events_state")),
        n_species_with_toxicity=tox["n_species_with_toxicity"],
        mean_max_toxicity_per_species=fmt(tox.get("mean_max_toxicity_per_species")),
        std_max_toxicity_per_species=fmt(tox.get("std_max_toxicity_per_species")),
        global_max_toxicity=fmt(tox.get("global_max_toxicity")),
    )


def build_row_c2(
    run_id: str,
    run_dir: Path,
    label_mode: str,
    workers: int,
) -> SpeciesMetricRow:
    t = load_json(run_dir / "EvolutionTracker.json")
    sc, ac, ngen, _ = species_tracker_final(t)
    ev = cumulative_events_from_tracker(t)
    ex = last_generation_speciation_extras(t)
    st = speciation_state_summary(run_dir)
    tox = toxicity_species_summary(run_dir)

    def fmt(v: Any) -> str:
        return "" if v is None else str(v)

    return SpeciesMetricRow(
        cohorts="c2",
        run_id=run_id,
        run_path=str(run_dir),
        run_mode=label_mode,
        num_workers=workers,
        n_generations=ngen,
        final_species_count=sc,
        final_active_species=ac,
        silhouette=fmt(ex.get("silhouette")),
        davies_bouldin=fmt(ex.get("davies_bouldin")),
        calinski_harabasz=fmt(ex.get("calinski_harabasz")),
        inter_species_diversity=fmt(ex.get("inter_species_diversity")),
        intra_species_diversity=fmt(ex.get("intra_species_diversity")),
        total_speciation_events_tracker=ev[0],
        total_merge_events_tracker=ev[1],
        total_extinction_events_tracker=ev[2],
        final_species_count_state=fmt(st.get("final_species_count_state")),
        total_speciation_events_state=fmt(st.get("total_speciation_events_state")),
        total_merge_events_state=fmt(st.get("total_merge_events_state")),
        total_extinction_events_state=fmt(st.get("total_extinction_events_state")),
        n_species_with_toxicity=tox["n_species_with_toxicity"],
        mean_max_toxicity_per_species=fmt(tox.get("mean_max_toxicity_per_species")),
        std_max_toxicity_per_species=fmt(tox.get("std_max_toxicity_per_species")),
        global_max_toxicity=fmt(tox.get("global_max_toxicity")),
    )


def merge_cohort_labels(a: str, b: str) -> str:
    s = set(x for x in a.split(";") if x) | set(x for x in b.split(";") if x)
    return ";".join(sorted(s))


def row_to_dict(r: SpeciesMetricRow) -> Dict[str, Any]:
    return {
        "cohorts": r.cohorts,
        "run_id": r.run_id,
        "run_path": r.run_path,
        "run_mode": r.run_mode,
        "num_workers": r.num_workers,
        "n_generations": r.n_generations,
        "final_species_count": r.final_species_count,
        "final_active_species": r.final_active_species,
        "silhouette": r.silhouette,
        "davies_bouldin": r.davies_bouldin,
        "calinski_harabasz": r.calinski_harabasz,
        "inter_species_diversity": r.inter_species_diversity,
        "intra_species_diversity": r.intra_species_diversity,
        "total_speciation_events_tracker": r.total_speciation_events_tracker,
        "total_merge_events_tracker": r.total_merge_events_tracker,
        "total_extinction_events_tracker": r.total_extinction_events_tracker,
        "final_species_count_state": r.final_species_count_state,
        "total_speciation_events_state": r.total_speciation_events_state,
        "total_merge_events_state": r.total_merge_events_state,
        "total_extinction_events_state": r.total_extinction_events_state,
        "n_species_with_toxicity": r.n_species_with_toxicity,
        "mean_max_toxicity_per_species": r.mean_max_toxicity_per_species,
        "std_max_toxicity_per_species": r.std_max_toxicity_per_species,
        "global_max_toxicity": r.global_max_toxicity,
    }


def main() -> int:
    c1_runs = load_c1_toxsearch_s_paths()
    seq_runs = discover_c2_runs(SEQ_ROOT, "sequential", 1)
    par2_runs = discover_c2_runs(PAR_ROOT, "parallel", 2)
    par4_runs = discover_c2_runs(PAR4_ROOT, "parallel", 4)

    manifest_rows: List[Dict[str, Any]] = []
    for run_id, p in c1_runs:
        manifest_rows.append(
            {
                "cohort": "c1",
                "method": "toxsearch_s",
                "run_id": run_id,
                "run_path": str(p),
                "run_mode": "sequential",
                "num_workers": 1,
            }
        )
    for run_id, p in seq_runs:
        manifest_rows.append(
            {
                "cohort": "c2",
                "method": "toxsearch_s",
                "run_id": run_id,
                "run_path": str(p),
                "run_mode": "sequential",
                "num_workers": 1,
            }
        )
    for run_id, p in par2_runs:
        manifest_rows.append(
            {
                "cohort": "c2",
                "method": "toxsearch_s_2w",
                "run_id": run_id,
                "run_path": str(p),
                "run_mode": "parallel",
                "num_workers": 2,
            }
        )
    for run_id, p in par4_runs:
        manifest_rows.append(
            {
                "cohort": "c2",
                "method": "toxsearch_s_4w",
                "run_id": run_id,
                "run_path": str(p),
                "run_mode": "parallel",
                "num_workers": 4,
            }
        )

    with (OUT / "run_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cohort", "method", "run_id", "run_path", "run_mode", "num_workers"],
        )
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    # Highest-performing species per run (C3-only artifact; avoids duplicating C1/C2 run-level outcome plots).
    write_top_species_outputs(manifest_rows, OUT, top_k=5)

    by_path: Dict[str, SpeciesMetricRow] = {}
    for run_id, p in c1_runs:
        key = str(p.resolve())
        if key not in by_path:
            by_path[key] = build_row_c1(run_id, p)
        else:
            by_path[key].cohorts = merge_cohort_labels(by_path[key].cohorts, "c1")
    for run_id, p in seq_runs:
        key = str(p.resolve())
        if key not in by_path:
            by_path[key] = build_row_c2(run_id, p, "sequential", 1)
        else:
            by_path[key].cohorts = merge_cohort_labels(by_path[key].cohorts, "c2")
    for run_id, p in par2_runs:
        key = str(p.resolve())
        if key not in by_path:
            by_path[key] = build_row_c2(run_id, p, "parallel", 2)
        else:
            by_path[key].cohorts = merge_cohort_labels(by_path[key].cohorts, "c2")
    for run_id, p in par4_runs:
        key = str(p.resolve())
        if key not in by_path:
            by_path[key] = build_row_c2(run_id, p, "parallel", 4)
        else:
            by_path[key].cohorts = merge_cohort_labels(by_path[key].cohorts, "c2")
    metrics = list(by_path.values())

    if metrics:
        fieldnames = list(row_to_dict(metrics[0]).keys())
        with (OUT / "species_metrics_per_run.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in metrics:
                w.writerow(row_to_dict(r))

    seq_paths = {str(p.resolve()) for _, p in seq_runs}
    par2_paths = {str(p.resolve()) for _, p in par2_runs}
    par4_paths = {str(p.resolve()) for _, p in par4_runs}
    seq_vals = [r for r in metrics if str(Path(r.run_path).resolve()) in seq_paths]
    par2_vals = [r for r in metrics if str(Path(r.run_path).resolve()) in par2_paths]
    par4_vals = [r for r in metrics if str(Path(r.run_path).resolve()) in par4_paths]

    # --- 3-group stats: Kruskal–Wallis + pairwise MWU with Holm (per metric) ---
    metrics_order = [
        "final_species_count",
        "final_active_species",
        "total_speciation_events_tracker",
        "total_merge_events_tracker",
        "total_extinction_events_tracker",
        "n_species_with_toxicity",
        "mean_max_toxicity_per_species",
        "global_max_toxicity",
    ]

    def to_float_list(xs: List[Any]) -> List[float]:
        out: List[float] = []
        for v in xs:
            try:
                fv = float(v)
            except Exception:
                continue
            if math.isnan(fv):
                continue
            out.append(fv)
        return out

    def holm_adjust(pvals: List[float]) -> List[float]:
        m = len(pvals)
        order = sorted(range(m), key=lambda i: float(pvals[i]))
        ps = [float(pvals[i]) for i in order]
        adj_sorted: List[float] = []
        for j in range(m):
            adj_sorted.append(max(min(1.0, (m - k) * ps[k]) for k in range(j + 1)))
        out = [0.0] * m
        for j in range(m):
            out[order[j]] = adj_sorted[j]
        return out

    stats_summary: Dict[str, Any] = {"n_by_mode": {"sequential": len(seq_vals), "parallel_2w": len(par2_vals), "parallel_4w": len(par4_vals)}}
    stats_table_rows: List[Dict[str, Any]] = []
    for mname in metrics_order:
        a = to_float_list([getattr(r, mname) for r in seq_vals])
        b = to_float_list([getattr(r, mname) for r in par2_vals])
        c = to_float_list([getattr(r, mname) for r in par4_vals])
        kw_p = float("nan")
        if a and b and c:
            _, p = kruskal(a, b, c)
            kw_p = float(p)

        comps = [("sequential", "parallel_2w", a, b), ("sequential", "parallel_4w", a, c), ("parallel_2w", "parallel_4w", b, c)]
        raw_ps: List[float] = []
        comp_rows: List[Dict[str, Any]] = []
        for left, right, x, y in comps:
            if x and y:
                _, p = mannwhitneyu(x, y, alternative="two-sided")
                raw_ps.append(float(p))
            else:
                raw_ps.append(float("nan"))
            comp_rows.append({"a": left, "b": right, "p_mwu": raw_ps[-1]})

        valid_idx = [i for i, p in enumerate(raw_ps) if not math.isnan(float(p))]
        adj = [float("nan")] * len(raw_ps)
        if valid_idx:
            adj_vals = holm_adjust([raw_ps[i] for i in valid_idx])
            for i, ap in zip(valid_idx, adj_vals):
                adj[i] = float(ap)
        for i in range(len(comp_rows)):
            comp_rows[i]["p_holm"] = adj[i]

        stats_summary.setdefault("kruskal_p", {})[mname] = kw_p
        stats_summary.setdefault("pairwise_mwu", {})[mname] = {"comparisons": comp_rows}

        for row in comp_rows:
            stats_table_rows.append(
                {
                    "metric": mname,
                    "p_kruskal": kw_p,
                    "a": row["a"],
                    "b": row["b"],
                    "p_mwu": row["p_mwu"],
                    "p_holm": row["p_holm"],
                }
            )

    (OUT / "species_stats_summary.json").write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    with (OUT / "species_stats_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "p_kruskal", "a", "b", "p_mwu", "p_holm"])
        w.writeheader()
        for row in stats_table_rows:
            w.writerow(row)

    write_speciation_milestone_long(
        c1_runs,
        seq_runs,
        par2_runs,
        par4_runs,
        OUT / "speciation_species_count_milestones_long.csv",
    )
    write_summary_by_milestone(c1_runs, OUT / "c1_speciation_summary_by_milestone.csv")
    write_c2_two_group_summary_by_milestone(
        seq_runs,
        par2_runs,
        par4_runs,
        OUT / "c2_speciation_summary_by_milestone.csv",
    )

    write_speciation_summary_table_pdf(seq_runs, par2_runs, par4_runs, FIG / "speciation_summary_table.pdf")
    plot_species_count_trajectory(seq_runs, par2_runs, par4_runs, FIG / "species_count_trajectory.pdf")
    plot_speciation_outcomes_raincloud(metrics, seq_paths, par2_paths, par4_paths, FIG / "speciation_outcomes_raincloud.pdf")

    print(f"Wrote {OUT}")
    print(
        f"  C1 toxsearch_s runs: {len(c1_runs)}, C2 seq: {len(seq_runs)}, C2 2w: {len(par2_runs)}, C2 4w: {len(par4_runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
