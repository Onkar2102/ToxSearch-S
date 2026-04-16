#!/usr/bin/env python3
"""
C3: Species / niches — PPSN2026 post-hoc analysis (no new runs).

Reads C1 manifest (toxsearch_s) and C2 seq vs 2-worker runs under data/outputs/ppsn2026/.

Writes:
  run_manifest.csv, species_metrics_per_run.csv (unique paths; cohorts c1 / c2 / c1;c2),
  c2_seq_vs_2w_species_stats.csv
  speciation_species_count_milestones_long.csv, c1/c2 speciation summary CSVs (milestones)
  figures/speciation_summary_table.pdf — single publication table (mean ± SD by condition)

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
from scipy.stats import mannwhitneyu

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
C1_MANIFEST = PROJ / "experiments" / "comparison_results" / "c1_ppsn2026_two_way" / "run_manifest.csv"

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
    par_runs: List[Tuple[str, Path]],
    out_path: Path,
) -> None:
    """
    Single publication table: metrics as rows, sequential vs parallel columns (mean ± SD).
    Sequential runs are the shared toxsearch-s corpus (C1 / C2 sequential).
    """
    seq_stats = [collect_per_run_speciation_stats(p) for _, p in seq_runs]
    par_stats = [collect_per_run_speciation_stats(p) for _, p in par_runs]
    if not seq_stats and not par_stats:
        return

    def col_seq(key: str, nd: int = 2) -> str:
        if not seq_stats:
            return "—"
        return mean_pm_sd([s.get(key) for s in seq_stats], nd=nd)

    def col_par(key: str, nd: int = 2) -> str:
        if not par_stats:
            return "—"
        return mean_pm_sd([s.get(key) for s in par_stats], nd=nd)

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
        cell_text.append([label, col_seq(key, nd=nd), col_par(key, nd=nd)])

    col_labels = [
        "Metric",
        "Sequential (toxsearch-s)",
        "Parallel (2 workers)",
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
    par_runs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    rows_out: List[Dict[str, Any]] = []
    groups: List[Tuple[str, List[Tuple[str, Path]]]] = [
        ("c1_toxsearch_s", c1_runs),
        ("c2_sequential", seq_runs),
        ("c2_parallel_2w", par_runs),
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
    par_runs: List[Tuple[str, Path]],
    out_path: Path,
    milestones: Sequence[int] = MILESTONES,
) -> None:
    s_mat = _series_matrix(seq_runs, milestones)
    p_mat = _series_matrix(par_runs, milestones)
    if not s_mat and not p_mat:
        return
    S = np.asarray(s_mat, dtype=float) if s_mat else np.empty((0, len(milestones)))
    P = np.asarray(p_mat, dtype=float) if p_mat else np.empty((0, len(milestones)))
    rows_out: List[Dict[str, Any]] = []
    for j, m in enumerate(milestones):
        ns, ms, q25s, q75s = _finite_col_stats(S, j)
        np_, mp, q25p, q75p = _finite_col_stats(P, j)
        rows_out.append(
            {
                "evaluated_genomes_milestone": m,
                "n_sequential": ns,
                "median_sequential": ms,
                "q25_sequential": q25s,
                "q75_sequential": q75s,
                "n_parallel_2w": np_,
                "median_parallel_2w": mp,
                "q25_parallel_2w": q25p,
                "q75_parallel_2w": q75p,
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
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows_out:
            w.writerow(row)


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
    par_runs = discover_c2_runs(PAR_ROOT, "parallel", 2)

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
    for run_id, p in par_runs:
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

    with (OUT / "run_manifest.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["cohort", "method", "run_id", "run_path", "run_mode", "num_workers"],
        )
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

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
    for run_id, p in par_runs:
        key = str(p.resolve())
        if key not in by_path:
            by_path[key] = build_row_c2(run_id, p, "parallel", 2)
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
    par_paths = {str(p.resolve()) for _, p in par_runs}
    seq_vals = [r for r in metrics if str(Path(r.run_path).resolve()) in seq_paths]
    par_vals = [r for r in metrics if str(Path(r.run_path).resolve()) in par_paths]

    stat_lines: List[Dict[str, Any]] = []
    pairs = [
        ("final_species_count", [r.final_species_count for r in seq_vals], [r.final_species_count for r in par_vals]),
        ("final_active_species", [r.final_active_species for r in seq_vals], [r.final_active_species for r in par_vals]),
        ("total_speciation_events_tracker", [r.total_speciation_events_tracker for r in seq_vals], [r.total_speciation_events_tracker for r in par_vals]),
        ("n_species_with_toxicity", [r.n_species_with_toxicity for r in seq_vals], [r.n_species_with_toxicity for r in par_vals]),
    ]
    for name, a, b in pairs:
        row: Dict[str, Any] = {
            "metric": name,
            "n_sequential": len(a),
            "n_parallel_2w": len(b),
            "median_sequential": float(np.median(a)) if a else "",
            "median_parallel_2w": float(np.median(b)) if b else "",
            "mann_whitney_p_two_sided": "",
        }
        if len(a) >= 1 and len(b) >= 1:
            _, p = mannwhitneyu(a, b, alternative="two-sided")
            row["mann_whitney_p_two_sided"] = float(p)
        stat_lines.append(row)

    with (OUT / "c2_seq_vs_2w_species_stats.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "n_sequential",
                "n_parallel_2w",
                "median_sequential",
                "median_parallel_2w",
                "mann_whitney_p_two_sided",
            ],
        )
        w.writeheader()
        for row in stat_lines:
            w.writerow(row)

    write_speciation_milestone_long(
        c1_runs,
        seq_runs,
        par_runs,
        OUT / "speciation_species_count_milestones_long.csv",
    )
    write_summary_by_milestone(c1_runs, OUT / "c1_speciation_summary_by_milestone.csv")
    write_c2_two_group_summary_by_milestone(
        seq_runs,
        par_runs,
        OUT / "c2_speciation_summary_by_milestone.csv",
    )

    write_speciation_summary_table_pdf(seq_runs, par_runs, FIG / "speciation_summary_table.pdf")

    print(f"Wrote {OUT}")
    print(f"  C1 toxsearch_s runs: {len(c1_runs)}, C2 seq: {len(seq_runs)}, C2 2w: {len(par_runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
