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
  - top_species_per_run.csv + top_species_summary.csv (highest-performing species within runs)
  - c3_top_species_by_labels_across_runs.csv + figures/c3_top_species_by_labels_across_runs.pdf
    (top-N species by max toxicity across pooled C2 cohorts; labels optional, any count)
  - figures/c3_prompt_embeddings_mds_by_species.pdf (RQ1-style landmark MDS + KNN: ToxSearch +
    ToxSearch-S sequential / 2w / 4w; prompt dedupe; marker size ~ toxicity; no RainbowPlus)

Run (from repo root):
  python experiments/comparison_results/c3_ppsn2026_species/c3_species_report.py
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

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


# --- Cross-run top species by peak toxicity (pooled C2) ----------------------

LNCS_WIDTH_IN = 160.0 / 25.4  # ~6.30 in; fits \begin{figure*} in LNCS double-column.

# Okabe–Ito-style qualitative colours (colour-blind friendly); shared by top-species boxplot + MDS.
TOP_SPECIES_QUALITATIVE = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#332288",
    "#000000",
    "#117733",
    "#882255",
)

# MDS pooled-embedding figure: one colour per methodology / execution mode (colour-blind friendly).
# C3 MDS map (aligned with RQ1 embedding_mds_landmark_map): four execution configurations only.
MDS_COHORT_ORDER = (
    "toxsearch",
    "toxsearch_s_seq",
    "toxsearch_s_2w",
    "toxsearch_s_4w",
)
MDS_COHORT_LABEL = {
    "toxsearch": "ToxSearch",
    "toxsearch_s_seq": "ToxSearch-S (sequential)",
    "toxsearch_s_2w": "ToxSearch-S (2 workers)",
    "toxsearch_s_4w": "ToxSearch-S (4 workers)",
}
# Tab10-style hues (same family as RQ1 toxsearch / toxsearch_s, extended for 2w/4w).
MDS_COHORT_COLOR = {
    "toxsearch": "#1f77b4",
    "toxsearch_s_seq": "#ff7f0e",
    "toxsearch_s_2w": "#2ca02c",
    "toxsearch_s_4w": "#9467bd",
}


def load_speciation_state(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Return the ``species`` mapping from ``speciation_state.json`` or ``{}``."""
    p = run_dir / "speciation_state.json"
    if not p.exists():
        return {}
    try:
        data = load_json(p)
    except Exception:
        return {}
    sp = data.get("species") if isinstance(data, dict) else None
    return sp if isinstance(sp, dict) else {}


def _per_species_toxicity_values(run_dir: Path) -> Dict[str, List[float]]:
    genomes = load_all_genomes(run_dir)
    vals: Dict[str, List[float]] = defaultdict(list)
    for g in genomes:
        if not is_valid_species_id(g.get("species_id")):
            continue
        t = get_toxicity(g)
        if t is None:
            continue
        vals[str(g["species_id"])].append(float(t))
    return vals


def _pub_species_y_tick_label(labels_display: Sequence[str], max_chars: int = 78) -> str:
    """Publication-style y tick: comma-joined c-TF-IDF tags only (no rank, run, or species ids)."""
    labs = [str(x).strip() for x in labels_display if str(x).strip()]
    if not labs:
        return "—"[:max_chars]
    body = ", ".join(labs)
    if len(body) <= max_chars:
        return body
    return body[: max(1, max_chars - 1)] + "…"


def build_top_species_by_labels_across_runs(
    run_paths: Sequence[Path],
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Pool species across the given runs and return the top-N (run, species) pairs by max toxicity.

    Each species is scored within its own run using all scored genomes in ``elites``/``reserves``.
    There is no minimum or fixed label count; ``labels_display`` comes from ``speciation_state.json``
    when present (any length).

    Each returned row matches the shape expected by ``plot_top_species_by_labels_across_runs``:
      - labels_display: list[str] from that species' state (possibly empty).
      - toxicities: per-genome toxicities for that species in that run only.
      - max_toxicity, mean_toxicity, n_genomes; n_runs and n_member_species are always 1.
      - contributors: singleton list with run_id, species_id, labels_order, max_toxicity, n_genomes.
    """
    candidates: List[Dict[str, Any]] = []
    for run_path in run_paths:
        state = load_speciation_state(run_path)
        if not state:
            continue
        per_genome = _per_species_toxicity_values(run_path)
        run_id = run_path.name
        for sid, v in state.items():
            if not isinstance(v, dict):
                continue
            tox = per_genome.get(str(sid), [])
            if not tox:
                continue
            labels = v.get("labels") or []
            labels_order = [str(x) for x in labels] if isinstance(labels, list) else []
            mt = float(max(tox))
            contrib = {
                "run_path": str(run_path),
                "run_id": run_id,
                "species_id": str(sid),
                "labels_order": labels_order,
                "max_toxicity": mt,
                "n_genomes": int(len(tox)),
            }
            candidates.append(
                {
                    "labels_display": list(labels_order),
                    "toxicities": [float(x) for x in tox],
                    "max_toxicity": mt,
                    "mean_toxicity": float(np.mean(tox)),
                    "n_genomes": int(len(tox)),
                    "n_runs": 1,
                    "n_member_species": 1,
                    "contributors": [contrib],
                }
            )

    candidates.sort(key=lambda r: float(r["max_toxicity"]), reverse=True)
    return candidates[: max(1, int(top_n))]


def plot_top_species_by_labels_across_runs(
    rows: Sequence[Dict[str, Any]],
    out_path: Path,
    n_runs_pooled: int,
) -> None:
    """Horizontal box + jitter: publication-style y-axis = rank + c-TF-IDF tags only (ids in CSV)."""
    if not rows:
        return
    n = len(rows)
    colors = [TOP_SPECIES_QUALITATIVE[i % len(TOP_SPECIES_QUALITATIVE)] for i in range(n)]

    fig_w_in = LNCS_WIDTH_IN
    fig_h_in = 0.62 + 0.44 * n
    fig, ax = plt.subplots(figsize=(fig_w_in, fig_h_in))

    positions = list(range(1, n + 1))
    data = [r["toxicities"] for r in rows]
    bp = ax.boxplot(
        data,
        positions=positions,
        vert=False,
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#1a1a1a", "linewidth": 1.15},
        whiskerprops={"color": "#555555", "linewidth": 0.85},
        capprops={"color": "#555555", "linewidth": 0.85},
        boxprops={"linewidth": 0.65},
    )
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_edgecolor("#2d2d2d")
        patch.set_alpha(0.88)

    rng = np.random.default_rng(0)
    for i, r in enumerate(rows):
        xs = np.asarray(r["toxicities"], dtype=float)
        if xs.size == 0:
            continue
        ys = positions[i] + rng.uniform(-0.18, 0.18, size=xs.size)
        ax.scatter(
            xs,
            ys,
            s=7.0,
            color=colors[i],
            edgecolor="#1f1f1f",
            linewidth=0.28,
            alpha=0.78,
            zorder=3,
        )

    tick_labels = [_pub_species_y_tick_label(r.get("labels_display") or []) for r in rows]
    ax.set_yticks(positions)
    ax.set_yticklabels(tick_labels, fontsize=8.0)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Toxicity (Perspective API score)", fontsize=9)
    ax.set_ylabel("c-TF-IDF tags (species annotation)", fontsize=9)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.38, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title(
        f"Top {n} species by peak toxicity (pooled C2 cohort, {n_runs_pooled} runs)",
        fontsize=9.5,
        loc="left",
        pad=10,
    )

    max_tick = max((len(s) for s in tick_labels), default=40)
    left_margin = min(0.56, 0.30 + max(0, (max_tick - 36)) * 0.0045)
    fig.subplots_adjust(left=left_margin, right=0.98, top=0.92, bottom=0.12)

    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def write_top_species_by_labels_across_runs(
    seq_runs: Sequence[Tuple[str, Path]],
    par2_runs: Sequence[Tuple[str, Path]],
    par4_runs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    fig_dir: Path,
    top_n: int = 10,
) -> Optional[Dict[str, Any]]:
    """Aggregate top-N (run, species) pairs by max toxicity across C2 cohorts and render the figure
    plus a sidecar CSV for traceability."""
    all_paths = [p for _, p in seq_runs] + [p for _, p in par2_runs] + [p for _, p in par4_runs]
    rows = build_top_species_by_labels_across_runs(all_paths, top_n=top_n)
    if not rows:
        return None
    fig_path = fig_dir / "c3_top_species_by_labels_across_runs.pdf"
    plot_top_species_by_labels_across_runs(rows, fig_path, n_runs_pooled=len(all_paths))

    csv_path = out_dir / "c3_top_species_by_labels_across_runs.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "rank",
                "run_id",
                "species_id",
                "n_labels",
                "labels",
                "max_toxicity",
                "mean_toxicity",
                "n_genomes",
            ]
        )
        for rank, r in enumerate(rows, start=1):
            c0 = (r.get("contributors") or [{}])[0]
            labs = r.get("labels_display") or []
            w.writerow(
                [
                    rank,
                    c0.get("run_id", ""),
                    c0.get("species_id", ""),
                    len(labs),
                    ", ".join(str(x) for x in labs),
                    f"{r['max_toxicity']:.6f}",
                    f"{r['mean_toxicity']:.6f}",
                    r["n_genomes"],
                ]
            )

    return {
        "figure": str(fig_path.relative_to(out_dir)),
        "csv": str(csv_path.relative_to(out_dir)),
        "n_runs_pooled": len(all_paths),
        "n_species_in_figure": len(rows),
    }


def canonicalize_prompt(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip())


def prompt_embedding_vec(genome: Dict[str, Any]) -> Optional[np.ndarray]:
    emb = genome.get("prompt_embedding")
    if isinstance(emb, list) and emb:
        try:
            v = np.asarray(emb, dtype=np.float32)
            if v.size and np.all(np.isfinite(v)):
                return v
        except Exception:
            return None
    return None


def load_c1_manifest_runs(method: str) -> List[Tuple[str, Path]]:
    if not C1_MANIFEST.exists():
        return []
    out: List[Tuple[str, Path]] = []
    with C1_MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("method") or "").strip() != method:
                continue
            rid = (row.get("run_id") or "").strip()
            rp = (row.get("run_path") or "").strip()
            if not rid or not rp:
                continue
            p = Path(rp)
            if method == "rainbow_plus":
                if p.suffix.lower() == ".jsonl" and p.exists():
                    out.append((rid, p))
            else:
                if (p / "EvolutionTracker.json").exists():
                    out.append((rid, p))
    return out


def load_toxsearch_run_genomes(run_dir: Path) -> List[Dict[str, Any]]:
    genomes: List[Dict[str, Any]] = []
    for fn in ("elites.json", "non_elites.json", "under_performing.json"):
        p = run_dir / fn
        if not p.exists():
            continue
        try:
            data = load_json(p)
        except Exception:
            continue
        if isinstance(data, list):
            genomes.extend(x for x in data if isinstance(x, dict))
    return genomes


def load_toxsearch_s_run_genomes_map(run_dir: Path) -> List[Dict[str, Any]]:
    """Same population files as RQ1 ``load_run_genomes(..., 'toxsearch_s')`` for embedding maps."""
    genomes: List[Dict[str, Any]] = []
    for fn in ("elites.json", "archive.json", "reserves.json"):
        p = run_dir / fn
        if not p.exists():
            continue
        try:
            data = load_json(p)
        except Exception:
            continue
        if isinstance(data, list):
            genomes.extend(x for x in data if isinstance(x, dict))
    return genomes


def _l2_normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.clip(norms, 1e-12, None)
    return mat / norms


def _merge_prompt_best_toxsearch_s(paths: Sequence[Path]) -> Dict[str, Tuple[float, Optional[np.ndarray]]]:
    """Canonical prompt -> (max toxicity, embedding) pooled across runs (RQ1 semantics)."""
    best: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}
    for run_path in paths:
        for g in load_toxsearch_s_run_genomes_map(run_path):
            p = canonicalize_prompt(g.get("prompt"))
            if not p:
                continue
            t = get_toxicity(g)
            if t is None:
                continue
            e = prompt_embedding_vec(g)
            if p not in best or float(t) > best[p][0]:
                best[p] = (float(t), e)
    return best


def _deterministic_sample_embedding_pts(
    pts: List[Tuple[str, np.ndarray, float]],
    target_n: int,
    keep_top_per_method_n: int,
) -> List[Tuple[str, np.ndarray, float]]:
    """Bias sample toward high-toxicity prompts per configuration (same structure as RQ1)."""
    if target_n <= 0 or len(pts) <= target_n:
        return pts
    rng = np.random.default_rng(0)
    kept: List[Tuple[str, np.ndarray, float]] = []
    for mm in MDS_COHORT_ORDER:
        subset = [t for t in pts if t[0] == mm]
        subset.sort(key=lambda x: x[2], reverse=True)
        kept.extend(subset[: min(keep_top_per_method_n, len(subset))])
    seen = {(mm, float(tt), ee.tobytes()) for mm, ee, tt in kept}
    remaining = [t for t in pts if (t[0], float(t[2]), t[1].tobytes()) not in seen]
    need = target_n - len(kept)
    if need > 0 and remaining:
        idx = rng.choice(len(remaining), size=min(need, len(remaining)), replace=False)
        kept.extend(remaining[int(i)] for i in idx)
    return kept


def _count_genomes_with_toxicity_toxsearch_s(paths: Sequence[Path]) -> int:
    n = 0
    for run_path in paths:
        for g in load_toxsearch_s_run_genomes_map(run_path):
            if get_toxicity(g) is not None:
                n += 1
    return n


def _count_genomes_with_toxicity_toxsearch(runs: Sequence[Tuple[str, Path]]) -> int:
    n = 0
    for _rid, rpath in runs:
        for g in load_toxsearch_run_genomes(rpath):
            if get_toxicity(g) is not None:
                n += 1
    return n


def collect_embedding_mds_points(
    seq_paths: Sequence[Path],
    par2_paths: Sequence[Path],
    par4_paths: Sequence[Path],
    toxsearch_runs: Sequence[Tuple[str, Path]],
    encoder: Any = None,
) -> Tuple[List[Tuple[str, np.ndarray, float]], Dict[str, Any]]:
    """RQ1-style: dedupe prompts per configuration, then list (method_key, embedding, max_tox).

    Each plotted point is one **canonical prompt** (text) with the **maximum** toxicity seen
    for that prompt across runs in that configuration. That is far fewer than ``7 runs ×
    budget`` genome rows because prompts repeat within and across runs; compare
    ``genomes_with_toxicity_by_cohort`` vs ``n_by_cohort`` in the summary JSON.
    """
    meta: Dict[str, Any] = {"encoder_used": False, "toxsearch_prompts_no_embedding": 0}

    best_ts: Dict[str, Tuple[float, Optional[np.ndarray]]] = {}
    for _rid, rpath in toxsearch_runs:
        for g in load_toxsearch_run_genomes(rpath):
            p = canonicalize_prompt(g.get("prompt"))
            if not p:
                continue
            t = get_toxicity(g)
            if t is None:
                continue
            e = prompt_embedding_vec(g)
            if p not in best_ts or float(t) > best_ts[p][0]:
                best_ts[p] = (float(t), e)

    best_seq = _merge_prompt_best_toxsearch_s(seq_paths)
    best_2w = _merge_prompt_best_toxsearch_s(par2_paths)
    best_4w = _merge_prompt_best_toxsearch_s(par4_paths)

    all_missing: List[str] = []
    for best in (best_ts, best_seq, best_2w, best_4w):
        all_missing.extend([p for p, (_t, e) in best.items() if e is None])
    all_missing = list(dict.fromkeys(all_missing))

    if all_missing and encoder is not None:
        vecs = encoder.encode(
            all_missing,
            batch_size=64,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        fill = {p: np.asarray(v, dtype=np.float32) for p, v in zip(all_missing, vecs)}
        for best in (best_ts, best_seq, best_2w, best_4w):
            for p in list(best.keys()):
                tox, e = best[p]
                if e is None and p in fill:
                    best[p] = (tox, fill[p])
        meta["encoder_used"] = True

    for p, (tox, e) in list(best_ts.items()):
        if e is None:
            meta["toxsearch_prompts_no_embedding"] = int(meta["toxsearch_prompts_no_embedding"]) + 1

    all_pts: List[Tuple[str, np.ndarray, float]] = []
    for mkey, best in (
        ("toxsearch", best_ts),
        ("toxsearch_s_seq", best_seq),
        ("toxsearch_s_2w", best_2w),
        ("toxsearch_s_4w", best_4w),
    ):
        for _p, (tox, e) in best.items():
            if e is None:
                continue
            all_pts.append((mkey, np.asarray(e, dtype=np.float32), float(tox)))

    meta["genomes_with_toxicity_by_cohort"] = {
        "toxsearch": _count_genomes_with_toxicity_toxsearch(toxsearch_runs),
        "toxsearch_s_seq": _count_genomes_with_toxicity_toxsearch_s(seq_paths),
        "toxsearch_s_2w": _count_genomes_with_toxicity_toxsearch_s(par2_paths),
        "toxsearch_s_4w": _count_genomes_with_toxicity_toxsearch_s(par4_paths),
    }

    counts: Dict[str, int] = defaultdict(int)
    for m, _e, _t in all_pts:
        counts[m] += 1
    meta["n_by_cohort"] = dict(counts)
    meta["n_points"] = len(all_pts)
    return all_pts, meta


def plot_prompt_embeddings_mds(
    all_pts: Sequence[Tuple[str, np.ndarray, float]],
    out_path: Path,
    landmark_n: int = 2000,
    keep_top_per_method: int = 500,
) -> None:
    """Landmark MDS + KNN extension, matching RQ1 ``embedding_mds_landmark_map`` (marker size ~ toxicity)."""
    if len(all_pts) < 3:
        return
    from sklearn.manifold import MDS
    from sklearn.neighbors import KNeighborsRegressor

    pts_for_fit = list(all_pts)
    ln_cap = min(int(landmark_n), len(pts_for_fit))
    if ln_cap < len(pts_for_fit):
        pts_for_fit = _deterministic_sample_embedding_pts(
            list(all_pts), ln_cap, keep_top_per_method_n=keep_top_per_method
        )

    X_fit = np.stack([np.asarray(e, dtype=np.float64) for _m, e, _t in pts_for_fit], axis=0)
    X_fit = _l2_normalize_rows(X_fit)
    X_all = np.stack([np.asarray(e, dtype=np.float64) for _m, e, _t in all_pts], axis=0)
    X_all = _l2_normalize_rows(X_all)

    mds = MDS(
        n_components=2,
        random_state=0,
        normalized_stress="auto",
        dissimilarity="euclidean",
        n_init=1,
        max_iter=300,
        init="random",
    )
    z_land = mds.fit_transform(X_fit)
    knn = KNeighborsRegressor(n_neighbors=min(15, max(1, len(pts_for_fit) - 1)), weights="distance")
    knn.fit(X_fit, z_land)
    z_all = knn.predict(X_all).astype(np.float64)

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    ax.minorticks_off()
    ax.grid(False)

    legend_handles: List[Patch] = []
    legend_labels: List[str] = []
    for m in MDS_COHORT_ORDER:
        idx = [i for i, (mm, _e, _t) in enumerate(all_pts) if mm == m]
        if not idx:
            continue
        tox = np.asarray([all_pts[i][2] for i in idx], dtype=float)
        sizes = 6.0 + 50.0 * np.clip(tox, 0.0, 1.0)
        ax.scatter(
            z_all[idx, 0],
            z_all[idx, 1],
            s=sizes,
            c=MDS_COHORT_COLOR[m],
            alpha=0.28,
            linewidths=0,
            rasterized=True,
        )
        legend_handles.append(Patch(facecolor=MDS_COHORT_COLOR[m], edgecolor="none"))
        legend_labels.append(f"{MDS_COHORT_LABEL[m]} (n={len(idx)})")

    ax.set_xlabel("MDS-1", fontsize=9)
    ax.set_ylabel("MDS-2", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.set_title(
        "Prompt embeddings (landmark MDS + KNN)\n"
        "One point per canonical prompt (max toxicity across runs); marker size ∝ toxicity",
        fontsize=9,
        loc="left",
    )

    if legend_handles:
        ax.legend(handles=legend_handles, labels=legend_labels, frameon=False, loc="best", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_prompt_embedding_mds_five_conditions(
    seq_runs: Sequence[Tuple[str, Path]],
    par2_runs: Sequence[Tuple[str, Path]],
    par4_runs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    fig_dir: Path,
) -> Optional[Dict[str, Any]]:
    """ToxSearch + ToxSearch-S (seq / 2w / 4w) only; RQ1-style landmark MDS map (deduped prompts)."""
    try:
        from sentence_transformers import SentenceTransformer
    except Exception:
        SentenceTransformer = None  # type: ignore

    seq_paths = [p for _, p in seq_runs]
    par2_paths = [p for _, p in par2_runs]
    par4_paths = [p for _, p in par4_runs]

    tox_runs = load_c1_manifest_runs("toxsearch")

    encoder = None
    if SentenceTransformer is not None and (tox_runs or seq_paths or par2_paths or par4_paths):
        encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    all_pts, meta = collect_embedding_mds_points(
        seq_paths,
        par2_paths,
        par4_paths,
        tox_runs,
        encoder=encoder,
    )
    if len(all_pts) < 3:
        return None

    out_pdf = fig_dir / "c3_prompt_embeddings_mds_by_species.pdf"
    plot_prompt_embeddings_mds(all_pts, out_pdf)
    summary_path = out_dir / "c3_prompt_embeddings_mds_summary.json"
    summary = {
        "figure": str(out_pdf.relative_to(out_dir)),
        "n_points": meta.get("n_points"),
        "encoder_used": meta.get("encoder_used"),
        "toxsearch_prompts_no_embedding": meta.get("toxsearch_prompts_no_embedding"),
        "genomes_with_toxicity_by_cohort": meta.get("genomes_with_toxicity_by_cohort"),
        "note": (
            "n_by_cohort = distinct canonical prompts per configuration (max toxicity across runs); "
            "genomes_with_toxicity_by_cohort = all genome rows with a toxicity score (includes repeats)."
        ),
        "n_by_cohort": meta.get("n_by_cohort"),
        "n_toxsearch_runs": len(tox_runs),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# =============================================================================
# 3D MDS-GDP projection (seq / 2w / 4w) with generation as the vertical axis
# =============================================================================

MDS_GDP_COHORT_ORDER = ("seq", "par2", "par4")
MDS_GDP_COHORT_LABEL = {
    "seq": "ToxSearch-S (sequential)",
    "par2": "ToxSearch-S (2 workers)",
    "par4": "ToxSearch-S (4 workers)",
}
MDS_GDP_COHORT_COLOR = {
    "seq": "#ff7f0e",
    "par2": "#2ca02c",
    "par4": "#9467bd",
}


def _merge_prompt_best_toxsearch_s_with_generation(
    paths: Sequence[Path],
) -> Dict[str, Tuple[float, Optional[np.ndarray], Optional[int]]]:
    """Canonical prompt -> (max toxicity, embedding, generation of that max) across runs.

    Same dedupe semantics as ``_merge_prompt_best_toxsearch_s`` but also carries the generation
    at which the kept (max-toxicity) genome was evaluated, which is required for the 3D
    MDS-GDP view.
    """
    best: Dict[str, Tuple[float, Optional[np.ndarray], Optional[int]]] = {}
    for run_path in paths:
        for g in load_toxsearch_s_run_genomes_map(run_path):
            p = canonicalize_prompt(g.get("prompt"))
            if not p:
                continue
            t = get_toxicity(g)
            if t is None:
                continue
            e = prompt_embedding_vec(g)
            gen_v = g.get("generation")
            try:
                gen_i: Optional[int] = int(gen_v) if gen_v is not None else None
            except Exception:
                gen_i = None
            if p not in best or float(t) > best[p][0]:
                best[p] = (float(t), e, gen_i)
    return best


def collect_mds_gdp_3d_points(
    seq_paths: Sequence[Path],
    par2_paths: Sequence[Path],
    par4_paths: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Per-cohort deduplicated prompts with (cohort, embedding, toxicity, generation).

    Prompts are canonicalised and deduplicated **within** each cohort (a prompt seen in
    multiple runs of the same cohort contributes a single point with the run-level max
    toxicity; the generation carried is the generation of the max-toxicity occurrence).
    Prompts without a usable embedding or without an integer ``generation`` are dropped.
    """
    best_seq = _merge_prompt_best_toxsearch_s_with_generation(seq_paths)
    best_2w = _merge_prompt_best_toxsearch_s_with_generation(par2_paths)
    best_4w = _merge_prompt_best_toxsearch_s_with_generation(par4_paths)

    points: List[Dict[str, Any]] = []
    dropped_no_emb = 0
    dropped_no_gen = 0
    for cohort_key, best in (("seq", best_seq), ("par2", best_2w), ("par4", best_4w)):
        for _p, (tox, emb, gen_i) in best.items():
            if emb is None:
                dropped_no_emb += 1
                continue
            if gen_i is None:
                dropped_no_gen += 1
                continue
            points.append(
                {
                    "cohort": cohort_key,
                    "emb": np.asarray(emb, dtype=np.float32),
                    "tox": float(tox),
                    "generation": int(gen_i),
                }
            )

    meta = {
        "n_points": len(points),
        "n_by_cohort": {
            ck: int(sum(1 for p in points if p["cohort"] == ck))
            for ck in ("seq", "par2", "par4")
        },
        "dropped_no_embedding": int(dropped_no_emb),
        "dropped_no_generation": int(dropped_no_gen),
    }
    return points, meta


def _cosine_mds_2d(embeddings: np.ndarray, random_state: int = 0) -> np.ndarray:
    """Cosine-MDS 2D reduction; mirrors ``src/utils/gdp_projection._reduce_using_cosine_mds``.

    Falls back to landmark MDS + nearest-neighbour projection for large inputs (>3000 points),
    since the full cosine pairwise-distance matrix becomes memory-heavy at that scale.
    """
    from sklearn.manifold import MDS
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.neighbors import KNeighborsRegressor

    n = int(embeddings.shape[0])
    if n < 2:
        return np.zeros((n, 2), dtype=np.float32)

    landmark_cap = 3000
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=FutureWarning)
        if n <= landmark_cap:
            D = cosine_distances(embeddings)
            mds = MDS(
                n_components=2,
                dissimilarity="precomputed",
                random_state=int(random_state),
                normalized_stress="auto",
                n_init=1,
                max_iter=300,
            )
            return mds.fit_transform(D).astype(np.float64)
        rng = np.random.default_rng(int(random_state))
        land_idx = np.sort(rng.choice(n, size=landmark_cap, replace=False))
        X_land = embeddings[land_idx]
        D_land = cosine_distances(X_land)
        mds = MDS(
            n_components=2,
            dissimilarity="precomputed",
            random_state=int(random_state),
            normalized_stress="auto",
            n_init=1,
            max_iter=300,
        )
        z_land = mds.fit_transform(D_land).astype(np.float64)
        knn = KNeighborsRegressor(n_neighbors=min(15, max(1, X_land.shape[0] - 1)), weights="distance")
        knn.fit(X_land, z_land)
        return knn.predict(embeddings).astype(np.float64)


def write_mds_gdp_3d_by_cohort(
    seq_runs: Sequence[Tuple[str, Path]],
    par2_runs: Sequence[Tuple[str, Path]],
    par4_runs: Sequence[Tuple[str, Path]],
    out_dir: Path,
    fig_dir: Path,
) -> Optional[Dict[str, Any]]:
    """3D MDS-GDP: cosine-MDS of pooled ToxSearch-S prompt embeddings for seq/2w/4w.

    Axes: (MDS-1, MDS-2, generation). Colour encodes cohort (seq/2w/4w), marker size is
    proportional to toxicity (size $\\propto$ tox). Points are deduplicated per cohort by
    canonical prompt (max-toxicity entry kept). The three cohorts are fit in a **single**
    cosine-MDS pass so their coordinates live in the same plane and are directly comparable.
    """
    seq_paths = [p for _, p in seq_runs]
    par2_paths = [p for _, p in par2_runs]
    par4_paths = [p for _, p in par4_runs]

    points, meta = collect_mds_gdp_3d_points(seq_paths, par2_paths, par4_paths)
    if len(points) < 3:
        return None

    X = np.stack([p["emb"].astype(np.float64) for p in points], axis=0)
    Xn = _l2_normalize_rows(X)
    z2 = _cosine_mds_2d(Xn, random_state=0)

    gens = np.asarray([p["generation"] for p in points], dtype=float)
    toxs = np.asarray([p["tox"] for p in points], dtype=float)
    cohorts = [p["cohort"] for p in points]

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (side-effect: registers 3d projection)

    fig = plt.figure(figsize=(LNCS_WIDTH_IN, LNCS_WIDTH_IN * 0.82))
    ax = fig.add_subplot(111, projection="3d")

    sizes = 6.0 + 90.0 * np.clip(toxs, 0.0, 1.0)

    for ck in MDS_GDP_COHORT_ORDER:
        idx = [i for i, c in enumerate(cohorts) if c == ck]
        if not idx:
            continue
        ax.scatter(
            z2[idx, 0],
            z2[idx, 1],
            gens[idx],
            c=MDS_GDP_COHORT_COLOR[ck],
            s=sizes[idx],
            marker="o",
            alpha=0.28,
            linewidths=0.0,
            label=f"{MDS_GDP_COHORT_LABEL[ck]} (n={len(idx)})",
            depthshade=False,
            rasterized=True,
        )

    ax.set_xlabel("MDS-1 (cosine)", fontsize=8.5, labelpad=2)
    ax.set_ylabel("MDS-2 (cosine)", fontsize=8.5, labelpad=2)
    ax.set_zlabel("Generation", fontsize=8.5, labelpad=2)
    ax.tick_params(axis="x", labelsize=7)
    ax.tick_params(axis="y", labelsize=7)
    ax.tick_params(axis="z", labelsize=7)
    try:
        ax.view_init(elev=22, azim=-55)
    except Exception:
        pass

    ax.set_title(
        "MDS-GDP 3D (cosine MDS of prompt embeddings; z = generation; size $\\propto$ toxicity)",
        fontsize=9,
        loc="left",
    )
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(0.0, 0.98))

    out_pdf = fig_dir / "c3_mds_gdp_3d_by_cohort.pdf"
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close(fig)

    csv_path = out_dir / "c3_mds_gdp_3d_points.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "mds1", "mds2", "generation", "toxicity"])
        for (x1, x2), pt in zip(z2.tolist(), points):
            w.writerow(
                [
                    pt["cohort"],
                    f"{float(x1):.6f}",
                    f"{float(x2):.6f}",
                    int(pt["generation"]),
                    f"{float(pt['tox']):.6f}",
                ]
            )

    html_rel: Optional[str] = None
    try:
        html_path = fig_dir / "c3_mds_gdp_3d_by_cohort_interactive.html"
        ok = _write_mds_gdp_3d_interactive_html(
            points,
            z2,
            html_path,
        )
        if ok:
            html_rel = str(html_path.relative_to(out_dir))
    except Exception:
        html_rel = None

    summary = {
        "figure": str(out_pdf.relative_to(out_dir)),
        "figure_interactive": html_rel,
        "csv": str(csv_path.relative_to(out_dir)),
        "n_points": int(meta["n_points"]),
        "n_by_cohort": meta["n_by_cohort"],
        "dropped_no_embedding": meta["dropped_no_embedding"],
        "dropped_no_generation": meta["dropped_no_generation"],
        "method": "cosine-MDS (landmark+KNN for n>3000)",
    }
    (out_dir / "c3_mds_gdp_3d_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _write_mds_gdp_3d_interactive_html(
    points: Sequence[Dict[str, Any]],
    z2: np.ndarray,
    out_html: Path,
) -> bool:
    """Interactive plotly HTML twin of the 3D MDS-GDP figure.

    Mirrors ``src/utils/gdp_projection.generate_gdp_3d_plotly_unified`` but is adapted to the
    C3 vocabulary: colour modes are **cohort** (seq/2w/4w) / **toxicity** / **generation**,
    Z-axis dropdown switches between Generation and Toxicity, and marker-size / opacity have
    their own dropdowns. The ``size ~ toxicity`` default that the user asked for is applied
    on load; the size dropdown exposes flat 2/4/6/8 px variants as well as the toxicity-scaled
    option.
    """
    try:
        import plotly.graph_objects as go
    except Exception:
        return False
    if z2.shape[0] != len(points) or len(points) == 0:
        return False

    xs = z2[:, 0].astype(float)
    ys = z2[:, 1].astype(float)
    gens = np.asarray([p["generation"] for p in points], dtype=float)
    toxs = np.asarray([p["tox"] for p in points], dtype=float)
    cohorts_str = [p["cohort"] for p in points]

    cohort_colour = {
        "seq": MDS_GDP_COHORT_COLOR["seq"],
        "par2": MDS_GDP_COHORT_COLOR["par2"],
        "par4": MDS_GDP_COHORT_COLOR["par4"],
    }
    cohort_hex = [cohort_colour[ck] for ck in cohorts_str]

    tox_size = (4.0 + 14.0 * np.clip(toxs, 0.0, 1.0)).tolist()

    hover = [
        f"cohort: {MDS_GDP_COHORT_LABEL[ck]}<br>"
        f"toxicity: {float(t):.3f}<br>"
        f"generation: {int(g)}<br>"
        f"MDS-1: {float(x):.3f}<br>MDS-2: {float(y):.3f}"
        for ck, t, g, x, y in zip(cohorts_str, toxs, gens, xs, ys)
    ]

    def colour_mode(mode: str) -> Dict[str, Any]:
        if mode == "cohort":
            return {
                "marker.color": [cohort_hex],
                "marker.colorscale": [None],
                "marker.autocolorscale": False,
                "marker.showscale": False,
                "marker.colorbar": [None],
            }
        if mode == "toxicity":
            return {
                "marker.color": [toxs.tolist()],
                "marker.colorscale": "Plasma",
                "marker.autocolorscale": False,
                "marker.showscale": True,
                "marker.colorbar": [{"title": {"text": "Toxicity"}}],
            }
        if mode == "generation":
            return {
                "marker.color": [gens.tolist()],
                "marker.colorscale": "Viridis",
                "marker.autocolorscale": False,
                "marker.showscale": True,
                "marker.colorbar": [{"title": {"text": "Generation"}}],
            }
        raise ValueError(f"unknown colour mode: {mode}")

    def view_patch(z_mode: str, c_mode: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        z_arr = gens.tolist() if z_mode == "generation" else toxs.tolist()
        z_title = "Generation" if z_mode == "generation" else "Toxicity"
        tp: Dict[str, Any] = {"z": [z_arr]}
        tp.update(colour_mode(c_mode))
        lp = {"scene": {"zaxis": {"title": {"text": z_title}}}}
        return tp, lp

    view_modes: List[Tuple[str, str, str]] = [
        ("Z: Generation · colour: Cohort (seq/2w/4w)", "generation", "cohort"),
        ("Z: Generation · colour: Toxicity", "generation", "toxicity"),
        ("Z: Toxicity · colour: Cohort", "toxicity", "cohort"),
        ("Z: Toxicity · colour: Generation", "toxicity", "generation"),
    ]
    view_buttons = []
    for label, zm, cm in view_modes:
        tp, lp = view_patch(zm, cm)
        view_buttons.append({"label": label, "method": "update", "args": [tp, lp]})

    size_buttons = [
        {"label": "Size ~ toxicity (default)", "method": "restyle", "args": [{"marker.size": [tox_size]}, [0]]},
        {"label": "Flat 3 px", "method": "restyle", "args": [{"marker.size": 3}, [0]]},
        {"label": "Flat 5 px", "method": "restyle", "args": [{"marker.size": 5}, [0]]},
        {"label": "Flat 7 px", "method": "restyle", "args": [{"marker.size": 7}, [0]]},
        {"label": "Flat 10 px", "method": "restyle", "args": [{"marker.size": 10}, [0]]},
    ]

    opacity_buttons = [
        {"label": "Opacity 30%", "method": "restyle", "args": [{"marker.opacity": 0.3}, [0]]},
        {"label": "Opacity 50%", "method": "restyle", "args": [{"marker.opacity": 0.5}, [0]]},
        {"label": "Opacity 70% (default)", "method": "restyle", "args": [{"marker.opacity": 0.7}, [0]]},
        {"label": "Opacity 100%", "method": "restyle", "args": [{"marker.opacity": 1.0}, [0]]},
    ]

    legend_labels = [
        ("ToxSearch-S (sequential)", MDS_GDP_COHORT_COLOR["seq"], "seq"),
        ("ToxSearch-S (2 workers)", MDS_GDP_COHORT_COLOR["par2"], "par2"),
        ("ToxSearch-S (4 workers)", MDS_GDP_COHORT_COLOR["par4"], "par4"),
    ]

    data = [
        go.Scatter3d(
            x=xs,
            y=ys,
            z=gens,
            mode="markers",
            marker=dict(
                size=tox_size,
                color=cohort_hex,
                opacity=0.7,
                line=dict(width=0),
                showscale=False,
            ),
            text=hover,
            hoverinfo="text",
            name="points",
            showlegend=False,
        )
    ]
    for label, hex_col, ck in legend_labels:
        n_ck = int(sum(1 for c in cohorts_str if c == ck))
        data.append(
            go.Scatter3d(
                x=[None],
                y=[None],
                z=[None],
                mode="markers",
                marker=dict(size=8, color=hex_col, opacity=0.9),
                name=f"{label} (n={n_ck})",
                showlegend=True,
                hoverinfo="skip",
            )
        )

    fig = go.Figure(data=data)
    fig.update_layout(
        title=dict(
            text=(
                "MDS-GDP 3D · ToxSearch-S (sequential / 2 workers / 4 workers)<br>"
                "<sub>cosine MDS of prompt embeddings; z = generation; size ∝ toxicity</sub>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yref="paper",
            yanchor="top",
            font=dict(size=16, family="Arial, sans-serif"),
        ),
        margin=dict(t=48, l=8, r=120, b=8),
        scene=dict(
            xaxis=dict(title="MDS 1 (cosine)"),
            yaxis=dict(title="MDS 2 (cosine)"),
            zaxis=dict(title="Generation"),
            bgcolor="white",
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
            domain=dict(x=[0.24, 0.99], y=[0.02, 0.92]),
        ),
        legend=dict(
            x=0.24,
            y=-0.02,
            xanchor="left",
            yanchor="top",
            orientation="h",
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#cccccc",
            borderwidth=1,
        ),
        hovermode="closest",
        height=720,
        width=1100,
        template="plotly_white",
        font=dict(family="Arial, sans-serif", size=11),
        uirevision="c3_mds_gdp_3d",
        updatemenus=[
            dict(
                active=0,
                buttons=view_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.85,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="#cccccc",
                borderwidth=1,
                font=dict(size=11),
            ),
            dict(
                active=0,
                buttons=size_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.58,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="#cccccc",
                borderwidth=1,
                font=dict(size=11),
            ),
            dict(
                active=2,
                buttons=opacity_buttons,
                direction="down",
                showactive=True,
                x=0.02,
                xanchor="left",
                y=0.34,
                yanchor="top",
                bgcolor="rgba(255,255,255,0.96)",
                bordercolor="#cccccc",
                borderwidth=1,
                font=dict(size=11),
            ),
        ],
        annotations=[
            dict(
                text="<b>View</b>",
                x=0.02,
                xref="paper",
                y=0.88,
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=11),
            ),
            dict(
                text="<b>Marker size</b>",
                x=0.02,
                xref="paper",
                y=0.61,
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=11),
            ),
            dict(
                text="<b>Marker opacity</b>",
                x=0.02,
                xref="paper",
                y=0.37,
                yref="paper",
                showarrow=False,
                xanchor="left",
                font=dict(size=11),
            ),
        ],
    )

    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_html), include_plotlyjs="inline", full_html=True)
    return True


# =============================================================================
# T5. Effect sizes (Cliff's delta + bootstrap 95% CI) for pairwise MWU contrasts
# =============================================================================


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta: (#a>b - #a<b) / (|a||b|); bounded in [-1, 1]. Ties contribute 0."""
    av = np.asarray([float(x) for x in a], dtype=float)
    bv = np.asarray([float(x) for x in b], dtype=float)
    if av.size == 0 or bv.size == 0:
        return float("nan")
    diff = av.reshape(-1, 1) - bv.reshape(1, -1)
    gt = float(np.sum(diff > 0))
    lt = float(np.sum(diff < 0))
    n = float(av.size * bv.size)
    return (gt - lt) / n


def cliffs_delta_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    random_state: int = 0,
) -> Tuple[float, float]:
    av = np.asarray([float(x) for x in a], dtype=float)
    bv = np.asarray([float(x) for x in b], dtype=float)
    if av.size == 0 or bv.size == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(int(random_state))
    deltas = np.empty(int(n_boot), dtype=float)
    for i in range(int(n_boot)):
        ai = rng.integers(0, av.size, size=av.size)
        bi = rng.integers(0, bv.size, size=bv.size)
        deltas[i] = cliffs_delta(av[ai].tolist(), bv[bi].tolist())
    lo = float(np.quantile(deltas, alpha / 2.0))
    hi = float(np.quantile(deltas, 1.0 - alpha / 2.0))
    return lo, hi


## (Deleted: species-level embedding map, label-jaccard heatmap, toxicity ridges, operator mix,
## toxicity coverage, lineage proxy, and MPI worker-attribution plotting.)


## (Deleted: label-Jaccard heatmap and associated helpers.)


## (Deleted: per-run toxicity ridges and representative-run selection helper.)


## (Deleted: operator-mix top-species plot and helpers.)


## (Deleted: toxicity-coverage plot and helper.)


## (Deleted: species lineage proxy plot.)


## (Deleted: MPI worker attribution plot and master_metrics loader.)


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
            delta = cliffs_delta(x, y) if (x and y) else float("nan")
            ci_lo, ci_hi = cliffs_delta_bootstrap_ci(x, y) if (x and y) else (float("nan"), float("nan"))
            comp_rows.append(
                {
                    "a": left,
                    "b": right,
                    "p_mwu": raw_ps[-1],
                    "cliffs_delta": float(delta) if not math.isnan(float(delta)) else float("nan"),
                    "cliffs_delta_ci_lo": float(ci_lo),
                    "cliffs_delta_ci_hi": float(ci_hi),
                }
            )

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
                    "cliffs_delta": row["cliffs_delta"],
                    "cliffs_delta_ci_lo": row["cliffs_delta_ci_lo"],
                    "cliffs_delta_ci_hi": row["cliffs_delta_ci_hi"],
                }
            )

    (OUT / "species_stats_summary.json").write_text(json.dumps(stats_summary, indent=2), encoding="utf-8")
    with (OUT / "species_stats_table.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "p_kruskal",
                "a",
                "b",
                "p_mwu",
                "p_holm",
                "cliffs_delta",
                "cliffs_delta_ci_lo",
                "cliffs_delta_ci_hi",
            ],
        )
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

    xrun_info = write_top_species_by_labels_across_runs(
        seq_runs, par2_runs, par4_runs, OUT, FIG, top_n=10
    )
    if xrun_info is not None:
        print(
            f"  cross-run top-species figure: {xrun_info['n_species_in_figure']} species "
            f"over {xrun_info['n_runs_pooled']} pooled C2 runs"
        )

    mds_info = write_prompt_embedding_mds_five_conditions(seq_runs, par2_runs, par4_runs, OUT, FIG)
    if mds_info is not None:
        print(
            f"  MDS prompt-embeddings figure: unique_prompts={mds_info.get('n_by_cohort')}, "
            f"genomes_w_tox={mds_info.get('genomes_with_toxicity_by_cohort')}, "
            f"encoder_used={mds_info.get('encoder_used')}"
        )

    gdp3d_info = write_mds_gdp_3d_by_cohort(seq_runs, par2_runs, par4_runs, OUT, FIG)
    if gdp3d_info is not None:
        print(
            f"  MDS-GDP 3D figure: {gdp3d_info['figure']}  "
            f"n_points={gdp3d_info['n_points']} ({gdp3d_info['n_by_cohort']})"
        )
        if gdp3d_info.get("figure_interactive"):
            print(f"  MDS-GDP 3D interactive: {gdp3d_info['figure_interactive']}")
        else:
            print("  MDS-GDP 3D interactive: skipped (plotly missing)")

    print(f"Wrote {OUT}")
    print(
        f"  C1 toxsearch_s runs: {len(c1_runs)}, C2 seq: {len(seq_runs)}, C2 2w: {len(par2_runs)}, C2 4w: {len(par4_runs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
