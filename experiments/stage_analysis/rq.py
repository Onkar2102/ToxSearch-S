"""RQ1-RQ4 aggregation over the per-run metrics table.

Expected input: ``runs.index_runs`` merged with ``metrics.run_metrics`` (one row per run).
Within one oracle the design is distance (d_E, d_N, d_EP) x clustering (LF, IncDBSCAN)
with >= 5 seeds, plus a ``control`` cell. Oracles are compared only through
within-oracle statistics (ranks, standardized effects vs control), never raw fitness.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .metrics import load_final_population, load_snapshots
from .runs import PRIMARY_CLUSTERINGS, PRIMARY_DISTANCES

PRIMARY_OUTCOMES = ("Q_max", "top10_mean", "top50_mean", "N1", "K_topic")
RQ1_DIAGNOSTICS = ("canvas_separation_ratio", "forming_separation_ratio", "canvas_spread_max",
                   "speciation_seconds", "nli_model_passes", "wall_seconds")
RQ3_OUTCOMES = ("n_species", "reserve_size", "archive_size", "breedable_size", "largest_species_size",
                "canvas_spread_max", "canvas_spread_max_median_over_species", "canvas_separation_ratio")


def bootstrap_ci(x: Sequence[float], n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    x = np.asarray([v for v in x if v == v], dtype=float)
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, x.size), replace=True).mean(axis=1)
    return (float(x.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def cell_summary(df: pd.DataFrame, outcome: str, by=("oracle", "distance", "clustering")) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(list(by)):
        m, lo, hi = bootstrap_ci(g[outcome])
        rows.append(dict(zip(by, keys), outcome=outcome, n=int(g[outcome].notna().sum()), mean=m, ci_lo=lo, ci_hi=hi,
                         sd=float(g[outcome].std(ddof=1)) if len(g) > 1 else np.nan))
    return pd.DataFrame(rows)


def two_way_anova(df: pd.DataFrame, outcome: str, a: str = "distance", b: str = "clustering") -> pd.DataFrame:
    """Balanced two-way ANOVA (interaction first). Raises if the design is unbalanced."""
    d = df[[a, b, outcome]].dropna()
    counts = d.groupby([a, b]).size()
    if counts.nunique() != 1:
        raise ValueError(f"Unbalanced design for {outcome}: {counts.to_dict()}")
    n = int(counts.iloc[0])
    la, lb = sorted(d[a].unique()), sorted(d[b].unique())
    grand = d[outcome].mean()
    ma = d.groupby(a)[outcome].mean()
    mb = d.groupby(b)[outcome].mean()
    mab = d.groupby([a, b])[outcome].mean()
    ss_a = n * len(lb) * float(((ma - grand) ** 2).sum())
    ss_b = n * len(la) * float(((mb - grand) ** 2).sum())
    ss_ab = n * float(sum((mab[(i, j)] - ma[i] - mb[j] + grand) ** 2 for i in la for j in lb))
    ss_e = float(sum(((g[outcome] - g[outcome].mean()) ** 2).sum() for _, g in d.groupby([a, b])))
    df_a, df_b = len(la) - 1, len(lb) - 1
    df_ab, df_e = df_a * df_b, len(la) * len(lb) * (n - 1)
    ms_e = ss_e / df_e if df_e > 0 else np.nan
    rows = []
    for name, ss, dof in ((f"{a}x{b}", ss_ab, df_ab), (a, ss_a, df_a), (b, ss_b, df_b)):
        f = (ss / dof) / ms_e if dof and ms_e and ms_e > 0 else np.nan
        rows.append({"term": name, "SS": ss, "df": dof, "F": f,
                     "p": float(stats.f.sf(f, dof, df_e)) if f == f else np.nan,
                     "partial_eta2": ss / (ss + ss_e) if ss + ss_e > 0 else np.nan})
    rows.append({"term": "residual", "SS": ss_e, "df": df_e})
    return pd.DataFrame(rows)


def simple_effects(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Distance contrasts within each clustering level, and LF vs IncDBSCAN within each distance (Welch t)."""
    rows = []
    for cl, g in df.groupby("clustering"):
        for i, d1 in enumerate(PRIMARY_DISTANCES):
            for d2 in PRIMARY_DISTANCES[i + 1:]:
                x, y = g.loc[g.distance == d1, outcome].dropna(), g.loc[g.distance == d2, outcome].dropna()
                if len(x) > 1 and len(y) > 1:
                    t = stats.ttest_ind(x, y, equal_var=False)
                    rows.append({"within": cl, "contrast": f"{d1}-{d2}", "diff": x.mean() - y.mean(), "t": t.statistic, "p": t.pvalue})
    for dist, g in df.groupby("distance"):
        x, y = g.loc[g.clustering == "LF", outcome].dropna(), g.loc[g.clustering == "IncDBSCAN", outcome].dropna()
        if len(x) > 1 and len(y) > 1:
            t = stats.ttest_ind(x, y, equal_var=False)
            rows.append({"within": dist, "contrast": "LF-IncDBSCAN", "diff": x.mean() - y.mean(), "t": t.statistic, "p": t.pvalue})
    return pd.DataFrame(rows)


def standardized_effects(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """(cell mean - control mean) / pooled seed SD, per oracle. Requires a control cell."""
    rows = []
    for oracle, g in df.groupby("oracle"):
        ctrl = g.loc[g.distance == "control", outcome].dropna()
        if len(ctrl) < 2:
            continue
        for (dist, cl), c in g[g.distance != "control"].groupby(["distance", "clustering"]):
            x = c[outcome].dropna()
            if len(x) < 2:
                continue
            pooled = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(ctrl) - 1) * ctrl.var(ddof=1)) / (len(x) + len(ctrl) - 2))
            rows.append({"oracle": oracle, "distance": dist, "clustering": cl, "outcome": outcome,
                         "effect": (x.mean() - ctrl.mean()) / pooled if pooled > 0 else np.nan})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- RQ1 / RQ2

def rq1_rq2(df: pd.DataFrame, outcomes: Sequence[str] = PRIMARY_OUTCOMES) -> Dict[str, pd.DataFrame]:
    """Per oracle: interaction-first ANOVA, simple effects, cell summaries; RQ1 diagnostics & cost."""
    out: Dict[str, pd.DataFrame] = {}
    primary = df[df.distance.isin(PRIMARY_DISTANCES) & df.clustering.isin(PRIMARY_CLUSTERINGS)]
    summaries, anovas, simple = [], [], []
    for outcome in [o for o in outcomes if o in df.columns]:
        summaries.append(cell_summary(df, outcome))
        for oracle, g in primary.groupby("oracle"):
            try:
                a = two_way_anova(g, outcome)
                a.insert(0, "outcome", outcome)
                a.insert(0, "oracle", oracle)
                anovas.append(a)
            except ValueError as e:
                print(f"  RQ1/2 {oracle} {outcome}: {e}")
            s = simple_effects(g, outcome)
            if not s.empty:
                s.insert(0, "outcome", outcome)
                s.insert(0, "oracle", oracle)
                simple.append(s)
    out["cell_summary"] = pd.concat(summaries) if summaries else pd.DataFrame()
    out["anova"] = pd.concat(anovas) if anovas else pd.DataFrame()
    out["simple_effects"] = pd.concat(simple) if simple else pd.DataFrame()
    diag = [cell_summary(df, o) for o in RQ1_DIAGNOSTICS if o in df.columns]
    out["rq1_diagnostics"] = pd.concat(diag) if diag else pd.DataFrame()
    # RQ2 anchor: every cell relative to the ToxSearch-S recipe (d_EP, LF) within oracle.
    rel = []
    for outcome in [o for o in outcomes if o in df.columns]:
        cs = cell_summary(primary, outcome)
        for oracle, g in cs.groupby("oracle"):
            ref = g[(g.distance == "d_EP") & (g.clustering == "LF")]["mean"]
            if len(ref):
                g = g.assign(delta_vs_toxsearch_s=g["mean"] - float(ref.iloc[0]))
                rel.append(g)
    out["rq2_vs_recipe"] = pd.concat(rel) if rel else pd.DataFrame()
    return out


# ---------------------------------------------------------------- RQ3

def trajectories(df: pd.DataFrame) -> pd.DataFrame:
    """Per-generation niche dynamics from species_snapshots.jsonl (one row per run x generation)."""
    rows = []
    for r in df.itertuples(index=False):
        snaps = load_snapshots(Path(r.run_dir))
        prev_leaders = None
        for s in snaps:
            labels = np.array(list((s.get("membership") or {}).values()), dtype=int)
            leaders = {k: v.get("leader_id") for k, v in (s.get("leaders") or {}).items() if int(k) > 0}
            changed = 0 if prev_leaders is None else sum(1 for k, v in leaders.items() if k in prev_leaders and prev_leaders[k] != v)
            prev_leaders = leaders
            states = [v.get("species_state") for k, v in (s.get("leaders") or {}).items() if int(k) > 0]
            rows.append({
                "run_dir": r.run_dir, "oracle": r.oracle, "distance": r.distance, "clustering": r.clustering, "seed": r.seed,
                "generation": s["generation"], "n_genomes": s.get("n_genomes"),
                "n_species": len(set(labels[labels > 0].tolist())),
                "reserve": int((labels == 0).sum()), "archive": int((labels == -1).sum()),
                "frozen": sum(1 for st in states if st == "frozen"),
                "leader_changes": changed,
                "speciation_seconds": s.get("speciation_duration_seconds"),
                "nli_model_passes": (s.get("nli") or {}).get("model_passes"),
            })
    return pd.DataFrame(rows)


def rq3(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """LF vs IncDBSCAN within each (oracle, distance): final structure + trajectories."""
    out = {}
    final = [cell_summary(df[df.clustering.isin(PRIMARY_CLUSTERINGS)], o) for o in RQ3_OUTCOMES if o in df.columns]
    out["final_structure"] = pd.concat(final) if final else pd.DataFrame()
    tr = trajectories(df[df.clustering.isin(PRIMARY_CLUSTERINGS)])
    out["trajectories"] = tr
    if not tr.empty:
        tr = tr.assign(budget_bin=(tr.n_genomes // 50) * 50)
        out["trajectory_summary"] = (tr.groupby(["oracle", "distance", "clustering", "budget_bin"])
                                     [["n_species", "reserve", "archive", "frozen", "leader_changes"]].mean().reset_index())
    return out


# ---------------------------------------------------------------- RQ4

def rq4_rank_agreement(df: pd.DataFrame, outcomes: Sequence[str] = PRIMARY_OUTCOMES) -> pd.DataFrame:
    """Kendall tau between oracles over the six niche cells (and over the d_E/d_N rows alone)."""
    rows = []
    primary = df[df.distance.isin(PRIMARY_DISTANCES) & df.clustering.isin(PRIMARY_CLUSTERINGS)]
    oracles = sorted(primary.oracle.unique())
    for outcome in [o for o in outcomes if o in df.columns]:
        means = primary.groupby(["oracle", "distance", "clustering"])[outcome].mean().unstack("oracle")
        for i, o1 in enumerate(oracles):
            for o2 in oracles[i + 1:]:
                for subset, m in (("all_6", means), ("prompt_only_dE_dN", means.loc[["d_E", "d_N"]] if {"d_E", "d_N"} <= set(means.index.get_level_values(0)) else means.iloc[0:0])):
                    m = m[[o1, o2]].dropna()
                    if len(m) >= 3:
                        t = stats.kendalltau(m[o1], m[o2])
                        rows.append({"outcome": outcome, "oracles": f"{o1}~{o2}", "cells": subset, "n": len(m), "tau": t.statistic, "p": t.pvalue})
    return pd.DataFrame(rows)


def rq4_sign_agreement(df: pd.DataFrame, outcomes: Sequence[str] = PRIMARY_OUTCOMES) -> pd.DataFrame:
    rows = []
    for outcome in [o for o in outcomes if o in df.columns]:
        eff = standardized_effects(df, outcome)
        if eff.empty or eff.oracle.nunique() < 2:
            continue
        w = eff.pivot_table(index=["distance", "clustering"], columns="oracle", values="effect")
        o1, o2 = list(w.columns)[:2]
        agree = np.sign(w[o1]) == np.sign(w[o2])
        rows.append({"outcome": outcome, "n_cells": int(w[[o1, o2]].dropna().shape[0]),
                     "sign_agreement": float(agree.mean()),
                     "sign_agreement_excl_dEP": float(agree[w.index.get_level_values(0) != "d_EP"].mean())})
    return pd.DataFrame(rows)


def export_topk_for_cross_scoring(df: pd.DataFrame, k: int, out_csv: Path) -> pd.DataFrame:
    """Write the final top-k prompts of every run for re-answering + scoring by every oracle."""
    from utils.population_io import _extract_north_star_score

    rows = []
    for r in df.itertuples(index=False):
        pop = load_final_population(Path(r.run_dir), r.clustering_method)
        pop.sort(key=lambda g: _extract_north_star_score(g, r.north_star_metric), reverse=True)
        for rank, g in enumerate(pop[:k]):
            rows.append({"run_dir": r.run_dir, "search_oracle": r.oracle, "distance": r.distance, "clustering": r.clustering,
                         "seed": r.seed, "rank": rank, "genome_id": g.get("id"), "prompt": g.get("prompt"),
                         "search_score": _extract_north_star_score(g, r.north_star_metric)})
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    return out


def rescore_topk(topk_csv: Path, out_csv: Path, measure_oracles: Sequence[str] = ("google", "openai")) -> None:
    """TODO: re-answer each prompt with theta_rg (same RG config, fixed seed), score the new response
    with every oracle in ``measure_oracles`` (north star of each), write columns
    [run_dir, search_oracle, measure_oracle, genome_id, measure_score]. Add 'laya' once M2 is live."""
    raise NotImplementedError


def rq4_cross_scoring(cross_csv: Path) -> pd.DataFrame:
    """Search-oracle x measure-oracle matrix of mean top-k fitness (per cell and overall)."""
    x = pd.read_csv(cross_csv)
    return x.pivot_table(index=["search_oracle", "distance", "clustering"], columns="measure_oracle",
                         values="measure_score", aggfunc="mean")


def write_tables(tables: Dict[str, pd.DataFrame], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        if t is not None and not t.empty:
            t.to_csv(out_dir / f"{prefix}_{name}.csv", index=False)
