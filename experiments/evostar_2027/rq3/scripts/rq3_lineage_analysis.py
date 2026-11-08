#!/usr/bin/env python3
"""RQ3 lineage-aware null and species reproductive contribution.

Uses stored parent IDs from the frozen 40 runs (no new evolutionary executions).

1. Lineage-aware null: permute fitness within connected components of the
   undirected parent--child kinship graph (genealogical blocks), keeping
   species labels fixed; recompute K10 / pi10.
2. Reproductive contribution for mature species s:
       R_s = (# top-k elites with an ancestor in s) / (# direct offspring of members of s)
   Also report parent-source entropy of the top-k set.

Example::

    PYTHONPATH=src:. venv/bin/python \\
      experiments/evostar_2027/rq3/scripts/rq3_lineage_analysis.py
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

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
NOTES = ROOT / "notes" / "rough_draft"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
}
DIST_TEX = {"embedding": r"$d_E$", "nli_entail": r"$d_{\neg E}$"}
CLUS_TEX = {"leader_follower": "LF", "dbscan": "IncDBSCAN"}
NOISE = {None, 0, -1, "0", "-1"}

N_PERM = 5000
PERM_SEED = 2027
K = 100  # primary elite set size (aligned with RQ2/RQ3 occupancy)


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text()).get("status")) == "complete"
    except Exception:
        return False


def sid_int(rec: Dict[str, Any]) -> Optional[int]:
    s = rec.get("species_id")
    if s in NOISE:
        return None
    try:
        i = int(s)
    except Exception:
        return None
    return None if i <= 0 else i


def parent_ids(rec: Dict[str, Any]) -> List[Any]:
    out = []
    for p in rec.get("parents") or []:
        if isinstance(p, dict):
            pid = p.get("id")
        else:
            pid = p
        if pid is not None:
            out.append(pid)
    return out


def occupancy(labels: Sequence[Optional[int]], fitness: Sequence[float], k: int = K) -> Dict[str, float]:
    order = np.argsort(-np.asarray(fitness, dtype=float))
    top = order[:k]
    top_labs = [labels[i] for i in top]
    assigned = [lab for lab in top_labs if lab is not None]
    u10 = float(sum(1 for lab in top_labs if lab is None))
    k10 = float(len(set(assigned)))
    pi10 = float(max(Counter(assigned).values()) / k) if assigned else 0.0
    return {"K10": k10, "U10": u10, "pi10": pi10}


def kinship_blocks(ids: Sequence[Any], parents: Dict[Any, List[Any]]) -> List[List[int]]:
    """Connected components of the undirected parent--child graph; return index lists."""
    id_to_i = {gid: i for i, gid in enumerate(ids)}
    adj: List[List[int]] = [[] for _ in ids]
    for i, gid in enumerate(ids):
        for pid in parents.get(gid, []):
            j = id_to_i.get(pid)
            if j is None:
                continue
            adj[i].append(j)
            adj[j].append(i)
    seen = [False] * len(ids)
    blocks: List[List[int]] = []
    for i in range(len(ids)):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    comp.append(v)
        blocks.append(comp)
    return blocks


def lineage_null(
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    blocks: Sequence[Sequence[int]],
    *,
    n_perm: int,
    seed: int,
) -> Dict[str, float]:
    fit = np.asarray(fitness, dtype=float)
    labs = list(labels)
    obs = occupancy(labs, fit)
    rng = np.random.default_rng(seed)
    null_k = np.empty(n_perm)
    null_pi = np.empty(n_perm)
    for b in range(n_perm):
        shuffled = fit.copy()
        for block in blocks:
            if len(block) < 2:
                continue
            idx = np.asarray(block, dtype=int)
            shuffled[idx] = fit[idx][rng.permutation(idx.size)]
        o = occupancy(labs, shuffled)
        null_k[b] = o["K10"]
        null_pi[b] = o["pi10"]
    def _p_lower(obs_v, null):
        return float((int(np.sum(null <= obs_v)) + 1) / (null.size + 1))
    def _p_upper(obs_v, null):
        return float((int(np.sum(null >= obs_v)) + 1) / (null.size + 1))
    mu_k, sd_k = float(null_k.mean()), float(null_k.std(ddof=1))
    mu_pi, sd_pi = float(null_pi.mean()), float(null_pi.std(ddof=1))
    return {
        "K10_obs": obs["K10"],
        "K10_null_mean": mu_k,
        "K10_z": (obs["K10"] - mu_k) / sd_k if sd_k > 0 else float("nan"),
        "K10_p": _p_lower(obs["K10"], null_k),
        "pi10_obs": obs["pi10"],
        "pi10_null_mean": mu_pi,
        "pi10_z": (obs["pi10"] - mu_pi) / sd_pi if sd_pi > 0 else float("nan"),
        "pi10_p": _p_upper(obs["pi10"], null_pi),
        "n_blocks": float(len(blocks)),
        "median_block_size": float(np.median([len(b) for b in blocks])),
        "max_block_size": float(max(len(b) for b in blocks)),
    }


def ancestors_of(gid: Any, parents: Dict[Any, List[Any]]) -> set:
    out = set()
    stack = list(parents.get(gid, []))
    seen = set()
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        out.add(u)
        stack.extend(parents.get(u, []))
    return out


def reproductive_stats(
    ids: Sequence[Any],
    labels: Sequence[Optional[int]],
    fitness: Sequence[float],
    parents: Dict[Any, List[Any]],
    k: int = K,
) -> Dict[str, float]:
    id_to_i = {gid: i for i, gid in enumerate(ids)}
    children: Dict[Any, List[Any]] = defaultdict(list)
    for gid, ps in parents.items():
        for pid in ps:
            children[pid].append(gid)

    order = np.argsort(-np.asarray(fitness, dtype=float))
    top_ids = [ids[i] for i in order[:k]]
    top_set = set(top_ids)

    by_sp: Dict[int, List[Any]] = defaultdict(list)
    for gid, lab in zip(ids, labels):
        if lab is not None:
            by_sp[int(lab)].append(gid)

    # Per-species R_s
    r_vals = []
    n_off_vals = []
    n_elite_vals = []
    for sid, members in by_sp.items():
        mem = set(members)
        n_off = sum(len(children.get(m, [])) for m in mem)
        n_elite = 0
        for e in top_ids:
            if e in mem or (mem & ancestors_of(e, parents)):
                n_elite += 1
        n_off_vals.append(n_off)
        n_elite_vals.append(n_elite)
        r_vals.append((n_elite / n_off) if n_off > 0 else float("nan"))

    # Parent-source entropy of top-k: species of immediate parents
    src_labs = []
    for e in top_ids:
        for pid in parents.get(e, []):
            j = id_to_i.get(pid)
            if j is None:
                continue
            lab = labels[j]
            src_labs.append(lab if lab is not None else -1)
    if src_labs:
        counts = np.asarray(list(Counter(src_labs).values()), dtype=float)
        p = counts / counts.sum()
        entropy = float(-(p * np.log(p)).sum())
        n_src = len(counts)
    else:
        entropy = float("nan")
        n_src = 0

    finite_r = [x for x in r_vals if x == x]
    return {
        "n_mature": float(len(by_sp)),
        "median_R": float(np.median(finite_r)) if finite_r else float("nan"),
        "max_R": float(np.max(finite_r)) if finite_r else float("nan"),
        "mean_R": float(np.mean(finite_r)) if finite_r else float("nan"),
        "sum_elite_descended": float(sum(n_elite_vals)),
        "sum_offspring": float(sum(n_off_vals)),
        "parent_source_entropy": entropy,
        "n_parent_sources": float(n_src),
        "frac_species_with_elite_desc": (
            float(sum(1 for e in n_elite_vals if e > 0) / len(n_elite_vals))
            if n_elite_vals
            else float("nan")
        ),
    }


def analyze_run(distance: str, clustering: str, seed: int, n_perm: int) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering)
    ids: List[Any] = []
    labels: List[Optional[int]] = []
    fitness: List[float] = []
    parents: Dict[Any, List[Any]] = {}
    for g in genomes:
        f = illicit_of(g)
        if f is None:
            continue
        gid = g.get("id")
        ids.append(gid)
        labels.append(sid_int(g))
        fitness.append(float(f))
        parents[gid] = parent_ids(g)
    if len(ids) < K:
        return None
    blocks = kinship_blocks(ids, parents)
    null = lineage_null(
        labels,
        fitness,
        blocks,
        n_perm=n_perm,
        seed=PERM_SEED + seed * 31 + hash((distance, clustering)) % 10000,
    )
    repro = reproductive_stats(ids, labels, fitness, parents, k=K)
    return {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "n_unique": len(ids),
        **{f"lin_{k}": v for k, v in null.items()},
        **{f"repro_{k}": v for k, v in repro.items()},
    }


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


def write_latex(path: Path, rows: List[Dict[str, Any]]) -> None:
    caption = (
        r"Lineage-aware occupancy null and reproductive contribution "
        rf"(median over $n{{=}}10$ runs; $B{{=}}5{{,}}000$ within-block permutations). "
        r"Fitness is shuffled inside kinship connected components of the "
        r"parent--child graph; species labels stay fixed. Most runs contain one "
        r"giant kinship component ($\sim$1950 individuals) plus singleton founders. "
        rf"$R_s$ uses the top-{K} elites; we report $\max_s R_s$ and the fraction of "
        r"mature species with any elite descendant. "
        rf"$H_{{\mathrm{{src}}}}$: entropy of immediate parent species among the top {K}."
    )
    lines = [
        "% Auto-generated by rq3_lineage_analysis.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{{caption}}}",
        r"\label{tab:rq3-lineage}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        rf"Dissim. & Method & $K_{{{K}}}$ & $K_{{{K}}}^{{\mathrm{{lin}}}}$ & "
        rf"$z_{{K}}^{{\mathrm{{lin}}}}$ & $\pi_{{{K}}}$ & $\pi_{{{K}}}^{{\mathrm{{lin}}}}$ & "
        r"$\max R$ & frac$_R$ & $H_{\mathrm{src}}$ \\",
        r"\midrule",
    ]
    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        lines.append(
            f"{DIST_TEX[distance]} & {CLUS_TEX[clustering]} & "
            f"{_fmt(np.median([r['lin_K10_obs'] for r in sub]), 1)} & "
            f"{_fmt(np.median([r['lin_K10_null_mean'] for r in sub]), 1)} & "
            f"{_fmt(np.nanmedian([r['lin_K10_z'] for r in sub]), 2)} & "
            f"{_fmt(np.median([r['lin_pi10_obs'] for r in sub]), 2)} & "
            f"{_fmt(np.median([r['lin_pi10_null_mean'] for r in sub]), 2)} & "
            f"{_fmt(np.nanmedian([r['repro_max_R'] for r in sub]), 3)} & "
            f"{_fmt(np.nanmedian([r['repro_frac_species_with_elite_desc'] for r in sub]), 2)} & "
            f"{_fmt(np.nanmedian([r['repro_parent_source_entropy'] for r in sub]), 2)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ3 lineage null and reproductive contribution")
    p.add_argument("--n-perm", type=int, default=N_PERM)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rows: List[Dict[str, Any]] = []
    for distance, clustering in LIVE_DIR:
        for seed in seeds:
            print(f"lineage {distance} {clustering} seed{seed}", flush=True)
            row = analyze_run(distance, clustering, seed, args.n_perm)
            if row:
                rows.append(row)

    REPORT.mkdir(parents=True, exist_ok=True)
    write_csv(REPORT / "rq3_lineage_runs.csv", rows)
    write_latex(NOTES / "rq3_lineage_table.tex", rows)

    meta = {
        "n_perm": args.n_perm,
        "k": K,
        "null": "permute fitness within kinship connected components",
        "R_definition": "(# top-k with ancestor in s) / (# direct offspring of members of s)",
        "n_runs": len(rows),
    }
    (REPORT / "rq3_lineage_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for distance, clustering in LIVE_DIR:
        sub = [r for r in rows if r["distance"] == distance and r["clustering"] == clustering]
        if not sub:
            continue
        print(
            DIST_TEX[distance],
            CLUS_TEX[clustering],
            f"K10={np.median([r['lin_K10_obs'] for r in sub]):.1f}",
            f"lin_null={np.median([r['lin_K10_null_mean'] for r in sub]):.1f}",
            f"z={np.nanmedian([r['lin_K10_z'] for r in sub]):.2f}",
            f"medR={np.nanmedian([r['repro_median_R'] for r in sub]):.3f}",
            f"Hsrc={np.nanmedian([r['repro_parent_source_entropy'] for r in sub]):.2f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
