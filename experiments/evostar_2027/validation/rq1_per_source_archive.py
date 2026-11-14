#!/usr/bin/env python3
"""Optional RQ1 robustness: neighborhood structure on each source-run subset.

Reuses existing B3139 distance matrices and p0_full.pkl provenance.
Compares d_E vs d_negE at t=0.10 on identical prompt subsets per source run.
No new metrics or models.

Example::

    PYTHONPATH=src:. ./venv/bin/python \\
      experiments/evostar_2027/validation/rq1_per_source_archive.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[3]
MAT_E = ROOT / "experiments" / "evostar_2027" / "rq1" / "mats" / "d_E_3139.npy"
MAT_NLI = ROOT / "experiments" / "evostar_2027" / "data" / "mats" / "B3139" / "d_negE.npy"
POOL = ROOT / "experiments" / "evostar_2027" / "data" / "pool" / "p0_full.pkl"
OUT = ROOT / "experiments" / "evostar_2027" / "validation" / "reports"

T = 0.10
MIN_PTS = 2


def graph_stats(D: np.ndarray, eps: float) -> Dict[str, float]:
    n = int(D.shape[0])
    adj = D <= eps
    np.fill_diagonal(adj, True)
    # undirected edges excluding self
    n_edges = int((np.triu(adj, k=1).sum()))
    max_edges = n * (n - 1) // 2
    dens = n_edges / max_edges if max_edges else float("nan")
    # giant CC via BFS on threshold graph
    seen = np.zeros(n, dtype=bool)
    largest = 0
    for i in range(n):
        if seen[i]:
            continue
        stack = [i]
        seen[i] = True
        size = 0
        while stack:
            u = stack.pop()
            size += 1
            nbrs = np.flatnonzero(adj[u])
            for v in nbrs:
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        largest = max(largest, size)
    labels = DBSCAN(eps=eps, min_samples=MIN_PTS, metric="precomputed").fit_predict(D)
    assigned = labels >= 0
    counts = Counter(int(x) for x in labels[assigned].tolist())
    mature = {k: v for k, v in counts.items() if v >= 2}
    n_mat = len(mature)
    n_asg = int(sum(mature.values()))
    d_asg = (max(mature.values()) / n_asg) if mature else float("nan")
    return {
        "n": float(n),
        "edge_density": float(dens),
        "giant_cc_frac": float(largest / n) if n else float("nan"),
        "n_mat": float(n_mat),
        "d_assigned": float(d_asg),
        "unassigned_frac": float((~assigned).mean()) if n else float("nan"),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = pickle.load(POOL.open("rb"))
    if hasattr(pool, "columns"):
        runs = pool["run"].astype(str).tolist()
    else:
        rows = pool["rows"] if isinstance(pool, dict) else pool
        runs = [str(r.get("run") or r.get("source_run")) for r in rows]
    runs_arr = np.asarray(runs)
    D_e = np.load(MAT_E)
    D_n = np.load(MAT_NLI)
    assert D_e.shape[0] == len(runs_arr) == D_n.shape[0]

    rows_out: List[Dict[str, Any]] = []
    for run in sorted(set(runs_arr.tolist())):
        idx = np.flatnonzero(runs_arr == run)
        sub_e = D_e[np.ix_(idx, idx)]
        sub_n = D_n[np.ix_(idx, idx)]
        se = graph_stats(sub_e, T)
        sn = graph_stats(sub_n, T)
        rows_out.append(
            {
                "run": run,
                "n": int(idx.size),
                "d_E_edge_density": se["edge_density"],
                "d_E_giant_cc_frac": se["giant_cc_frac"],
                "d_E_n_mat": se["n_mat"],
                "d_E_d_assigned": se["d_assigned"],
                "d_negE_edge_density": sn["edge_density"],
                "d_negE_giant_cc_frac": sn["giant_cc_frac"],
                "d_negE_n_mat": sn["n_mat"],
                "d_negE_d_assigned": sn["d_assigned"],
                "nli_higher_giant": int(sn["giant_cc_frac"] > se["giant_cc_frac"]),
                "nli_higher_dominance": int(sn["d_assigned"] > se["d_assigned"]),
            }
        )
        print(
            f"{run} n={idx.size}: "
            f"dE dens={se['edge_density']:.4f} giant={se['giant_cc_frac']:.2f} "
            f"D={se['d_assigned']:.2f} n_mat={se['n_mat']:.0f} | "
            f"NLI dens={sn['edge_density']:.4f} giant={sn['giant_cc_frac']:.2f} "
            f"D={sn['d_assigned']:.2f} n_mat={sn['n_mat']:.0f}"
        )

    # Pooled reference
    pe = graph_stats(D_e, T)
    pn = graph_stats(D_n, T)
    pooled = {
        "run": "B3139_pooled",
        "n": int(D_e.shape[0]),
        "d_E_edge_density": pe["edge_density"],
        "d_E_giant_cc_frac": pe["giant_cc_frac"],
        "d_E_n_mat": pe["n_mat"],
        "d_E_d_assigned": pe["d_assigned"],
        "d_negE_edge_density": pn["edge_density"],
        "d_negE_giant_cc_frac": pn["giant_cc_frac"],
        "d_negE_n_mat": pn["n_mat"],
        "d_negE_d_assigned": pn["d_assigned"],
        "nli_higher_giant": int(pn["giant_cc_frac"] > pe["giant_cc_frac"]),
        "nli_higher_dominance": int(pn["d_assigned"] > pe["d_assigned"]),
    }
    rows_out.append(pooled)

    csv_path = OUT / "rq1_per_source_archive.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    consistent_giant = all(r["nli_higher_giant"] for r in rows_out if r["run"] != "B3139_pooled")
    consistent_dom = all(r["nli_higher_dominance"] for r in rows_out if r["run"] != "B3139_pooled")
    md = [
        "# RQ1 per-source archive robustness",
        "",
        f"Threshold $t={T}$, batch DBSCAN MinPts={MIN_PTS}, identical subsets per source run.",
        "",
        f"- NLI giant CC > embedding on every source archive: **{consistent_giant}**",
        f"- NLI $D_{{assigned}}$ > embedding on every source archive: **{consistent_dom}**",
        "",
        "| run | n | dE dens | dE giant | dE D | NLI dens | NLI giant | NLI D |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows_out:
        md.append(
            f"| {r['run']} | {r['n']} | {r['d_E_edge_density']:.4f} | "
            f"{r['d_E_giant_cc_frac']:.2f} | {r['d_E_d_assigned']:.2f} | "
            f"{r['d_negE_edge_density']:.4f} | {r['d_negE_giant_cc_frac']:.2f} | "
            f"{r['d_negE_d_assigned']:.2f} |"
        )
    md += [
        "",
        "Disclosure: subset sizes differ by source run; this is not a large population of",
        "independent datasets and does not claim universal generalization.",
        "",
    ]
    md_path = OUT / "rq1_per_source_archive.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    meta = {
        "threshold": T,
        "min_pts": MIN_PTS,
        "consistent_nli_higher_giant": consistent_giant,
        "consistent_nli_higher_dominance": consistent_dom,
        "csv": str(csv_path),
    }
    (OUT / "rq1_per_source_archive_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
