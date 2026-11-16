#!/usr/bin/env python3
"""RQ1 mechanism diagnostics on B3139 matrices (no new EA / NLI).

Computes:
  - finer percolation grid t=0.05..0.20 step 0.01
  - cross-source edge share
  - degree heterogeneity (max degree, top-1% edge share)
  - hub-removal (top 1% degree) at t=0.10 and dens-matched thresholds
  - operating-window scan under D_assigned<0.70 and assigned≥70%

Example::

    PYTHONPATH=src:. ./venv/bin/python \\
      experiments/evostar_2027/validation/rq1_mechanism_diagnostics.py
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.cluster import DBSCAN

ROOT = Path(__file__).resolve().parents[3]
MAT_E = ROOT / "experiments/evostar_2027/data/mats/B3139/d_E.npy"
MAT_NLI = ROOT / "experiments/evostar_2027/data/mats/B3139/d_negE.npy"
POOL = ROOT / "experiments/evostar_2027/data/pool/p0_full.pkl"
GRID_CSV = ROOT / "experiments/evostar_2027/rq1/sweeps/B3139/threshold_grid.csv"
OUT = ROOT / "experiments/evostar_2027/validation/reports"


def _load_runs(n: int) -> np.ndarray:
    pool = pickle.load(POOL.open("rb"))
    if hasattr(pool, "columns"):
        runs = pool["run"].astype(str).to_numpy()
    else:
        rows = pool["rows"] if isinstance(pool, dict) else pool
        runs = np.asarray([str(r.get("run") or r.get("source_run")) for r in rows])
    assert len(runs) == n
    return runs


def _adj_stats(D: np.ndarray, t: float) -> Dict[str, float]:
    n = int(D.shape[0])
    # undirected edges, no self
    mask = (D <= t) & ~np.eye(n, dtype=bool)
    # symmetrize for safety
    mask = np.logical_or(mask, mask.T)
    deg = mask.sum(axis=1).astype(np.int64)
    n_edges = int(deg.sum() // 2)
    max_edges = n * (n - 1) // 2
    dens = n_edges / max_edges if max_edges else float("nan")
    # giant CC
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
            for v in np.flatnonzero(mask[u]):
                if not seen[v]:
                    seen[v] = True
                    stack.append(int(v))
        largest = max(largest, size)
    return {
        "n": float(n),
        "n_edges": float(n_edges),
        "edge_density": float(dens),
        "giant_cc_frac": float(largest / n) if n else float("nan"),
        "max_degree": float(deg.max() if n else 0),
        "mean_degree": float(deg.mean() if n else 0),
    }, deg, mask


def cross_source_edge_share(mask: np.ndarray, runs: np.ndarray) -> float:
    n = mask.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    edges = mask[iu, ju]
    if not edges.any():
        return float("nan")
    cross = runs[iu[edges]] != runs[ju[edges]]
    return float(cross.mean())


def top_hub_edge_share(deg: np.ndarray, mask: np.ndarray, frac: float = 0.01) -> float:
    n = int(deg.size)
    k = max(1, int(round(frac * n)))
    hubs = np.argpartition(-deg, k - 1)[:k]
    hub_set = set(int(i) for i in hubs.tolist())
    n_edges = int(deg.sum() // 2)
    if n_edges == 0:
        return float("nan")
    # undirected edges incident to at least one hub
    iu, ju = np.triu_indices(n, k=1)
    edges = mask[iu, ju]
    if not edges.any():
        return float("nan")
    incident = np.array([(int(i) in hub_set) or (int(j) in hub_set) for i, j in zip(iu[edges], ju[edges])])
    return float(incident.mean())


def hub_removal_giant(D: np.ndarray, t: float, frac: float = 0.01) -> Dict[str, float]:
    stats, deg, mask = _adj_stats(D, t)
    n = int(D.shape[0])
    k = max(1, int(round(frac * n)))
    hubs = np.argpartition(-deg, k - 1)[:k]
    keep = np.ones(n, dtype=bool)
    keep[hubs] = False
    idx = np.flatnonzero(keep)
    sub = D[np.ix_(idx, idx)]
    st2, _, _ = _adj_stats(sub, t)
    return {
        "t": t,
        "n_removed": float(k),
        "giant_before": stats["giant_cc_frac"],
        "giant_after": st2["giant_cc_frac"],
        "edge_density": stats["edge_density"],
        "hub_edge_share": top_hub_edge_share(deg, mask, frac),
        "max_degree": stats["max_degree"],
        "cross_source_share": float("nan"),  # filled by caller
    }


def dens_match_t(D: np.ndarray, target: float, grid: Sequence[float]) -> float:
    best_t, best_gap = float(grid[0]), float("inf")
    for t in grid:
        st, _, _ = _adj_stats(D, float(t))
        gap = abs(st["edge_density"] - target)
        if gap < best_gap:
            best_gap, best_t = gap, float(t)
    return best_t


def operating_window(grid_csv: Path) -> List[Dict[str, Any]]:
    """Apply D_assigned<0.70 and assigned>=70% on threshold_grid rows."""
    rows = list(csv.DictReader(grid_csv.open(encoding="utf-8")))
    out = []
    for r in rows:
        t = float(r["threshold"])
        method = r["method"]
        dist = r["distance"]
        ms = r["min_samples"]
        if method == "incdbscan":
            unass = float(r["noise_frac"])
            n_mat = int(float(r["n_mature"]))
            max_c = float(r["max_cluster"])
        else:
            unass = 1.0 - float(r["frac_in_mature"])
            n_mat = int(float(r["n_mature"]))
            max_c = float(r["max_cluster"])
        assigned = 1.0 - unass
        d_asg = (max_c / (assigned * 3139)) if assigned > 0 else float("nan")
        feasible = (d_asg < 0.70) and (assigned >= 0.70) and (n_mat >= 3)
        out.append(
            {
                "distance": dist,
                "method": method,
                "min_samples": ms,
                "t": t,
                "n_mature": n_mat,
                "unassigned": unass,
                "d_assigned": d_asg,
                "feasible": int(feasible),
            }
        )
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    De = np.load(MAT_E)
    Dn = np.load(MAT_NLI)
    assert De.shape == Dn.shape == (3139, 3139)
    runs = _load_runs(3139)

    fine = [round(0.05 + 0.01 * i, 2) for i in range(16)]  # 0.05..0.20
    perc_rows = []
    for name, D in (("d_E", De), ("d_negE", Dn)):
        for t in fine:
            st, deg, mask = _adj_stats(D, t)
            cross = cross_source_edge_share(mask, runs)
            hub_share = top_hub_edge_share(deg, mask, 0.01)
            perc_rows.append(
                {
                    "distance": name,
                    "t": t,
                    **{k: st[k] for k in ("edge_density", "giant_cc_frac", "max_degree", "mean_degree", "n_edges")},
                    "cross_source_edge_share": cross,
                    "top1pct_hub_edge_share": hub_share,
                }
            )
            print(
                f"{name} t={t:.2f}: dens={st['edge_density']:.4f} giant={st['giant_cc_frac']:.3f} "
                f"cross={cross:.3f} hub1%={hub_share:.3f} maxdeg={st['max_degree']:.0f}"
            )

    # Hub removal at t=0.10 and dens-matched
    hub_rows = []
    st_e, _, _ = _adj_stats(De, 0.10)
    st_n, _, _ = _adj_stats(Dn, 0.10)
    dens_e, dens_n = st_e["edge_density"], st_n["edge_density"]
    # match NLI density to MiniLM @0.10 and vice versa
    match_grid = [round(0.01 + 0.01 * i, 2) for i in range(50)]
    t_n_to_e = dens_match_t(Dn, dens_e, match_grid)
    t_e_to_n = dens_match_t(De, dens_n, match_grid)

    for label, D, t in [
        ("d_E@0.10", De, 0.10),
        ("d_negE@0.10", Dn, 0.10),
        (f"d_negE@dens-match({t_n_to_e:.2f})", Dn, t_n_to_e),
        (f"d_E@dens-match({t_e_to_n:.2f})", De, t_e_to_n),
    ]:
        dist = "d_E" if label.startswith("d_E") else "d_negE"
        st, deg, mask = _adj_stats(D, t)
        row = hub_removal_giant(D, t, 0.01)
        row["label"] = label
        row["distance"] = dist
        row["cross_source_share"] = cross_source_edge_share(mask, runs)
        row["hub_edge_share"] = top_hub_edge_share(deg, mask, 0.01)
        row["max_degree"] = st["max_degree"]
        hub_rows.append(row)
        print(
            f"hub-removal {label}: giant {row['giant_before']:.3f} → {row['giant_after']:.3f} "
            f"(cross={row['cross_source_share']:.3f}, hub_share={row['hub_edge_share']:.3f})"
        )

    win = operating_window(GRID_CSV)
    feasible = [r for r in win if r["feasible"] == 1]
    # summarize windows
    windows: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in feasible:
        key = (r["distance"], r["method"], str(r["min_samples"]))
        windows[key].append(r["t"])

    # write CSVs
    def write(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        fields = list(rows[0].keys())
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    write(OUT / "rq1_percolation_fine.csv", perc_rows)
    write(OUT / "rq1_hub_removal.csv", hub_rows)
    write(OUT / "rq1_operating_window.csv", win)

    summary = {
        "percolation_note": (
            "NLI already has giant≈0.53 at dens≈0.0019 (t=0.05); "
            "MiniLM transitions sharply between dens≈0.0049 and ≈0.0177."
        ),
        "cross_source_at_t010": {
            r["distance"]: r["cross_source_edge_share"]
            for r in perc_rows
            if abs(r["t"] - 0.10) < 1e-9
        },
        "hub_removal": hub_rows,
        "feasible_windows": {
            f"{d}|{m}|ms={ms}": {"t_min": min(ts), "t_max": max(ts), "n_t": len(ts)}
            for (d, m, ms), ts in windows.items()
        },
        "dens_matched_thresholds": {
            "nli_to_match_embedding_010": t_n_to_e,
            "embedding_to_match_nli_010": t_e_to_n,
            "embedding_010_density": dens_e,
            "nli_010_density": dens_n,
        },
    }
    (OUT / "rq1_mechanism_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # markdown
    md = [
        "# RQ1 mechanism diagnostics",
        "",
        "## Cross-source edge share at t=0.10",
        "",
    ]
    for r in perc_rows:
        if abs(r["t"] - 0.10) < 1e-9:
            md.append(
                f"- `{r['distance']}`: cross-source={r['cross_source_edge_share']:.3f}, "
                f"hub1% edges={r['top1pct_hub_edge_share']:.3f}, "
                f"maxdeg={r['max_degree']:.0f}, giant={r['giant_cc_frac']:.3f}"
            )
    md += ["", "## Hub removal (top 1% degree)", ""]
    for r in hub_rows:
        md.append(
            f"- {r['label']}: giant {r['giant_before']:.3f} → {r['giant_after']:.3f}"
        )
    md += ["", "## Feasible operating windows (D_asg<0.70 & assigned≥70% & n_mat≥3)", ""]
    if not windows:
        md.append("- none")
    for (d, m, ms), ts in sorted(windows.items()):
        md.append(f"- `{d}` {m} MinPts={ms}: t∈[{min(ts):.2f}, {max(ts):.2f}] ({len(ts)} points)")
    (OUT / "rq1_mechanism_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"wrote {OUT / 'rq1_mechanism_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
