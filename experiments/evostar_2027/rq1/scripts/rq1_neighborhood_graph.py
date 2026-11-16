#!/usr/bin/env python3
"""RQ1 neighborhood-graph diagnostics for MiniLM vs NLI collapse.

On the fixed N=3139 calibration archive:
  - edge density, degree quartiles, core fraction, giant (core) component
  - matched-density threshold search (equalize edge density across representations)
  - bridge / high-betweenness removal diagnostic on the NLI giant component
  - online counterpart from live ``inc_dbscan_state.json`` graphs

Example::

    PYTHONPATH=src:. venv/bin/python \\
      experiments/evostar_2027/rq1/scripts/rq1_neighborhood_graph.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from speciation.clustering.inc_dbscan import seed_from_precomputed_matrix  # noqa: E402

MAT_E = ROOT / "experiments" / "evostar_2027" / "data" / "mats" / "B3139" / "d_E.npy"
MAT_NLI = ROOT / "experiments" / "evostar_2027" / "data" / "mats" / "B3139" / "d_negE.npy"
POOL = ROOT / "experiments" / "evostar_2027" / "data" / "pool" / "p0_full.pkl"
LIVE = ROOT / "data" / "outputs" / "evostar2027"
OUT = ROOT / "experiments" / "evostar_2027" / "rq1" / "reports"
NOTES = ROOT / "notes" / "rough_draft"

THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.25)
MIN_SAMPLES = (2, 3, 5)
ONLINE_CELLS = (("embedding", "inc_embedding"), ("nli_entail", "inc_nli"))


def _fitness_order(n: int) -> np.ndarray:
    if POOL.is_file():
        import pickle

        with POOL.open("rb") as f:
            pool = pickle.load(f)
        rows = pool.get("rows") if isinstance(pool, dict) else pool
        fits: List[Tuple[float, int]] = []
        if isinstance(rows, list):
            for i, rec in enumerate(rows[:n]):
                if not isinstance(rec, dict):
                    fits.append((0.0, i))
                    continue
                fit = rec.get("fitness", rec.get("illicit", 0.0))
                try:
                    fits.append((float(fit), i))
                except (TypeError, ValueError):
                    fits.append((0.0, i))
            if len(fits) == n:
                fits.sort(key=lambda t: (-t[0], t[1]))
                return np.asarray([i for _, i in fits], dtype=np.int64)
    return np.arange(n, dtype=np.int64)


def _adj_from_matrix(D: np.ndarray, t: float) -> List[List[int]]:
    n = D.shape[0]
    adj: List[List[int]] = [[] for _ in range(n)]
    # Upper triangle for speed; add both directions + self
    for i in range(n):
        nbrs = np.flatnonzero(D[i] <= t)
        adj[i] = nbrs.tolist()
    return adj


def _connected_components(adj: Sequence[Sequence[int]], mask: Optional[np.ndarray] = None) -> List[List[int]]:
    n = len(adj)
    seen = np.zeros(n, dtype=bool)
    comps: List[List[int]] = []
    for i in range(n):
        if seen[i]:
            continue
        if mask is not None and not mask[i]:
            continue
        stack = [i]
        seen[i] = True
        comp = [i]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if seen[v]:
                    continue
                if mask is not None and not mask[v]:
                    continue
                seen[v] = True
                stack.append(v)
                comp.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def graph_stats(D: np.ndarray, t: float, min_samples_list: Sequence[int]) -> Dict[str, Any]:
    n = int(D.shape[0])
    # Degrees excluding self for reporting; core test includes self
    deg = np.sum(D <= t, axis=1).astype(np.int32)  # includes self
    n_edges = int((deg.sum() - n) // 2)  # undirected, no self
    max_edges = n * (n - 1) // 2
    edge_density = n_edges / max_edges if max_edges else float("nan")
    deg_ext = deg - 1  # exclude self for quartiles
    q25, q50, q75 = (float(x) for x in np.percentile(deg_ext, [25, 50, 75]))
    adj = _adj_from_matrix(D, t)
    comps = _connected_components(adj)
    giant = len(comps[0]) if comps else 0
    row: Dict[str, Any] = {
        "threshold": t,
        "n": n,
        "n_edges": n_edges,
        "edge_density": edge_density,
        "mean_degree": float(deg_ext.mean()),
        "deg_q25": q25,
        "deg_q50": q50,
        "deg_q75": q75,
        "giant_cc": giant,
        "giant_cc_frac": giant / n if n else float("nan"),
        "n_components": len(comps),
    }
    for ms in min_samples_list:
        is_core = deg >= ms
        core_frac = float(is_core.mean())
        core_comps = _connected_components(adj, mask=is_core)
        giant_core = len(core_comps[0]) if core_comps else 0
        # Assigned = cores in components of size>=2 plus their borders
        # For simplicity report giant core / all and / cores
        n_core = int(is_core.sum())
        row[f"core_frac_m{ms}"] = core_frac
        row[f"giant_core_m{ms}"] = giant_core
        row[f"giant_core_frac_all_m{ms}"] = giant_core / n if n else float("nan")
        row[f"giant_core_frac_cores_m{ms}"] = (
            giant_core / n_core if n_core else float("nan")
        )
    return row


def batch_partition_stats(D: np.ndarray, t: float, min_samples: int, order: np.ndarray) -> Dict[str, float]:
    from sklearn.cluster import DBSCAN

    batch = DBSCAN(eps=t, min_samples=min_samples, metric="precomputed").fit_predict(D)
    labs = np.asarray(batch)
    n = labs.size
    assigned = labs >= 0
    counts = Counter(int(x) for x in labs[assigned].tolist())
    mature = {k: v for k, v in counts.items() if v >= 2}
    n_mat = len(mature)
    n_ass = sum(mature.values())
    largest = max(mature.values()) if mature else 0
    return {
        "n_mat": float(n_mat),
        "unassigned_frac": float((~assigned).mean()),
        "d_assigned": (largest / n_ass) if n_ass else float("nan"),
    }


def lf_partition_stats(D: np.ndarray, t: float, order: np.ndarray) -> Dict[str, float]:
    """Simple fitness-order leader-follower: assign to nearest leader if d < t."""
    n = D.shape[0]
    leaders: List[int] = []
    labels = np.full(n, -1, dtype=np.int32)
    for orig in order.tolist():
        i = int(orig)
        if not leaders:
            leaders.append(i)
            labels[i] = 0
            continue
        dists = D[i, leaders]
        j = int(np.argmin(dists))
        if dists[j] < t:  # strict, matching live LF
            labels[i] = j  # temporary: leader index in leaders list
        else:
            labels[i] = len(leaders)
            leaders.append(i)
    # Remap: singleton leaders (size 1) → unassigned
    counts = Counter(int(x) for x in labels.tolist())
    remap: Dict[int, int] = {}
    next_id = 0
    for lab, c in counts.items():
        if c >= 2:
            remap[lab] = next_id
            next_id += 1
    out = np.full(n, -1, dtype=np.int32)
    for i, lab in enumerate(labels.tolist()):
        if lab in remap:
            out[i] = remap[lab]
    assigned = out >= 0
    counts2 = Counter(int(x) for x in out[assigned].tolist())
    n_mat = len(counts2)
    n_ass = int(assigned.sum())
    largest = max(counts2.values()) if counts2 else 0
    return {
        "n_mat": float(n_mat),
        "unassigned_frac": float((~assigned).mean()),
        "d_assigned": (largest / n_ass) if n_ass else float("nan"),
    }


def find_matched_threshold(
    D_src: np.ndarray,
    target_density: float,
    grid: Sequence[float],
) -> Tuple[float, float]:
    """Pick threshold on ``grid`` whose edge density is closest to ``target_density``."""
    best_t, best_dens, best_err = grid[0], float("nan"), float("inf")
    n = D_src.shape[0]
    max_edges = n * (n - 1) // 2
    for t in grid:
        deg = np.sum(D_src <= t, axis=1)
        n_edges = int((deg.sum() - n) // 2)
        dens = n_edges / max_edges if max_edges else 0.0
        err = abs(dens - target_density)
        if err < best_err:
            best_t, best_dens, best_err = float(t), dens, err
    return best_t, best_dens


def dense_threshold_search(
    D: np.ndarray,
    target_density: float,
    lo: float = 0.01,
    hi: float = 0.50,
    steps: int = 50,
) -> Tuple[float, float]:
    grid = np.linspace(lo, hi, steps).tolist()
    return find_matched_threshold(D, target_density, grid)


def bridge_removal(
    D: np.ndarray,
    t: float,
    ks: Sequence[int] = (5, 10, 20),
    sample_betweenness: int = 200,
) -> List[Dict[str, Any]]:
    """Remove top-k betweenness nodes from the NLI giant component; track shrinkage."""
    n = D.shape[0]
    adj = _adj_from_matrix(D, t)
    comps = _connected_components(adj)
    if not comps:
        return []
    giant = set(comps[0])
    # Approximate betweenness on the giant via sampled Brandes
    nodes = sorted(giant)
    idx = {u: i for i, u in enumerate(nodes)}
    m = len(nodes)
    # Build giant-restricted adjacency as lists of local indices
    g_adj = [[] for _ in range(m)]
    for u in nodes:
        ui = idx[u]
        for v in adj[u]:
            if v in idx:
                g_adj[ui].append(idx[v])

    rng = np.random.default_rng(0)
    sources = (
        rng.choice(m, size=min(sample_betweenness, m), replace=False).tolist()
        if m > 0
        else []
    )
    bet = np.zeros(m, dtype=float)
    for s in sources:
        stack: List[int] = []
        pred: List[List[int]] = [[] for _ in range(m)]
        sigma = np.zeros(m, dtype=float)
        dist = np.full(m, -1, dtype=int)
        sigma[s] = 1.0
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in g_adj[v]:
                if dist[w] < 0:
                    dist[w] = dist[v] + 1
                    q.append(w)
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = np.zeros(m, dtype=float)
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bet[w] += delta[w]
    order = np.argsort(-bet)

    rows: List[Dict[str, Any]] = []
    for k in ks:
        remove = set(int(nodes[i]) for i in order[:k].tolist())
        # Rebuild components excluding removed nodes
        mask = np.ones(n, dtype=bool)
        for u in remove:
            mask[u] = False
        # Also zero out edges to removed by mask in BFS
        comps_k = _connected_components(adj, mask=mask)
        # Restrict to remaining giant members
        remaining_giant = [c for c in comps_k if any(u in giant and u not in remove for u in c)]
        remaining_giant.sort(key=len, reverse=True)
        new_giant = len(remaining_giant[0]) if remaining_giant else 0
        rows.append(
            {
                "threshold": t,
                "k_removed": k,
                "original_giant": len(giant),
                "new_giant": new_giant,
                "shrink_frac": 1.0 - (new_giant / len(giant)) if giant else float("nan"),
                "removed_ids": ",".join(str(nodes[i]) for i in order[:k].tolist()),
            }
        )
    return rows


def online_graph_stats(live_root: Path, seeds: Sequence[int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for distance, folder in ONLINE_CELLS:
        for seed in seeds:
            state_path = live_root / folder / f"rep{seed}_s{seed}" / "inc_dbscan_state.json"
            cfg_path = live_root / folder / f"rep{seed}_s{seed}" / "run_config.json"
            if not state_path.is_file():
                continue
            state = json.loads(state_path.read_text(encoding="utf-8"))
            cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
            spec = cfg.get("speciation") or {}
            ms = int(spec.get("dbscan_min_samples") or state.get("min_samples") or 2)
            neighbors = state["neighbors"]
            is_core = np.asarray(state["is_core"], dtype=bool)
            labels = np.asarray(state["labels"], dtype=np.int32)
            n = len(neighbors)
            n_edges = (sum(len(nbrs) for nbrs in neighbors) - n) // 2
            max_edges = n * (n - 1) // 2
            deg_ext = np.asarray([len(nbrs) - 1 for nbrs in neighbors], dtype=float)
            # Build adj for CC
            adj = [list(nbrs) for nbrs in neighbors]
            comps = _connected_components(adj)
            giant = len(comps[0]) if comps else 0
            core_comps = _connected_components(adj, mask=is_core)
            giant_core = len(core_comps[0]) if core_comps else 0
            mature = Counter(int(x) for x in labels.tolist() if int(x) > 0)
            n_mat = sum(1 for c in mature.values() if c >= 2)
            rows.append(
                {
                    "distance": distance,
                    "folder": folder,
                    "seed": seed,
                    "min_samples": ms,
                    "n": n,
                    "n_edges": n_edges,
                    "edge_density": n_edges / max_edges if max_edges else float("nan"),
                    "mean_degree": float(deg_ext.mean()),
                    "core_frac": float(is_core.mean()),
                    "giant_cc_frac": giant / n if n else float("nan"),
                    "giant_core_frac": giant_core / n if n else float("nan"),
                    "n_mat_live": float(n_mat),
                }
            )
    return rows


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


def write_latex_main(path: Path, matched: List[Dict[str, Any]], dens_rows: List[Dict[str, Any]]) -> None:
    """Compact RQ1 matched-density table for the main text."""
    # dens at t=0.10 for both
    by = {(r["distance"], float(r["threshold"])): r for r in dens_rows}
    lines = [
        "% Auto-generated by rq1_neighborhood_graph.py — do not hand-edit.",
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Matched neighborhood-density analysis on the $N{=}3139$ archive. "
        r"At the shared numeric threshold $t{=}0.10$, MiniLM ($d_E$) and NLI "
        r"($d_{\neg E}$) induce different edge densities (cosine $\ge 0.80$ vs.\ "
        r"mean bidirectional entailment $\ge 0.90$). Matching edge density "
        r"across representations shows whether NLI concentration is a "
        r"representation effect or a threshold/MinPts artifact. "
        r"IncDBSCAN columns use batch DBSCAN; $\mathrm{MinPts}{=}2$ equals "
        r"connected-component niching.}",
        r"\label{tab:rq1-matched-density}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llcccc@{}}",
        r"\toprule",
        r"Setting & $t$ & Edge dens. & $\mathrm{MinPts}$ & $n_{\mathrm{mat}}$ & $D_{\mathrm{assigned}}$ \\",
        r"\midrule",
    ]
    for r in matched:
        lines.append(
            f"{r['label']} & {r['threshold']:.3f} & {float(r['edge_density']):.4f} & "
            f"{r['min_samples']} & {float(r['n_mat']):.0f} & {float(r['d_assigned']):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="RQ1 neighborhood-graph diagnostics")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--seeds", type=str, default="0,1,2,3,4,5,6,7,8,9")
    p.add_argument("--skip-online", action="store_true")
    p.add_argument("--skip-bridge", action="store_true")
    args = p.parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    mats = {"d_E": MAT_E, "d_negE": MAT_NLI}
    loaded: Dict[str, np.ndarray] = {}
    for name, path in mats.items():
        if not path.is_file():
            print(f"[warn] missing {path}")
            continue
        loaded[name] = np.load(path)
        print(f"Loaded {name} {loaded[name].shape}")

    dens_rows: List[Dict[str, Any]] = []
    for name, D in loaded.items():
        for t in THRESHOLDS:
            row = graph_stats(D, t, MIN_SAMPLES)
            row["distance"] = name
            dens_rows.append(row)
            print(
                f"{name} t={t}: dens={row['edge_density']:.5f} "
                f"mean_deg={row['mean_degree']:.1f} "
                f"giant={row['giant_cc_frac']:.3f} "
                f"core5={row.get('core_frac_m5', float('nan')):.3f}"
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "rq1_neighborhood_density.csv", dens_rows)

    # Matched-density analysis
    order = _fitness_order(next(iter(loaded.values())).shape[0])
    matched: List[Dict[str, Any]] = []
    if "d_E" in loaded and "d_negE" in loaded:
        # Reference: MiniLM at t=0.10
        ref = next(r for r in dens_rows if r["distance"] == "d_E" and float(r["threshold"]) == 0.10)
        target = float(ref["edge_density"])
        t_nli, dens_nli = dense_threshold_search(loaded["d_negE"], target)
        print(f"Matched NLI threshold for MiniLM dens@{0.10}={target:.5f}: t*={t_nli:.3f} dens={dens_nli:.5f}")

        # Reverse: MiniLM matched to NLI at 0.10
        ref_n = next(r for r in dens_rows if r["distance"] == "d_negE" and float(r["threshold"]) == 0.10)
        target_n = float(ref_n["edge_density"])
        t_e, dens_e = dense_threshold_search(loaded["d_E"], target_n)
        print(f"Matched MiniLM threshold for NLI dens@{0.10}={target_n:.5f}: t*={t_e:.3f} dens={dens_e:.5f}")

        configs = [
            ("MiniLM @0.10", "d_E", 0.10, float(ref["edge_density"])),
            ("NLI @0.10", "d_negE", 0.10, float(ref_n["edge_density"])),
            (f"NLI dens-matched (t={t_nli:.3f})", "d_negE", t_nli, dens_nli),
            (f"MiniLM dens-matched (t={t_e:.3f})", "d_E", t_e, dens_e),
        ]
        for label, name, t, dens in configs:
            D = loaded[name]
            for ms in MIN_SAMPLES:
                st = batch_partition_stats(D, t, ms, order)
                matched.append(
                    {
                        "label": f"{label}, batch MinPts={ms}",
                        "distance": name,
                        "threshold": t,
                        "edge_density": dens,
                        "min_samples": ms,
                        **st,
                        "method": "batch_dbscan",
                    }
                )
            # LF at same threshold
            st_lf = lf_partition_stats(D, t, order)
            matched.append(
                {
                    "label": f"{label}, LF",
                    "distance": name,
                    "threshold": t,
                    "edge_density": dens,
                    "min_samples": "",
                    **st_lf,
                    "method": "leader_follower",
                }
            )

    write_csv(args.out_dir / "rq1_matched_density.csv", matched)
    write_latex_main(NOTES / "rq1_matched_density_table.tex", matched, dens_rows)

    # Bridge removal on NLI at t=0.10
    bridge_rows: List[Dict[str, Any]] = []
    if not args.skip_bridge and "d_negE" in loaded:
        print("Bridge-removal diagnostic on NLI t=0.10 ...")
        bridge_rows = bridge_removal(loaded["d_negE"], t=0.10)
        write_csv(args.out_dir / "rq1_nli_bridge_removal.csv", bridge_rows)
        for r in bridge_rows:
            print(
                f"  k={r['k_removed']}: giant {r['original_giant']} → {r['new_giant']} "
                f"(shrink {r['shrink_frac']:.2%}); ids={r['removed_ids']}"
            )

    online_rows: List[Dict[str, Any]] = []
    if not args.skip_online:
        online_rows = online_graph_stats(LIVE, seeds)
        write_csv(args.out_dir / "rq1_online_graph_stats.csv", online_rows)
        for folder in ("inc_embedding", "inc_nli"):
            sub = [r for r in online_rows if r["folder"] == folder]
            if sub:
                print(
                    f"Online {folder}: median dens={np.median([r['edge_density'] for r in sub]):.5f}, "
                    f"core={np.median([r['core_frac'] for r in sub]):.3f}, "
                    f"giant_core={np.median([r['giant_core_frac'] for r in sub]):.3f}"
                )

    meta = {
        "thresholds": list(THRESHOLDS),
        "min_samples": list(MIN_SAMPLES),
        "mats": {k: str(mats[k]) for k in loaded},
        "n_density_rows": len(dens_rows),
        "n_matched_rows": len(matched),
        "n_bridge_rows": len(bridge_rows),
        "n_online_rows": len(online_rows),
        "semantics_t010": {
            "d_E": "cosine similarity >= 0.80",
            "d_negE": "mean bidirectional entailment probability >= 0.90",
        },
        "minpts2_note": (
            "With self-inclusion, MinPts=2 makes every non-isolated point a core; "
            "DBSCAN clusters equal nontrivial connected components of the threshold graph."
        ),
    }
    (args.out_dir / "rq1_neighborhood_meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Wrote reports under {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
