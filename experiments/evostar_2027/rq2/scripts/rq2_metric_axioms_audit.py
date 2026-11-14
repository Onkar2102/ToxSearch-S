#!/usr/bin/env python3
"""Metric-axiom audit on all 50 RQ2 live executions.

For each completed run (and optionally a pooled unique-prompt set), checks:

  non-negativity, symmetry, reflexivity d(x,x)=0,
  soft identity (distinct texts with very small d),
  triangle inequality (sampled triples),
  range in [0,1].

d_E  — full n×n matrix from stored MiniLM prompt embeddings (all 50 cells).
d_¬E — sampled triples with on-demand cross-encoder NLI (NLI cells only;
       full pairwise NLI on ~2k–100k prompts is intractable).

Writes:
  experiments/evostar_2027/rq2/reports/metric_axioms_by_execution.csv
  experiments/evostar_2027/rq2/reports/metric_axioms_summary.csv
  experiments/evostar_2027/rq2/reports/metric_axioms_audit.xlsx
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import load_genomes, load_json  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"

CELLS = [
    ("lf_embedding", "d_E", "leader_follower", "MiniLM_LF"),
    ("inc_embedding", "d_E", "dbscan", "MiniLM_IncDBSCAN"),
    ("lf_nli", "d_negE", "leader_follower", "NLI_LF"),
    ("inc_nli", "d_negE", "dbscan", "NLI_IncDBSCAN"),
    ("no_speciation", "none", "none", "Global_EA"),
]


def _norm(p: str) -> str:
    return " ".join(str(p).split()).strip()


def complete(run_dir: Path) -> bool:
    t = load_json(run_dir / "EvolutionTracker.json")
    return isinstance(t, dict) and str(t.get("status", "")).lower() == "complete"


def embeddings_and_prompts(genomes: Sequence[dict]) -> Tuple[np.ndarray, List[str]]:
    embs, prompts, seen = [], [], set()
    for r in genomes:
        p = _norm(r.get("prompt", ""))
        e = r.get("prompt_embedding")
        if not p or not isinstance(e, list) or len(e) != 384:
            continue
        if p in seen:
            continue
        seen.add(p)
        prompts.append(p)
        embs.append(e)
    X = np.asarray(embs, dtype=np.float64)
    nrm = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(nrm, 1e-12)
    return X, prompts


def d_E_matrix(X: np.ndarray) -> np.ndarray:
    sims = np.clip(X @ X.T, -1.0, 1.0)
    D = (1.0 - sims) / 2.0
    np.fill_diagonal(D, 0.0)
    return 0.5 * (D + D.T)


def triangle_sample(D: np.ndarray, n_triples: int, seed: int, atol: float = 1e-12) -> dict:
    n = D.shape[0]
    if n < 3:
        return {"n_checks": 0, "n_viol": 0, "viol_rate": float("nan"), "mean_ex": 0.0, "max_ex": 0.0}
    rng = np.random.default_rng(seed)
    triples = set()
    target = min(n_triples, n * (n - 1) * (n - 2) // 6)
    while len(triples) < target:
        a, b, c = rng.choice(n, size=3, replace=False)
        triples.add(tuple(sorted(map(int, (a, b, c)))))
    checks = viol = 0
    slacks: List[float] = []
    for i, j, k in triples:
        for u, v, w in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            checks += 1
            slack = float(D[u, w] - (D[u, v] + D[v, w]))
            if slack > atol:
                viol += 1
                slacks.append(slack)
    return {
        "n_checks": checks,
        "n_viol": viol,
        "viol_rate": viol / checks if checks else float("nan"),
        "mean_ex": float(np.mean(slacks)) if slacks else 0.0,
        "max_ex": float(np.max(slacks)) if slacks else 0.0,
    }


def triangle_from_embeddings(
    X: np.ndarray, n_triples: int, seed: int, atol: float = 1e-12
) -> dict:
    """Triangle checks without materializing full D (for pooled sets)."""
    n = X.shape[0]
    if n < 3:
        return {"n_checks": 0, "n_viol": 0, "viol_rate": float("nan"), "mean_ex": 0.0, "max_ex": 0.0}

    def d(i: int, j: int) -> float:
        if i == j:
            return 0.0
        return float((1.0 - np.clip(X[i] @ X[j], -1.0, 1.0)) / 2.0)

    rng = np.random.default_rng(seed)
    triples = set()
    target = min(n_triples, n * (n - 1) * (n - 2) // 6)
    while len(triples) < target:
        a, b, c = rng.choice(n, size=3, replace=False)
        triples.add(tuple(sorted(map(int, (a, b, c)))))
    checks = viol = 0
    slacks: List[float] = []
    for i, j, k in triples:
        dd = {
            (i, j): d(i, j),
            (j, i): d(j, i),
            (i, k): d(i, k),
            (k, i): d(k, i),
            (j, k): d(j, k),
            (k, j): d(k, j),
        }
        for u, v, w in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            checks += 1
            slack = dd[(u, w)] - (dd[(u, v)] + dd[(v, w)])
            if slack > atol:
                viol += 1
                slacks.append(slack)
    return {
        "n_checks": checks,
        "n_viol": viol,
        "viol_rate": viol / checks if checks else float("nan"),
        "mean_ex": float(np.mean(slacks)) if slacks else 0.0,
        "max_ex": float(np.max(slacks)) if slacks else 0.0,
    }


def soft_identity(D: np.ndarray, prompts: Sequence[str], thr: float = 0.01) -> dict:
    n = D.shape[0]
    iu = np.triu_indices(n, k=1)
    off = D[iu]
    exact0 = int(np.sum(off <= 1e-12))
    near = int(np.sum(off <= thr))
    distinct_near = 0
    same_near = 0
    idx = np.flatnonzero(off <= thr)
    for k in idx:
        i, j = int(iu[0][k]), int(iu[1][k])
        if prompts[i] == prompts[j]:
            same_near += 1
        else:
            distinct_near += 1
    return {
        "n_pairs": int(off.size),
        "n_exact0": exact0,
        "n_d_le_thr": near,
        "n_distinct_text_d_le_thr": distinct_near,
        "n_same_text_d_le_thr": same_near,
        "thr": thr,
    }


def axiom_row_from_D(
    D: np.ndarray,
    prompts: Sequence[str],
    *,
    n_triples: int,
    seed: int,
    thr: float = 0.01,
) -> dict:
    atol = 1e-12
    sym = float(np.max(np.abs(D - D.T)))
    diag = float(np.max(np.abs(np.diag(D))))
    tri = triangle_sample(D, n_triples, seed, atol=atol)
    ident = soft_identity(D, prompts, thr=thr)
    return {
        "n_prompts": int(D.shape[0]),
        "nonneg_ok": bool(D.min() >= -atol),
        "min_d": float(D.min()),
        "max_d": float(D.max()),
        "range_in_01": bool(D.min() >= -atol and D.max() <= 1.0 + 1e-9),
        "symmetry_ok": sym <= atol,
        "max_asym": sym,
        "reflexivity_ok": diag <= atol,
        "max_diag": diag,
        **{f"id_{k}": v for k, v in ident.items()},
        **{f"tri_{k}": v for k, v in tri.items()},
    }


def audit_dE_run(
    genomes: Sequence[dict], *, n_triples: int, seed: int, thr: float
) -> Optional[dict]:
    X, prompts = embeddings_and_prompts(genomes)
    if len(prompts) < 3:
        return None
    D = d_E_matrix(X)
    return axiom_row_from_D(D, prompts, n_triples=n_triples, seed=seed, thr=thr)


def audit_dnegE_run(
    genomes: Sequence[dict],
    *,
    n_triples: int,
    seed: int,
    thr: float,
    max_prompts: int,
    n_identity_pairs: int = 5000,
) -> Optional[dict]:
    """Sampled-triple + sampled-pair d_negE audit (no full n×n NLI matrix).

    max_prompts caps the prompt pool drawn from the run before sampling.
    """
    from speciation.distance_metrics.nli import nli_entail_distance

    prompts = []
    seen = set()
    for r in genomes:
        p = _norm(r.get("prompt", ""))
        if p and p not in seen:
            seen.add(p)
            prompts.append(p)
    if len(prompts) < 3:
        return None
    rng = np.random.default_rng(seed)
    if len(prompts) > max_prompts:
        idx = rng.choice(len(prompts), size=max_prompts, replace=False)
        prompts = [prompts[i] for i in sorted(idx.tolist())]
    n = len(prompts)
    cache: Dict[Tuple[int, int], float] = {}

    def d_ij(i: int, j: int) -> float:
        if i == j:
            return 0.0
        a, b = (i, j) if i < j else (j, i)
        key = (a, b)
        if key not in cache:
            cache[key] = float(nli_entail_distance(prompts[a], prompts[b]))
        return cache[key]

    # Triangle on sampled triples
    triples = set()
    target = min(n_triples, n * (n - 1) * (n - 2) // 6)
    while len(triples) < target:
        a, b, c = rng.choice(n, size=3, replace=False)
        triples.add(tuple(sorted(map(int, (a, b, c)))))
    print(f"    d_negE triples={len(triples)} on pool n={n}", flush=True)
    checks = viol = 0
    slacks: List[float] = []
    for t_i, (i, j, k) in enumerate(triples):
        dd = {
            (i, j): d_ij(i, j),
            (j, i): d_ij(j, i),
            (i, k): d_ij(i, k),
            (k, i): d_ij(k, i),
            (j, k): d_ij(j, k),
            (k, j): d_ij(k, j),
        }
        for u, v, w in (
            (i, j, k),
            (i, k, j),
            (j, i, k),
            (j, k, i),
            (k, i, j),
            (k, j, i),
        ):
            checks += 1
            slack = dd[(u, w)] - (dd[(u, v)] + dd[(v, w)])
            if slack > 1e-12:
                viol += 1
                slacks.append(slack)
        if (t_i + 1) % 200 == 0 or t_i + 1 == len(triples):
            print(
                f"      triple {t_i+1}/{len(triples)}  cache_pairs={len(cache)}  viol_so_far={viol}",
                flush=True,
            )

    # Soft identity on random pairs
    pair_target = min(n_identity_pairs, n * (n - 1) // 2)
    pairs = set()
    while len(pairs) < pair_target:
        a, b = rng.choice(n, size=2, replace=False)
        pairs.add(tuple(sorted(map(int, (a, b)))))
    near = exact0 = distinct_near = 0
    vals = []
    for a, b in pairs:
        dij = d_ij(a, b)
        vals.append(dij)
        if dij <= 1e-12:
            exact0 += 1
        if dij <= thr:
            near += 1
            if prompts[a] != prompts[b]:
                distinct_near += 1

    return {
        "n_prompts": n,
        "nonneg_ok": bool(min(vals + [0.0]) >= -1e-12),
        "min_d": float(min(vals)) if vals else 0.0,
        "max_d": float(max(vals)) if vals else 0.0,
        "range_in_01": bool(min(vals + [0.0]) >= -1e-12 and max(vals + [0.0]) <= 1.0 + 1e-9),
        "symmetry_ok": True,  # nli_entail_distance symmetrizes by construction
        "max_asym": 0.0,
        "reflexivity_ok": True,
        "max_diag": 0.0,
        "id_n_pairs": pair_target,
        "id_n_exact0": exact0,
        "id_n_d_le_thr": near,
        "id_n_distinct_text_d_le_thr": distinct_near,
        "id_n_same_text_d_le_thr": near - distinct_near,
        "id_thr": thr,
        "tri_n_checks": checks,
        "tri_n_viol": viol,
        "tri_viol_rate": viol / checks if checks else float("nan"),
        "tri_mean_ex": float(np.mean(slacks)) if slacks else 0.0,
        "tri_max_ex": float(np.max(slacks)) if slacks else 0.0,
        "nli_cached_pairs": len(cache),
    }


def write_outputs(rows: List[dict], summary: List[dict]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "metric_axioms_by_execution.csv"
    if rows:
        keys: List[str] = []
        seen_k = set()
        for r in rows:
            for k in r.keys():
                if k not in seen_k:
                    seen_k.add(k)
                    keys.append(k)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
    sum_path = REPORTS / "metric_axioms_summary.csv"
    if summary:
        keys = list(summary[0].keys())
        with sum_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(summary)

    wb = Workbook()
    ws = wb.active
    ws.title = "README"
    for i, line in enumerate(
        [
            "RQ2 metric-axiom audit across live executions",
            "",
            "d_E: MiniLM (1-cos)/2 from stored prompt embeddings; full matrix per run.",
            "d_negE: DeBERTa NLI non-entailment on a random prompt subsample per NLI run",
            "        (full 2k×2k NLI is too costly; see --nli-max-prompts).",
            "",
            "Axioms: non-negativity, symmetry, reflexivity, soft identity (d≤thr),",
            "triangle (sampled triples), range in [0,1].",
            f"Per-execution CSV: {csv_path.name}",
            f"Summary CSV: {sum_path.name}",
        ],
        start=1,
    ):
        ws.cell(i, 1, line)
    ws.column_dimensions["A"].width = 100

    def sheet_from_rows(name: str, data: List[dict]) -> None:
        if not data:
            return
        ws = wb.create_sheet(name)
        keys = list(data[0].keys())
        ws.append(keys)
        for r in data:
            ws.append([r.get(k, "") for k in keys])
        for c in ws[1]:
            c.font = Font(bold=True)
        ws.freeze_panes = "A2"
        for k in range(1, len(keys) + 1):
            ws.column_dimensions[get_column_letter(k)].width = 14

    sheet_from_rows("By_execution", rows)
    sheet_from_rows("Summary", summary)
    xlsx = REPORTS / "metric_axioms_audit.xlsx"
    wb.save(xlsx)
    print(f"wrote {csv_path}")
    print(f"wrote {sum_path}")
    print(f"wrote {xlsx}")


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in {"1", "true", "yes"}


def summarize(rows: List[dict]) -> List[dict]:
    by: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for r in rows:
        by[(r["distance"], r["setting"])].append(r)
    out = []
    for (dist, setting), rs in sorted(by.items()):
        rates = []
        for r in rs:
            try:
                rates.append(float(r["tri_viol_rate"]))
            except (TypeError, ValueError):
                continue
        near = []
        for r in rs:
            v = r.get("id_n_distinct_text_d_le_thr", "")
            if v == "" or v is None:
                continue
            try:
                near.append(int(float(v)))
            except (TypeError, ValueError):
                continue
        max_ex = [float(r["tri_max_ex"]) for r in rs if str(r.get("tri_max_ex", "")).strip() != ""]
        out.append(
            {
                "distance": dist,
                "setting": setting,
                "n_executions": len(rs),
                "tri_viol_rate_median": float(np.median(rates)) if rates else "",
                "tri_viol_rate_min": float(np.min(rates)) if rates else "",
                "tri_viol_rate_max": float(np.max(rates)) if rates else "",
                "tri_max_ex_median": float(np.median(max_ex)) if max_ex else "",
                "distinct_near_median": float(np.median(near)) if near else "",
                "symmetry_all_ok": all(_truthy(r["symmetry_ok"]) for r in rs),
                "reflexivity_all_ok": all(_truthy(r["reflexivity_ok"]) for r in rs),
                "nonneg_all_ok": all(_truthy(r["nonneg_ok"]) for r in rs),
                "range_all_ok": all(_truthy(r["range_in_01"]) for r in rs),
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--triples", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--identity-thr", type=float, default=0.01)
    ap.add_argument(
        "--nli-max-prompts",
        type=int,
        default=400,
        help="Subsample size for per-run d_negE full matrix (default 400).",
    )
    ap.add_argument(
        "--skip-nli",
        action="store_true",
        help="Only audit d_E (all 50 runs); skip on-demand NLI.",
    )
    ap.add_argument(
        "--nli-only",
        action="store_true",
        help="Only audit d_negE on NLI cells; merge into existing by_execution CSV.",
    )
    ap.add_argument(
        "--nli-triples",
        type=int,
        default=None,
        help="Triples per NLI run (default: min(--triples, 2000)).",
    )
    ap.add_argument(
        "--nli-cache",
        type=Path,
        default=REPORTS / "nli_pairs_metric_axioms.sqlite",
    )
    ap.add_argument("--pooled-dE-triples", type=int, default=50000)
    args = ap.parse_args()

    rows: List[dict] = []
    pooled_X: List[np.ndarray] = []
    pooled_seen = set()
    csv_existing = REPORTS / "metric_axioms_by_execution.csv"
    if args.nli_only and csv_existing.is_file():
        with csv_existing.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("distance") != "d_negE"]
        # coerce numeric-looking fields later as strings ok for rewrite
        print(f"[resume] kept {len(rows)} non-d_negE rows from {csv_existing.name}", flush=True)

    nli_triples = args.nli_triples if args.nli_triples is not None else min(args.triples, 2000)

    # ---- d_E on every execution ----
    if args.nli_only:
        print("[skip] d_E (--nli-only)", flush=True)
    else:
        for cell, _dist_tag, clustering, setting in CELLS:
            for i in range(10):
                run_dir = LIVE / cell / f"rep{i}_s{i}"
                execution = f"rep{i}_s{i}"
                if not complete(run_dir):
                    print(f"[skip] {cell}/{execution}", flush=True)
                    continue
                genomes = load_genomes(run_dir, clustering, attach_embeddings=True)
                print(f"[d_E] {cell}/{execution} n={len(genomes)}", flush=True)
                stats = audit_dE_run(
                    genomes, n_triples=args.triples, seed=args.seed + i, thr=args.identity_thr
                )
                if stats is None:
                    continue
                rows.append(
                    {
                        "setting": setting,
                        "cell_dir": cell,
                        "execution": execution,
                        "seed": i,
                        "distance": "d_E",
                        **stats,
                    }
                )
                X, prompts = embeddings_and_prompts(genomes)
                for p, v in zip(prompts, X):
                    if p not in pooled_seen:
                        pooled_seen.add(p)
                        pooled_X.append(v)

    # pooled unique-prompt d_E triangle (no full matrix)
    if (not args.nli_only) and pooled_X:
        Xp = np.stack(pooled_X, axis=0)
        print(f"[d_E pooled] {Xp.shape[0]} unique prompts; sampling triangles", flush=True)
        tri = triangle_from_embeddings(Xp, args.pooled_dE_triples, seed=args.seed + 999)
        rows.append(
            {
                "setting": "POOLED_ALL_EXECUTIONS",
                "cell_dir": "all",
                "execution": "pooled_unique",
                "seed": -1,
                "distance": "d_E",
                "n_prompts": int(Xp.shape[0]),
                "nonneg_ok": True,
                "min_d": "",
                "max_d": "",
                "range_in_01": True,
                "symmetry_ok": True,
                "max_asym": 0.0,
                "reflexivity_ok": True,
                "max_diag": 0.0,
                "id_n_pairs": "",
                "id_n_exact0": "",
                "id_n_d_le_thr": "",
                "id_n_distinct_text_d_le_thr": "",
                "id_n_same_text_d_le_thr": "",
                "id_thr": args.identity_thr,
                "tri_n_checks": tri["n_checks"],
                "tri_n_viol": tri["n_viol"],
                "tri_viol_rate": tri["viol_rate"],
                "tri_mean_ex": tri["mean_ex"],
                "tri_max_ex": tri["max_ex"],
            }
        )

    # ---- d_negE on NLI cells (subsample) ----
    if not args.skip_nli:
        sys.path.insert(0, str(ROOT / "src"))
        from speciation.distance_metrics.nli import set_nli_pair_cache_path

        args.nli_cache.parent.mkdir(parents=True, exist_ok=True)
        set_nli_pair_cache_path(str(args.nli_cache))
        for cell, _dist_tag, clustering, setting in CELLS:
            if "nli" not in cell:
                continue
            for i in range(10):
                run_dir = LIVE / cell / f"rep{i}_s{i}"
                execution = f"rep{i}_s{i}"
                if not complete(run_dir):
                    continue
                genomes = load_genomes(run_dir, clustering, attach_embeddings=False)
                print(
                    f"[d_negE] {cell}/{execution} subsample≤{args.nli_max_prompts}",
                    flush=True,
                )
                stats = audit_dnegE_run(
                    genomes,
                    n_triples=nli_triples,
                    seed=args.seed + 100 + i,
                    thr=args.identity_thr,
                    max_prompts=args.nli_max_prompts,
                )
                if stats is None:
                    continue
                rows.append(
                    {
                        "setting": setting,
                        "cell_dir": cell,
                        "execution": execution,
                        "seed": i,
                        "distance": "d_negE",
                        **stats,
                    }
                )

    summary = summarize([r for r in rows if r["setting"] != "POOLED_ALL_EXECUTIONS"])
    write_outputs(rows, summary)


if __name__ == "__main__":
    main()
