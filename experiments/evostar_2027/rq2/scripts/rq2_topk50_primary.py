#!/usr/bin/env python3
"""Switch analysis: top-50 primary at exact budget 2000; top-100 sensitivity.

Writes:
  experiments/evostar_2027/rq2/reports/rq2_topk_b2000_runs.csv
  experiments/evostar_2027/rq2/reports/rq2_mwu_topk_b2000.csv
  notes/rough_draft/rq2_factorial_tables.tex          (primary k=50)
  notes/rough_draft/rq2_topk100_sensitivity.tex       (appendix k=100)

Uses Mann--Whitney U with label-permutation p-values (same as rq2_factorial_report._mwu).
Includes Global EA (no_speciation) as a non-speciated baseline row/contrasts.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_factorial_report import (  # noqa: E402
    PRIMARY_CONTRASTS,
    _mwu,
    cliffs_delta,
    cliffs_delta_ci,
    holm_adjust,
)
from rq2_metrics import illicit_of, load_genomes, moderated_genomes, topk_mean_truncated  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "dbscan"): "inc_nli",
    ("none", "none"): "no_speciation",
}
DIST_TEX = {
    "embedding": r"$d_E$",
    "nli_entail": r"$d_{\neg E}$",
    "none": r"---",
}
CLUS_TEX = {
    "leader_follower": "LF",
    "dbscan": "IncDBSCAN",
    "none": "Global EA",
}
# Speciation 2x2 first, then Global EA baseline.
ORDER = [
    ("embedding", "leader_follower"),
    ("embedding", "dbscan"),
    ("nli_entail", "leader_follower"),
    ("nli_entail", "dbscan"),
    ("none", "none"),
]
GLOBAL_KEY = ("none", "none")
# Global EA vs each speciation cell (Holm within metric over these four).
GLOBAL_CONTRASTS: List[Tuple[str, str, Tuple[str, str], Tuple[str, str]]] = [
    (
        "global_vs_lf_emb",
        r"Global EA vs LF ($d_E$)",
        GLOBAL_KEY,
        ("embedding", "leader_follower"),
    ),
    (
        "global_vs_inc_emb",
        r"Global EA vs IncDBSCAN ($d_E$)",
        GLOBAL_KEY,
        ("embedding", "dbscan"),
    ),
    (
        "global_vs_lf_nli",
        r"Global EA vs LF ($d_{\neg E}$)",
        GLOBAL_KEY,
        ("nli_entail", "leader_follower"),
    ),
    (
        "global_vs_inc_nli",
        r"Global EA vs IncDBSCAN ($d_{\neg E}$)",
        GLOBAL_KEY,
        ("nli_entail", "dbscan"),
    ),
]
OUT = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
NOTES = ROOT / "notes" / "rough_draft"
BUDGET = 2000


def complete(run_dir: Path) -> bool:
    p = run_dir / "EvolutionTracker.json"
    if not p.is_file():
        return False
    try:
        return str(json.loads(p.read_text()).get("status")) == "complete"
    except Exception:
        return False


def wall_hours(run_dir: Path) -> float:
    tr = json.loads((run_dir / "EvolutionTracker.json").read_text())
    md = tr.get("run_metadata") or {}
    wall = tr.get("execution_time_seconds") or md.get("run_duration_seconds")
    return float(wall) / 3600.0 if isinstance(wall, (int, float)) else float("nan")


def l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-12)


def mean_pairwise_dE(embs: np.ndarray) -> float:
    if embs.shape[0] < 2:
        return float("nan")
    e = l2(embs.astype(np.float64))
    cos = np.clip(e @ e.T, -1.0, 1.0)
    D = (1.0 - cos) / 2.0
    iu = np.triu_indices(D.shape[0], k=1)
    return float(np.mean(D[iu]))


def elite_disp(recs: Sequence[Dict[str, Any]], k: int) -> float:
    scored: List[Tuple[float, np.ndarray]] = []
    for r in recs:
        f = illicit_of(r)
        e = r.get("prompt_embedding")
        if f is None or not isinstance(e, list) or len(e) != 384:
            continue
        scored.append((float(f), np.asarray(e, dtype=np.float64)))
    scored.sort(key=lambda t: t[0], reverse=True)
    top = scored[:k]
    if len(top) < k:
        return float("nan")
    return mean_pairwise_dE(np.stack([e for _, e in top], axis=0))


def analyze_run(distance: str, clustering: str, seed: int) -> Optional[Dict[str, Any]]:
    run_dir = LIVE / LIVE_DIR[(distance, clustering)] / f"rep{seed}_s{seed}"
    if not complete(run_dir):
        return None
    genomes = load_genomes(run_dir, clustering, attach_embeddings=True)
    first = moderated_genomes(genomes)[:BUDGET]
    fit50 = topk_mean_truncated(genomes, k=50, budget=BUDGET)
    fit100 = topk_mean_truncated(genomes, k=100, budget=BUDGET)
    return {
        "distance": distance,
        "clustering": clustering,
        "seed": seed,
        "n_first": len(first),
        "top50_mean_b2000": fit50,
        "top100_mean_b2000": fit100,
        "disp_top50_b2000": elite_disp(first, 50),
        "disp_top100_b2000": elite_disp(first, 100),
        "wall_hours": wall_hours(run_dir),
    }


def iqr(vals: Sequence[float], fmt: str = ".3f") -> str:
    v = sorted(float(x) for x in vals if x == x and np.isfinite(x))
    if not v:
        return "---"
    med = float(np.median(v))
    q1, q3 = np.percentile(v, [25, 75])
    return f"{med:{fmt}} [{q1:{fmt}}, {q3:{fmt}}]"


def _contrast_block(
    rows: List[Dict[str, Any]],
    metric: str,
    contrasts: Sequence[Tuple[str, str, Tuple[str, str], Tuple[str, str]]],
    family: str,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    raw_p: List[float] = []
    built: List[Dict[str, Any]] = []
    for cid, ctex, akey, bkey in contrasts:
        xa = [
            float(r[metric])
            for r in rows
            if (r["distance"], r["clustering"]) == akey and r[metric] == r[metric] and np.isfinite(r[metric])
        ]
        xb = [
            float(r[metric])
            for r in rows
            if (r["distance"], r["clustering"]) == bkey and r[metric] == r[metric] and np.isfinite(r[metric])
        ]
        u, p = _mwu(xa, xb)
        delta = cliffs_delta(xa, xb)
        lo, hi = cliffs_delta_ci(xa, xb)
        built.append(
            {
                "metric": metric,
                "family": family,
                "contrast": ctex,
                "contrast_id": cid,
                "n_a": len(xa),
                "n_b": len(xb),
                "median_a": float(np.median(xa)) if xa else float("nan"),
                "median_b": float(np.median(xb)) if xb else float("nan"),
                "U": u,
                "p": p,
                "cliffs_delta": delta,
                "cliffs_ci_lo": lo,
                "cliffs_ci_hi": hi,
            }
        )
        raw_p.append(p)
    holm = holm_adjust(raw_p)
    for row, hp in zip(built, holm):
        row["holm_p"] = hp
        out.append(row)
    return out


def contrast_metric(rows: List[Dict[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """LF vs Inc (primary) and Global vs each cell (baseline family)."""
    out = _contrast_block(rows, metric, PRIMARY_CONTRASTS, family="lf_vs_inc")
    out.extend(_contrast_block(rows, metric, GLOBAL_CONTRASTS, family="global_vs_cell"))
    return out


def write_summary_table(path: Path, rows: List[Dict[str, Any]], k: int, primary: bool) -> None:
    fit_key = f"top{k}_mean_b2000"
    disp_key = f"disp_top{k}_b2000"
    label = "primary" if primary else "sensitivity"
    lines = [
        f"% Auto-generated by rq2_topk50_primary.py ({label} k={k}, budget={BUDGET}).",
        r"\begin{table}[ht]",
        r"\centering",
        rf"\caption{{RQ2 end-of-run outcomes at matched budget $N_{{\max}}={BUDGET}$ "
        rf"(median [IQR], $n{{=}}10$). Top-{k} fitness is the mean of the {k} fittest "
        rf"unique individuals among the first {BUDGET} moderated genomes. "
        rf"Elite semantic dispersion $D_{{r,{k}}}$ is mean pairwise MiniLM $d_E$ on that set. "
        r"Wall time is end-to-end clock hours on one NVIDIA A100. "
        r"Global EA is the non-speciated baseline. "
        r"Mann--Whitney $U$ contrasts (label-permutation $p$) are reported in the text.}",
        rf"\label{{tab:rq2-factorial-summary{'' if primary else '-k100'}}}",
        r"\scriptsize",
        r"\begin{tabular}{@{}llccc@{}}",
        r"\toprule",
        rf"Dissim. & Method & Top-{k} fitness & $D_{{r,{k}}}$ & Wall (h) \\",
        r"\midrule",
    ]
    for d, c in ORDER:
        sub = [r for r in rows if r["distance"] == d and r["clustering"] == c]
        lines.append(
            f"{DIST_TEX[d]} & {CLUS_TEX[c]} & {iqr([r[fit_key] for r in sub])} & "
            f"{iqr([r[disp_key] for r in sub])} & {iqr([r['wall_hours'] for r in sub], '.1f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows: List[Dict[str, Any]] = []
    for d, c in ORDER:
        for seed in range(10):
            print(f"topk {d} {c} seed{seed}", flush=True)
            row = analyze_run(d, c, seed)
            if row:
                rows.append(row)
    if not rows:
        raise SystemExit("no complete runs found")
    OUT.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with (OUT / "rq2_topk_b2000_runs.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    tests: List[Dict[str, Any]] = []
    for metric in ("top50_mean_b2000", "disp_top50_b2000", "top100_mean_b2000", "disp_top100_b2000"):
        tests.extend(contrast_metric(rows, metric))
    with (OUT / "rq2_mwu_topk_b2000.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(tests[0].keys()))
        w.writeheader()
        w.writerows(tests)

    write_summary_table(NOTES / "rq2_factorial_tables.tex", rows, k=50, primary=True)
    write_summary_table(NOTES / "rq2_topk100_sensitivity.tex", rows, k=100, primary=False)

    for t in tests:
        if t["metric"].startswith("top50") or t["metric"].startswith("disp_top50"):
            print(
                t["family"],
                t["metric"],
                t["contrast"],
                f"d={t['cliffs_delta']:.2f} [{t['cliffs_ci_lo']:.2f},{t['cliffs_ci_hi']:.2f}]",
                f"p={t['p']:.3f}",
                f"Holm={t['holm_p']:.3f}",
            )
    print(f"wrote {len(rows)} runs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
