#!/usr/bin/env python3
"""Mandatory experimental validation checklist (no new EA runs).

0. Reconcile Appendix Table 6 / online ARI with on-disk CSV + code snapshot
1. Deterministic IncDBSCAN star (MinPts=5) and MinPts=2 CC tests
2. RQ2 genome-id vs prompt-text uniqueness + MPNet outlier inspection
3. RQ3 final labels / archive status / K50 / U50 consistency

Example::

    PYTHONPATH=src:. ./venv/bin/python \\
      experiments/evostar_2027/validation/mandatory_checks.py
"""

from __future__ import annotations

import csv
import json
import pickle
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))

from rq2_metrics import illicit_of, load_genomes, moderated_genomes  # noqa: E402

OUT = ROOT / "experiments" / "evostar_2027" / "validation" / "reports"
ONLINE_CSV = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "incdbscan_exactness_online.csv"
ARCHIVE_CSV = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "incdbscan_exactness_archive.csv"
SUMMARY_MD = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "incdbscan_exactness_summary.md"
META_JSON = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "incdbscan_exactness_meta.json"
APP_TEX = ROOT / "notes" / "rough_draft" / "app_a2_correctness.tex"
RQ3_RUNS = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports" / "rq3_occupancy_null_runs.csv"
RQ3_VAR = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports" / "rq3_occupancy_null_variants.csv"
RQ3_TEX = ROOT / "notes" / "rough_draft" / "rq3_occupancy_main.tex"
MPNET_CSV = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "rq2_mpnet_disp_runs.csv"
MPNET_CACHE = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports" / "mpnet_disp_cache" / "lf_nli_s5_top50.npz"
LIVE = ROOT / "data" / "outputs" / "evostar2027"
POOL = ROOT / "experiments" / "evostar_2027" / "data" / "pool" / "p0_full.pkl"

CELLS = (
    ("embedding", "leader_follower", "lf_embedding"),
    ("embedding", "dbscan", "inc_embedding"),
    ("nli_entail", "leader_follower", "lf_nli"),
    ("nli_entail", "dbscan", "inc_nli"),
)


def _median(xs: Sequence[float]) -> float:
    return float(statistics.median(xs)) if xs else float("nan")


def check_table6_snapshot() -> Dict[str, Any]:
    online = list(csv.DictReader(ONLINE_CSV.open(encoding="utf-8")))
    archive = list(csv.DictReader(ARCHIVE_CSV.open(encoding="utf-8")))
    by_folder: Dict[str, List[float]] = defaultdict(list)
    for r in online:
        by_folder[r["folder"]].append(float(r["ari"]))
    emb = by_folder["inc_embedding"]
    nli = by_folder["inc_nli"]
    emb_sum = {
        "median": _median(emb),
        "min": min(emb) if emb else float("nan"),
        "max": max(emb) if emb else float("nan"),
        "n": len(emb),
        "all_exact": all(int(r["exact_match"]) == 1 for r in online if r["folder"] == "inc_embedding"),
    }
    nli_sum = {
        "median": _median(nli),
        "min": min(nli) if nli else float("nan"),
        "max": max(nli) if nli else float("nan"),
        "n": len(nli),
    }
    # Paper Table 6 claims
    paper = {
        "live_dE": (1.000, 1.000, 1.000),
        "live_dNegE": (0.995, 0.974, 0.997),
    }
    match_e = (
        abs(emb_sum["median"] - paper["live_dE"][0]) < 1e-3
        and abs(emb_sum["min"] - paper["live_dE"][1]) < 1e-3
        and abs(emb_sum["max"] - paper["live_dE"][2]) < 1e-3
    )
    match_n = (
        abs(nli_sum["median"] - paper["live_dNegE"][0]) < 5e-3
        and abs(nli_sum["min"] - paper["live_dNegE"][1]) < 5e-3
        and abs(nli_sum["max"] - paper["live_dNegE"][2]) < 5e-3
    )
    # Archive t=0.10 rows cited in Table 6
    arch_t10 = [
        r
        for r in archive
        if abs(float(r["eps"]) - 0.1) < 1e-9 and int(r["min_samples"]) in (2, 3, 5)
    ]
    # Regenerate summary.md from CSV so it cannot drift again
    _rewrite_summary(archive, online)
    return {
        "csv_online": str(ONLINE_CSV),
        "csv_archive": str(ARCHIVE_CSV),
        "meta": str(META_JSON) if META_JSON.is_file() else None,
        "app_tex": str(APP_TEX),
        "embedding_live_ari": emb_sum,
        "nli_live_ari": nli_sum,
        "matches_table6_embedding": match_e,
        "matches_table6_nli": match_n,
        "paper_claims": paper,
        "archive_t010_rows": [
            {
                "distance": r["distance"],
                "min_samples": int(r["min_samples"]),
                "inc_n_mat": float(r["inc_n_mat"]),
                "batch_n_mat": float(r["batch_n_mat"]),
                "ari": float(r["ari"]),
            }
            for r in arch_t10
        ],
        "note": (
            "Online ARI uses final-graph batch rebuild with noise canonicalization "
            "(live labels ≤0 and batch -1 → unassigned). Embedding MinPts=2 is exact; "
            "NLI MinPts=5 is near-exact on partition agreement but under-counts mature "
            "clusters relative to batch. Stale summary.md (low ARI) was regenerated from CSV."
        ),
    }


def _rewrite_summary(archive: List[Dict[str, Any]], online: List[Dict[str, Any]]) -> None:
    lines = [
        "# IncDBSCAN exactness audit",
        "",
        "Compares grow-only IncDBSCAN partitions against classical batch DBSCAN.",
        "At MinPts=2 the methods must match (connected-component niching).",
        "At MinPts≥3 the live insert path can under-count mature clusters relative to batch.",
        "",
        "## Archive (calibration matrices)",
        "",
        "| distance | t | MinPts | inc n_mat | batch n_mat | ARI | exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in archive:
        lines.append(
            f"| {r['distance']} | {r['eps']} | {r['min_samples']} | "
            f"{float(r['inc_n_mat']):.0f} | {float(r['batch_n_mat']):.0f} | "
            f"{float(r['ari']):.4f} | {r['exact_match']} |"
        )
    lines += [
        "",
        "## Online (final live graphs)",
        "",
        "| cell | seed | MinPts | live n_mat | batch n_mat | ARI | exact |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in online:
        lines.append(
            f"| {r['folder']} | {r['seed']} | {r['min_samples']} | "
            f"{float(r['inc_n_mat']):.0f} | {float(r['batch_n_mat']):.0f} | "
            f"{float(r['ari']):.4f} | {r['exact_match']} |"
        )
    lines += ["", "## Online ARI summary (median [min, max])", ""]
    by_folder: Dict[str, List[float]] = defaultdict(list)
    for r in online:
        by_folder[r["folder"]].append(float(r["ari"]))
    for folder, vals in by_folder.items():
        lines.append(
            f"- `{folder}`: {_median(vals):.4f} [{min(vals):.4f}, {max(vals):.4f}] (n={len(vals)})"
        )
    lines += [
        "",
        "## Caveat",
        "",
        "The online comparison applies batch DBSCAN to the *final* ε-graph and",
        "therefore bounds final-partition disagreement. It does not replay the",
        "evolutionary trajectory under exact IncrementalDBSCAN updates.",
        "",
        "Regenerated from CSV by `validation/mandatory_checks.py`.",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")


def check_incdbscan_star() -> Dict[str, Any]:
    from sklearn.cluster import DBSCAN
    from speciation.clustering.inc_dbscan import (
        insert_point,
        seed_from_incremental,
        seed_from_precomputed_matrix,
    )
    from speciation.species import Individual, SpeciesIdGenerator
    from speciation.distance_metrics.embedding import embedding_distance
    from utils.evaluator_profiles import set_active_evaluator, set_active_north_star

    set_active_evaluator("google")
    set_active_north_star("toxicity")

    def _unit(v):
        a = np.asarray(v, dtype=np.float64)
        return a / np.linalg.norm(a)

    def _genome(gid, emb, fitness=0.5):
        scores = {
            "toxicity": fitness,
            "threat": max(0.0001, fitness * 0.5),
            "profanity": max(0.0001, fitness * 0.4),
            "sexually_explicit": max(0.0001, fitness * 0.3),
            "identity_attack": max(0.0001, fitness * 0.2),
            "flirtation": max(0.0001, fitness * 0.1),
            "insult": max(0.0001, fitness * 0.45),
            "severe_toxicity": max(0.0001, fitness * 0.25),
        }
        return {
            "id": gid,
            "prompt": f"prompt-{gid}",
            "prompt_embedding": emb.tolist(),
            "fitness": fitness,
            "toxicity": fitness,
            "generation": 0,
            "status": "complete",
            "moderation_result": {"google": {"scores": scores}},
        }

    D = np.ones((5, 5), dtype=np.float64)
    np.fill_diagonal(D, 0.0)
    for i in range(1, 5):
        D[0, i] = D[i, 0] = 0.05

    SpeciesIdGenerator.reset(0)
    cc = seed_from_precomputed_matrix(D, np.arange(5), eps=0.1, min_samples=2)
    minpts2_ok = len({int(x) for x in cc.tolist() if int(x) >= 0}) == 1 and all(
        int(x) >= 0 for x in cc.tolist()
    )

    SpeciesIdGenerator.reset(0)
    inc = seed_from_precomputed_matrix(D, np.arange(5), eps=0.1, min_samples=5)
    batch = DBSCAN(eps=0.1, min_samples=5, metric="precomputed").fit_predict(D)
    inc_n = len({int(x) for x in inc.tolist() if int(x) >= 0})
    batch_n = len({int(x) for x in batch.tolist() if int(x) >= 0})
    core_promotion_miss = batch_n == 1 and inc_n == 0

    # Live insert_point path
    SpeciesIdGenerator.reset(0)
    a = 0.7
    vecs = [
        np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
        _unit([1.0, a, 0.0, 0.0, 0.0]),
        _unit([1.0, 0.0, a, 0.0, 0.0]),
        _unit([1.0, 0.0, 0.0, a, 0.0]),
        _unit([1.0, 0.0, 0.0, 0.0, a]),
    ]
    eps = 0.1
    genomes = [_genome(i + 1, vecs[i], 1.0 - 0.05 * i) for i in range(4)]
    state, species, _ = seed_from_incremental(
        genomes, eps=eps, min_samples=5, distance_method="embedding"
    )
    g5 = _genome(5, vecs[4], 0.7)
    genomes.append(g5)
    ev = insert_point(state, g5, Individual.from_genome(g5), current_generation=1, species=species)
    live_mature = [int(x) for x in state.labels if int(x) > 0]
    live_miss = len(live_mature) == 0 and state.is_core[0] and not state.is_core[4]

    # Also run pytest subset
    env = {k: v for k, v in __import__("os").environ.items()}
    env["PYTHONPATH"] = f"src:.:{env.get('PYTHONPATH', '')}"
    proc = subprocess.run(
        [
            str(ROOT / "venv" / "bin" / "python"),
            "-m",
            "unittest",
            "tests.test_inc_dbscan.TestIncDBSCANCorePromotionStar",
            "-v",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        env=env,
    )

    return {
        "minpts2_connected_component": bool(minpts2_ok),
        "minpts5_batch_n_mat": int(batch_n),
        "minpts5_inc_n_mat": int(inc_n),
        "core_promotion_miss_precomputed": bool(core_promotion_miss),
        "live_insert_event": ev.get("event"),
        "live_hub_became_core": bool(state.is_core[0]),
        "live_leaf_is_core": bool(state.is_core[4]),
        "live_mature_count": len(set(live_mature)),
        "live_core_promotion_miss": bool(live_miss),
        "unittest_returncode": proc.returncode,
        "unittest_stdout_tail": "\n".join(proc.stdout.strip().splitlines()[-20:]),
        "unittest_stderr_tail": "\n".join(proc.stderr.strip().splitlines()[-20:]),
        "geometry_check": {
            "hub_leaf_dE": [float(embedding_distance(vecs[0], vecs[i])) for i in range(1, 5)],
            "leaf_leaf_min_dE": float(
                min(
                    embedding_distance(vecs[i], vecs[j])
                    for i in range(1, 5)
                    for j in range(i + 1, 5)
                )
            ),
        },
        "verdict": (
            "CONFIRMED core-promotion miss at MinPts=5: batch finds 1 mature cluster; "
            "grow-only insert leaves the star unassigned after the hub becomes core."
            if core_promotion_miss and live_miss
            else "UNEXPECTED: star behaviour did not match the documented failure mode."
        ),
    }


def check_rq2_uniqueness_and_outlier() -> Dict[str, Any]:
    # Cross-cell uniqueness
    cell_stats = []
    for dist, clus, folder in CELLS:
        id_dup_prompt = 0
        prompt_dups_in_topk = 0
        n_runs = 0
        for seed in range(10):
            run_dir = LIVE / folder / f"rep{seed}_s{seed}"
            if not (run_dir / "EvolutionTracker.json").is_file():
                continue
            n_runs += 1
            recs = load_genomes(run_dir, clus, attach_embeddings=False)
            mod = moderated_genomes(recs)[:2000]
            prompts = [str(r.get("prompt") or "") for r in mod]
            ids = [r.get("id") for r in mod]
            # same prompt, different ids inside budget window
            by_prompt: Dict[str, set] = defaultdict(set)
            for pid, pr in zip(ids, prompts):
                by_prompt[pr].add(pid)
            id_dup_prompt += sum(1 for s in by_prompt.values() if len(s) > 1)
            # top-50 by illicit
            scored = [(illicit_of(r), r) for r in mod]
            scored = [(s, r) for s, r in scored if s is not None]
            scored.sort(key=lambda t: (-t[0], str(t[1].get("id"))))
            top = [r for _, r in scored[:50]]
            top_prompts = [str(r.get("prompt") or "") for r in top]
            if len(top_prompts) != len(set(top_prompts)):
                prompt_dups_in_topk += 1
        cell_stats.append(
            {
                "distance": dist,
                "clustering": clus,
                "folder": folder,
                "n_runs": n_runs,
                "budget_windows_with_shared_prompt_across_ids": id_dup_prompt,
                "runs_with_duplicate_prompt_in_top50": prompt_dups_in_topk,
            }
        )

    # MPNet outlier
    mpnet_rows = list(csv.DictReader(MPNET_CSV.open(encoding="utf-8")))
    outlier = min(
        (r for r in mpnet_rows if r["distance"] == "nli_entail" and r["clustering"] == "leader_follower"),
        key=lambda r: float(r["disp_mpnet_top50_b2000"]),
    )
    seed = int(outlier["seed"])
    run_dir = LIVE / "lf_nli" / f"rep{seed}_s{seed}"
    recs = load_genomes(run_dir, "leader_follower", attach_embeddings=False)
    mod = moderated_genomes(recs)[:2000]
    scored = [(illicit_of(r), r) for r in mod]
    scored = [(s, r) for s, r in scored if s is not None]
    scored.sort(key=lambda t: (-t[0], str(t[1].get("id"))))
    top = [r for _, r in scored[:50]]
    prompts = [str(r.get("prompt") or "") for r in top]
    ids = [r.get("id") for r in top]
    # pairwise prompt identity / near-dup via exact string
    n_exact_pairs = 0
    for i in range(len(prompts)):
        for j in range(i + 1, len(prompts)):
            if prompts[i] == prompts[j]:
                n_exact_pairs += 1
    cache_disp = None
    if MPNET_CACHE.is_file():
        z = np.load(MPNET_CACHE)
        emb = z["embeddings"] if "embeddings" in z.files else z[z.files[0]]
        # mean pairwise (1-cos)/2 on L2-normalized rows
        X = emb.astype(np.float64)
        X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-12, None)
        sims = X @ X.T
        n = sims.shape[0]
        tri = sims[np.triu_indices(n, k=1)]
        cache_disp = float(np.mean((1.0 - tri) / 2.0))

    samples = [
        {"id": ids[i], "prompt_preview": prompts[i][:160].replace("\n", " ")}
        for i in range(min(5, len(prompts)))
    ]
    return {
        "dedupe_policy": "RQ2 metrics dedupe by genome id only; prompt text is not unique-keyed",
        "cell_stats": cell_stats,
        "outlier": {
            "seed": seed,
            "csv_disp": float(outlier["disp_mpnet_top50_b2000"]),
            "cache_disp": cache_disp,
            "n_top50": len(top),
            "n_unique_ids": len(set(ids)),
            "n_unique_prompts": len(set(prompts)),
            "exact_duplicate_prompt_pairs": n_exact_pairs,
            "prompt_samples": samples,
            "interpretation": (
                "Near-zero MPNet dispersion with distinct prompt strings → near-paraphrase "
                "elite concentration, not an aggregation artifact from identical strings."
                if len(set(prompts)) == len(prompts) and float(outlier["disp_mpnet_top50_b2000"]) < 0.01
                else "Inspect samples; dispersion may reflect exact repeats or mixed causes."
            ),
        },
    }


def check_rq3_labels() -> Dict[str, Any]:
    NOISE = {None, 0, -1, "0", "-1"}

    def sid_int(rec: Dict[str, Any]) -> Optional[int]:
        s = rec.get("species_id")
        if s in NOISE:
            return None
        try:
            i = int(s)
        except Exception:
            return None
        return None if i <= 0 else i

    # Archive status on LF runs
    archive_audit = []
    for dist, clus, folder in CELLS:
        if clus != "leader_follower":
            continue
        for seed in range(10):
            run_dir = LIVE / folder / f"rep{seed}_s{seed}"
            arch_path = run_dir / "archive.json"
            elites_path = run_dir / "elites.json"
            if not arch_path.is_file():
                continue
            archive = json.loads(arch_path.read_text(encoding="utf-8"))
            elites = json.loads(elites_path.read_text(encoding="utf-8")) if elites_path.is_file() else []
            arch_sids = [sid_int(r) if isinstance(r, dict) else None for r in archive]
            elite_sids = [sid_int(r) if isinstance(r, dict) else None for r in elites]
            raw_arch = [
                r.get("species_id") for r in archive if isinstance(r, dict)
            ]
            archive_audit.append(
                {
                    "folder": folder,
                    "seed": seed,
                    "n_archive": len(archive),
                    "archive_mature_frac": (sum(1 for s in arch_sids if s is not None) / len(arch_sids))
                    if arch_sids
                    else float("nan"),
                    "archive_all_unassigned": all(s is None for s in arch_sids),
                    "raw_archive_sid_counts": dict(Counter(str(x) for x in raw_arch)),
                    "elites_mature_frac": (sum(1 for s in elite_sids if s is not None) / len(elite_sids))
                    if elite_sids
                    else float("nan"),
                }
            )

    # Recompute K50/U50 medians from runs CSV and compare to TeX
    runs = list(csv.DictReader(RQ3_RUNS.open(encoding="utf-8")))
    # Prefer final + assigned mode rows if present; else aggregate by cell
    def _col(row: Dict[str, str], *names: str) -> Optional[float]:
        for n in names:
            if n in row and row[n] not in ("", None):
                try:
                    return float(row[n])
                except ValueError:
                    continue
        return None

    medians = {}
    for dist, clus, folder in CELLS:
        sub = [
            r
            for r in runs
            if r.get("distance") == dist
            and r.get("clustering") == clus
            and r.get("label_source", "final") in ("final", "", None)
            and r.get("mode", "assigned") in ("assigned", "obs", "", None)
        ]
        # If mode filter emptied, fall back to all rows for cell
        if not sub:
            sub = [r for r in runs if r.get("distance") == dist and r.get("clustering") == clus]
        # Prefer mode=assigned if column exists
        if any(r.get("mode") == "assigned" for r in runs):
            sub = [
                r
                for r in runs
                if r.get("distance") == dist
                and r.get("clustering") == clus
                and r.get("mode") == "assigned"
                and r.get("label_source", "final") == "final"
            ]
        ks = [_col(r, "K10_obs", "K50_obs", "K10", "K50") for r in sub]
        us = [_col(r, "U10_obs", "U50_obs", "U10", "U50") for r in sub]
        ks = [x for x in ks if x is not None]
        us = [x for x in us if x is not None]
        medians[(dist, clus)] = {
            "n": len(sub),
            "K50_median": _median(ks),
            "U50_median": _median(us),
        }

    tex = RQ3_TEX.read_text(encoding="utf-8")
    expected = {
        ("embedding", "leader_follower"): (5.5, 13.5),
        ("embedding", "dbscan"): (5.0, 0.5),
        ("nli_entail", "leader_follower"): (6.5, 32.0),
        ("nli_entail", "dbscan"): (1.0, 5.5),
    }
    compare = []
    for key, (ek, eu) in expected.items():
        got = medians[key]
        compare.append(
            {
                "cell": key,
                "tex_K50": ek,
                "tex_U50": eu,
                "csv_K50": got["K50_median"],
                "csv_U50": got["U50_median"],
                "K_match": abs(got["K50_median"] - ek) < 1e-6,
                "U_match": abs(got["U50_median"] - eu) < 1e-6,
            }
        )

    lf_all_unassigned = all(r["archive_all_unassigned"] for r in archive_audit)
    return {
        "convention": (
            "Mature = species_id > 0; LF archive write path sets species_id=-1; "
            "unassigned includes 0 (singleton/noise pool) and -1 (archived)."
        ),
        "lf_archive_all_unassigned_across_runs": lf_all_unassigned,
        "archive_audit_sample": archive_audit[:4],
        "n_lf_runs_audited": len(archive_audit),
        "median_compare_to_tex": compare,
        "tex_path": str(RQ3_TEX),
        "variants_path": str(RQ3_VAR) if RQ3_VAR.is_file() else None,
        "tex_present": "K_{50}" in tex or "K_{50}" in tex.replace(" ", ""),
    }


def check_rq1_provenance() -> Dict[str, Any]:
    if not POOL.is_file():
        return {"available": False, "reason": "p0_full.pkl missing"}
    with POOL.open("rb") as f:
        pool = pickle.load(f)
    try:
        import pandas as pd  # noqa: F401
    except Exception:
        pd = None
    if hasattr(pool, "columns"):  # DataFrame
        n_rows = int(len(pool))
        col = "run" if "run" in pool.columns else ("source_run" if "source_run" in pool.columns else None)
        by_run = dict(Counter(pool[col].astype(str).tolist())) if col else {}
        cols = list(pool.columns)
    else:
        rows = pool["rows"] if isinstance(pool, dict) and "rows" in pool else pool
        n_rows = len(rows)
        by_run = Counter()
        cols = sorted(rows[0].keys()) if rows and isinstance(rows[0], dict) else []
        for rec in rows:
            if isinstance(rec, dict):
                by_run[str(rec.get("run") or rec.get("source_run"))] += 1
        by_run = dict(by_run)
    return {
        "available": True,
        "n_rows": n_rows,
        "columns": cols[:30],
        "per_run_counts": by_run,
        "note": (
            "Provenance exists for optional per-source-archive RQ1 robustness."
        ),
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {}
    print("=== 1/4 Table 6 snapshot ===")
    report["table6"] = check_table6_snapshot()
    print(json.dumps(report["table6"], indent=2, default=str)[:2000])

    print("\n=== 2/4 IncDBSCAN star / MinPts=2 ===")
    report["incdbscan"] = check_incdbscan_star()
    print(json.dumps({k: v for k, v in report["incdbscan"].items() if k != "unittest_stdout_tail"}, indent=2, default=str)[:2500])

    print("\n=== 3/4 RQ2 uniqueness + MPNet outlier ===")
    report["rq2"] = check_rq2_uniqueness_and_outlier()
    print(json.dumps(report["rq2"], indent=2, default=str)[:2500])

    print("\n=== 4/4 RQ3 labels / K50 U50 ===")
    report["rq3"] = check_rq3_labels()
    print(json.dumps(report["rq3"], indent=2, default=str)[:2500])

    print("\n=== Optional provenance ===")
    report["rq1_provenance"] = check_rq1_provenance()
    print(json.dumps(report["rq1_provenance"], indent=2, default=str))

    out_json = OUT / "mandatory_checks.json"
    out_md = OUT / "mandatory_checks.md"
    out_json.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    out_md.write_text(_to_md(report), encoding="utf-8")
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    print(f"Rewrote {SUMMARY_MD}")
    return 0


def _to_md(report: Dict[str, Any]) -> str:
    t6 = report["table6"]
    inc = report["incdbscan"]
    rq2 = report["rq2"]
    rq3 = report["rq3"]
    lines = [
        "# Mandatory validation checklist",
        "",
        "## 1. Appendix Table 6 snapshot",
        "",
        f"- Online CSV: `{t6['csv_online']}`",
        f"- Embedding live ARI median [min, max]: "
        f"**{t6['embedding_live_ari']['median']:.3f}** "
        f"[{t6['embedding_live_ari']['min']:.3f}, {t6['embedding_live_ari']['max']:.3f}] "
        f"(matches Table 6: {t6['matches_table6_embedding']})",
        f"- NLI live ARI median [min, max]: "
        f"**{t6['nli_live_ari']['median']:.3f}** "
        f"[{t6['nli_live_ari']['min']:.3f}, {t6['nli_live_ari']['max']:.3f}] "
        f"(matches Table 6: {t6['matches_table6_nli']})",
        f"- Note: {t6['note']}",
        "",
        "## 2. IncDBSCAN correctness",
        "",
        f"- MinPts=2 star = connected component: **{inc['minpts2_connected_component']}**",
        f"- MinPts=5 star: batch n_mat={inc['minpts5_batch_n_mat']}, "
        f"inc n_mat={inc['minpts5_inc_n_mat']}, "
        f"core-promotion miss={inc['core_promotion_miss_precomputed']}",
        f"- Live insert_point: event=`{inc['live_insert_event']}`, "
        f"hub_core={inc['live_hub_became_core']}, leaf_core={inc['live_leaf_is_core']}, "
        f"miss={inc['live_core_promotion_miss']}",
        f"- Unittest return code: {inc['unittest_returncode']}",
        f"- Verdict: {inc['verdict']}",
        "",
        "## 3. RQ2 uniqueness / MPNet outlier",
        "",
        f"- Policy: {rq2['dedupe_policy']}",
    ]
    for c in rq2["cell_stats"]:
        lines.append(
            f"- `{c['folder']}`: shared-prompt-across-ids windows={c['budget_windows_with_shared_prompt_across_ids']}, "
            f"top50 prompt-dup runs={c['runs_with_duplicate_prompt_in_top50']}"
        )
    o = rq2["outlier"]
    lines += [
        f"- Outlier: lf_nli seed={o['seed']}, CSV D={o['csv_disp']:.6f}, cache D={o['cache_disp']}",
        f"- Unique ids/prompts in top-50: {o['n_unique_ids']}/{o['n_unique_prompts']}; "
        f"exact duplicate pairs={o['exact_duplicate_prompt_pairs']}",
        f"- Interpretation: {o['interpretation']}",
        "",
        "## 4. RQ3 labels / occupancy",
        "",
        f"- Convention: {rq3['convention']}",
        f"- LF archives all unassigned across audited runs: **{rq3['lf_archive_all_unassigned_across_runs']}**",
    ]
    for c in rq3["median_compare_to_tex"]:
        lines.append(
            f"- {c['cell']}: TeX K/U={c['tex_K50']}/{c['tex_U50']} vs CSV "
            f"{c['csv_K50']}/{c['csv_U50']} (K_match={c['K_match']}, U_match={c['U_match']})"
        )
    lines += [
        "",
        "## Optional RQ1 provenance",
        "",
        json.dumps(report.get("rq1_provenance", {}), indent=2),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
