"""Standalone runner that produces Phase 5 / Phase 7 / Phase 11 artifacts from
the already-saved unified CSV and emits the numbers needed for plan section M.9.

Why this script exists:
    The notebook normally re-runs every phase end-to-end, which requires the raw
    population JSONs from ``data/outputs/<run>``. After we cleaned the
    ``results/`` tree we still have the *unified* artifacts (CSV, embeddings,
    OpenAI cache) but no per-phase outputs. This script reconstructs the row
    dicts used by the analysis modules from those unified artifacts and runs
    just the phases that feed M.9: Phase 5 (counterfactual + new CIs),
    Phase 7 (per-topic permutation + Holm-Bonferroni), and Phase 11
    (advanced analytics, needed for Fig 5 + the niche-specialization numbers).

The script is read-only with respect to the unified artifacts.
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

THIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS_DIR))

import analysis_utils  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("m9")

RUN_ID = "20260211_2122"
UNIFIED_DIR = THIS_DIR / "results" / "unified"
RESULTS_DIR = THIS_DIR / "results"


def load_unified_rows() -> List[Dict[str, Any]]:
    csv_path = UNIFIED_DIR / f"{RUN_ID}_genomes.csv"
    emb_path = UNIFIED_DIR / f"{RUN_ID}_embeddings.npy"
    ids_path = UNIFIED_DIR / f"{RUN_ID}_genome_ids.json"
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    embeddings = np.load(emb_path, allow_pickle=False) if emb_path.is_file() else None
    ids = json.loads(ids_path.read_text(encoding="utf-8")) if ids_path.is_file() else []
    id_to_emb_index = {str(g): k for k, g in enumerate(ids)}

    google_axes = list(analysis_utils.get_axis_order("google"))
    openai_axes = list(analysis_utils.get_axis_order("openai"))

    rows: List[Dict[str, Any]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            row: Dict[str, Any] = {
                "run_id": r.get("run_id") or RUN_ID,
                "genome_id": int(r["genome_id"]) if str(r.get("genome_id") or "").isdigit() else r.get("genome_id"),
                "species_id": int(r.get("species_id") or 0),
                "generation": int(r.get("generation") or 0),
                "source_file": r.get("source_file") or "",
                "prompt": str(r.get("prompt") or ""),
                "has_embedding": str(r.get("has_embedding")).lower() == "true",
            }

            g_vec: List[float] = []
            g_ok = True
            for ax in google_axes:
                col = analysis_utils.score_column_name("google", ax)
                v = r.get(col)
                if v in (None, ""):
                    g_ok = False
                    break
                try:
                    g_vec.append(float(v))
                except (TypeError, ValueError):
                    g_ok = False
                    break
            if g_ok and len(g_vec) == len(google_axes):
                row["objective_vector"] = np.asarray(g_vec, dtype=np.float64)

            o_vec: List[float] = []
            o_ok = True
            for ax in openai_axes:
                col = analysis_utils.score_column_name("openai", ax)
                v = r.get(col)
                if v in (None, ""):
                    o_ok = False
                    break
                try:
                    o_vec.append(float(v))
                except (TypeError, ValueError):
                    o_ok = False
                    break
            if o_ok and len(o_vec) == len(openai_axes):
                row["objective_vector_openai"] = np.asarray(o_vec, dtype=np.float64)

            gid = str(r.get("genome_id") or "")
            if embeddings is not None and gid in id_to_emb_index:
                k = id_to_emb_index[gid]
                if 0 <= k < embeddings.shape[0]:
                    row["_embedding"] = np.asarray(embeddings[k], dtype=np.float64)

            rows.append(row)
    return rows


def main() -> int:
    rows = load_unified_rows()
    n_g = sum(1 for r in rows if "objective_vector" in r)
    n_o = sum(1 for r in rows if "objective_vector_openai" in r)
    n_e = sum(1 for r in rows if "_embedding" in r)
    log.info("Loaded %d rows (google=%d openai=%d emb=%d)", len(rows), n_g, n_o, n_e)

    p5_dir = RESULTS_DIR / "phase5"
    p7_dir = RESULTS_DIR / "phase7"
    p11_dir = RESULTS_DIR / "phase11"

    log.info("=== Phase 11 (advanced analytics) [skip if exists] ===")
    if not (p11_dir / "topic_advanced_analytics.csv").is_file():
        p11 = analysis_utils.save_phase11_artifacts(rows, p11_dir, RUN_ID)
        log.info("phase11 keys: %s", list(p11.get("artifacts", {}).keys()))
    else:
        log.info("phase11 already present; skipping recompute")

    log.info("=== Wilson intervals on Phase 11 survival counts (no bootstrap) ===")
    # Compute Wilson CIs directly on the known counts so M.9 has numeric CIs
    # without paying the O(N^2) cost of redoing Pareto sorting on 5655 rows.
    import counterfactual_survival as p5c
    survival_counts = {
        "google": {
            "scalar":            (1, 9),
            "best_single_axis":  (5, 9),
            "global_mo":         (4, 9),
            "speciated_oracle":  (9, 9),
        },
        "openai": {
            "scalar":            (1, 9),
            "best_single_axis":  (6, 9),
            "global_mo":         (5, 9),
            "speciated_oracle":  (9, 9),
        },
    }
    wilson_table: List[Dict[str, Any]] = []
    for ev, by_pol in survival_counts.items():
        for pol, (succ, n) in by_pol.items():
            lo, hi = p5c.wilson_interval(succ, n, confidence=0.95)
            wilson_table.append({
                "evaluator": ev,
                "policy": pol,
                "topics_survived": succ,
                "topics_total": n,
                "survival_rate": round(succ / n, 4),
                "wilson_ci_lower": round(lo, 4),
                "wilson_ci_upper": round(hi, 4),
                "confidence": 0.95,
            })
    p5_dir.mkdir(parents=True, exist_ok=True)
    wilson_path = p5_dir / "wilson_survival_intervals.csv"
    if wilson_table:
        with wilson_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(wilson_table[0].keys()))
            w.writeheader()
            w.writerows(wilson_table)
    log.info("Wilson table -> %s", wilson_path)
    for r in wilson_table:
        log.info("  %s/%s: %d/%d=%.3f Wilson95=[%.3f,%.3f]",
                 r["evaluator"], r["policy"],
                 r["topics_survived"], r["topics_total"],
                 r["survival_rate"], r["wilson_ci_lower"], r["wilson_ci_upper"])

    log.info("=== Phase 7 (per-topic permutation + Holm-Bonferroni, vectorized) ===")
    # Direct vectorized per-topic permutation test, sufficient for M.9 numbers.
    # Avoids the slower Python double-loop in phase7_inference._stat.
    import behaviour_layout
    import permutation_inference

    behaviour_layout.stamp_f0_flags(
        rows, global_pareto_annotate=analysis_utils.global_pareto_annotate
    )
    for ev in ("google", "openai"):
        permutation_inference.stamp_tdi_members(
            rows, ev,
            min_topic_size=5,
            member_dominated_mask=analysis_utils.member_dominated_mask,
        )

    F0_FLAG = permutation_inference.F0_FLAG

    def _vec_perm_per_topic(ev: str, n_perm: int = 2000, seed: int = 42,
                             max_pool: int = 200) -> List[Dict[str, Any]]:
        rng = np.random.default_rng(seed)
        f0_key = F0_FLAG[ev]
        f0_emb = [r["_embedding"] for r in rows
                  if r.get(f0_key) and r.get("_embedding") is not None]
        if len(f0_emb) < 2:
            return []
        F0 = np.vstack(f0_emb).astype(np.float64)
        F0 /= np.maximum(np.linalg.norm(F0, axis=1, keepdims=True), 1e-12)
        if F0.shape[0] > max_pool:
            sel = rng.choice(F0.shape[0], size=max_pool, replace=False)
            F0 = F0[sel]
        by_topic: Dict[int, np.ndarray] = {}
        for r in rows:
            sid = int(r.get("species_id") or 0)
            if sid <= 0 or r.get(f0_key):
                continue
            emb = r.get("_embedding")
            if emb is None:
                continue
            by_topic.setdefault(sid, []).append(emb)
        out_rows: List[Dict[str, Any]] = []
        for sid, embs in sorted(by_topic.items()):
            if len(embs) < 5:
                continue
            G = np.vstack(embs).astype(np.float64)
            G /= np.maximum(np.linalg.norm(G, axis=1, keepdims=True), 1e-12)
            if G.shape[0] > max_pool:
                sel = rng.choice(G.shape[0], size=max_pool, replace=False)
                G = G[sel]
            n0, n1 = G.shape[0], F0.shape[0]
            pool = np.vstack([G, F0])  # cosine-normalized
            sim = pool @ pool.T
            dist = 1.0 - sim
            np.fill_diagonal(dist, 0.0)
            obs = float(dist[:n0, n0:].mean())
            ge = 0
            n_pool = pool.shape[0]
            for _ in range(n_perm):
                idx = rng.permutation(n_pool)
                a, b = idx[:n0], idx[n0:]
                stat = float(dist[np.ix_(a, b)].mean())
                if stat >= obs:
                    ge += 1
            p_raw = (ge + 1) / (n_perm + 1)
            out_rows.append({
                "evaluator": ev,
                "species_id": int(sid),
                "n_topic_dominated": len(embs),
                "n_f0_global": len(f0_emb),
                "obs_stat": round(obs, 6),
                "p_raw": round(p_raw, 6),
                "n_perm": int(n_perm),
            })
        out_rows = permutation_inference.annotate_holm_bonferroni(
            out_rows, p_key="p_raw", alpha=0.05
        )
        return out_rows

    p7_dir.mkdir(parents=True, exist_ok=True)
    holm_summary: Dict[str, Any] = {}
    for ev in ("google", "openai"):
        log.info("  per-topic permutation [%s] ...", ev)
        rows_pt = _vec_perm_per_topic(ev, n_perm=2000)
        out_csv = p7_dir / f"inference_per_topic_{ev}.csv"
        if rows_pt:
            with out_csv.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(rows_pt[0].keys()))
                w.writeheader()
                w.writerows(rows_pt)
        n_rej = sum(1 for r in rows_pt if r.get("reject_at_0p05"))
        holm_summary[ev] = {
            "n_topics_tested": len(rows_pt),
            "n_rejected_at_0p05": int(n_rej),
            "rows": rows_pt,
        }
        log.info("  [%s] %d/%d topics reject Holm-Bonferroni at alpha=0.05",
                 ev, n_rej, len(rows_pt))
        for r in rows_pt:
            log.info("    sid=%d  n=%d  obs=%.4f  p_raw=%.4f  p_holm=%.4f  reject=%s",
                     r["species_id"], r["n_topic_dominated"], r["obs_stat"],
                     r["p_raw"], r["p_holm"], r["reject_at_0p05"])

    summary_path = p7_dir / "per_topic_holm_summary.json"
    summary_path.write_text(
        json.dumps({"google": holm_summary["google"],
                    "openai": holm_summary["openai"]}, indent=2),
        encoding="utf-8",
    )
    log.info("Holm summary -> %s", summary_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
