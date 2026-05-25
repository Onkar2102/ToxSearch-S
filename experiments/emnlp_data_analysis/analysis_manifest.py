"""Phase 10: manifest, results snippet, validation checklist, figure index."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


PHASE_DIRS = (
    "gate0_smoke.json",
    "unified",
    "phase2",
    "phase3",
    "phase4",
    "phase5",
    "phase6",
    "phase7",
    "phase8",
    "phase9",
    "phase11",
    "phase12",
)


def _collect_artifacts(results_dir: Path) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for phase in (
        "phase2", "phase3", "phase4", "phase5", "phase6",
        "phase7", "phase8", "phase9", "phase11", "phase12",
    ):
        pdir = results_dir / phase
        if not pdir.is_dir():
            continue
        files = sorted(str(f.relative_to(results_dir)) for f in pdir.rglob("*") if f.is_file())
        out[phase] = files
    for name in ("gate0_smoke.json", "phase1_manifest.json"):
        fp = results_dir / name
        if fp.is_file():
            out.setdefault("phase0_1", []).append(name)
    unified = results_dir / "unified"
    if unified.is_dir():
        out["unified"] = sorted(str(f.relative_to(results_dir)) for f in unified.iterdir() if f.is_file())
    return out


def build_analysis_manifest(results_dir: Path, *, run_id: str = "") -> Dict[str, Any]:
    results_dir = Path(results_dir)
    manifest: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(results_dir),
        "run_id": run_id,
        "phases": _collect_artifacts(results_dir),
    }
    for key in (
        "phase1_manifest.json",
        "phase2/phase2_global_pareto.json",
        "phase3/phase3_manifest.json",
        "phase11/phase11_summary.json",
        "phase12/phase12_summary.json",
    ):
        fp = results_dir / key
        if fp.is_file():
            try:
                manifest[key.replace("/", "_").replace(".", "_")] = json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                pass
    out_path = results_dir / "ANALYSIS_MANIFEST.json"
    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_validation_checklist(
    results_dir: Path,
    *,
    anchors: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compare measured stats to paper anchor numbers."""
    results_dir = Path(results_dir)
    anchors = anchors or {
        "n_f0_google": 100,
        "n_fully_dominated_topics_google": 5,
        "n_topics": 9,
        "n_generations_collapsed_to_one": 18,
        "n_genomes": 5655,
        "google_mo_topics_survived": 4,
        "openai_mo_topics_survived": 5,
        "speciated_oracle_topics_survived": 9,
        "n_topics_no_single_axis_recovers_google": 3,
    }
    measured: Dict[str, Any] = {}

    p2 = results_dir / "phase2" / "phase2_global_pareto.json"
    if p2.is_file():
        s = json.loads(p2.read_text(encoding="utf-8"))
        measured["n_f0_google"] = s.get("n_f0")

    p3g = results_dir / "phase3" / "google" / "topic_domination_global_stats.json"
    if p3g.is_file():
        s = json.loads(p3g.read_text(encoding="utf-8"))
        measured["n_fully_dominated_topics_google"] = s.get("n_fully_dominated_topics")
        measured["n_topics"] = s.get("n_topics_summarized")
        measured["n_topics_summarized"] = s.get("n_topics_summarized")

    p4 = results_dir / "phase4"
    for stats_file in p4.glob("timeseries_*_stats.json"):
        s = json.loads(stats_file.read_text(encoding="utf-8"))
        measured["n_generations_collapsed_to_one"] = s.get("n_generations_collapsed_to_one")
        break

    unified_stats = list((results_dir / "unified").glob("*_stats.json"))
    if unified_stats:
        s = json.loads(unified_stats[0].read_text(encoding="utf-8"))
        measured["n_genomes"] = s.get("n_genomes")

    p11 = results_dir / "phase11" / "phase11_summary.json"
    if p11.is_file():
        s = json.loads(p11.read_text(encoding="utf-8"))
        for key in (
            "google_mo_topics_survived",
            "openai_mo_topics_survived",
            "speciated_oracle_topics_survived",
            "n_topics_no_single_axis_recovers_google",
            "n_topics_no_single_axis_recovers_openai",
        ):
            if key in s:
                measured[key] = s.get(key)

    rows = []
    for key, anchor in anchors.items():
        m = measured.get(key)
        rows.append({
            "metric": key,
            "anchor": anchor,
            "measured": m,
            "delta": (m - anchor) if isinstance(m, (int, float)) and isinstance(anchor, (int, float)) else None,
        })

    checklist = {"anchors": anchors, "measured": measured, "rows": rows}
    out_path = results_dir / "validation_checklist.json"
    out_path.write_text(json.dumps(checklist, indent=2), encoding="utf-8")
    return checklist


def build_results_snippet(results_dir: Path) -> str:
    results_dir = Path(results_dir)
    bullets: List[str] = []

    p2 = results_dir / "phase2" / "phase2_global_pareto.json"
    if p2.is_file():
        s = json.loads(p2.read_text(encoding="utf-8"))
        bullets.append(
            f"- Global Pareto (Google): |F₀|={s.get('n_f0')}, "
            f"fraction on F₀={s.get('f0_fraction', 0):.3f}, fronts={s.get('n_fronts')}."
        )

    rank_path = results_dir / "phase3" / "evaluator_rank_agreement_summary.json"
    if rank_path.is_file():
        s = json.loads(rank_path.read_text(encoding="utf-8"))
        bullets.append(
            f"- Rank agreement (both evaluators, n={s.get('n_genomes_both_evaluators')}): "
            f"Spearman ρ={s.get('spearman_rank_correlation')}, "
            f"same rank={s.get('frac_same_rank')}, both on F₀={s.get('n_both_on_f0')}."
        )

    p3g = results_dir / "phase3" / "google" / "topic_domination_global_stats.json"
    if p3g.is_file():
        s = json.loads(p3g.read_text(encoding="utf-8"))
        bullets.append(
            f"- Topic domination (Google): {s.get('n_fully_dominated_topics')}/"
            f"{s.get('n_topics_summarized')} topics fully dominated; "
            f"{s.get('n_distinct_fully_dominated')} distinct+dominated."
        )

    p5cmp = results_dir / "phase5" / "counterfactual_evaluator_comparison.json"
    if p5cmp.is_file():
        s = json.loads(p5cmp.read_text(encoding="utf-8"))
        bullets.append(
            f"- Counterfactual hierarchy preserved: Google={s.get('hierarchy_google')}, "
            f"OpenAI={s.get('hierarchy_openai')}."
        )

    p7 = results_dir / "phase7" / "inference_summary.json"
    if p7.is_file():
        s = json.loads(p7.read_text(encoding="utf-8"))
        for ev in ("google", "openai"):
            pe = s.get(ev, {})
            if pe and not pe.get("skipped"):
                bullets.append(
                    f"- Permutation ({ev}): p={pe.get('p_value')}, "
                    f"mean d_g F₀ vs dominated={pe.get('obs_stat')}."
                )
        holm = s.get("per_topic_holm") or {}
        for ev, info in holm.items():
            if info:
                bullets.append(
                    f"- Per-topic Holm-Bonferroni ({ev}): "
                    f"{info.get('n_rejected_at_0p05')}/{info.get('n_topics_tested')} "
                    f"topics reject H0 at α=0.05."
                )

    p9 = results_dir / "phase9" / "cross_evaluator_robustness.json"
    if p9.is_file():
        s = json.loads(p9.read_text(encoding="utf-8"))
        bullets.append(
            f"- Cross-evaluator Jaccard (fully dominated topics): "
            f"{s.get('jaccard_fully_dominated_topics')}."
        )

    p12 = results_dir / "phase12" / "phase12_summary.json"
    if p12.is_file():
        s = json.loads(p12.read_text(encoding="utf-8"))
        bullets.append(
            f"- Phase 12 topic labels: {s.get('n_topics')} topics labelled "
            f"(LLM={s.get('llm_enabled')}, "
            f"cache_hits={s.get('n_label_cache_hits')}, "
            f"api_calls={s.get('n_label_api_calls')})."
        )

    p11 = results_dir / "phase11" / "phase11_summary.json"
    if p11.is_file():
        s = json.loads(p11.read_text(encoding="utf-8"))
        n_topics = s.get("n_topics")
        bullets.append(
            f"- Phase 11 advanced analytics: best single Google axis = "
            f"{s.get('best_single_axis_google', {}).get('axis')} "
            f"({s.get('best_single_axis_google', {}).get('topics_survived')}/{n_topics}); "
            f"best single OpenAI axis = "
            f"{s.get('best_single_axis_openai', {}).get('axis')} "
            f"({s.get('best_single_axis_openai', {}).get('topics_survived')}/{n_topics}); "
            f"global Pareto retains {s.get('google_mo_topics_survived')}/{n_topics} (G), "
            f"{s.get('openai_mo_topics_survived')}/{n_topics} (O); "
            f"speciated oracle = {s.get('speciated_oracle_topics_survived')}/{n_topics}."
        )
        bullets.append(
            f"- Topics not recovered by ANY single axis: "
            f"{s.get('n_topics_no_single_axis_recovers_google')} (Google), "
            f"{s.get('n_topics_no_single_axis_recovers_openai')} (OpenAI)."
        )

    text = "# EMNLP analysis results\n\n" + "\n".join(bullets) + "\n"
    (results_dir / "RESULTS_SNIPPET.md").write_text(text, encoding="utf-8")
    return text


PAPER_FIGURE_INDEX = """# Paper figure index

Main figures; appendix PCPs under `figures/appendix/`.

| File | Section / role |
|------|----------------|
| phase4/figures/fig1_temporal_species_count.png | Fig 1A — species collapse + reserves |
| phase6/figures/fig1_umap_dual.png | Fig 1B — embedding layout, Google + OpenAI panels |
| phase2/figures/google/pymoo_pcp_pareto_google.png | Fig 2 — global Pareto PCP (Google) |
| phase2/figures/openai/pymoo_pcp_pareto_openai.png | Fig 2 — global Pareto PCP (OpenAI) |
| phase5/figures/fig3_counterfactual_dual_panel.png | Fig 3 — counterfactual retention |
| phase3/figures/topic_tdi_google_vs_openai.png | Topic TDI comparison |
| phase3/figures/topic_domination_heatmap_dual.png | Topic domination structure (dual) |
| phase3/figures/topic_domination_graph_google.png | Topic domination graph (Google) |
| phase6/figures/topic_axis_heatmap_dual.png | Per-topic axis profiles (dual) |
| phase6/figures/fig_dg_dp_dual.png | Topic centroid d_g vs d_p (dual) |
| phase7/figures/rank_contingency_heatmap.png | Rank contingency |
| phase8/figures/epsilon_sweep_topics.png | Sensitivity appendix |
| phase9/figures/cross_evaluator_summary.png | Robustness summary |
| phase11/figures/fig11_single_axis_survival.png | Single-axis vs MO vs oracle survival |
| phase11/figures/fig11_topic_profile_vs_domination.png | Per-topic profile, TDI, NSI |
| phase11/figures/fig11_distinct_but_dominated.png | Distinct-but-dominated topics |
| phase12/figures/fig5_niche_specialization_vs_tdi_labeled.png | Fig 5 — NSI vs TDI (labelled topics) |
"""


def write_paper_figure_index(results_dir: Path) -> str:
    fig_dir = Path(results_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    path = fig_dir / "PAPER_FIGURE_INDEX.md"
    path.write_text(PAPER_FIGURE_INDEX, encoding="utf-8")
    return str(path)


__all__ = [
    "build_analysis_manifest",
    "build_results_snippet",
    "build_validation_checklist",
    "write_paper_figure_index",
]
