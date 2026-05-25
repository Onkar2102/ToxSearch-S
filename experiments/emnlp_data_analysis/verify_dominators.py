"""Dominator species mix for TDI=1 topics (paper sanity check)."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from analysis_utils import (
    _evaluator_vector_key,
    dominates,
    global_pareto_annotate,
    load_unified_from_artifacts,
    member_dominated_mask,
)


def _species_label(row: Dict[str, Any]) -> str:
    sid = int(row.get("species_id") or 0)
    if sid > 0:
        return f"species_{sid}"
    source = str(row.get("source_file") or "")
    if "reserves.json" in source:
        return "reserves"
    if "archive.json" in source:
        return "archive"
    return "other"


def _dominator_species_counts(
    F: np.ndarray,
    valid: Sequence[Dict[str, Any]],
    topic_idxs: List[int],
    *,
    restrict_f0: bool,
    on_f0: np.ndarray,
) -> Tuple[Counter[str], int, int]:
    """Tally dominator species over topic_idxs × candidates."""
    counts: Counter[str] = Counter()
    pair_count = 0
    member_pair_count = 0
    n = F.shape[0]
    candidate_idxs = np.where(on_f0)[0].tolist() if restrict_f0 else list(range(n))

    for gi in topic_idxs:
        g_vec = F[gi]
        member_had = False
        for gj in candidate_idxs:
            if gj == gi:
                continue
            if dominates(F[gj], g_vec):
                counts[_species_label(valid[gj])] += 1
                pair_count += 1
                member_had = True
        if member_had:
            member_pair_count += 1
    return counts, pair_count, member_pair_count


def _share(counts: Counter[str], key: str) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts[key] / total


def analyze_evaluator(
    rows: Sequence[Dict[str, Any]],
    evaluator: str,
) -> Dict[str, Any]:
    vec_key = _evaluator_vector_key(evaluator)
    valid = [r for r in rows if r.get(vec_key) is not None]
    F = np.vstack([r[vec_key] for r in valid])
    _, on_f0, _ = global_pareto_annotate(F)
    dominated = member_dominated_mask(F)

    by_topic: Dict[int, List[int]] = defaultdict(list)
    for i, r in enumerate(valid):
        sid = int(r.get("species_id") or 0)
        if sid > 0:
            by_topic[sid].append(i)

    f0_species_counts = Counter(
        _species_label(valid[i]) for i in range(len(valid)) if on_f0[i]
    )

    topic_results: List[Dict[str, Any]] = []
    fully_dominated_ids: List[int] = []

    for sid, idxs in sorted(by_topic.items()):
        tdi = float(np.mean(dominated[idxs]))
        n_on_f0 = int(np.sum(on_f0[idxs]))
        fully_dom = n_on_f0 == 0
        entry: Dict[str, Any] = {
            "species_id": sid,
            "n_members": len(idxs),
            "tdi": round(tdi, 4),
            "fully_dominated": fully_dom,
            "n_on_f0": n_on_f0,
        }
        if fully_dom:
            fully_dominated_ids.append(sid)
            all_g, all_pairs, members_with = _dominator_species_counts(
                F, valid, idxs, restrict_f0=False, on_f0=on_f0
            )
            f0_g, f0_pairs, f0_members_with = _dominator_species_counts(
                F, valid, idxs, restrict_f0=True, on_f0=on_f0
            )
            entry.update({
                "dominator_pairs_all_g": all_pairs,
                "dominator_pairs_f0_only": f0_pairs,
                "members_with_dominator_all_g": members_with,
                "members_with_dominator_f0_only": f0_members_with,
                "dominator_species_all_g": dict(all_g),
                "dominator_species_f0_only": dict(f0_g),
                "share_dominators_from_2421_all_g": round(_share(all_g, "species_2421"), 4),
                "share_dominators_from_2421_f0_only": round(_share(f0_g, "species_2421"), 4),
                "share_dominators_from_archive_all_g": round(_share(all_g, "archive"), 4),
                "share_dominators_from_archive_f0_only": round(_share(f0_g, "archive"), 4),
            })
        topic_results.append(entry)

    # Aggregate over fully-dominated topics
    agg_all = Counter()
    agg_f0 = Counter()
    for t in topic_results:
        if not t.get("fully_dominated"):
            continue
        agg_all.update(t.get("dominator_species_all_g", {}))
        agg_f0.update(t.get("dominator_species_f0_only", {}))

    return {
        "evaluator": evaluator,
        "n_genomes": len(valid),
        "n_f0": int(np.sum(on_f0)),
        "f0_species_composition": dict(f0_species_counts),
        "fully_dominated_topic_ids": fully_dominated_ids,
        "topics": topic_results,
        "aggregate_fully_dominated": {
            "dominator_species_all_g": validate_counter(agg_all),
            "dominator_species_f0_only": validate_counter(agg_f0),
            "share_from_2421_all_g": round(_share(agg_all, "species_2421"), 4),
            "share_from_2421_f0_only": round(_share(agg_f0, "species_2421"), 4),
            "share_from_archive_all_g": round(_share(agg_all, "archive"), 4),
            "share_from_archive_f0_only": round(_share(agg_f0, "archive"), 4),
            "total_dominator_pairs_all_g": sum(agg_all.values()),
            "total_dominator_pairs_f0_only": sum(agg_f0.values()),
        },
    }


def validate_counter(c: Counter) -> Dict[str, int]:
    return {k: int(v) for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))}


def main() -> None:
    results_dir = Path(__file__).resolve().parent / "results"
    run_id = "20260211_2122"
    rows, _ = load_unified_from_artifacts(results_dir, run_id)

    out_dir = results_dir / "phase_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {"run_id": run_id, "evaluators": {}}
    for evaluator in ("google", "openai"):
        summary["evaluators"][evaluator] = analyze_evaluator(rows, evaluator)

    json_path = out_dir / "dominator_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Dominator species composition (fully-dominated topics, TDI=1)",
        "",
        f"Run: `{run_id}`",
        "",
    ]
    for ev in ("google", "openai"):
        ev_data = summary["evaluators"][ev]
        agg = ev_data["aggregate_fully_dominated"]
        md_lines.extend([
            f"## {ev}",
            "",
            f"- Fully dominated topics: {ev_data['fully_dominated_topic_ids']}",
            f"- F0 size: {ev_data['n_f0']}",
            f"- F0 species composition: `{ev_data['f0_species_composition']}`",
            f"- Aggregate dominator share from species_2421 (all G): **{agg['share_from_2421_all_g']:.1%}**",
            f"- Aggregate dominator share from species_2421 (F0 only): **{agg['share_from_2421_f0_only']:.1%}**",
            f"- Aggregate dominator share from archive (F0 only): **{agg['share_from_archive_f0_only']:.1%}**",
            "",
            "| topic | pairs (F0) | share 2421 (F0) | top dominators (F0) |",
            "|-------|------------|-----------------|---------------------|",
        ])
        for t in ev_data["topics"]:
            if not t.get("fully_dominated"):
                continue
            top = sorted(
                t.get("dominator_species_f0_only", {}).items(),
                key=lambda kv: -kv[1],
            )[:5]
            top_str = ", ".join(f"{k}:{v}" for k, v in top)
            md_lines.append(
                f"| {t['species_id']} | {t['dominator_pairs_f0_only']} | "
                f"{t['share_dominators_from_2421_f0_only']:.1%} | {top_str} |"
            )
        md_lines.append("")

    md_path = out_dir / "dominator_summary.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps(summary["evaluators"]["google"]["aggregate_fully_dominated"], indent=2))
    print(json.dumps(summary["evaluators"]["openai"]["aggregate_fully_dominated"], indent=2))
    print(f"Wrote {json_path} and {md_path}")


if __name__ == "__main__":
    main()
