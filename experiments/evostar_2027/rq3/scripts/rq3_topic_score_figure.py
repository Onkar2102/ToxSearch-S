#!/usr/bin/env python3
"""RQ3 Fig.2: toxicity-by-species box+strip matching preprint Fig.4 style.

Each panel (one config):
  - take one illustrative seed with the richest top-K species list
  - one row per species (K<=10), labeled by top-10 TF-IDF keywords
  - horizontal box + gray strip of member illicit scores
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments/evostar_2027/rq2/scripts"))
from rq2_metrics import NOISE_SIDS, illicit_of, load_genomes, moderated_genomes  # noqa: E402

LIVE = ROOT / "data/outputs/evostar2027"
REP = ROOT / "experiments/evostar_2027/rq3/reports/rq3_representatives.csv"
OUT_REPORTS = ROOT / "experiments/evostar_2027/rq3/reports"
FIG = ROOT / "notes/rough_draft/figures"

LIVE_DIR = {
    ("embedding", "leader_follower"): "lf_embedding",
    ("embedding", "dbscan"): "inc_embedding",
    ("nli_entail", "leader_follower"): "lf_nli",
    ("nli_entail", "dbscan"): "inc_nli",
}
CELLS: List[Tuple[str, str, str]] = [
    ("embedding", "leader_follower", r"LF $d_E$"),
    ("embedding", "dbscan", r"Inc $d_E$"),
    ("nli_entail", "leader_follower", r"LF $d_{\neg E}$"),
    ("nli_entail", "dbscan", r"Inc $d_{\neg E}$"),
]
# Distinct per-row colors like the preprint figure.
ROW_COLORS = [
    "#2ca02c",
    "#8c564b",
    "#c49a6c",
    "#9467bd",
    "#1f77b4",
    "#7f7f7f",
    "#c5b0d5",
    "#d62728",
    "#2ca02c",
    "#e377c2",
]
TOPIC_LABEL = {
    "cyber": "cyber",
    "hate": "hate",
    "privacy": "privacy",
    "weapons": "weapons",
    "nonviolent_crime": "nonviolent crime",
    "violent_crime": "violent crime",
    "self_harm": "self-harm",
    "sexual": "sexual",
    "bio_chem": "bio/chem",
    "specialized_advice": "specialized advice",
    "mixed": "mixed",
}

plt.rcParams.update(
    {
        "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 8.0,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z\-']+")


def _sid(rec: Dict[str, Any]) -> Optional[int]:
    sid = rec.get("species_id")
    if sid in NOISE_SIDS:
        return None
    try:
        return int(sid)
    except (TypeError, ValueError):
        return None


def load_reps() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with REP.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            topic = r.get("topic") or "mixed"
            if topic not in TOPIC_LABEL:
                topic = "mixed"
            rows.append(
                {
                    "distance": r["distance"],
                    "clustering": r["clustering"],
                    "seed": int(r["seed"]),
                    "rank": int(r["rank"]),
                    "species_id": int(r["species_id"]),
                    "topic": topic,
                    "source_illicit": float(r["source_illicit"]),
                }
            )
    return rows


def choose_seed(reps: Sequence[Dict[str, Any]], d: str, c: str) -> int:
    """Prefer the seed with the most ranked species; break ties by lowest seed."""
    by_seed: Dict[int, int] = defaultdict(int)
    for r in reps:
        if r["distance"] == d and r["clustering"] == c:
            by_seed[r["seed"]] += 1
    if not by_seed:
        return 0
    best = sorted(by_seed.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return int(best)


def collect_species_bundle(
    d: str, c: str, seed: int, reps: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Load one run; return top-K species with prompts, scores, prototype topic."""
    run_reps = sorted(
        [r for r in reps if r["distance"] == d and r["clustering"] == c and r["seed"] == seed],
        key=lambda r: r["rank"],
    )
    run = LIVE / LIVE_DIR[(d, c)] / f"rep{seed}_s{seed}"
    gens = moderated_genomes(load_genomes(run, c, attach_embeddings=False))[:2000]
    by_sid_scores: Dict[int, List[float]] = defaultdict(list)
    by_sid_prompts: Dict[int, List[str]] = defaultdict(list)
    for g in gens:
        sid = _sid(g)
        if sid is None:
            continue
        s = illicit_of(g)
        if s is None:
            continue
        by_sid_scores[sid].append(float(s))
        prompt = str(g.get("prompt") or "").strip()
        if prompt:
            by_sid_prompts[sid].append(prompt)

    out: List[Dict[str, Any]] = []
    for r in run_reps:
        sid = r["species_id"]
        scores = by_sid_scores.get(sid, [])
        prompts = by_sid_prompts.get(sid, [])
        if not scores:
            continue
        out.append(
            {
                "rank": r["rank"],
                "species_id": sid,
                "topic": r["topic"],
                "scores": scores,
                "prompts": prompts,
                "source_illicit": r["source_illicit"],
            }
        )
    return out


def keyword_labels(species: Sequence[Dict[str, Any]], n_words: int = 10) -> List[str]:
    """Top-n TF-IDF words per species (one concatenated document each)."""
    docs = []
    for sp in species:
        text = " ".join(sp["prompts"]) if sp["prompts"] else "empty"
        docs.append(text)
    if not docs:
        return []
    try:
        vec = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            max_features=5000,
            token_pattern=r"[a-zA-Z][a-zA-Z\-']{2,}",
            min_df=1,
            sublinear_tf=True,
        )
        x = vec.fit_transform(docs)
        terms = np.asarray(vec.get_feature_names_out())
    except ValueError:
        return [f"species {sp['rank']}" for sp in species]

    labels: List[str] = []
    for i in range(x.shape[0]):
        row = x.getrow(i).toarray().ravel()
        if float(row.max()) <= 0:
            labels.append(f"species {species[i]['rank']}")
            continue
        idx = np.argsort(row)[::-1][:n_words]
        words = [str(terms[j]) for j in idx if row[j] > 0]
        labels.append(", ".join(words[:n_words]) if words else f"species {species[i]['rank']}")
    return labels


def effective_n1(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 0:
        return float("nan")
    h = 0.0
    for v in counts.values():
        p = v / total
        if p > 0:
            h -= p * math.log(p)
    return float(math.exp(h))


def main() -> int:
    reps = load_reps()
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), layout="constrained", sharex=True)
    rng = np.random.default_rng(2027)

    coverage_species: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
    for r in reps:
        coverage_species[(r["distance"], r["clustering"])][r["topic"]] += 1

    panel_meta = []
    for ax, (d, c, title) in zip(axes.ravel(), CELLS):
        seed = choose_seed(reps, d, c)
        species = collect_species_bundle(d, c, seed, reps)
        # Keep fitness-rank order (rank 1 at top).
        species = sorted(species, key=lambda sp: sp["rank"])
        labels = keyword_labels(species, n_words=10)
        # Plot top rank at top of axis.
        plot_species = list(reversed(species))
        plot_labels = list(reversed(labels))
        plot_vals = [sp["scores"] for sp in plot_species]
        positions = list(range(len(plot_species)))

        for i, (vals, sp) in enumerate(zip(plot_vals, plot_species)):
            color = ROW_COLORS[sp["rank"] % len(ROW_COLORS)]
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(
                vals,
                np.full(len(vals), i) + jitter,
                s=7,
                color="#666666",
                alpha=0.35,
                linewidths=0,
                zorder=1,
            )
            bp = ax.boxplot(
                [vals],
                positions=[i],
                vert=False,
                widths=0.55,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "black", "lw": 1.2},
                whiskerprops={"color": color, "lw": 1.0},
                capprops={"color": color, "lw": 1.0},
                boxprops={"facecolor": color, "alpha": 0.55, "edgecolor": color, "lw": 1.0},
            )

        ax.set_yticks(positions)
        ax.set_yticklabels(plot_labels, fontsize=5.5)
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(-0.7, max(len(plot_species) - 0.3, 0.3))
        ax.set_title(f"{title}  (seed {seed})", loc="left", fontsize=9)
        ax.set_xlabel("Illicit score")
        if ax in (axes[0, 0], axes[1, 0]):
            ax.set_ylabel("Species (by semantic label)", fontsize=8)
        n_pts = sum(len(v) for v in plot_vals)
        ax.text(
            0.98,
            0.02,
            f"K={len(plot_species)}, n={n_pts}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            color="#555",
        )
        panel_meta.append(
            {
                "distance": d,
                "clustering": c,
                "seed": seed,
                "n_species": len(plot_species),
                "labels": plot_labels[::-1],  # rank order
                "ranks": [sp["rank"] for sp in species],
            }
        )
        print(f"{title}: seed={seed} K={len(plot_species)} n={n_pts}", flush=True)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "fig_rq3_topics.pdf")
    fig.savefig(FIG / "fig_rq3_topics.png", dpi=160)
    plt.close(fig)
    print(f"wrote {FIG / 'fig_rq3_topics.pdf'}")

    summary_rows = []
    for d, c, title in CELLS:
        counts = coverage_species[(d, c)]
        k = len([t for t, v in counts.items() if v > 0 and t != "mixed"])
        k_all = len([t for t, v in counts.items() if v > 0])
        n1 = effective_n1(counts)
        n1_nomix = effective_n1(Counter({t: v for t, v in counts.items() if t != "mixed"}))
        summary_rows.append(
            {
                "distance": d,
                "clustering": c,
                "n_labeled_species": int(sum(counts.values())),
                "unique_topics_incl_mixed": int(k_all),
                "unique_topics_excl_mixed": int(k),
                "N1_incl_mixed": n1,
                "N1_excl_mixed": n1_nomix,
                "topic_counts": dict(counts),
            }
        )
        print(
            f"{title}: inventory species={sum(counts.values())} K={k} N1={n1_nomix:.2f}",
            flush=True,
        )

    OUT_REPORTS.mkdir(parents=True, exist_ok=True)
    with (OUT_REPORTS / "rq3_topic_coverage.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "distance",
                "clustering",
                "n_labeled_species",
                "unique_topics_incl_mixed",
                "unique_topics_excl_mixed",
                "N1_incl_mixed",
                "N1_excl_mixed",
            ],
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r[k] for k in w.fieldnames})
    (OUT_REPORTS / "rq3_topic_coverage_meta.json").write_text(
        json.dumps({"rows": summary_rows, "panels": panel_meta}, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUT_REPORTS / 'rq3_topic_coverage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
