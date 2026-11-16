#!/usr/bin/env python3
"""RQ2 text diagnostics. Writes reports only; does not touch the manuscript.

1. Response duration (seconds) for genomes with refusal-penalized fitness > 0.6.
2. Same-seed prompt length versus species membership (four speciated cells).

Prompt perplexity is not computed here: the generation GGUF
(Meta-Llama-3.1-8B-Instruct.Q8_0.gguf) is not on disk, and no other model
is substituted.
"""
from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from rq2_metrics import NOISE_SIDS, illicit_of, load_json  # noqa: E402

LIVE = ROOT / "data" / "outputs" / "evostar2027"
REPORTS = ROOT / "experiments" / "evostar_2027" / "rq2" / "reports"
TOPIC_XLSX = REPORTS / "topic_vocabulary_by_execution.xlsx"
GGUF = ROOT / "models/llama3.1-8b-instruct-gguf/Meta-Llama-3.1-8B-Instruct.Q8_0.gguf"

CELLS: List[Tuple[str, str, str]] = [
    ("lf_embedding", "leader_follower", "MiniLM LF"),
    ("inc_embedding", "dbscan", "MiniLM Inc"),
    ("lf_nli", "leader_follower", "NLI LF"),
    ("inc_nli", "dbscan", "NLI Inc"),
    ("no_speciation", "none", "Global EA"),
]
SPECIATED = {"lf_embedding", "inc_embedding", "lf_nli", "inc_nli"}
MIN_TOPIC_N = 8
BIN_WIDTH = 5.0


def pool_names(clustering: str) -> Tuple[str, ...]:
    if clustering == "dbscan":
        return ("temp.json",)
    return ("elites.json", "reserves.json", "archive.json")


def load_pool(run_dir: Path, clustering: str) -> List[dict]:
    recs: List[dict] = []
    seen = set()
    for name in pool_names(clustering):
        data = load_json(run_dir / name)
        if not isinstance(data, list):
            continue
        for rec in data:
            if not isinstance(rec, dict):
                continue
            rid = rec.get("id")
            key = str(rid)
            if key in seen:
                continue
            seen.add(key)
            recs.append(rec)
    return recs


def parse_duration(value: Any) -> Tuple[Optional[float], bool]:
    """Return (seconds, missing). Missing is not coerced to zero."""
    if value is None or value == "":
        return None, True
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None, True
    if x != x:
        return None, True
    return x, False


def species_int(value: Any) -> Optional[int]:
    if value in NOISE_SIDS:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def summarize(durations: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(durations), dtype=float)
    if arr.size == 0:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan, "iqr": np.nan, "min": np.nan, "max": np.nan}
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(med),
        "q25": float(q25),
        "q75": float(q75),
        "iqr": float(q75 - q25),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_vals = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a = np.asarray(x, dtype=float)
    b = np.asarray(y, dtype=float)
    if a.size < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_topic_index() -> Dict[Tuple[str, int, str], Tuple[str, str]]:
    df = pd.read_excel(
        TOPIC_XLSX,
        sheet_name="Prompt_topics",
        usecols=["cell_dir", "seed", "genome_id", "ancestry_topic_ids", "topic_source"],
    )
    out: Dict[Tuple[str, int, str], Tuple[str, str]] = {}
    for rec in df.itertuples(index=False):
        anc = "" if pd.isna(rec.ancestry_topic_ids) else str(rec.ancestry_topic_ids).strip()
        src = "" if pd.isna(rec.topic_source) else str(rec.topic_source).strip()
        out[(str(rec.cell_dir), int(rec.seed), str(rec.genome_id))] = (anc, src)
    return out


def single_seed(ancestry: str, source: str) -> Optional[str]:
    parts = [p for p in ancestry.split() if p]
    if len(parts) != 1:
        return None
    if source == "conflict":
        return None
    if source in {"ancestry", "ancestry and text agree"} or source.startswith("archive:"):
        return parts[0]
    return None


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    topics = load_topic_index()
    print(f"[topics] {len(topics)} labeled genomes", flush=True)

    rows: List[dict] = []
    missing_topic = 0
    for cell, clustering, label in CELLS:
        for seed in range(10):
            run_dir = LIVE / cell / f"rep{seed}_s{seed}"
            genomes = load_pool(run_dir, clustering)
            sizes: Counter = Counter()
            prepared = []
            for rec in genomes:
                sid = species_int(rec.get("species_id"))
                if sid is not None and sid >= 1:
                    sizes[sid] += 1
                fit = illicit_of(rec)
                dur, missing = parse_duration(rec.get("response_duration"))
                prompt = rec.get("prompt") or ""
                gid = str(rec.get("id"))
                anc, src = topics.get((cell, seed, gid), ("", ""))
                if (cell, seed, gid) not in topics:
                    missing_topic += 1
                prepared.append(
                    {
                        "cell": cell,
                        "label": label,
                        "seed": seed,
                        "genome_id": gid,
                        "fitness": fit,
                        "response_duration": dur,
                        "duration_missing": missing,
                        "prompt_chars": len(prompt),
                        "species_id": sid if sid is not None else rec.get("species_id"),
                        "ancestry_topic_ids": anc,
                        "topic_source": src,
                        "topic_id": single_seed(anc, src),
                    }
                )
            mature_size = {sid: n for sid, n in sizes.items() if n >= 2}
            for row in prepared:
                sid = row["species_id"]
                try:
                    sid_i = int(sid)
                except (TypeError, ValueError):
                    sid_i = None
                row["species_size"] = int(sizes.get(sid_i, 0)) if sid_i is not None else 0
                row["mature"] = bool(sid_i is not None and sid_i >= 1 and sid_i in mature_size)
                row["unassigned"] = not row["mature"]
                rows.append(row)
            print(f"[pool] {label} seed {seed}: {len(prepared)} genomes", flush=True)

    print(f"[topics] genomes with no vocabulary row: {missing_topic}", flush=True)

    high = [r for r in rows if r["fitness"] is not None and r["fitness"] > 0.6]
    high_ok = [r for r in high if not r["duration_missing"]]
    high_miss = [r for r in high if r["duration_missing"]]

    per_path = REPORTS / "rq2_response_duration_gt06.csv"
    with per_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "cell", "label", "seed", "genome_id", "fitness", "response_duration",
            "duration_missing", "prompt_chars", "species_id", "mature", "unassigned",
            "topic_id", "topic_source",
        ]
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in high:
            w.writerow(r)

    summaries = []
    grouped: Dict[str, List[float]] = defaultdict(list)
    for r in high_ok:
        grouped[r["label"]].append(r["response_duration"])
    order = [label for _, _, label in CELLS]
    blocks = [("Overall", [r["response_duration"] for r in high_ok])] + [(lab, grouped[lab]) for lab in order]
    for name, vals in blocks:
        stats = summarize(vals)
        stats["configuration"] = name
        stats["n_missing_duration"] = (
            len(high_miss) if name == "Overall" else sum(1 for r in high_miss if r["label"] == name)
        )
        stats["n_fitness_gt_0.6"] = stats["n"] + stats["n_missing_duration"]
        summaries.append(stats)

    sum_path = REPORTS / "rq2_response_duration_summary.csv"
    with sum_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["configuration", "n_fitness_gt_0.6", "n", "n_missing_duration", "mean", "median", "q25", "q75", "iqr", "min", "max"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stats in summaries:
            w.writerow({k: stats[k] for k in fields})

    observed = np.asarray([r["response_duration"] for r in high_ok], dtype=float)
    hi = float(np.ceil(observed.max() / BIN_WIDTH) * BIN_WIDTH) if observed.size else BIN_WIDTH
    edges = np.arange(0.0, hi + BIN_WIDTH, BIN_WIDTH)
    hist_rows = []
    hist_path = REPORTS / "rq2_response_duration_hist.csv"
    with hist_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["configuration", "bin_lo", "bin_hi", "count"])
        w.writeheader()
        for name, vals in blocks:
            counts, _ = np.histogram(vals, bins=edges)
            for lo, hi_edge, count in zip(edges[:-1], edges[1:], counts):
                rec = {"configuration": name, "bin_lo": lo, "bin_hi": hi_edge, "count": int(count)}
                w.writerow(rec)
                hist_rows.append(rec)

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True)
    palette = {
        "Overall": "#333333",
        "MiniLM LF": "#0F4D92",
        "MiniLM Inc": "#2A9D8F",
        "NLI LF": "#B64342",
        "NLI Inc": "#E9C46A",
        "Global EA": "#4D4D4D",
    }
    for ax, (name, vals) in zip(axes.ravel(), blocks):
        ax.hist(vals, bins=edges, color=palette[name], edgecolor="white", linewidth=0.4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Response duration (s)")
        ax.set_ylabel("Prompts")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes.ravel()[-1].axis("off")
    fig.suptitle("Refusal-penalized fitness > 0.6", fontsize=11)
    fig.tight_layout()
    fig.savefig(REPORTS / "rq2_response_duration_hist.png", dpi=160)
    fig.savefig(REPORTS / "rq2_response_duration_hist.pdf")
    plt.close(fig)

    # Length within single-seed topics, four speciated configurations.
    by_topic: Dict[Tuple[str, int, str], List[dict]] = defaultdict(list)
    for r in rows:
        if r["cell"] not in SPECIATED or not r["topic_id"]:
            continue
        by_topic[(r["cell"], r["seed"], r["topic_id"])].append(r)

    topic_rows = []
    for (cell, seed, topic_id), members in by_topic.items():
        if len(members) < MIN_TOPIC_N:
            continue
        mature_counts: Counter = Counter()
        for m in members:
            if m["mature"]:
                mature_counts[int(m["species_id"])] += 1
        if not mature_counts:
            continue
        largest = max(mature_counts, key=lambda sid: (mature_counts[sid], -sid))
        groups = {"largest": [], "other_mature": [], "unassigned": []}
        for m in members:
            if m["unassigned"]:
                groups["unassigned"].append(m["prompt_chars"])
            elif int(m["species_id"]) == largest:
                groups["largest"].append(m["prompt_chars"])
            else:
                groups["other_mature"].append(m["prompt_chars"])
        lengths = np.asarray([m["prompt_chars"] for m in members], dtype=float)
        in_largest = np.asarray(
            [0.0 if m["unassigned"] or int(m["species_id"]) != largest else 1.0 for m in members]
        )
        q75 = float(np.percentile(lengths, 75))
        long_mask = lengths >= q75
        base = float(in_largest.mean())
        long_rate = float(in_largest[long_mask].mean()) if long_mask.any() else float("nan")
        topic_rows.append(
            {
                "cell": cell,
                "label": members[0]["label"],
                "seed": seed,
                "topic_id": topic_id,
                "n": len(members),
                "n_largest": len(groups["largest"]),
                "n_other_mature": len(groups["other_mature"]),
                "n_unassigned": len(groups["unassigned"]),
                "largest_species_id": largest,
                "median_largest": float(np.median(groups["largest"])) if groups["largest"] else "",
                "median_other_mature": float(np.median(groups["other_mature"])) if groups["other_mature"] else "",
                "median_unassigned": float(np.median(groups["unassigned"])) if groups["unassigned"] else "",
                "base_share_largest": base,
                "long_quartile_share_largest": long_rate,
                "long_excess": long_rate - base,
                "spearman_length_in_largest": spearman(lengths, in_largest),
            }
        )

    topic_path = REPORTS / "rq2_length_species_topics.csv"
    if topic_rows:
        with topic_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(topic_rows[0].keys()))
            w.writeheader()
            w.writerows(topic_rows)

    def _nums(records: Sequence[dict], key: str) -> np.ndarray:
        vals = []
        for rec in records:
            v = rec[key]
            if v == "" or v is None:
                continue
            vals.append(float(v))
        return np.asarray(vals, dtype=float)

    length_summary = []
    for cell, _, label in CELLS:
        if cell not in SPECIATED:
            continue
        subset = [r for r in topic_rows if r["cell"] == cell]
        rho = _nums(subset, "spearman_length_in_largest")
        excess = _nums(subset, "long_excess")
        base = _nums(subset, "base_share_largest")
        length_summary.append(
            {
                "configuration": label,
                "n_topics": len(subset),
                "median_of_median_largest": float(np.median(_nums(subset, "median_largest"))) if _nums(subset, "median_largest").size else "",
                "median_of_median_other_mature": float(np.median(_nums(subset, "median_other_mature"))) if _nums(subset, "median_other_mature").size else "",
                "median_of_median_unassigned": float(np.median(_nums(subset, "median_unassigned"))) if _nums(subset, "median_unassigned").size else "",
                "n_topics_with_other_mature": int(_nums(subset, "median_other_mature").size),
                "n_topics_with_unassigned": int(_nums(subset, "median_unassigned").size),
                "median_base_share_largest": float(np.median(base)) if base.size else "",
                "median_long_quartile_share": float(np.median(_nums(subset, "long_quartile_share_largest"))) if subset else "",
                "median_long_excess": float(np.median(excess)) if excess.size else "",
                "median_spearman": float(np.nanmedian(rho)) if rho.size else "",
                "n_spearman": int(np.isfinite(rho).sum()) if rho.size else 0,
            }
        )
    len_sum_path = REPORTS / "rq2_length_species_summary.csv"
    with len_sum_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(length_summary[0].keys()))
        w.writeheader()
        w.writerows(length_summary)

    print("[duration] fitness>0.6", len(high), "with duration", len(high_ok), "missing", len(high_miss))
    for stats in summaries:
        print(
            f"  {stats['configuration']}: n={stats['n']} missing={stats['n_missing_duration']} "
            f"mean={stats['mean']:.2f} median={stats['median']:.2f} IQR={stats['iqr']:.2f}"
        )
    print("[length] topics", len(topic_rows))
    for rec in length_summary:
        print(
            f"  {rec['configuration']}: topics={rec['n_topics']} "
            f"medL={rec['median_of_median_largest']} medO={rec['median_of_median_other_mature']} "
            f"medU={rec['median_of_median_unassigned']} excess={rec['median_long_excess']} "
            f"rho={rec['median_spearman']}"
        )
    print("[gguf]", "present" if GGUF.is_file() else f"ABSENT {GGUF}")


if __name__ == "__main__":
    main()
