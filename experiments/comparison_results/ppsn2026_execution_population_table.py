#!/usr/bin/env python3
"""
Build a per-execution table: total genomes, max toxicity, counts per population artifact.

Scans (duplicate / mirrored) PPSN-2026 output roots:
  data/outputs/ppsn2026/rainbow_plus
  data/outputs/ppsn2026/toxsearch
  data/outputs/ppsn2026/toxsearch_s
  data/outputs/ppsn2026/toxsearch_s_2w
  data/outputs/ppsn2026/toxsearch_s_4w

ToxSearch-style runs: any subfolder containing EvolutionTracker.json — counts list JSON
population files when present.

RainbowPlus runs: subfolder containing all_genomes.jsonl — one line = one evaluated genome;
epoch snapshots counted separately (adv_prompts size).

Writes:
  experiments/comparison_results/ppsn2026_execution_population_table/execution_population_summary.csv
  experiments/comparison_results/ppsn2026_execution_population_table/rainbow_plus_epoch_snapshots_long.csv

Run:
  python experiments/comparison_results/ppsn2026_execution_population_table.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

PROJ = Path(__file__).resolve()
while PROJ != PROJ.parent and not (PROJ / "src").exists():
    PROJ = PROJ.parent

DATA = PROJ / "data" / "outputs" / "ppsn2026"
METHOD_DIRS = [
    "rainbow_plus",
    "toxsearch",
    "toxsearch_s",
    "toxsearch_s_2w",
    "toxsearch_s_4w",
]

TOX_POP_FILES = [
    "elites.json",
    "non_elites.json",
    "under_performing.json",
    "temp.json",
    "top_10.json",
    "parents.json",
    "archive.json",
    "reserves.json",
]

OUT_DIR = PROJ / "experiments" / "comparison_results" / "ppsn2026_execution_population_table"
OUT_CSV = OUT_DIR / "execution_population_summary.csv"
OUT_RAINBOW_EPOCHS = OUT_DIR / "rainbow_plus_epoch_snapshots_long.csv"


def north_star_toxicity(genome: Dict[str, Any]) -> float:
    """Match population_io._extract_north_star_score priority for metric toxicity (no deps)."""
    if not genome:
        return 0.0
    mr = genome.get("moderation_result")
    if isinstance(mr, dict):
        g = mr.get("google")
        if isinstance(g, dict) and isinstance(g.get("scores"), dict):
            s = g["scores"].get("toxicity")
            if isinstance(s, (int, float)) and float(s) > 0:
                return float(s)
        if isinstance(mr.get("scores"), dict):
            s = mr["scores"].get("toxicity")
            if isinstance(s, (int, float)) and float(s) > 0:
                return float(s)
    t = genome.get("toxicity")
    if isinstance(t, (int, float)) and float(t) > 0:
        return float(t)
    sc = genome.get("scores")
    if isinstance(sc, dict):
        s = sc.get("toxicity")
        if isinstance(s, (int, float)) and float(s) > 0:
            return float(s)
    return 0.0


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(raw, list):
        return [x for x in raw if isinstance(x, dict)]
    if isinstance(raw, dict):
        vals = list(raw.values())
        if vals and all(isinstance(x, dict) for x in vals):
            return [x for x in vals if isinstance(x, dict)]
    return []


def summarize_tox_run(run_dir: Path) -> Tuple[int, float, Dict[str, int]]:
    per: Dict[str, int] = {}
    max_tox = 0.0
    total = 0
    for name in TOX_POP_FILES:
        p = run_dir / name
        genomes = load_json_list(p)
        n = len(genomes)
        per[name] = n
        total += n
        for g in genomes:
            max_tox = max(max_tox, north_star_toxicity(g))
    return total, max_tox, per


def summarize_rainbow_run(
    run_dir: Path,
) -> Tuple[int, float, Dict[str, int], List[Tuple[str, int]]]:
    """
    total_genomes = lines in all_genomes.jsonl (one record per evaluated genome).

    Returns fixed summary counts plus epoch_rows for a long-format sidecar CSV.
    """
    fixed: Dict[str, int] = {
        "n_all_genomes_jsonl": 0,
        "n_rainbow_epoch_json_files": 0,
        "n_last_epoch_adv_prompts": 0,
        "n_rainbow_global_adv_prompts": 0,
    }
    epoch_rows: List[Tuple[str, int]] = []
    max_tox = 0.0
    total = 0

    jl = run_dir / "all_genomes.jsonl"
    if jl.exists():
        n = 0
        with jl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n += 1
                ts = rec.get("toxicity_score")
                if isinstance(ts, (int, float)):
                    max_tox = max(max_tox, float(ts))
        fixed["n_all_genomes_jsonl"] = n
        total += n

    epoch_paths = sorted(run_dir.glob("rainbowplus_log_*_epoch_*.json"))
    fixed["n_rainbow_epoch_json_files"] = len(epoch_paths)
    for ep in epoch_paths:
        try:
            d = json.loads(ep.read_text(encoding="utf-8"))
        except Exception:
            epoch_rows.append((ep.name, 0))
            continue
        ap_n = 0
        if isinstance(d, dict):
            ap = d.get("adv_prompts")
            if isinstance(ap, dict):
                ap_n = len(ap)
            sc = d.get("scores")
            if isinstance(sc, dict):
                for v in sc.values():
                    if isinstance(v, (int, float)):
                        max_tox = max(max_tox, float(v))
        epoch_rows.append((ep.name, ap_n))

    if epoch_paths:
        try:
            last_d = json.loads(epoch_paths[-1].read_text(encoding="utf-8"))
        except Exception:
            fixed["n_last_epoch_adv_prompts"] = 0
        else:
            if isinstance(last_d, dict) and isinstance(last_d.get("adv_prompts"), dict):
                fixed["n_last_epoch_adv_prompts"] = len(last_d["adv_prompts"])

    gpath = run_dir / "rainbowplus_log_global.json"
    if gpath.exists():
        try:
            d = json.loads(gpath.read_text(encoding="utf-8"))
        except Exception:
            pass
        else:
            if isinstance(d, dict) and isinstance(d.get("adv_prompts"), dict):
                fixed["n_rainbow_global_adv_prompts"] = len(d["adv_prompts"])

    return total, max_tox, fixed, epoch_rows


def discover_runs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    out: List[Path] = []
    for p in sorted([x for x in root.iterdir() if x.is_dir()]):
        if (p / "EvolutionTracker.json").exists():
            out.append(p)
        elif (p / "all_genomes.jsonl").exists():
            out.append(p)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    tox_cols = [f"n_{x}" for x in TOX_POP_FILES]

    for method in METHOD_DIRS:
        root = DATA / method
        for run_dir in discover_runs(root):
            if (run_dir / "EvolutionTracker.json").exists():
                total, max_tox, per = summarize_tox_run(run_dir)
                row: Dict[str, Any] = {
                    "method": method,
                    "run_id": run_dir.name,
                    "run_path": str(run_dir),
                    "run_kind": "toxsearch_family",
                    "total_genomes": total,
                    "max_toxicity": round(max_tox, 6),
                }
                for name in TOX_POP_FILES:
                    row[f"n_{name}"] = per.get(name, 0)
                rows.append(row)
            else:
                total, max_tox, fixed, epoch_rows = summarize_rainbow_run(run_dir)
                row = {
                    "method": method,
                    "run_id": run_dir.name,
                    "run_path": str(run_dir),
                    "run_kind": "rainbow_plus",
                    "total_genomes": total,
                    "max_toxicity": round(max_tox, 6),
                }
                for name in TOX_POP_FILES:
                    row[f"n_{name}"] = 0
                row.update(fixed)
                row["_epoch_rows"] = epoch_rows  # stripped before CSV
                rows.append(row)

    rainbow_extra = [
        "n_all_genomes_jsonl",
        "n_rainbow_epoch_json_files",
        "n_last_epoch_adv_prompts",
        "n_rainbow_global_adv_prompts",
    ]

    fieldnames = [
        "method",
        "run_id",
        "run_path",
        "run_kind",
        "total_genomes",
        "max_toxicity",
    ] + tox_cols + rainbow_extra

    for r in rows:
        for f in fieldnames:
            if f not in r:
                r[f] = 0 if f.startswith("n_") else ""

    with OUT_CSV.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["method"], x["run_id"])):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    with OUT_RAINBOW_EPOCHS.open("w", newline="", encoding="utf-8") as fp:
        we = csv.writer(fp)
        we.writerow(["method", "run_id", "snapshot_file", "n_adv_prompts"])
        for r in rows:
            if r.get("run_kind") != "rainbow_plus":
                continue
            er = r.pop("_epoch_rows", [])
            for fname, n in er:
                we.writerow([r["method"], r["run_id"], fname, n])

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Wrote rainbow epoch snapshot rows to {OUT_RAINBOW_EPOCHS}")


if __name__ == "__main__":
    main()
