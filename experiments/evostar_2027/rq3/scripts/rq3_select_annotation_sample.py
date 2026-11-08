#!/usr/bin/env python3
"""Stratified sample of high-fitness genomes for blinded RQ3 annotation.

Writes ID-level CSV only (run path, genome id, fitness, config). Does not write
prompt/response text into the report CSV. Optional --export-packets writes a
private JSONL with texts for annotators.

  PYTHONPATH=src:. venv/bin/python \\
    experiments/evostar_2027/rq3/scripts/rq3_select_annotation_sample.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "experiments" / "evostar_2027" / "rq2" / "scripts"))
sys.path.insert(0, str(ROOT / "notes" / "rough_draft"))

from generate_rq2_coverage import (  # noqa: E402
    LIVE_DIR,
    _assign,
    _complete,
    _prototype_matrix,
    _run_dir,
)
from rq2_metrics import illicit_of, load_genomes  # noqa: E402

REPORT = ROOT / "experiments" / "evostar_2027" / "rq3" / "reports"
NOISE = {None, 0, -1, "0", "-1"}


def _sid(rec: Dict[str, Any]) -> Any:
    return rec.get("species_id", rec.get("density_label", rec.get("label")))


def _response(rec: Dict[str, Any]) -> str:
    for k in ("response", "model_response", "output", "completion"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.80)
    ap.add_argument("--per-config", type=int, default=30)
    ap.add_argument("--dupes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--export-packets", type=Path, default=None, help="Private JSONL with texts")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    risk_m, style_m, risk_names, style_names = _prototype_matrix(encoder)
    cache: Dict[str, Tuple[str, str]] = {}
    rng = np.random.default_rng(args.seed)

    pool: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for distance, clustering in LIVE_DIR:
        for seed in range(10):
            run_dir = _run_dir(distance, clustering, seed)
            if not _complete(run_dir):
                continue
            for rec in load_genomes(run_dir, clustering):
                f = illicit_of(rec)
                if f is None or f < args.tau:
                    continue
                prompt = str(rec.get("prompt") or "").strip()
                if not prompt:
                    continue
                z = _assign(rec, risk_m, style_m, risk_names, style_names, encoder, cache)
                risk, style = z if z else ("unknown", "unknown")
                fit_bin = "hi" if f >= 0.90 else "mid"
                item = {
                    "distance": distance,
                    "clustering": clustering,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "genome_id": rec.get("id"),
                    "species_id": _sid(rec),
                    "fitness": float(f),
                    "fitness_bin": fit_bin,
                    "proto_risk": risk,
                    "proto_style": style,
                    "is_noise": _sid(rec) in NOISE,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16],
                    "prompt": prompt,
                    "response": _response(rec),
                }
                pool[(distance, clustering)].append(item)

    selected: List[Dict[str, Any]] = []
    for key, items in pool.items():
        if not items:
            continue
        # diversify by risk then fitness bin
        by_risk: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for it in items:
            by_risk[it["proto_risk"]].append(it)
        risks = list(by_risk.keys())
        rng.shuffle(risks)
        picks: List[Dict[str, Any]] = []
        # round-robin across risks
        while len(picks) < min(args.per_config, len(items)):
            progressed = False
            for rk in risks:
                cand = [x for x in by_risk[rk] if x not in picks]
                if not cand:
                    continue
                # prefer mix of fitness bins
                mid = [x for x in cand if x["fitness_bin"] == "mid"]
                hi = [x for x in cand if x["fitness_bin"] == "hi"]
                bag = mid if mid and (len([p for p in picks if p["fitness_bin"] == "mid"]) <= len([p for p in picks if p["fitness_bin"] == "hi"])) else (hi or mid or cand)
                choice = bag[int(rng.integers(0, len(bag)))]
                picks.append(choice)
                progressed = True
                if len(picks) >= args.per_config:
                    break
            if not progressed:
                break
        selected.extend(picks)

    # duplicates for agreement
    if selected and args.dupes > 0:
        dup_src = list(selected)
        rng.shuffle(dup_src)
        for it in dup_src[: args.dupes]:
            d = dict(it)
            d["is_duplicate"] = True
            selected.append(d)
    for it in selected:
        it.setdefault("is_duplicate", False)

    rng.shuffle(selected)
    for i, it in enumerate(selected):
        it["annot_id"] = f"A{i:04d}"

    REPORT.mkdir(parents=True, exist_ok=True)
    id_fields = [
        "annot_id",
        "is_duplicate",
        "distance",
        "clustering",
        "seed",
        "run_dir",
        "genome_id",
        "species_id",
        "fitness",
        "fitness_bin",
        "proto_risk",
        "proto_style",
        "is_noise",
        "prompt_sha256",
    ]
    out_csv = REPORT / "annotation_sample_ids.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=id_fields, extrasaction="ignore")
        w.writeheader()
        for it in selected:
            w.writerow(it)

    meta = {
        "n_selected": len(selected),
        "n_unique": sum(1 for x in selected if not x["is_duplicate"]),
        "n_dupes": sum(1 for x in selected if x["is_duplicate"]),
        "tau": args.tau,
        "per_config": args.per_config,
        "seed": args.seed,
        "pool_sizes": {f"{d}__{c}": len(v) for (d, c), v in pool.items()},
    }
    (REPORT / "annotation_sample_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.export_packets is not None:
        args.export_packets.parent.mkdir(parents=True, exist_ok=True)
        with args.export_packets.open("w", encoding="utf-8") as f:
            for it in selected:
                packet = {
                    "annot_id": it["annot_id"],
                    "is_duplicate": it["is_duplicate"],
                    "prompt": it["prompt"],
                    "response": it["response"],
                    # gold labels hidden from annotators in UI; kept for scoring only
                    "_gold_proto_risk": it["proto_risk"],
                    "_gold_proto_style": it["proto_style"],
                    "_gold_fitness": it["fitness"],
                }
                f.write(json.dumps(packet, ensure_ascii=False) + "\n")
        print(f"wrote private packets {args.export_packets}")

    print(f"wrote {out_csv} ({meta['n_unique']} unique + {meta['n_dupes']} dupes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
