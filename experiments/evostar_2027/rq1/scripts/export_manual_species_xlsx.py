#!/usr/bin/env python3
"""Manual species-coherence workbook for the locked RQ1 partitions.

Every prompt in every mature species (at least two members) at t=0.10 gets one
row, for leader--follower and for IncDBSCAN (MinPts=2) under d_E (MiniLM) and
d_negE (NLI). Unjoined LF leaders are not species and are omitted. Topic
judgments come only from reading the prompt text; no encoder, NLI model, or
archive_topic_labels.csv topic column is used.

  --chunks   write reading chunks (whole species, except the largest NLI
             species, which is split) to manual_species/chunks/
  default    build manual_species_coherence.xlsx, filling judgments from
             manual_species/judgments/*.csv when present

Judgment CSV columns: prompt_index,species_id,species_topic_label,
matches_species,notes (matches_species is yes or no).

The largest NLI species is too large to label in one reading, so each of its
prompts was read and given a topic from giant_topic_vocabulary.txt (files in
manual_species/giant_topics/). Its species label is the most frequent read
topic, and a prompt matches when its read topic equals that label.
"""
from __future__ import annotations

import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[4]
ARCHIVE = ROOT / "experiments/evostar_2027/data/batches/B3139/p0.pkl"
SWEEP = ROOT / "experiments/evostar_2027/rq1/sweeps/B3139"
SPECIES = SWEEP / "species"
WORK = SWEEP / "manual_species"
CHUNKS = WORK / "chunks"
JUDGMENTS = WORK / "judgments"
GIANT_TOPICS = WORK / "giant_topics"
GIANT_VOCAB = WORK / "giant_topic_vocabulary.txt"
OUT = SWEEP / "manual_species_coherence.xlsx"

SETTINGS = [
    ("MiniLM_IncDBSCAN", "incdbscan__d_E__t0.10_ms2"),
    ("MiniLM_LF", "leader_follower__d_E__t0.10"),
    ("NLI_IncDBSCAN", "incdbscan__d_negE__t0.10_ms2"),
    ("NLI_LF", "leader_follower__d_negE__t0.10"),
]
COLUMNS = [
    "species_id",
    "species_size",
    "prompt_index",
    "fitness",
    "prompt",
    "species_topic_label",
    "matches_species",
    "notes",
]
CHUNK_PROMPTS = 350


def species_rows(archive, stem: str) -> list[dict]:
    labels = np.load(SPECIES / f"{stem}_labels.npy")
    counts = Counter(int(s) for s in labels.tolist())
    sizes = {s: n for s, n in counts.items() if s >= 0 and n >= 2}
    order = sorted(sizes, key=lambda s: (-sizes[s], s))
    rows = []
    for sid in order:
        members = np.flatnonzero(labels == sid)
        members = sorted(members, key=lambda i: -float(archive.at[i, "fitness"]))
        for i in members:
            rows.append(
                {
                    "species_id": sid,
                    "species_size": sizes[sid],
                    "prompt_index": int(i),
                    "fitness": round(float(archive.at[i, "fitness"]), 4),
                    "prompt": str(archive.at[i, "prompt"]),
                }
            )
    return rows


def write_chunks(sheet: str, rows: list[dict]) -> None:
    CHUNKS.mkdir(parents=True, exist_ok=True)
    by_species: dict[int, list[dict]] = {}
    for r in rows:
        by_species.setdefault(r["species_id"], []).append(r)
    chunks: list[list[dict]] = []
    current: list[dict] = []
    for members in by_species.values():
        if len(members) > CHUNK_PROMPTS:
            if current:
                chunks.append(current)
                current = []
            for k in range(0, len(members), CHUNK_PROMPTS):
                chunks.append(members[k : k + CHUNK_PROMPTS])
            continue
        if current and len(current) + len(members) > CHUNK_PROMPTS:
            chunks.append(current)
            current = []
        current.extend(members)
    if current:
        chunks.append(current)
    for n, chunk in enumerate(chunks):
        path = CHUNKS / f"{sheet}_chunk{n:02d}.tsv"
        with path.open("w", newline="") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["prompt_index", "species_id", "species_size", "prompt"])
            for r in chunk:
                prompt = " ".join(r["prompt"].split())
                w.writerow([r["prompt_index"], r["species_id"], r["species_size"], prompt])
        print(f"{path.name}: {len(chunk)} prompts, species {sorted({r['species_id'] for r in chunk})[:6]}...")


def giant_judgments(sheet: str) -> dict[int, dict]:
    paths = sorted(GIANT_TOPICS.glob(f"{sheet}_*.csv"))
    if not paths:
        return {}
    vocab = {}
    for line in GIANT_VOCAB.read_text().splitlines():
        tid, name = line.split("\t", 1)
        vocab[tid] = name
    reads = []
    for path in paths:
        with path.open(newline="") as f:
            reads.extend(csv.DictReader(f))

    def topic_of(r: dict) -> str:
        tid = r["topic_id"].strip()
        if tid == "NEW":
            return " ".join(r["new_topic_name"].split()).rstrip(".").capitalize()
        return vocab[tid]

    topics = Counter(topic_of(r) for r in reads)
    label = topics.most_common(1)[0][0]
    out = {}
    for r in reads:
        topic = topic_of(r)
        notes = [] if topic == label else [f"read topic: {topic}"]
        if r.get("notes", "").strip():
            notes.append(r["notes"].strip())
        out[int(r["prompt_index"])] = {
            "species_id": r["species_id"],
            "species_topic_label": label,
            "matches_species": "yes" if topic == label else "no",
            "notes": "; ".join(notes),
        }
    return out


def load_judgments(sheet: str) -> dict[int, dict]:
    out = giant_judgments(sheet)
    for path in sorted(JUDGMENTS.glob(f"{sheet}_*.csv")):
        with path.open(newline="") as f:
            for r in csv.DictReader(f):
                out[int(r["prompt_index"])] = r
    return out


def build_workbook(archive) -> None:
    wb = Workbook()
    wb.remove(wb.active)
    summary = []
    for sheet, stem in SETTINGS:
        rows = species_rows(archive, stem)
        judged = load_judgments(sheet)
        ws = wb.create_sheet(sheet)
        ws.append(COLUMNS)
        n_yes = n_no = 0
        mixed: set[int] = set()
        for r in rows:
            j = judged.get(r["prompt_index"])
            if j is not None and int(j["species_id"]) != r["species_id"]:
                raise ValueError(f"{sheet} prompt {r['prompt_index']}: species mismatch in judgments")
            match = (j or {}).get("matches_species", "").strip().lower()
            if match == "yes":
                n_yes += 1
            elif match == "no":
                n_no += 1
                mixed.add(r["species_id"])
            ws.append(
                [
                    r["species_id"],
                    r["species_size"],
                    r["prompt_index"],
                    r["fitness"],
                    r["prompt"],
                    (j or {}).get("species_topic_label", ""),
                    match,
                    (j or {}).get("notes", ""),
                ]
            )
        widths = [11, 13, 13, 9, 90, 40, 16, 40]
        for k, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(k)].width = w
        for cell in ws[1]:
            cell.font = Font(bold=True)
        for row in ws.iter_rows(min_row=2, min_col=5, max_col=5):
            row[0].alignment = Alignment(wrap_text=True, vertical="top")
        ws.freeze_panes = "A2"
        ws.auto_filter.ref = ws.dimensions
        n_species = len({r["species_id"] for r in rows})
        judged_n = n_yes + n_no
        summary.append(
            [
                sheet,
                n_species,
                len(rows),
                judged_n,
                n_no,
                round(n_no / judged_n, 4) if judged_n else None,
                len(mixed),
            ]
        )
        print(f"{sheet}: {n_species} species, {len(rows)} prompts, judged {judged_n}, mismatches {n_no}")

    ws = wb.create_sheet("Summary", 0)
    ws.append(
        [
            "setting",
            "mature_species",
            "prompts",
            "prompts_judged",
            "mismatches_no",
            "mismatch_rate",
            "species_with_any_mismatch",
        ]
    )
    for row in summary:
        ws.append(row)
    for cell in ws[1]:
        cell.font = Font(bold=True)
    for k in range(1, 8):
        ws.column_dimensions[get_column_letter(k)].width = 22
    ws.append([])
    ws.append(["Partitions: t=0.10. IncDBSCAN uses MinPts=2. LF sheets are leader--follower mature species only (size >= 2)."])
    ws.append(["d_E = MiniLM, d_negE = DeBERTa NLI non-entailment."])
    ws.append(["Judgments from reading prompt text only; matches_species = prompt request topic matches species_topic_label."])
    wb.save(OUT)
    print(f"wrote {OUT}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", action="store_true")
    args = ap.parse_args()
    with ARCHIVE.open("rb") as f:
        archive = pickle.load(f).reset_index(drop=True)
    if args.chunks:
        for sheet, stem in SETTINGS:
            write_chunks(sheet, species_rows(archive, stem))
        return
    build_workbook(archive)


if __name__ == "__main__":
    main()
