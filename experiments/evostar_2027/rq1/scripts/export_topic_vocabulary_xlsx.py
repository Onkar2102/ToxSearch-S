#!/usr/bin/env python3
"""Excel topic vocabulary for the B3139 archive, by distance × method × execution.

Reads archive_topic_labels.csv (seed ancestry + MPNet nearest-seed + hand reads)
and the manual giant_topic_vocabulary.txt codebook. Writes one workbook with:

  README
  Vocabulary              all topic ids / names used on the archive
  Giant_codebook          T01–T99 manual labels for the giant NLI species
  Topics_by_cell          topic counts for each (distance, method, execution)
  Species_topics          mature species × dominant topic, with per-execution mass
  Prompt_topics           one row per archive prompt (compact reference)

Locked online cells are listed first; other offline partitions remain available.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parents[4]
SWEEP = ROOT / "experiments/evostar_2027/rq1/sweeps/B3139"
LABELS = SWEEP / "archive_topic_labels.csv"
NEW_TOPICS = SWEEP / "archive_new_topics.csv"
GIANT_VOCAB = SWEEP / "manual_species" / "giant_topic_vocabulary.txt"
OUT = SWEEP / "manual_species" / "topic_vocabulary_by_execution.xlsx"

# (sheet-friendly name, species column suffix in archive_topic_labels.csv,
#  distance display, method display, min_samples or "")
SETTINGS = [
    ("MiniLM_LF", "leader_follower__d_E__t0.10", "d_E", "leader_follower", ""),
    ("MiniLM_IncDBSCAN", "incdbscan__d_E__t0.10_ms2", "d_E", "incdbscan", "2"),
    ("NLI_LF", "leader_follower__d_negE__t0.10", "d_negE", "leader_follower", ""),
    ("NLI_IncDBSCAN_ms5", "incdbscan__d_negE__t0.10_ms5", "d_negE", "incdbscan", "5"),
    ("NLI_IncDBSCAN_ms2", "incdbscan__d_negE__t0.10_ms2", "d_negE", "incdbscan", "2"),
    ("dEP_LF", "leader_follower__d_EP__t0.10", "d_EP", "leader_follower", ""),
    ("dEP_IncDBSCAN", "incdbscan__d_EP__t0.10_ms2", "d_EP", "incdbscan", "2"),
]

LOCKED = {
    "MiniLM_LF",
    "MiniLM_IncDBSCAN",
    "NLI_LF",
    "NLI_IncDBSCAN_ms5",
}


def _style_header(ws) -> None:
    for cell in ws[1]:
        cell.font = Font(bold=True)
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions


def _widths(ws, widths: list[int]) -> None:
    for k, w in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(k)].width = w


def load_labels() -> list[dict]:
    with LABELS.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def topic_catalog(rows: list[dict]) -> dict[str, str]:
    names: dict[str, str] = {}
    if NEW_TOPICS.is_file():
        with NEW_TOPICS.open(newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                names[r["topic_id"].strip()] = r["topic"].strip()
    for r in rows:
        ids = r["topic_ids"].split()
        parts = [p.strip() for p in r["topic"].split("|")]
        if len(parts) == len(ids):
            for tid, name in zip(ids, parts):
                names.setdefault(tid, name.strip())
        else:
            for tid in ids:
                names.setdefault(tid, "one-off prompt")
    return names


def load_giant_vocab() -> list[tuple[str, str]]:
    out = []
    for line in GIANT_VOCAB.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        tid, name = line.split("\t", 1)
        out.append((tid.strip(), name.strip()))
    return out


def mature_members(rows: list[dict], col: str) -> dict[int, list[dict]]:
    """Mature species = label id with size >= 2 (pool/noise ids with size 1 excluded)."""
    counts: Counter[int] = Counter()
    for r in rows:
        sid = int(r[col])
        if sid >= 0:
            counts[sid] += 1
    keep = {sid for sid, n in counts.items() if n >= 2}
    by: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        sid = int(r[col])
        if sid in keep:
            by[sid].append(r)
    return by


def write_readme(wb: Workbook) -> None:
    ws = wb.create_sheet("README", 0)
    lines = [
        "RQ1 archive topic vocabulary (B3139) — five preliminary source executions",
        "",
        "THIS workbook covers the offline archive only:",
        "  N=3139 unique prompts from run01_speciated … run05_speciated.",
        "  Topic labels: archive_topic_labels.csv (seed ancestry + MPNet nearest",
        "  seed, with hand reads when those signals disagree or a new theme appears).",
        "",
        "For the 50 RQ2 live executions (rep0_s0…rep9_s9 × five cells), see:",
        "  experiments/evostar_2027/rq2/reports/topic_vocabulary_by_execution.xlsx",
        "  and data/outputs/evostar2027/<cell>/topic_vocabulary.xlsx",
        "",
        "Sheets:",
        "  Vocabulary              union of all topic ids used on the archive",
        "  Giant_codebook          manual T01–T99 labels for the giant NLI species",
        "  Topics_by_cell          topic counts per (distance, method, execution)",
        "  Species_topics          mature species with dominant topic + per-run mass",
        "  Prompt_topics           one row per archive prompt",
        "",
        "Locked online RQ2 cells (flagged locked=yes): MiniLM LF, MiniLM IncDBSCAN",
        "  (MinPts=2), NLI LF, NLI IncDBSCAN (MinPts=5), all at t=0.10.",
        "Execution here = archive source run (run01…run05), not RQ2 replicates.",
    ]
    for i, line in enumerate(lines, start=1):
        ws.cell(i, 1, line)
    ws.column_dimensions["A"].width = 100


def write_vocabulary(wb: Workbook, rows: list[dict], names: dict[str, str]) -> None:
    counts: Counter[str] = Counter()
    sources: dict[str, Counter[str]] = defaultdict(Counter)
    runs: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        for tid in r["topic_ids"].split():
            counts[tid] += 1
            sources[tid][r["topic_source"]] += 1
            runs[tid].add(r["run"])
    ws = wb.create_sheet("Vocabulary")
    ws.append(
        [
            "topic_id",
            "topic",
            "n_prompts",
            "n_source_executions",
            "source_executions",
            "n_ancestry_and_text_agree",
            "n_read",
        ]
    )
    for tid in sorted(counts, key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        ws.append(
            [
                tid,
                names.get(tid, ""),
                counts[tid],
                len(runs[tid]),
                " ".join(sorted(runs[tid])),
                sources[tid].get("ancestry and text agree", 0),
                sources[tid].get("read", 0),
            ]
        )
    _style_header(ws)
    _widths(ws, [10, 70, 12, 18, 55, 24, 10])


def write_giant(wb: Workbook) -> None:
    ws = wb.create_sheet("Giant_codebook")
    ws.append(["topic_id", "topic", "notes"])
    for tid, name in load_giant_vocab():
        note = (
            "catch-all for garbled / meta text"
            if tid == "T99"
            else "manual label for giant NLI IncDBSCAN species prompts"
        )
        ws.append([tid, name, note])
    _style_header(ws)
    _widths(ws, [10, 70, 55])


def write_topics_by_cell(
    wb: Workbook, rows: list[dict], names: dict[str, str]
) -> None:
    ws = wb.create_sheet("Topics_by_cell")
    ws.append(
        [
            "setting",
            "distance",
            "speciation_method",
            "min_samples",
            "locked_online",
            "execution",
            "topic_id",
            "topic",
            "n_prompts_in_mature_species",
            "n_mature_species_containing_topic",
            "share_of_execution_mature_prompts",
        ]
    )
    for setting, suffix, dist, method, ms in SETTINGS:
        col = f"species__{suffix}"
        by_species = mature_members(rows, col)
        # topic -> species set, and topic -> counts per execution
        species_with: dict[str, set[int]] = defaultdict(set)
        cell_topic_run: dict[tuple[str, str], int] = Counter()
        run_totals: Counter[str] = Counter()
        for sid, members in by_species.items():
            for r in members:
                run_totals[r["run"]] += 1
                for tid in r["topic_ids"].split():
                    species_with[tid].add(sid)
                    cell_topic_run[(r["run"], tid)] += 1
        for (run, tid), n in sorted(
            cell_topic_run.items(), key=lambda x: (x[0][0], -x[1], x[0][1])
        ):
            total = run_totals[run] or 1
            ws.append(
                [
                    setting,
                    dist,
                    method,
                    ms,
                    "yes" if setting in LOCKED else "no",
                    run,
                    tid,
                    names.get(tid, ""),
                    n,
                    len(species_with[tid]),
                    round(n / total, 4),
                ]
            )
    _style_header(ws)
    _widths(ws, [18, 10, 16, 12, 14, 18, 10, 70, 18, 18, 16])


def write_species_topics(
    wb: Workbook, rows: list[dict], names: dict[str, str]
) -> None:
    ws = wb.create_sheet("Species_topics")
    ws.append(
        [
            "setting",
            "distance",
            "speciation_method",
            "min_samples",
            "locked_online",
            "species_id",
            "species_size",
            "dominant_topic_id",
            "dominant_topic",
            "n_topics_in_species",
            "prompts_in_dominant_topic",
            "prompts_outside_dominant_topic",
            "mixed",
            "execution",
            "n_prompts_from_execution",
            "n_dominant_from_execution",
        ]
    )
    for setting, suffix, dist, method, ms in SETTINGS:
        col = f"species__{suffix}"
        by_species = mature_members(rows, col)
        for sid in sorted(by_species, key=lambda s: (-len(by_species[s]), s)):
            members = by_species[sid]
            tally: Counter[str] = Counter()
            member_topics = []
            for r in members:
                ts = r["topic_ids"].split()
                member_topics.append(frozenset(ts))
                for tid in ts:
                    tally[tid] += 1
            dom = tally.most_common(1)[0][0]
            n_dom = sum(1 for ts in member_topics if dom in ts)
            mixed = "yes" if not frozenset.intersection(*member_topics) else "no"
            by_run: dict[str, list[dict]] = defaultdict(list)
            for r in members:
                by_run[r["run"]].append(r)
            for run in sorted(by_run):
                sub = by_run[run]
                n_dom_run = sum(1 for r in sub if dom in r["topic_ids"].split())
                ws.append(
                    [
                        setting,
                        dist,
                        method,
                        ms,
                        "yes" if setting in LOCKED else "no",
                        sid,
                        len(members),
                        dom,
                        names.get(dom, ""),
                        len(tally),
                        n_dom,
                        len(members) - n_dom,
                        mixed,
                        run,
                        len(sub),
                        n_dom_run,
                    ]
                )
    _style_header(ws)
    _widths(ws, [18, 10, 16, 12, 14, 12, 12, 14, 55, 12, 14, 14, 10, 18, 14, 14])


def write_prompt_topics(wb: Workbook, rows: list[dict]) -> None:
    ws = wb.create_sheet("Prompt_topics")
    header = [
        "prompt_index",
        "execution",
        "genome_id",
        "fitness",
        "topic_ids",
        "topic",
        "topic_source",
        "ancestry_topic_ids",
        "text_nearest_topic_id",
        "prompt",
    ]
    # locked species columns only, to keep the sheet readable
    locked_cols = [
        (name, f"species__{suffix}")
        for name, suffix, *_ in SETTINGS
        if name in LOCKED
    ]
    for name, _ in locked_cols:
        header.append(f"species_{name}")
    ws.append(header)
    for r in rows:
        row = [
            int(r["prompt_index"]),
            r["run"],
            r["genome_id"],
            float(r["fitness"]),
            r["topic_ids"],
            r["topic"],
            r["topic_source"],
            r["ancestry_topic_ids"],
            r["text_nearest_topic_id"],
            " ".join(r["prompt"].split()),
        ]
        for _, col in locked_cols:
            row.append(int(r[col]))
        ws.append(row)
    _style_header(ws)
    widths = [12, 18, 12, 10, 14, 55, 22, 16, 16, 80] + [14] * len(locked_cols)
    _widths(ws, widths)
    for cell in ws.iter_rows(min_row=2, min_col=10, max_col=10):
        cell[0].alignment = Alignment(wrap_text=True, vertical="top")


def build(out: Path) -> Path:
    rows = load_labels()
    names = topic_catalog(rows)
    wb = Workbook()
    wb.remove(wb.active)
    write_readme(wb)
    write_vocabulary(wb, rows, names)
    write_giant(wb)
    write_topics_by_cell(wb, rows, names)
    write_species_topics(wb, rows, names)
    write_prompt_topics(wb, rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not LABELS.is_file():
        raise SystemExit(f"missing {LABELS}; run rq1_species_topic_audit.py first")
    path = build(args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
