# RQ3 gate

**Decision (locked):** Demote RQ3 to **protocol + selection prep**. No transfer generalization claims until approved cross-model scoring exists.

**Date:** 2026-11-07

## What already exists

| Artifact | Content |
|----------|---------|
| `experiments/evostar_2027/rq3/reports/rq3_representatives.csv` | 216 selected prompts from 24 complete runs |
| `rq3_run_spread.csv` | Per-run MiniLM spread / source illicit summaries |
| `rq3_selection_meta.json` | `k=10`, design `A_transfer_of_evolved_species`, oracle `openai_illicit` |
| `notes/rough_draft/rq3_transfer_table.tex` | Placeholder `---` cells only |

Selection script (`rq3_select_representatives.py`) does **not** call response generators or moderation for transfer.

## What does *not* exist

- Target-model responses for Mistral / Qwen (or any other)
- Independent source re-scores for retention denominators
- Filled \(T_{r,m}\) / \(R_{r,m}(0.80)\) tables

## Manuscript policy

- Abstract / intro / conclusion: no claim that elites transfer.
- Experiments §RQ3: keep selection protocol; state scores are **pending / gated**.
- Appendix RQ3 map: qualitative protocol only.
- Do not invent numbers or silently start target-model evaluation.

## Approval gate

New response-generator + moderation evaluations require explicit user authorization and must be labeled as new data, not as part of the frozen RQ2 archive.
