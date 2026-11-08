# GECCO audit (thin additive package)

Derived from `notes/ToxSearchS_GECCO_Agent_Handoff.md`, scoped to what is worth keeping for the paper.

**Policy:** do not mutate historical RQ2 runs; reuse `experiments/evostar_2027/` reports; do not rebuild a parallel analysis pipeline.

| File | Role |
|------|------|
| [`reports/01_framing_lock.md`](reports/01_framing_lock.md) | Locked C1/C2 contribution framing; rejected claims |
| [`reports/02_integrity_and_implementation_audit.md`](reports/02_integrity_and_implementation_audit.md) | 37/40 status + code↔paper audit |
| [`reports/03_missing_runs_decision.md`](reports/03_missing_runs_decision.md) | Explicit decision on the 3 missing RQ2 cells |
| [`reports/04_rq3_gate.md`](reports/04_rq3_gate.md) | RQ3 demoted until transfer scoring is approved |
| [`reports/05_novelty_claims_matrix.md`](reports/05_novelty_claims_matrix.md) | Claim → evidence → permitted wording |
| [`reports/06_manuscript_patches.md`](reports/06_manuscript_patches.md) | What was patched in `notes/rough_draft/` |
| [`reports/07_rq3_annotation_protocol.md`](reports/07_rq3_annotation_protocol.md) | Path B blinded annotation protocol |

Primary artifact roots (do not duplicate):

- RQ1: `experiments/evostar_2027/rq1/`
- RQ2: `experiments/evostar_2027/rq2/reports/`, `…/figures/` (**40/40 complete**)
- RQ3 selection only: `experiments/evostar_2027/rq3/reports/`
- Manuscript source: `notes/rough_draft/main.tex`
- Canonical PDF: `notes/rough_draft/paper.pdf` (build with `latexmk -pdf -jobname=paper main.tex`; do **not** keep `main.pdf`)
