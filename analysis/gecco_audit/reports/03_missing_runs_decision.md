# Decision: three missing RQ2 cells

**Decision (updated):** **Finished.** The three previously missing cells are on disk and the factorial is **40/40 complete**.

**Date:** 2026-11-07

## Cells that were missing (now complete)

| cell_id | distance | clustering | seed | live path |
|---------|----------|------------|-----:|-----------|
| `embedding__leader_follower__seed9` | embedding | leader_follower | 9 | `data/outputs/evostar2027/lf_embedding/rep9_s9` |
| `nli_entail__dbscan__seed8` | nli_entail | dbscan | 8 | `data/outputs/evostar2027/inc_nli/rep8_s8` |
| `nli_entail__dbscan__seed9` | nli_entail | dbscan | 9 | `data/outputs/evostar2027/inc_nli/rep9_s9` |

## Regenerated artifacts

- `experiments/evostar_2027/rq2/reports/rq2_status.json` → `n_complete: 40`
- Factorial tables, MWU, species dynamics, coverage, and RQ2 figures
- Manuscript prose updated from unbalanced 37/40 to full grid
- Canonical PDF: `notes/rough_draft/paper.pdf` (built with `-jobname=paper`; do not use `main.pdf`)
