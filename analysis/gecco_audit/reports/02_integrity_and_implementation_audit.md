# Integrity and implementation audit

**Status:** one-pass read-only audit  
**Date:** 2026-11-07  
**Inputs:** `experiments/evostar_2027/rq2/reports/rq2_status.json`, `rq2_run_level.csv`, `rq2_mannwhitney.csv`; code under `src/speciation/`; manuscript under `notes/rough_draft/`

## A. Forty-run integrity

| Metric | Value | Evidence |
|--------|------:|----------|
| Planned cells | 40 | \(2\times2\times10\) design |
| Complete | **37** | `rq2_status.json`, `rq2_run_level.csv` |
| Missing | **3** | see below |

Missing cells (`status=missing_run`):

1. `embedding__leader_follower__seed9`
2. `nli_entail__dbscan__seed8`
3. `nli_entail__dbscan__seed9`

Completed sample sizes by cell: LF+\(d_E\) \(n{=}9\); Inc+\(d_E\) \(n{=}10\); LF+\(d_{\neg E}\) \(n{=}10\); Inc+\(d_{\neg E}\) \(n{=}8\).

All complete runs report `n_unique` ≈ 2000 (Inc embedding/NLI sometimes 2000–2013). Do not treat genomes as independent experimental units.

Interim top-10 medians from `rq2_mannwhitney.csv` (match draft Table 2 story):

| Contrast | median LF | median Inc | Holm \(p\) (top10) | Cliff \(\delta\) CI includes 0? |
|----------|----------:|-----------:|------------------:|:--|
| embedding LF vs Inc | 0.919 | 0.928 | 0.598 | yes |
| nli_entail LF vs Inc | 0.853 | 0.912 | 0.135 | yes (CI touches 0) |

**Implication:** manuscript must keep interim / unbalanced wording; no superiority claim.

## B. Code ↔ paper consistency

| Manuscript claim | Code location | Match? | Implication |
|------------------|---------------|--------|-------------|
| \(d_E=(1-\mathrm{clip}(\cos,-1,1))/2\) on L2-normalized embeddings | `distance_metrics/embedding.py`, `embeddings.py` (`normalize_embeddings=True`) | **match** | Keep Eq. embedding distance |
| LF join uses strict \(d<\theta_{\mathrm{sim}}\) | `leader_follower.py` (`dist < theta_sim`) | **match** | Keep Alg. LF; note default 0.25 vs RQ1-selected 0.10 |
| Inc neighborhood uses \(d\le\varepsilon\) | `inc_dbscan.py` (`D[i,j] <= eps`) | **match** | Keep inclusive \(\varepsilon\) |
| Inc `min_samples` counts self | `len(neighbors[i]) >= min_samples` with self in set | **match** | MinPts=2 ⇒ need ≥1 other neighbor to be core |
| Size-1 mature labels demoted to pool 0 | `_demote_size1_*` in `inc_dbscan.py` | **match** | Mature niches require ≥2 members |
| Merge assigns expanded component members | `_expand_cluster` + relabel on merge (lines ~580–614) | **match in code** | Pseudocode Else branch must state this explicitly (patched) |
| G7: “borders do not exist for \(m{=}2\)” | Border branch exists in code but is unreachable when self-inclusive MinPts=2 | **supported with caveat** | Soften “exact / order-independent theorem” language; keep as special-case characterization |
| G7: “exact batch DBSCAN / order-independent partition” | Incremental insert path + demotion; not a one-shot CC call | **partial** | Prefer “MinPts=2 is a weak density threshold; niches are neighborhood-connected” over “proved single-linkage theorem” |
| RQ2 fitness = OpenAI `illicit` | RQ2 run configs / factorial reports | **match** (vs Perspective method examples) | Keep explicit oracle distinction |
| NLI Inc uses MinPts=5 online | `rq2_run_level.csv` `min_samples=5` for `nli_entail__dbscan` | **match** | Label as full configuration, not pure representation effect |
| Embedding Inc uses MinPts=2 | `min_samples=2` for embedding dbscan cells | **match** | — |
| Prototype grid ≠ algorithm species | coverage appendix prose | **match** | Keep proxy wording |

## C. Algorithm 3 display issue (handoff flag)

Paper Alg. IncDBSCAN Else branch previously said only “Merge all species intersecting \(G\)” without assigning new members of \(G\). **Code does assign** via `_expand_cluster` and parent relabel. Classification: **display omission**, not a silent historical search bug. Manuscript patched to state assignment of \(G\) onto the survivor species.

## D. What was *not* re-done

- No re-execution of evolutionary search
- No wholesale recompute of RQ2 figures/tables (existing `rq2_*` artifacts reused)
- No new RQ3 target-model scoring
- No human labeling of the 60-cell grid
