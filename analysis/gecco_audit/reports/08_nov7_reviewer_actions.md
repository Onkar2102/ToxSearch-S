# Actions taken on the 7 November reviewer notes (analysis only)

## Scope
No new evolutionary runs; no new oracle/model evaluations. All work is offline reanalysis of the frozen 40 runs plus the RQ1 archive matrices, plus manuscript corrections.

## Reviewer priorities

| # | Request | Status | Artifact |
|---|---|---|---|
| 1 | Species-size null for RQ3 | **Done** | `experiments/evostar_2027/rq3/reports/rq3_occupancy_null_*.csv`, updated `rq3_species_tables.tex`, null-overlay figure |
| 2 | NLI matched-neighborhood / threshold sensitivity | **Done** | `experiments/evostar_2027/rq1/reports/rq1_neighborhood_*.csv`, `rq1_matched_density_table.tex` |
| 3 | Independent-encoder coherence | **Done** | `rq3_coherence_independent_*.csv`, `rq3_coherence_independent_table.tex` |
| 4 | Non-speciated baseline | **Deferred** (user: no new runs); stated as limitation | Conclusion / intro |
| 5 | Retention ablation | **Deferred**; stated as limitation | Conclusion |
| 6 | Independent evaluator / transfer | **Deferred** (user: skip); stated as limitation | Conclusion |

## Additional integrity findings acted on

1. **Budget counters.** IncDBSCAN `tracker_api_calls` / `total_llm_calls` inflated (~2.4×). Report now uses `n_evaluated` (unique moderated genomes, 2000–2012) and `llm_calls_recount`. Truncation to first 2000 IDs does not change top-10 medians.
2. **IncDBSCAN exactness.** Exact at MinPts=2 (archive ARI=1.0). Under-merges at MinPts≥3. Online NLI (MinPts=5) final-graph ARI often ≪1. Pseudocode and complexity rewritten to match the live walk (no full core-graph rebuild).
3. **Parent fallback.** Documented: EXPLORATION samples with replacement; pool 0 is eligible.
4. **Bibliography.** Deduplicated Reimers / Pugh / Qiu / Dixon / MiniLM / ToxSearch-S keys; added Lee et al. ICLR 2025 GFlowNet red-teaming citation.

## Headline empirical results for the reframed claim

- **Matched density (RQ1):** At $t=0.10$, MiniLM edge density (0.0049) *exceeds* NLI (0.0026), yet NLI giant CC is 61% vs 11%. Dens-matched NLI ($t^*=0.33$) still has $D_{assigned}=0.96$ at MinPts=2 → **representational connectivity**, not threshold scale alone.
- **Occupancy null (RQ3):** Embedding IncDBSCAN $K_{10}=3$ vs null $5.5$ ($z=-2.06$, Stouffer–Holm $p<10^{-7}$). Embedding LF $\pi_{10}=0.75$ vs null $0.11$. NLI–IncDBSCAN $K_{10}=1$ matches the size null.
- **Claim supported:** species cardinality can misrepresent effective high-quality diversity; cannot claim speciation beats no-speciation without a baseline.

## GECCO packaging
Anonymous ACM `sigconf` scaffold in `notes/gecco2027/` with 8-page target main + supplement README.
