# Mandatory validation checklist

## 1. Appendix Table 6 snapshot

- Online CSV: `experiments/evostar_2027/rq2/reports/incdbscan_exactness_online.csv`
- Embedding live ARI median [min, max]: **1.000** [1.000, 1.000] (matches Table 6: True)
- NLI live ARI median [min, max]: **0.994** [0.974, 0.997] (matches Table 6: True)
- Note: Online ARI uses final-graph batch rebuild with noise canonicalization (live labels ≤0 and batch -1 → unassigned). Embedding MinPts=2 is exact; NLI MinPts=5 is near-exact on partition agreement but under-counts mature clusters relative to batch. Stale summary.md (low ARI) was regenerated from CSV.

## 2. IncDBSCAN correctness

- MinPts=2 star = connected component: **True**
- MinPts=5 star: batch n_mat=1, inc n_mat=0, core-promotion miss=True
- Live insert_point: event=`inc_dbscan_insert_noise`, hub_core=True, leaf_core=False, miss=True
- Unittest return code: 0
- Verdict: CONFIRMED core-promotion miss at MinPts=5: batch finds 1 mature cluster; grow-only insert leaves the star unassigned after the hub becomes core.

## 3. RQ2 uniqueness / MPNet outlier

- Policy: RQ2 metrics dedupe by genome id only; prompt text is not unique-keyed
- `lf_embedding`: shared-prompt-across-ids windows=0, top50 prompt-dup runs=0
- `inc_embedding`: shared-prompt-across-ids windows=0, top50 prompt-dup runs=0
- `lf_nli`: shared-prompt-across-ids windows=0, top50 prompt-dup runs=0
- `inc_nli`: shared-prompt-across-ids windows=0, top50 prompt-dup runs=0
- Outlier: lf_nli seed=5, CSV D=0.001355, cache D=0.0013545130920368088
- Unique ids/prompts in top-50: 50/50; exact duplicate pairs=0
- Interpretation: Near-zero MPNet dispersion with distinct prompt strings → near-paraphrase elite concentration, not an aggregation artifact from identical strings.

## 4. RQ3 labels / occupancy

- Convention: Mature = species_id > 0; LF archive write path sets species_id=-1; unassigned includes 0 (singleton/noise pool) and -1 (archived).
- LF archives all unassigned across audited runs: **True**
- ('embedding', 'leader_follower'): TeX K/U=5.5/13.5 vs CSV 5.5/13.5 (K_match=True, U_match=True)
- ('embedding', 'dbscan'): TeX K/U=5.0/0.5 vs CSV 5.0/0.5 (K_match=True, U_match=True)
- ('nli_entail', 'leader_follower'): TeX K/U=6.5/32.0 vs CSV 6.5/32.0 (K_match=True, U_match=True)
- ('nli_entail', 'dbscan'): TeX K/U=1.0/5.5 vs CSV 1.0/5.5 (K_match=True, U_match=True)

## Optional RQ1 provenance

{
  "available": true,
  "n_rows": 3139,
  "columns": [
    "run",
    "source_file",
    "genome_id",
    "generation",
    "variant_type",
    "prompt",
    "response",
    "fitness",
    "emb",
    "scores_google",
    "scores_openai"
  ],
  "per_run_counts": {
    "run02_speciated": 573,
    "run05_speciated": 594,
    "run01_speciated": 691,
    "run03_speciated": 642,
    "run04_speciated": 639
  },
  "note": "Provenance exists for optional per-source-archive RQ1 robustness."
}
