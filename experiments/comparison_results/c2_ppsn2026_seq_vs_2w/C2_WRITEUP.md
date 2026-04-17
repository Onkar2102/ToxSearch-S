# C2 — Sequential vs Distributed (2w, 4w) ToxSearch-S

This section reports **Category 2 (C2)**: a publication-grade comparison of **sequential** ToxSearch-S vs **MPI distributed** ToxSearch-S at **2 workers** and **4 workers** (each with a single master rank).

## Research questions (C2)

- **Performance**: At a fixed evaluation budget, how much do 2-worker and 4-worker MPI runs reduce **wall-clock time** and increase **throughput (integrated genomes/s)** vs sequential?
- **Outcome preservation**: Do distributed runs preserve **search quality** (toxicity best-so-far) and **diversity** (speciation/cluster quality proxies) relative to sequential?

## Experimental controls

- **Budget definition**: one “evaluation” is one scored prompt–response pair (Perspective `TOXICITY`). The cohort analyzed here uses **`max_total_genomes = 1000`** (PPSN2026 cohort).
- **Termination**: `--max-total-genomes` (primary termination in both sequential and MPI modes).
- **Environment controls to report**: hardware (GPU/CPU), MPI stack, `mpi4py`, Python + pinned deps.

## Data and artifacts (in-repo)

All C2 paper-ready artifacts are written under:

- `experiments/comparison_results/c2_ppsn2026_seq_vs_2w/`
  - `run_manifest.csv`: run directory manifest (seq, 2w, 4w).
  - `metrics_per_run.csv`: per-run endpoints (performance + quality + diversity).
  - `stats_summary.json`: Kruskal–Wallis + pairwise MWU (+ Holm) + paired-by-index sensitivity.
  - `stats_table.csv`: flattened stats table.
  - `figures/` (PDF): performance, outcomes, diversity.

The underlying run outputs live under:

- `data/outputs/ppsn2026/toxsearch_s/` (sequential)
- `data/outputs/ppsn2026/toxsearch_s_2w/` (parallel 2 workers)
- `data/outputs/ppsn2026/toxsearch_s_4w/` (parallel 4 workers)

## How to reproduce this analysis

From the repository root:

```bash
.venv/bin/python experiments/comparison_results/c2_ppsn2026_seq_vs_2w/c2_seq_vs_2w_report.py
```

This regenerates the CSVs/JSON and all PDFs under `experiments/comparison_results/c2_ppsn2026_seq_vs_2w/`.

## Endpoints

### Performance

- **Wall-clock (s)**: `EvolutionTracker.json → run_metadata.run_duration_seconds`
- **Throughput**: `total_integrated / wall_s`, where `total_integrated` is the sum across generations of:
  - `variants_integrated` when present (MPI), else fallbacks (`total_evaluated`, then `budget.llm_calls`).
- **Time breakdown** (per-run totals + fractions): summed over generations from tracker fields:
  - LLM time: `budget.total_response_time + budget.total_variant_creation_time`
  - Moderation/eval time: `budget.total_evaluation_time + budget.total_evaluation_api_wait_seconds`
  - Speciation time: `speciation.speciation_duration_seconds`
  - Overhead: `generation_duration_seconds - (llm + moderation + speciation)` when wall is available

### Quality

- **`qmax_tracker`**: max of `generation.max_score_variants` across generations (best-so-far toxicity proxy).
- **AUC(best-so-far vs evals)**: trapezoidal AUC of cumulative best toxicity vs cumulative evaluated genomes.
- **Time-to-threshold**: first cumulative wall time where best-so-far ≥ {0.10, 0.15, 0.20}.

### Diversity (publication-friendly, tracker-native)

From the **final generation** speciation record:

- `final_species_count`, `final_active_species_count`
- `final_inter_species_diversity`, `final_intra_species_diversity`
- `final_silhouette`, `final_davies_bouldin`, `final_calinski_harabasz` (cluster quality)

## Statistical analysis

Primary (unpaired) analysis is reported for each metric:

- **3-group**: Kruskal–Wallis (seq vs 2w vs 4w).
- **Pairwise**: Mann–Whitney U for (seq,2w), (seq,4w), (2w,4w), with **Holm** adjustment per metric.
- **Effect sizes**: Cliff’s delta + bootstrap CIs (median diff and delta).

Sensitivity (paired-by-index) analysis:

- Pair runs within each mode by sorted `run_id` index and run **Wilcoxon signed-rank** on paired differences.
- This is a deterministic robustness check; it is **not a guarantee of identical RNG seeds** (the seed is not currently recorded in `EvolutionTracker.json`).

## Results (PPSN2026 cohort, n=7 runs/mode, 1000 max_total_genomes)

Key medians with IQR (Q1–Q3), extracted from `metrics_per_run.csv`:

- **Wall-clock (s)**:
  - Sequential: median \(4.595\\times 10^4\) (4.224e4–4.600e4)
  - 2 workers: median \(2.506\\times 10^4\) (2.481e4–2.530e4)
  - 4 workers: median \(1.427\\times 10^4\) (1.402e4–1.522e4)
  - Kruskal \(p=1.35\\times 10^{-4}\)
- **Throughput (integrated genomes/s)**:
  - Sequential: median 0.02233 (0.02188–0.02394)
  - 2 workers: median 0.04030 (0.04004–0.04074)
  - 4 workers: median 0.07124 (0.06687–0.07277)
  - Kruskal \(p=1.35\\times 10^{-4}\)
- **Quality (`qmax_tracker`)**:
  - Similar across modes (Kruskal \(p=0.929\))

Recommended figures to cite in the paper (all PDFs under `figures/`):

- `throughput_wall_vs_evaluated_genomes.pdf` (two panels: wall time left, throughput right; common $x\in[0,1000]$; mean and min–max band per mode)
- `time_breakdown_mean_fractions.pdf`
- `toxicity_diversity_vs_evaluated_genomes.pdf` (2$\times$2: top inter/intra diversity with legend on intra; bottom toxicity + species count; $x$ ticks every 200 genomes; median + IQR)
- `best_so_far_vs_wall_time.pdf`

## Limitations / reporting notes

- **Seed pairing**: The run metadata in `EvolutionTracker.json` does not currently include the RNG seed, so strict paired-seed statistics cannot be guaranteed from the tracker alone. We report both unpaired tests and a paired-by-index sensitivity check.
- **Per-generation timing comparability**: Per-generation durations are aligned by a shared trailing edge (`generation_duration_scope = through_evolution_tracker_statistics_write`), but **start anchors differ** between sequential and MPI; interpret per-generation wall-time slices accordingly.

