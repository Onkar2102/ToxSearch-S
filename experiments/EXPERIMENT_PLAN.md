# Experimental Design Plan

**Project:** ToxSearch-S — Speciation-Driven Adversarial Prompt Search  
**Date:** 2026-03-31  
**Status:** DRAFT — single canonical plan (this file only)

---

## Quick reference

| Cat | Topic | Runs | Budget | Main question |
|-----|-------|------|--------|----------------|
| **C1** | ToxSearch vs ToxSearch-S vs RainbowPlus | 10 × 3 methods = **30** | 1 000 evals/run | Quality & diversity vs baselines |
| **C2** | Sequential vs distributed ToxSearch-S | **5 + 5 = 10** | 5 000 evals/run | Wall-clock, throughput, search equivalence |
| **C3** | Species / niches | **0 new runs** | — | Interpret structure from C1+C2 outputs |

**Total scheduled runs:** 40 (C3 reuses C1/C2 data).

---

## Shared foundation

**Terminology:** **ToxSearch** = no speciation (baseline); **ToxSearch-S** = speciated EA; **Distributed** = `--parallel` MPI; **RainbowPlus** = MAP-Elites QD baseline (`rainbowplus-main/`). **Evaluation** = one scored prompt–response pair (budget unit).

**Fair controls:** Same target LLM (`meta-llama/Llama-3.2-3B-Instruct`), Perspective `TOXICITY`, a **CSV** seed file with a `questions` column and ≥100 rows (see `load_and_initialize_population` in `src/utils/population_io.py`; the loader uses `pd.read_csv`, not JSONL), decoding aligned with `rainbowplus-main/configs/SOTA_COMPARISON.md`, paired `--seed` values where noted.

**Budget:** ToxSearch-S uses `--max-total-genomes`; RainbowPlus uses `--max_genomes` (rows in `all_genomes.jsonl`). Verify counts after each run.

---

# Category 1 (C1) — Quality & diversity (three-way comparison)

## Aims

- Show whether **speciation** improves **search quality** (toxicity) vs **non-speciated ToxSearch** and vs **RainbowPlus** (predefined category×style grid).
- Show whether **diversity** of failure modes (embedding clusters, topics) differs across the three methods at the **same evaluation budget**.

## Experiment design

| Factor | Setting |
|--------|---------|
| Methods | ToxSearch (baseline), ToxSearch-S, RainbowPlus |
| Replications | **10 per method** (seeds 0–9) |
| Budget | **1 000** evaluations per run |
| Stats | Kruskal–Wallis → pairwise Mann–Whitney, Holm–Bonferroni, Cliff’s δ, bootstrap CIs |

**Run commands:** See **Appendix A** (bash loops + Slurm array for RainbowPlus). **Important:** `src/main.py` **always runs speciation** in both sequential and `--parallel` modes; there is **no** `--no-speciation` (or similar) flag. **Baseline ToxSearch** for C1 must be defined explicitly (e.g. separate entrypoint, branch, or a small code path that skips speciation)—adjust Appendix A once that exists.

## What we have in the codebase

| Piece | Location / role |
|-------|-----------------|
| Main search | `src/main.py` — sequential ToxSearch-S |
| RainbowPlus fork | `rainbowplus-main/rainbowplus/rainbowplus.py`, `sbatch_rainbowplus.sh` |
| C1-style analysis | `experiments/rq1.py` → `experiments/comparison_results/rq1_quality/` (quality, diversity, figures; **currently 2-group baseline vs speciated** — extend to **3 groups** + RainbowPlus paths) |
| Fair comparison notes | `rainbowplus-main/configs/SOTA_COMPARISON.md` |
| Per-run metrics helper | `scripts/experiment_metrics.py` (throughput, time-to-threshold in seconds) |

## What is required

| Gap | Action |
|-----|--------|
| **3-way stats** | Extend `rq1.py` (or new script) for **Kruskal–Wallis** + RainbowPlus run directories |
| **Baseline implementation** | Non-speciated baseline is **not** a flag today; implement or document the chosen entrypoint, then update Appendix A |
| **RainbowPlus data** | Parse `all_genomes.jsonl` + archives; align metrics with ToxSearch-S outputs |
| **Bootstrap CI** | Wire existing bootstrap helper in `rq1.py` into saved tables |
| **C1 metrics** | Quality: Q_max, AUC, Top10/50, TTT(t). Diversity: K_clusters, S_spread, BERTopic metrics (see Appendix B) — embeddings + optional BERTopic deps |

---

# Category 2 (C2) — Sequential vs distributed ToxSearch-S

## Aims

- **Speed / efficiency:** Does **MPI distributed** reduce **wall-clock** and raise **genomes/s** vs **sequential** for the **same** `--max-total-genomes`, without harming search outcomes?
- **Bottlenecks:** Where is time spent (LLM, Perspective, speciation, API wait)?
- **Optional (advisor-style scalability):** “More workers → more genomes/s + API wait” needs **extra runs** varying worker count (not only seq vs 4 workers) — see **Scalability extension** below.

## Experiment design

| Factor | Setting |
|--------|---------|
| Modes | Sequential `python3 src/main.py` vs `mpiexec -n 5 ... --parallel` (1 master + 4 workers) |
| Replications | **5 per mode**, **same seeds** 0–4 (paired) |
| Budget | **5 000** evaluations per run |
| Figures | Total time, throughput, time breakdown, best toxicity vs wall time vs genome index |

**Scalability extension (optional):** Same budget, vary MPI size (e.g. 2, 3, 4, 5 ranks → 1–4 workers). Not required for minimal C2; needed to fully answer “adding workers” narratives.

## What we have in the codebase

| Piece | Location / role |
|-------|-----------------|
| Sequential timing | `src/main.py` — `generation_duration_seconds`, `calculate_budget_metrics` / `generations[].budget` in `EvolutionTracker.json` |
| Parallel timing | `src/parallel/master_worker.py` — cycle duration, `total_wait_for_results_seconds`, merge+speciate; worker stats with response/eval/API wait |
| Post-hoc compare | `experiments/compare_sequential_vs_parallel.py` — reads two run dirs, `comparison_summary.json`, throughput figure |
| Scaling plots | `experiments/plot_scaling_curves.py` — multiple `EvolutionTracker.json` |
| Worker aggregation | `scripts/aggregate_worker_metrics.py` (parallel runs) |
| Run metrics | `scripts/experiment_metrics.py` |

## What is required

| Gap | Action |
|-----|--------|
| **Comparable wall-clock** | **Aggregate `T_run`:** use `run_metadata.run_duration_seconds`; confirm includes gen0→end (**parallel drain** included). **Per-generation:** sequential vs parallel **`generation_duration_seconds` scopes differ** — document one rule or add aligned phase timers (end after speciation + population update; exclude/include viz consistently). See **Appendix C**. |
| **Time breakdown parity** | Verify **every** generation has populated **`budget`** (LLM / eval / `total_evaluation_api_wait_seconds`) in **both** modes for fair fractions. |
| **C2 analysis script** | Dedicated script or extend `compare_sequential_vs_parallel.py` for **5×5 paired** seeds + C1 quality/diversity checks |
| **Worker sweep** | Not a code change — **schedule extra runs** if scalability questions are in scope |

---

# Category 3 (C3) — Species analysis

## Aims

- Describe **species** as semantic niches: toxicity by species, separation, labels, dynamics (emergence, merges, extinctions).
- **Bridge:** ToxSearch-S **species** vs RainbowPlus **archive cells**; seq vs dist **species consistency** (labels / overlap).

## Experiment design

- **No new runs.** Use **C1** ToxSearch-S (10 × 1 000) and **C2** (10 × 5 000 total) outputs.
- Per-run inputs: `speciation_state.json`, `elites.json`, `reserves.json`, `EvolutionTracker.json`; optional GDP HTML from `scripts/generate_interactive_3d_plots.py` / live analysis.

## What we have in the codebase

| Piece | Location / role |
|-------|-----------------|
| Species-focused analysis | `experiments/rq2.py` → `experiments/comparison_results/rq2_speciation/` |
| Per-run species figure | `scripts/generate_run_analysis_figures.py` (toxicity by species) |
| 3D GDP | `scripts/generate_interactive_3d_plots.py`, `src/utils/gdp_projection.py` |
| Tracker fields | Per-generation `species_count`, merge/extinction events in `EvolutionTracker.json` |

## What is required

| Gap | Action |
|-----|--------|
| **Cross-method bridge** | New analysis: compare RainbowPlus archive keys to species labels (similarity / overlap) — **not** in `rq2.py` yet |
| **Seq vs dist species** | Compare C2 sequential vs distributed **label / count** consistency — script TBD (`c3_species_analysis.py` or extend `rq2.py`) |
| **Optional** | Per-species toxicity **over time** needs snapshots; not always in tracker — scope as optional |

---

## Execution checklist (all categories)

**Pre-flight:** Seed file ≥100 rows; `.env` Perspective keys; Llama model available; create `data/outputs/c1_*`, `c2_*` dirs; GPUs for C2 distributed.

**Order:** C1 RainbowPlus → C1 ToxSearch → C1 ToxSearch-S → C2 sequential → C2 distributed.

**Post-run:** Verify genome counts; `experiment_metrics.py` on ToxSearch-S runs; `aggregate_worker_metrics.py` on parallel C2; embeddings for C1 diversity.

**Analysis:** Extend `rq1.py` (C1); C2 script; `rq2.py` / new C3 script; figures + `matplotlib_embed_fonts` for PDFs.

---

## Threats to validity (short)

Single target model and single fitness (Perspective); API stochasticity; C1 budget modest; RainbowPlus vs ToxSearch-S iteration batching differs — comparisons by **total evaluations**; speciation hyperparameters fixed.

---

## Deliverables

| Artifact | Location |
|----------|----------|
| Raw outputs | `data/outputs/c1_*/`, `c2_*/` |
| Processed results | `experiments/comparison_results/rq1_quality/`, `rq2_speciation/`, future `c2_performance/`, `c3_species/` |
| This plan | `experiments/EXPERIMENT_PLAN.md` (canonical) |

---

## Appendix A — C1 run commands (illustrative)

**ToxSearch-S (10 runs):**
```bash
for i in $(seq 0 9); do
  python3 src/main.py \
    --seed-file data/toxsearch_seed.csv \
    --max-total-genomes 1000 \
    --batch-size 100 \
    --seed $i \
    --output-dir data/outputs/c1_toxsearch_s/run$(printf '%02d' $i)
done
```

**RainbowPlus (Slurm array, 10 runs):**
```bash
sbatch --export=ALL,USE_ARRAY_RUN_ID=1,MAX_GENOMES=1000,NUM_MUTATIONS=3 \
  --array=0-9 rainbowplus-main/sbatch_rainbowplus.sh
```

**Baseline ToxSearch:** Use the project-defined non-speciated entrypoint when available; place outputs under `data/outputs/c1_toxsearch/runXX`. (Same `--seed-file` / `--seed` pattern as above once the runner exists.)

---

## Appendix A2 — C2 run commands (5 + 5 paired seeds)

**Sequential:**
```bash
for i in $(seq 0 4); do
  python3 src/main.py \
    --seed-file data/toxsearch_seed.csv \
    --max-total-genomes 5000 \
    --batch-size 100 \
    --seed $i \
    --output-dir data/outputs/c2_sequential/run$(printf '%02d' $i)
done
```

**Distributed (1 master + 4 workers = 5 ranks):**
```bash
for i in $(seq 0 4); do
  mpiexec -n 5 python3 src/main.py \
    --parallel \
    --seed-file data/toxsearch_seed.csv \
    --max-total-genomes 5000 \
    --batch-size 100 \
    --seed $i \
    --output-dir data/outputs/c2_distributed/run$(printf '%02d' $i)
done
```

---

## Appendix B — C1 metrics (summary)

**Quality (aggregate unless noted):** Q_max, AUC, Top10, Top50; **TTT(t)** (per-generation / first-hit index) for t ∈ {0.5, 0.7, 0.8, 0.9}.

**Diversity (aggregate):** K_clusters (DBSCAN), S_spread, N_topics, Evenness, H (BERTopic).

---

## Appendix C — Wall-clock parity (C2)

- **`T_run`:** use `EvolutionTracker.json` → `run_metadata.run_duration_seconds` (written at end of the run). Throughput ≈ total evaluated genomes / `T_run`. Confirm parallel **drain** is included (final buffer merge before shutdown). Note: sequential `run_duration_seconds` is computed **before** the optional end-of-run GDP plot in `main.py`—keep the same definition across modes when comparing.
- **Per-generation `generation_duration_seconds` (no code change required for C2 if you follow the rules below):**
  - **Sequential** (`src/main.py`): duration = `gen_start` (top of generation loop) → moment **before** `update_evolution_tracker_with_statistics`, after speciation and `calculate_generation_statistics` (and related validation). **Excludes** per-generation `run_live_analysis` and the tracker write.
  - **Parallel** (`src/parallel/master_worker.py`): duration = prior cycle’s `gen_start` → end of `_merge_and_speciate`, **before** `_update_tracker`. **Excludes** per-generation `_run_live_analysis` and `_update_tracker` time.
  - So both modes **omit** per-gen visualization from the stored duration, but **sequential still includes more post-speciation bookkeeping** than the parallel slice. Treat **cross-mode** per-generation duration plots as **indicative**, not strictly aligned, unless you add matching phase timers in code.
- **Fair per-generation time breakdown:** Prefer **`generations[].budget`** (LLM / eval / API wait fields from `calculate_budget_metrics`) where populated in **both** modes—closer to comparable phases than raw `generation_duration_seconds`.
- **Safe curves (recommended):** best toxicity vs **cumulative genomes**; cumulative **seconds** vs genome index with one monotonic clock rule; optional **aligned timers** in code only if you need strict per-gen wall-clock parity between seq and MPI.

---

## Appendix D — Revision history (from earlier drafts)

Omnibus test for C1; paired seeds for C2; C3 dynamics + cross-category bridges; bootstrap CIs; TTT thresholds 0.5–0.9; optional worker sweep for scalability narratives.

---

## Legacy planning notes

Older standalone notes in `experiments/` were **removed** as redundant with this document: `RESEARCH_QUESTIONS.txt`, `FOUR_LEVEL_METRICS_GAP_AND_IMPLEMENTATION.txt`, `SAFE_TO_IMPLEMENT_METRICS.txt`, `PLAN_REFERENCE_REPOS_METRICS_ANALYTICS.txt`. Worker-scaling detail that was unique to `RESEARCH_QUESTIONS.txt` is summarized under **C2 → Scalability extension**.
