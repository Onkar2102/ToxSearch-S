# Experimental Design Plan

**Project:** ToxSearch-S — Speciation-Driven Adversarial Prompt Search  
**Date:** 2026-04-15 (documentation sync with codebase)  
**Status:** DRAFT — single canonical plan (this file only)

---

## Quick reference

| Cat | Topic | Runs | Budget | Main question |
|-----|-------|------|--------|----------------|
| **C1** | ToxSearch vs ToxSearch-S vs RainbowPlus | 10 × 3 methods = **30** | 1 000 evals/run | Same-budget quality **and** diversity across three methods |
| **C2** | Sequential vs distributed ToxSearch-S | **5 + 5 = 10** | 5 000 evals/run | Wall-clock, throughput, search equivalence |
| **C3** | Species / niches | **0 new runs** | — | Interpret structure from C1+C2 outputs |

**Total scheduled runs:** 40 (C3 reuses C1/C2 data).

**Paper-ready aggregates (PPSN2026):** Precomputed tables and PDFs for a **1 000 genome** cohort live under `experiments/comparison_results/c1_ppsn2026_two_way/`, `c2_ppsn2026_seq_vs_2w/`, `c3_ppsn2026_species/`, and `ppsn2026_operator_delta/`. Regenerate from repo root: `python experiments/comparison_results/c2_ppsn2026_seq_vs_2w/c2_seq_vs_2w_report.py`, `python experiments/comparison_results/c3_ppsn2026_species/c3_species_report.py`, `python experiments/comparison_results/ppsn2026_operator_delta/operator_delta_report.py` (C1 two-way figures: `experiments/comparison_results/c1_two_way_figures.py` when needed).

---

## Shared foundation

**Terminology:** **ToxSearch** = no speciation (baseline); **ToxSearch-S** = speciated EA; **Distributed** = `--parallel` MPI; **RainbowPlus** = MAP-Elites QD baseline (`rainbowplus-main/`). **Evaluation** = one scored prompt–response pair (budget unit).

**Fair controls:** Same target LLM (`meta-llama/Llama-3.2-3B-Instruct`), Perspective `TOXICITY`, a **CSV** seed file with a `questions` column and ≥100 rows (see `load_and_initialize_population` in `src/utils/population_io.py`; the loader uses `pd.read_csv`, not JSONL), decoding aligned with `rainbowplus-main/configs/SOTA_COMPARISON.md`, paired `--seed` values where noted.

**Budget:** ToxSearch-S uses `--max-total-genomes`; RainbowPlus uses `--max_genomes` (rows in `all_genomes.jsonl`). Verify counts after each run.

---

# Category 1 (C1) — Quality & diversity (three-way comparison)

## Research question (C1)

Under a **common evaluation budget** and matched experimental controls, **how do non-speciated ToxSearch, speciated ToxSearch-S, and RainbowPlus (MAP-Elites on a fixed category×style archive) compare jointly on toxicity-focused search quality and on diversity of harmful failure modes** (e.g., embedding-space clustering and/or topic structure of successful attacks)?

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
| **Comparable wall-clock** | **Aggregate `T_run`:** use `run_metadata.run_duration_seconds`; confirm includes gen0→end (**parallel drain** included). **Per-generation:** both modes now share `generation_duration_scope = through_evolution_tracker_statistics_write` (see README + Appendix C); start anchors still differ—use **`generations[].budget`** for phase-like fractions when possible. |
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
| PPSN2026 species tables / milestones | `experiments/comparison_results/c3_ppsn2026_species/c3_species_report.py` |
| RainbowPlus ↔ species bridge (optional) | `experiments/c3_species_bridge.py` — batch CSV of `tracker_path,rainbow_jsonl` |
| Per-run species figure | `scripts/generate_run_analysis_figures.py` (toxicity by species) |
| 3D GDP | `scripts/generate_interactive_3d_plots.py`, `src/utils/gdp_projection.py` |
| Tracker fields | Per-generation `species_count`, merge/extinction events in `EvolutionTracker.json` |

## What is required

| Gap | Action |
|-----|--------|
| **Cross-method bridge** | Optional: run `c3_species_bridge.py` when `all_genomes.jsonl` paths exist; extend metrics if the paper needs finer overlap than the bridge table |
| **Seq vs dist species** | C3 report (`c3_species_report.py`) compares C2 sequential vs 2-worker cohorts; extend if you need more MPI widths or seed-paired C2 |
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
| Raw outputs | `data/outputs/c1_*/`, `c2_*/`, and paper cohorts under `data/outputs/ppsn2026/` (e.g. `toxsearch_s`, `toxsearch_s_2w`, `rainbow_plus`) |
| Processed results | `experiments/comparison_results/rq1_quality/`, `rq2_speciation/`, `c1_ppsn2026_two_way/`, `c2_ppsn2026_seq_vs_2w/`, `c3_ppsn2026_species/`, `ppsn2026_operator_delta/` |
| This plan | `experiments/EXPERIMENT_PLAN.md` (canonical) |

---

## Appendix A — C1 run commands (illustrative)

Use `export PYTHONPATH=src` and run from the **ToxSearch-S repository root**. **`--batch-size` applies only to `--parallel` MPI runs** (merge `K`); sequential `src/main.py` ignores it.

**ToxSearch-S (10 runs, sequential):**
```bash
export PYTHONPATH=src
for i in $(seq 0 9); do
  python3 src/main.py \
    --seed-file data/toxsearch_seed.csv \
    --max-total-genomes 1000 \
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
export PYTHONPATH=src
for i in $(seq 0 4); do
  python3 src/main.py \
    --seed-file data/toxsearch_seed.csv \
    --max-total-genomes 5000 \
    --seed $i \
    --output-dir data/outputs/c2_sequential/run$(printf '%02d' $i)
done
```

**Distributed (1 master + 4 workers = 5 ranks):**
```bash
export PYTHONPATH=src
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

- **`T_run`:** use `EvolutionTracker.json` → `run_metadata.run_duration_seconds` (written at end of the run). Throughput ≈ total integrated genomes / `T_run`. Confirm parallel **drain** is included (final buffer merge before shutdown). Sequential `run_duration_seconds` may be computed before optional end-of-run visualization (e.g. GDP plots)—document whether your comparison includes that tail wall time.
- **Per-generation `generation_duration_seconds` + `generation_duration_scope`:** Both modes set `generation_duration_scope` to **`through_evolution_tracker_statistics_write`** — wall time until the generation row is **persisted** by `update_evolution_tracker_with_statistics` (trailing edge aligned). **Start anchors still differ** (sequential: generation loop / gen0 bootstrap; parallel: master clock after the previous generation’s tracker update through merge, dedup, speciation, and master-side stats prep). **Not included:** adaptive selection and visualization passes **after** that tracker write. See [README.md](../README.md) (run artifacts under **Documentation**).
- **Fair per-generation time breakdown:** Prefer **`generations[].budget`** (LLM / eval / API wait from `calculate_budget_metrics` in `src/utils/population_io.py`) where populated in **both** modes.
- **Safe curves (recommended):** best toxicity vs **cumulative genomes**; cumulative wall time vs genome index with a single definition of `T_run`; use optional extra phase timers only if you need stricter per-generation parity than the shared tracker-write trailing edge.

---

## Appendix D — Revision history (from earlier drafts)

Omnibus test for C1; paired seeds for C2; C3 dynamics + cross-category bridges; bootstrap CIs; TTT thresholds 0.5–0.9; optional worker sweep for scalability narratives.

---

## Legacy planning notes

Older standalone notes in `experiments/` were **removed** as redundant with this document: `RESEARCH_QUESTIONS.txt`, `FOUR_LEVEL_METRICS_GAP_AND_IMPLEMENTATION.txt`, `SAFE_TO_IMPLEMENT_METRICS.txt`, `PLAN_REFERENCE_REPOS_METRICS_ANALYTICS.txt`. Worker-scaling detail that was unique to `RESEARCH_QUESTIONS.txt` is summarized under **C2 → Scalability extension**.
