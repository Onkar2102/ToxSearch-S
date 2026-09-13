# ToxSearch-S mathematical audit (`verfier/`)

Lean 4 + Mathlib formalization of the algorithmic core. Build with
`lake build` (see [README.md](README.md)).

Status legend:

- **Correct** — proved in Lean as stated (or with a harmless strengthening).
- **Incorrect** — false or inconsistent with the code/spec; correction given.
- **Empirical-only** — not a Lean claim (depends on models / data / APIs).

---

## Correct

| Claim | Lean reference | Notes |
|-------|----------------|-------|
| On unit embeddings, genotype distance `dg = 1 - ⟪e₁,e₂⟫` is in `[0,2]` | `semanticDistance_nonneg`, `semanticDistance_le_two` | Clip in Python is redundant under `‖e‖=1` (Cauchy–Schwarz). |
| Normalized genotype `dgn = dg/2` lies in `[0,1]` | `semanticDistanceNorm_mem_Icc` | Matches `/2` in `distance.py`. |
| Genotype distance is symmetric | `semanticDistance_symm`, `semanticDistanceNorm_symm` | |
| `dg = 0 ↔ e₁ = e₂` (unit embeddings) | `semanticDistance_eq_zero_iff` | Pseudometric identity on the sphere. |
| Phenotype distance is in `[0,1]`, symmetric, nonnegative | `phenotypeDistance_nonneg`, `phenotypeDistance_le_one`, `phenotypeDistance_symm` | |
| On unit-cube phenotypes, `‖p₁−p₂‖ ≤ √k`, so outer `min(·,1)` is redundant | `phenotype_diameter_le_sqrt_card`, `phenotypeDistance_eq_normalized` | Harmless in Python. |
| `dp = 0 ↔ p₁ = p₂` | `phenotypeDistance_eq_zero_iff` | |
| Ensemble `d = w_g dgn + w_p dp` with `WeightsOK` stays in `[0,1]`, symmetric | `ensembleDistance_nonneg`, `ensembleDistance_le_one`, `ensembleDistance_symm` | |
| With positive weights, `d=0 ↔` same embedding and phenotype | `ensembleDistance_eq_zero_iff` | |
| `θ_merge ≤ θ_sim`, weights sum to 1, thresholds in `[0,1]` | `SpeciationConfig.WellFormed`, `merge_implies_membership_radius` | Mirrors `config.py` asserts. |
| Default weights `(0.7, 0.3)` are well-formed | `default_weights_ok` | |
| Additive **2-relaxed triangle** for genotype / ensemble | `semanticDistance_two_relaxed_triangle`, `ensembleDistance_two_relaxed_triangle` | Paper’s “2-inframetric” claim in additive form: `d(x,z) ≤ 2(d(x,y)+d(y,z))`. |
| Phenotype (unit cube) satisfies ordinary triangle | `phenotypeDistance_triangle_on_unit_cube` | Scaled Euclidean. |
| NSGA-II dominance is irreflexive, asymmetric, transitive | `dominates_irrefl`, `dominates_asymm`, `dominates_trans` | Matches `_dominates` in `reserve_selection.py`. |
| Front capacity selection keeps `≤ capacity` individuals | `selectByFronts_length_le` | |
| Leader has max fitness among members; species id ≠ 0 | `leader_has_max_fitness`, `species_id_nonzero` | Spec of `Species.WellFormed`. |
| Population: unique ids; reserves = cluster 0; elites `species_id > 0` | `PopulationState.WellFormed` theorems | Spec of `validation.py` checks. |
| Refusal penalty `s ↦ 0.85·s` is nonnegative and ≤ `s` for `s ≥ 0` | `applyRefusalPenalty_*` | |

---

## Incorrect / inconsistent (with corrections)

**Full mathematical write-up of how to fix each item:** [CORRECTIONS.md](CORRECTIONS.md).

| Issue | Verdict | How to correct mathematically (summary) |
|-------|---------|----------------------------------------|
| “Ensemble / cosine distance is a **metric**” | **Incorrect** | Drop triangle. Replace with: nonnegative + symmetric + bounded + **additive 2-relaxed triangle** `d(x,z) ≤ 2(d(x,y)+d(y,z))`. Refutation: `cosine_not_a_metric`. Correct lemmas: `ensembleDistance_two_relaxed_triangle`, etc. |
| Wording “**2-inframetric**” | **Ambiguous** | State form (A) `d ≤ 2(d+d)` (proved). Do **not** claim max-form `d ≤ 2 max(d,d)` with constant 2 without a new proof (likely needs ~4). |
| Batch path without `‖e‖=1` | **Spec violation** | All genotype theorems require `UnitEmbedding`. Correct precondition: L2-normalize before distance. |
| Missing phenotype as `0` in ensemble | **Wrong semantics** | Correct: missing → `1` via `phenotypeDistanceOpt` / `phenotypeDistanceOpt_missing_eq_one`. |
| Silhouette = ensemble validation | **Category error** | Report silhouette in cosine-embedding space only; speciation uses ensemble `d`. |

### Corrected claim (canonical)

On unit embeddings and unit-cube phenotypes, with `w_g + w_p = 1` and `w_g,w_p ≥ 0`:

```text
0 ≤ d(a,b) ≤ 1
d(a,b) = d(b,a)
d(a,b) = 0  ↔  (e_a = e_b ∧ p_a = p_b)   (if w_g > 0 and w_p > 0)
d(a,c) ≤ 2 (d(a,b) + d(b,c))
```

That is the statement to use in place of “`d` is a metric.”

---

## Empirical-only (out of Lean scope)

- Sentence-transformer “semantic” meaning of embeddings.
- GGUF sampling / LLM mutation operators.
- Perspective / OpenAI moderation score generation.
- MPI scheduling and wall-clock behaviour.
- Paper statistics (Kruskal–Wallis, Cliff’s δ, bootstrap, topic domination).

---

## Module map

| File | Role |
|------|------|
| `ToxSearch/Distance.lean` | Definitions |
| `ToxSearch/DistanceProperties.lean` | Algebraic properties |
| `ToxSearch/Inframetric.lean` | Metricity audit + 2-relaxed triangle |
| `ToxSearch/Config.lean` | Config well-formedness |
| `ToxSearch/NSGA2.lean` | Pareto dominance + capacity selection |
| `ToxSearch/SpeciationSpec.lean` | Abstract species/population invariants |
| `ToxSearch/RefusalPenalty.lean` | `0.85` multiplier |

---

## How to re-check

```bash
cd verfier
lake build
```

All Tier-A theorems above are compiled with **no `sorry`**.
