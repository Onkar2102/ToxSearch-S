# Mathematical corrections (inside `verfier/`)

This document states **how to correct** each incorrect or overstated claim
using the formal model already in this directory. Prefer these corrected
statements in papers, docs, and code comments.

Rebuild proofs with `lake build` from `verfier/`.

---

## 1. Do **not** call ensemble / cosine distance a metric

### What was wrong

A metric requires, for all `x,y,z`:

```text
d(x,z) ≤ d(x,y) + d(y,z)     (triangle inequality)
```

Cosine / semantic distance `dg(e₁,e₂) = 1 − ⟪e₁,e₂⟫` on unit embeddings **fails**
this. Lean refutation:

- `ToxSearch.cosine_triangle_counterexample_arith`
- `ToxSearch.cosine_not_a_metric`

Witness (unit vectors in `ℝ²`): `(1,0)`, `(1/√2,1/√2)`, `(0,1)` give

```text
dg(X,Z) = 1
dg(X,Y) + dg(Y,Z) = 2(1 − 1/√2) < 1
```

### How to correct (use this instead)

On the product space of **unit embeddings × phenotype vectors**, state that `d`
is a **bounded symmetric nonnegative** function with:

| Property | Correct statement | Lean |
|----------|-------------------|------|
| Nonnegativity / bound | `0 ≤ d ≤ 1` | `ensembleDistance_nonneg`, `ensembleDistance_le_one` |
| Symmetry | `d(a,b) = d(b,a)` | `ensembleDistance_symm` |
| Identity of indiscernibles (on the pair model) | `d=0 ↔` same `(e,p)` (positive weights) | `ensembleDistance_eq_zero_iff` |
| Relaxed triangle | `d(x,z) ≤ 2(d(x,y)+d(y,z))` | `ensembleDistance_two_relaxed_triangle` |

**Paper / README wording to use:**

> Ensemble distance is not a classical metric on prompts. On L2-normalized
> embeddings and unit-cube phenotypes it is a pseudometric-style score with a
> proved additive 2-relaxed triangle inequality
> `d(x,z) ≤ 2(d(x,y)+d(y,z))`, which is enough for pairwise leader/merge tests.

Do **not** claim metric-tree / covering-radius arguments that need strict triangle.

---

## 2. Clarify “2-inframetric” wording

### What was ambiguous

“2-inframetric” can mean either:

```text
(A)  d(x,z) ≤ 2 · (d(x,y) + d(y,z))      # additive relaxation  — PROVED
(B)  d(x,z) ≤ 2 · max(d(x,y), d(y,z))    # max-inframetric     — NOT claimed at const 2
```

For squared-Euclidean / cosine (`dg = ‖·−·‖²/2`), (A) holds with constant **2**.
Form (B) with constant 2 is **not** what we prove; a max-form would need a
larger constant (order 4 from `‖x−z‖ ≤ 2 max(‖x−y‖,‖y−z‖)` after squaring).

### How to correct

In prose, prefer the **additive** lemma and cite Lean:

```text
∀ unit e₁,e₂,e₃,
  semanticDistance e₁ e₃
    ≤ 2 · (semanticDistance e₁ e₂ + semanticDistance e₂ e₃)
```

- Genotype: `semanticDistance_two_relaxed_triangle`
- Normalized: `semanticDistanceNorm_two_relaxed_triangle`
- Ensemble (unit-cube phenotypes): `ensembleDistance_two_relaxed_triangle`

If you need a max-inframetric, prove a separate lemma with the correct constant;
do not reuse “2” without that proof.

---

## 3. Missing phenotype must be distance `1`, not `0`

### What was wrong

Scalar `phenotype_distance(None, ·)` returned `1.0` (max separation), while
`ensemble_distance` historically used `d_phenotype = 0.0` (treated as identical).
Those contradict each other mathematically.

### How to correct (formal semantics)

Use the optional phenotype map already in Lean:

```lean
-- ToxSearch/Distance.lean
phenotypeDistanceOpt : Option (EuclideanSpace ℝ n) → Option … → ℝ
-- none, _  ↦  1
-- _, none  ↦  1
-- some a, some b  ↦  phenotypeDistance a b
```

Proved: `phenotypeDistanceOpt_missing_eq_one`.

**Correct rule:** absent phenotype ⇒ **maximal** phenotype contribution `1`, so
ensemble cannot understate separation when moderation scores are missing.

Python was aligned to this in `src/speciation/distance.py`; the Lean definition
is the source of truth for the math.

---

## 4. Batch genotype distance must assume unit embeddings

### What was wrong

The scalar path required `‖e‖₂ ≈ 1`; the batch path did not. The theorems
about range `[0,1]` and the 2-relaxed triangle **assume** `UnitEmbedding`.

### How to correct

Every genotype argument in the formal model is under:

```lean
UnitEmbedding e  :=  ‖e‖ = 1
```

Without that hypothesis, `dg = 1 − ⟪e₁,e₂⟫` need not lie in `[0,2]`, and the
`/2` normalization is no longer a map into `[0,1]`.

**Correct precondition (state everywhere distances are used):**

> Embeddings are L2-normalized before genotype / ensemble distance.

Lean theorems that encode this: `abs_inner_le_one_of_unit`,
`semanticDistanceNorm_mem_Icc`, all inframetric lemmas.

---

## 5. Silhouette ≠ ensemble geometry

### What was wrong

Reporting silhouette (cosine on embeddings) as if it validated the **ensemble**
speciation geometry mixes two different distances.

### How to correct

Treat them as separate objects:

| Quantity | Distance used | What it validates |
|----------|---------------|-------------------|
| Species assignment / merge | Ensemble `d = w_g dgn + w_p dp` | QD structure used at runtime |
| Silhouette / DB / CH in analysis | Typically cosine on embeddings only | Embedding-space cluster shape |

Correct reporting sentence:

> Silhouette is computed in embedding (cosine) space and does not certify
> properties of the ensemble distance used for speciation.

No Lean change required; keep analysis claims scoped.

---

## Quick reference: replace false claim → corrected claim

| False / sloppy claim | Correct mathematical claim | Where proved |
|----------------------|----------------------------|--------------|
| “`d` is a metric” | “`d` is nonnegative, symmetric, bounded, with 2-relaxed triangle” | `Inframetric.lean`, `DistanceProperties.lean` |
| “2-inframetric (max form, const 2)” | “Additive relaxation `d ≤ 2(d+d)`” | `semanticDistance_two_relaxed_triangle` |
| “Missing phenotype ⇒ distance 0” | “Missing phenotype ⇒ distance 1” | `phenotypeDistanceOpt` |
| “Any embedding vector is fine” | “Require `‖e‖ = 1`” | `UnitEmbedding` |
| “Silhouette validates speciation distance” | “Silhouette ≠ ensemble `d`” | (documentation only) |

---

## Optional future Lean (not required for Tier A)

If you want stronger paper claims formalized next:

1. Prove a **max-inframetric** lemma with explicit constant `C` (likely `C = 4` for raw cosine before `/2`).
2. Show ensemble is a **pseudometric** after replacing genotype by angular / chordal metric (true metric on the sphere), then take convex combination with phenotype Euclidean distance.

Those would live as new theorems beside `ToxSearch/Inframetric.lean`.
