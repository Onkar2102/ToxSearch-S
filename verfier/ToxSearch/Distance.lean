import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

namespace ToxSearch

open scoped Real InnerProductSpace

/-- Genotype (semantic) distance on unit embeddings: `1 - cosine similarity`. -/
noncomputable def semanticDistance {n : Type*} [Fintype n]
    (e1 e2 : EuclideanSpace ℝ n) : ℝ :=
  1 - ⟪e1, e2⟫_ℝ

/-- Normalized genotype distance in `[0,1]` (divide raw cosine distance by 2). -/
noncomputable def semanticDistanceNorm {n : Type*} [Fintype n]
    (e1 e2 : EuclideanSpace ℝ n) : ℝ :=
  semanticDistance e1 e2 / 2

/-- Phenotype distance: `min(‖p1 - p2‖ / √k, 1)` with `k = Fintype.card n`. -/
noncomputable def phenotypeDistance {n : Type*} [Fintype n] [Nonempty n]
    (p1 p2 : EuclideanSpace ℝ n) : ℝ :=
  min (‖p1 - p2‖ / Real.sqrt (Fintype.card n : ℝ)) 1

/-- Ensemble distance with genotype / phenotype weights summing to 1. -/
noncomputable def ensembleDistance {nG nP : Type*} [Fintype nG] [Fintype nP] [Nonempty nP]
    (wG wP : ℝ) (e1 e2 : EuclideanSpace ℝ nG) (p1 p2 : EuclideanSpace ℝ nP) : ℝ :=
  wG * semanticDistanceNorm e1 e2 + wP * phenotypeDistance p1 p2

/-- Well-formed ensemble weights (as enforced by `SpeciationConfig` / runtime asserts). -/
def WeightsOK (wG wP : ℝ) : Prop :=
  wG + wP = 1 ∧ 0 ≤ wG ∧ 0 ≤ wP

/-- Unit embedding precondition (matches scalar `semantic_distance` in Python). -/
def UnitEmbedding {n : Type*} [Fintype n] (e : EuclideanSpace ℝ n) : Prop :=
  ‖e‖ = 1

/-- Phenotype coordinates lie in `[0,1]` (moderation scores). -/
def PhenotypeInUnitCube {n : Type*} [Fintype n] (p : EuclideanSpace ℝ n) : Prop :=
  ∀ i, p i ∈ Set.Icc (0 : ℝ) 1

/-- Intentional semantics for a missing phenotype: treat as maximal distance `1`.
Python historically disagreed between scalar (`1.0`) and ensemble (`0.0`); the
formal model picks the conservative scalar convention. -/
noncomputable def phenotypeDistanceOpt {n : Type*} [Fintype n] [Nonempty n]
    (p1 p2 : Option (EuclideanSpace ℝ n)) : ℝ :=
  match p1, p2 with
  | some a, some b => phenotypeDistance a b
  | _, _ => 1

end ToxSearch
