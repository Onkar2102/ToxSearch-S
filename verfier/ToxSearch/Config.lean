import ToxSearch.Distance
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# SpeciationConfig well-formedness

Mirrors the assertions in `src/speciation/config.py` `__post_init__`.
-/

namespace ToxSearch

/-- Abstract speciation configuration (numeric fields only). -/
structure SpeciationConfig where
  thetaSim : ℝ
  thetaMerge : ℝ
  minStabilityGens : ℕ
  cluster0MaxCapacity : ℕ
  speciesCapacity : ℕ
  wGenotype : ℝ
  wPhenotype : ℝ

/-- Runtime well-formedness predicate. -/
def SpeciationConfig.WellFormed (c : SpeciationConfig) : Prop :=
  c.thetaSim ∈ Set.Icc (0 : ℝ) 1 ∧
  c.thetaMerge ∈ Set.Icc (0 : ℝ) 1 ∧
  c.thetaMerge ≤ c.thetaSim ∧
  c.wGenotype + c.wPhenotype = 1 ∧
  0 < c.cluster0MaxCapacity ∧
  0 < c.speciesCapacity

theorem wellFormed_merge_le_sim {c : SpeciationConfig} (h : c.WellFormed) :
    c.thetaMerge ≤ c.thetaSim :=
  h.2.2.1

theorem wellFormed_weights_sum {c : SpeciationConfig} (h : c.WellFormed) :
    c.wGenotype + c.wPhenotype = 1 :=
  h.2.2.2.1

theorem wellFormed_thetas_mem_unit {c : SpeciationConfig} (h : c.WellFormed) :
    c.thetaSim ∈ Set.Icc (0 : ℝ) 1 ∧ c.thetaMerge ∈ Set.Icc (0 : ℝ) 1 :=
  ⟨h.1, h.2.1⟩

/-- Default paper/runtime weights are well-formed when thresholds are. -/
theorem default_weights_ok :
    WeightsOK (0.7 : ℝ) 0.3 := by
  refine ⟨?_, by norm_num, by norm_num⟩
  norm_num

/-- Membership radius uses `θ_sim`; merge uses the stricter `θ_merge`. -/
theorem merge_implies_membership_radius
    {c : SpeciationConfig} (h : c.WellFormed) {d : ℝ}
    (hd : d < c.thetaMerge) : d < c.thetaSim :=
  lt_of_lt_of_le hd (wellFormed_merge_le_sim h)

end ToxSearch
