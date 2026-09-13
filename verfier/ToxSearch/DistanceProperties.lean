import ToxSearch.Distance
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Tactic

/-!
# Basic algebraic properties of ToxSearch-S distances
-/

namespace ToxSearch

open scoped Real InnerProductSpace

variable {n : Type*} [Fintype n]

theorem abs_inner_le_one_of_unit
    {e1 e2 : EuclideanSpace ℝ n} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2) :
    |⟪e1, e2⟫_ℝ| ≤ 1 := by
  calc
    |⟪e1, e2⟫_ℝ| ≤ ‖e1‖ * ‖e2‖ := abs_real_inner_le_norm e1 e2
    _ = 1 * 1 := by rw [h1, h2]
    _ = 1 := by norm_num

theorem semanticDistance_nonneg
    {e1 e2 : EuclideanSpace ℝ n} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2) :
    0 ≤ semanticDistance e1 e2 := by
  have : ⟪e1, e2⟫_ℝ ≤ 1 := (abs_le.mp (abs_inner_le_one_of_unit h1 h2)).2
  simp only [semanticDistance]
  linarith

theorem semanticDistance_le_two
    {e1 e2 : EuclideanSpace ℝ n} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2) :
    semanticDistance e1 e2 ≤ 2 := by
  have : -1 ≤ ⟪e1, e2⟫_ℝ := (abs_le.mp (abs_inner_le_one_of_unit h1 h2)).1
  simp only [semanticDistance]
  linarith

theorem semanticDistance_symm (e1 e2 : EuclideanSpace ℝ n) :
    semanticDistance e1 e2 = semanticDistance e2 e1 := by
  simp [semanticDistance, real_inner_comm]

theorem semanticDistanceNorm_mem_Icc
    {e1 e2 : EuclideanSpace ℝ n} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2) :
    semanticDistanceNorm e1 e2 ∈ Set.Icc (0 : ℝ) 1 := by
  constructor
  · have := semanticDistance_nonneg h1 h2
    simp only [semanticDistanceNorm]
    linarith
  · have := semanticDistance_le_two h1 h2
    simp only [semanticDistanceNorm]
    linarith

theorem semanticDistanceNorm_symm (e1 e2 : EuclideanSpace ℝ n) :
    semanticDistanceNorm e1 e2 = semanticDistanceNorm e2 e1 := by
  simp [semanticDistanceNorm, semanticDistance_symm]

theorem semanticDistance_eq_zero_iff
    {e1 e2 : EuclideanSpace ℝ n} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2) :
    semanticDistance e1 e2 = 0 ↔ e1 = e2 := by
  constructor
  · intro h
    have hinn : ⟪e1, e2⟫_ℝ = 1 := by
      simp only [semanticDistance] at h
      linarith
    have hsq : ‖e1 - e2‖ ^ 2 = 0 := by
      rw [norm_sub_sq_real, h1, h2, hinn]
      ring
    have hnorm : ‖e1 - e2‖ = 0 := sq_eq_zero_iff.mp hsq
    exact eq_of_sub_eq_zero (norm_eq_zero.mp hnorm)
  · intro h
    subst h
    simp only [semanticDistance]
    have : ⟪e1, e1⟫_ℝ = ‖e1‖ ^ 2 := real_inner_self_eq_norm_sq e1
    rw [this, h1]
    ring

variable {nP : Type*} [Fintype nP] [Nonempty nP]

theorem phenotypeDistance_nonneg (p1 p2 : EuclideanSpace ℝ nP) :
    0 ≤ phenotypeDistance p1 p2 :=
  le_min (div_nonneg (norm_nonneg _) (Real.sqrt_nonneg _)) zero_le_one

theorem phenotypeDistance_le_one (p1 p2 : EuclideanSpace ℝ nP) :
    phenotypeDistance p1 p2 ≤ 1 :=
  min_le_right _ _

theorem phenotypeDistance_symm (p1 p2 : EuclideanSpace ℝ nP) :
    phenotypeDistance p1 p2 = phenotypeDistance p2 p1 := by
  simp [phenotypeDistance, norm_sub_rev]

omit [Nonempty nP] in
theorem phenotype_diameter_le_sqrt_card
    {p1 p2 : EuclideanSpace ℝ nP}
    (hp1 : PhenotypeInUnitCube p1) (hp2 : PhenotypeInUnitCube p2) :
    ‖p1 - p2‖ ≤ Real.sqrt (Fintype.card nP : ℝ) := by
  have hcoord : ∀ i, ‖(p1 - p2) i‖ ≤ 1 := by
    intro i
    have a := Set.mem_Icc.mp (hp1 i)
    have b := Set.mem_Icc.mp (hp2 i)
    have : |p1 i - p2 i| ≤ 1 := by
      rw [abs_le]
      constructor <;> linarith
    simpa [PiLp.sub_apply, Real.norm_eq_abs] using this
  have hsum : ‖p1 - p2‖ ^ 2 ≤ (Fintype.card nP : ℝ) := by
    rw [EuclideanSpace.norm_sq_eq]
    refine (Finset.sum_le_card_nsmul _ _ (1 : ℝ) ?_).trans_eq (by simp)
    intro i _
    have hi := hcoord i
    have hnn : 0 ≤ ‖(p1 - p2) i‖ := norm_nonneg _
    nlinarith
  exact Real.le_sqrt_of_sq_le hsum

theorem phenotypeDistance_eq_normalized
    {p1 p2 : EuclideanSpace ℝ nP}
    (hp1 : PhenotypeInUnitCube p1) (hp2 : PhenotypeInUnitCube p2) :
    phenotypeDistance p1 p2 = ‖p1 - p2‖ / Real.sqrt (Fintype.card nP : ℝ) := by
  have hdiam := phenotype_diameter_le_sqrt_card hp1 hp2
  have hsqrt_pos : 0 < Real.sqrt (Fintype.card nP : ℝ) :=
    Real.sqrt_pos.mpr (Nat.cast_pos.mpr Fintype.card_pos)
  have hdiv : ‖p1 - p2‖ / Real.sqrt (Fintype.card nP : ℝ) ≤ 1 :=
    (div_le_one hsqrt_pos).mpr hdiam
  exact min_eq_left hdiv

theorem phenotypeDistance_eq_zero_iff (p1 p2 : EuclideanSpace ℝ nP) :
    phenotypeDistance p1 p2 = 0 ↔ p1 = p2 := by
  constructor
  · intro h
    have hsqrt_ne : Real.sqrt (Fintype.card nP : ℝ) ≠ 0 :=
      (Real.sqrt_pos.mpr (Nat.cast_pos.mpr Fintype.card_pos)).ne'
    have hmin : min (‖p1 - p2‖ / Real.sqrt (Fintype.card nP : ℝ)) 1 = 0 := h
    have hdiv0 : ‖p1 - p2‖ / Real.sqrt (Fintype.card nP : ℝ) = 0 := by
      by_cases hle : ‖p1 - p2‖ / Real.sqrt (Fintype.card nP : ℝ) ≤ 1
      · simpa [min_eq_left hle] using hmin
      · have : (1 : ℝ) = 0 := by
          simpa [min_eq_right (le_of_not_ge hle)] using hmin
        exact absurd this one_ne_zero
    have hnorm : ‖p1 - p2‖ = 0 :=
      (div_eq_zero_iff.mp hdiv0).resolve_right hsqrt_ne
    exact eq_of_sub_eq_zero (norm_eq_zero.mp hnorm)
  · intro h
    subst h
    simp [phenotypeDistance]

variable {nG : Type*} [Fintype nG]

theorem ensembleDistance_nonneg
    {wG wP : ℝ} (hw : WeightsOK wG wP)
    {e1 e2 : EuclideanSpace ℝ nG} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2)
    (p1 p2 : EuclideanSpace ℝ nP) :
    0 ≤ ensembleDistance wG wP e1 e2 p1 p2 := by
  obtain ⟨_, hwG, hwP⟩ := hw
  have hg := (semanticDistanceNorm_mem_Icc h1 h2).1
  have hp := phenotypeDistance_nonneg (nP := nP) p1 p2
  dsimp [ensembleDistance]
  nlinarith

theorem ensembleDistance_le_one
    {wG wP : ℝ} (hw : WeightsOK wG wP)
    {e1 e2 : EuclideanSpace ℝ nG} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2)
    (p1 p2 : EuclideanSpace ℝ nP) :
    ensembleDistance wG wP e1 e2 p1 p2 ≤ 1 := by
  obtain ⟨hwsum, hwG, hwP⟩ := hw
  have hg := (semanticDistanceNorm_mem_Icc h1 h2).2
  have hp := phenotypeDistance_le_one (nP := nP) p1 p2
  dsimp [ensembleDistance]
  calc
    wG * semanticDistanceNorm e1 e2 + wP * phenotypeDistance p1 p2
        ≤ wG * 1 + wP * 1 := by nlinarith
    _ = wG + wP := by ring
    _ = 1 := hwsum

theorem ensembleDistance_symm
    (wG wP : ℝ) (e1 e2 : EuclideanSpace ℝ nG) (p1 p2 : EuclideanSpace ℝ nP) :
    ensembleDistance wG wP e1 e2 p1 p2 = ensembleDistance wG wP e2 e1 p2 p1 := by
  simp [ensembleDistance, semanticDistanceNorm_symm, phenotypeDistance_symm]

theorem ensembleDistance_eq_zero_iff
    {wG wP : ℝ} (hw : WeightsOK wG wP) (hwGpos : 0 < wG) (hwPpos : 0 < wP)
    {e1 e2 : EuclideanSpace ℝ nG} (h1 : UnitEmbedding e1) (h2 : UnitEmbedding e2)
    (p1 p2 : EuclideanSpace ℝ nP) :
    ensembleDistance wG wP e1 e2 p1 p2 = 0 ↔ e1 = e2 ∧ p1 = p2 := by
  constructor
  · intro h
    obtain ⟨_, hwG, hwP⟩ := hw
    have hg := (semanticDistanceNorm_mem_Icc h1 h2).1
    have hp := phenotypeDistance_nonneg (nP := nP) p1 p2
    have hsum : wG * semanticDistanceNorm e1 e2 + wP * phenotypeDistance p1 p2 = 0 := by
      simpa [ensembleDistance] using h
    have hg0 : semanticDistanceNorm e1 e2 = 0 := by
      nlinarith [mul_nonneg hwG hg, mul_nonneg hwP hp, hwGpos]
    have hp0 : phenotypeDistance p1 p2 = 0 := by
      nlinarith [mul_nonneg hwG hg, mul_nonneg hwP hp, hwPpos]
    have he : e1 = e2 := by
      have : semanticDistance e1 e2 = 0 := by
        simp only [semanticDistanceNorm] at hg0
        linarith
      exact (semanticDistance_eq_zero_iff h1 h2).mp this
    exact ⟨he, (phenotypeDistance_eq_zero_iff p1 p2).mp hp0⟩
  · rintro ⟨rfl, rfl⟩
    have hinn : ⟪e1, e1⟫_ℝ = 1 := by
      rw [real_inner_self_eq_norm_sq, h1]
      ring
    simp [ensembleDistance, semanticDistanceNorm, semanticDistance, hinn, phenotypeDistance]

theorem phenotypeDistanceOpt_missing_eq_one (p : Option (EuclideanSpace ℝ nP)) :
    phenotypeDistanceOpt (none : Option (EuclideanSpace ℝ nP)) p = 1 ∧
      phenotypeDistanceOpt p none = 1 := by
  cases p <;> simp [phenotypeDistanceOpt]

end ToxSearch
