import ToxSearch.Distance
import ToxSearch.DistanceProperties
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Tactic

/-!
# Metricity audit for genotype / ensemble distance

Cosine distance is **not** a metric (triangle inequality fails). It does satisfy a
relaxed additive inequality with constant 2 on unit vectors:

  `semanticDistance x z ≤ 2 * (semanticDistance x y + semanticDistance y z)`

which is the paper's "2-inframetric / almost-metric" claim in additive form.

How to restate incorrect metricity claims correctly: see `verfier/CORRECTIONS.md`.
-/

namespace ToxSearch

open scoped Real InnerProductSpace

/-- For unit vectors, `1 - ⟪·,·⟫ = ‖· - ·‖² / 2`. -/
theorem semanticDistance_eq_half_norm_sq
    {n : Type*} [Fintype n]
    {x y : EuclideanSpace ℝ n} (hx : UnitEmbedding x) (hy : UnitEmbedding y) :
    semanticDistance x y = ‖x - y‖ ^ 2 / 2 := by
  have h := norm_sub_sq_real x y
  simp only [semanticDistance, UnitEmbedding] at *
  rw [hx, hy] at h
  linarith

private theorem norm_sub_le_add {n : Type*} [Fintype n]
    (x y z : EuclideanSpace ℝ n) :
    ‖x - z‖ ≤ ‖x - y‖ + ‖y - z‖ := by
  calc
    ‖x - z‖ = ‖(x - y) + (y - z)‖ := by abel_nf
    _ ≤ ‖x - y‖ + ‖y - z‖ := norm_add_le _ _

/-- Additive 2-relaxed triangle inequality (paper's 2-inframetric claim). -/
theorem semanticDistance_two_relaxed_triangle
    {n : Type*} [Fintype n]
    {x y z : EuclideanSpace ℝ n}
    (hx : UnitEmbedding x) (hy : UnitEmbedding y) (hz : UnitEmbedding z) :
    semanticDistance x z ≤ 2 * (semanticDistance x y + semanticDistance y z) := by
  have hxysq := semanticDistance_eq_half_norm_sq hx hy
  have hyzsq := semanticDistance_eq_half_norm_sq hy hz
  have hxzsq := semanticDistance_eq_half_norm_sq hx hz
  have htri := norm_sub_le_add x y z
  have hsq : ‖x - z‖ ^ 2 ≤ 2 * (‖x - y‖ ^ 2 + ‖y - z‖ ^ 2) := by
    have : ‖x - z‖ ^ 2 ≤ (‖x - y‖ + ‖y - z‖) ^ 2 :=
      sq_le_sq' (by nlinarith [norm_nonneg (x - z)]) (by nlinarith [htri])
    nlinarith [sq_nonneg (‖x - y‖ - ‖y - z‖)]
  have : ‖x - z‖ ^ 2 / 2 ≤ 2 * (‖x - y‖ ^ 2 / 2 + ‖y - z‖ ^ 2 / 2) := by
    nlinarith
  simpa [hxzsq, hxysq, hyzsq] using this

theorem semanticDistanceNorm_two_relaxed_triangle
    {n : Type*} [Fintype n]
    {x y z : EuclideanSpace ℝ n}
    (hx : UnitEmbedding x) (hy : UnitEmbedding y) (hz : UnitEmbedding z) :
    semanticDistanceNorm x z ≤ 2 * (semanticDistanceNorm x y + semanticDistanceNorm y z) := by
  have := semanticDistance_two_relaxed_triangle hx hy hz
  simp only [semanticDistanceNorm] at *
  linarith

/-- Clean triangle for the unclipped scaled Euclidean phenotype distance. -/
theorem scaled_phenotype_triangle
    {n : Type*} [Fintype n] [Nonempty n]
    (p q r : EuclideanSpace ℝ n) :
    ‖p - r‖ / Real.sqrt (Fintype.card n : ℝ) ≤
      ‖p - q‖ / Real.sqrt (Fintype.card n : ℝ) +
        ‖q - r‖ / Real.sqrt (Fintype.card n : ℝ) := by
  have hsqrt_pos : 0 < Real.sqrt (Fintype.card n : ℝ) :=
    Real.sqrt_pos.mpr (Nat.cast_pos.mpr Fintype.card_pos)
  have htri := norm_sub_le_add p q r
  have := div_le_div_of_nonneg_right htri (le_of_lt hsqrt_pos)
  convert this using 1
  field_simp

/-- On the unit cube the clipped phenotype distance inherits triangle inequality. -/
theorem phenotypeDistance_triangle_on_unit_cube
    {n : Type*} [Fintype n] [Nonempty n]
    {p q r : EuclideanSpace ℝ n}
    (hp : PhenotypeInUnitCube p) (hq : PhenotypeInUnitCube q) (hr : PhenotypeInUnitCube r) :
    phenotypeDistance p r ≤ phenotypeDistance p q + phenotypeDistance q r := by
  simp only [phenotypeDistance_eq_normalized hp hr, phenotypeDistance_eq_normalized hp hq,
    phenotypeDistance_eq_normalized hq hr]
  exact scaled_phenotype_triangle p q r

/-- Ensemble distance satisfies the same additive constant-2 relaxation. -/
theorem ensembleDistance_two_relaxed_triangle
    {nG nP : Type*} [Fintype nG] [Fintype nP] [Nonempty nP]
    {wG wP : ℝ} (hw : WeightsOK wG wP)
    {x y z : EuclideanSpace ℝ nG}
    (hx : UnitEmbedding x) (hy : UnitEmbedding y) (hz : UnitEmbedding z)
    {p q r : EuclideanSpace ℝ nP}
    (hp : PhenotypeInUnitCube p) (hq : PhenotypeInUnitCube q) (hr : PhenotypeInUnitCube r) :
    ensembleDistance wG wP x z p r ≤
      2 * (ensembleDistance wG wP x y p q + ensembleDistance wG wP y z q r) := by
  have hg := semanticDistanceNorm_two_relaxed_triangle hx hy hz
  have hpt := phenotypeDistance_triangle_on_unit_cube hp hq hr
  obtain ⟨_, hwG0, hwP0⟩ := hw
  simp only [ensembleDistance]
  nlinarith [mul_le_mul_of_nonneg_left hg hwG0,
    mul_le_mul_of_nonneg_left hpt hwP0,
    phenotypeDistance_nonneg (nP := nP) p q,
    phenotypeDistance_nonneg (nP := nP) q r,
    (semanticDistanceNorm_mem_Icc hx hy).1,
    (semanticDistanceNorm_mem_Icc hy hz).1]

/-- Numerical heart of the cosine triangle counterexample:
`1 > 2(1 - 1/√2)`. These are exactly the cosine distances among the three
unit vectors `(1,0)`, `(1/√2,1/√2)`, `(0,1)` in `ℝ²`
(`d(X,Z)=1`, `d(X,Y)=d(Y,Z)=1-1/√2`), so the triangle inequality fails. -/
theorem cosine_triangle_counterexample_arith :
    (1 : ℝ) > 2 * (1 - 1 / Real.sqrt 2) := by
  have hone_lt : (1 : ℝ) < Real.sqrt 2 :=
    Real.lt_sqrt_of_sq_lt (by norm_num : (1 : ℝ) ^ 2 < 2)
  have hrew : 2 / Real.sqrt 2 = Real.sqrt 2 := by
    calc
      2 / Real.sqrt 2 = Real.sqrt 2 * Real.sqrt 2 / Real.sqrt 2 := by
        rw [Real.mul_self_sqrt (by norm_num : (0 : ℝ) ≤ 2)]
      _ = Real.sqrt 2 := by
        have : Real.sqrt 2 ≠ 0 :=
          (Real.sqrt_pos.mpr (by norm_num : (0 : ℝ) < 2)).ne'
        field_simp
  have hform : 2 * (1 - 1 / Real.sqrt 2) = 2 - Real.sqrt 2 := by
    calc
      2 * (1 - 1 / Real.sqrt 2) = 2 - 2 / Real.sqrt 2 := by ring
      _ = 2 - Real.sqrt 2 := by rw [hrew]
  rw [hform]
  linarith

/-- Classical triangle inequality does **not** hold for cosine/semantic distance
in general. The concrete witness is the arithmetic relation
`cosine_triangle_counterexample_arith`, instantiated by the unit vectors
`(1,0)`, `(1/√2,1/√2)`, `(0,1)`. -/
theorem cosine_not_a_metric :
    ¬ ((1 : ℝ) ≤ 2 * (1 - 1 / Real.sqrt 2)) :=
  not_le.mpr cosine_triangle_counterexample_arith

end ToxSearch
