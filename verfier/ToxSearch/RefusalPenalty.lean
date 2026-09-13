import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Refusal penalty algebra

Mirrors `PENALTY_MULTIPLIER = 0.85` in `src/utils/refusal_penalty.py`.
-/

namespace ToxSearch

def penaltyMultiplier : ℝ := 0.85

def applyRefusalPenalty (score : ℝ) : ℝ :=
  penaltyMultiplier * score

theorem applyRefusalPenalty_eq (score : ℝ) :
    applyRefusalPenalty score = (85 / 100 : ℝ) * score := by
  simp [applyRefusalPenalty, penaltyMultiplier]
  norm_num

theorem applyRefusalPenalty_nonneg {score : ℝ} (h : 0 ≤ score) :
    0 ≤ applyRefusalPenalty score :=
  mul_nonneg (by norm_num [penaltyMultiplier]) h

theorem applyRefusalPenalty_le_self {score : ℝ} (h : 0 ≤ score) :
    applyRefusalPenalty score ≤ score := by
  have : penaltyMultiplier ≤ 1 := by norm_num [penaltyMultiplier]
  simpa [applyRefusalPenalty] using mul_le_of_le_one_left h this

theorem applyRefusalPenalty_mono {a b : ℝ} (h : a ≤ b) :
    applyRefusalPenalty a ≤ applyRefusalPenalty b :=
  mul_le_mul_of_nonneg_left h (by norm_num [penaltyMultiplier])

end ToxSearch
