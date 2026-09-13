import Mathlib.Data.Real.Basic
import Mathlib.Data.List.Basic
import Mathlib.Tactic

/-!
# NSGA-II dominance and selection properties

Mirrors `_dominates` and capacity truncation in
`src/speciation/reserve_selection.py` (maximize diversity and toxicity).
-/

namespace ToxSearch.NSGA2

/-- Pareto dominance for two maximization objectives (diversity, toxicity). -/
def Dominates (aDiv aTox bDiv bTox : ℝ) : Prop :=
  (aDiv ≥ bDiv ∧ aTox ≥ bTox) ∧ (aDiv > bDiv ∨ aTox > bTox)

theorem dominates_irrefl (d t : ℝ) : ¬ Dominates d t d t := by
  rintro ⟨_, hstrict⟩
  cases hstrict with
  | inl h => exact lt_irrefl _ h
  | inr h => exact lt_irrefl _ h

theorem dominates_asymm {aDiv aTox bDiv bTox : ℝ}
    (h : Dominates aDiv aTox bDiv bTox) : ¬ Dominates bDiv bTox aDiv aTox := by
  rintro ⟨⟨hle1', hle2'⟩, _⟩
  rcases h with ⟨⟨hle1, hle2⟩, hs⟩
  have eq1 : aDiv = bDiv := le_antisymm hle1' hle1
  have eq2 : aTox = bTox := le_antisymm hle2' hle2
  subst eq1; subst eq2
  cases hs with
  | inl hlt => exact lt_irrefl _ hlt
  | inr hlt => exact lt_irrefl _ hlt

/-- Pareto dominance is transitive (strict partial order on objective pairs). -/
theorem dominates_trans {aDiv aTox bDiv bTox cDiv cTox : ℝ}
    (h₁ : Dominates aDiv aTox bDiv bTox) (h₂ : Dominates bDiv bTox cDiv cTox) :
    Dominates aDiv aTox cDiv cTox := by
  rcases h₁ with ⟨⟨hle_ab_d, hle_ab_t⟩, hs_ab⟩
  rcases h₂ with ⟨⟨hle_bc_d, hle_bc_t⟩, hs_bc⟩
  refine ⟨⟨le_trans hle_bc_d hle_ab_d, le_trans hle_bc_t hle_ab_t⟩, ?_⟩
  cases hs_ab with
  | inl h => exact Or.inl (lt_of_le_of_lt hle_bc_d h)
  | inr h =>
    cases hs_bc with
    | inl h' => exact Or.inl (lt_of_lt_of_le h' hle_ab_d)
    | inr h' => exact Or.inr (lt_of_lt_of_le h' hle_ab_t)

/-- Objective pair. -/
structure Point where
  div : ℝ
  tox : ℝ

def DominatesPoint (a b : Point) : Prop :=
  Dominates a.div a.tox b.div b.tox

/-- Index `i` is non-dominated in a finite list. -/
def IsNonDominated (pts : List Point) (i : ℕ) : Prop :=
  ∃ hi : i < pts.length,
    ∀ j (hj : j < pts.length), j ≠ i →
      ¬ DominatesPoint (pts.get ⟨j, hj⟩) (pts.get ⟨i, hi⟩)

theorem isNonDominated_lt {pts : List Point} {i : ℕ}
    (h : IsNonDominated pts i) : i < pts.length := by
  obtain ⟨hi, _⟩ := h
  exact hi

/-- Functional selection: greedily take whole fronts, then a prefix of the next.
Matches the capacity logic of `select_reserves_nsga2`. -/
def selectByFronts (fronts : List (List ℕ)) (capacity : ℕ) : List ℕ :=
  match fronts with
  | [] => []
  | f :: rest =>
    if f.length ≤ capacity then
      f ++ selectByFronts rest (capacity - f.length)
    else
      f.take capacity

theorem selectByFronts_length_le (fronts : List (List ℕ)) (capacity : ℕ) :
    (selectByFronts fronts capacity).length ≤ capacity := by
  induction fronts generalizing capacity with
  | nil => simp [selectByFronts]
  | cons f rest ih =>
    simp only [selectByFronts]
    split_ifs with h
    · have hrest := ih (capacity - f.length)
      have : f.length + (selectByFronts rest (capacity - f.length)).length ≤ capacity := by
        omega
      simpa [List.length_append] using this
    · simpa using (List.length_take_le (n := capacity) (l := f))

theorem selectByFronts_empty (capacity : ℕ) :
    selectByFronts ([] : List (List ℕ)) capacity = [] := rfl

/-- If total population size is already within capacity, selection keeps everyone
when presented as a single front of all indices. -/
theorem selectByFronts_keeps_all_when_small
    (indices : List ℕ) {capacity : ℕ} (h : indices.length ≤ capacity) :
    selectByFronts [indices] capacity = indices := by
  simp [selectByFronts, h]

end ToxSearch.NSGA2
