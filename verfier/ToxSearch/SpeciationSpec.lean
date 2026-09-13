import ToxSearch.Config
import ToxSearch.Distance
import Mathlib.Data.Real.Basic
import Mathlib.Tactic

/-!
# Abstract speciation state invariants

Idealized specification matching checks in `src/speciation/validation.py` and
leader rules in `src/speciation/species.py` / `leader_follower.py`.
-/

namespace ToxSearch.Speciation

/-- A genome in the abstract model. -/
structure Genome where
  id : ℕ
  fitness : ℝ
  speciesId : ℕ

/-- A species: leader must be a member; id `0` is reserved for Cluster 0. -/
structure Species where
  id : ℕ
  members : List Genome
  leader : Genome
  radius : ℝ

/-- Species invariants used by ToxSearch-S. -/
def Species.WellFormed (s : Species) (thetaSim : ℝ) : Prop :=
  s.id ≠ 0 ∧
  s.leader ∈ s.members ∧
  s.leader.speciesId = s.id ∧
  (∀ g ∈ s.members, g.speciesId = s.id) ∧
  (∀ g ∈ s.members, g.fitness ≤ s.leader.fitness) ∧
  s.radius = thetaSim

theorem leader_has_max_fitness {s : Species} {thetaSim : ℝ}
    (h : s.WellFormed thetaSim) {g : Genome} (hg : g ∈ s.members) :
    g.fitness ≤ s.leader.fitness :=
  h.2.2.2.2.1 g hg

theorem leader_is_member {s : Species} {thetaSim : ℝ}
    (h : s.WellFormed thetaSim) : s.leader ∈ s.members :=
  h.2.1

theorem species_id_nonzero {s : Species} {thetaSim : ℝ}
    (h : s.WellFormed thetaSim) : s.id ≠ 0 :=
  h.1

/-- Global population partition. -/
structure PopulationState where
  elites : List Genome
  reserves : List Genome
  archive : List Genome
  species : List Species
  thetaSim : ℝ

def PopulationState.allGenomes (st : PopulationState) : List Genome :=
  st.elites ++ st.reserves ++ st.archive

/-- Consistency invariants (unique IDs; Cluster 0 = reserves; elite species tracked). -/
def PopulationState.WellFormed (st : PopulationState) : Prop :=
  (st.allGenomes.map (·.id)).Nodup ∧
  (∀ g ∈ st.reserves, g.speciesId = 0) ∧
  (∀ g ∈ st.elites, 0 < g.speciesId) ∧
  (∀ g ∈ st.elites, ∃ s ∈ st.species, s.id = g.speciesId ∧ s.WellFormed st.thetaSim) ∧
  (st.species.map (·.id)).Nodup ∧
  (∀ s ∈ st.species, s.id ≠ 0) ∧
  (∀ s ∈ st.species, ∀ g ∈ s.members, g ∈ st.elites)

theorem wellFormed_unique_ids {st : PopulationState} (h : st.WellFormed) :
    (st.allGenomes.map (·.id)).Nodup :=
  h.1

theorem wellFormed_reserves_cluster0 {st : PopulationState} (h : st.WellFormed)
    {g : Genome} (hg : g ∈ st.reserves) : g.speciesId = 0 :=
  h.2.1 g hg

theorem wellFormed_elites_positive_species {st : PopulationState} (h : st.WellFormed)
    {g : Genome} (hg : g ∈ st.elites) : 0 < g.speciesId :=
  h.2.2.1 g hg

/-- Membership predicate used by leader–follower assignment. -/
def WithinRadius (d thetaSim : ℝ) : Prop :=
  d < thetaSim

theorem withinRadius_of_merge
    {c : SpeciationConfig} (hw : c.WellFormed) {d : ℝ}
    (h : d < c.thetaMerge) : WithinRadius d c.thetaSim :=
  merge_implies_membership_radius hw h

end ToxSearch.Speciation
