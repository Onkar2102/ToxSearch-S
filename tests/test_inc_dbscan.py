"""Tests for Incremental DBSCAN seed/insert and parent eligibility (noise kept)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from speciation.clustering._io import SINGLETON_SPECIES_ID
from speciation.clustering.inc_dbscan import (
    IncDBSCANState,
    _build_species_from_state,
    insert_point,
    process_unlabeled,
    seed_from_incremental,
    seed_from_precomputed_matrix,
)
from speciation.density_memory import NOISE_LABEL, is_labeled, set_density_label
from speciation.species import Individual, Species, SpeciesIdGenerator
from utils.evaluator_profiles import set_active_evaluator, set_active_north_star


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _genome(gid, emb, fitness=0.5, prompt=None):
    # Full Google phenotype vector required by extract_phenotype_vector (fail-fast).
    google_scores = {
        "toxicity": fitness,
        "threat": max(0.0001, fitness * 0.5),
        "profanity": max(0.0001, fitness * 0.4),
        "sexually_explicit": max(0.0001, fitness * 0.3),
        "identity_attack": max(0.0001, fitness * 0.2),
        "flirtation": max(0.0001, fitness * 0.1),
        "insult": max(0.0001, fitness * 0.45),
        "severe_toxicity": max(0.0001, fitness * 0.25),
    }
    return {
        "id": gid,
        "prompt": prompt or f"prompt-{gid}",
        "prompt_embedding": emb.tolist(),
        "fitness": fitness,
        "toxicity": fitness,
        "generation": 0,
        "status": "complete",
        "moderation_result": {"google": {"scores": google_scores}},
    }


class TestIncDBSCANSeedAndNoise(unittest.TestCase):
    def setUp(self):
        SpeciesIdGenerator.reset(0)
        # Isolate from other tests that leave openai/violence as active north-star.
        set_active_evaluator("google")
        set_active_north_star("toxicity")

    def test_seed_promotes_noise_to_singleton_pool(self):
        from speciation.clustering._io import SINGLETON_SPECIES_ID

        a, b = _unit([1, 0, 0]), _unit([0.99, 0.1, 0])
        c, d = _unit([0, 1, 0]), _unit([0.1, 0.99, 0])
        noise = _unit([0, 0, 1])
        genomes = [
            _genome(1, a, 0.9),
            _genome(2, b, 0.8),
            _genome(3, c, 0.7),
            _genome(4, d, 0.6),
            _genome(5, noise, 0.1),
        ]
        state, species, ev = seed_from_incremental(
            genomes, eps=0.05, min_samples=2, current_generation=0, distance_method="embedding"
        )
        self.assertEqual(ev["event"], "inc_dbscan_seed")
        self.assertEqual(ev["noise"], 0)
        self.assertEqual(ev["noise_before_singleton_promotion"], 1)
        self.assertEqual(ev["singletons"], 1)
        # 2 mature multi-member clusters; noise shares pool 0 (not a Species entry)
        self.assertEqual(len(species), 2)
        self.assertTrue(all(sid > 0 for sid in species))
        noise_g = next(g for g in genomes if g["id"] == 5)
        self.assertTrue(is_labeled(noise_g))
        self.assertEqual(int(noise_g["density_label"]), SINGLETON_SPECIES_ID)
        self.assertEqual(noise_g["species_id"], SINGLETON_SPECIES_ID)
        self.assertNotIn(SINGLETON_SPECIES_ID, species)
        self.assertNotIn(NOISE_LABEL, state.labels)
        # Mature IDs start at 1; pool 0 does not consume the ID counter
        self.assertEqual(sorted(species.keys()), [1, 2])
        self.assertEqual(SpeciesIdGenerator.peek_next(), 3)

    def test_speciation_state_noise_lists_non_mature_ids(self):
        """speciation_state.noise lists pool-0 ids; noise_count matches that list."""
        import json
        from speciation.clustering.inc_dbscan import write_species_and_config

        a, b = _unit([1, 0, 0]), _unit([0.99, 0.1, 0])
        c, d = _unit([0, 1, 0]), _unit([0.1, 0.99, 0])
        noise = _unit([0, 0, 1])
        genomes = [
            _genome(1, a, 0.9),
            _genome(2, b, 0.8),
            _genome(3, c, 0.7),
            _genome(4, d, 0.6),
            _genome(5, noise, 0.1),
        ]
        state, species, _ = seed_from_incremental(
            genomes, eps=0.05, min_samples=2, current_generation=0, distance_method="embedding"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "speciation_state.json"
            write_species_and_config(path, species, 0, state)
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            noise_block = payload["noise"]
            self.assertEqual(noise_block["species_id"], SINGLETON_SPECIES_ID)
            self.assertEqual(noise_block["genome_ids"], ["5"])
            self.assertEqual(noise_block["count"], 1)
            self.assertEqual(payload["noise_count"], 1)
            self.assertEqual(payload["singleton_species_count"], 1)
            self.assertEqual(payload.get("residual_dbscan_noise_count", 0), 0)
            self.assertNotIn("0", payload["species"])
            self.assertNotIn("-1", payload["species"])

    def test_sequential_insert_far_point_stays_non_mature(self):
        """Seed with two clusters, then insert a far point into the singleton pool."""
        vecs = [
            _unit([1, 0, 0]),
            _unit([0.98, 0.1, 0]),
            _unit([0, 1, 0]),
            _unit([0.1, 0.98, 0]),
            _unit([0, 0, 1]),
        ]
        genomes = [_genome(i + 1, vecs[i], fitness=1.0 - 0.1 * i) for i in range(4)]
        state, species, _ = seed_from_incremental(
            genomes, eps=0.08, min_samples=2, distance_method="embedding"
        )
        self.assertEqual(len(species), 2)
        g5 = _genome(5, vecs[4], fitness=0.05)
        genomes.append(g5)
        ind = Individual.from_genome(g5)
        ev = insert_point(state, g5, ind, current_generation=1, species=species)
        self.assertIn(ev.get("event"), ("inc_dbscan_insert_noise", "inc_dbscan_singleton_pool"))
        self.assertTrue(int(state.labels[-1]) <= 0)

    def test_process_unlabeled_absorb(self):
        a, b = _unit([1, 0]), _unit([0.99, 0.05])
        genomes = [_genome(1, a, 0.9), _genome(2, b, 0.8)]
        state, species, _ = seed_from_incremental(
            genomes, eps=0.1, min_samples=2, distance_method="embedding"
        )
        # Near cluster member
        g3 = _genome(3, _unit([0.97, 0.1]), fitness=0.95)
        genomes.append(g3)
        evs = process_unlabeled(state, genomes, current_generation=1, species=species)
        self.assertTrue(any(e.get("event") in ("inc_dbscan_absorb", "inc_dbscan_new_species") for e in evs))
        self.assertTrue(is_labeled(g3))
        self.assertNotEqual(g3["density_label"], NOISE_LABEL)

    def test_merge_keeps_oldest_parent_id(self):
        """Bridge insert merges two clusters onto min(parent_ids), not a new ID."""
        SpeciesIdGenerator.reset(0)
        vecs = [
            _unit([1, 0, 0]),
            _unit([0.98, 0.2, 0]),
            _unit([0, 1, 0]),
            _unit([0.2, 0.98, 0]),
            _unit([0.75, 0.75, 0]),  # bridge between the two clusters
        ]
        genomes = [_genome(i + 1, vecs[i], fitness=1.0 - 0.1 * i) for i in range(4)]
        state, species, _ = seed_from_incremental(
            genomes, eps=0.12, min_samples=2, distance_method="embedding"
        )
        self.assertEqual(len(species), 2)
        parent_ids_before = sorted(species.keys())
        self.assertEqual(parent_ids_before, [1, 2])

        g5 = _genome(5, vecs[4], fitness=0.5)
        genomes.append(g5)
        ind = Individual.from_genome(g5)
        ev = insert_point(state, g5, ind, current_generation=1, species=species)
        self.assertEqual(ev.get("event"), "inc_dbscan_merge")
        self.assertEqual(ev.get("species_id"), 1)  # oldest / smallest parent
        self.assertEqual(sorted(ev.get("parent_ids", [])), [1, 2])
        self.assertEqual(ev.get("absorbed_ids"), [2])
        self.assertIn(1, species)
        self.assertNotIn(2, species)
        # All previously clustered points + bridge share survivor label 1
        self.assertTrue(all(int(lab) == 1 for lab in state.labels))
        self.assertEqual(species[1].parent_ids, [1, 2])
        self.assertEqual(species[1].cluster_origin, "inc_dbscan_merge")


class TestLeaderFollowerMergeKeepsOldest(unittest.TestCase):
    def setUp(self):
        set_active_evaluator("google")
        set_active_north_star("toxicity")
        SpeciesIdGenerator.reset(0)

    def test_merge_islands_survivor_is_min_id(self):
        from speciation.merging import merge_islands
        from speciation.species import Species

        SpeciesIdGenerator.reset(0)
        emb = _unit([1, 0])
        m1 = Individual(id=10, prompt="a", fitness=0.9, embedding=emb, species_id=1)
        m2 = Individual(id=11, prompt="b", fitness=0.8, embedding=emb, species_id=1)
        m3 = Individual(id=20, prompt="c", fitness=0.7, embedding=_unit([0.99, 0.1]), species_id=2)
        m4 = Individual(id=21, prompt="d", fitness=0.6, embedding=_unit([0.98, 0.2]), species_id=2)
        sp1 = Species(id=1, leader=m1, members=[m1, m2], created_at=0, cluster_origin="natural")
        sp2 = Species(id=2, leader=m3, members=[m3, m4], created_at=3, cluster_origin="natural")
        merged, _ = merge_islands(sp1, sp2, current_generation=5)
        self.assertEqual(merged.id, 1)
        self.assertEqual(merged.parent_ids, [1, 2])
        self.assertEqual(merged.created_at, 0)  # preserved from older parent
        self.assertTrue(all(m.species_id == 1 for m in merged.members))


class TestIncDBSCANParentSelector(unittest.TestCase):
    def test_singleton_pool_is_parent_eligible(self):
        from ea.inc_dbscan_parent_selector import IncDBSCANParentSelector
        from speciation.clustering._io import SINGLETON_SPECIES_ID

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            genomes = [
                _genome(1, _unit([1, 0]), 0.9),
                _genome(2, _unit([0.99, 0.1]), 0.8),
                _genome(3, _unit([0, 1]), 0.1),
            ]
            set_density_label(genomes[0], 1)
            set_density_label(genomes[1], 1)
            set_density_label(genomes[2], SINGLETON_SPECIES_ID)
            import json
            with open(td / "temp.json", "w") as f:
                json.dump(genomes, f)
            sel = IncDBSCANParentSelector("toxicity")
            sel.adaptive_tournament_selection(outputs_path=str(td))
            with open(td / "parents.json") as f:
                parents = json.load(f)
            self.assertGreaterEqual(len(parents), 2)


class TestIncDBSCANPrecomputedMatrix(unittest.TestCase):
    def setUp(self):
        SpeciesIdGenerator.reset(0)

    def test_precomputed_seed_deterministic_and_aligned(self):
        # Two tight pairs far apart + one isolated point
        rng = np.random.default_rng(0)
        pts = np.array([
            [0.0, 0.0],
            [0.01, 0.0],
            [1.0, 0.0],
            [1.01, 0.0],
            [5.0, 5.0],
        ], dtype=np.float64)
        n = len(pts)
        D = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                D[i, j] = D[j, i] = d
        order = np.arange(n)  # index order
        a = seed_from_precomputed_matrix(D, order, eps=0.05, min_samples=2)
        b = seed_from_precomputed_matrix(D, order, eps=0.05, min_samples=2)
        self.assertEqual(a.shape, (n,))
        np.testing.assert_array_equal(a, b)
        # Isolated point must be non-mature (-1)
        self.assertEqual(int(a[4]), -1)
        # Each tight pair should share a mature label (or both noise if min_samples blocks)
        # With min_samples=2 and eps covering pair distance 0.01, pairs should mature
        self.assertNotEqual(int(a[0]), -1)
        self.assertEqual(int(a[0]), int(a[1]))
        self.assertNotEqual(int(a[2]), -1)
        self.assertEqual(int(a[2]), int(a[3]))
        self.assertNotEqual(int(a[0]), int(a[2]))
        # Fitness-desc order permutation still deterministic
        order2 = np.array([4, 3, 2, 1, 0])
        c = seed_from_precomputed_matrix(D, order2, eps=0.05, min_samples=2)
        d = seed_from_precomputed_matrix(D, order2, eps=0.05, min_samples=2)
        np.testing.assert_array_equal(c, d)
        del rng


class TestRebuildStagnation(unittest.TestCase):
    """Tied champion must not wipe species.stagnation (RQ2 Inc freeze bug)."""

    def _state(self, genome_ids, labels):
        return IncDBSCANState(eps=0.1, min_samples=2, genome_ids=list(genome_ids), labels=list(labels))

    def test_tied_champ_preserves_stagnation(self):
        champ = Individual(id=1, prompt="q?", fitness=0.9, species_id=1)
        other = Individual(id=2, prompt="r?", fitness=0.4, species_id=1)
        prev = Species(
            id=1, leader=champ, members=[champ, other], radius=0.1,
            stagnation=6, max_fitness=0.9, species_state="active",
            created_at=1, last_improvement=3,
        )
        rebuilt = _build_species_from_state(
            self._state([1, 2], [1, 1]),
            [{"id": 1}, {"id": 2}],
            [champ, other],
            current_generation=20,
            eps=0.1,
            origin="inc_dbscan_growth",
            existing={1: prev},
        )
        sp = rebuilt[1]
        self.assertEqual(sp.stagnation, 6)
        self.assertEqual(sp.last_improvement, 3)
        self.assertAlmostEqual(sp.max_fitness, 0.9)
        self.assertEqual(sp.species_state, "active")

    def test_true_gain_resets_stagnation(self):
        old = Individual(id=1, prompt="q?", fitness=0.9, species_id=1)
        better = Individual(id=3, prompt="s?", fitness=0.95, species_id=1)
        prev = Species(
            id=1, leader=old, members=[old], radius=0.1,
            stagnation=6, max_fitness=0.9, species_state="active",
            created_at=1, last_improvement=3,
        )
        rebuilt = _build_species_from_state(
            self._state([1, 3], [1, 1]),
            [{"id": 1}, {"id": 3}],
            [old, better],
            current_generation=20,
            eps=0.1,
            origin="inc_dbscan_growth",
            existing={1: prev},
        )
        sp = rebuilt[1]
        self.assertEqual(sp.stagnation, 0)
        self.assertEqual(sp.last_improvement, 20)
        self.assertAlmostEqual(sp.max_fitness, 0.95)


if __name__ == "__main__":
    unittest.main()
