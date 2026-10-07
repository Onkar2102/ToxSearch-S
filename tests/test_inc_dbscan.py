"""Tests for Incremental DBSCAN seed/insert and parent eligibility (noise kept)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from speciation.clustering.dbscan import dbscan_precomputed
from speciation.clustering.inc_dbscan import (
    insert_point,
    seed_from_batch,
    process_unlabeled,
    validate_against_batch,
    IncDBSCANState,
)
from speciation.density_memory import NOISE_LABEL, is_labeled, set_density_label
from speciation.distance import pairwise_embedding_matrix
from speciation.species import Individual, SpeciesIdGenerator


def _unit(v):
    a = np.asarray(v, dtype=np.float64)
    return a / np.linalg.norm(a)


def _genome(gid, emb, fitness=0.5, prompt=None):
    return {
        "id": gid,
        "prompt": prompt or f"prompt-{gid}",
        "prompt_embedding": emb.tolist(),
        "fitness": fitness,
        "toxicity": fitness,
        "generation": 0,
        "status": "complete",
    }


class TestIncDBSCANSeedAndNoise(unittest.TestCase):
    def setUp(self):
        SpeciesIdGenerator.reset(0)

    def test_seed_keeps_noise_as_minus_one(self):
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
        state, species, ev = seed_from_batch(
            genomes, eps=0.05, min_samples=2, current_generation=0, distance_method="embedding"
        )
        self.assertEqual(ev["noise"], 1)
        self.assertEqual(len(species), 2)
        noise_g = next(g for g in genomes if g["id"] == 5)
        self.assertTrue(is_labeled(noise_g))
        self.assertEqual(noise_g["density_label"], NOISE_LABEL)
        self.assertEqual(noise_g["species_id"], NOISE_LABEL)
        # Noise not a Species entry
        self.assertNotIn(NOISE_LABEL, species)

    def test_sequential_inserts_match_batch_partition(self):
        """Insert one-by-one then compare partition to batch DBSCAN (label permutation)."""
        vecs = [
            _unit([1, 0, 0]),
            _unit([0.98, 0.1, 0]),
            _unit([0, 1, 0]),
            _unit([0.1, 0.98, 0]),
            _unit([0, 0, 1]),
        ]
        # Seed with first 4, insert 5th as new far point (noise)
        genomes = [_genome(i + 1, vecs[i], fitness=1.0 - 0.1 * i) for i in range(4)]
        state, species, _ = seed_from_batch(
            genomes, eps=0.08, min_samples=2, distance_method="embedding"
        )
        g5 = _genome(5, vecs[4], fitness=0.05)
        genomes.append(g5)
        ind = Individual.from_genome(g5)
        insert_point(state, g5, ind, current_generation=1, species=species)
        self.assertTrue(validate_against_batch(state))

    def test_process_unlabeled_absorb(self):
        a, b = _unit([1, 0]), _unit([0.99, 0.05])
        genomes = [_genome(1, a, 0.9), _genome(2, b, 0.8)]
        state, species, _ = seed_from_batch(
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
        state, species, _ = seed_from_batch(
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
    def test_noise_is_parent_eligible(self):
        from ea.inc_dbscan_parent_selector import IncDBSCANParentSelector

        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            genomes = [
                _genome(1, _unit([1, 0]), 0.9),
                _genome(2, _unit([0.99, 0.1]), 0.8),
                _genome(3, _unit([0, 1]), 0.1),
            ]
            set_density_label(genomes[0], 1)
            set_density_label(genomes[1], 1)
            set_density_label(genomes[2], NOISE_LABEL)
            import json
            with open(td / "temp.json", "w") as f:
                json.dump(genomes, f)
            sel = IncDBSCANParentSelector("toxicity")
            sel.adaptive_tournament_selection(outputs_path=str(td))
            with open(td / "parents.json") as f:
                parents = json.load(f)
            self.assertGreaterEqual(len(parents), 2)


class TestDBSCANPrecomputedStillWorks(unittest.TestCase):
    def test_precomputed(self):
        emb = np.stack([_unit([1, 0]), _unit([0.99, 0.1]), _unit([0, 1])])
        dist = pairwise_embedding_matrix(emb)
        labels = dbscan_precomputed(dist, eps=0.05, min_samples=2)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(int(labels[2]), -1)


if __name__ == "__main__":
    unittest.main()
