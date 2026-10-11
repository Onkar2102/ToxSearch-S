import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speciation.species import SpeciesIdGenerator, generate_species_id, Species, Individual
from speciation.clustering._io import load_species_state, write_species_state


class TestSpeciesIdGenerator(unittest.TestCase):
    def tearDown(self):
        SpeciesIdGenerator.reset(0)

    def test_sequential_from_zero(self):
        SpeciesIdGenerator.reset(0)
        self.assertEqual(generate_species_id(), 1)
        self.assertEqual(generate_species_id(), 2)

    def test_set_min_id_does_not_skip(self):
        # After max existing species id=31, callers pass set_min_id(32).
        SpeciesIdGenerator.reset(0)
        SpeciesIdGenerator.set_min_id(31 + 1)
        self.assertEqual(generate_species_id(), 32)
        self.assertEqual(generate_species_id(), 33)

    def test_set_min_id_respects_higher_cursor(self):
        SpeciesIdGenerator.reset(0)
        self.assertEqual(generate_species_id(), 1)
        # Cursor already past requested min — do not rewind.
        SpeciesIdGenerator.set_min_id(1)
        self.assertEqual(generate_species_id(), 2)

    def test_note_id_prevents_reuse_of_absorbed_high_id(self):
        """After absorbing 47, the next new species ID must be 48, not 47."""
        SpeciesIdGenerator.reset(0)
        for _ in range(47):
            generate_species_id()
        self.assertEqual(SpeciesIdGenerator.peek_next(), 48)
        # 47 absorbed into older survivor; live max may look like 46 afterward.
        SpeciesIdGenerator.note_id(47)
        # Restoring from live-only max must not undercut the high-water mark.
        SpeciesIdGenerator.set_min_id(46 + 1)
        self.assertEqual(generate_species_id(), 48)

    def test_restore_next_from_persisted_counter(self):
        SpeciesIdGenerator.reset(0)
        SpeciesIdGenerator.restore_next(48)
        self.assertEqual(generate_species_id(), 48)
        self.assertEqual(generate_species_id(), 49)

    def test_write_load_roundtrip_preserves_counter_after_merge(self):
        SpeciesIdGenerator.reset(0)
        emb = [1.0, 0.0]
        # Live survivor 12 (merged from 12+47); 47 extinct but must stay past high-water.
        leader = Individual(id=1, prompt="p", fitness=0.5, embedding=emb, species_id=12)
        sp = Species(
            id=12,
            leader=leader,
            members=[leader],
            parent_ids=[12, 47],
            cluster_origin="merge",
        )
        SpeciesIdGenerator.note_id(47)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "speciation_state.json"
            write_species_state(path, {12: sp}, generation=5)
            SpeciesIdGenerator.reset(0)
            loaded = load_species_state(path, theta_sim=0.25)
            self.assertIn(12, loaded)
            self.assertEqual(generate_species_id(), 48)


if __name__ == "__main__":
    unittest.main()
