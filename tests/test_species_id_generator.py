import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from speciation.species import SpeciesIdGenerator, generate_species_id


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


if __name__ == "__main__":
    unittest.main()
