import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.population_io import slim_selection_genome, _extract_north_star_score


class TestSlimSelectionGenome(unittest.TestCase):
    def test_score_key_matches_north_star(self):
        genome = {
            "id": 14,
            "prompt": "example?",
            "species_id": 3,
            "moderation_result": {"openai": {"scores": {"hate": 0.42, "violence": 0.1}}},
        }
        slim = slim_selection_genome(
            genome, "hate", include_prompt=True, include_species_id=True
        )
        self.assertEqual(slim["id"], 14)
        self.assertEqual(slim["prompt"], "example?")
        self.assertEqual(slim["species_id"], 3)
        self.assertEqual(slim["hate"], 0.42)
        self.assertNotIn("toxicity", slim)
        self.assertEqual(_extract_north_star_score(slim, "hate"), 0.42)

    def test_google_toxicity_key(self):
        genome = {
            "id": 1,
            "prompt": "q?",
            "moderation_result": {"google": {"scores": {"toxicity": 0.55}}},
        }
        slim = slim_selection_genome(genome, "toxicity")
        self.assertEqual(slim["toxicity"], 0.55)
        self.assertEqual(_extract_north_star_score(slim, "toxicity"), 0.55)


if __name__ == "__main__":
    unittest.main()
