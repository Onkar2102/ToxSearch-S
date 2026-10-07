import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gne.evaluator import HybridModerationEvaluator


class TestNormalizeScores(unittest.TestCase):
    def setUp(self):
        # Bypass heavy __init__; only exercise _normalize_scores.
        self.ev = object.__new__(HybridModerationEvaluator)

    def test_exact_zero_becomes_floor(self):
        out = self.ev._normalize_scores({"toxicity": 0.0, "threat": 0.000000})
        self.assertEqual(out["toxicity"], 0.0001)
        self.assertEqual(out["threat"], 0.0001)

    def test_tiny_positive_rounds_to_floor(self):
        # Values that round to 0.0000 at 4 decimals must become 0.0001
        out = self.ev._normalize_scores({"hate": 4e-8, "violence": 0.00004})
        self.assertEqual(out["hate"], 0.0001)
        self.assertEqual(out["violence"], 0.0001)

    def test_tiny_nonzero_not_floored(self):
        out = self.ev._normalize_scores({"toxicity": 0.0002})
        self.assertEqual(out["toxicity"], 0.0002)

    def test_normal_scores_rounded(self):
        out = self.ev._normalize_scores({"toxicity": 0.123456})
        self.assertEqual(out["toxicity"], 0.1235)

    def test_out_of_range_raises(self):
        with self.assertRaises(ValueError):
            self.ev._normalize_scores({"toxicity": -0.1})
        with self.assertRaises(ValueError):
            self.ev._normalize_scores({"toxicity": 1.1})


if __name__ == "__main__":
    unittest.main()
