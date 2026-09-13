import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

for _mod in ("pandas", "dotenv"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from utils.population_io import _extract_north_star_score, get_moderation_scores
from utils.evaluator_profiles import resolve_evaluator, set_active_evaluator, set_active_north_star
from gne.evaluator import HybridModerationEvaluator


class _FakeCategoryScores:
    def model_dump(self):
        return {
            "harassment": 0.12,
            "violence": 0.87,
            "hate": 0.05,
        }


class _FakeResult:
    def __init__(self):
        self.category_scores = _FakeCategoryScores()


class _FakeResponse:
    results = [_FakeResult()]


def _make_openai_evaluator():
    evaluator = HybridModerationEvaluator.__new__(HybridModerationEvaluator)
    evaluator.profile = resolve_evaluator("openai")
    evaluator.openai_model = "omni-moderation-latest"
    evaluator.openai_available = True
    evaluator.google_available = False
    evaluator.logger = MagicMock()
    evaluator.openai_client = MagicMock()
    evaluator.openai_client.moderations.create.return_value = _FakeResponse()
    return evaluator


def _make_google_evaluator():
    evaluator = HybridModerationEvaluator.__new__(HybridModerationEvaluator)
    evaluator.profile = resolve_evaluator("google")
    evaluator.google_available = True
    evaluator.openai_available = False
    evaluator.logger = MagicMock()
    evaluator.google_client = MagicMock()
    evaluator.google_client.comments.return_value.analyze.return_value.execute.return_value = {
        "attributeScores": {
            "TOXICITY": {"summaryScore": {"value": 0.42}},
            "THREAT": {"summaryScore": {"value": 0.1}},
        }
    }
    return evaluator


class TestEvaluatorOpenAI(unittest.TestCase):
    def test_evaluate_with_openai_normalizes_scores(self):
        evaluator = _make_openai_evaluator()
        result, _info = evaluator._evaluate_with_openai("harmful text", "g1")
        self.assertIsNotNone(result)
        scores = result["scores"]
        self.assertIn("violence", scores)
        self.assertGreaterEqual(scores["violence"], 0.0001)
        self.assertEqual(result["api_version"], "openai")

    def test_extract_north_star_openai_nested(self):
        set_active_evaluator("openai")
        set_active_north_star("violence")
        genome = {
            "evaluator": "openai",
            "moderation_result": {
                "openai": {
                    "scores": {
                        "harassment": 0.1,
                        "violence": 0.75,
                    }
                }
            },
        }
        self.assertEqual(_extract_north_star_score(genome, "violence"), 0.75)
        scores = get_moderation_scores(genome)
        self.assertIsNotNone(scores)
        self.assertIn("violence", scores)

    def test_hybrid_evaluate_text_stamps_openai_backend(self):
        evaluator = _make_openai_evaluator()
        out = evaluator._evaluate_text_hybrid("text", "g2")
        self.assertIn("openai", out)
        self.assertNotIn("error", out)

    def test_google_path_unchanged(self):
        evaluator = _make_google_evaluator()
        out = evaluator._evaluate_text_hybrid("text", "g3")
        self.assertIn("google", out)
        genome = {
            "evaluator": "google",
            "moderation_result": out,
        }
        self.assertEqual(_extract_north_star_score(genome, "toxicity"), 0.42)

if __name__ == "__main__":
    unittest.main()
