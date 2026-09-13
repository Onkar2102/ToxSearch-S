import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.evaluator_profiles import (
    GOOGLE_PROFILE,
    OPENAI_PROFILE,
    resolve_evaluator,
    validate_north_star,
    moderation_methods_to_evaluator,
    moderation_methods_to_backend_list,
    set_active_evaluator,
)


class TestEvaluatorProfiles(unittest.TestCase):
    def test_resolve_google_and_openai(self):
        self.assertEqual(resolve_evaluator("google").backend_key, "google")
        self.assertEqual(resolve_evaluator("openai").backend_key, "openai")
        self.assertEqual(resolve_evaluator("perspective").name, "google")
        self.assertEqual(resolve_evaluator("omni").name, "openai")

    def test_unknown_evaluator_raises(self):
        with self.assertRaises(ValueError):
            resolve_evaluator("unknown-backend")

    def test_validate_north_star_per_profile(self):
        self.assertEqual(validate_north_star(GOOGLE_PROFILE, "threat"), "threat")
        self.assertEqual(validate_north_star(OPENAI_PROFILE, "violence"), "violence")
        with self.assertRaises(ValueError):
            validate_north_star(GOOGLE_PROFILE, "violence")
        with self.assertRaises(ValueError):
            validate_north_star(OPENAI_PROFILE, "toxicity")

    def test_openai_metric_alias(self):
        self.assertEqual(OPENAI_PROFILE.resolve_metric_alias("toxicity"), "harassment")
        self.assertEqual(OPENAI_PROFILE.resolve_metric_alias("violence"), "violence")

    def test_moderation_methods_to_evaluator(self):
        self.assertEqual(moderation_methods_to_evaluator(["google"]), "google")
        self.assertEqual(moderation_methods_to_evaluator(["perspective", "all"]), "google")
        self.assertEqual(moderation_methods_to_evaluator(["openai"]), "openai")
        self.assertEqual(moderation_methods_to_evaluator(["omni"]), "openai")

    def test_backend_list_exclusive(self):
        profile = set_active_evaluator("openai")
        self.assertEqual(moderation_methods_to_backend_list(profile), ["openai"])


if __name__ == "__main__":
    unittest.main()
