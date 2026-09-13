
import sys
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


                                                                           

class TestGenerateSingleVariant:
    """Tests for ea.evolution_engine.generate_single_variant."""

    def _make_parent(self, pid=1, prompt="hello", toxicity=0.5):
        return {
            "id": pid,
            "prompt": prompt,
            "moderation_result": {"google": {"scores": {"toxicity": toxicity}}},
        }

    @patch("ea.evolution_engine.get_custom_logging")
    def test_returns_genome_dicts_with_required_fields(self, mock_logging):
        mock_logger = MagicMock()
        mock_logging.return_value = (lambda *a, **k: mock_logger, lambda: "test.log", None, None)

        mock_pg = MagicMock()

        mock_op = MagicMock()
        mock_op.name = "MockMutation"
        mock_op.apply.return_value = ["evolved prompt 1"]

        parents = [self._make_parent()]

        with patch("ea.evolution_engine.LLM_POSAwareSynonymReplacement", return_value=mock_op), \
             patch("ea.evolution_engine.POSAwareAntonymReplacement", return_value=mock_op), \
             patch("ea.evolution_engine.MLMOperator", return_value=mock_op), \
             patch("ea.evolution_engine.LLMBasedParaphrasingOperator", return_value=mock_op), \
             patch("ea.evolution_engine.StylisticMutator", return_value=mock_op), \
             patch("ea.evolution_engine.LLMBackTranslationHIOperator", return_value=mock_op), \
             patch("ea.evolution_engine.NegationOperator", return_value=mock_op), \
             patch("ea.evolution_engine.TypographicalErrorsOperator", return_value=mock_op), \
             patch("ea.evolution_engine.ConceptAdditionOperator", return_value=mock_op), \
             patch("ea.evolution_engine.InformedEvolutionOperator", return_value=mock_op), \
             patch("ea.evolution_engine.random") as mock_random:

            mock_random.choice.return_value = mock_op

            from ea.evolution_engine import generate_single_variant
            results = generate_single_variant(parents, mock_pg,
                                              north_star_metric="toxicity",
                                              operators_mode="all")

        assert len(results) >= 1
        g = results[0]
        assert "prompt" in g
        assert g["prompt"] == "evolved prompt 1"
        assert g["status"] == "pending_generation"
        assert "operator" in g
        assert "variant_type" in g
        assert "parents" in g
        assert "id" not in g                     

    @patch("ea.evolution_engine.get_custom_logging")
    def test_empty_parents_returns_empty(self, mock_logging):
        mock_logger = MagicMock()
        mock_logging.return_value = (lambda *a, **k: mock_logger, lambda: "test.log", None, None)

        from ea.evolution_engine import generate_single_variant
        assert generate_single_variant([], MagicMock()) == []


                                                                           

class TestProcessSingleGenome:
    def test_updates_genome_in_place(self):
        mock_rg = MagicMock()
        mock_rg.generate_response.return_value = ("mock response text", 0.123)
        mock_rg.model_name = "test-model"

        genome = {"prompt": "test prompt", "status": "pending_generation"}

        from gne.response_generator import process_single_genome
        result = process_single_genome(mock_rg, genome)

        assert result is genome
        assert genome["generated_output"] == "mock response text"
        assert genome["model_name"] == "test-model"
        assert genome["response_duration"] == 0.123
        assert genome["status"] == "pending_evaluation"

    def test_empty_response(self):
        mock_rg = MagicMock()
        mock_rg.generate_response.return_value = ("", 0.01)
        mock_rg.model_name = None

        genome = {"prompt": "x", "status": "pending_generation"}

        from gne.response_generator import process_single_genome
        process_single_genome(mock_rg, genome)

        assert genome["generated_output"] == ""
        assert genome["status"] == "pending_evaluation"


                                                                           

class TestEvaluateSingleGenome:
    def test_updates_genome_on_success(self):
        mock_eval = MagicMock()
        mock_eval._evaluate_text_hybrid.return_value = {
            "google": {"scores": {"toxicity": 0.75}}
        }
        mock_eval._last_evaluation_time = {"duration": 0.456}

        genome = {
            "generated_output": "some text",
            "local_variant_id": "1_0",
            "status": "pending_evaluation",
        }

        from gne.evaluator import evaluate_single_genome
        result = evaluate_single_genome(mock_eval, genome)

        assert result is genome
        assert genome["status"] == "complete"
        assert genome["moderation_result"]["google"]["scores"]["toxicity"] == 0.75
        assert genome["evaluation_duration"] == 0.456

    def test_error_when_no_output(self):
        mock_eval = MagicMock()
        genome = {"generated_output": "", "status": "pending_evaluation"}

        from gne.evaluator import evaluate_single_genome
        evaluate_single_genome(mock_eval, genome)

        assert genome["status"] == "error"
        mock_eval._evaluate_text_hybrid.assert_not_called()

    def test_error_when_all_apis_fail(self):
        mock_eval = MagicMock()
        mock_eval._evaluate_text_hybrid.return_value = {"error": "All available APIs failed"}

        genome = {
            "generated_output": "text",
            "local_variant_id": "1_0",
            "status": "pending_evaluation",
        }

        from gne.evaluator import evaluate_single_genome
        evaluate_single_genome(mock_eval, genome)

        assert genome["status"] == "error"


                                                                           

class TestApplyRefusalPenaltySingle:
    def test_no_penalty_for_non_refusal(self):
        genome = {
            "generated_output": "This is a long response with lots of details about the topic at hand that clearly is not a refusal.",
            "moderation_result": {"google": {"scores": {"toxicity": 0.8}}},
        }

        from utils.refusal_penalty import apply_refusal_penalty_single
        result = apply_refusal_penalty_single(genome, "toxicity")

        assert result is genome
        assert genome["is_refusal"] == 0
        assert genome["moderation_result"]["google"]["scores"]["toxicity"] == 0.8

    @patch("utils.refusal_penalty.is_refusal", return_value=True)
    def test_penalty_applied_for_refusal(self, mock_is_refusal):
        genome = {
            "generated_output": "I cannot help with that.",
            "moderation_result": {"google": {"scores": {"toxicity": 0.8}}},
        }

        from utils.refusal_penalty import apply_refusal_penalty_single
        result = apply_refusal_penalty_single(genome, "toxicity")

        assert result is genome
        assert genome["is_refusal"] == 1
        expected = round(0.8 * 0.85, 4)
        assert genome["moderation_result"]["google"]["scores"]["toxicity"] == expected
        assert genome["north_star_score"] == expected

    @patch("utils.refusal_penalty.is_refusal", return_value=True)
    def test_no_penalty_for_tiny_score(self, mock_is_refusal):
        genome = {
            "generated_output": "No.",
            "moderation_result": {"google": {"scores": {"toxicity": 0.0001}}},
        }

        from utils.refusal_penalty import apply_refusal_penalty_single
        apply_refusal_penalty_single(genome, "toxicity")

        assert genome["is_refusal"] == 1
        assert genome["moderation_result"]["google"]["scores"]["toxicity"] == 0.0001
