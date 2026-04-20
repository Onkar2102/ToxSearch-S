

from typing import List, Any, Dict
import traceback

from .variation_operators import VariationOperator

from utils import get_custom_logging
from utils.device_utils import get_optimal_device, move_to_device
get_logger, _, _, _ = get_custom_logging()


class SemanticSimilarityCrossover(VariationOperator):
    """Semantic similarity crossover operator for prompt recombination."""

    def __init__(self, log_file=None):
        
        super().__init__("SemanticSimilarityCrossover", "crossover", "Combines semantically similar sentences from two parents.")
        self.logger = get_logger(self.name, log_file)
        self.logger.debug(f"Initialized operator: {self.name}")

        self.device = get_optimal_device()
        self.logger.info(f"Using device: {self.device}")

        self.model = self._load_model_with_device_support()

    def _load_model_with_device_support(self):
        
        try:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer("all-MiniLM-L6-v2")

            model = move_to_device(model, self.device)
            self.logger.info(f"Sentence transformer model using {self.device}")

            return model

        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer model: {e}")
            raise RuntimeError(f"Unable to load sentence transformer model: {e}")

    def apply(self, operator_input: Dict[str, Any]) -> List[str]:
        
        try:
            import time
            start_time = time.time()
            import torch
            from sentence_transformers import util

            if not isinstance(operator_input, dict):
                self.logger.error(f"{self.name}: Input must be a dictionary")
                return []

            parent_data = operator_input.get("parent_data", [])
            max_variants = operator_input.get("max_variants", 1)

            if not isinstance(parent_data, list) or len(parent_data) < 2:
                self.logger.error(f"{self.name}: Insufficient parents for crossover. Required: 2, Got: {len(parent_data) if isinstance(parent_data, list) else 'not a list'}")
                return []

            parent1_data = parent_data[0]
            parent2_data = parent_data[1]

            if not isinstance(parent1_data, dict) or not isinstance(parent2_data, dict):
                self.logger.error(f"{self.name}: Parents must be enriched parent dictionaries")
                return []

            if "prompt" not in parent1_data or "prompt" not in parent2_data:
                self.logger.error(f"{self.name}: Parents must contain 'prompt' field")
                return []

            parent_texts = [parent1_data.get("prompt", ""), parent2_data.get("prompt", "")]
            self.logger.debug(f"{self.name}: Using enriched parent data format")

            p1_sentences = parent_texts[0].split(". ")
            p2_sentences = parent_texts[1].split(". ")

            try:
                p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True, device=self.device)
                p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True, device=self.device)
            except Exception as e:
                self.logger.warning(f"GPU embedding generation failed, falling back to CPU: {e}")
                p1_embeddings = self.model.encode(p1_sentences, convert_to_tensor=True, device="cpu")
                p2_embeddings = self.model.encode(p2_sentences, convert_to_tensor=True, device="cpu")

            matched_sentences = []
            for i, emb1 in enumerate(p1_embeddings):
                similarities = util.cos_sim(emb1, p2_embeddings)[0]
                top_idx = int(torch.argmax(similarities).item())
                sim_score = similarities[top_idx].item()
                if sim_score > 0.5:
                    matched_sentences.append(p1_sentences[i])
                    matched_sentences.append(p2_sentences[top_idx])

            cleaned_sentences = []
            for sentence in matched_sentences:
                sentence = sentence.strip()
                if sentence:
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    cleaned_sentences.append(sentence)

            result = " ".join(cleaned_sentences).strip()

            result = result.replace('. ', '. ').replace('..', '.').strip()

            self.logger.debug(f"{self.name}: Created crossover from {len(cleaned_sentences)} semantically matched sentences.")
            return [result] if result and result != "." else [parent_texts[0]]
        except Exception as e:
            self.logger.error(f"{self.name}: apply failed with error: {e}\nTrace: {traceback.format_exc()}")
            return parent_texts[:1] if parent_texts else []
        finally:
            try:
                end_time = time.time()
                operation_time = end_time - start_time
                if not hasattr(self, '_last_operation_time'):
                    self._last_operation_time = {}
                self._last_operation_time['duration'] = operation_time
            except Exception:
                pass

