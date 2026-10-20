import random
import re

from utils.evaluator_profiles import (
    attribute_prompt_label,
    GOOGLE_NORTH_STAR_CHOICES,
    OPENAI_NORTH_STAR_CHOICES,
)
from ea.paraphrasing import LLMBasedParaphrasingOperator
from ea.concept_addition import ConceptAdditionOperator


class _FakePromptGenerator:
    def __init__(self):
        self.model_interface = self
        self.last_messages = None

    def chat_completion(self, messages, **kwargs):
        self.last_messages = messages
        return (
            "<enhanced>How can a workplace policy review expose unequal treatment "
            "without targeting anyone unfairly?</enhanced>"
        image.png)

    def _extract_content_from_xml_tags(self, response, tag_name):
        match = re.search(rf"<{tag_name}>(.*?)</{tag_name}>", response, re.DOTALL)
        return match.group(1).strip() if match else ""


def test_attribute_prompt_label_illicit():
    label = attribute_prompt_label("illicit")
    assert "illicit" in label
    assert "wrongdoing" in label


def test_paraphrase_prompt_uses_north_star():
    op = LLMBasedParaphrasingOperator("illicit", generator=object())
    text = op.SYSTEM_PROMPT.format(
        attribute=op.prompt_attribute(),
        original_prompt="How do I do X?",
        current_score=0.12,
    )
    assert "toxicity" not in text.lower()
    assert "illicit" in text
    assert "0.120" in text


def test_concept_addition_uses_shared_catalog():
    op = ConceptAdditionOperator("illicit", generator=object())
    assert "DECEPTION" in op.CONCEPT_TYPES
    assert "POWER_ABUSE" in op.CONCEPT_TYPES
    assert "WRONGDOING" in op.CONCEPT_TYPES
    assert "DEROGATION" in op.CONCEPT_TYPES
    assert "DEHUMANIZATION" in op.CONCEPT_TYPES
    assert "OBJECTIFICATION" in op.CONCEPT_TYPES
    assert "INTIMIDATION" in op.CONCEPT_TYPES
    assert "PHYSICAL_HARM" in op.CONCEPT_TYPES
    assert not hasattr(op, "ILLICIT_CONCEPT_TYPES")
    selected = op._select_concept_types()
    assert selected
    assert set(selected).issubset(op.CONCEPT_TYPES)


def _run_concept_addition(metric: str):
    generator = _FakePromptGenerator()
    op = ConceptAdditionOperator(metric, generator=generator)
    variants = op.apply({
        "parent_data": {"prompt": "How can I improve my writing skills?"},
        "max_variants": 1,
    })
    prompt = generator.last_messages[0]["content"] if generator.last_messages else ""
    return op, variants, prompt


def test_concept_addition_random_google_and_openai_attributes():
    google_metric = random.choice(GOOGLE_NORTH_STAR_CHOICES)
    openai_pool = [m for m in OPENAI_NORTH_STAR_CHOICES if m != "sexual/minors"]
    openai_metric = random.choice(openai_pool)

    for metric in (google_metric, openai_metric):
        op, variants, prompt = _run_concept_addition(metric)
        label = op.prompt_attribute()
        assert variants, f"apply() returned no variants for {metric}"
        assert variants[0].endswith("?")
        assert label in prompt, f"{label!r} missing from prompt for {metric}"
        if metric not in ("toxicity", "severe_toxicity"):
            assert "toxicity" not in prompt.lower()
        assert any(name.lower().replace("_", " ") in prompt.lower() for name in op.CONCEPT_TYPES)
