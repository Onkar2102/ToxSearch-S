from utils.evaluator_profiles import attribute_prompt_label
from ea.paraphrasing import LLMBasedParaphrasingOperator
from ea.concept_addition import ConceptAdditionOperator


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
