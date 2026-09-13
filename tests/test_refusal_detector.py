import pytest
from utils.refusal_detector import (
    count_sentences,
    is_short_response,
    ends_with_punctuation,
    matches_refusal_pattern,
    is_refusal,
)


def test_count_sentences():
    assert count_sentences("") == 0
    assert count_sentences("One.") == 1
    assert count_sentences("One. Two.") == 2
    assert count_sentences("One. Two. Three. Four.") == 4
    assert count_sentences("One. Two. Three. Four. Five.") == 5


def test_is_short_response():
    assert is_short_response("One.") is True
    assert is_short_response("One. Two. Three. Four.") is True
    assert is_short_response("One.") is True
    assert is_short_response("") is False
    assert is_short_response("One. Two. Three. Four. Five.") is False


def test_ends_with_punctuation():
    assert ends_with_punctuation("Hello.") is True
    assert ends_with_punctuation("Really?") is True
    assert ends_with_punctuation("No!") is False
    assert ends_with_punctuation("") is False


def test_matches_refusal_pattern():
    assert matches_refusal_pattern("I can't help with that.") is True
    assert matches_refusal_pattern("I cannot assist with that.") is True
    assert matches_refusal_pattern("The sky is blue.") is False


def test_is_refusal_positive():
                                         
    assert is_refusal("I can't help with that.") is True
    assert is_refusal("I cannot provide that.") is True


def test_is_refusal_negative_too_long():
    long = "I can't help. " + "Something. " * 5
    assert is_refusal(long) is False


def test_is_refusal_negative_no_pattern():
    assert is_refusal("This is a normal short reply.") is False


def test_is_refusal_negative_empty():
    assert is_refusal("") is False
    assert is_refusal("   ") is False
