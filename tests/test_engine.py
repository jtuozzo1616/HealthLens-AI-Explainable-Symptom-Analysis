"""Unit tests check software behavior, not clinical performance."""

import pytest

from healthlens.engine import (
    SymptomAnalyzer, MAX_INPUT_CHARS, find_warnings, normalize_text,
)


@pytest.fixture(scope="module")
def model():
    return SymptomAnalyzer()


@pytest.mark.parametrize("text", [
    "I have chest pain and a cough.",
    "I cannot breathe.",
    "He is coughing up blood.",
    "My speech is slurred.",
    "I cannot stop bleeding from a wound.",
    "I have severe abdominal pain.",
    "I fainted and lost consciousness.",
    "My throat is swelling.",
    "I have the worst headache of my life.",
    "I am having a seizure.",
    "I am having suicidal thoughts.",
])
def test_warning_examples_are_flagged(model, text):
    answer = model.analyze(text)
    assert answer.state == "warning"
    assert answer.warning_signs
    assert answer.topic is None


@pytest.mark.parametrize("text", [
    "I do not have chest pain, but my arm is itchy.",
    "No chest pain, just an itchy rash.",
    "I deny shortness of breath and have a sore neck.",
    "I have no trouble breathing but my knee aches.",
    "I have not lost consciousness and my eyes are watery.",
])
def test_simple_negations_do_not_trigger_warning(model, text):
    assert not find_warnings(text)
    assert model.analyze(text).state != "warning"


def test_clause_negation_does_not_hide_real_warning(model):
    assert model.analyze("No chest pain, but I cannot breathe.").state == "warning"


def test_safety_rules_precede_similarity_model(model):
    result = model.analyze("A cough, runny nose, and severe abdominal pain.")
    assert result.state == "warning"
    assert result.similarity is None


def test_topic_has_explainable_model_trace(model):
    result = model.analyze("My itchy red rash is bothering me.")
    assert result.state == "topic"
    assert result.topic == "skin"
    assert result.matched_example
    assert 0 < result.similarity <= 1


def test_unknown_content_abstains(model):
    assert model.analyze("I went shopping and cooked pasta.").state == "abstain"
    assert model.analyze("").state == "abstain"
    assert model.analyze("z" * (MAX_INPUT_CHARS + 1)).state == "abstain"


def test_no_false_probability(model):
    result = model.analyze("My eyes are watery and itchy.")
    assert result.state == "topic"
    assert result.similarity < 1
    assert "NOT a diagnosis" in result.detail


def test_normalization():
    assert normalize_text("  My    skin\n is itchy.  ") == "My skin is itchy."


def test_dataset_has_expected_categories(model):
    assert len(model.examples) == 56
    assert len(set(model.labels)) == 7


def test_negated_chest_phrase_does_not_contaminate_topic(model):
    result = model.analyze("No chest pain, but I have a sore back after working out.")
    assert result.state == "topic"
    assert result.topic == "musculoskeletal"
