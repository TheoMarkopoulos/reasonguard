from src.reasonguard.scoring.verbalized import parse_verbalized_confidence


def test_standard_format():
    assert parse_verbalized_confidence("\\boxed{B} Confidence: 95%") == 0.95


def test_after_think_tag():
    text = "<think>maybe it's A</think>The answer is B. Confidence: 90%"
    assert parse_verbalized_confidence(text) == 0.90


def test_no_confidence():
    assert parse_verbalized_confidence("The answer is B.") is None


def test_confidence_equals_format():
    assert parse_verbalized_confidence("confidence = 80%") == 0.80


def test_confidence_with_brackets():
    assert parse_verbalized_confidence("Confidence: [75]%") == 0.75


def test_decimal_confidence():
    assert parse_verbalized_confidence("Confidence: 92.5%") == 0.925


def test_zero_confidence():
    assert parse_verbalized_confidence("Confidence: 0%") == 0.0


def test_hundred_confidence():
    assert parse_verbalized_confidence("Confidence: 100%") == 1.0


def test_ignores_confidence_in_think_tags():
    text = "<think>Confidence: 30%</think>The answer is B. Confidence: 95%"
    assert parse_verbalized_confidence(text) == 0.95


def test_fallback_to_percentage():
    text = "The answer is B with 85% certainty."
    assert parse_verbalized_confidence(text) == 0.85


def test_last_match_wins():
    text = "Confidence: 50%. Actually, Confidence: 90%"
    assert parse_verbalized_confidence(text) == 0.90


def test_out_of_range_ignored():
    # 150% is > 100, should return None
    assert parse_verbalized_confidence("Confidence: 150%") is None


def test_multiple_think_tags():
    text = "<think>first thought</think><think>second Confidence: 20%</think>Final answer. Confidence: 88%"
    assert parse_verbalized_confidence(text) == 0.88


def test_empty_string():
    assert parse_verbalized_confidence("") is None
