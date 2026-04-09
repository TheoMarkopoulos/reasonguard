from src.reasonguard.scoring.hvr import compute_hvr
from src.reasonguard.scoring.markers import DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN


def test_no_hedge_markers():
    """Trace with no hedge words -> HVR = 0"""
    trace = "The answer is clearly B. Let me verify by substituting back."
    hvr, h, v = compute_hvr(trace, DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN)
    assert hvr == 0.0
    assert h == 0


def test_high_uncertainty():
    """Trace with many hedge words, no verify -> high HVR"""
    trace = "Maybe the answer is A. Perhaps B. Possibly C. I'm not sure, probably A."
    hvr, h, v = compute_hvr(trace, DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN)
    assert hvr > 1.0


def test_balanced_trace():
    """Hedge words offset by verify words -> moderate HVR"""
    trace = "Maybe it's A. Let me check... perhaps B. Let me verify the calculation."
    hvr, h, v = compute_hvr(trace, DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN)
    assert 0 < hvr < 2.0


def test_word_boundary():
    """'check' inside 'checkout' should NOT match"""
    trace = "I'll go to checkout now."
    _, h, v = compute_hvr(trace, DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN)
    assert v == 0


def test_empty_trace():
    hvr, h, v = compute_hvr("")
    assert hvr == 0.0
    assert h == 0
    assert v == 0


def test_only_verify_markers():
    trace = "Let me check. Let me verify. I need to prove this. Let me recheck."
    hvr, h, v = compute_hvr(trace)
    assert hvr == 0.0
    assert h == 0
    assert v > 0


def test_hvr_formula():
    """HVR = h / (v + 1). With 3 hedges and 2 verifies: 3 / (2+1) = 1.0"""
    trace = "Maybe A. Perhaps B. Probably C. Let me check. Let me verify."
    hvr, h, v = compute_hvr(trace)
    assert h == 3
    assert v == 2
    assert hvr == 3.0 / (2 + 1)


def test_case_insensitive():
    trace = "MAYBE the answer is A. PERHAPS B."
    hvr, h, v = compute_hvr(trace)
    assert h == 2
