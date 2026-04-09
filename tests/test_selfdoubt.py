import math

from src.reasonguard.scoring.selfdoubt import compute_selfdoubt_score


def test_zero_hvr_high_conf():
    """HVR=0, high confidence -> high positive score.
    z(-HVR) = (0 - 0) / 1 = 0, z(Verb) = (0.95 - 0.5) / 0.2 = 2.25 -> positive.
    """
    score = compute_selfdoubt_score(
        hvr=0.0, verbalized_conf=0.95,
        mu_hvr=0.0, sigma_hvr=1.0,
        mu_verb=0.5, sigma_verb=0.2,
    )
    assert score > 0


def test_high_hvr_low_conf():
    """High HVR, low confidence -> low (negative) score."""
    score = compute_selfdoubt_score(
        hvr=5.0, verbalized_conf=0.2,
        mu_hvr=1.0, sigma_hvr=0.5,
        mu_verb=0.7, sigma_verb=0.15,
    )
    assert score < 0


def test_z_score_formula():
    """Verify the exact formula: z(-HVR) + z(Verb)."""
    hvr, verb = 2.0, 0.8
    mu_hvr, sigma_hvr = 1.0, 0.5
    mu_verb, sigma_verb = 0.6, 0.2

    expected_z_hvr = (-hvr - mu_hvr) / (sigma_hvr + 1e-8)
    expected_z_verb = (verb - mu_verb) / (sigma_verb + 1e-8)
    expected = expected_z_hvr + expected_z_verb

    score = compute_selfdoubt_score(hvr, verb, mu_hvr, sigma_hvr, mu_verb, sigma_verb)
    assert math.isclose(score, expected, rel_tol=1e-6)


def test_zero_sigma_no_crash():
    """sigma=0 should not cause division by zero (epsilon guard)."""
    score = compute_selfdoubt_score(
        hvr=1.0, verbalized_conf=0.5,
        mu_hvr=0.0, sigma_hvr=0.0,
        mu_verb=0.0, sigma_verb=0.0,
    )
    assert math.isfinite(score)


def test_symmetric_around_mean():
    """At mean values, z-scores are 0, so score should be ~0."""
    score = compute_selfdoubt_score(
        hvr=0.0, verbalized_conf=0.7,
        mu_hvr=0.0, sigma_hvr=1.0,
        mu_verb=0.7, sigma_verb=0.2,
    )
    assert math.isclose(score, 0.0, abs_tol=1e-6)
