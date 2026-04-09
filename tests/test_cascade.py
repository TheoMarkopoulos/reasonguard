from src.reasonguard.cascade import Decision, cascade_decision


def test_tier1_hvr_zero_accepts():
    decision, tier = cascade_decision(hvr=0.0, selfdoubt_score=-10.0)
    assert decision == Decision.ACCEPT
    assert tier == 1


def test_tier1_gate_disabled():
    """When gate disabled, HVR=0 should NOT auto-accept."""
    decision, tier = cascade_decision(
        hvr=0.0, selfdoubt_score=-10.0, hvr_gate_enabled=False
    )
    assert tier == 2
    assert decision == Decision.DEFER


def test_tier2_accept_above_tau():
    decision, tier = cascade_decision(hvr=1.0, selfdoubt_score=0.5, tau=0.0)
    assert decision == Decision.ACCEPT
    assert tier == 2


def test_tier2_defer_below_tau():
    decision, tier = cascade_decision(hvr=1.0, selfdoubt_score=-0.5, tau=0.0)
    assert decision == Decision.DEFER
    assert tier == 2


def test_tier2_exact_tau_accepts():
    """Score exactly at tau should ACCEPT."""
    decision, tier = cascade_decision(hvr=1.0, selfdoubt_score=0.0, tau=0.0)
    assert decision == Decision.ACCEPT
    assert tier == 2


def test_custom_tau():
    # With tau=1.0, score=0.5 should DEFER
    decision, _ = cascade_decision(hvr=1.0, selfdoubt_score=0.5, tau=1.0)
    assert decision == Decision.DEFER

    # With tau=-1.0, score=-0.5 should ACCEPT
    decision, _ = cascade_decision(hvr=1.0, selfdoubt_score=-0.5, tau=-1.0)
    assert decision == Decision.ACCEPT


def test_decision_enum_values():
    assert Decision.ACCEPT == "ACCEPT"
    assert Decision.DEFER == "DEFER"


def test_tier1_takes_precedence():
    """Even with a terrible selfdoubt score, HVR=0 should accept at tier 1."""
    decision, tier = cascade_decision(hvr=0.0, selfdoubt_score=-100.0)
    assert decision == Decision.ACCEPT
    assert tier == 1
