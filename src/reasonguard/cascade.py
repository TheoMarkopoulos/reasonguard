from enum import Enum


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    DEFER = "DEFER"


def cascade_decision(
    hvr: float,
    selfdoubt_score: float,
    tau: float = 0.0,
    hvr_gate_enabled: bool = True,
) -> tuple[Decision, int]:
    """Two-tier deployment cascade.

    Returns (decision, tier).
    Tier 1: HVR=0 gate (96% precision)
    Tier 2: Calibrated z-sum threshold
    """
    # Tier 1: HVR = 0 gate
    if hvr_gate_enabled and hvr == 0.0:
        return Decision.ACCEPT, 1

    # Tier 2: Calibrated score threshold
    if selfdoubt_score >= tau:
        return Decision.ACCEPT, 2
    else:
        return Decision.DEFER, 2
