import re

from .markers import DEFAULT_HEDGE_PATTERN, DEFAULT_VERIFY_PATTERN, count_markers


def compute_hvr(
    trace: str,
    hedge_pattern: re.Pattern = DEFAULT_HEDGE_PATTERN,
    verify_pattern: re.Pattern = DEFAULT_VERIFY_PATTERN,
) -> tuple[float, int, int]:
    """Compute Hedge-to-Verify Ratio.

    HVR(T) = h(T) / (v(T) + 1)

    Returns (hvr, hedge_count, verify_count).
    """
    h = count_markers(trace, hedge_pattern)
    v = count_markers(trace, verify_pattern)
    hvr = h / (v + 1)
    return hvr, h, v
