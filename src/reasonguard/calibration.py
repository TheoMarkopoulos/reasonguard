"""Calibration statistics manager.

Collects traces, computes per-model μ and σ for HVR and verbalized confidence,
and stores them via the cache/pipeline layer.
"""

import math

from .models import CalibrateResponse
from .scoring.hvr import compute_hvr
from .scoring.pipeline import set_calibration
from .scoring.verbalized import parse_verbalized_confidence


def calibrate(
    model: str,
    traces: list[dict],
) -> CalibrateResponse:
    """Run calibration over a batch of traces.

    Each entry in `traces` must have keys "reasoning_trace" and "final_answer".
    Computes μ/σ for HVR and verbalized confidence, stores them, and returns stats.
    """
    hvr_values: list[float] = []
    verb_values: list[float] = []

    for entry in traces:
        reasoning_trace = entry.get("reasoning_trace", "")
        final_answer = entry.get("final_answer", "")

        hvr, _, _ = compute_hvr(reasoning_trace)
        hvr_values.append(hvr)

        verb = parse_verbalized_confidence(final_answer)
        if verb is not None:
            verb_values.append(verb)

    mu_hvr = _mean(hvr_values)
    sigma_hvr = _std(hvr_values)

    mu_verb = _mean(verb_values) if verb_values else 0.8
    sigma_verb = _std(verb_values) if verb_values else 0.15

    set_calibration(model, mu_hvr, sigma_hvr, mu_verb, sigma_verb)

    return CalibrateResponse(
        model=model,
        mu_hvr=mu_hvr,
        sigma_hvr=sigma_hvr,
        mu_verb=mu_verb,
        sigma_verb=sigma_verb,
        n_traces=len(traces),
    )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 1.0
    mu = _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / len(values)
    return math.sqrt(variance) if variance > 0 else 1e-8
