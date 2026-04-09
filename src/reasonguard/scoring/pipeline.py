"""Scoring pipeline — runs HVR + verbalized + SELFDOUBT + cascade in one call."""

from .. import cache
from ..cascade import Decision, cascade_decision
from ..config import settings
from ..metrics import SCORING_LATENCY, record_cache_hit, record_cache_miss, record_scoring_result
from ..models import ReasonGuardResult
from .hvr import compute_hvr
from .selfdoubt import compute_selfdoubt_score
from .verbalized import parse_verbalized_confidence

# Default calibration stats (sensible priors before /calibrate is called).
DEFAULT_MU_HVR = 0.5
DEFAULT_SIGMA_HVR = 1.0
DEFAULT_MU_VERB = 0.8
DEFAULT_SIGMA_VERB = 0.15

# In-memory calibration store (fallback when Redis is unavailable)
_calibration_store: dict[str, dict[str, float]] = {}


def set_calibration(model: str, mu_hvr: float, sigma_hvr: float, mu_verb: float, sigma_verb: float) -> None:
    stats = {
        "mu_hvr": mu_hvr,
        "sigma_hvr": sigma_hvr,
        "mu_verb": mu_verb,
        "sigma_verb": sigma_verb,
    }
    _calibration_store[model] = stats
    cache.set_calibration(model, stats)


def get_calibration(model: str) -> dict[str, float]:
    # Try Redis first
    redis_cal = cache.get_calibration(model)
    if redis_cal is not None:
        return redis_cal
    # Fall back to in-memory
    return _calibration_store.get(model, {
        "mu_hvr": DEFAULT_MU_HVR,
        "sigma_hvr": DEFAULT_SIGMA_HVR,
        "mu_verb": DEFAULT_MU_VERB,
        "sigma_verb": DEFAULT_SIGMA_VERB,
    })


def _compute(
    reasoning_trace: str,
    final_answer: str,
    model: str,
    tau: float,
    hvr_gate_enabled: bool,
) -> ReasonGuardResult:
    """Pure scoring computation (no cache interaction)."""
    # Step 1: HVR
    hvr, hedge_count, verify_count = compute_hvr(reasoning_trace)

    # Step 2: Verbalized confidence (parsed from final answer)
    verbalized_conf = parse_verbalized_confidence(final_answer)

    # Step 3: SELFDOUBT score (requires calibration + verbalized confidence)
    cal = get_calibration(model)
    selfdoubt_score: float | None = None
    if verbalized_conf is not None:
        selfdoubt_score = compute_selfdoubt_score(
            hvr=hvr,
            verbalized_conf=verbalized_conf,
            mu_hvr=cal["mu_hvr"],
            sigma_hvr=cal["sigma_hvr"],
            mu_verb=cal["mu_verb"],
            sigma_verb=cal["sigma_verb"],
        )

    # Step 4: Cascade decision
    decision, tier = cascade_decision(
        hvr=hvr,
        selfdoubt_score=selfdoubt_score if selfdoubt_score is not None else 0.0,
        tau=tau,
        hvr_gate_enabled=hvr_gate_enabled,
    )

    return ReasonGuardResult(
        reliable=decision == Decision.ACCEPT,
        tier=tier,
        hvr=hvr,
        hedge_count=hedge_count,
        verify_count=verify_count,
        verbalized_confidence=verbalized_conf,
        selfdoubt_score=selfdoubt_score,
        decision=decision.value,
        mi_estimate=None,
    )


def score_trace(
    reasoning_trace: str,
    final_answer: str,
    model: str = "unknown",
    tau: float | None = None,
    hvr_gate_enabled: bool | None = None,
) -> ReasonGuardResult:
    """Run the full scoring pipeline on a single trace.

    Checks Redis cache first; computes and caches on miss.
    """
    if tau is None:
        tau = settings.default_tau
    if hvr_gate_enabled is None:
        hvr_gate_enabled = settings.hvr_gate_enabled

    # Check cache
    cached = cache.get_cached_score(reasoning_trace, final_answer, model)
    if cached is not None:
        record_cache_hit()
        return ReasonGuardResult(**cached)
    record_cache_miss()

    # Compute
    with SCORING_LATENCY.time():
        result = _compute(reasoning_trace, final_answer, model, tau, hvr_gate_enabled)

    # Record metrics
    record_scoring_result(
        hvr=result.hvr,
        decision=result.decision,
        tier=result.tier,
        selfdoubt_score=result.selfdoubt_score,
    )

    # Store in cache
    cache.set_cached_score(reasoning_trace, final_answer, model, result.model_dump())

    return result
