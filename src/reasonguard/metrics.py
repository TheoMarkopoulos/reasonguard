"""Prometheus metrics for ReasonGuard."""

from prometheus_client import Counter, Gauge, Histogram

REQUESTS_TOTAL = Counter(
    "reasonguard_requests_total",
    "Total requests",
    ["endpoint"],
)

SCORING_LATENCY = Histogram(
    "reasonguard_scoring_seconds",
    "Scoring latency",
)

HVR_VALUES = Histogram(
    "reasonguard_hvr",
    "HVR distribution",
    buckets=[0, 0.1, 0.5, 1, 2, 5, 10],
)

DECISIONS = Counter(
    "reasonguard_decisions_total",
    "Cascade decisions",
    ["decision", "tier"],
)

TIER1_COVERAGE = Gauge(
    "reasonguard_tier1_coverage",
    "Fraction of requests accepted by Tier 1",
)

SELFDOUBT_SCORES = Histogram(
    "reasonguard_selfdoubt_score",
    "SELFDOUBT score distribution",
)

CACHE_HITS = Counter(
    "reasonguard_cache_hits_total",
    "Redis cache hits",
)

CACHE_MISSES = Counter(
    "reasonguard_cache_misses_total",
    "Redis cache misses",
)

# Internal counters for Tier 1 coverage gauge calculation
_total_scored = 0
_tier1_accepted = 0


def record_scoring_result(
    hvr: float,
    decision: str,
    tier: int,
    selfdoubt_score: float | None,
) -> None:
    """Record metrics for a completed scoring operation."""
    global _total_scored, _tier1_accepted

    HVR_VALUES.observe(hvr)
    DECISIONS.labels(decision=decision, tier=str(tier)).inc()

    if selfdoubt_score is not None:
        SELFDOUBT_SCORES.observe(selfdoubt_score)

    _total_scored += 1
    if tier == 1 and decision == "ACCEPT":
        _tier1_accepted += 1
    TIER1_COVERAGE.set(_tier1_accepted / _total_scored)


def record_cache_hit() -> None:
    CACHE_HITS.inc()


def record_cache_miss() -> None:
    CACHE_MISSES.inc()
