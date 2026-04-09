"""Redis caching layer — optional, app works without Redis."""

import hashlib
import json
import logging

from .config import settings

logger = logging.getLogger(__name__)

_redis_client = None
_redis_available = False

SCORE_CACHE_TTL = 3600  # 1 hour
CALIBRATION_PREFIX = "reasonguard:cal:"
SCORE_PREFIX = "reasonguard:score:"

# In-process hit/miss counters (also exposed via Prometheus in metrics.py)
_cache_hits = 0
_cache_misses = 0


def get_cache_stats() -> dict[str, int]:
    return {"hits": _cache_hits, "misses": _cache_misses}


def _connect() -> None:
    """Attempt to connect to Redis. Fails silently if unavailable."""
    global _redis_client, _redis_available
    if _redis_client is not None:
        return
    try:
        import redis

        _redis_client = redis.Redis.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=2
        )
        _redis_client.ping()
        _redis_available = True
        logger.info("Redis connected at %s", settings.redis_url)
    except Exception as exc:
        _redis_client = None
        _redis_available = False
        logger.warning("Redis unavailable, caching disabled: %s", exc)


def is_available() -> bool:
    _connect()
    return _redis_available


def _score_key(reasoning_trace: str, final_answer: str, model: str) -> str:
    """Build a cache key from a hash of the inputs."""
    content = f"{model}:{reasoning_trace}:{final_answer}"
    digest = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{SCORE_PREFIX}{digest}"


def get_cached_score(
    reasoning_trace: str, final_answer: str, model: str
) -> dict | None:
    """Look up a cached scoring result. Returns dict or None."""
    global _cache_hits, _cache_misses
    _connect()
    if not _redis_available:
        _cache_misses += 1
        return None
    try:
        key = _score_key(reasoning_trace, final_answer, model)
        raw = _redis_client.get(key)
        if raw is not None:
            _cache_hits += 1
            return json.loads(raw)
        _cache_misses += 1
        return None
    except Exception:
        _cache_misses += 1
        return None


def set_cached_score(
    reasoning_trace: str, final_answer: str, model: str, result: dict
) -> None:
    """Store a scoring result in cache."""
    _connect()
    if not _redis_available:
        return
    try:
        key = _score_key(reasoning_trace, final_answer, model)
        _redis_client.setex(key, SCORE_CACHE_TTL, json.dumps(result))
    except Exception:
        pass


def get_calibration(model: str) -> dict[str, float] | None:
    """Retrieve stored calibration stats for a model from Redis."""
    _connect()
    if not _redis_available:
        return None
    try:
        raw = _redis_client.get(f"{CALIBRATION_PREFIX}{model}")
        if raw is not None:
            return json.loads(raw)
        return None
    except Exception:
        return None


def set_calibration(model: str, stats: dict[str, float]) -> None:
    """Persist calibration stats for a model to Redis."""
    _connect()
    if not _redis_available:
        return
    try:
        _redis_client.set(f"{CALIBRATION_PREFIX}{model}", json.dumps(stats))
    except Exception:
        pass
