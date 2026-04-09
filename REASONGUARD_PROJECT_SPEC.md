# ReasonGuard — Project Specification for Claude Code

## Overview
Build **ReasonGuard**, a production-ready FastAPI middleware that scores the reliability of any reasoning LLM's output in a single pass, with zero access to model internals. Based on two research papers: SELFDOUBT (arXiv:2604.06389) and RAGEN-2 (arXiv:2604.06268).

GitHub repo name: `reasonguard`

---

## Architecture

```
Client → ReasonGuard FastAPI Proxy → Upstream LLM (OpenAI-compatible API)
                ↓
        1. Intercept response + reasoning trace
        2. Compute HVR score (SELFDOUBT)
        3. Compute MI metrics (RAGEN-2, optional)
        4. Return augmented response with confidence object
                ↓
        Redis cache ← Prometheus metrics → Grafana dashboard
```

---

## Core Algorithm #1: Hedge-to-Verify Ratio (HVR) — from SELFDOUBT

### What it does
Parses a reasoning trace for **hedging markers** (uncertainty language) and **verification markers** (self-checking language), then computes a ratio.

### Formula
```
HVR(T) = h(T) / (v(T) + 1)
```
Where:
- `h(T)` = total hedge marker occurrences in trace T
- `v(T)` = total verify marker occurrences in trace T
- `+1` prevents division by zero

### Key insight: HVR = 0 Gate
Traces with **zero hedging markers** are correct **96.1% of the time** (across 7 models, 3 datasets). This is a free, high-precision confidence gate.

### Marker Dictionaries
Use word-boundary regex matching (`\b...\b`, case-insensitive, non-overlapping, longest-match-first).

**Default Hedge Seeds (top-10 from paper Table 21):**
```
possibly, seemingly, maybe, apparently, probably, presumably, perhaps, likely, reportedly, seems
```

**Default Verify Seeds (top-10 from paper Table 21):**
```
check, reassess, reevaluate, re-evaluate, reinspect, rechecking, verifying, reconfirming, recheck, prove
```

**Extended markers (from paper Figure 2 examples):**
- Hedge n-grams: "might be", "not sure", "hmm", "so perhaps", "so maybe", "plausible"
- Verify n-grams: "let me check", "lets check", "me verify", "let me verify", "examine", "evaluate", "compute", "determine", "confirm", "recalculate", "substitute back"

### SELFDOUBT Score (full fusion)
```python
score = z(-1 * HVR) + z(Verb)
```
Where:
- `z(x) = (x - μ) / σ` — z-score normalization
- `Verb` = verbalized confidence parsed from the model's response (0.0 to 1.0)
- `μ_hvr, σ_hvr, μ_verb, σ_verb` = calibration statistics from 90 unlabeled traces

### Deployment Cascade (2-tier)
- **Tier 1**: If `HVR == 0` → ACCEPT (96% precision, ~25% coverage)
- **Tier 2**: Compute `score = z_cal(-HVR) + z_cal(Verb)`. If `score >= τ` → ACCEPT, else → DEFER
  - `τ = 0` is the balanced default (71% coverage, 89.7% accuracy)
  - Higher τ = more conservative, lower τ = more permissive

---

## Core Algorithm #2: Mutual Information Proxy — from RAGEN-2

### What it does
Detects **template collapse** — when reasoning looks diverse but is actually input-agnostic (same boilerplate regardless of the actual question).

### Key concept
```
H(Z) = I(X; Z) + H(Z | X)
```
- `H(Z | X)` = within-input diversity (entropy) — can stay high even during collapse
- `I(X; Z)` = cross-input distinguishability (mutual information) — drops during collapse

### Simplified implementation for ReasonGuard
Since we don't have access to model internals or training rollouts, implement a **batch-level MI estimate** using embedding similarity:

1. Collect recent reasoning traces in a sliding window (e.g., last 100 traces)
2. Embed each trace using a sentence transformer
3. For each trace, compute cosine similarity to all other traces
4. **High average similarity across different inputs = low MI = template collapse warning**

### Metrics to expose
- `mi_estimate`: float — estimated mutual information (higher = better, reasoning is input-grounded)
- `template_collapse_risk`: bool — True if MI drops below threshold
- `trace_diversity`: float — average pairwise distance of recent traces

---

## Project Structure

```
reasonguard/
├── README.md
├── LICENSE                      # MIT
├── pyproject.toml               # Python 3.11+, dependencies
├── Dockerfile
├── docker-compose.yml           # FastAPI + Redis + Prometheus + Grafana
├── .github/
│   └── workflows/
│       └── ci.yml               # pytest + linting
├── src/
│   └── reasonguard/
│       ├── __init__.py
│       ├── main.py              # FastAPI app entry point
│       ├── config.py            # Settings via pydantic-settings
│       ├── proxy.py             # LLM proxy layer (intercept + forward)
│       ├── scoring/
│       │   ├── __init__.py
│       │   ├── hvr.py           # HVR computation + marker matching
│       │   ├── markers.py       # Hedge/verify marker dictionaries + regex
│       │   ├── verbalized.py    # Verbalized confidence parser
│       │   ├── selfdoubt.py     # Full SELFDOUBT score (z-score fusion)
│       │   └── mi_proxy.py      # Mutual information proxy (batch embedding)
│       ├── cascade.py           # 2-tier deployment cascade logic
│       ├── calibration.py       # Calibration statistics manager
│       ├── cache.py             # Redis caching layer
│       ├── metrics.py           # Prometheus metrics
│       ├── models.py            # Pydantic request/response models
│       └── trace_parser.py      # Extract reasoning trace from <think> tags / CoT
├── tests/
│   ├── test_hvr.py
│   ├── test_markers.py
│   ├── test_selfdoubt.py
│   ├── test_cascade.py
│   ├── test_verbalized.py
│   ├── test_trace_parser.py
│   ├── test_mi_proxy.py
│   └── test_api.py
├── grafana/
│   └── dashboards/
│       └── reasonguard.json     # Pre-built Grafana dashboard
├── prometheus/
│   └── prometheus.yml           # Prometheus scrape config
└── examples/
    ├── basic_usage.py
    └── batch_eval.py
```

---

## API Endpoints

### `POST /v1/chat/completions`
Drop-in replacement for OpenAI chat completions. Forwards request to upstream LLM, intercepts response, computes scores, returns augmented response.

**Added to response:**
```json
{
  "reasonguard": {
    "reliable": true,
    "tier": 1,
    "hvr": 0.0,
    "hedge_count": 0,
    "verify_count": 0,
    "verbalized_confidence": 0.95,
    "selfdoubt_score": 1.82,
    "decision": "ACCEPT",
    "mi_estimate": null
  }
}
```

### `POST /score`
Score a pre-existing reasoning trace directly (no LLM call).

**Request:**
```json
{
  "reasoning_trace": "<the model's thinking text>",
  "final_answer": "The answer is (B). Confidence: 95%",
  "model": "qwen3-4b"
}
```

**Response:**
```json
{
  "reliable": true,
  "tier": 1,
  "hvr": 0.0,
  "hedge_count": 0,
  "verify_count": 3,
  "verbalized_confidence": 0.95,
  "selfdoubt_score": 1.82,
  "decision": "ACCEPT"
}
```

### `GET /health`
Health check.

### `GET /metrics`
Prometheus metrics endpoint.

### `POST /calibrate`
Run calibration: send 90 unlabeled traces, compute μ/σ for HVR and verbalized confidence per model. Stores calibration stats in Redis.

---

## Implementation Details

### Marker Matching (hvr.py)
```python
import re
from typing import Dict, List, Tuple

def build_marker_patterns(markers: List[str]) -> re.Pattern:
    """Build compiled regex from marker list.
    Sort by length descending for longest-match-first.
    Use word boundaries, case-insensitive.
    """
    sorted_markers = sorted(markers, key=len, reverse=True)
    escaped = [re.escape(m) for m in sorted_markers]
    pattern = r'\b(?:' + '|'.join(escaped) + r')\b'
    return re.compile(pattern, re.IGNORECASE)

def count_markers(text: str, pattern: re.Pattern) -> int:
    """Count non-overlapping marker matches."""
    return len(pattern.findall(text))

def compute_hvr(trace: str, hedge_pattern: re.Pattern, verify_pattern: re.Pattern) -> Tuple[float, int, int]:
    """Compute Hedge-to-Verify Ratio.
    Returns (hvr, hedge_count, verify_count)
    """
    h = count_markers(trace, hedge_pattern)
    v = count_markers(trace, verify_pattern)
    hvr = h / (v + 1)
    return hvr, h, v
```

### Verbalized Confidence Parser (verbalized.py)
```python
import re

def parse_verbalized_confidence(text: str) -> float | None:
    """Extract verbalized confidence from model response.
    
    Looks for patterns like:
    - "Confidence: 95%"
    - "Confidence: [95]%"  
    - "confidence = 95%"
    
    Returns float in [0, 1] or None if not found.
    
    Per the paper: parse from the answer region (after last </think> tag).
    Use strict parser first, then fallback.
    """
    # Get answer region (after last </think>)
    think_split = text.rsplit('</think>', 1)
    answer_region = think_split[-1] if len(think_split) > 1 else text
    
    # Strict parser: "Confidence: X%" or "Confidence= X%"
    strict = re.findall(r'[Cc]onfidence[:\s=]+\[?(\d+(?:\.\d+)?)\]?\s*%', answer_region)
    if strict:
        val = float(strict[-1])  # last valid match
        if 0 <= val <= 100:
            return val / 100.0
    
    # Fallback: any percentage in answer region
    fallback = re.findall(r'(\d+(?:\.\d+)?)\s*%', answer_region)
    if fallback:
        val = float(fallback[-1])
        if 0 <= val <= 100:
            return val / 100.0
    
    return None
```

### Trace Parser (trace_parser.py)
```python
def extract_reasoning_trace(response_text: str) -> str | None:
    """Extract reasoning trace from various formats:
    - <think>...</think> tags (DeepSeek, Qwen, etc.)
    - Chain-of-thought before final answer
    - Thought summaries from commercial APIs
    """
    # Try <think> tags first
    import re
    think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()
    
    # Try reasoning field if structured response
    # Fallback: return full text (for CoT-style responses)
    return response_text
```

### SELFDOUBT Score (selfdoubt.py)
```python
def compute_selfdoubt_score(
    hvr: float,
    verbalized_conf: float,
    mu_hvr: float,
    sigma_hvr: float, 
    mu_verb: float,
    sigma_verb: float
) -> float:
    """Compute full SELFDOUBT score via z-score fusion.
    
    score = z(-HVR) + z(Verb)
    
    Higher score = more reliable/confident.
    """
    z_hvr = (-hvr - mu_hvr) / (sigma_hvr + 1e-8)
    z_verb = (verbalized_conf - mu_verb) / (sigma_verb + 1e-8)
    return z_hvr + z_verb
```

### Cascade Decision (cascade.py)
```python
from enum import Enum

class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    DEFER = "DEFER"

def cascade_decision(
    hvr: float,
    selfdoubt_score: float,
    tau: float = 0.0,
    hvr_gate_enabled: bool = True
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
```

---

## Tech Stack & Dependencies

```toml
[project]
name = "reasonguard"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.115",
    "uvicorn[standard]>=0.30",
    "httpx>=0.27",              # async HTTP client for proxying
    "pydantic>=2.0",
    "pydantic-settings>=2.0",
    "redis[hiredis]>=5.0",      # caching + calibration storage
    "prometheus-client>=0.21",  # metrics
    "sentence-transformers>=3.0", # MI proxy embeddings (optional)
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "pytest-httpx>=0.30",
    "ruff>=0.6",
]
```

---

## Docker Setup

### docker-compose.yml
Services:
1. **reasonguard** — FastAPI app on port 8000
2. **redis** — Redis 7 on port 6379
3. **prometheus** — Prometheus on port 9090
4. **grafana** — Grafana on port 3000

### Environment Variables
```env
UPSTREAM_BASE_URL=https://api.openai.com/v1     # or any OpenAI-compatible API
UPSTREAM_API_KEY=sk-...                          # API key for upstream
REDIS_URL=redis://redis:6379/0
DEFAULT_TAU=0.0                                  # cascade threshold
HVR_GATE_ENABLED=true
ENABLE_MI_PROXY=false                            # opt-in, requires GPU/embeddings
```

---

## Prometheus Metrics

```python
# metrics.py
from prometheus_client import Counter, Histogram, Gauge

REQUESTS_TOTAL = Counter('reasonguard_requests_total', 'Total requests', ['endpoint'])
SCORING_LATENCY = Histogram('reasonguard_scoring_seconds', 'Scoring latency')
HVR_VALUES = Histogram('reasonguard_hvr', 'HVR distribution', buckets=[0, 0.1, 0.5, 1, 2, 5, 10])
DECISIONS = Counter('reasonguard_decisions_total', 'Cascade decisions', ['decision', 'tier'])
TIER1_COVERAGE = Gauge('reasonguard_tier1_coverage', 'Fraction of requests accepted by Tier 1')
SELFDOUBT_SCORES = Histogram('reasonguard_selfdoubt_score', 'SELFDOUBT score distribution')
CACHE_HITS = Counter('reasonguard_cache_hits_total', 'Redis cache hits')
CACHE_MISSES = Counter('reasonguard_cache_misses_total', 'Redis cache misses')
```

---

## Grafana Dashboard

Pre-built dashboard showing:
- Requests per second
- HVR distribution histogram
- ACCEPT vs DEFER ratio over time
- Tier 1 vs Tier 2 coverage
- SELFDOUBT score distribution
- Scoring latency p50/p95/p99
- Cache hit rate
- Per-model reliability drift (if multiple models used)

---

## Tests

### test_hvr.py — Critical test cases:
```python
def test_no_hedge_markers():
    """Trace with no hedge words → HVR = 0"""
    trace = "The answer is clearly B. Let me verify by substituting back."
    hvr, h, v = compute_hvr(trace, hedge_pattern, verify_pattern)
    assert hvr == 0.0
    assert h == 0

def test_high_uncertainty():
    """Trace with many hedge words, no verify → high HVR"""
    trace = "Maybe the answer is A. Perhaps B. Possibly C. I'm not sure, probably A."
    hvr, h, v = compute_hvr(trace, hedge_pattern, verify_pattern)
    assert hvr > 1.0

def test_balanced_trace():
    """Hedge words offset by verify words → moderate HVR"""
    trace = "Maybe it's A. Let me check... perhaps B. Let me verify the calculation."
    hvr, h, v = compute_hvr(trace, hedge_pattern, verify_pattern)
    assert 0 < hvr < 2.0

def test_word_boundary():
    """'check' inside 'checkout' should NOT match"""
    trace = "I'll go to checkout now."
    _, h, v = compute_hvr(trace, hedge_pattern, verify_pattern)
    assert v == 0
```

### test_verbalized.py — Critical test cases:
```python
def test_standard_format():
    assert parse_verbalized_confidence("\\boxed{B} Confidence: 95%") == 0.95

def test_after_think_tag():
    text = "<think>maybe it's A</think>The answer is B. Confidence: 90%"
    assert parse_verbalized_confidence(text) == 0.90

def test_no_confidence():
    assert parse_verbalized_confidence("The answer is B.") is None
```

---

## README Template

Include:
- What ReasonGuard does (1 paragraph)
- Quick start (docker-compose up)
- API usage examples with curl
- How scoring works (brief explanation of HVR + SELFDOUBT)
- Configuration reference
- Citation of the two papers
- License (MIT)

---

## Implementation Order (for Claude Code)

1. **Scaffold**: pyproject.toml, project structure, config.py, models.py
2. **Core scoring**: markers.py → hvr.py → verbalized.py → selfdoubt.py → cascade.py
3. **Tests for scoring**: test_hvr.py, test_markers.py, test_verbalized.py, test_cascade.py
4. **Trace parser**: trace_parser.py + tests
5. **FastAPI app**: main.py, proxy.py, /score endpoint, /health
6. **Redis cache**: cache.py
7. **Calibration**: calibration.py + /calibrate endpoint
8. **Prometheus metrics**: metrics.py + /metrics endpoint
9. **Docker**: Dockerfile, docker-compose.yml, prometheus config, grafana dashboard
10. **MI proxy** (optional/phase 2): mi_proxy.py
11. **CI**: GitHub Actions workflow
12. **README + examples**
13. **Git init + push to GitHub**

---

## Key Paper References

- **SELFDOUBT** (arXiv:2604.06389): Sections 3.1-3.4 for marker discovery + HVR formula, Section 6 for deployment cascade, Appendix O (Algorithm 1 & 2) for pseudocode, Table 21 for seed words
- **RAGEN-2** (arXiv:2604.06268): Section 2.2-2.3 for MI decomposition + proxy family, Table 1 for MI proxy variants

## Notes for Claude Code

- The SELFDOUBT paper has a public GitHub repo: https://github.com/satwik2711/SelfDoubt — you can reference it for marker dictionaries
- For the MI proxy, start with the simple embedding-similarity approach. The full RAGEN-2 cross-scoring method requires model internals we won't have in production
- The paper uses BAAI/bge-m3 for embeddings in marker discovery — you can use the same or any sentence-transformer for the MI proxy
- Keep the codebase clean and well-documented — this is a portfolio project
- All scoring must be synchronous and fast (target <12ms for HVR + SELFDOUBT on a single trace)
