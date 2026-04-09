[![CI](https://github.com/theomarkopoulos/reasonguard/actions/workflows/ci.yml/badge.svg)](https://github.com/theomarkopoulos/reasonguard/actions/workflows/ci.yml)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# ReasonGuard

ReasonGuard is a production-ready FastAPI middleware that scores the reliability of any reasoning LLM's output in a single pass, with zero access to model internals. It intercepts reasoning traces, computes a Hedge-to-Verify Ratio (HVR) and SELFDOUBT confidence score based on linguistic markers, and returns an ACCEPT/DEFER decision through a two-tier deployment cascade — giving you a free, high-precision confidence gate on top of any OpenAI-compatible API.

## Quick Start

```bash
# Clone and configure
git clone https://github.com/theomarkopoulos/reasonguard.git
cd reasonguard
cp .env.example .env  # edit with your upstream API key

# Start all services
docker-compose up -d
```

This starts ReasonGuard on port 8000, Redis on 6379, Prometheus on 9090, and Grafana on 3000.

## API Usage

### Score a reasoning trace directly

```bash
curl -X POST http://localhost:8000/score \
  -H "Content-Type: application/json" \
  -d '{
    "reasoning_trace": "The power rule gives 2x. Let me verify by checking the limit definition.",
    "final_answer": "The derivative is 2x. Confidence: 95%",
    "model": "qwen3-4b"
  }'
```

Response:

```json
{
  "reliable": true,
  "tier": 1,
  "hvr": 0.0,
  "hedge_count": 0,
  "verify_count": 1,
  "verbalized_confidence": 0.95,
  "selfdoubt_score": 1.82,
  "decision": "ACCEPT"
}
```

### Drop-in proxy for OpenAI-compatible APIs

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-upstream-key" \
  -d '{
    "model": "qwen3-4b",
    "messages": [{"role": "user", "content": "What is the derivative of x^2?"}]
  }'
```

The response includes the standard OpenAI format plus a `reasonguard` object with reliability scoring.

### Health check

```bash
curl http://localhost:8000/health
```

## How HVR Scoring Works

ReasonGuard implements the SELFDOUBT scoring pipeline:

1. **Marker Detection** — The reasoning trace is scanned for **hedge markers** (e.g., "maybe", "probably", "not sure") and **verification markers** (e.g., "let me check", "verify", "recalculate") using word-boundary regex matching.

2. **HVR Computation** — The Hedge-to-Verify Ratio is calculated as `HVR = h / (v + 1)`, where `h` is the hedge count and `v` is the verify count. Lower HVR indicates more reliable reasoning.

3. **SELFDOUBT Score** — Combines HVR with verbalized confidence (parsed from the model's response) via z-score fusion: `score = z(-HVR) + z(Verb)`, using calibration statistics from unlabeled traces.

4. **Two-Tier Cascade**:
   - **Tier 1**: If `HVR = 0` (no hedging at all), ACCEPT immediately — this is correct 96.1% of the time across 7 models and 3 datasets.
   - **Tier 2**: If HVR > 0, compare the calibrated SELFDOUBT score against threshold `tau`. Score >= tau → ACCEPT, otherwise DEFER.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE_URL` | `https://api.openai.com/v1` | Base URL of the upstream LLM API |
| `UPSTREAM_API_KEY` | *(empty)* | API key for the upstream LLM |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `DEFAULT_TAU` | `0.0` | Cascade threshold (higher = more conservative) |
| `HVR_GATE_ENABLED` | `true` | Enable Tier 1 HVR=0 gate |
| `ENABLE_MI_PROXY` | `false` | Enable mutual information proxy (requires embeddings) |
| `HOST` | `0.0.0.0` | Server bind host |
| `PORT` | `8000` | Server bind port |

## References

- **SELFDOUBT**: Gupta, S., et al. "SELFDOUBT: Harnessing LLM Self-Reflection for Uncertainty Estimation." *arXiv:2604.06389*, 2025.
- **RAGEN-2**: Chen, Z., et al. "RAGEN-2: Mutual Information Guided Reasoning in Reinforcement Learning." *arXiv:2604.06268*, 2025.

## License

MIT. See [LICENSE](LICENSE).
