"""Basic usage example for the ReasonGuard /score endpoint."""

import httpx

REASONGUARD_URL = "http://localhost:8000"


def score_trace(reasoning_trace: str, final_answer: str, model: str = "qwen3-4b") -> dict:
    """Score a reasoning trace via the /score endpoint."""
    response = httpx.post(
        f"{REASONGUARD_URL}/score",
        json={
            "reasoning_trace": reasoning_trace,
            "final_answer": final_answer,
            "model": model,
        },
    )
    response.raise_for_status()
    return response.json()


def main():
    # Example 1: High-confidence trace (no hedging, has verification)
    result = score_trace(
        reasoning_trace=(
            "The question asks for the derivative of x^2. "
            "The power rule gives 2x. Let me verify by checking the limit definition. "
            "Yes, the derivative is 2x."
        ),
        final_answer="The derivative is 2x. Confidence: 99%",
    )
    print("Example 1 — High confidence:")
    print(f"  Decision: {result['decision']}, Tier: {result['tier']}, HVR: {result['hvr']}")
    print()

    # Example 2: Uncertain trace (heavy hedging)
    result = score_trace(
        reasoning_trace=(
            "Maybe the answer is A. Perhaps it could be B. "
            "I'm not sure, but probably A seems right. "
            "Possibly C is also valid."
        ),
        final_answer="The answer is A. Confidence: 40%",
    )
    print("Example 2 — Uncertain:")
    print(f"  Decision: {result['decision']}, Tier: {result['tier']}, HVR: {result['hvr']}")
    print()

    # Example 3: Balanced trace
    result = score_trace(
        reasoning_trace=(
            "Perhaps the integral is x^3/3. Let me check by differentiating. "
            "d/dx(x^3/3) = x^2. That confirms the answer."
        ),
        final_answer="The integral of x^2 is x^3/3 + C. Confidence: 85%",
    )
    print("Example 3 — Balanced:")
    print(f"  Decision: {result['decision']}, Tier: {result['tier']}, HVR: {result['hvr']}")


if __name__ == "__main__":
    main()
