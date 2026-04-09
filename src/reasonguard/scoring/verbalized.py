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
    think_split = text.rsplit("</think>", 1)
    answer_region = think_split[-1] if len(think_split) > 1 else text

    # Strict parser: "Confidence: X%" or "Confidence= X%"
    strict = re.findall(
        r"[Cc]onfidence[:\s=]+\[?(\d+(?:\.\d+)?)\]?\s*%", answer_region
    )
    if strict:
        val = float(strict[-1])  # last valid match
        if 0 <= val <= 100:
            return val / 100.0

    # Fallback: any percentage in answer region
    fallback = re.findall(r"(\d+(?:\.\d+)?)\s*%", answer_region)
    if fallback:
        val = float(fallback[-1])
        if 0 <= val <= 100:
            return val / 100.0

    return None
