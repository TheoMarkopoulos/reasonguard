import re


def extract_reasoning_trace(response_text: str) -> str | None:
    """Extract reasoning trace from various formats:
    - <think>...</think> tags (DeepSeek, Qwen, etc.)
    - Chain-of-thought before final answer
    - Thought summaries from commercial APIs

    Returns the reasoning trace text, or the full text as fallback.
    Returns None only if input is empty.
    """
    if not response_text or not response_text.strip():
        return None

    # Try <think> tags first — collect all think blocks
    think_matches = re.findall(r"<think>(.*?)</think>", response_text, re.DOTALL)
    if think_matches:
        return "\n".join(m.strip() for m in think_matches if m.strip())

    # Try <reasoning>...</reasoning> tags (some APIs use this)
    reasoning_matches = re.findall(
        r"<reasoning>(.*?)</reasoning>", response_text, re.DOTALL
    )
    if reasoning_matches:
        return "\n".join(m.strip() for m in reasoning_matches if m.strip())

    # Fallback: return full text (for CoT-style responses without tags)
    return response_text


def split_trace_and_answer(response_text: str) -> tuple[str | None, str]:
    """Split response into (reasoning_trace, final_answer).

    For tagged formats, the trace is inside tags and the answer is outside.
    For plain text, trace is the full text and answer is also the full text.
    """
    if not response_text or not response_text.strip():
        return None, response_text

    # Try <think> tags — answer is everything after the last </think>
    think_split = response_text.rsplit("</think>", 1)
    if len(think_split) > 1:
        trace = extract_reasoning_trace(response_text)
        answer = think_split[-1].strip()
        return trace, answer if answer else response_text

    # Try <reasoning> tags
    reasoning_split = response_text.rsplit("</reasoning>", 1)
    if len(reasoning_split) > 1:
        trace = extract_reasoning_trace(response_text)
        answer = reasoning_split[-1].strip()
        return trace, answer if answer else response_text

    # No tags — full text is both trace and answer
    return response_text, response_text
