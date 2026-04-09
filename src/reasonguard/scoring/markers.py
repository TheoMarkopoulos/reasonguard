import re

# Default hedge seeds (top-10 from paper Table 21)
HEDGE_SEEDS: list[str] = [
    "possibly",
    "seemingly",
    "maybe",
    "apparently",
    "probably",
    "presumably",
    "perhaps",
    "likely",
    "reportedly",
    "seems",
]

# Extended hedge n-grams (from paper Figure 2 examples)
HEDGE_EXTENDED: list[str] = [
    "might be",
    "not sure",
    "hmm",
    "so perhaps",
    "so maybe",
    "plausible",
]

# Default verify seeds (top-10 from paper Table 21)
VERIFY_SEEDS: list[str] = [
    "check",
    "reassess",
    "reevaluate",
    "re-evaluate",
    "reinspect",
    "rechecking",
    "verifying",
    "reconfirming",
    "recheck",
    "prove",
]

# Extended verify n-grams (from paper Figure 2 examples)
VERIFY_EXTENDED: list[str] = [
    "let me check",
    "lets check",
    "me verify",
    "let me verify",
    "examine",
    "evaluate",
    "compute",
    "determine",
    "confirm",
    "recalculate",
    "substitute back",
]

DEFAULT_HEDGE_MARKERS: list[str] = HEDGE_SEEDS + HEDGE_EXTENDED
DEFAULT_VERIFY_MARKERS: list[str] = VERIFY_SEEDS + VERIFY_EXTENDED


def build_marker_pattern(markers: list[str]) -> re.Pattern:
    """Build compiled regex from marker list.
    Sort by length descending for longest-match-first.
    Use word boundaries, case-insensitive.
    """
    if not markers:
        return re.compile(r"(?!x)x", re.IGNORECASE)  # matches nothing
    sorted_markers = sorted(markers, key=len, reverse=True)
    escaped = [re.escape(m) for m in sorted_markers]
    pattern = r"\b(?:" + "|".join(escaped) + r")\b"
    return re.compile(pattern, re.IGNORECASE)


def count_markers(text: str, pattern: re.Pattern) -> int:
    """Count non-overlapping marker matches."""
    return len(pattern.findall(text))


DEFAULT_HEDGE_PATTERN: re.Pattern = build_marker_pattern(DEFAULT_HEDGE_MARKERS)
DEFAULT_VERIFY_PATTERN: re.Pattern = build_marker_pattern(DEFAULT_VERIFY_MARKERS)
