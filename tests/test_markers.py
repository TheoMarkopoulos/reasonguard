import re

from src.reasonguard.scoring.markers import (
    DEFAULT_HEDGE_MARKERS,
    DEFAULT_HEDGE_PATTERN,
    DEFAULT_VERIFY_MARKERS,
    DEFAULT_VERIFY_PATTERN,
    build_marker_pattern,
    count_markers,
)


class TestBuildMarkerPattern:
    def test_returns_compiled_pattern(self):
        pattern = build_marker_pattern(["maybe", "perhaps"])
        assert isinstance(pattern, re.Pattern)

    def test_case_insensitive(self):
        pattern = build_marker_pattern(["maybe"])
        assert count_markers("Maybe MAYBE maybe", pattern) == 3

    def test_word_boundary(self):
        pattern = build_marker_pattern(["check"])
        assert count_markers("check this", pattern) == 1
        assert count_markers("checkout counter", pattern) == 0
        assert count_markers("recheck it", pattern) == 0
        assert count_markers("double-check", pattern) == 1

    def test_longest_match_first(self):
        pattern = build_marker_pattern(["let me check", "check"])
        # "let me check" should match as one, not also trigger "check"
        assert count_markers("let me check", pattern) == 1

    def test_ngram_matching(self):
        pattern = build_marker_pattern(["not sure", "might be"])
        assert count_markers("I'm not sure if it might be correct", pattern) == 2

    def test_empty_text(self):
        pattern = build_marker_pattern(["maybe"])
        assert count_markers("", pattern) == 0

    def test_empty_markers(self):
        pattern = build_marker_pattern([])
        assert count_markers("maybe perhaps", pattern) == 0

    def test_no_matches(self):
        pattern = build_marker_pattern(["maybe"])
        assert count_markers("The answer is definitely B.", pattern) == 0


class TestDefaultDictionaries:
    def test_hedge_markers_nonempty(self):
        assert len(DEFAULT_HEDGE_MARKERS) >= 10

    def test_verify_markers_nonempty(self):
        assert len(DEFAULT_VERIFY_MARKERS) >= 10

    def test_no_overlap_between_hedge_and_verify(self):
        hedge_set = {m.lower() for m in DEFAULT_HEDGE_MARKERS}
        verify_set = {m.lower() for m in DEFAULT_VERIFY_MARKERS}
        assert hedge_set.isdisjoint(verify_set)

    def test_default_patterns_compile(self):
        assert isinstance(DEFAULT_HEDGE_PATTERN, re.Pattern)
        assert isinstance(DEFAULT_VERIFY_PATTERN, re.Pattern)

    def test_hedge_seeds_all_match(self):
        for marker in DEFAULT_HEDGE_MARKERS:
            assert count_markers(marker, DEFAULT_HEDGE_PATTERN) == 1, f"'{marker}' not matched"

    def test_verify_seeds_all_match(self):
        for marker in DEFAULT_VERIFY_MARKERS:
            assert count_markers(marker, DEFAULT_VERIFY_PATTERN) == 1, f"'{marker}' not matched"
