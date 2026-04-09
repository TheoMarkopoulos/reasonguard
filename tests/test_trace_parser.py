from src.reasonguard.trace_parser import extract_reasoning_trace, split_trace_and_answer


class TestExtractReasoningTrace:
    def test_single_think_tag(self):
        text = "<think>Step 1: consider A. Step 2: verify B.</think>The answer is B."
        trace = extract_reasoning_trace(text)
        assert trace == "Step 1: consider A. Step 2: verify B."

    def test_multiple_think_tags(self):
        text = "<think>first thought</think>middle<think>second thought</think>answer"
        trace = extract_reasoning_trace(text)
        assert "first thought" in trace
        assert "second thought" in trace

    def test_multiline_think_tag(self):
        text = "<think>\nLine 1\nLine 2\nLine 3\n</think>Answer."
        trace = extract_reasoning_trace(text)
        assert "Line 1" in trace
        assert "Line 3" in trace

    def test_empty_think_tag(self):
        text = "<think>   </think>The answer is B."
        trace = extract_reasoning_trace(text)
        # Empty think block with only whitespace — fallback to full text
        assert trace is not None

    def test_reasoning_tags(self):
        text = "<reasoning>My analysis here.</reasoning>Final: B."
        trace = extract_reasoning_trace(text)
        assert trace == "My analysis here."

    def test_plain_text_fallback(self):
        text = "I think the answer is B because of X and Y."
        trace = extract_reasoning_trace(text)
        assert trace == text

    def test_empty_string(self):
        assert extract_reasoning_trace("") is None

    def test_whitespace_only(self):
        assert extract_reasoning_trace("   \n  ") is None

    def test_none_not_returned_for_content(self):
        assert extract_reasoning_trace("any content") is not None


class TestSplitTraceAndAnswer:
    def test_think_tag_split(self):
        text = "<think>reasoning here</think>The answer is B."
        trace, answer = split_trace_and_answer(text)
        assert trace == "reasoning here"
        assert answer == "The answer is B."

    def test_multiple_think_tags_answer_after_last(self):
        text = "<think>thought 1</think>interim<think>thought 2</think>Final answer."
        trace, answer = split_trace_and_answer(text)
        assert "thought 1" in trace
        assert "thought 2" in trace
        assert answer == "Final answer."

    def test_reasoning_tag_split(self):
        text = "<reasoning>analysis</reasoning>Result: C."
        trace, answer = split_trace_and_answer(text)
        assert trace == "analysis"
        assert answer == "Result: C."

    def test_no_tags(self):
        text = "The answer is B because X."
        trace, answer = split_trace_and_answer(text)
        assert trace == text
        assert answer == text

    def test_empty_string(self):
        trace, answer = split_trace_and_answer("")
        assert trace is None
        assert answer == ""

    def test_think_tag_no_answer_after(self):
        text = "<think>all reasoning no answer</think>"
        trace, answer = split_trace_and_answer(text)
        assert trace == "all reasoning no answer"
        # When nothing after </think>, falls back to full text
        assert answer == text

    def test_whitespace_only_answer(self):
        text = "<think>reasoning</think>   "
        trace, answer = split_trace_and_answer(text)
        assert trace == "reasoning"
        # Whitespace-only answer falls back to full text
        assert answer == text
