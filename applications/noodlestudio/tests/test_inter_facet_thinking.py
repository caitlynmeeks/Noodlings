# ──────────────────────────────────────────────────────────────
#
#   Tests for Inter-Facet Thinking Cleanup (Commit C)
#
#   Verifies that thinking tags from upstream facets (e.g., Perception)
#   don't contaminate downstream facets (e.g., Response).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_inter_facet_thinking
# PURPOSE:  Verify thinking tag stripping at facet boundaries
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


class TestInterFacetThinkingCleanup:
    """Verify thinking tags are stripped from all facet LLM outputs."""

    def test_strip_thinking_tags_public_method(self):
        """strip_thinking_tags() is accessible as a public method on HeadlessLLMClient."""
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        client = HeadlessLLMClient(LLMConfig())

        text = "<think>I need to analyze this carefully</think>The weather is sunny."
        result = client.strip_thinking_tags(text)
        assert result == "The weather is sunny."

    def test_multiple_tag_variants_stripped(self):
        """All thinking tag variants are stripped."""
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        client = HeadlessLLMClient(LLMConfig())

        variants = [
            ("<think>reasoning</think>Answer.", "Answer."),
            ("<thinking>deep thought</thinking>Result.", "Result."),
            ("<reasoning>step by step</reasoning>Output.", "Output."),
            ("<reflection>hmm</reflection>Done.", "Done."),
            ("<inner_thoughts>ponder</inner_thoughts>Yes.", "Yes."),
            ("<analysis>data</analysis>Conclusion.", "Conclusion."),
        ]

        for input_text, expected in variants:
            result = client.strip_thinking_tags(input_text)
            assert result == expected, f"Failed for: {input_text}"

    def test_thinking_tag_filter_always_suppresses(self):
        """ThinkingTagFilter with suppress=True discards thinking content."""
        from noodlestudio.runtime.llm_client import ThinkingTagFilter

        # This is the mode used by generate_stream (always True now)
        f = ThinkingTagFilter(suppress=True)
        result = f.feed("<think>I should analyze the mood here</think>The character feels happy.")
        result += f.flush()
        assert "analyze the mood" not in result
        assert "The character feels happy." in result

    def test_chained_facet_output_clean(self):
        """Simulates Perception facet output with thinking tags fed to Response facet."""
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        client = HeadlessLLMClient(LLMConfig())

        # Perception facet output (LLM response with thinking tags)
        perception_output = (
            "<think>The user seems curious about marine biology. "
            "I should focus on the exploratory aspects.</think>"
            "The user is asking about ocean creatures with a sense of wonder."
        )

        # After executor stripping, this is what Response facet would receive
        cleaned = client.strip_thinking_tags(perception_output)
        assert "<think>" not in cleaned
        assert "</think>" not in cleaned
        assert "ocean creatures" in cleaned
        assert "should focus" not in cleaned
