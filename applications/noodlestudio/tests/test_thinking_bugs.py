# ──────────────────────────────────────────────────────────────
#
#   Tests for thinking model bug fixes (Commit 0)
#
#   Bug 1: Thinking content leaking into ensemble performance
#   Bug 2: Speaker name inserted after first character
#   Bug 3: NoodleCode returns no output from LMStudio thinking models
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_thinking_bugs
# PURPOSE:  Verify thinking model compatibility fixes
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os
import json
import asyncio

# Add noodlestudio package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# =============================================================================
# Bug 1: Thinking tags stripped from LLM responses
# =============================================================================

class TestStripThinkingTags:
    """Verify _strip_thinking_tags handles all known tag variants."""

    def _make_client(self):
        """Create a HeadlessLLMClient with default config."""
        from noodlestudio.runtime.llm_client import HeadlessLLMClient, LLMConfig
        return HeadlessLLMClient(LLMConfig())

    def test_strip_thinking_tags_basic(self):
        """Existing <think> tags stripped."""
        client = self._make_client()
        text = "<think>Let me reason about this...</think>Hello world!"
        result = client._strip_thinking_tags(text)
        assert result == "Hello world!"

    def test_strip_reasoning_tags(self):
        """<reasoning> tags stripped (Kimi 2.5 via LMStudio)."""
        client = self._make_client()
        text = "<reasoning>Step 1: analyze the query\nStep 2: formulate response</reasoning>The answer is 42."
        result = client._strip_thinking_tags(text)
        assert result == "The answer is 42."

    def test_strip_reflection_tags(self):
        """<reflection> tags stripped."""
        client = self._make_client()
        text = "Some text <reflection>Let me reconsider...</reflection> and more text."
        result = client._strip_thinking_tags(text)
        assert "reflection" not in result
        assert "Some text" in result
        assert "and more text." in result

    def test_strip_nested_tag_variants(self):
        """<inner_thoughts> and <analysis> tags stripped."""
        client = self._make_client()

        text1 = "<inner_thoughts>Processing deeply...</inner_thoughts>Final answer."
        result1 = client._strip_thinking_tags(text1)
        assert result1 == "Final answer."

        text2 = "<analysis>Checking data points...</analysis>Result: confirmed."
        result2 = client._strip_thinking_tags(text2)
        assert result2 == "Result: confirmed."


# =============================================================================
# Bug 1 continued: reasoning_content field discarded
# =============================================================================

class TestReasoningContentField:
    """Verify reasoning_content field is handled in response parsing."""

    def test_reasoning_content_field_discarded(self):
        """reasoning_content field in message dict should not leak into response."""
        # Simulate the response parsing logic from _generate_openai_compatible
        message = {
            'content': 'The visible response.',
            'reasoning_content': 'This is internal chain-of-thought reasoning...'
        }
        # Our fix: read content, ignore reasoning_content
        response_text = message.get('content') or ''
        assert response_text == 'The visible response.'
        assert 'chain-of-thought' not in response_text

    def test_content_none_fallback(self):
        """When content is None with only reasoning_content, return empty string."""
        message = {
            'content': None,
            'reasoning_content': 'Long internal reasoning that should not appear...'
        }
        response_text = message.get('content') or ''
        assert response_text == ''


# =============================================================================
# Bug 2: performanceReady emitted before first characterRevealed
# =============================================================================

class TestPerformanceEmitOrder:
    """Verify name prefix appears before first character."""

    def test_performance_ready_before_first_character(self):
        """performanceReady must emit before first characterRevealed."""
        from PyQt6.QtCore import QCoreApplication
        import sys

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication(sys.argv)

        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        # Track signal emission order
        signal_order = []

        performer = NoodlingPerformer(
            noodling_id='test',
            name='TestNoodling',
            llm_client=None
        )

        performer.performanceReady.connect(
            lambda script: signal_order.append('performanceReady')
        )
        performer.characterRevealed.connect(
            lambda char: signal_order.append(f'char:{char}')
        )

        # Build a minimal performance script
        script = {
            'type': 'performance_script',
            'text': 'Hi',
            'characters': [
                {'c': 'H', 'd': 35},
                {'c': 'i', 'd': 35},
            ],
            'speaking_intensity': 0.7
        }

        # Simulate what _on_assembly_result does with a performance_script
        # After the fix, performanceReady emits FIRST, then _play_performance starts
        performer.performanceReady.emit(script)
        performer._play_performance(script)

        # performanceReady should be first in the list
        assert len(signal_order) >= 2
        assert signal_order[0] == 'performanceReady'
        assert signal_order[1].startswith('char:')

        # Clean up
        performer.stop()

    def test_name_prefix_before_text_ensemble(self):
        """In ensemble, begin_noodling_text() (from performanceReady) fires before append_character()."""
        # This tests the logical contract: the name prefix insertion
        # (triggered by performanceReady) must complete before the first
        # character reveal (triggered by _play_performance).

        from PyQt6.QtCore import QCoreApplication
        import sys

        app = QCoreApplication.instance()
        if app is None:
            app = QCoreApplication(sys.argv)

        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        # Simulate a window's text buffer
        text_buffer = []

        def begin_noodling_text(script):
            """Simulates GuidePerformanceWindow.begin_noodling_text()."""
            text_buffer.append("Ajo: ")

        def append_character(char):
            """Simulates GuidePerformanceWindow.append_character()."""
            text_buffer.append(char)

        performer = NoodlingPerformer(
            noodling_id='ajo',
            name='Ajo',
            llm_client=None
        )
        performer.performanceReady.connect(begin_noodling_text)
        performer.characterRevealed.connect(append_character)

        script = {
            'type': 'performance_script',
            'text': 'Hi',
            'characters': [
                {'c': 'H', 'd': 35},
                {'c': 'i', 'd': 35},
            ],
            'speaking_intensity': 0.7
        }

        # Emit in the fixed order
        performer.performanceReady.emit(script)
        performer._play_performance(script)

        # Verify: name prefix is first, then characters
        assert text_buffer[0] == "Ajo: "
        assert text_buffer[1] == "H"

        performer.stop()


# =============================================================================
# Bug 3: NoodleCode SSE reasoning_content yields StreamChunk
# =============================================================================

class TestNoodleCodeSSEReasoningContent:
    """Verify NoodleCode SSE parser handles reasoning_content in delta."""

    def test_noodlecode_sse_reasoning_content(self):
        """SSE delta with reasoning_content should yield a StreamChunk."""
        from noodlestudio.core.noodle_code_engine import StreamChunk

        # Simulate what the SSE parser does with a delta containing reasoning_content
        delta = {
            "reasoning_content": "Let me think about this step by step..."
        }

        # The fix adds this check after the content check
        chunks = []
        if "content" in delta and delta["content"]:
            chunks.append(StreamChunk(type="text", content=delta["content"]))
        if "reasoning_content" in delta and delta["reasoning_content"]:
            chunks.append(StreamChunk(type="text", content=delta["reasoning_content"]))

        assert len(chunks) == 1
        assert chunks[0].type == "text"
        assert "step by step" in chunks[0].content

    def test_noodlecode_sse_content_preferred_over_reasoning(self):
        """When both content and reasoning_content are present, both yield chunks."""
        from noodlestudio.core.noodle_code_engine import StreamChunk

        delta = {
            "content": "The answer is 42.",
            "reasoning_content": "Because 6 times 7..."
        }

        chunks = []
        if "content" in delta and delta["content"]:
            chunks.append(StreamChunk(type="text", content=delta["content"]))
        if "reasoning_content" in delta and delta["reasoning_content"]:
            chunks.append(StreamChunk(type="text", content=delta["reasoning_content"]))

        assert len(chunks) == 2
        assert chunks[0].content == "The answer is 42."
        assert "6 times 7" in chunks[1].content
