# ──────────────────────────────────────────────────────────────
#
#   Tests for Streaming Delivery Modes (Commit 2)
#
#   Facet delivery field, PerformancePlayer streaming methods,
#   ThinkingTagFilter state machine.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   tests.test_streaming_delivery
# PURPOSE:  Verify streaming delivery mode infrastructure
# LAYER:    Tests
# ──────────────────────────────────────────────────────────────

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest


# =============================================================================
# Facet delivery field
# =============================================================================

class TestFacetDeliveryField:
    """Verify delivery field on Facet dataclass."""

    def _make_facet(self, **kwargs):
        from noodlestudio.core.facet_system import Facet
        defaults = {
            'id': 'test_facet',
            'name': 'Test',
            'facet_type': 'LLMFacet',
            'prompt': 'Test prompt',
        }
        defaults.update(kwargs)
        return Facet(**defaults)

    def test_facet_delivery_default_buffered(self):
        """Facet().delivery == 'buffered'."""
        facet = self._make_facet()
        assert facet.delivery == "buffered"

    def test_facet_delivery_to_dict_omits_default(self):
        """to_dict() omits delivery when it's the default 'buffered'."""
        facet = self._make_facet()
        d = facet.to_dict()
        assert 'delivery' not in d

    def test_facet_delivery_to_dict_includes_non_default(self):
        """to_dict() includes delivery when set to stream_animated."""
        facet = self._make_facet(delivery='stream_animated')
        d = facet.to_dict()
        assert d['delivery'] == 'stream_animated'

    def test_facet_delivery_from_dict(self):
        """from_dict() reads delivery field."""
        from noodlestudio.core.facet_system import Facet
        data = {
            'id': 'f1', 'name': 'F1', 'type': 'LLMFacet',
            'prompt': 'test', 'delivery': 'stream_raw'
        }
        facet = Facet.from_dict(data)
        assert facet.delivery == 'stream_raw'

    def test_facet_delivery_from_dict_missing(self):
        """Defaults to 'buffered' when delivery absent from dict."""
        from noodlestudio.core.facet_system import Facet
        data = {
            'id': 'f2', 'name': 'F2', 'type': 'LLMFacet',
            'prompt': 'test'
        }
        facet = Facet.from_dict(data)
        assert facet.delivery == 'buffered'


# =============================================================================
# PerformancePlayer streaming mode
# =============================================================================

class TestPerformancePlayerStreaming:
    """Verify PerformancePlayer streaming methods."""

    # Hold references to prevent Qt GC
    _app = None
    _players = []

    def _make_player(self):
        from PyQt6.QtCore import QCoreApplication, QObject
        import sys
        app = QCoreApplication.instance()
        if app is None:
            TestPerformancePlayerStreaming._app = QCoreApplication(sys.argv)
            app = TestPerformancePlayerStreaming._app

        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        # Parent to a holder QObject to prevent GC
        holder = QObject()
        player = PerformancePlayer(parent=holder)
        TestPerformancePlayerStreaming._players.append((holder, player))
        return player

    def test_player_streaming_basic(self):
        """start_streaming + append_text + finish = all chars emitted."""
        player = self._make_player()

        revealed_chars = []
        player.characterRevealed.connect(lambda c: revealed_chars.append(c))

        finished_count = []
        player.finished.connect(lambda: finished_count.append(1))

        player.start_streaming()
        player.append_text("Hello")  # 5 chars, >= 3 threshold -> starts

        # Process events to let QTimer fire
        from PyQt6.QtCore import QCoreApplication
        import time
        deadline = time.time() + 2.0
        while len(revealed_chars) < 5 and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        player.finish_streaming()

        # Drain remaining
        deadline = time.time() + 2.0
        while not finished_count and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        assert "".join(revealed_chars) == "Hello"

    def test_player_streaming_punctuation_delays(self):
        """Period delay > base delay."""
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        assert PerformancePlayer._char_delay('.') > PerformancePlayer._char_delay('a')
        assert PerformancePlayer._char_delay('!') > PerformancePlayer._char_delay('a')

    def test_player_streaming_buffer_stall(self):
        """Buffer empty + stream ongoing = waits, doesn't crash."""
        player = self._make_player()

        revealed_chars = []
        player.characterRevealed.connect(lambda c: revealed_chars.append(c))

        player.start_streaming()
        # Append just 1 char -- below threshold, should NOT start yet
        player.append_text("A")

        from PyQt6.QtCore import QCoreApplication
        import time
        # Process events briefly -- should not crash
        deadline = time.time() + 0.2
        while time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        # Now add more to cross threshold
        player.append_text("BC")
        deadline = time.time() + 2.0
        while len(revealed_chars) < 3 and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        player.finish_streaming()

        deadline = time.time() + 2.0
        while len(revealed_chars) < 3 and time.time() < deadline:
            QCoreApplication.processEvents()
            time.sleep(0.01)

        assert "".join(revealed_chars) == "ABC"


# =============================================================================
# ThinkingTagFilter state machine
# =============================================================================

class TestThinkingTagFilter:
    """Verify ThinkingTagFilter handles tags within and across chunks."""

    def test_thinking_tag_filter_strips(self):
        """ThinkingTagFilter removes <think> blocks."""
        from noodlestudio.runtime.llm_client import ThinkingTagFilter

        f = ThinkingTagFilter(suppress=True)
        result = f.feed("<think>reasoning</think>The answer.")
        result += f.flush()
        assert result == "The answer."

    def test_thinking_tag_filter_passthrough(self):
        """Normal text passes through unmodified."""
        from noodlestudio.runtime.llm_client import ThinkingTagFilter

        f = ThinkingTagFilter(suppress=True)
        result = f.feed("Just normal text here.")
        result += f.flush()
        assert result == "Just normal text here."

    def test_thinking_tag_filter_no_suppress(self):
        """When suppress=False, everything passes through."""
        from noodlestudio.runtime.llm_client import ThinkingTagFilter

        f = ThinkingTagFilter(suppress=False)
        result = f.feed("<think>reasoning</think>The answer.")
        result += f.flush()
        assert result == "<think>reasoning</think>The answer."

    def test_thinking_tag_filter_across_chunks(self):
        """Tags split across chunks are handled correctly."""
        from noodlestudio.runtime.llm_client import ThinkingTagFilter

        f = ThinkingTagFilter(suppress=True)
        # Feed in chunks that split the tag
        r1 = f.feed("Hello <thi")
        r2 = f.feed("nk>internal thoughts</think>World")
        r3 = f.flush()
        result = r1 + r2 + r3
        assert "internal thoughts" not in result
        assert "Hello" in result
        assert "World" in result
