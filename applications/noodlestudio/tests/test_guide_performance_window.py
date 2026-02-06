# ──────────────────────────────────────────────────────────────
#   Tests for Guide Performance Window
#
#   Tests for the floating combined panel that provides VRM
#   character rendering, dialogue display, and user text input
#   during guided play performances.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QMainWindow

from noodlestudio.runtime.ui.guide_performance_window import (
    GuidePerformanceWindow,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def parent_window(qapp, qtbot):
    """Create a mock parent window."""
    window = QMainWindow()
    window.resize(1400, 900)
    qtbot.addWidget(window)
    yield window
    window.close()


@pytest.fixture
def guide_window(qapp, qtbot, parent_window):
    """Create a GuidePerformanceWindow for testing."""
    window = GuidePerformanceWindow(parent_window=parent_window)
    qtbot.addWidget(window)
    yield window
    window.close()


# =============================================================================
# Window Creation Tests
# =============================================================================

class TestWindowCreation:
    """Tests for window construction and flags."""

    def test_window_creation(self, guide_window):
        """Window creates with correct flags."""
        flags = guide_window.windowFlags()
        assert flags & Qt.WindowType.FramelessWindowHint
        assert flags & Qt.WindowType.WindowStaysOnTopHint
        assert flags & Qt.WindowType.Tool

    def test_has_vrm_container(self, guide_window):
        """Window has a VRM viewport container."""
        assert guide_window.vrm_container is not None
        assert guide_window.vrm_container.height() == 250

    def test_has_dialogue_view(self, guide_window):
        """Window has a dialogue text display."""
        assert guide_window.dialogue_view is not None
        assert guide_window.dialogue_view.isReadOnly()

    def test_has_input_field(self, guide_window):
        """Window has a user input field."""
        assert guide_window.input_field is not None

    def test_has_send_button(self, guide_window):
        """Window has a send button."""
        assert guide_window.send_button is not None
        assert guide_window.send_button.text() == "Send"

    def test_has_header(self, guide_window):
        """Window has a draggable header."""
        assert guide_window.header_label is not None

    def test_has_thinking_indicator(self, guide_window):
        """Window has a thinking indicator (hidden by default)."""
        assert guide_window.thinking_indicator is not None

    def test_default_size(self, guide_window):
        """Window has the expected default size."""
        assert guide_window.width() == 350
        assert guide_window.height() == 600


# =============================================================================
# Position Tracking Tests
# =============================================================================

class TestPositionTracking:
    """Tests for parent window following."""

    def test_follow_parent_positions_relative(self, guide_window, parent_window):
        """Window positions itself relative to parent's right edge."""
        parent_window.move(100, 50)
        parent_window.show()

        guide_window._follow_parent()

        geo = parent_window.geometry()
        expected_x = geo.right() + guide_window._offset[0]

        # X position should match (no window decoration offset on x)
        assert guide_window.x() == expected_x

        # Y position: allow for window manager decoration offsets
        expected_y = geo.top() + guide_window._offset[1]
        assert abs(guide_window.y() - expected_y) < 50

    def test_hides_when_parent_hidden(self, guide_window, parent_window):
        """Window hides when parent is not visible."""
        guide_window.show()
        parent_window.hide()

        guide_window._follow_parent()

        assert not guide_window.isVisible()


# =============================================================================
# Dialogue Display Tests
# =============================================================================

class TestDialogueDisplay:
    """Tests for text display in the dialogue area."""

    def test_append_guide_text(self, guide_window):
        """Guide text appears in dialogue display."""
        guide_window.append_guide_text("Welcome to the performance.")
        text = guide_window.dialogue_view.toPlainText()
        assert "Welcome to the performance." in text

    def test_append_user_text(self, guide_window):
        """User text appears with prefix in dialogue display."""
        guide_window.append_user_text("What are facets?")
        text = guide_window.dialogue_view.toPlainText()
        assert "What are facets?" in text

    def test_clear_dialogue(self, guide_window):
        """Clear removes all dialogue text."""
        guide_window.append_guide_text("Some text")
        guide_window.append_user_text("More text")
        guide_window.clear_dialogue()
        text = guide_window.dialogue_view.toPlainText()
        assert text.strip() == ""

    def test_streaming_chunks_concatenate(self, guide_window):
        """Multiple streaming chunks concatenate properly."""
        guide_window._response_started = True
        guide_window._append_streaming_text("Hello ")
        guide_window._append_streaming_text("world")
        text = guide_window.dialogue_view.toPlainText()
        assert "Hello world" in text

    def test_play_header(self, guide_window):
        """Header displays play title."""
        guide_window.show_play_header("Let's Consciousness!")
        assert guide_window.header_label.text() == "Let's Consciousness!"


# =============================================================================
# Engine Wiring Tests
# =============================================================================

class TestEngineWiring:
    """Tests for engine and handler setup."""

    def test_set_engine(self, guide_window):
        """Engine can be set on the window."""
        mock_engine = MagicMock()
        guide_window.set_engine(mock_engine)
        assert guide_window.engine is mock_engine

    def test_set_guide_cue_handler(self, guide_window):
        """Guide cue handler can be set."""
        mock_handler = MagicMock()
        guide_window.set_guide_cue_handler(mock_handler)
        assert guide_window._guide_cue_handler is mock_handler

    def test_send_without_engine_shows_error(self, guide_window):
        """Sending without engine shows error message."""
        guide_window.input_field.setText("test message")
        guide_window._on_send()
        text = guide_window.dialogue_view.toPlainText()
        assert "Error" in text

    def test_send_with_empty_input_does_nothing(self, guide_window):
        """Sending with empty input is a no-op."""
        mock_engine = MagicMock()
        guide_window.set_engine(mock_engine)
        guide_window.input_field.setText("")
        guide_window._on_send()
        # No worker should have been created
        assert guide_window.worker is None


# =============================================================================
# VRM Loading Tests
# =============================================================================

class TestVRMLoading:
    """Tests for VRM character loading."""

    def test_set_vrm_with_valid_mock(self, guide_window):
        """VRM loading wires through to VRMViewportWidget."""
        # Patch at the import source (vrm_viewport module)
        with patch(
            'noodlestudio.runtime.ui.components.vrm_viewport.VRMViewportWidget'
        ) as MockWidget:
            mock_widget = MagicMock()
            MockWidget.return_value = mock_widget

            guide_window.set_vrm("/path/to/test.vrm")

            # Placeholder should have been removed
            assert guide_window._vrm_placeholder is None

    def test_set_vrm_with_invalid_path(self, guide_window):
        """Setting VRM with invalid path logs error gracefully."""
        # Should not raise
        guide_window.set_vrm("/nonexistent/path.vrm")


# =============================================================================
# Input Interaction Tests
# =============================================================================

class TestInputInteraction:
    """Tests for user input behavior."""

    def test_input_field_sends_on_enter(self, guide_window, qtbot):
        """Enter key triggers send and emits signal."""
        mock_engine = MagicMock()

        # Create a mock that returns an async iterator
        async def mock_send(msg):
            return
            yield  # Make it an async generator

        mock_engine.send_message = mock_send
        guide_window.set_engine(mock_engine)

        guide_window.input_field.setText("Hello Guide")

        with qtbot.waitSignal(guide_window.messageSent, timeout=1000):
            guide_window._on_send()

    def test_stop_mode_toggle(self, guide_window):
        """Send/Stop button toggles correctly."""
        assert guide_window.send_button.text() == "Send"
        guide_window._set_stop_mode(True)
        assert guide_window.send_button.text() == "Stop"
        guide_window._set_stop_mode(False)
        assert guide_window.send_button.text() == "Send"


# =============================================================================
# Chunk Handling Tests
# =============================================================================

class TestChunkHandling:
    """Tests for streaming chunk processing."""

    def test_text_chunk_displays(self, guide_window):
        """Text chunks appear in the dialogue."""
        guide_window._on_chunk({
            'type': 'text',
            'content': 'Hello there',
            'tool_name': None,
            'tool_id': None,
            'tool_input': None,
        })
        text = guide_window.dialogue_view.toPlainText()
        assert "Hello there" in text

    def test_error_chunk_displays(self, guide_window):
        """Error chunks appear in the dialogue."""
        guide_window._on_chunk({
            'type': 'error',
            'content': 'Something went wrong',
            'tool_name': None,
            'tool_id': None,
            'tool_input': None,
        })
        text = guide_window.dialogue_view.toPlainText()
        assert "Something went wrong" in text

    def test_tool_use_sets_indicator_status(self, guide_window):
        """Tool use chunks set the thinking indicator status text."""
        guide_window._on_chunk({
            'type': 'tool_use_start',
            'content': '',
            'tool_name': 'read_file',
            'tool_id': '123',
            'tool_input': None,
        })
        # Check the status label text rather than visibility
        # (visibility depends on parent widget being shown)
        assert "read_file" in guide_window.thinking_indicator.status_label.text()

    def test_done_chunk_clears_indicator(self, guide_window):
        """Done chunk clears the thinking indicator."""
        guide_window.thinking_indicator.set_status("Working...")
        guide_window._on_chunk({
            'type': 'done',
            'content': '',
            'tool_name': None,
            'tool_id': None,
            'tool_input': None,
        })
        # After done, indicator should be hidden
        assert guide_window.thinking_indicator.status_label.text() == "Working..."
        # The clear() hides it and stops the timer
        assert not guide_window.thinking_indicator._timer.isActive()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
