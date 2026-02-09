# ──────────────────────────────────────────────────────────────
#   Tests for Guide Performance Window
#
#   Tests for the floating combined panel that provides VRM
#   character rendering, dialogue display, and user text input
#   during guided play performances.
#
#   The window is a pure renderer -- it does NOT make LLM calls.
#   All cognition is handled by GuidePerformanceManager.
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

class TestInitialPosition:
    """Tests for initial window positioning."""

    def test_positions_at_parent_right_edge(self, parent_window, qapp, qtbot):
        """Window initially positions near the right edge of parent."""
        parent_window.move(100, 50)
        parent_window.show()

        window = GuidePerformanceWindow(parent_window=parent_window)
        qtbot.addWidget(window)

        geo = parent_window.geometry()
        # Should be near the right edge (allow tolerance for WM)
        assert abs(window.x() - (geo.right() - 350 - 20)) < 50
        window.close()

    def test_stays_put_when_parent_moves(self, guide_window, parent_window):
        """Window stays at its position when parent moves (independent)."""
        parent_window.move(100, 50)
        parent_window.show()
        guide_window.show()

        original_pos = guide_window.pos()

        # Move the parent window
        parent_window.move(500, 300)

        # Guide window should NOT have moved
        assert guide_window.pos() == original_pos


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

    def test_play_header(self, guide_window):
        """Header displays play title."""
        guide_window.show_play_header("Let's Consciousness!")
        assert guide_window.header_label.text() == "Let's Consciousness!"


# =============================================================================
# Signal Tests
# =============================================================================

class TestSignals:
    """Tests for window signals (pure renderer pattern)."""

    def test_message_submitted_signal(self, guide_window, qtbot):
        """Sending a message emits messageSubmitted signal."""
        guide_window.input_field.setText("Hello Guide")

        with qtbot.waitSignal(guide_window.messageSubmitted, timeout=1000):
            guide_window._on_send()

    def test_message_sent_signal(self, guide_window, qtbot):
        """Sending a message emits messageSent signal for channel bus."""
        guide_window.input_field.setText("Hello Guide")

        with qtbot.waitSignal(guide_window.messageSent, timeout=1000):
            guide_window._on_send()

    def test_send_clears_input(self, guide_window):
        """Sending a message clears the input field."""
        guide_window.input_field.setText("Hello Guide")
        guide_window._on_send()
        assert guide_window.input_field.text() == ""

    def test_send_displays_user_text(self, guide_window):
        """Sending a message displays it in the dialogue."""
        guide_window.input_field.setText("Hello Guide")
        guide_window._on_send()
        text = guide_window.dialogue_view.toPlainText()
        assert "Hello Guide" in text

    def test_empty_send_does_nothing(self, guide_window, qtbot):
        """Empty input does not emit signals."""
        guide_window.input_field.setText("")

        # messageSubmitted should NOT fire
        emitted = []
        guide_window.messageSubmitted.connect(lambda msg: emitted.append(msg))
        guide_window._on_send()
        assert len(emitted) == 0


# =============================================================================
# Busy State Tests
# =============================================================================

class TestBusyState:
    """Tests for set_busy() (controlled by manager during assembly execution)."""

    def test_set_busy_disables_input(self, guide_window):
        """set_busy(True) disables input field and send button."""
        guide_window.set_busy(True)
        assert not guide_window.input_field.isEnabled()
        assert not guide_window.send_button.isEnabled()

    def test_set_busy_shows_thinking(self, guide_window):
        """set_busy(True) shows thinking indicator."""
        guide_window.set_busy(True)
        assert "Thinking" in guide_window.thinking_indicator.status_label.text()

    def test_clear_busy_enables_input(self, guide_window):
        """set_busy(False) re-enables input."""
        guide_window.set_busy(True)
        guide_window.set_busy(False)
        assert guide_window.input_field.isEnabled()
        assert guide_window.send_button.isEnabled()

    def test_clear_busy_hides_thinking(self, guide_window):
        """set_busy(False) stops the thinking indicator timer."""
        guide_window.set_busy(True)
        guide_window.set_busy(False)
        assert not guide_window.thinking_indicator._timer.isActive()


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
# Error Display Tests
# =============================================================================

class TestErrorDisplay:
    """Tests for error message display."""

    def test_show_error(self, guide_window):
        """_show_error displays error text in dialogue."""
        guide_window._show_error("Something went wrong")
        text = guide_window.dialogue_view.toPlainText()
        assert "Something went wrong" in text


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
