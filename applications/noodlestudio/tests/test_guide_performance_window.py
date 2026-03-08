# ──────────────────────────────────────────────────────────────
#   Tests for Performance Panel (formerly GuidePerformanceWindow)
#
#   Tests for the embeddable panel that provides VRM
#   character rendering, dialogue display, and user text input
#   during guided play performances.
#
#   The panel is a pure renderer -- it does NOT make LLM calls.
#   All cognition is handled by GuidePerformanceManager.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtCore import Qt

from noodlestudio.runtime.ui.guide_performance_window import (
    GuidePerformanceWindow,
    PerformancePanel,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def guide_window(qapp, qtbot):
    """Create a PerformancePanel (single mode) for testing."""
    window = PerformancePanel(ensemble_mode=False)
    qtbot.addWidget(window)
    yield window
    window.close()


# =============================================================================
# Window Creation Tests
# =============================================================================

class TestWindowCreation:
    """Tests for panel construction."""

    def test_panel_creation(self, guide_window):
        """Panel creates as a QWidget."""
        assert isinstance(guide_window, PerformancePanel)

    def test_has_vrm_container(self, guide_window):
        """Panel has a VRM viewport container for the 'left' slot."""
        assert 'left' in guide_window._vrm_containers
        assert guide_window._vrm_containers['left'] is not None

    def test_has_dialogue_view(self, guide_window):
        """Panel has a dialogue text display."""
        assert guide_window.dialogue_view is not None
        assert guide_window.dialogue_view.isReadOnly()

    def test_has_input_field(self, guide_window):
        """Panel has a user input field."""
        assert guide_window.input_field is not None

    def test_has_send_button(self, guide_window):
        """Panel has a send button."""
        assert guide_window.send_button is not None
        assert guide_window.send_button.text() == "Send"

    def test_has_header(self, guide_window):
        """Panel has a header label."""
        assert guide_window.header_label is not None

    def test_has_thinking_indicator(self, guide_window):
        """Panel has a thinking indicator (hidden by default)."""
        assert guide_window.thinking_indicator is not None


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
    """Tests for panel signals (pure renderer pattern)."""

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

            # Placeholder should have been removed for the 'left' slot
            assert guide_window._vrm_placeholders.get('left') is None

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


# Made with love. Use with love.
# Caitlyn Meeks 2026
