# ------------------------------------------------------------------
#   Tests for Shared Stage Window (Ensemble Mode)
#
#   Tests that the GuidePerformanceWindow supports both single-noodling
#   and ensemble modes. Ensemble mode provides two VRM viewports
#   side by side, noodling-aware dialogue methods, and named thinking.
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

import pytest
from PyQt6.QtWidgets import QMainWindow
from PyQt6.QtGui import QColor

from noodlestudio.runtime.ui.guide_performance_window import (
    GuidePerformanceWindow,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def parent_window(qapp, qtbot):
    """Create a parent window for positioning."""
    window = QMainWindow()
    window.resize(1400, 900)
    window.move(100, 100)
    qtbot.addWidget(window)
    yield window
    window.close()


@pytest.fixture
def single_window(parent_window, qtbot):
    """Create a single-mode performance window."""
    window = GuidePerformanceWindow(parent_window=parent_window)
    qtbot.addWidget(window)
    yield window
    window.close()


@pytest.fixture
def ensemble_window(parent_window, qtbot):
    """Create an ensemble-mode performance window."""
    window = GuidePerformanceWindow(
        parent_window=parent_window, ensemble_mode=True
    )
    qtbot.addWidget(window)
    yield window
    window.close()


# =============================================================================
# Single Mode Backward Compatibility
# =============================================================================

class TestSingleModeCompat:
    """Single mode works exactly as before the ensemble changes."""

    def test_single_mode_default(self, single_window):
        """Default window is not in ensemble mode."""
        assert not single_window.ensemble_mode

    def test_single_mode_size(self, single_window):
        """Default window size is 350x600."""
        assert single_window._size == (350, 600)

    def test_single_mode_has_default_container(self, single_window):
        """Single mode has a 'default' VRM container."""
        assert 'default' in single_window._vrm_containers
        assert 'default' in single_window._vrm_container_layouts
        assert 'default' in single_window._vrm_placeholders

    def test_single_mode_legacy_aliases(self, single_window):
        """Single mode keeps legacy vrm_container and vrm_container_layout."""
        assert single_window.vrm_container is not None
        assert single_window.vrm_container_layout is not None
        assert single_window._vrm_placeholder is not None

    def test_single_mode_get_slot_returns_default(self, single_window):
        """_get_slot always returns 'default' in single mode."""
        assert single_window._get_slot('ajo') == 'default'
        assert single_window._get_slot('yuki') == 'default'
        assert single_window._get_slot() == 'default'

    def test_set_busy_without_name(self, single_window):
        """set_busy works without name parameter (backward compat)."""
        single_window.set_busy(True)
        assert not single_window.input_field.isEnabled()
        # Timer running means indicator is active (isVisible requires shown parent)
        assert single_window.thinking_indicator._timer.isActive()
        assert single_window.thinking_indicator.status_label.text() == "Thinking..."

        single_window.set_busy(False)
        assert single_window.input_field.isEnabled()
        assert not single_window.thinking_indicator._timer.isActive()

    def test_append_guide_text_uses_icon(self, single_window):
        """append_guide_text uses the guide icon prefix in single mode."""
        single_window.append_guide_text("Hello world")
        text = single_window.dialogue_view.toPlainText()
        assert "\ua69c Hello world" in text

    def test_begin_guide_text_uses_icon(self, single_window):
        """begin_guide_text inserts guide icon prefix."""
        single_window.begin_guide_text()
        text = single_window.dialogue_view.toPlainText()
        assert "\ua69c " in text

    def test_append_character_works(self, single_window):
        """append_character adds individual characters."""
        single_window.begin_guide_text()
        single_window.append_character("H")
        single_window.append_character("i")
        text = single_window.dialogue_view.toPlainText()
        assert "Hi" in text

    def test_end_guide_text_adds_newlines(self, single_window):
        """end_guide_text finalizes the text block."""
        single_window.begin_guide_text()
        single_window.append_character("X")
        single_window.end_guide_text()
        text = single_window.dialogue_view.toPlainText()
        assert "X" in text

    def test_append_user_text(self, single_window):
        """append_user_text shows user message with icon."""
        single_window.append_user_text("Hello from user")
        text = single_window.dialogue_view.toPlainText()
        assert "Hello from user" in text

    def test_show_play_header(self, single_window):
        """show_play_header sets header text."""
        single_window.show_play_header("Ajo Alive")
        assert single_window.header_label.text() == "Ajo Alive"

    def test_input_placeholder_single(self, single_window):
        """Single mode shows 'Talk to Guide...' placeholder."""
        assert single_window.input_field.placeholderText() == "Talk to Guide..."


# =============================================================================
# Ensemble Mode Layout
# =============================================================================

class TestEnsembleLayout:
    """Ensemble mode creates three VRM viewports side by side."""

    def test_ensemble_mode_flag(self, ensemble_window):
        """Ensemble mode is correctly set."""
        assert ensemble_window.ensemble_mode

    def test_ensemble_size_auto_widens(self, parent_window, qtbot):
        """Ensemble mode with default size auto-widens to 900x650."""
        window = GuidePerformanceWindow(
            parent_window=parent_window, ensemble_mode=True
        )
        qtbot.addWidget(window)
        assert window._size == (900, 650)
        window.close()

    def test_ensemble_custom_size_respected(self, parent_window, qtbot):
        """Custom size is not overridden by ensemble auto-widening."""
        window = GuidePerformanceWindow(
            parent_window=parent_window,
            size=(800, 700),
            ensemble_mode=True
        )
        qtbot.addWidget(window)
        assert window._size == (800, 700)
        window.close()

    def test_ensemble_has_three_containers(self, ensemble_window):
        """Ensemble mode creates left, center, and right VRM containers."""
        assert 'left' in ensemble_window._vrm_containers
        assert 'center' in ensemble_window._vrm_containers
        assert 'right' in ensemble_window._vrm_containers
        assert len(ensemble_window._vrm_containers) == 3

    def test_ensemble_has_three_placeholders(self, ensemble_window):
        """Ensemble mode creates left, center, and right placeholders."""
        assert 'left' in ensemble_window._vrm_placeholders
        assert 'center' in ensemble_window._vrm_placeholders
        assert 'right' in ensemble_window._vrm_placeholders

    def test_ensemble_vrm_row_height(self, ensemble_window):
        """VRM row has correct height in ensemble mode."""
        assert ensemble_window._vrm_row.maximumHeight() == 280

    def test_input_placeholder_ensemble(self, ensemble_window):
        """Ensemble mode shows 'Talk to the ensemble...' placeholder."""
        assert ensemble_window.input_field.placeholderText() == "Talk to the ensemble..."


# =============================================================================
# VRM Slot Assignment
# =============================================================================

class TestVRMSlotAssignment:
    """Tests for noodling_id to slot routing in ensemble mode."""

    def test_first_noodling_gets_left(self, ensemble_window):
        """First noodling_id is assigned to left slot."""
        slot = ensemble_window._get_slot('ajo')
        assert slot == 'left'
        assert ensemble_window._noodling_to_slot['ajo'] == 'left'

    def test_second_noodling_gets_center(self, ensemble_window):
        """Second noodling_id is assigned to center slot."""
        ensemble_window._get_slot('ajo')
        slot = ensemble_window._get_slot('krampus')
        assert slot == 'center'
        assert ensemble_window._noodling_to_slot['krampus'] == 'center'

    def test_third_noodling_gets_right(self, ensemble_window):
        """Third noodling_id is assigned to right slot."""
        ensemble_window._get_slot('ajo')
        ensemble_window._get_slot('krampus')
        slot = ensemble_window._get_slot('juanita')
        assert slot == 'right'
        assert ensemble_window._noodling_to_slot['juanita'] == 'right'

    def test_same_noodling_returns_same_slot(self, ensemble_window):
        """Calling _get_slot twice with same id returns same slot."""
        slot1 = ensemble_window._get_slot('ajo')
        slot2 = ensemble_window._get_slot('ajo')
        assert slot1 == slot2

    def test_fourth_noodling_falls_back_to_left(self, ensemble_window):
        """Fourth noodling_id falls back to left (all slots taken)."""
        ensemble_window._get_slot('ajo')
        ensemble_window._get_slot('krampus')
        ensemble_window._get_slot('juanita')
        slot = ensemble_window._get_slot('four')
        assert slot == 'left'

    def test_single_mode_ignores_noodling_id(self, single_window):
        """Single mode always routes to 'default' slot."""
        assert single_window._get_slot('ajo') == 'default'
        assert single_window._get_slot('yuki') == 'default'


# =============================================================================
# Ensemble Dialogue
# =============================================================================

class TestEnsembleDialogue:
    """Tests for noodling-aware dialogue in ensemble mode."""

    def test_begin_noodling_text_with_name(self, ensemble_window):
        """begin_noodling_text inserts name prefix in ensemble mode."""
        ensemble_window.begin_noodling_text('ajo', 'Ajo')
        text = ensemble_window.dialogue_view.toPlainText()
        assert "Ajo: " in text

    def test_begin_noodling_text_without_name(self, ensemble_window):
        """begin_noodling_text uses icon prefix when name is None."""
        ensemble_window.begin_noodling_text('ajo', None)
        text = ensemble_window.dialogue_view.toPlainText()
        assert "\ua69c " in text

    def test_append_noodling_text_with_name(self, ensemble_window):
        """append_noodling_text includes name prefix in ensemble mode."""
        ensemble_window.append_noodling_text('ajo', 'Ajo', 'Hello there!')
        text = ensemble_window.dialogue_view.toPlainText()
        assert "Ajo: Hello there!" in text

    def test_append_noodling_text_single_mode_uses_icon(self, single_window):
        """append_noodling_text uses icon prefix in single mode."""
        single_window.append_noodling_text('ajo', 'Ajo', 'Hello')
        text = single_window.dialogue_view.toPlainText()
        assert "\ua69c Hello" in text
        assert "Ajo:" not in text

    def test_interleaved_dialogue(self, ensemble_window):
        """Multiple noodlings' text appears interleaved."""
        ensemble_window.append_user_text("Tell me about feelings")
        ensemble_window.append_noodling_text('ajo', 'Ajo', 'Feelings are fascinating!')
        ensemble_window.append_noodling_text('yuki', 'Yuki', 'An old fox has seen many.')

        text = ensemble_window.dialogue_view.toPlainText()
        ajo_pos = text.index("Ajo:")
        yuki_pos = text.index("Yuki:")
        assert ajo_pos < yuki_pos

    def test_end_noodling_text_clears_typing_state(self, ensemble_window):
        """end_noodling_text clears the current typing noodling."""
        ensemble_window.begin_noodling_text('yuki', 'Yuki')
        assert ensemble_window._current_typing_noodling == 'yuki'
        ensemble_window.end_noodling_text()
        assert ensemble_window._current_typing_noodling is None


# =============================================================================
# Noodling Text Colors
# =============================================================================

class TestNoodlingColors:
    """Tests for per-noodling text colors."""

    def test_ajo_color(self, ensemble_window):
        """Ajo's color is warm gray (#B0B0B0)."""
        color = ensemble_window._noodling_colors['ajo']
        assert color.red() == 176
        assert color.green() == 176
        assert color.blue() == 176

    def test_krampus_color(self, ensemble_window):
        """Krampus color is warm brownish gray (#B0A090)."""
        color = ensemble_window._noodling_colors['krampus']
        assert color.red() == 176
        assert color.green() == 160
        assert color.blue() == 144

    def test_juanita_color(self, ensemble_window):
        """Juanita color is subtle sage gray (#A0B0A0)."""
        color = ensemble_window._noodling_colors['juanita']
        assert color.red() == 160
        assert color.green() == 176
        assert color.blue() == 160

    def test_typing_noodling_tracked(self, ensemble_window):
        """Current typing noodling is tracked for append_character color."""
        ensemble_window.begin_noodling_text('yuki', 'Yuki')
        assert ensemble_window._current_typing_noodling == 'yuki'

    def test_default_color_for_unknown_noodling(self, ensemble_window):
        """Unknown noodling_id gets default color."""
        color = ensemble_window._noodling_colors.get(
            'unknown', ensemble_window._noodling_colors['default']
        )
        assert color.red() == 176  # Same as default


# =============================================================================
# Named Thinking Indicator
# =============================================================================

class TestNamedThinking:
    """Tests for named thinking indicator."""

    def test_set_busy_with_name(self, ensemble_window):
        """set_busy with name shows named thinking text."""
        ensemble_window.set_busy(True, name='Ajo')
        assert ensemble_window.thinking_indicator.status_label.text() == "Ajo is thinking..."

    def test_set_busy_with_different_name(self, ensemble_window):
        """set_busy with Yuki's name shows her name."""
        ensemble_window.set_busy(True, name='Yuki')
        assert ensemble_window.thinking_indicator.status_label.text() == "Yuki is thinking..."

    def test_set_thinking_shows_name(self, ensemble_window):
        """set_thinking with a name shows it in the indicator."""
        ensemble_window.set_thinking('ajo', 'Ajo', True)
        assert ensemble_window.thinking_indicator._timer.isActive()
        assert ensemble_window.thinking_indicator.status_label.text() == "Ajo is thinking..."

    def test_set_thinking_clear(self, ensemble_window):
        """set_thinking(False) hides the indicator."""
        ensemble_window.set_thinking('ajo', 'Ajo', True)
        ensemble_window.set_thinking('ajo', 'Ajo', False)
        assert not ensemble_window.thinking_indicator._timer.isActive()

    def test_set_thinking_without_name(self, ensemble_window):
        """set_thinking without name shows generic text."""
        ensemble_window.set_thinking('ajo', None, True)
        assert ensemble_window.thinking_indicator.status_label.text() == "Thinking..."


# =============================================================================
# Speaking Mode Routing
# =============================================================================

class TestSpeakingMode:
    """Tests for speaking mode routing via the window API."""

    def test_set_speaking_mode_no_viewport_no_crash(self, ensemble_window):
        """set_speaking_mode does not crash when no viewport loaded."""
        # Should not raise
        ensemble_window.set_speaking_mode(True, 0.7, 'ajo')

    def test_set_speaking_mode_no_viewport_single(self, single_window):
        """set_speaking_mode does not crash in single mode without viewport."""
        single_window.set_speaking_mode(True, 0.5)


# Made with love. Use with love.
# Caitlyn Meeks 2026
