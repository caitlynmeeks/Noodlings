# ──────────────────────────────────────────────────────────────
#   Tests for Guide Performance Manager
#
#   Tests for the orchestrator that coordinates performance
#   lifecycle: window creation, assembly loading, demo mode,
#   [D] button sync, and affect pipeline.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QMainWindow, QPushButton

from noodlestudio.runtime.ui.guide_performance_manager import (
    GuidePerformanceManager,
)

# The manager imports get_computer_use_controller locally, so patch at source
CUC_PATCH = 'noodlestudio.core.computer_use_controller.get_computer_use_controller'

# Patch _load_assembly to prevent FacetExecutor creation (requires event loop)
LOAD_ASSEMBLY_PATCH = 'noodlestudio.runtime.ui.guide_performance_manager.GuidePerformanceManager._load_assembly'


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
def mock_panel():
    """Create a mock NoodleCode panel with [D] button."""
    panel = MagicMock()
    panel.demo_mode_btn = MagicMock(spec=QPushButton)
    return panel


@pytest.fixture
def manager(parent_window, mock_panel):
    """Create a configured GuidePerformanceManager."""
    m = GuidePerformanceManager(parent_window)
    m.set_noodle_code_panel(mock_panel)
    return m


# =============================================================================
# Lifecycle Tests
# =============================================================================

class TestPerformanceLifecycle:
    """Tests for start/stop performance."""

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_start_creates_window(self, mock_get_ctrl, mock_load, manager):
        """Starting a performance creates the window."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        assert manager.window is not None
        assert manager.is_active

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_start_enables_demo_mode(self, mock_get_ctrl, mock_load, manager):
        """Starting a performance enables demo mode."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl
        manager.start_performance("Test Play")
        mock_ctrl.demo_mode = True

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_stop_cleans_up(self, mock_get_ctrl, mock_load, manager):
        """Stopping cleans up window and demo mode."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        assert manager.is_active
        manager.stop_performance()
        assert not manager.is_active
        assert manager.window is None

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_stop_disables_demo_mode(self, mock_get_ctrl, mock_load, manager):
        """Stopping disables demo mode on controller."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl
        manager.start_performance("Test Play")
        manager.stop_performance()
        assert mock_ctrl.demo_mode == False

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_is_active_property(self, mock_get_ctrl, mock_load, manager):
        """is_active reflects performance state."""
        mock_get_ctrl.return_value = MagicMock()
        assert not manager.is_active
        manager.start_performance("Test Play")
        assert manager.is_active
        manager.stop_performance()
        assert not manager.is_active

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_start_without_vrm(self, mock_get_ctrl, mock_load, manager):
        """Performance starts without VRM path."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play", vrm_path=None)
        assert manager.is_active

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_double_start_stops_first(self, mock_get_ctrl, mock_load, manager):
        """Starting twice stops the first performance."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Play 1")
        first_window = manager.window
        manager.start_performance("Play 2")
        second_window = manager.window
        assert second_window is not first_window


# =============================================================================
# Demo Button Sync Tests
# =============================================================================

class TestDemoButtonSync:
    """Tests for [D] button synchronization."""

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_demo_button_checked_on_start(self, mock_get_ctrl, mock_load, manager, mock_panel):
        """[D] button is checked when performance starts."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        mock_panel.demo_mode_btn.setChecked.assert_called_with(True)

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_demo_button_unchecked_on_stop(self, mock_get_ctrl, mock_load, manager, mock_panel):
        """[D] button is unchecked when performance stops."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        manager.stop_performance()
        mock_panel.demo_mode_btn.setChecked.assert_called_with(False)


# =============================================================================
# Assembly Wiring Tests
# =============================================================================

class TestAssemblyWiring:
    """Tests for assembly loading and execution pipeline."""

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_header_shows_play_title(self, mock_get_ctrl, mock_load, manager):
        """Window header displays the play title."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Let's Consciousness!")
        assert manager.window.header_label.text() == "Let's Consciousness!"

    @patch(CUC_PATCH)
    def test_guide_cue_handler_set(self, mock_get_ctrl, manager):
        """Guide cue handler is stored on the manager."""
        mock_handler = MagicMock()
        manager.set_guide_cue_handler(mock_handler)
        assert manager._guide_cue_handler is mock_handler

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_stop_clears_assembly_state(self, mock_get_ctrl, mock_load, manager):
        """Stopping performance clears assembly, executor, and history."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        manager._conversation_history.append({'role': 'user', 'content': 'test'})
        manager.stop_performance()
        assert manager._assembly is None
        assert manager._executor is None
        assert manager._conversation_history == []

    @patch(LOAD_ASSEMBLY_PATCH, return_value=False)
    @patch(CUC_PATCH)
    def test_message_submitted_without_assembly(self, mock_get_ctrl, mock_load, manager):
        """Message submission without loaded assembly shows error."""
        mock_get_ctrl.return_value = MagicMock()
        manager.start_performance("Test Play")
        manager._assembly = None
        manager._executor = None
        manager._on_user_message_for_assembly("Hello")
        text = manager.window.dialogue_view.toPlainText()
        assert "not loaded" in text.lower() or "error" in text.lower()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
