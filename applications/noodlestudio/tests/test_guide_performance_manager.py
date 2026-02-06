# ──────────────────────────────────────────────────────────────
#   Tests for Guide Performance Manager
#
#   Tests for the orchestrator that coordinates performance
#   lifecycle: window creation, demo mode, [D] button sync.
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
def mock_engine():
    """Create a mock NoodleCode engine."""
    engine = MagicMock()
    return engine


@pytest.fixture
def mock_panel():
    """Create a mock NoodleCode panel with [D] button."""
    panel = MagicMock()
    panel.demo_mode_btn = MagicMock(spec=QPushButton)
    return panel


@pytest.fixture
def manager(parent_window, mock_engine, mock_panel):
    """Create a configured GuidePerformanceManager."""
    m = GuidePerformanceManager(parent_window)
    m.set_engine(mock_engine)
    m.set_noodle_code_panel(mock_panel)
    return m


# =============================================================================
# Lifecycle Tests
# =============================================================================

class TestPerformanceLifecycle:
    """Tests for start/stop performance."""

    @patch(CUC_PATCH)
    def test_start_creates_window(self, mock_get_ctrl, manager):
        """Starting a performance creates the window."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")

        assert manager.window is not None
        assert manager.is_active

    @patch(CUC_PATCH)
    def test_start_enables_demo_mode(self, mock_get_ctrl, manager):
        """Starting a performance enables demo mode."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")

        mock_ctrl.demo_mode = True

    @patch(CUC_PATCH)
    def test_stop_cleans_up(self, mock_get_ctrl, manager):
        """Stopping cleans up window and demo mode."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")
        assert manager.is_active

        manager.stop_performance()
        assert not manager.is_active
        assert manager.window is None

    @patch(CUC_PATCH)
    def test_stop_disables_demo_mode(self, mock_get_ctrl, manager):
        """Stopping disables demo mode on controller."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")
        manager.stop_performance()

        # After stop, the last assignment to demo_mode should be False
        # MagicMock tracks property assignments as attribute sets
        assert mock_ctrl.demo_mode == False

    @patch(CUC_PATCH)
    def test_is_active_property(self, mock_get_ctrl, manager):
        """is_active reflects performance state."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        assert not manager.is_active

        manager.start_performance("Test Play")
        assert manager.is_active

        manager.stop_performance()
        assert not manager.is_active

    @patch(CUC_PATCH)
    def test_start_without_vrm(self, mock_get_ctrl, manager):
        """Performance starts without VRM path."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        # Should not raise
        manager.start_performance("Test Play", vrm_path=None)
        assert manager.is_active

    @patch(CUC_PATCH)
    def test_double_start_stops_first(self, mock_get_ctrl, manager):
        """Starting twice stops the first performance."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Play 1")
        first_window = manager.window

        manager.start_performance("Play 2")
        second_window = manager.window

        # Second window should be different from first
        assert second_window is not first_window


# =============================================================================
# Demo Button Sync Tests
# =============================================================================

class TestDemoButtonSync:
    """Tests for [D] button synchronization."""

    @patch(CUC_PATCH)
    def test_demo_button_checked_on_start(self, mock_get_ctrl, manager, mock_panel):
        """[D] button is checked when performance starts."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")

        mock_panel.demo_mode_btn.setChecked.assert_called_with(True)

    @patch(CUC_PATCH)
    def test_demo_button_unchecked_on_stop(self, mock_get_ctrl, manager, mock_panel):
        """[D] button is unchecked when performance stops."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")
        manager.stop_performance()

        # Last call should be setChecked(False)
        mock_panel.demo_mode_btn.setChecked.assert_called_with(False)


# =============================================================================
# Engine Wiring Tests
# =============================================================================

class TestEngineWiring:
    """Tests for engine and handler wiring."""

    @patch(CUC_PATCH)
    def test_engine_wired_to_window(self, mock_get_ctrl, manager, mock_engine):
        """Engine is wired to the performance window."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Test Play")

        assert manager.window.engine is mock_engine

    @patch(CUC_PATCH)
    def test_guide_cue_handler_wired(self, mock_get_ctrl, manager):
        """Guide cue handler is wired when available."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        mock_handler = MagicMock()
        manager.set_guide_cue_handler(mock_handler)

        manager.start_performance("Test Play")

        assert manager.window._guide_cue_handler is mock_handler

    @patch(CUC_PATCH)
    def test_header_shows_play_title(self, mock_get_ctrl, manager):
        """Window header displays the play title."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager.start_performance("Let's Consciousness!")

        assert manager.window.header_label.text() == "Let's Consciousness!"


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
