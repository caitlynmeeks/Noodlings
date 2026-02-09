# ──────────────────────────────────────────────────────────────
#   Tests for Guide Screen Space Integration
#
#   Integration tests verifying the floating performance window
#   remains visible during tab switches and can be launched
#   via CLI flags.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget
from PyQt6.QtCore import QTimer

from noodlestudio.runtime.ui.guide_performance_window import (
    GuidePerformanceWindow,
)
from noodlestudio.runtime.ui.guide_performance_manager import (
    GuidePerformanceManager,
)

# Patch path for ComputerUseController
CUC_PATCH = 'noodlestudio.core.computer_use_controller.get_computer_use_controller'


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def parent_window(qapp, qtbot):
    """Create a parent window with tab widget."""
    window = QMainWindow()
    window.resize(1400, 900)
    window.move(100, 100)

    tabs = QTabWidget()
    tabs.addTab(QWidget(), "Noodle Code")
    tabs.addTab(QWidget(), "Facets")
    tabs.addTab(QWidget(), "Gaussian Viewer")
    window.setCentralWidget(tabs)
    window.tabs = tabs

    qtbot.addWidget(window)
    yield window
    window.close()


# =============================================================================
# Screen Space Tests
# =============================================================================

class TestGuideScreenSpace:
    """Tests for window visibility during UI interaction."""

    @patch('noodlestudio.runtime.ui.guide_performance_manager.GuidePerformanceManager._load_assembly', return_value=False)
    @patch(CUC_PATCH)
    def test_guide_visible_during_tab_switch(self, mock_get_ctrl, mock_load, parent_window, qtbot):
        """Performance window remains visible when switching center tabs."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager = GuidePerformanceManager(parent_window)

        parent_window.show()
        manager.start_performance("Test Play")

        assert manager.is_active
        assert manager.window is not None

        # Switch tabs
        parent_window.tabs.setCurrentIndex(1)  # Facets
        qtbot.wait(100)

        # Performance window should still exist and be active
        assert manager.is_active
        assert manager.window is not None

        # Switch again
        parent_window.tabs.setCurrentIndex(2)  # Gaussian Viewer
        qtbot.wait(100)

        assert manager.is_active

        manager.stop_performance()


class TestPlayCLILaunch:
    """Tests for --play CLI argument integration."""

    def test_parse_args_accepts_play(self):
        """Argument parser accepts --play flag."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--play', '/tmp/test.play.yaml']
        try:
            args = parse_args()
            assert args.play == '/tmp/test.play.yaml'
        finally:
            sys.argv = original_argv

    def test_parse_args_play_is_optional(self):
        """--play is optional."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--no-splash']
        try:
            args = parse_args()
            assert args.play is None
        finally:
            sys.argv = original_argv


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
