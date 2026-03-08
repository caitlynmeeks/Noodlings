# ------------------------------------------------------------------
#   Tests for Guide Screen Space Integration
#
#   Integration tests verifying the performance panel remains
#   accessible during tab switches and can be launched via CLI flags.
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

from unittest.mock import MagicMock, patch
import pytest

from PyQt6.QtWidgets import QMainWindow, QTabWidget, QWidget
from PyQt6.QtCore import QTimer

from noodlestudio.runtime.ui.guide_performance_window import (
    PerformancePanel,
)
from noodlestudio.runtime.ui.guide_performance_manager import (
    GuidePerformanceManager,
)

# Patch path for ComputerUseController
CUC_PATCH = 'noodlestudio.core.computer_use_controller.get_computer_use_controller'

# Patch create_llm_client to prevent real provider config lookups
CREATE_LLM_PATCH = 'noodlestudio.runtime.ui.guide_performance_manager.create_llm_client'

# Patch NoodlingPerformer.load_assembly to prevent real assembly loading
LOAD_ASSEMBLY_PATCH = 'noodlestudio.runtime.ui.noodling_performer.NoodlingPerformer.load_assembly'


class FakeLLMClient:
    """Lightweight stand-in for HeadlessLLMClient."""
    async def close(self):
        pass


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


@pytest.fixture
def performance_panel(qapp, qtbot):
    """Create a real PerformancePanel for screen space tests."""
    panel = PerformancePanel(ensemble_mode=False)
    qtbot.addWidget(panel)
    return panel


# =============================================================================
# Screen Space Tests
# =============================================================================

class TestGuideScreenSpace:
    """Tests for panel accessibility during UI interaction."""

    @patch(LOAD_ASSEMBLY_PATCH, return_value=True)
    @patch(CREATE_LLM_PATCH, return_value=FakeLLMClient())
    @patch(CUC_PATCH)
    def test_guide_visible_during_tab_switch(self, mock_get_ctrl, mock_create_llm, mock_load, parent_window, performance_panel, qtbot):
        """Performance panel remains active when switching center tabs."""
        mock_ctrl = MagicMock()
        mock_get_ctrl.return_value = mock_ctrl

        manager = GuidePerformanceManager(parent_window)
        manager.set_performance_panel(performance_panel)

        parent_window.show()
        manager.start_performance("Test Play")

        assert manager.is_active
        # Panel is persistent (embedded tab) -- always non-None
        assert manager.window is not None

        # Switch tabs
        parent_window.tabs.setCurrentIndex(1)  # Facets
        qtbot.wait(100)

        # Performance should still be active
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


# Made with love. Use with love.
# Caitlyn Meeks 2026
