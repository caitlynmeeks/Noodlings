# ──────────────────────────────────────────────────────────────
#   Tests for PerformancePanel (embedded center-pane tab)
# ──────────────────────────────────────────────────────────────

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def panel(qapp):
    """Create a PerformancePanel for testing."""
    from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
    p = PerformancePanel(ensemble_mode=True)
    yield p
    p.close()


class TestPerformancePanelExtract:
    """Commit 1: PerformancePanel is QWidget, not QMainWindow."""

    def test_is_qwidget_not_qmainwindow(self, panel):
        from PyQt6.QtWidgets import QWidget, QMainWindow
        assert isinstance(panel, QWidget)
        assert not isinstance(panel, QMainWindow)

    def test_creates_without_parent_window_arg(self, qapp):
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        p = PerformancePanel()
        assert p is not None
        p.close()

    def test_not_fixed_size(self, panel):
        """Panel should be resizable (no setFixedSize)."""
        # QWidget.maximumSize defaults to (16777215, 16777215) unless fixed
        max_w = panel.maximumWidth()
        max_h = panel.maximumHeight()
        assert max_w > 1000
        assert max_h > 1000

    def test_has_three_vrm_slots(self, panel):
        """Should have left, center, right VRM containers."""
        assert 'left' in panel._vrm_containers
        assert 'center' in panel._vrm_containers
        assert 'right' in panel._vrm_containers

    def test_signals_exist(self, panel):
        """All required signals must exist."""
        assert hasattr(panel, 'messageSubmitted')
        assert hasattr(panel, 'messageSent')
        assert hasattr(panel, 'noodlingSelected')

    def test_dialogue_methods_work(self, panel):
        """Basic dialogue append should not crash."""
        panel.append_user_text("Hello")
        panel.append_guide_text("Hi there")
        text = panel.dialogue_view.toPlainText()
        assert "Hello" in text
        assert "Hi there" in text

    def test_set_input_enabled(self, panel):
        """Input field and send button toggle together."""
        panel.set_input_enabled(False)
        assert not panel.input_field.isEnabled()
        assert not panel.send_button.isEnabled()
        panel.set_input_enabled(True)
        assert panel.input_field.isEnabled()
        assert panel.send_button.isEnabled()

    def test_set_ensemble_visible_hides_slots(self, panel):
        """set_ensemble_visible(False) hides name bar + center/right."""
        panel.set_ensemble_visible(False)
        # Use isHidden() -- isVisible() requires parent to be shown
        assert panel._name_bar.isHidden()
        assert panel._vrm_containers['center'].isHidden()
        assert panel._vrm_containers['right'].isHidden()
        # Left should NOT be hidden
        assert not panel._vrm_containers['left'].isHidden()

    def test_set_ensemble_visible_shows_slots(self, panel):
        """set_ensemble_visible(True) shows everything."""
        panel.set_ensemble_visible(False)
        panel.set_ensemble_visible(True)
        assert not panel._name_bar.isHidden()
        assert not panel._vrm_containers['center'].isHidden()
        assert not panel._vrm_containers['right'].isHidden()

    def test_backward_compat_alias(self):
        """GuidePerformanceWindow should be an alias for PerformancePanel."""
        from noodlestudio.runtime.ui.guide_performance_window import (
            PerformancePanel, GuidePerformanceWindow
        )
        assert GuidePerformanceWindow is PerformancePanel


class TestPerformancePanelCenterPane:
    """Commit 2: Performance tab in center pane."""

    def test_performance_tab_exists(self, main_window):
        """'Performance' tab should exist in center pane."""
        tabs = main_window.center_tabs
        tab_names = [tabs.tabText(i) for i in range(tabs.count())]
        assert "Performance" in tab_names

    def test_performance_tab_is_panel_instance(self, main_window):
        """Performance tab widget should be a PerformancePanel."""
        from noodlestudio.runtime.ui.guide_performance_window import PerformancePanel
        assert isinstance(main_window.performance_panel, PerformancePanel)

    def test_manager_window_is_embedded_panel(self, main_window, qtbot):
        """After init, manager._window should be the embedded panel."""
        from PyQt6.QtCore import QTimer
        # _init_guide_performance runs on a 600ms timer; process events
        qtbot.wait(700)
        mgr = getattr(main_window, 'guide_performance_manager', None)
        if mgr:
            assert mgr._window is main_window.performance_panel
