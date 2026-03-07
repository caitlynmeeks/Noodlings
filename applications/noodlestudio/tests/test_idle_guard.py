# ──────────────────────────────────────────────────────────────
#   Tests for Idle Animation Guard
# ──────────────────────────────────────────────────────────────

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class StubViewport:
    """Minimal stand-in for VRMViewportWidget.

    Tracks idle guard method calls for test assertions.
    """

    def __init__(self):
        self.idle_enabled = None
        self.frozen = False

    def set_idle_enabled(self, enabled):
        self.idle_enabled = enabled
        if enabled:
            self.frozen = False

    def freeze_idle(self):
        self.frozen = True


class TestIdleGuardViewport:
    """Idle guard methods on VRMViewportWidget."""

    def test_set_idle_enabled_false_resets(self, qapp):
        """set_idle_enabled(False) stops timer, resets muscles + phase."""
        from noodlestudio.runtime.ui.components.vrm_viewport import VRMViewport, VRMViewportWidget
        component = VRMViewport("test_idle")
        widget = VRMViewportWidget(component)

        # Manually set some idle state
        widget._idle_phase = 5.0
        widget._idle_muscles = {'Head.NodDownUp': 0.5}

        widget.set_idle_enabled(False)

        assert widget._idle_phase == 0.0
        assert widget._idle_muscles == {}
        if widget._idle_timer:
            assert not widget._idle_timer.isActive()

        widget.close()

    def test_set_idle_enabled_true_starts(self, qapp):
        """set_idle_enabled(True) starts the idle timer."""
        from noodlestudio.runtime.ui.components.vrm_viewport import VRMViewport, VRMViewportWidget
        component = VRMViewport("test_idle2")
        widget = VRMViewportWidget(component)

        widget.set_idle_enabled(True)

        assert widget._idle_timer is not None
        assert widget._idle_timer.isActive()

        widget._idle_timer.stop()
        widget.close()

    def test_freeze_idle_preserves_state(self, qapp):
        """freeze_idle() stops timer but preserves muscles + phase."""
        from noodlestudio.runtime.ui.components.vrm_viewport import VRMViewport, VRMViewportWidget
        component = VRMViewport("test_freeze")
        widget = VRMViewportWidget(component)

        # Start idle first so timer exists
        widget.set_idle_enabled(True)

        # Simulate that some time has passed
        widget._idle_phase = 3.14
        widget._idle_muscles = {'Chest.FrontBack': 0.06}

        widget.freeze_idle()

        # Timer stopped
        assert not widget._idle_timer.isActive()
        # State preserved (not reset)
        assert widget._idle_phase == 3.14
        assert widget._idle_muscles == {'Chest.FrontBack': 0.06}

        widget._idle_timer.stop()
        widget.close()


class TestIdleGuardManager:
    """Manager dispatches idle modes to viewports."""

    def _make_manager_with_viewports(self):
        """Create a manager with stub viewports wired in."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager, PerformanceState,
        )
        from tests.conftest import StubMainWindow, StubWindow

        stub_main = StubMainWindow()
        manager = GuidePerformanceManager(stub_main)

        # Wire stub window with stub viewports
        window = StubWindow()
        vp1 = StubViewport()
        vp2 = StubViewport()
        window._vrm_viewports = {'left': vp1, 'center': vp2}
        manager._window = window
        manager._performance_state = PerformanceState.IDLE

        return manager, vp1, vp2

    def test_playing_enables_idle(self):
        """PLAYING state calls set_idle_enabled(True)."""
        from noodlestudio.runtime.ui.guide_performance_manager import PerformanceState
        manager, vp1, vp2 = self._make_manager_with_viewports()
        manager._set_performance_state(PerformanceState.PLAYING)
        assert vp1.idle_enabled is True
        assert vp2.idle_enabled is True

    def test_paused_freezes_idle(self):
        """PAUSED state calls freeze_idle()."""
        from noodlestudio.runtime.ui.guide_performance_manager import PerformanceState
        manager, vp1, vp2 = self._make_manager_with_viewports()
        manager._set_performance_state(PerformanceState.PAUSED)
        assert vp1.frozen is True
        assert vp2.frozen is True

    def test_stopped_disables_idle(self):
        """STOPPED state calls set_idle_enabled(False)."""
        from noodlestudio.runtime.ui.guide_performance_manager import PerformanceState
        manager, vp1, vp2 = self._make_manager_with_viewports()
        manager._set_performance_state(PerformanceState.STOPPED)
        assert vp1.idle_enabled is False
        assert vp2.idle_enabled is False
