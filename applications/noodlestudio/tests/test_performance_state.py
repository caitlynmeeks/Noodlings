# ──────────────────────────────────────────────────────────────
#
#   Performance State Tests
#
#   Verifies the PerformanceState enum, state machine transitions,
#   persistent window lifecycle, pause/resume, and transport button
#   synchronization.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_performance_state
# PURPOSE:  Performance Lifecycle State Machine Tests
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from noodlestudio.runtime.ui.guide_performance_manager import (
    GuidePerformanceManager,
    PerformanceState,
)
from conftest import StubMainWindow, StubFacetsEditor, StubWindow, FakeLLMClient


# ============================================================================
# Helpers
# ============================================================================

def _make_manager():
    """Create a GuidePerformanceManager with stub dependencies."""
    stub_editor = StubFacetsEditor()
    stub_main = StubMainWindow(unified_editor=stub_editor)
    manager = GuidePerformanceManager(stub_main)
    manager._assembly_editor = stub_editor
    return manager


def _make_manager_with_window():
    """Create a manager with a StubWindow pre-attached (simulates active perf)."""
    manager = _make_manager()
    manager._window = StubWindow()
    return manager


# ============================================================================
# Commit 1: PerformanceState Enum + Persistent Window Lifecycle
# ============================================================================

class TestPerformanceStateEnum:
    """PerformanceState enum has all expected values."""

    def test_idle_exists(self):
        assert PerformanceState.IDLE.value == "idle"

    def test_playing_exists(self):
        assert PerformanceState.PLAYING.value == "playing"

    def test_paused_exists(self):
        assert PerformanceState.PAUSED.value == "paused"

    def test_stopped_exists(self):
        assert PerformanceState.STOPPED.value == "stopped"


class TestInitialState:
    """Manager starts in IDLE with no cached paths."""

    def test_starts_idle(self):
        manager = _make_manager()
        assert manager.performance_state == PerformanceState.IDLE

    def test_no_cached_stage_path(self):
        manager = _make_manager()
        assert manager._last_stage_path is None

    def test_no_cached_play_title(self):
        manager = _make_manager()
        assert manager._last_play_title is None


class TestStopPerformance:
    """stop_performance transitions to STOPPED, keeps window alive."""

    def test_stop_transitions_to_stopped(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.stop_performance()
        assert manager.performance_state == PerformanceState.STOPPED

    def test_window_survives_stop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.stop_performance()
        assert manager._window is not None

    def test_dialogue_dimmed_on_stop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.stop_performance()
        assert manager._window._dialogue_dimmed is True

    def test_input_disabled_on_stop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.stop_performance()
        assert manager._window._input_enabled is False

    def test_performers_cleaned_up_on_stop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager._ensemble_mode = True

        class _StubPerformer:
            def stop(self):
                pass

        manager._performers = {'test': _StubPerformer()}
        manager.stop_performance()
        assert manager._performers == {}
        assert manager._performer is None


class TestSetPerformanceState:
    """_set_performance_state updates window UI correctly."""

    def test_playing_enables_input(self):
        manager = _make_manager_with_window()
        manager._window._input_enabled = False
        manager._set_performance_state(PerformanceState.PLAYING)
        assert manager._window._input_enabled is True

    def test_paused_disables_input(self):
        manager = _make_manager_with_window()
        manager._set_performance_state(PerformanceState.PAUSED)
        assert manager._window._input_enabled is False

    def test_stopped_disables_input_and_dims(self):
        manager = _make_manager_with_window()
        manager._set_performance_state(PerformanceState.STOPPED)
        assert manager._window._input_enabled is False
        assert manager._window._dialogue_dimmed is True


class TestIsActive:
    """is_active reflects PLAYING/PAUSED states."""

    def test_idle_not_active(self):
        manager = _make_manager()
        assert manager.is_active is False

    def test_playing_is_active(self):
        manager = _make_manager()
        manager._performance_state = PerformanceState.PLAYING
        assert manager.is_active is True

    def test_paused_is_active(self):
        manager = _make_manager()
        manager._performance_state = PerformanceState.PAUSED
        assert manager.is_active is True

    def test_stopped_not_active(self):
        manager = _make_manager()
        manager._performance_state = PerformanceState.STOPPED
        assert manager.is_active is False


class TestStageCaching:
    """start_ensemble_from_stage caches stage_path and play_title."""

    def test_caches_stage_path(self):
        manager = _make_manager()
        # Call with invalid path -- won't find instances but will cache
        manager.start_ensemble_from_stage("/nonexistent/stage", "Test Title")
        assert manager._last_stage_path == "/nonexistent/stage"

    def test_caches_play_title(self):
        manager = _make_manager()
        manager.start_ensemble_from_stage("/nonexistent/stage", "My Show")
        assert manager._last_play_title == "My Show"


# ============================================================================
# Commit 2: Pause / Resume
# ============================================================================

class TestPauseResume:
    """Pause and resume state transitions."""

    def test_pause_from_playing(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.pause_ensemble()
        assert manager.performance_state == PerformanceState.PAUSED

    def test_resume_from_paused(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PAUSED
        manager.resume_ensemble()
        assert manager.performance_state == PerformanceState.PLAYING

    def test_pause_from_non_playing_is_noop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.STOPPED
        manager.pause_ensemble()
        assert manager.performance_state == PerformanceState.STOPPED

    def test_resume_from_non_paused_is_noop(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PLAYING
        manager.resume_ensemble()
        assert manager.performance_state == PerformanceState.PLAYING

    def test_advance_turn_blocked_when_paused(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PAUSED
        manager._turn_queue = ['ajo', 'krampus']
        manager._advance_ensemble_turn()
        # Queue should not have been consumed
        assert manager._turn_queue == ['ajo', 'krampus']

    def test_user_message_blocked_when_paused(self):
        manager = _make_manager_with_window()
        manager._performance_state = PerformanceState.PAUSED
        manager._ensemble_mode = True
        manager._pending_message = None
        manager._on_user_message("hello")
        # Message should not have been routed
        assert manager._pending_message is None


class TestPerformancePlayerPauseResume:
    """PerformancePlayer pause/resume freezes and continues typing."""

    def test_pause_stops_timer(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        player.play({'characters': [
            {'c': 'H', 'd': 35}, {'c': 'i', 'd': 35}
        ]})
        player.pause()
        assert player._paused is True
        assert not player._timer.isActive()

    def test_resume_continues(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        player._characters = [
            {'c': 'H', 'd': 35}, {'c': 'i', 'd': 35}
        ]
        player._index = 0
        player._paused = True
        # resume should clear pause and start revealing
        player.resume()
        assert player._paused is False

    def test_streaming_buffer_fills_during_pause(self, qapp):
        from noodlestudio.runtime.ui.performance_player import PerformancePlayer
        player = PerformancePlayer()
        player.start_streaming()
        # Pause before any tokens arrive
        player.pause()
        # Text without a newline goes into _line_buffer (pending flush)
        player.append_text("Hi")
        assert player._line_buffer == "Hi"
        assert player._paused is True
        # finish_streaming flushes the line buffer into stream buffer
        player.finish_streaming()
        # Streaming consumes first char synchronously; check suffix present
        assert "i" in player._stream_buffer


class TestNoodlingPerformerPauseAnimation:
    """NoodlingPerformer delegates pause/resume to player."""

    def test_pause_animation_without_player(self, qapp):
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        p = NoodlingPerformer(
            noodling_id='test', name='Test',
            llm_client=FakeLLMClient()
        )
        # Should not raise even with no player
        p.pause_animation()

    def test_resume_animation_without_player(self, qapp):
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        p = NoodlingPerformer(
            noodling_id='test', name='Test',
            llm_client=FakeLLMClient()
        )
        p.resume_animation()


# ============================================================================
# Commit 3: Transport Buttons
# ============================================================================

class _FakeButtonMixin:
    """Minimal stand-in providing transport button attributes."""

    def __init__(self):
        self._play_enabled = True
        self._pause_enabled = False
        self._stop_enabled = False

    class _Btn:
        def __init__(self, enabled=True):
            self._enabled = enabled
        def setEnabled(self, val):
            self._enabled = val
        def isEnabled(self):
            return self._enabled


def _make_button_mixin_with_manager():
    """Create a mixin-like object with transport buttons and a manager."""
    from noodlestudio.core.main_window_project_mixin import MainWindowProjectMixin

    manager = _make_manager_with_window()

    class FakeHost(MainWindowProjectMixin):
        def __init__(self):
            self.guide_performance_manager = manager
            self._play_button = _FakeButtonMixin._Btn(True)
            self._pause_button = _FakeButtonMixin._Btn(False)
            self._stop_button = _FakeButtonMixin._Btn(False)

    host = FakeHost()
    return host, manager


class TestSyncTransportButtons:
    """_sync_transport_buttons enables correct buttons per state."""

    def test_idle_state(self):
        host, manager = _make_button_mixin_with_manager()
        manager._performance_state = PerformanceState.IDLE
        host._sync_transport_buttons()
        assert host._play_button.isEnabled() is True
        assert host._pause_button.isEnabled() is False
        assert host._stop_button.isEnabled() is False

    def test_playing_state(self):
        host, manager = _make_button_mixin_with_manager()
        manager._performance_state = PerformanceState.PLAYING
        host._sync_transport_buttons()
        assert host._play_button.isEnabled() is False
        assert host._pause_button.isEnabled() is True
        assert host._stop_button.isEnabled() is True

    def test_paused_state(self):
        host, manager = _make_button_mixin_with_manager()
        manager._performance_state = PerformanceState.PAUSED
        host._sync_transport_buttons()
        assert host._play_button.isEnabled() is True
        assert host._pause_button.isEnabled() is False
        assert host._stop_button.isEnabled() is True

    def test_stopped_state(self):
        host, manager = _make_button_mixin_with_manager()
        manager._performance_state = PerformanceState.STOPPED
        host._sync_transport_buttons()
        assert host._play_button.isEnabled() is True
        assert host._pause_button.isEnabled() is False
        assert host._stop_button.isEnabled() is False


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
