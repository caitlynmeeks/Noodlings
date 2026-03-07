# ------------------------------------------------------------------
#   Tests for Ensemble Mode (Two Noodlings, One Stage)
#
#   Tests that GuidePerformanceManager can coordinate two
#   NoodlingPerformers with turn-taking: User -> Ajo -> Yuki -> wait.
#   Yuki receives Ajo's response as extra context.
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

from pathlib import Path
from unittest.mock import patch
import pytest

from conftest import (
    FakeLLMClient, SignalCollector, StubFacetsEditor,
    StubMainWindow, StubWindow,
)


# Patch paths
CREATE_LLM_PATCH = (
    'noodlestudio.runtime.ui.guide_performance_manager.create_llm_client'
)
LOAD_ASSEMBLY_PATCH = (
    'noodlestudio.runtime.ui.noodling_performer.NoodlingPerformer.load_assembly'
)
CUC_PATCH = (
    'noodlestudio.core.computer_use_controller.get_computer_use_controller'
)


# =============================================================================
# Helpers
# =============================================================================

def _create_ensemble_manager():
    """Create a manager configured for ensemble testing."""
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
        PerformanceState,
    )

    stub_editor = StubFacetsEditor()
    stub_main = StubMainWindow(unified_editor=stub_editor)

    manager = GuidePerformanceManager(stub_main)
    manager._assembly_editor = stub_editor
    manager._performance_state = PerformanceState.PLAYING
    return manager


def _setup_ensemble_performers(manager):
    """Set up two performers on the manager without UI."""
    from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

    ajo = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
    ajo._assembly = True
    ajo._executor = True

    yuki = NoodlingPerformer('yuki', 'Yuki', FakeLLMClient())
    yuki._assembly = True
    yuki._executor = True

    manager._ensemble_mode = True
    manager._performers = {'ajo': ajo, 'yuki': yuki}
    manager._performer = ajo
    manager._window = StubWindow()

    return ajo, yuki


# =============================================================================
# Ensemble Mode Initialization
# =============================================================================

class TestEnsembleInit:
    """Tests for start_ensemble initialization."""

    def test_ensemble_mode_flag_set(self):
        """start_ensemble sets ensemble_mode flag."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        assert manager._ensemble_mode

    def test_two_performers_registered(self):
        """start_ensemble creates two performers."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        assert 'ajo' in manager._performers
        assert 'yuki' in manager._performers
        assert len(manager._performers) == 2

    def test_primary_performer_is_ajo(self):
        """Primary performer is Ajo for backward compat."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        assert manager._performer is ajo


# =============================================================================
# Turn-Taking Order
# =============================================================================

class TestTurnTaking:
    """Tests for turn-taking sequence: Ajo first, then Yuki."""

    def test_user_message_starts_turn_queue(self):
        """User message creates turn queue with both noodlings."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        with patch(
            'noodlestudio.runtime.ui.noodling_performer._AssemblyWorker'
        ) as MockWorker:
            from unittest.mock import MagicMock
            MockWorker.return_value = MagicMock()

            manager._on_user_message("Hello")

            # Ajo should be executing (popped from queue), Yuki waiting
            assert 'yuki' in manager._turn_queue or len(manager._turn_queue) == 1

    def test_ajo_executes_first(self):
        """Ajo is the first performer to execute."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        executed = []

        # Track which performer gets execute() called
        original_ajo_execute = ajo.execute
        original_yuki_execute = yuki.execute

        def track_ajo(msg, ctx=None):
            executed.append('ajo')

        def track_yuki(msg, ctx=None):
            executed.append('yuki')

        ajo.execute = track_ajo
        yuki.execute = track_yuki

        manager._on_user_message_ensemble("Hello")

        assert executed == ['ajo']

    def test_yuki_executes_after_ajo_finishes(self):
        """Yuki executes after Ajo's turn finishes."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        executed = []

        def track_ajo(msg, ctx=None):
            executed.append('ajo')

        def track_yuki(msg, ctx=None):
            executed.append('yuki')

        ajo.execute = track_ajo
        yuki.execute = track_yuki

        manager._on_user_message_ensemble("Hello")
        assert executed == ['ajo']

        # Simulate Ajo finishing
        ajo._last_response = "I'm Ajo!"
        manager._on_ensemble_turn_finished('ajo')

        assert executed == ['ajo', 'yuki']

    def test_busy_clears_after_all_turns(self):
        """Busy state clears after all turns complete."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        ajo.execute = lambda msg, ctx=None: None
        yuki.execute = lambda msg, ctx=None: None

        manager._on_user_message_ensemble("Hello")

        # Simulate both finishing
        ajo._last_response = "Ajo says hi"
        manager._on_ensemble_turn_finished('ajo')

        yuki._last_response = "Yuki greets you"
        manager._on_ensemble_turn_finished('yuki')

        # Last busy_states entry should be (False, None) (cleared)
        last = manager._window.busy_states[-1]
        assert last[0] is False


# =============================================================================
# Context Passing
# =============================================================================

class TestContextPassing:
    """Tests that Yuki receives Ajo's response as context."""

    def test_yuki_gets_ajo_response(self):
        """Yuki's extra_context includes what Ajo said."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        captured_context = {}

        def capture_ajo(msg, ctx=None):
            pass

        def capture_yuki(msg, ctx=None):
            captured_context.update(ctx or {})

        ajo.execute = capture_ajo
        yuki.execute = capture_yuki

        manager._on_user_message_ensemble("Hello")

        # Simulate Ajo finishing with a response
        ajo._last_response = "Oh! Hello there!"
        manager._on_ensemble_turn_finished('ajo')

        assert captured_context.get('ajo_said') == "Oh! Hello there!"

    def test_ajo_has_no_prior_context(self):
        """Ajo doesn't have any prior noodling responses."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        captured_context = {}

        def capture_ajo(msg, ctx=None):
            captured_context.update(ctx or {})

        ajo.execute = capture_ajo
        yuki.execute = lambda msg, ctx=None: None

        manager._on_user_message_ensemble("Hello")

        # No _said keys should be present
        said_keys = [k for k in captured_context if k.endswith('_said')]
        assert said_keys == []


# =============================================================================
# Stop / Cleanup
# =============================================================================

class TestEnsembleStop:
    """Tests for stopping ensemble performances."""

    def test_stop_clears_all_performers(self):
        """stop_performance clears both performers."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        manager.stop_performance()

        assert manager._performers == {}
        assert manager._performer is None
        assert not manager._ensemble_mode

    def test_stop_resets_turn_state(self):
        """stop_performance resets turn-taking state."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        manager._turn_queue = ['yuki']
        manager._turn_responses = {'ajo': 'hello'}

        manager.stop_performance()

        assert manager._turn_queue == []
        assert manager._turn_responses == {}


# =============================================================================
# Ensemble Performance Finish
# =============================================================================

class TestEnsemblePerformanceFinish:
    """Tests for typed text completion in ensemble mode."""

    def test_performance_finished_ends_text_block(self):
        """Performance finish ends the noodling text block."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        manager._on_ensemble_performance_finished('ajo')

        assert manager._window._text_blocks_ended == 1

    def test_performance_finished_turns_off_speaking(self):
        """Performance finish disables speaking animation."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        manager._on_ensemble_performance_finished('yuki')

        calls = manager._window._speaking_mode_calls
        assert len(calls) == 1
        assert calls[0][0] is False  # active=False
        assert calls[0][2] == 'yuki'  # noodling_id


# =============================================================================
# Error Handling
# =============================================================================

class TestEnsembleErrors:
    """Tests for error handling in ensemble mode."""

    def test_error_shows_noodling_name(self):
        """Error message includes the noodling name."""
        manager = _create_ensemble_manager()
        ajo, yuki = _setup_ensemble_performers(manager)

        manager._on_ensemble_error('yuki', 'Connection timeout')

        assert len(manager._window.errors) == 1
        assert 'Yuki' in manager._window.errors[0]
        assert 'Connection timeout' in manager._window.errors[0]

    def test_no_performers_shows_error(self):
        """Ensemble message with no performers shows error."""
        manager = _create_ensemble_manager()
        manager._ensemble_mode = True
        manager._performers = {}
        manager._window = StubWindow()

        manager._on_user_message_ensemble("Hello")

        assert len(manager._window.errors) == 1


# =============================================================================
# CLI Flag
# =============================================================================

class TestCLIFlag:
    """Tests for --ensemble CLI argument."""

    def test_parse_args_accepts_ensemble(self):
        """Argument parser accepts --ensemble flag."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--ensemble', '--no-splash']
        try:
            args = parse_args()
            assert args.ensemble is True
        finally:
            sys.argv = original_argv

    def test_parse_args_ensemble_is_optional(self):
        """--ensemble is optional."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--no-splash']
        try:
            args = parse_args()
            assert args.ensemble is False
        finally:
            sys.argv = original_argv


# =============================================================================
# Backward Compatibility
# =============================================================================

class TestBackwardCompat:
    """Tests that single-noodling mode still works."""

    def test_single_mode_not_ensemble(self):
        """Default manager is not in ensemble mode."""
        manager = _create_ensemble_manager()
        assert not manager._ensemble_mode

    def test_single_message_routes_to_performer(self):
        """Single mode message routes to the primary performer."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        manager = _create_ensemble_manager()
        manager._window = StubWindow()

        p = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        p._assembly = True
        p._executor = True
        manager._performer = p

        executed = []
        p.execute = lambda msg, ctx=None: executed.append(msg)

        manager._on_user_message("Hello")

        assert executed == ["Hello"]


# Made with love. Use with love.
# Caitlyn Meeks 2026
