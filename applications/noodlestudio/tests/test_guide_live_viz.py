# ------------------------------------------------------------------
#   Tests for Guide Performance Live Visualization
#
#   Tests that the facets editor receives execution events when Ajo's
#   assembly runs -- nodes pulse, wires animate, cycle completes.
#   Bypasses WebSocket; events are delivered directly.
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

import json
from dataclasses import dataclass, field
from typing import Dict, Any, List
from unittest.mock import MagicMock, patch, call
import pytest


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class MockExecutionResult:
    """Mock of facet_executor.ExecutionResult."""
    response: str = "Hello from assembly"
    total_time: float = 0.5
    total_tokens: int = 50
    facets_executed: int = 2
    facets_skipped: int = 0
    facet_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    facet_times: Dict[str, float] = field(default_factory=dict)
    facet_tokens: Dict[str, int] = field(default_factory=dict)
    execution_id: str = ""


def _make_manager(with_editor=True):
    """Create a GuidePerformanceManager with mock deps for testing."""
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
    )

    manager = GuidePerformanceManager.__new__(GuidePerformanceManager)
    manager._window = MagicMock()
    manager._conversation_history = []
    manager._last_user_message = "Hello"
    manager._guide_cue_handler = None
    manager._worker = None
    manager._current_execution_id = None

    # Mock main window with optional facets editor
    manager._main_window = MagicMock()
    mock_editor = MagicMock() if with_editor else None
    manager._main_window.facets_editor = mock_editor
    manager._facets_editor = mock_editor

    return manager


# =============================================================================
# Event Delivery Tests
# =============================================================================

class TestEventDelivery:
    """Tests that _emit_execution_event reaches the facets editor."""

    def test_event_delivered_to_editor(self):
        """Events are passed to facets editor's _handle_execution_event."""
        manager = _make_manager()
        event = {'type': 'facet_execution', 'subtype': 'cycle_start'}

        manager._emit_execution_event(event)

        manager._facets_editor._handle_execution_event.assert_called_once_with(event)

    def test_no_crash_without_editor(self):
        """No crash when facets editor is not available."""
        manager = _make_manager(with_editor=False)
        manager._facets_editor = None
        manager._main_window.facets_editor = None

        # Should not raise
        manager._emit_execution_event({'type': 'facet_execution', 'subtype': 'cycle_start'})

    def test_no_crash_without_main_window(self):
        """No crash when _main_window is missing (test via __new__)."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )

        manager = GuidePerformanceManager.__new__(GuidePerformanceManager)
        # Intentionally don't set _main_window or _facets_editor

        # _get_facets_editor uses getattr, so this should not raise
        editor = manager._get_facets_editor()
        assert editor is None

    def test_editor_cached_after_first_lookup(self):
        """Facets editor is cached after first successful lookup."""
        manager = _make_manager()
        manager._facets_editor = None  # Force re-lookup

        editor1 = manager._get_facets_editor()
        editor2 = manager._get_facets_editor()

        assert editor1 is editor2
        assert editor1 is manager._main_window.facets_editor


# =============================================================================
# Start Event Tests
# =============================================================================

class TestStartEvents:
    """Tests for _emit_execution_start_events."""

    def test_cycle_start_emitted(self):
        """cycle_start event is emitted when execution begins."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"
        manager._last_user_message = "Hello Ajo"

        manager._emit_execution_start_events()

        # First call should be cycle_start
        calls = manager._facets_editor._handle_execution_event.call_args_list
        assert len(calls) >= 1
        first_event = calls[0][0][0]
        assert first_event['type'] == 'facet_execution'
        assert first_event['subtype'] == 'cycle_start'
        assert first_event['execution_id'] == 'abc123'

    def test_incoming_facet_start_emitted(self):
        """facet_start for 'incoming' is emitted."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"
        manager._last_user_message = "Hello Ajo"

        manager._emit_execution_start_events()

        calls = manager._facets_editor._handle_execution_event.call_args_list
        start_events = [
            c[0][0] for c in calls
            if c[0][0].get('subtype') == 'facet_start'
            and c[0][0].get('source_id') == 'incoming'
        ]
        assert len(start_events) == 1
        assert start_events[0]['data']['inputs']['user_message'] == 'Hello Ajo'

    def test_no_events_without_execution_id(self):
        """No events emitted when execution_id is None."""
        manager = _make_manager()
        manager._current_execution_id = None

        manager._emit_execution_start_events()

        manager._facets_editor._handle_execution_event.assert_not_called()

    def test_execution_id_consistency(self):
        """All start events share the same execution_id."""
        manager = _make_manager()
        manager._current_execution_id = "xyz789"
        manager._last_user_message = "test"

        manager._emit_execution_start_events()

        calls = manager._facets_editor._handle_execution_event.call_args_list
        for c in calls:
            event = c[0][0]
            eid = event.get('execution_id') or event.get('data', {}).get('execution_id')
            assert eid == "xyz789", f"Event {event['subtype']} has wrong execution_id"


# =============================================================================
# Complete Event Tests
# =============================================================================

class TestCompleteEvents:
    """Tests for _emit_execution_complete_events."""

    def test_response_and_sentiment_complete(self):
        """facet_complete emitted for both response and sentiment."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        result = MockExecutionResult(
            facet_outputs={
                'response': {'out': 'Hello!'},
                'sentiment': {'out': '{"valence": 0.8}'},
            }
        )

        manager._emit_execution_complete_events(result)

        calls = manager._facets_editor._handle_execution_event.call_args_list
        complete_events = [
            c[0][0] for c in calls
            if c[0][0].get('subtype') == 'facet_complete'
        ]

        facet_ids = {e['source_id'] for e in complete_events}
        assert 'response' in facet_ids
        assert 'sentiment' in facet_ids

    def test_data_flows_to_outgoing(self):
        """data_flow events emitted from response and sentiment to outgoing."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        result = MockExecutionResult()
        manager._emit_execution_complete_events(result)

        calls = manager._facets_editor._handle_execution_event.call_args_list
        flow_events = [
            c[0][0] for c in calls
            if c[0][0].get('subtype') == 'data_flow'
        ]

        sources = {e['from_facet'] for e in flow_events}
        assert 'response' in sources
        assert 'sentiment' in sources
        for e in flow_events:
            assert e['to_facet'] == 'outgoing'

    def test_outputs_included_in_complete(self):
        """Facet outputs from ExecutionResult are included in complete events."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        result = MockExecutionResult(
            facet_outputs={
                'response': {'out': 'Ajo says hi'},
                'sentiment': {'out': '{"valence": 0.9}'},
            }
        )

        manager._emit_execution_complete_events(result)

        calls = manager._facets_editor._handle_execution_event.call_args_list
        response_complete = [
            c[0][0] for c in calls
            if c[0][0].get('subtype') == 'facet_complete'
            and c[0][0].get('source_id') == 'response'
        ]

        assert len(response_complete) == 1
        assert response_complete[0]['data']['outputs'] == {'out': 'Ajo says hi'}

    def test_no_events_without_execution_id(self):
        """No completion events when execution_id is None."""
        manager = _make_manager()
        manager._current_execution_id = None

        result = MockExecutionResult()
        manager._emit_execution_complete_events(result)

        manager._facets_editor._handle_execution_event.assert_not_called()


# =============================================================================
# Error Event Tests
# =============================================================================

class TestErrorEvents:
    """Tests for _emit_execution_error_events."""

    def test_error_emitted_for_processing_facets(self):
        """facet_error emitted for response and sentiment nodes."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        manager._emit_execution_error_events("LLM timeout")

        calls = manager._facets_editor._handle_execution_event.call_args_list
        error_events = [
            c[0][0] for c in calls
            if c[0][0].get('subtype') == 'facet_error'
        ]

        assert len(error_events) == 2
        facet_ids = {e['source_id'] for e in error_events}
        assert facet_ids == {'response', 'sentiment'}

        for e in error_events:
            assert e['data']['error'] == 'LLM timeout'

    def test_cycle_complete_after_error(self):
        """cycle_complete is emitted after error events."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        manager._emit_execution_error_events("fail")

        calls = manager._facets_editor._handle_execution_event.call_args_list
        last_event = calls[-1][0][0]
        assert last_event['subtype'] == 'cycle_complete'

    def test_execution_id_cleared_after_error(self):
        """Execution ID is cleared after error events."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        manager._emit_execution_error_events("fail")

        assert manager._current_execution_id is None

    def test_no_events_without_execution_id(self):
        """No error events when execution_id is None."""
        manager = _make_manager()
        manager._current_execution_id = None

        manager._emit_execution_error_events("fail")

        manager._facets_editor._handle_execution_event.assert_not_called()


# =============================================================================
# Integration: Message -> Events Flow
# =============================================================================

class TestMessageToEventsFlow:
    """Tests that user message triggers execution events."""

    def test_message_creates_execution_id(self):
        """Sending a message creates a non-None execution_id."""
        manager = _make_manager()
        manager._assembly = MagicMock()
        manager._executor = MagicMock()

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            manager._on_user_message_for_assembly("Hello")

            assert manager._current_execution_id is not None
            assert len(manager._current_execution_id) == 8

    def test_message_emits_start_events(self):
        """Sending a message emits start events to facets editor."""
        manager = _make_manager()
        manager._assembly = MagicMock()
        manager._executor = MagicMock()

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            manager._on_user_message_for_assembly("Hello")

            # At least cycle_start and incoming facet_start should fire
            assert manager._facets_editor._handle_execution_event.call_count >= 2

    def test_result_emits_complete_events(self):
        """Assembly result emits completion events to facets editor."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        result = MockExecutionResult(
            response="Hi there!",
            facet_outputs={
                'response': {'out': 'Hi there!'},
                'sentiment': {'out': '{"valence": 0.7}'},
            }
        )

        manager._on_assembly_result(result)

        # Should have facet_complete events + data_flow events
        calls = manager._facets_editor._handle_execution_event.call_args_list
        event_types = [c[0][0]['subtype'] for c in calls]
        assert 'facet_complete' in event_types
        assert 'data_flow' in event_types

    def test_error_emits_error_events(self):
        """Assembly error emits error events to facets editor."""
        manager = _make_manager()
        manager._current_execution_id = "abc123"

        manager._on_assembly_error("Connection timeout")

        calls = manager._facets_editor._handle_execution_event.call_args_list
        event_types = [c[0][0]['subtype'] for c in calls]
        assert 'facet_error' in event_types
        assert 'cycle_complete' in event_types


# =============================================================================
# Tab Switching Tests
# =============================================================================

class TestTabSwitching:
    """Tests for Facets Editor tab activation on performance start."""

    def test_tab_switch_code_path(self):
        """Verify tab switching logic finds Facets Editor tab."""
        # Simulate the tab switching logic from start_performance
        mock_tabs = MagicMock()
        mock_tabs.count.return_value = 5
        mock_tabs.tabText.side_effect = [
            "Noodle Code", "Text View", "Spatial View", "Facets Editor", "Settings"
        ]

        # This mirrors the logic in start_performance
        for i in range(mock_tabs.count()):
            if mock_tabs.tabText(i) == "Facets Editor":
                mock_tabs.setCurrentIndex(i)
                break

        mock_tabs.setCurrentIndex.assert_called_once_with(3)


# =============================================================================
# Stale Execution Guard Tests
# =============================================================================

class TestStaleExecutionGuard:
    """Tests that stale timer callbacks are discarded."""

    def test_complete_events_skipped_when_id_changes(self):
        """Completion events are skipped if execution_id has changed."""
        manager = _make_manager()
        manager._current_execution_id = "old_id"

        result = MockExecutionResult()

        # Simulate: before calling, the execution_id changes
        # (e.g., new message sent)
        manager._current_execution_id = None

        manager._emit_execution_complete_events(result)

        manager._facets_editor._handle_execution_event.assert_not_called()

    def test_error_events_skipped_when_id_cleared(self):
        """Error events are skipped if execution_id is already None."""
        manager = _make_manager()
        manager._current_execution_id = None

        manager._emit_execution_error_events("fail")

        manager._facets_editor._handle_execution_event.assert_not_called()


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Made with love. Use with love.
# Caitlyn Meeks 2026
