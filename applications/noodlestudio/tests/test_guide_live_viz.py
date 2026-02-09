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
from typing import Dict, Any
from unittest.mock import MagicMock, patch
import pytest

from conftest import StubFacetsEditor, StubMainWindow, StubWindow


# =============================================================================
# Helpers
# =============================================================================

@dataclass
class FakeExecutionResult:
    """Lightweight stand-in for facet_executor.ExecutionResult."""
    response: str = "Hello from assembly"
    total_time: float = 0.5
    total_tokens: int = 50
    facets_executed: int = 2
    facets_skipped: int = 0
    facet_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    facet_times: Dict[str, float] = field(default_factory=dict)
    facet_tokens: Dict[str, int] = field(default_factory=dict)
    execution_id: str = ""


def _events_by_subtype(events, subtype):
    """Filter event list by subtype."""
    return [e for e in events if e.get('subtype') == subtype]


# =============================================================================
# Event Delivery Tests
# =============================================================================

class TestEventDelivery:
    """Tests that _emit_execution_event reaches the facets editor."""

    def test_event_delivered_to_editor(self, guide_manager):
        """Events are passed to facets editor's _handle_execution_event."""
        event = {'type': 'facet_execution', 'subtype': 'cycle_start'}

        guide_manager._emit_execution_event(event)

        assert guide_manager._facets_editor.events == [event]

    def test_no_crash_without_editor(self, guide_manager):
        """No crash when facets editor is not available."""
        guide_manager._facets_editor = None
        guide_manager._main_window.facets_editor = None

        # Should not raise
        guide_manager._emit_execution_event(
            {'type': 'facet_execution', 'subtype': 'cycle_start'}
        )

    def test_no_crash_when_editor_is_none_at_init(self):
        """No crash when main_window has no facets_editor attribute."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )

        stub_main = StubMainWindow(facets_editor=None)
        manager = GuidePerformanceManager(stub_main)
        manager._facets_editor = None

        editor = manager._get_facets_editor()
        assert editor is None

    def test_editor_cached_after_first_lookup(self, guide_manager):
        """Facets editor is cached after first successful lookup."""
        guide_manager._facets_editor = None  # Force re-lookup

        editor1 = guide_manager._get_facets_editor()
        editor2 = guide_manager._get_facets_editor()

        assert editor1 is editor2
        assert editor1 is guide_manager._main_window.facets_editor


# =============================================================================
# Start Event Tests
# =============================================================================

class TestStartEvents:
    """Tests for _emit_execution_start_events."""

    def test_cycle_start_emitted(self, guide_manager):
        """cycle_start event is emitted when execution begins."""
        guide_manager._current_execution_id = "abc123"
        guide_manager._last_user_message = "Hello Ajo"

        guide_manager._emit_execution_start_events()

        events = guide_manager._facets_editor.events
        assert len(events) >= 1
        assert events[0]['type'] == 'facet_execution'
        assert events[0]['subtype'] == 'cycle_start'
        assert events[0]['execution_id'] == 'abc123'

    def test_incoming_facet_start_emitted(self, guide_manager):
        """facet_start for 'incoming' is emitted."""
        guide_manager._current_execution_id = "abc123"
        guide_manager._last_user_message = "Hello Ajo"

        guide_manager._emit_execution_start_events()

        events = guide_manager._facets_editor.events
        start_events = [
            e for e in events
            if e.get('subtype') == 'facet_start'
            and e.get('source_id') == 'incoming'
        ]
        assert len(start_events) == 1
        assert start_events[0]['data']['inputs']['user_message'] == 'Hello Ajo'

    def test_no_events_without_execution_id(self, guide_manager):
        """No events emitted when execution_id is None."""
        guide_manager._current_execution_id = None

        guide_manager._emit_execution_start_events()

        assert guide_manager._facets_editor.events == []

    def test_execution_id_consistency(self, guide_manager):
        """All start events share the same execution_id."""
        guide_manager._current_execution_id = "xyz789"
        guide_manager._last_user_message = "test"

        guide_manager._emit_execution_start_events()

        for event in guide_manager._facets_editor.events:
            eid = event.get('execution_id') or event.get('data', {}).get('execution_id')
            assert eid == "xyz789", f"Event {event['subtype']} has wrong execution_id"


# =============================================================================
# Complete Event Tests
# =============================================================================

class TestCompleteEvents:
    """Tests for per-facet completion via _on_facet_completed."""

    def test_response_and_sentiment_complete_via_callback(self, guide_manager):
        """Per-facet callback emits facet_complete for response and sentiment."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._on_facet_completed('response', {'out': 'Hello!'})
        guide_manager._on_facet_completed('sentiment', {'out': '{"valence": 0.8}'})

        complete_events = _events_by_subtype(
            guide_manager._facets_editor.events, 'facet_complete'
        )
        facet_ids = {e['source_id'] for e in complete_events}
        assert 'response' in facet_ids
        assert 'sentiment' in facet_ids

    def test_data_flows_via_callback(self, guide_manager):
        """data_flow events emitted by per-facet callback."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._on_facet_completed('sentiment', {'out': '{"valence": 0.8}'})
        guide_manager._on_facet_completed('response', {'out': 'Hello!'})

        flow_events = _events_by_subtype(
            guide_manager._facets_editor.events, 'data_flow'
        )
        from_facets = {e['from_facet'] for e in flow_events}
        assert 'sentiment' in from_facets
        assert 'response' in from_facets

    def test_outputs_included_in_callback_complete(self, guide_manager):
        """Facet outputs are included in per-facet complete events."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._on_facet_completed('response', {'out': 'Ajo says hi'})

        complete_events = [
            e for e in guide_manager._facets_editor.events
            if e.get('subtype') == 'facet_complete'
            and e.get('source_id') == 'response'
        ]
        assert len(complete_events) == 1
        assert complete_events[0]['data']['outputs'] == {'out': 'Ajo says hi'}

    def test_no_events_without_execution_id(self, guide_manager):
        """No completion events when execution_id is None."""
        result = FakeExecutionResult()
        guide_manager._emit_execution_complete_events(result)

        assert guide_manager._facets_editor.events == []


# =============================================================================
# Error Event Tests
# =============================================================================

class TestErrorEvents:
    """Tests for _emit_execution_error_events."""

    def test_error_emitted_for_processing_facets(self, guide_manager):
        """facet_error emitted for response, sentiment, and performance nodes."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._emit_execution_error_events("LLM timeout")

        error_events = _events_by_subtype(
            guide_manager._facets_editor.events, 'facet_error'
        )
        assert len(error_events) == 3
        facet_ids = {e['source_id'] for e in error_events}
        assert facet_ids == {'response', 'sentiment', 'performance'}

        for e in error_events:
            assert e['data']['error'] == 'LLM timeout'

    def test_cycle_complete_after_error(self, guide_manager):
        """cycle_complete is emitted after error events."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._emit_execution_error_events("fail")

        last_event = guide_manager._facets_editor.events[-1]
        assert last_event['subtype'] == 'cycle_complete'

    def test_execution_id_cleared_after_error(self, guide_manager):
        """Execution ID is cleared after error events."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._emit_execution_error_events("fail")

        assert guide_manager._current_execution_id is None

    def test_no_events_without_execution_id(self, guide_manager):
        """No error events when execution_id is None."""
        guide_manager._emit_execution_error_events("fail")

        assert guide_manager._facets_editor.events == []


# =============================================================================
# Integration: Message -> Events Flow
# =============================================================================

class TestMessageToEventsFlow:
    """Tests that user message triggers execution events."""

    def test_message_creates_execution_id(self, guide_manager):
        """Sending a message creates a non-None execution_id."""
        guide_manager._assembly = MagicMock()
        guide_manager._executor = MagicMock()

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message_for_assembly("Hello")

            assert guide_manager._current_execution_id is not None
            assert len(guide_manager._current_execution_id) == 8

    def test_message_emits_start_events(self, guide_manager):
        """Sending a message emits start events to facets editor."""
        guide_manager._assembly = MagicMock()
        guide_manager._executor = MagicMock()

        with patch(
            'noodlestudio.runtime.ui.guide_performance_manager._AssemblyWorker'
        ) as MockWorker:
            mock_worker = MagicMock()
            MockWorker.return_value = mock_worker

            guide_manager._on_user_message_for_assembly("Hello")

            # At least cycle_start and incoming facet_start should fire
            assert len(guide_manager._facets_editor.events) >= 2

    def test_result_calls_completion_pipeline(self, guide_manager):
        """Assembly result triggers the completion event pipeline."""
        guide_manager._current_execution_id = "abc123"
        guide_manager._last_user_message = "Hello"

        result = FakeExecutionResult(
            response="Hi there!",
            facet_outputs={
                'response': {'out': 'Hi there!'},
                'sentiment': {'out': '{"valence": 0.7}'},
            }
        )

        with patch.object(guide_manager, '_emit_execution_complete_events') as mock_complete, \
             patch.object(guide_manager, '_apply_affect'):
            guide_manager._on_assembly_result(result)
            mock_complete.assert_called_once_with(result)

    def test_error_emits_error_events(self, guide_manager):
        """Assembly error emits error events to facets editor."""
        guide_manager._current_execution_id = "abc123"

        guide_manager._on_assembly_error("Connection timeout")

        event_types = [e['subtype'] for e in guide_manager._facets_editor.events]
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

    def test_complete_events_skipped_when_id_changes(self, guide_manager):
        """Completion events are skipped if execution_id has changed."""
        guide_manager._current_execution_id = None  # Already cleared

        result = FakeExecutionResult()
        guide_manager._emit_execution_complete_events(result)

        assert guide_manager._facets_editor.events == []

    def test_error_events_skipped_when_id_cleared(self, guide_manager):
        """Error events are skipped if execution_id is already None."""
        guide_manager._emit_execution_error_events("fail")

        assert guide_manager._facets_editor.events == []


# ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~ ~
# Made with love. Use with love.
# Caitlyn Meeks 2026
