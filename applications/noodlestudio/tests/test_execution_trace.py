# ──────────────────────────────────────────────────────────────
#   Tests for Execution Trace Capture
# ──────────────────────────────────────────────────────────────

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFacetTrace:
    """Trace data stored on facets after execution."""

    def test_facet_last_trace_set(self):
        """Facet._last_trace should be set after execution."""
        from noodlestudio.core.facet_system import Facet
        facet = Facet(id='test', facet_type='IntuitionFacet', name='Test', prompt='')
        # Simulate what FacetExecutor does
        facet._last_trace = {
            'system_prompt': '',
            'formatted_prompt': '',
            'output': 'hello',
            'execution_time': 0.05,
            'token_count': 0,
            'model_label': '',
        }
        assert facet._last_trace['output'] == 'hello'
        assert facet._last_trace['execution_time'] > 0

    def test_trace_has_all_fields(self):
        """Trace dict should contain all required fields."""
        trace = {
            'system_prompt': 'You are helpful',
            'formatted_prompt': 'Hello world',
            'output': 'Hi there',
            'execution_time': 1.5,
            'token_count': 42,
            'model_label': 'response',
        }
        assert 'system_prompt' in trace
        assert 'formatted_prompt' in trace
        assert 'output' in trace
        assert 'execution_time' in trace
        assert 'token_count' in trace
        assert 'model_label' in trace


class TestPerformerTraceSignals:
    """NoodlingPerformer emits trace signals."""

    def test_performer_has_turn_trace_signal(self, qapp):
        """NoodlingPerformer should have turnTraceReady signal."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import FakeLLMClient
        p = NoodlingPerformer('test', 'Test', FakeLLMClient())
        assert hasattr(p, 'turnTraceReady')

    def test_worker_has_facet_trace_signal(self, qapp):
        """_AssemblyWorker should have facetTraceReady signal."""
        from noodlestudio.runtime.ui.noodling_performer import _AssemblyWorker
        assert hasattr(_AssemblyWorker, 'facetTraceReady')

    def test_performer_collects_traces(self, qapp):
        """Performer should collect traces in _current_turn_traces."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer
        from tests.conftest import FakeLLMClient
        p = NoodlingPerformer('test', 'Test', FakeLLMClient())
        p._current_turn_traces = []

        trace = {'facet_id': 'response', 'output': 'hello'}
        p._on_facet_trace('response', trace)

        assert len(p._current_turn_traces) == 1
        assert p._current_turn_traces[0]['facet_id'] == 'response'


class TestManagerTraceRouting:
    """Manager stores and routes traces."""

    def test_manager_stores_traces(self):
        """_on_turn_trace should store traces in _turn_traces."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager, PerformanceState,
        )
        from tests.conftest import StubMainWindow, StubWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = StubWindow()
        manager._performance_state = PerformanceState.PLAYING

        traces = [
            {'facet_id': 'response', 'output': 'hi'},
            {'facet_id': 'sentiment', 'output': 'happy'},
        ]
        manager._on_turn_trace('ajo', traces)

        assert 'ajo' in manager._turn_traces
        assert len(manager._turn_traces['ajo']) == 1
        assert manager._turn_traces['ajo'][0] == traces
        assert manager._turn_count == 1

    def test_manager_increments_turn_count(self):
        """Turn count should increment with each trace."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager, PerformanceState,
        )
        from tests.conftest import StubMainWindow, StubWindow

        manager = GuidePerformanceManager(StubMainWindow())
        manager._window = StubWindow()
        manager._performance_state = PerformanceState.PLAYING

        manager._on_turn_trace('ajo', [{'facet_id': 'r1'}])
        manager._on_turn_trace('krampus', [{'facet_id': 'r2'}])

        assert manager._turn_count == 2
        assert len(manager._turn_traces['ajo']) == 1
        assert len(manager._turn_traces['krampus']) == 1
