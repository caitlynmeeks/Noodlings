"""Tests for execution visualization in AssemblyEditorView (C.4).

Covers: node state machine, wire packet animation, event dispatch,
cycle tracking, ensemble filtering, scene transition lock, sound toggle,
pause/resume, timer cleanup.

Uses real FacetAssembly objects (no mocks per project policy).
"""

import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer
from PyQt6.QtGui import QColor

from noodlestudio.core.facet_system import (
    Facet, FacetAssembly, FacetConnection,
)
from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
from noodlestudio.panels.editors.assembly_graphics_items import (
    FacetNodeItem, FacetConnectionItem,
)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_assembly():
    """Create a 3-node INCOMING -> LLM -> OUTGOING assembly."""
    incoming = Facet(
        id="incoming", name="INCOMING", facet_type="INCOMING",
        prompt="", position={'x': 100, 'y': 200}
    )
    incoming.add_output_pad("out", "Output")

    llm = Facet(
        id="response", name="Response", facet_type="LLMFacet",
        prompt="Think carefully.", position={'x': 400, 'y': 200}
    )
    llm.add_input_pad("in", "Input")
    llm.add_output_pad("out", "Output")

    outgoing = Facet(
        id="outgoing", name="OUTGOING", facet_type="OUTGOING",
        prompt="", position={'x': 700, 'y': 200}
    )
    outgoing.add_input_pad("in", "Input")

    assembly = FacetAssembly(name="test_viz_assembly")
    assembly.facets = [incoming, llm, outgoing]
    assembly.connections = [
        FacetConnection("incoming", "out", "response", "in"),
        FacetConnection("response", "out", "outgoing", "in"),
    ]
    return assembly


@pytest.fixture
def view(qapp):
    v = AssemblyEditorView()
    v.show()
    yield v
    v.close()


@pytest.fixture
def loaded_view(view):
    assembly = _make_assembly()
    view.load_assembly_from_data(assembly)
    return view


# ============================================================================
# Node state machine
# ============================================================================

class TestNodeStateMachine:
    """Test FacetNodeItem execution state transitions."""

    def test_processing_starts_timer(self, loaded_view):
        """Processing state starts a pulse timer."""
        node = loaded_view._node_items["response"]
        assert node.execution_state == "idle"
        assert node.animation_timer is None

        node.set_execution_state("processing")
        assert node.execution_state == "processing"
        assert node.animation_timer is not None
        assert node.animation_timer.isActive()

        # Cleanup
        node.stop_animation()

    def test_idle_stops_timer(self, loaded_view):
        """Idle state stops any running timer."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("processing")
        assert node.animation_timer is not None

        node.set_execution_state("idle")
        assert node.execution_state == "idle"
        assert node.animation_timer is None

    def test_complete_auto_resets(self, qapp, loaded_view):
        """Complete state returns to idle after 200ms."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("complete")
        assert node.execution_state == "complete"

        # Process events to let QTimer.singleShot fire
        qapp.processEvents()
        QTimer.singleShot(250, qapp.quit)
        qapp.exec()

        assert node.execution_state == "idle"

    def test_error_flash_cycles(self, loaded_view):
        """Error state flashes 5 cycles (10 flashes) then returns to idle."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("error")
        assert node.execution_state == "error"
        assert node.animation_timer is not None

        # Manually advance the timer 10 times
        for _ in range(10):
            node._flash_error_border()

        assert node.execution_state == "idle"
        assert node.animation_timer is None

    def test_quantum_collapse_fades(self, loaded_view):
        """Quantum collapse starts with alpha 1.0 and decrements."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("quantum_collapse")
        assert node.execution_state == "quantum_collapse"
        assert node.collapse_flash_alpha == 1.0

        # Advance fade
        node._fade_quantum_flash()
        assert node.collapse_flash_alpha < 1.0

        # Cleanup
        node.stop_animation()

    def test_stop_animation_cleanup(self, loaded_view):
        """stop_animation() stops timer and resets state."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("processing")
        assert node.animation_timer is not None

        node.stop_animation()
        assert node.animation_timer is None
        assert node.execution_state == "idle"


# ============================================================================
# Wire packet animation
# ============================================================================

class TestWirePacketAnimation:
    """Test FacetConnectionItem data flow packets."""

    def test_animate_data_flow_starts(self, loaded_view):
        """animate_data_flow() starts packet animation."""
        wire = loaded_view._wire_items[0]
        assert not wire.packet_animating

        wire.animate_data_flow()
        assert wire.packet_animating
        assert wire.packet_progress == 0.0
        assert wire.packet_timer is not None
        assert wire.packet_timer.isActive()

        # Cleanup
        wire.stop_animation()

    def test_advance_packet_increments(self, loaded_view):
        """_advance_packet() increments progress."""
        wire = loaded_view._wire_items[0]
        wire.animate_data_flow()

        wire._advance_packet()
        assert wire.packet_progress > 0.0

        wire.stop_animation()

    def test_packet_stops_at_completion(self, loaded_view):
        """Packet animation stops when progress reaches 1.0."""
        wire = loaded_view._wire_items[0]
        wire.animate_data_flow()

        # Advance to completion
        for _ in range(25):  # 25 * 0.05 = 1.25, will hit >= 1.0
            wire._advance_packet()

        assert wire.packet_progress == 1.0
        assert not wire.packet_animating

    def test_no_double_animation(self, loaded_view):
        """animate_data_flow() is a no-op if already animating."""
        wire = loaded_view._wire_items[0]
        wire.animate_data_flow()
        wire.packet_progress = 0.5  # Mid-animation

        wire.animate_data_flow()  # Should not reset
        assert wire.packet_progress == 0.5

        wire.stop_animation()

    def test_stop_animation_resets(self, loaded_view):
        """stop_animation() resets all packet state."""
        wire = loaded_view._wire_items[0]
        wire.animate_data_flow()
        wire.packet_progress = 0.5

        wire.stop_animation()
        assert not wire.packet_animating
        assert wire.packet_progress == 0.0
        assert wire.packet_timer is None


# ============================================================================
# Event dispatch
# ============================================================================

class TestEventDispatch:
    """Test _handle_execution_event routing."""

    def test_facet_start_sets_processing(self, loaded_view):
        """facet_start event puts node in processing state."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {
                'execution_id': 'exec_1',
                'inputs': {'in': 'hello'}
            }
        })

        node = loaded_view._node_items["response"]
        assert node.execution_state == "processing"

        # Cleanup
        node.stop_animation()

    def test_facet_complete_sets_complete(self, loaded_view):
        """facet_complete event puts node in complete state."""
        node = loaded_view._node_items["response"]
        # First start it so the state change from idle -> complete triggers
        node.set_execution_state("processing")

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_complete',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {
                'execution_id': 'exec_1',
                'outputs': {'out': 'world'}
            }
        })

        assert node.execution_state == "complete"
        # Stop to prevent QTimer.singleShot from firing after test teardown
        node.stop_animation()

    def test_facet_error_sets_error(self, loaded_view):
        """facet_error event puts node in error state."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_error',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'error': 'test error'}
        })

        node = loaded_view._node_items["response"]
        assert node.execution_state == "error"
        node.stop_animation()

    def test_data_flow_animates_wire(self, loaded_view):
        """data_flow event triggers wire packet animation."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'data_flow',
            'from_facet': 'incoming',
            'to_facet': 'response',
            'execution_id': 'exec_1',
        })

        # Find the wire from incoming -> response
        wire = loaded_view._wire_items[0]
        assert wire.from_port.facet_node.facet.id == "incoming"
        assert wire.to_port.facet_node.facet.id == "response"
        assert wire.packet_animating

        wire.stop_animation()

    def test_unknown_facet_id_ignored(self, loaded_view):
        """Events with unknown source_id are silently ignored."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'nonexistent_facet',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {}}
        })
        # No crash, no state changes
        assert loaded_view._node_items["response"].execution_state == "idle"

    def test_non_execution_type_ignored(self, loaded_view):
        """Events with type != 'facet_execution' are ignored."""
        loaded_view._handle_execution_event({
            'type': 'something_else',
            'subtype': 'facet_start',
            'source_id': 'response',
        })
        assert loaded_view._node_items["response"].execution_state == "idle"


# ============================================================================
# Cycle tracking
# ============================================================================

class TestCycleTracking:
    """Test cycle color assignment and badge tracking."""

    def test_cycle_color_assigned(self, loaded_view):
        """facet_start assigns a cycle color from the palette."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {'in': 'hi'}}
        })

        assert 'exec_1' in loaded_view.cycle_colors
        assert isinstance(loaded_view.cycle_colors['exec_1'], QColor)

        loaded_view._node_items["response"].stop_animation()

    def test_different_cycles_get_different_colors(self, loaded_view):
        """Two different execution_ids get different colors."""
        for eid in ('exec_1', 'exec_2'):
            loaded_view._handle_execution_event({
                'type': 'facet_execution',
                'subtype': 'facet_start',
                'source_id': 'response',
                'execution_id': eid,
                'data': {'execution_id': eid, 'inputs': {'in': 'hi'}}
            })

        assert loaded_view.cycle_colors['exec_1'] != loaded_view.cycle_colors['exec_2']

        loaded_view._node_items["response"].stop_animation()

    def test_active_cycles_tracked_on_node(self, loaded_view):
        """facet_start adds to node.active_cycles."""
        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {'in': 'hi'}}
        })

        node = loaded_view._node_items["response"]
        assert len(node.active_cycles) == 1
        assert node.active_cycles[0][0] == 'exec_1'

        node.stop_animation()

    def test_cycle_complete_cleans_up(self, loaded_view):
        """cycle_complete removes color from cycle_colors."""
        loaded_view.cycle_colors['exec_1'] = QColor("#00BFFF")

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_complete',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1'}
        })

        assert 'exec_1' not in loaded_view.cycle_colors

    def test_last_inputs_outputs_stored(self, loaded_view):
        """facet_start/complete store last_inputs and last_outputs on node."""
        node = loaded_view._node_items["response"]

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {'in': 'hello'}}
        })
        assert node.last_inputs == {'in': 'hello'}

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_complete',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'outputs': {'out': 'world'}}
        })
        assert node.last_outputs == {'out': 'world'}

        node.stop_animation()


# ============================================================================
# Ensemble filtering
# ============================================================================

class TestEnsembleFiltering:
    """Test ensemble noodling event filtering."""

    def test_filter_by_noodling_id(self, loaded_view):
        """Events for non-selected noodling are ignored."""
        loaded_view._selected_noodling_id = 'ajo'

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'noodling_id': 'yuki',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {}}
        })

        # Should be ignored (yuki != ajo)
        assert loaded_view._node_items["response"].execution_state == "idle"

    def test_accept_matching_noodling(self, loaded_view):
        """Events for selected noodling are processed."""
        loaded_view._selected_noodling_id = 'ajo'

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'noodling_id': 'ajo',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {}}
        })

        node = loaded_view._node_items["response"]
        assert node.execution_state == "processing"
        node.stop_animation()

    def test_no_filter_when_none(self, loaded_view):
        """When _selected_noodling_id is None, all events pass."""
        loaded_view._selected_noodling_id = None

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'noodling_id': 'yuki',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {}}
        })

        node = loaded_view._node_items["response"]
        assert node.execution_state == "processing"
        node.stop_animation()


# ============================================================================
# Scene transition lock
# ============================================================================

class TestSceneTransitionLock:
    """Test scene_transition_lock blocks event processing."""

    def test_lock_blocks_events(self, loaded_view):
        """Events are dropped when scene_transition_lock is True."""
        loaded_view.scene_transition_lock = True

        loaded_view._handle_execution_event({
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'exec_1',
            'data': {'execution_id': 'exec_1', 'inputs': {}}
        })

        assert loaded_view._node_items["response"].execution_state == "idle"

        loaded_view.scene_transition_lock = False

    def test_render_assembly_sets_lock(self, loaded_view):
        """_render_assembly sets and clears scene_transition_lock."""
        # After rendering, lock should be False
        assert loaded_view.scene_transition_lock is False


# ============================================================================
# Sound toggle
# ============================================================================

class TestSoundToggle:
    """Test sound_enabled toggle."""

    def test_sound_toggle(self, loaded_view):
        """toggle_sound() toggles sound_enabled."""
        assert loaded_view.sound_enabled is True

        loaded_view.toggle_sound(False)
        assert loaded_view.sound_enabled is False

        loaded_view.toggle_sound(True)
        assert loaded_view.sound_enabled is True


# ============================================================================
# Pause/resume
# ============================================================================

class TestPauseResume:
    """Test cognition pause/resume state."""

    def test_pause_sets_flag(self, loaded_view):
        """toggle_pause_cognition(True) sets cognition_paused."""
        loaded_view.current_agent_id = 'test_agent'

        loaded_view.toggle_pause_cognition(True)
        assert loaded_view.cognition_paused is True

        loaded_view.toggle_pause_cognition(False)
        assert loaded_view.cognition_paused is False

    def test_pause_without_agent_is_noop(self, loaded_view):
        """Pause with no current_agent_id is a no-op."""
        loaded_view.current_agent_id = None
        loaded_view.toggle_pause_cognition(True)
        assert loaded_view.cognition_paused is False


# ============================================================================
# Ensemble selector
# ============================================================================

class TestEnsembleSelector:
    """Test ensemble noodling selector UI."""

    def test_set_ensemble_noodlings(self, loaded_view):
        """set_ensemble_noodlings shows dropdown with items."""
        noodlings = [
            {'id': 'ajo', 'name': 'Ajo Majo', 'assembly': None, 'assembly_path': None},
            {'id': 'yuki', 'name': 'Yuki Cyberfox', 'assembly': None, 'assembly_path': None},
        ]
        loaded_view.set_ensemble_noodlings(noodlings)

        # Use isHidden() -- isVisible() requires the full ancestor chain
        # (toolbar widget) to be shown, which only happens when embedded
        # in UnifiedEditorPanel. The mixin's API controls show/hide on
        # the selector itself.
        assert not loaded_view._noodling_selector.isHidden()
        assert loaded_view._noodling_selector.count() == 2
        assert loaded_view._noodling_selector.itemText(0) == 'Ajo Majo'
        assert loaded_view._noodling_selector.itemText(1) == 'Yuki Cyberfox'
        assert loaded_view._selected_noodling_id == 'ajo'

    def test_clear_ensemble_noodlings(self, loaded_view):
        """clear_ensemble_noodlings hides dropdown and clears filter."""
        noodlings = [
            {'id': 'ajo', 'name': 'Ajo Majo', 'assembly': None, 'assembly_path': None},
        ]
        loaded_view.set_ensemble_noodlings(noodlings)
        assert not loaded_view._noodling_selector.isHidden()

        loaded_view.clear_ensemble_noodlings()
        assert loaded_view._noodling_selector.isHidden()
        assert loaded_view._noodling_selector.count() == 0
        assert loaded_view._selected_noodling_id is None

    def test_select_noodling(self, loaded_view):
        """select_noodling() switches the dropdown programmatically."""
        noodlings = [
            {'id': 'ajo', 'name': 'Ajo Majo', 'assembly': None, 'assembly_path': None},
            {'id': 'yuki', 'name': 'Yuki Cyberfox', 'assembly': None, 'assembly_path': None},
        ]
        loaded_view.set_ensemble_noodlings(noodlings)
        assert loaded_view._selected_noodling_id == 'ajo'

        loaded_view.select_noodling('yuki')
        assert loaded_view._selected_noodling_id == 'yuki'
        assert loaded_view._noodling_selector.currentIndex() == 1


# ============================================================================
# Timer cleanup on re-render
# ============================================================================

class TestTimerCleanup:
    """Test that re-rendering stops all running animation timers."""

    def test_rerender_stops_node_timers(self, loaded_view):
        """_render_assembly() stops animation timers on existing nodes."""
        node = loaded_view._node_items["response"]
        node.set_execution_state("processing")
        assert node.animation_timer is not None

        # Re-render (creates new nodes, old ones get cleaned up)
        assembly = _make_assembly()
        loaded_view.load_assembly_from_data(assembly, force_reload=True)

        # Old node's timer was stopped by stop_animation() in _render_assembly
        # (node is now detached from scene, timer stopped)
        assert node.animation_timer is None

    def test_rerender_stops_wire_timers(self, loaded_view):
        """_render_assembly() stops packet animation timers on wires."""
        wire = loaded_view._wire_items[0]
        wire.animate_data_flow()
        assert wire.packet_timer is not None

        assembly = _make_assembly()
        loaded_view.load_assembly_from_data(assembly, force_reload=True)

        assert wire.packet_timer is None
        assert not wire.packet_animating
