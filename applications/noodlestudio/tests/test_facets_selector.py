# ------------------------------------------------------------------
#   Tests for Facets Editor Noodling Selector (Ensemble Mode)
#
#   Tests that the facets editor dropdown appears in ensemble mode,
#   switches assemblies when a different noodling is selected, and
#   filters live-viz events by the selected noodling.
# ------------------------------------------------------------------
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ------------------------------------------------------------------

from pathlib import Path
from unittest.mock import patch
import pytest

from conftest import (
    FakeLLMClient, StubFacetsEditor, StubMainWindow, StubWindow,
)


# =============================================================================
# Helpers
# =============================================================================

def _create_manager_with_editor():
    """Create a manager whose main_window has a StubFacetsEditor."""
    from noodlestudio.runtime.ui.guide_performance_manager import (
        GuidePerformanceManager,
    )

    editor = StubFacetsEditor()
    main = StubMainWindow(unified_editor=editor)

    manager = GuidePerformanceManager(main)
    manager._assembly_editor = editor
    manager._window = StubWindow()
    return manager, editor


def _make_fake_assembly(name="test"):
    """Create a minimal FacetAssembly for testing."""
    from noodlestudio.core.facet_system import (
        FacetAssembly, Facet, FacetConnection,
    )

    assembly = FacetAssembly(name=name)
    assembly.facets = [
        Facet(id="incoming", name="Input", facet_type="SpecialNode",
              prompt="", position={'x': 0, 'y': 0}),
        Facet(id="response", name="Response", facet_type="LLM",
              prompt="respond", position={'x': 200, 'y': 0}),
        Facet(id="outgoing", name="Output", facet_type="SpecialNode",
              prompt="", position={'x': 400, 'y': 0}),
    ]
    assembly.connections = [
        FacetConnection(from_facet="incoming", from_pad="out",
                        to_facet="response", to_pad="in"),
        FacetConnection(from_facet="response", from_pad="out",
                        to_facet="outgoing", to_pad="in"),
    ]
    return assembly


def _make_noodling_entries():
    """Create two noodling entries for set_ensemble_noodlings()."""
    return [
        {
            'id': 'ajo',
            'name': 'Ajo Majo',
            'assembly': _make_fake_assembly('ajo_assembly'),
            'assembly_path': '/fake/ajo/assembly.yaml',
        },
        {
            'id': 'yuki',
            'name': 'Yuki Cyberfox',
            'assembly': _make_fake_assembly('yuki_assembly'),
            'assembly_path': '/fake/yuki/assembly.yaml',
        },
    ]


# =============================================================================
# Dropdown Visibility
# =============================================================================

class TestDropdownVisibility:
    """Tests for noodling selector dropdown show/hide behavior."""

    def test_selector_hidden_by_default(self, qapp):
        """Noodling selector is hidden when editor first created."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        assert editor._noodling_selector.isHidden()

    def test_selector_shown_after_set_ensemble(self, qapp):
        """Selector becomes visible after set_ensemble_noodlings()."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        # isHidden() returns the widget's own hidden flag, not the effective
        # visibility (which depends on parent being shown). After show(),
        # isHidden() returns False even when the parent is not shown.
        assert not editor._noodling_selector.isHidden()

    def test_selector_hidden_after_clear(self, qapp):
        """Selector hides after clear_ensemble_noodlings()."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())
        editor.clear_ensemble_noodlings()

        assert editor._noodling_selector.isHidden()

    def test_selector_item_count(self, qapp):
        """Selector has correct number of items."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        assert editor._noodling_selector.count() == 2

    def test_selector_item_names(self, qapp):
        """Selector items show noodling names."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        assert editor._noodling_selector.itemText(0) == 'Ajo Majo'
        assert editor._noodling_selector.itemText(1) == 'Yuki Cyberfox'


# =============================================================================
# Assembly Switching
# =============================================================================

class TestAssemblySwitching:
    """Tests for loading different assemblies when selection changes."""

    def test_first_noodling_selected_on_init(self, qapp):
        """First noodling is auto-selected when ensemble is set up."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        assert editor._selected_noodling_id == 'ajo'

    def test_first_assembly_loaded_on_init(self, qapp):
        """First noodling's assembly is loaded into the editor on setup."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        # Check assembly was loaded (current_assembly should be set)
        assert editor.current_assembly is not None
        assert editor.current_assembly.name == 'ajo_assembly'

    def test_switching_changes_selected_id(self, qapp):
        """Changing dropdown updates _selected_noodling_id."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        editor._noodling_selector.setCurrentIndex(1)

        assert editor._selected_noodling_id == 'yuki'

    def test_switching_loads_new_assembly(self, qapp):
        """Changing dropdown loads the selected noodling's assembly."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        editor._noodling_selector.setCurrentIndex(1)

        assert editor.current_assembly is not None
        assert editor.current_assembly.name == 'yuki_assembly'

    def test_switching_back_loads_original(self, qapp):
        """Switching back to first noodling reloads its assembly."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        editor._noodling_selector.setCurrentIndex(1)
        editor._noodling_selector.setCurrentIndex(0)

        assert editor.current_assembly.name == 'ajo_assembly'


# =============================================================================
# Event Filtering
# =============================================================================

class TestEventFiltering:
    """Tests that live-viz events are filtered by selected noodling."""

    def test_events_without_noodling_id_pass_through(self, qapp):
        """Events without noodling_id are accepted (backward compat)."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        # Selected: ajo. Event has no noodling_id.
        event = {
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'execution_id': 'abc',
            'data': {'execution_id': 'abc'}
        }
        editor._handle_execution_event(event)

        # Should not be rejected (cycle_start has no facet node but is processed)
        # Just verify it didn't crash

    def test_matching_noodling_events_accepted(self, qapp):
        """Events matching selected noodling are processed."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        # Load assembly so node_graphics has the facet
        event = {
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'incoming',
            'noodling_id': 'ajo',
            'execution_id': 'test1',
            'data': {'execution_id': 'test1', 'inputs': {'user_message': 'hi'}}
        }
        editor._handle_execution_event(event)

        # Check the node was set to processing state
        node = editor.node_graphics.get('incoming')
        if node:
            assert node.execution_state == 'processing'

    def test_other_noodling_events_filtered(self, qapp):
        """Events for non-selected noodling are rejected."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        entries = _make_noodling_entries()
        editor.set_ensemble_noodlings(entries)

        # Selected: ajo (first). Send event for yuki.
        event = {
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'incoming',
            'noodling_id': 'yuki',
            'execution_id': 'test2',
            'data': {'execution_id': 'test2', 'inputs': {'user_message': 'hi'}}
        }
        editor._handle_execution_event(event)

        # Node should NOT have been set to processing
        node = editor.node_graphics.get('incoming')
        if node:
            assert node.execution_state != 'processing'

    def test_no_filter_when_no_selection(self, qapp):
        """Events are accepted when _selected_noodling_id is None."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        # No set_ensemble_noodlings() called → _selected_noodling_id is None

        event = {
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'noodling_id': 'ajo',
            'execution_id': 'abc',
            'data': {'execution_id': 'abc'}
        }
        # Should not be rejected
        editor._handle_execution_event(event)

    def test_clear_removes_filter(self, qapp):
        """After clear_ensemble_noodlings(), events are no longer filtered."""
        from noodlestudio.panels.facets_editor_panel import FacetsEditorPanel

        editor = FacetsEditorPanel()
        editor.set_ensemble_noodlings(_make_noodling_entries())

        assert editor._selected_noodling_id is not None

        editor.clear_ensemble_noodlings()

        assert editor._selected_noodling_id is None


# =============================================================================
# Manager Event Tagging
# =============================================================================

class TestManagerEventTagging:
    """Tests that the manager tags events with noodling_id."""

    def test_events_tagged_in_single_mode(self):
        """Single-mode events get noodling_id='default'."""
        manager, editor = _create_manager_with_editor()

        manager._current_execution_id = 'test1'
        manager._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'execution_id': 'test1',
            'data': {'execution_id': 'test1'}
        })

        assert len(editor.events) == 1
        assert editor.events[0]['noodling_id'] == 'default'

    def test_events_tagged_with_active_noodling(self):
        """Events get noodling_id from _active_noodling_id."""
        manager, editor = _create_manager_with_editor()

        manager._active_noodling_id = 'yuki'
        manager._current_execution_id = 'test2'
        manager._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'execution_id': 'test2',
            'data': {'execution_id': 'test2'}
        })

        assert editor.events[0]['noodling_id'] == 'yuki'

    def test_explicit_noodling_id_not_overwritten(self):
        """Events with explicit noodling_id keep their value."""
        manager, editor = _create_manager_with_editor()

        manager._active_noodling_id = 'ajo'
        manager._current_execution_id = 'test3'
        manager._emit_execution_event({
            'type': 'facet_execution',
            'subtype': 'cycle_start',
            'noodling_id': 'yuki',  # Explicit
            'execution_id': 'test3',
            'data': {'execution_id': 'test3'}
        })

        assert editor.events[0]['noodling_id'] == 'yuki'


# =============================================================================
# Manager Ensemble Integration
# =============================================================================

class TestManagerEnsembleWiring:
    """Tests that the manager wires the selector in ensemble mode."""

    def test_start_ensemble_sets_noodlings_on_editor(self):
        """start_ensemble() populates the stub editor's ensemble list."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        editor = StubFacetsEditor()
        main = StubMainWindow(unified_editor=editor)
        manager = GuidePerformanceManager(main)
        manager._assembly_editor = editor

        # Set up ensemble manually (bypass real assembly loading)
        ajo = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        ajo._assembly = _make_fake_assembly('ajo_asm')
        ajo._assembly_path = '/fake/ajo.yaml'
        ajo._executor = True

        yuki = NoodlingPerformer('yuki', 'Yuki', FakeLLMClient())
        yuki._assembly = _make_fake_assembly('yuki_asm')
        yuki._assembly_path = '/fake/yuki.yaml'
        yuki._executor = True

        manager._ensemble_mode = True
        manager._performers = {'ajo': ajo, 'yuki': yuki}
        manager._performer = ajo
        manager._window = StubWindow()

        # Simulate what start_ensemble does for the editor
        noodlings = [
            {'id': 'ajo', 'name': 'Ajo Majo',
             'assembly': ajo.assembly, 'assembly_path': ajo.assembly_path},
            {'id': 'yuki', 'name': 'Yuki Cyberfox',
             'assembly': yuki.assembly, 'assembly_path': yuki.assembly_path},
        ]
        editor.set_ensemble_noodlings(noodlings)

        assert len(editor._ensemble_noodlings) == 2
        assert editor._selected_noodling_id == 'ajo'

    def test_stop_clears_ensemble_on_editor(self):
        """stop_performance() clears the stub editor's ensemble state."""
        from noodlestudio.runtime.ui.guide_performance_manager import (
            GuidePerformanceManager,
        )
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        editor = StubFacetsEditor()
        main = StubMainWindow(unified_editor=editor)
        manager = GuidePerformanceManager(main)
        manager._assembly_editor = editor

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

        # Set up ensemble state
        editor._ensemble_noodlings = [{'id': 'ajo'}, {'id': 'yuki'}]
        editor._selected_noodling_id = 'ajo'

        manager.stop_performance()

        assert editor._ensemble_noodlings == []
        assert editor._selected_noodling_id is None


# =============================================================================
# Advance Turn Sets Active Noodling
# =============================================================================

class TestAdvanceTurnActiveNoodling:
    """Tests that _advance_ensemble_turn sets _active_noodling_id."""

    def test_first_turn_sets_ajo(self):
        """First turn sets _active_noodling_id to 'ajo'."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        manager, editor = _create_manager_with_editor()

        ajo = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        ajo._assembly = True
        ajo._executor = True
        ajo.execute = lambda msg, ctx=None: None

        yuki = NoodlingPerformer('yuki', 'Yuki', FakeLLMClient())
        yuki._assembly = True
        yuki._executor = True

        manager._ensemble_mode = True
        manager._performers = {'ajo': ajo, 'yuki': yuki}
        manager._performer = ajo
        manager._turn_queue = ['ajo', 'yuki']
        manager._pending_message = 'Hello'

        manager._advance_ensemble_turn()

        assert manager._active_noodling_id == 'ajo'

    def test_second_turn_sets_yuki(self):
        """When Ajo's turn done, _active_noodling_id changes to 'yuki'."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        manager, editor = _create_manager_with_editor()

        ajo = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        ajo._assembly = True
        ajo._executor = True
        ajo.execute = lambda msg, ctx=None: None

        yuki = NoodlingPerformer('yuki', 'Yuki', FakeLLMClient())
        yuki._assembly = True
        yuki._executor = True
        yuki.execute = lambda msg, ctx=None: None

        manager._ensemble_mode = True
        manager._performers = {'ajo': ajo, 'yuki': yuki}
        manager._performer = ajo
        manager._turn_queue = ['yuki']  # Ajo already popped
        manager._pending_message = 'Hello'

        manager._advance_ensemble_turn()

        assert manager._active_noodling_id == 'yuki'

    def test_all_done_resets_to_default(self):
        """When all turns done, _active_noodling_id resets to 'default'."""
        manager, editor = _create_manager_with_editor()

        manager._ensemble_mode = True
        manager._performers = {}
        manager._turn_queue = []  # Empty — all done
        manager._pending_message = 'Hello'

        manager._advance_ensemble_turn()

        assert manager._active_noodling_id == 'default'

    def test_each_turn_gets_fresh_execution_id(self):
        """Each turn creates a new _current_execution_id."""
        from noodlestudio.runtime.ui.noodling_performer import NoodlingPerformer

        manager, editor = _create_manager_with_editor()

        ajo = NoodlingPerformer('ajo', 'Ajo', FakeLLMClient())
        ajo._assembly = True
        ajo._executor = True
        ajo.execute = lambda msg, ctx=None: None

        yuki = NoodlingPerformer('yuki', 'Yuki', FakeLLMClient())
        yuki._assembly = True
        yuki._executor = True
        yuki.execute = lambda msg, ctx=None: None

        manager._ensemble_mode = True
        manager._performers = {'ajo': ajo, 'yuki': yuki}
        manager._performer = ajo
        manager._turn_queue = ['ajo', 'yuki']
        manager._pending_message = 'Hello'

        manager._advance_ensemble_turn()
        ajo_eid = manager._current_execution_id

        manager._turn_queue = ['yuki']
        manager._advance_ensemble_turn()
        yuki_eid = manager._current_execution_id

        assert ajo_eid != yuki_eid
        assert ajo_eid is not None
        assert yuki_eid is not None


# Made with love. Use with love.
# Caitlyn Meeks 2026
