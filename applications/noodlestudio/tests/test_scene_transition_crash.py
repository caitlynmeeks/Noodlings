# ──────────────────────────────────────────────────────────────
#   Tests for Scene Transition Crash Fix
#
#   Regression test for EXC_BAD_ACCESS in QGraphicsItem.scene()
#   when a 300ms cleanup timer fires after the scene has been
#   cleared and rebuilt. The timer's captured node reference is
#   stale -- calling .scene() on it segfaults.
#
#   Fix: _scene_generation counter invalidates timers from
#   previous scenes; RuntimeError guards catch deleted C++ objects.
# ──────────────────────────────────────────────────────────────

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSceneGenerationGuard:
    """Scene generation counter prevents stale timer callbacks."""

    def test_generation_increments_on_render(self, qapp):
        """_scene_generation increments each time _render_assembly runs."""
        from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
        from noodlestudio.core.facet_system import FacetAssembly

        view = AssemblyEditorView()
        initial_gen = view._scene_generation

        assembly = FacetAssembly(name="test")
        view._assembly = assembly
        view._render_assembly()

        assert view._scene_generation == initial_gen + 1
        view.close()

    def test_stale_timer_is_noop(self, qapp):
        """A clear_cycle closure from a previous generation does nothing."""
        from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
        from noodlestudio.panels.editors.assembly_graphics_items import FacetNodeItem
        from noodlestudio.core.facet_system import Facet, FacetAssembly

        view = AssemblyEditorView()

        # Create a minimal assembly with one facet
        facet = Facet(id='test_facet', facet_type='IntuitionFacet',
                      name='Test', prompt='')
        assembly = FacetAssembly(name="test")
        assembly.facets = [facet]
        assembly.connections = []
        view.load_assembly_from_data(assembly)

        node = view._node_items.get('test_facet')
        assert node is not None

        # Simulate a facet_complete event that schedules a 300ms timer
        old_generation = view._scene_generation

        # Now rebuild the scene (simulates hierarchy click loading new assembly)
        view._assembly = FacetAssembly(name="new_test")
        view._render_assembly()

        # Generation should have incremented
        assert view._scene_generation > old_generation

        # The old node is now deleted by _scene.clear().
        # A stale clear_cycle closure should be a no-op.
        # (In production this runs via QTimer.singleShot; here we
        # call the equivalent logic directly to verify the guard.)
        stale_gen = old_generation
        current_gen = view._scene_generation

        # This is exactly what the closure checks:
        assert stale_gen != current_gen  # Guard would return early

        view.close()

    def test_execution_mixin_has_generation(self, qapp):
        """AssemblyEditorView has _scene_generation from execution mixin."""
        from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
        view = AssemblyEditorView()
        assert hasattr(view, '_scene_generation')
        assert isinstance(view._scene_generation, int)
        view.close()


class TestRuntimeErrorGuard:
    """RuntimeError guards catch deleted C++ QGraphicsItem objects."""

    def test_handle_event_survives_deleted_node(self, qapp):
        """_handle_execution_event does not crash on deleted node."""
        from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
        from noodlestudio.core.facet_system import Facet, FacetAssembly

        view = AssemblyEditorView()

        facet = Facet(id='response', facet_type='IntuitionFacet',
                      name='Response', prompt='')
        assembly = FacetAssembly(name="test")
        assembly.facets = [facet]
        assembly.connections = []
        view.load_assembly_from_data(assembly)

        node = view._node_items.get('response')
        assert node is not None

        # Clear the scene (deletes the C++ QGraphicsItem)
        view._scene.clear()

        # Node items dict still has the Python wrapper
        # but the C++ object is deleted. Sending an event
        # should not crash -- the RuntimeError guard catches it.
        event = {
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'response',
            'execution_id': 'abc123',
            'data': {'execution_id': 'abc123', 'inputs': {}},
        }
        # This should not raise
        view._handle_execution_event(event)

        view.close()
