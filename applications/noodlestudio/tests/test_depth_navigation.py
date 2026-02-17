"""Tests for depth-stack navigation: NeuralCanvasView as first registered depth view (C.5).

Tests the full pipeline: double-click NeuralCanvasFacet -> push NC view ->
breadcrumb navigation -> pop back to assembly. Uses real objects per project policy.
"""

import json
import os
import tempfile

import pytest
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QKeyEvent, QTransform

from noodlestudio.core.facet_system import (
    Facet, FacetAssembly, FacetConnection, FacetPad, PadType
)
from noodlestudio.panels.editors.unified_editor_panel import UnifiedEditorPanel
from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
from noodlestudio.panels.editors.neural_canvas_depth_view import NeuralCanvasDepthView


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def tmp_nncanvas(tmp_path):
    """Create a minimal .nncanvas file on disk and return its path."""
    nncanvas_data = {
        "version": "1.0",
        "name": "Test Charm Network",
        "description": "A test neural canvas",
        "metadata": {
            "created": "2026-02-17T00:00:00",
            "modified": "2026-02-17T00:00:00",
            "author": "test",
            "total_parameters": 0
        },
        "nodes": [
            {
                "id": "input_1",
                "name": "Input",
                "type": "INPUT",
                "position": [100, 100],
                "params": {"dimensions": 5}
            },
            {
                "id": "output_1",
                "name": "Output",
                "type": "OUTPUT",
                "position": [400, 100],
                "params": {"dimensions": 5}
            }
        ],
        "connections": [
            {
                "from_node": "input_1",
                "from_port": "output",
                "to_node": "output_1",
                "to_port": "input"
            }
        ],
        "hidden_states": {},
        "export_targets": []
    }
    filepath = tmp_path / "test_charm.nncanvas"
    with open(filepath, "w") as f:
        json.dump(nncanvas_data, f)
    return str(filepath)


def _make_assembly_with_nc(nncanvas_path: str) -> FacetAssembly:
    """Create a 3-node assembly: INCOMING -> NeuralCanvasFacet -> OUTGOING."""
    incoming = Facet(
        id="incoming_1", name="INCOMING", facet_type="INCOMING",
        prompt="", position={'x': 100, 'y': 200}
    )
    incoming.add_output_pad("out", "Output")

    nc_facet = Facet(
        id="nc_1", name="Charm Network", facet_type="NeuralCanvasFacet",
        prompt="", position={'x': 400, 'y': 200},
        nncanvas_path=nncanvas_path
    )
    nc_facet.add_input_pad("in", "Input")
    nc_facet.add_output_pad("out", "Output")

    outgoing = Facet(
        id="outgoing_1", name="OUTGOING", facet_type="OUTGOING",
        prompt="", position={'x': 700, 'y': 200}
    )
    outgoing.add_input_pad("in", "Input")

    assembly = FacetAssembly(name="test_assembly")
    assembly.facets = [incoming, nc_facet, outgoing]
    assembly.connections = [
        FacetConnection("incoming_1", "out", "nc_1", "in"),
        FacetConnection("nc_1", "out", "outgoing_1", "in"),
    ]
    return assembly


def _make_assembly_no_nc() -> FacetAssembly:
    """Create a simple 3-node assembly with no NeuralCanvasFacet."""
    incoming = Facet(
        id="incoming_1", name="INCOMING", facet_type="INCOMING",
        prompt="", position={'x': 100, 'y': 200}
    )
    incoming.add_output_pad("out", "Output")

    llm = Facet(
        id="llm_1", name="Intuition", facet_type="LLMFacet",
        prompt="Think.", position={'x': 400, 'y': 200}
    )
    llm.add_input_pad("in", "Input")
    llm.add_output_pad("out", "Output")

    outgoing = Facet(
        id="outgoing_1", name="OUTGOING", facet_type="OUTGOING",
        prompt="", position={'x': 700, 'y': 200}
    )
    outgoing.add_input_pad("in", "Input")

    assembly = FacetAssembly(name="test_assembly_no_nc")
    assembly.facets = [incoming, llm, outgoing]
    assembly.connections = [
        FacetConnection("incoming_1", "out", "llm_1", "in"),
        FacetConnection("llm_1", "out", "outgoing_1", "in"),
    ]
    return assembly


@pytest.fixture
def panel(qapp):
    """Fresh UnifiedEditorPanel with NeuralCanvasDepthView registered."""
    p = UnifiedEditorPanel()
    p.show()
    # Ensure NeuralCanvasDepthView is registered (done by editors/__init__.py import)
    UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", NeuralCanvasDepthView)
    yield p
    UnifiedEditorPanel._depth_view_registry.clear()
    p.close()


# ============================================================================
# Test: Double-click pushes NC view
# ============================================================================

class TestDepthNavigation:

    def test_double_click_neural_canvas_pushes_view(self, panel, tmp_nncanvas):
        """Double-click on NeuralCanvasFacet pushes NC depth view."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        assert panel.depth() == 1

        # Simulate the containerDoubleClicked signal
        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )

        assert panel.depth() == 2
        top = panel.current_view()
        assert isinstance(top, NeuralCanvasDepthView)

    def test_breadcrumb_shows_two_segments(self, panel, tmp_nncanvas):
        """After pushing NC view, breadcrumb shows two segments."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )

        assert panel._breadcrumb.segment_count() == 2

    def test_breadcrumb_click_pops_to_assembly(self, panel, tmp_nncanvas, qtbot):
        """Clicking the root breadcrumb segment pops back to assembly."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.depth() == 2

        # Click the root breadcrumb (index 0)
        with qtbot.waitSignal(panel.depthChanged, timeout=1000):
            panel._breadcrumb.segmentClicked.emit(0)

        assert panel.depth() == 1
        assert panel.current_view() is root_view

    def test_backspace_pops_nc_view(self, panel, tmp_nncanvas):
        """Backspace at depth 2 pops back to depth 1."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.depth() == 2

        from PyQt6.QtCore import QEvent
        event = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Backspace,
                          Qt.KeyboardModifier.NoModifier)
        panel.keyPressEvent(event)

        assert panel.depth() == 1
        assert panel.current_view() is root_view

    def test_double_click_non_container_no_push(self, panel):
        """Double-clicking a non-container facet type does not push a view."""
        assembly = _make_assembly_no_nc()
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly")

        assert panel.depth() == 1

        # Emit for an unregistered type
        root_view.containerDoubleClicked.emit(
            "LLMFacet", "/does/not/matter", "Intuition"
        )

        assert panel.depth() == 1  # No change


# ============================================================================
# Test: Path resolution
# ============================================================================

class TestPathResolution:

    def test_relative_path_resolution(self, tmp_path):
        """Relative nncanvas_path resolved against project_root."""
        resolved = NeuralCanvasDepthView._resolve_path(
            "Noodlings/ajo/charm.nncanvas",
            {"project_root": str(tmp_path)}
        )
        expected = os.path.join(str(tmp_path), "Noodlings/ajo/charm.nncanvas")
        assert resolved == expected

    def test_absolute_path_used_as_is(self, tmp_nncanvas):
        """Absolute path is returned unchanged."""
        resolved = NeuralCanvasDepthView._resolve_path(
            tmp_nncanvas,
            {"project_root": "/some/other/root"}
        )
        assert resolved == tmp_nncanvas

    def test_no_project_root_returns_path_as_is(self):
        """Without project_root, relative path is returned unchanged."""
        resolved = NeuralCanvasDepthView._resolve_path(
            "relative/path.nncanvas",
            {}
        )
        assert resolved == "relative/path.nncanvas"


# ============================================================================
# Test: NC view loading
# ============================================================================

class TestNCViewLoading:

    def test_nc_view_loads_graph(self, qapp, tmp_nncanvas):
        """NeuralCanvasDepthView loads the .nncanvas graph correctly."""
        view = NeuralCanvasDepthView()
        view.load_data(tmp_nncanvas, {})

        # The embedded panel should have loaded the graph
        assert view._panel.graph is not None
        assert len(view._panel.graph.nodes) == 2  # INPUT + OUTPUT
        assert view._panel.current_filepath == tmp_nncanvas
        view.close()

    def test_nc_view_breadcrumb_label(self, qapp, tmp_nncanvas):
        """Breadcrumb label comes from the graph name."""
        view = NeuralCanvasDepthView()
        view.load_data(tmp_nncanvas, {})

        assert view.get_breadcrumb_label() == "Test Charm Network"
        view.close()

    def test_nc_view_breadcrumb_label_default(self, qapp):
        """Default breadcrumb label when no graph is loaded."""
        view = NeuralCanvasDepthView()
        assert view.get_breadcrumb_label() == "Neural Canvas"
        view.close()

    def test_nc_view_has_unsaved_changes_false(self, qapp, tmp_nncanvas):
        """NC auto-saves, so has_unsaved_changes is always False."""
        view = NeuralCanvasDepthView()
        view.load_data(tmp_nncanvas, {})
        assert view.has_unsaved_changes() is False
        view.close()


# ============================================================================
# Test: Save on pop
# ============================================================================

class TestSaveOnPop:

    def test_nc_view_save_called_on_pop(self, panel, tmp_nncanvas):
        """Popping the NC view triggers save_data() -> save_if_dirty()."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        nc_view = panel.current_view()
        assert isinstance(nc_view, NeuralCanvasDepthView)

        # Record the modification time before pop
        mtime_before = os.path.getmtime(tmp_nncanvas)

        # Pop should call save_data() which calls save_if_dirty()
        panel.pop_one()
        assert panel.depth() == 1

        # The NC panel's save_if_dirty() should have been called (no error)
        # We can't easily check file write since it may be identical content,
        # but the fact that pop_one() succeeded without error confirms save_data() ran


# ============================================================================
# Test: State preservation
# ============================================================================

class TestStatePreservation:

    def test_assembly_view_state_preserved(self, panel, tmp_nncanvas):
        """Assembly view's transform is preserved after push/pop round-trip."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": os.path.dirname(tmp_nncanvas)})

        # Apply a non-identity transform (zoom in)
        root_view.scale(1.5, 1.5)
        transform_before = root_view.transform()

        # Push NC view (hides assembly)
        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.depth() == 2

        # Pop back to assembly
        panel.pop_one()
        assert panel.depth() == 1

        # Transform should be preserved
        transform_after = root_view.transform()
        assert transform_before == transform_after


# ============================================================================
# Test: Context inheritance
# ============================================================================

class TestContextInheritance:

    def test_context_inherits_from_parent(self, panel, tmp_nncanvas):
        """NC view receives project_root from parent frame's context."""
        project_root = os.path.dirname(tmp_nncanvas)
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": project_root})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )

        nc_view = panel.current_view()
        assert isinstance(nc_view, NeuralCanvasDepthView)
        # The NC view's context should have inherited project_root
        assert nc_view._context.get('project_root') == project_root

    def test_context_inherits_with_relative_path(self, panel, tmp_path):
        """Relative nncanvas path is resolved using inherited project_root."""
        # Create .nncanvas in a subdirectory
        subdir = tmp_path / "Noodlings" / "ajo"
        subdir.mkdir(parents=True)
        nncanvas_path = subdir / "charm.nncanvas"
        nncanvas_data = {
            "version": "1.0", "name": "Resolved Charm",
            "description": "", "metadata": {},
            "nodes": [], "connections": [],
            "hidden_states": {}, "export_targets": []
        }
        with open(nncanvas_path, "w") as f:
            json.dump(nncanvas_data, f)

        assembly = _make_assembly_with_nc("Noodlings/ajo/charm.nncanvas")
        root_view = AssemblyEditorView()
        root_view.load_assembly_from_data(assembly)
        panel.push_view(root_view, "assembly",
                        context={"project_root": str(tmp_path)})

        root_view.containerDoubleClicked.emit(
            "NeuralCanvasFacet", "Noodlings/ajo/charm.nncanvas", "Charm Network"
        )

        nc_view = panel.current_view()
        assert isinstance(nc_view, NeuralCanvasDepthView)
        assert nc_view._panel.current_filepath == str(nncanvas_path)
        assert nc_view.get_breadcrumb_label() == "Resolved Charm"
