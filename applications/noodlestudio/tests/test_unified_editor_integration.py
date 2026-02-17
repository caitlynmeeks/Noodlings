"""Integration tests for UnifiedEditorPanel facade API (C.6).

Tests verify that the facade methods on UnifiedEditorPanel correctly
delegate to the underlying AssemblyEditorView and that NC signal
forwarding works through the depth stack. Uses real objects per
project policy (no mocks).
"""

import json
import os
import tempfile

import pytest
from PyQt6.QtWidgets import QApplication, QPushButton
from PyQt6.QtCore import Qt

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


def _make_assembly(name="test_assembly") -> FacetAssembly:
    """Create a minimal 3-node assembly: INCOMING -> LLMFacet -> OUTGOING."""
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

    assembly = FacetAssembly(name=name)
    assembly.facets = [incoming, llm, outgoing]
    assembly.connections = [
        FacetConnection("incoming_1", "out", "llm_1", "in"),
        FacetConnection("llm_1", "out", "outgoing_1", "in"),
    ]
    return assembly


def _make_assembly_with_nc(nncanvas_path: str) -> FacetAssembly:
    """Create a 3-node assembly with a NeuralCanvasFacet."""
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

    assembly = FacetAssembly(name="nc_assembly")
    assembly.facets = [incoming, nc_facet, outgoing]
    assembly.connections = [
        FacetConnection("incoming_1", "out", "nc_1", "in"),
        FacetConnection("nc_1", "out", "outgoing_1", "in"),
    ]
    return assembly


@pytest.fixture
def tmp_assembly_yaml(tmp_path):
    """Create a minimal assembly YAML on disk and return its path."""
    assembly = _make_assembly("disk_assembly")
    path = tmp_path / "assembly.yaml"
    assembly.save_yaml(str(path))
    return str(path)


@pytest.fixture
def tmp_nncanvas(tmp_path):
    """Create a minimal .nncanvas file on disk."""
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
                "id": "input_1", "name": "Input", "type": "INPUT",
                "position": [100, 100], "params": {"dimensions": 5}
            },
            {
                "id": "output_1", "name": "Output", "type": "OUTPUT",
                "position": [400, 100], "params": {"dimensions": 5}
            }
        ],
        "connections": [
            {
                "from_node": "input_1", "from_port": "output",
                "to_node": "output_1", "to_port": "input"
            }
        ],
        "hidden_states": {},
        "export_targets": []
    }
    filepath = tmp_path / "test_charm.nncanvas"
    with open(filepath, "w") as f:
        json.dump(nncanvas_data, f)
    return str(filepath)


@pytest.fixture
def panel(qapp):
    """Fresh UnifiedEditorPanel for each test."""
    p = UnifiedEditorPanel()
    p.show()
    UnifiedEditorPanel.register_depth_view("NeuralCanvasFacet", NeuralCanvasDepthView)
    yield p
    UnifiedEditorPanel._depth_view_registry.clear()
    p.close()


# ============================================================================
# Facade: Assembly Loading
# ============================================================================

class TestFacadeAssemblyLoading:

    def test_load_assembly_from_data_creates_root_view(self, panel):
        """Calling load_assembly_from_data on empty panel creates root view."""
        assembly = _make_assembly()
        panel.load_assembly_from_data(assembly)

        assert panel.depth() == 1
        root = panel._root_view()
        assert isinstance(root, AssemblyEditorView)
        assert root._assembly is assembly

    def test_load_assembly_from_data_with_source_path(self, panel, tmp_assembly_yaml):
        """source_path is forwarded to the root view."""
        assembly = _make_assembly()
        panel.load_assembly_from_data(
            assembly, source_path=tmp_assembly_yaml
        )

        assert panel.current_assembly_path == tmp_assembly_yaml

    def test_load_assembly_from_data_replaces_existing(self, panel):
        """Calling twice replaces the assembly without creating a new view."""
        a1 = _make_assembly("first")
        a2 = _make_assembly("second")

        panel.load_assembly_from_data(a1)
        assert panel.depth() == 1
        root1 = panel._root_view()

        panel.load_assembly_from_data(a2, force_reload=True)
        assert panel.depth() == 1
        # Same root view, different assembly
        assert panel.current_assembly is a2

    def test_load_assembly_from_data_pops_depth_views(self, panel, tmp_nncanvas):
        """Loading a new assembly pops any NC depth views back to root."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        # Push an NC depth view
        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.depth() == 2

        # Loading new assembly should pop back to root
        new_assembly = _make_assembly("new")
        panel.load_assembly_from_data(new_assembly, force_reload=True)
        assert panel.depth() == 1

    def test_clear_editor_clears_stack(self, panel):
        """clear_editor() empties the stack."""
        panel.load_assembly_from_data(_make_assembly())
        assert panel.depth() == 1

        panel.clear_editor()
        assert panel.depth() == 0
        assert panel.current_assembly is None

    def test_set_current_agent(self, panel):
        """set_current_agent stores ID on root view."""
        panel.load_assembly_from_data(_make_assembly())
        panel.set_current_agent("agent_ajo")

        assert panel.current_agent_id == "agent_ajo"

    def test_load_assembly_from_path(self, panel, tmp_assembly_yaml):
        """load_assembly() loads from a file path (soft_restart usage)."""
        panel.load_assembly(tmp_assembly_yaml)

        assert panel.depth() == 1
        assert panel.current_assembly is not None
        assert panel.current_assembly.name == "disk_assembly"
        assert panel.current_assembly_path == tmp_assembly_yaml


# ============================================================================
# Facade: Ensemble
# ============================================================================

class TestFacadeEnsemble:

    def test_set_ensemble_noodlings(self, panel):
        """set_ensemble_noodlings delegates to root view."""
        panel.load_assembly_from_data(_make_assembly())

        noodlings = [
            {'id': 'ajo', 'name': 'Ajo Majo',
             'assembly': _make_assembly("ajo"), 'assembly_path': '/path/ajo'},
            {'id': 'yuki', 'name': 'Yuki Cyberfox',
             'assembly': _make_assembly("yuki"), 'assembly_path': '/path/yuki'},
        ]
        panel.set_ensemble_noodlings(noodlings)

        root = panel._root_view()
        # Ensemble mixin stores the noodlings list
        assert hasattr(root, '_ensemble_noodlings')
        assert len(root._ensemble_noodlings) == 2

    def test_clear_ensemble_noodlings(self, panel):
        """clear_ensemble_noodlings hides selector."""
        panel.load_assembly_from_data(_make_assembly())

        noodlings = [
            {'id': 'ajo', 'name': 'Ajo',
             'assembly': _make_assembly(), 'assembly_path': '/path/ajo'},
        ]
        panel.set_ensemble_noodlings(noodlings)
        panel.clear_ensemble_noodlings()

        root = panel._root_view()
        assert len(root._ensemble_noodlings) == 0

    def test_select_noodling(self, panel):
        """select_noodling delegates to root view."""
        ajo_assembly = _make_assembly("ajo")
        panel.load_assembly_from_data(ajo_assembly)

        noodlings = [
            {'id': 'ajo', 'name': 'Ajo',
             'assembly': ajo_assembly, 'assembly_path': '/path/ajo'},
            {'id': 'yuki', 'name': 'Yuki',
             'assembly': _make_assembly("yuki"), 'assembly_path': '/path/yuki'},
        ]
        panel.set_ensemble_noodlings(noodlings)
        panel.select_noodling('yuki')

        root = panel._root_view()
        assert root._selected_noodling_id == 'yuki'


# ============================================================================
# Facade: Execution Events
# ============================================================================

class TestFacadeExecution:

    def test_handle_execution_event(self, panel):
        """_handle_execution_event forwards to root view without error."""
        panel.load_assembly_from_data(_make_assembly())

        event = {
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'llm_1',
            'execution_id': 'exec_001',
            'noodling_id': 'ajo',
        }
        # Should not raise
        panel._handle_execution_event(event)

    def test_handle_execution_event_empty_stack(self, panel):
        """_handle_execution_event on empty stack is a no-op."""
        event = {
            'type': 'facet_execution',
            'subtype': 'facet_start',
            'source_id': 'llm_1',
        }
        # Should not raise
        panel._handle_execution_event(event)


# ============================================================================
# Facade: Save
# ============================================================================

class TestFacadeSave:

    def test_save_if_dirty_calls_save_all(self, panel):
        """save_if_dirty() delegates to save_all()."""
        panel.load_assembly_from_data(_make_assembly())
        # Should not raise (assembly has no path, so save is a no-op)
        panel.save_if_dirty()

    def test_save_assembly_to_disk_with_path(self, panel, tmp_assembly_yaml):
        """_save_assembly_to_disk() persists via root view."""
        assembly = _make_assembly("save_test")
        panel.load_assembly_from_data(
            assembly, source_path=tmp_assembly_yaml
        )

        # Move the graphics node (save syncs positions from graphics first)
        root = panel._root_view()
        llm_node = root._node_items.get("llm_1")
        assert llm_node is not None
        llm_node.setPos(999, 888)

        panel._save_assembly_to_disk()

        # Reload and verify the position was persisted
        reloaded = FacetAssembly.load_yaml(tmp_assembly_yaml)
        saved_facet = next(
            f for f in reloaded.facets if f.id == "llm_1"
        )
        assert saved_facet.position['x'] == 999
        assert saved_facet.position['y'] == 888

    def test_current_assembly_property(self, panel):
        """current_assembly returns the loaded assembly."""
        assembly = _make_assembly()
        panel.load_assembly_from_data(assembly)
        assert panel.current_assembly is assembly

    def test_current_assembly_none_when_empty(self, panel):
        """current_assembly is None on empty stack."""
        assert panel.current_assembly is None

    def test_current_assembly_path_property(self, panel, tmp_assembly_yaml):
        """current_assembly_path returns the source path."""
        panel.load_assembly_from_data(
            _make_assembly(), source_path=tmp_assembly_yaml
        )
        assert panel.current_assembly_path == tmp_assembly_yaml


# ============================================================================
# Facade: Properties
# ============================================================================

class TestFacadeProperties:

    def test_node_graphics_maps_to_node_items(self, panel):
        """node_graphics returns the root view's _node_items dict."""
        assembly = _make_assembly()
        panel.load_assembly_from_data(assembly)

        ng = panel.node_graphics
        assert isinstance(ng, dict)
        # Should have entries for all 3 facets
        assert "incoming_1" in ng
        assert "llm_1" in ng
        assert "outgoing_1" in ng

    def test_node_graphics_empty_when_no_assembly(self, panel):
        """node_graphics is empty dict when stack is empty."""
        assert panel.node_graphics == {}

    def test_cognition_paused_property(self, panel):
        """cognition_paused read/write works through facade."""
        panel.load_assembly_from_data(_make_assembly())

        assert panel.cognition_paused is False
        panel.cognition_paused = True
        assert panel.cognition_paused is True
        panel.cognition_paused = False
        assert panel.cognition_paused is False

    def test_pause_button_property(self, panel):
        """pause_button returns a QPushButton."""
        panel.load_assembly_from_data(_make_assembly())

        btn = panel.pause_button
        assert isinstance(btn, QPushButton)

    def test_bottom_pause_btn_returns_same_button(self, panel):
        """bottom_pause_btn returns the same button as pause_button."""
        panel.load_assembly_from_data(_make_assembly())

        assert panel.bottom_pause_btn is panel.pause_button

    def test_set_pause_state(self, panel):
        """set_pause_state updates both cognition_paused and button."""
        panel.load_assembly_from_data(_make_assembly())

        panel.set_pause_state(True)
        assert panel.cognition_paused is True
        assert panel.pause_button.isChecked() is True

        panel.set_pause_state(False)
        assert panel.cognition_paused is False
        assert panel.pause_button.isChecked() is False

    def test_refresh_node_for_facet(self, panel):
        """refresh_node_for_facet delegates without error."""
        panel.load_assembly_from_data(_make_assembly())
        # Should not raise
        panel.refresh_node_for_facet("llm_1")

    def test_refresh_node_nonexistent_facet(self, panel):
        """refresh_node_for_facet on nonexistent ID is a no-op."""
        panel.load_assembly_from_data(_make_assembly())
        panel.refresh_node_for_facet("nonexistent_id")


# ============================================================================
# Signal Forwarding: facetSelected
# ============================================================================

class TestFacetSelectedSignal:

    def test_facet_selected_forwards_from_root_view(self, panel, qtbot):
        """Selecting a node in AssemblyEditorView emits facetSelected on panel."""
        assembly = _make_assembly()
        panel.load_assembly_from_data(assembly)

        root = panel._root_view()
        llm_node = root._node_items.get("llm_1")
        assert llm_node is not None

        with qtbot.waitSignal(panel.facetSelected, timeout=1000) as blocker:
            # Select the LLM node in the scene
            root._scene.clearSelection()
            llm_node.setSelected(True)

        assert blocker.args[0] is not None
        assert blocker.args[0].id == "llm_1"


# ============================================================================
# Signal Forwarding: NC signals
# ============================================================================

class TestNCSignalForwarding:

    def test_nc_node_selected_forwarded_on_push(self, panel, tmp_nncanvas, qtbot):
        """Pushing NC depth view connects node_selected to ncNodeSelected."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        # Push NC depth view
        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        nc_view = panel.current_view()
        assert isinstance(nc_view, NeuralCanvasDepthView)

        # Emit node_selected from the inner NC panel
        with qtbot.waitSignal(panel.ncNodeSelected, timeout=1000) as blocker:
            nc_view._panel.node_selected.emit("input_1")

        assert blocker.args == ["input_1"]

    def test_nc_graph_loaded_forwarded(self, panel, tmp_nncanvas, qtbot):
        """NC graph_loaded signal is forwarded through panel."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        nc_view = panel.current_view()

        with qtbot.waitSignal(panel.ncGraphLoaded, timeout=1000):
            nc_view._panel.graph_loaded.emit()

    def test_nc_signals_disconnected_on_pop(self, panel, tmp_nncanvas, qtbot):
        """After popping NC view, ncNodeSelected no longer fires."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        nc_view = panel.current_view()
        nc_panel = nc_view._panel

        # Pop back to assembly
        panel.pop_one()
        assert panel.depth() == 1

        # Emitting node_selected on the (now-popped) NC panel should NOT
        # trigger ncNodeSelected on the unified panel
        received = []
        panel.ncNodeSelected.connect(lambda nid: received.append(nid))
        try:
            nc_panel.node_selected.emit("input_1")
        except RuntimeError:
            pass  # Panel may have been destroyed
        assert len(received) == 0

    def test_get_current_nc_graph(self, panel, tmp_nncanvas):
        """get_current_nc_graph returns graph when NC view is on top."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        # At assembly level, no NC graph
        assert panel.get_current_nc_graph() is None

        # Push NC view
        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )

        graph = panel.get_current_nc_graph()
        assert graph is not None
        assert len(graph.nodes) == 2

    def test_get_current_nc_graph_none_after_pop(self, panel, tmp_nncanvas):
        """get_current_nc_graph is None after popping NC view."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.get_current_nc_graph() is not None

        panel.pop_one()
        assert panel.get_current_nc_graph() is None


# ============================================================================
# Ctrl+S Cascade
# ============================================================================

class TestCtrlSCascade:

    def test_save_all_saves_every_stack_frame(self, panel, tmp_nncanvas):
        """save_all() calls save_data() on every view in the stack."""
        assembly = _make_assembly_with_nc(tmp_nncanvas)
        panel.load_assembly_from_data(assembly)

        panel.on_double_click_container(
            "NeuralCanvasFacet", tmp_nncanvas, "Charm Network"
        )
        assert panel.depth() == 2

        # save_all should succeed without error on both views
        panel.save_all()
