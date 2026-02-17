"""Tests for AssemblyEditorView -- rendering, interaction, editing, clipboard.

Uses real FacetAssembly objects (no mocks per project policy).
"""

import os
import pytest

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QPointF

from noodlestudio.core.facet_system import (
    Facet, FacetAssembly, FacetConnection, FacetPad, PadType
)
from noodlestudio.panels.editors.assembly_editor_view import AssemblyEditorView
from noodlestudio.panels.editors.assembly_graphics_items import (
    FacetNodeItem, FacetPortItem, FacetConnectionItem, get_facet_header_color
)
from noodlestudio.panels.editors.facet_editor_protocol import FacetEditorProtocol
from noodlestudio.core.undo_manager import undo_manager


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_assembly():
    """Create a 3-node INCOMING -> LLM -> OUTGOING assembly."""
    incoming = Facet(
        id="incoming_1", name="INCOMING", facet_type="INCOMING",
        prompt="", position={'x': 100, 'y': 200}
    )
    incoming.add_output_pad("out", "Output")

    llm = Facet(
        id="llm_1", name="Intuition", facet_type="LLMFacet",
        prompt="Think carefully.", position={'x': 400, 'y': 200}
    )
    llm.add_input_pad("in", "Input")
    llm.add_output_pad("out", "Output")

    outgoing = Facet(
        id="outgoing_1", name="OUTGOING", facet_type="OUTGOING",
        prompt="", position={'x': 700, 'y': 200}
    )
    outgoing.add_input_pad("in", "Input")

    assembly = FacetAssembly(name="test_assembly")
    assembly.facets = [incoming, llm, outgoing]
    assembly.connections = [
        FacetConnection("incoming_1", "out", "llm_1", "in"),
        FacetConnection("llm_1", "out", "outgoing_1", "in"),
    ]
    return assembly


@pytest.fixture
def view(qapp):
    """Fresh AssemblyEditorView for each test."""
    undo_manager.clear()
    v = AssemblyEditorView()
    v.show()
    yield v
    v.close()


@pytest.fixture
def loaded_view(view):
    """AssemblyEditorView with a 3-node assembly loaded."""
    assembly = _make_assembly()
    view.load_assembly_from_data(assembly)
    return view


# ============================================================================
# TestRendering
# ============================================================================

class TestRendering:
    """Verify assembly loads correctly into the scene."""

    def test_node_count(self, loaded_view):
        assert len(loaded_view._node_items) == 3

    def test_wire_count(self, loaded_view):
        assert len(loaded_view._wire_items) == 2

    def test_node_positions_match_data(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        assert abs(node.pos().x() - 400) < 1
        assert abs(node.pos().y() - 200) < 1

    def test_incoming_is_special_compact(self, loaded_view):
        node = loaded_view._node_items["incoming_1"]
        assert node.is_special_node is True

    def test_outgoing_is_special_compact(self, loaded_view):
        node = loaded_view._node_items["outgoing_1"]
        assert node.is_special_node is True

    def test_llm_is_not_special(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        assert node.is_special_node is False

    def test_header_color_for_llm(self):
        facet = Facet(id="x", name="Test", facet_type="LLMFacet", prompt="")
        assert get_facet_header_color(facet) == "#6A4A6A"

    def test_header_color_for_incoming(self):
        facet = Facet(id="x", name="INCOMING", facet_type="INCOMING", prompt="")
        assert get_facet_header_color(facet) == "#5A7A5A"

    def test_header_color_default(self):
        facet = Facet(id="x", name="Test", facet_type="UnknownType", prompt="")
        assert get_facet_header_color(facet) == "#5A5A5A"

    def test_empty_assembly_renders_nothing(self, view):
        assembly = FacetAssembly(name="empty")
        view.load_assembly_from_data(assembly)
        assert len(view._node_items) == 0
        assert len(view._wire_items) == 0

    def test_ports_created(self, loaded_view):
        llm_node = loaded_view._node_items["llm_1"]
        assert "in" in llm_node.input_pads
        assert "out" in llm_node.output_pads

    def test_convergence_facet_has_two_inputs(self, view):
        facet = Facet(
            id="conv_1", name="Merge", facet_type="ConvergenceFacet",
            prompt="", position={'x': 0, 'y': 0}
        )
        facet.add_input_pad("input1", "First")
        facet.add_input_pad("input2", "Second")
        facet.add_output_pad("output", "Out")
        assembly = FacetAssembly(name="conv_test")
        assembly.facets = [facet]
        view.load_assembly_from_data(assembly)
        node = view._node_items["conv_1"]
        assert len(node.input_pads) == 2


# ============================================================================
# TestSelection
# ============================================================================

class TestSelection:
    """Verify selection signals and state."""

    def test_select_node_emits_facet(self, loaded_view, qtbot):
        node = loaded_view._node_items["llm_1"]
        with qtbot.waitSignal(loaded_view.facetSelected, timeout=1000) as blocker:
            node.setSelected(True)
        assert blocker.args[0] is not None
        assert blocker.args[0].id == "llm_1"

    def test_deselect_emits_none(self, loaded_view, qtbot):
        node = loaded_view._node_items["llm_1"]
        node.setSelected(True)
        QApplication.processEvents()

        with qtbot.waitSignal(loaded_view.facetSelected, timeout=1000) as blocker:
            node.setSelected(False)
        assert blocker.args[0] is None

    def test_get_selected_node_ids(self, loaded_view):
        loaded_view._node_items["llm_1"].setSelected(True)
        loaded_view._node_items["outgoing_1"].setSelected(True)
        ids = loaded_view.get_selected_node_ids()
        assert set(ids) == {"llm_1", "outgoing_1"}


# ============================================================================
# TestUndoCreateDelete
# ============================================================================

class TestUndoCreateDelete:
    """Test create/delete facets with undo."""

    def test_create_facet_and_undo(self, loaded_view):
        assert len(loaded_view._node_items) == 3

        # Create a new facet via internal method (simulates undo command)
        new_facet = Facet(
            id="new_1", name="New Facet", facet_type="EmotionFacet",
            prompt="test", position={'x': 500, 'y': 300}
        )
        new_facet.add_input_pad("in", "Input")
        new_facet.add_output_pad("out", "Output")

        from noodlestudio.core.commands import CreateFacetCommand
        cmd = CreateFacetCommand(
            editor=loaded_view,
            facet_data=new_facet.to_dict(),
            facet_name="New Facet"
        )
        undo_manager.push(cmd)
        assert len(loaded_view._node_items) == 4
        assert "new_1" in loaded_view._node_items

        # Undo
        undo_manager.undo()
        assert len(loaded_view._node_items) == 3
        assert "new_1" not in loaded_view._node_items

        # Redo
        undo_manager.redo()
        assert len(loaded_view._node_items) == 4
        assert "new_1" in loaded_view._node_items

    def test_delete_facet_and_undo(self, loaded_view):
        assert len(loaded_view._node_items) == 3
        initial_conn_count = len(loaded_view._wire_items)

        # Collect connection data for llm_1
        conns_data = [
            c.to_dict() for c in loaded_view._assembly.connections
            if c.from_facet == "llm_1" or c.to_facet == "llm_1"
        ]

        from noodlestudio.core.commands import DeleteFacetCommand
        cmd = DeleteFacetCommand(
            editor=loaded_view,
            facet_data=loaded_view._node_items["llm_1"].facet.to_dict(),
            connections_data=conns_data,
            facet_name="Intuition",
        )
        undo_manager.push(cmd)

        assert len(loaded_view._node_items) == 2
        assert "llm_1" not in loaded_view._node_items
        # Both wires should be gone (both connected to llm_1)
        assert len(loaded_view._wire_items) == 0

        # Undo restores facet + connections
        undo_manager.undo()
        assert len(loaded_view._node_items) == 3
        assert "llm_1" in loaded_view._node_items
        assert len(loaded_view._wire_items) == initial_conn_count


# ============================================================================
# TestWireConnection
# ============================================================================

class TestWireConnection:
    """Test wire creation/deletion with undo."""

    def test_create_connection_undo(self, view):
        # Set up two disconnected facets
        f1 = Facet(
            id="f1", name="Source", facet_type="LLMFacet",
            prompt="", position={'x': 100, 'y': 100}
        )
        f1.add_output_pad("out", "Output")

        f2 = Facet(
            id="f2", name="Dest", facet_type="LLMFacet",
            prompt="", position={'x': 400, 'y': 100}
        )
        f2.add_input_pad("in", "Input")

        assembly = FacetAssembly(name="wire_test")
        assembly.facets = [f1, f2]
        view.load_assembly_from_data(assembly)

        assert len(view._wire_items) == 0

        # Create connection
        from noodlestudio.core.commands import CreateConnectionCommand
        cmd = CreateConnectionCommand(
            editor=view, from_facet="f1", from_pad="out",
            to_facet="f2", to_pad="in"
        )
        undo_manager.push(cmd)

        assert len(view._wire_items) == 1
        assert len(view._assembly.connections) == 1

        # Undo
        undo_manager.undo()
        assert len(view._wire_items) == 0
        assert len(view._assembly.connections) == 0

    def test_wire_registers_with_ports(self, loaded_view):
        """Verify wires register themselves with their port items."""
        # The llm_1 node should have connections on its ports
        llm_node = loaded_view._node_items["llm_1"]
        in_port = llm_node.input_pads["in"]
        out_port = llm_node.output_pads["out"]
        assert len(in_port.connections) == 1
        assert len(out_port.connections) == 1


# ============================================================================
# TestCopyPaste
# ============================================================================

class TestCopyPaste:
    """Test clipboard operations."""

    def test_copy_paste_creates_new_ids(self, loaded_view):
        # Select the LLM node
        loaded_view._node_items["llm_1"].setSelected(True)
        loaded_view.copy_selection()

        assert len(loaded_view._clipboard_items) == 1

        initial_count = len(loaded_view._node_items)
        loaded_view.paste_selection()

        assert len(loaded_view._node_items) == initial_count + 1

        # Verify new node has a different ID
        new_ids = set(loaded_view._node_items.keys()) - {"incoming_1", "llm_1", "outgoing_1"}
        assert len(new_ids) == 1
        new_id = new_ids.pop()
        assert new_id != "llm_1"

    def test_special_nodes_not_copied(self, loaded_view):
        """INCOMING/OUTGOING should not be copyable."""
        loaded_view._node_items["incoming_1"].setSelected(True)
        loaded_view.copy_selection()
        assert len(loaded_view._clipboard_items) == 0

    def test_paste_preserves_internal_connections(self, view):
        """When pasting a group, internal connections are duplicated."""
        # Create two connected non-special facets
        f1 = Facet(
            id="f1", name="A", facet_type="LLMFacet",
            prompt="", position={'x': 100, 'y': 100}
        )
        f1.add_input_pad("in", "Input")
        f1.add_output_pad("out", "Output")

        f2 = Facet(
            id="f2", name="B", facet_type="LLMFacet",
            prompt="", position={'x': 400, 'y': 100}
        )
        f2.add_input_pad("in", "Input")
        f2.add_output_pad("out", "Output")

        assembly = FacetAssembly(name="paste_test")
        assembly.facets = [f1, f2]
        assembly.connections = [
            FacetConnection("f1", "out", "f2", "in")
        ]
        view.load_assembly_from_data(assembly)

        # Select both and copy
        view._node_items["f1"].setSelected(True)
        view._node_items["f2"].setSelected(True)
        view.copy_selection()

        assert len(view._clipboard_connections) == 1

        initial_wires = len(view._wire_items)
        view.paste_selection()

        # Should have created the internal connection between the pasted pair
        assert len(view._wire_items) == initial_wires + 1


# ============================================================================
# TestAutoArrange
# ============================================================================

class TestAutoArrange:
    """Test topological layout."""

    def test_auto_arrange_produces_left_to_right(self, loaded_view):
        # Scramble positions
        loaded_view._node_items["incoming_1"].setPos(500, 500)
        loaded_view._node_items["llm_1"].setPos(100, 100)
        loaded_view._node_items["outgoing_1"].setPos(300, 300)

        loaded_view.auto_arrange()

        # After topological sort: INCOMING (layer 0) should be leftmost
        incoming_x = loaded_view._node_items["incoming_1"].pos().x()
        llm_x = loaded_view._node_items["llm_1"].pos().x()
        outgoing_x = loaded_view._node_items["outgoing_1"].pos().x()

        assert incoming_x < llm_x
        assert llm_x < outgoing_x


# ============================================================================
# TestFocusFrameAll
# ============================================================================

class TestFocusFrameAll:
    """Test focus (F) and frame-all (A) operations."""

    def test_frame_all(self, loaded_view):
        # Just verify it doesn't crash
        loaded_view.frame_all_nodes()

    def test_focus_toggle(self, loaded_view):
        loaded_view._node_items["llm_1"].setSelected(True)

        # First focus
        loaded_view.focus_selection()
        assert loaded_view._is_focused is True

        # Toggle back
        loaded_view.focus_selection()
        assert loaded_view._is_focused is False


# ============================================================================
# TestSaveRoundTrip
# ============================================================================

class TestSaveRoundTrip:
    """Test save to disk and reload."""

    def test_save_and_reload(self, loaded_view, tmp_path):
        save_path = str(tmp_path / "test_assembly.yaml")
        loaded_view._assembly_path = save_path

        # Move a node to verify position round-trip
        loaded_view._node_items["llm_1"].setPos(555, 333)

        loaded_view.save_data()
        assert os.path.exists(save_path)

        # Reload into a fresh view
        view2 = AssemblyEditorView()
        view2.load_data(save_path, {})

        assert len(view2._node_items) == 3
        assert len(view2._wire_items) == 2

        # Verify position was persisted
        node = view2._node_items["llm_1"]
        assert abs(node.pos().x() - 555) < 1
        assert abs(node.pos().y() - 333) < 1
        view2.close()


# ============================================================================
# TestProtocolCompliance
# ============================================================================

class TestProtocolCompliance:
    """Verify AssemblyEditorView satisfies FacetEditorProtocol."""

    def test_has_all_protocol_methods(self, view):
        protocol_methods = [
            '_save_assembly_to_disk',
            '_set_facet_position_internal',
            '_create_facet_internal',
            '_delete_facet_internal',
            '_create_connection_internal',
            '_delete_connection_internal',
            '_set_facet_property_internal',
        ]
        for method in protocol_methods:
            assert hasattr(view, method), f"Missing protocol method: {method}"
            assert callable(getattr(view, method)), f"Not callable: {method}"


# ============================================================================
# TestDepthViewProtocol
# ============================================================================

class TestDepthViewProtocol:
    """Verify DepthViewProtocol implementation."""

    def test_has_depth_view_methods(self, view):
        assert hasattr(view, 'load_data')
        assert hasattr(view, 'save_data')
        assert hasattr(view, 'get_breadcrumb_label')
        assert hasattr(view, 'has_unsaved_changes')

    def test_breadcrumb_label(self, loaded_view):
        assert loaded_view.get_breadcrumb_label() == "test_assembly"

    def test_has_unsaved_changes_initially_false(self, loaded_view):
        assert loaded_view.has_unsaved_changes() is False

    def test_dirty_after_create(self, loaded_view):
        new_facet = Facet(
            id="dirty_1", name="Dirty", facet_type="LLMFacet",
            prompt="", position={'x': 0, 'y': 0}
        )
        new_facet.add_input_pad("in", "Input")
        new_facet.add_output_pad("out", "Output")

        loaded_view._create_facet_internal(new_facet.to_dict())
        assert loaded_view.has_unsaved_changes() is True

    def test_load_from_yaml(self, view, tmp_path):
        assembly = _make_assembly()
        save_path = str(tmp_path / "load_test.yaml")
        assembly.save_yaml(save_path)

        view.load_data(save_path, {})
        assert len(view._node_items) == 3
        assert view.get_breadcrumb_label() == "test_assembly"


# ============================================================================
# TestPortDuckTyping
# ============================================================================

class TestPortDuckTyping:
    """Verify ports satisfy SharedWireMixin's duck-typing contract."""

    def test_port_has_is_output(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        assert node.input_pads["in"].is_output is False
        assert node.output_pads["out"].is_output is True

    def test_port_has_parent_node_id(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        assert node.input_pads["in"].get_parent_node_id() == "llm_1"

    def test_port_has_port_name(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        assert node.input_pads["in"].get_port_name() == "in"
        assert node.output_pads["out"].get_port_name() == "out"

    def test_port_has_scene_position(self, loaded_view):
        node = loaded_view._node_items["llm_1"]
        pos = node.output_pads["out"].get_scene_position()
        assert isinstance(pos, QPointF)


# ============================================================================
# TestMoveUndo
# ============================================================================

class TestMoveUndo:
    """Test move facet with undo."""

    def test_move_and_undo(self, loaded_view):
        from noodlestudio.core.commands import MoveFacetCommand

        old_pos = (400, 200)
        new_pos = (600, 300)

        cmd = MoveFacetCommand(
            editor=loaded_view,
            facet_id="llm_1",
            old_pos=old_pos,
            new_pos=new_pos,
            facet_name="Intuition",
        )
        undo_manager.push(cmd)

        # Undo should restore position
        undo_manager.undo()
        node = loaded_view._node_items["llm_1"]
        assert abs(node.pos().x() - 400) < 1
        assert abs(node.pos().y() - 200) < 1

        # Redo should move back
        undo_manager.redo()
        assert abs(node.pos().x() - 600) < 1
        assert abs(node.pos().y() - 300) < 1


# ============================================================================
# TestSharedMixinIntegration
# ============================================================================

class TestSharedMixinIntegration:
    """Verify the shared mixin abstract methods are properly implemented."""

    def test_get_node_items(self, loaded_view):
        items = loaded_view.get_node_items()
        assert len(items) == 3
        assert all(isinstance(v, FacetNodeItem) for v in items.values())

    def test_get_graph_edges(self, loaded_view):
        edges = loaded_view.get_graph_edges()
        assert len(edges) == 2
        assert ("incoming_1", "llm_1") in edges
        assert ("llm_1", "outgoing_1") in edges

    def test_get_existing_connections(self, loaded_view):
        conns = loaded_view.get_existing_connections()
        assert len(conns) == 2
        # Each is (from_facet, from_pad, to_facet, to_pad)
        assert conns[0] == ("incoming_1", "out", "llm_1", "in")

    def test_can_connect_validates(self, loaded_view):
        """SharedWireMixin.can_connect should validate connections."""
        in_node = loaded_view._node_items["incoming_1"]
        llm_node = loaded_view._node_items["llm_1"]

        out_port = in_node.output_pads["out"]
        in_port = llm_node.input_pads["in"]

        # Same direction should fail
        assert loaded_view.can_connect(in_port, in_port) is False

        # Duplicate connection should fail (already connected)
        assert loaded_view.can_connect(out_port, in_port) is False

    def test_grid_mixin_initialized(self, view):
        """Verify grid state was initialized by the mixin."""
        assert hasattr(view, '_snap_to_grid')
        assert hasattr(view, '_grid_size')
        assert hasattr(view, '_grid_visible')

    def test_input_mixin_initialized(self, view):
        """Verify input state was initialized by the mixin."""
        assert hasattr(view, '_space_pressed')
        assert hasattr(view, '_middle_panning')

    def test_clipboard_mixin_initialized(self, view):
        """Verify clipboard state was initialized by the mixin."""
        assert hasattr(view, '_clipboard_items')
        assert hasattr(view, '_clipboard_connections')

    def test_wire_mixin_initialized(self, view):
        """Verify wire state was initialized by the mixin."""
        assert hasattr(view, '_wire_being_drawn')
        assert hasattr(view, '_wire_start_port')
        assert view.is_drawing_wire is False
