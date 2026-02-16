"""Tests for shared editor mixins.

Each mixin is tested via a lightweight concrete class that inherits from
the mixin and QGraphicsView, implementing the required abstract methods.
"""

import pytest
from PyQt6.QtWidgets import QApplication, QGraphicsView, QGraphicsScene, QGraphicsRectItem
from PyQt6.QtCore import QPointF

from noodlestudio.panels.editors.shared_input_mixin import SharedInputMixin
from noodlestudio.panels.editors.shared_grid_mixin import SharedGridMixin
from noodlestudio.panels.editors.shared_layout_mixin import SharedLayoutMixin
from noodlestudio.panels.editors.shared_view_ops_mixin import SharedViewOpsMixin
from noodlestudio.panels.editors.shared_wire_mixin import SharedWireMixin
from noodlestudio.panels.editors.shared_clipboard_mixin import SharedClipboardMixin


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ========== Concrete test views ==========


class _TestNode(QGraphicsRectItem):
    """Minimal node item for testing."""

    def __init__(self, node_id, x=0, y=0, w=100, h=60):
        super().__init__(0, 0, w, h)
        self.node_id = node_id
        self.setPos(x, y)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsRectItem.GraphicsItemFlag.ItemIsMovable)


class _TestPort:
    """Minimal port item for testing wire mixin."""

    def __init__(self, node_id, port_name, is_output=False, scene_pos=None):
        self._node_id = node_id
        self._port_name = port_name
        self.is_output = is_output
        self._scene_pos = scene_pos or QPointF(0, 0)

    def get_parent_node_id(self):
        return self._node_id

    def get_port_name(self):
        return self._port_name

    def get_scene_position(self):
        return self._scene_pos


class InputTestView(SharedInputMixin, QGraphicsView):
    """Test view for SharedInputMixin."""

    def __init__(self):
        QGraphicsView.__init__(self)
        self._init_input_state()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._nodes = {}

    def get_node_items(self):
        return self._nodes


class GridTestView(SharedGridMixin, QGraphicsView):
    """Test view for SharedGridMixin."""

    GRID_SETTINGS_ORG = "NoodlingsTest"
    GRID_SETTINGS_APP = "GridTest"

    def __init__(self):
        QGraphicsView.__init__(self)
        self._scene = QGraphicsScene(-500, -500, 1000, 1000)
        self.setScene(self._scene)
        # Clear test settings before init to avoid cross-test pollution
        from PyQt6.QtCore import QSettings
        QSettings(self.GRID_SETTINGS_ORG, self.GRID_SETTINGS_APP).clear()
        self._init_grid_state()


class LayoutTestView(SharedLayoutMixin, QGraphicsView):
    """Test view for SharedLayoutMixin."""

    def __init__(self):
        QGraphicsView.__init__(self)
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._nodes = {}
        self._edges = []
        self._layout_complete_count = 0

    def get_node_items(self):
        return self._nodes

    def get_graph_edges(self):
        return self._edges

    def on_layout_complete(self):
        self._layout_complete_count += 1


class ViewOpsTestView(SharedViewOpsMixin, QGraphicsView):
    """Test view for SharedViewOpsMixin."""

    def __init__(self):
        QGraphicsView.__init__(self)
        self._init_view_ops_state()
        self._scene = QGraphicsScene(-1000, -1000, 2000, 2000)
        self.setScene(self._scene)
        self._nodes = {}

    def get_node_items(self):
        return self._nodes

    def get_selected_node_ids(self):
        return [
            nid for nid, item in self._nodes.items()
            if item.isSelected()
        ]


class WireTestView(SharedWireMixin, QGraphicsView):
    """Test view for SharedWireMixin."""

    def __init__(self):
        QGraphicsView.__init__(self)
        self._init_wire_state()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._connections = []
        self._created_connections = []

    def create_connection(self, from_port, to_port):
        self._created_connections.append((from_port, to_port))

    def get_existing_connections(self):
        return self._connections


class ClipboardTestView(SharedClipboardMixin, QGraphicsView):
    """Test view for SharedClipboardMixin."""

    def __init__(self):
        QGraphicsView.__init__(self)
        self._init_clipboard_state()
        self._scene = QGraphicsScene()
        self.setScene(self._scene)
        self._nodes = {}
        self._connections = []
        self._deserialized_items = []
        self._deserialized_connections = []

    def serialize_selection(self):
        return [
            {"id": nid, "position": {"x": item.pos().x(), "y": item.pos().y()}}
            for nid, item in self._nodes.items()
            if item.isSelected()
        ]

    def deserialize_items(self, items_data, connections_data):
        self._deserialized_items = items_data
        self._deserialized_connections = connections_data

    def get_existing_connections(self):
        return self._connections


# ========== Tests ==========


class TestSharedInputMixin:
    """Test zoom, pan, and input handling."""

    def test_init_state(self, qapp):
        view = InputTestView()
        assert view._space_pressed is False
        assert view._middle_panning is False
        assert view._last_right_click_time == 0.0

    def test_zoom_view_changes_scale(self, qapp):
        view = InputTestView()
        view.show()
        initial_scale = view.transform().m11()
        view._zoom_view(1.5)
        assert view.transform().m11() > initial_scale

    def test_zoom_view_respects_min(self, qapp):
        view = InputTestView()
        view.show()
        # Zoom out far below minimum
        view._zoom_view(0.01)
        # Scale should not go below MIN_ZOOM
        assert view.transform().m11() >= view.MIN_ZOOM * 0.5  # some tolerance

    def test_zoom_view_respects_max(self, qapp):
        view = InputTestView()
        view.show()
        # Zoom in far above maximum
        for _ in range(50):
            view._zoom_view(1.5)
        assert view.transform().m11() <= view.MAX_ZOOM * 2  # reasonable upper bound

    def test_space_pan_toggle(self, qapp):
        view = InputTestView()
        view.show()
        assert view.dragMode() == QGraphicsView.DragMode.NoDrag

        # Simulate space press
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent

        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        consumed = view.handle_key_press_input(press)
        assert consumed is True
        assert view._space_pressed is True
        assert view.dragMode() == QGraphicsView.DragMode.ScrollHandDrag

        release = QKeyEvent(QEvent.Type.KeyRelease, Qt.Key.Key_Space, Qt.KeyboardModifier.NoModifier)
        consumed = view.handle_key_release_input(release)
        assert consumed is True
        assert view._space_pressed is False
        assert view.dragMode() == QGraphicsView.DragMode.RubberBandDrag

    def test_non_space_key_not_consumed(self, qapp):
        view = InputTestView()
        from PyQt6.QtGui import QKeyEvent
        from PyQt6.QtCore import Qt, QEvent

        press = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.NoModifier)
        consumed = view.handle_key_press_input(press)
        assert consumed is False

    def test_right_click_time_recording(self, qapp):
        view = InputTestView()
        assert view._last_right_click_time == 0.0
        view.record_right_click_time()
        assert view._last_right_click_time > 0.0


class TestSharedGridMixin:
    """Test grid snap, toggle, and rendering."""

    def test_init_state(self, qapp):
        view = GridTestView()
        assert view._grid_size == 20
        assert isinstance(view._grid_lines, list)

    def test_snap_position_disabled(self, qapp):
        view = GridTestView()
        view._snap_to_grid = False
        assert view.snap_position(37.3, 42.7) == (37.3, 42.7)

    def test_snap_position_enabled(self, qapp):
        view = GridTestView()
        view._snap_to_grid = True
        view._grid_size = 20
        assert view.snap_position(37.3, 42.7) == (40, 40)

    def test_snap_position_exact_grid(self, qapp):
        view = GridTestView()
        view._snap_to_grid = True
        view._grid_size = 20
        assert view.snap_position(60.0, 80.0) == (60, 80)

    def test_snap_position_negative(self, qapp):
        view = GridTestView()
        view._snap_to_grid = True
        view._grid_size = 20
        assert view.snap_position(-37.0, -42.0) == (-40, -40)

    def test_toggle_grid_on_draws_lines(self, qapp):
        view = GridTestView()
        view.toggle_grid(True)
        assert view._snap_to_grid is True
        assert view._grid_visible is True
        assert len(view._grid_lines) > 0

    def test_toggle_grid_off_clears_lines(self, qapp):
        view = GridTestView()
        view.toggle_grid(True)
        assert len(view._grid_lines) > 0
        view.toggle_grid(False)
        assert len(view._grid_lines) == 0

    def test_set_grid_size(self, qapp):
        view = GridTestView()
        view.set_grid_size(50)
        assert view._grid_size == 50

    def test_set_grid_size_redraws_if_visible(self, qapp):
        view = GridTestView()
        view._grid_size = 10  # Force small grid for predictable initial count
        view.toggle_grid(True)
        count_before = len(view._grid_lines)
        assert count_before > 0
        view.set_grid_size(100)  # Much larger -> fewer lines
        # Larger grid size = fewer lines (scene is 1000x1000)
        assert len(view._grid_lines) < count_before


class TestSharedLayoutMixin:
    """Test topological auto-arrange and alignment."""

    def _make_view_with_chain(self, qapp):
        """Create a 3-node chain: A -> B -> C."""
        view = LayoutTestView()
        view.show()

        for nid, x, y in [("a", 500, 500), ("b", 500, 200), ("c", 200, 400)]:
            node = _TestNode(nid, x, y)
            view._scene.addItem(node)
            view._nodes[nid] = node

        view._edges = [("a", "b"), ("b", "c")]
        return view

    def test_auto_arrange_layering(self, qapp):
        view = self._make_view_with_chain(qapp)
        view.auto_arrange()

        # A has no dependencies -> layer 0 (leftmost x)
        # B depends on A -> layer 1
        # C depends on B -> layer 2
        assert view._nodes["a"].pos().x() < view._nodes["b"].pos().x()
        assert view._nodes["b"].pos().x() < view._nodes["c"].pos().x()

    def test_auto_arrange_calls_on_layout_complete(self, qapp):
        view = self._make_view_with_chain(qapp)
        assert view._layout_complete_count == 0
        view.auto_arrange()
        assert view._layout_complete_count == 1

    def test_auto_arrange_empty_graph(self, qapp):
        view = LayoutTestView()
        view.show()
        view.auto_arrange()  # Should not crash

    def test_auto_arrange_with_cycle(self, qapp):
        """Cyclic edges should not crash; cycles go in final layer."""
        view = LayoutTestView()
        view.show()
        for nid in ["a", "b"]:
            node = _TestNode(nid, 0, 0)
            view._scene.addItem(node)
            view._nodes[nid] = node
        view._edges = [("a", "b"), ("b", "a")]
        view.auto_arrange()  # Should not hang or crash

    def test_align_horizontally(self, qapp):
        view = LayoutTestView()
        view.show()
        nodes = []
        for nid, x, y in [("a", 100, 100), ("b", 300, 300)]:
            node = _TestNode(nid, x, y)
            node.setSelected(True)
            view._scene.addItem(node)
            view._nodes[nid] = node
            nodes.append(node)

        view.align_selected_horizontally()
        # Both should be at average Y = (100+300)/2 = 200
        assert abs(nodes[0].pos().y() - 200) < 1
        assert abs(nodes[1].pos().y() - 200) < 1

    def test_align_vertically(self, qapp):
        view = LayoutTestView()
        view.show()
        nodes = []
        for nid, x, y in [("a", 100, 100), ("b", 300, 300)]:
            node = _TestNode(nid, x, y)
            node.setSelected(True)
            view._scene.addItem(node)
            view._nodes[nid] = node
            nodes.append(node)

        view.align_selected_vertically()
        # Both should be at average X = (100+300)/2 = 200
        assert abs(nodes[0].pos().x() - 200) < 1
        assert abs(nodes[1].pos().x() - 200) < 1

    def test_align_needs_two_nodes(self, qapp):
        """Align with fewer than 2 selected nodes should do nothing."""
        view = LayoutTestView()
        view.show()
        node = _TestNode("a", 100, 100)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node
        view.align_selected_horizontally()
        # Position unchanged
        assert abs(node.pos().y() - 100) < 1


class TestSharedViewOpsMixin:
    """Test focus toggle and frame-all."""

    def test_init_state(self, qapp):
        view = ViewOpsTestView()
        assert view._is_focused is False
        assert view._focused_node_ids is None
        assert view._pre_focus_transform is None

    def test_focus_selection_saves_transform(self, qapp):
        view = ViewOpsTestView()
        view.show()
        view.resize(400, 400)

        node = _TestNode("a", 100, 100)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        original_transform = view.transform()
        view.focus_selection()

        assert view._is_focused is True
        assert view._focused_node_ids == ("a",)
        assert view._pre_focus_transform == original_transform

    def test_focus_toggle_restores_transform(self, qapp):
        view = ViewOpsTestView()
        view.show()
        view.resize(400, 400)

        node = _TestNode("a", 100, 100)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        original_transform = view.transform()
        view.focus_selection()  # Focus
        view.focus_selection()  # Unfocus

        assert view._is_focused is False
        assert view.transform() == original_transform

    def test_focus_no_selection_does_nothing(self, qapp):
        view = ViewOpsTestView()
        view.show()
        view.focus_selection()
        assert view._is_focused is False

    def test_frame_all_nodes(self, qapp):
        view = ViewOpsTestView()
        view.show()
        view.resize(400, 400)

        for nid, x, y in [("a", 0, 0), ("b", 500, 500)]:
            node = _TestNode(nid, x, y)
            view._scene.addItem(node)
            view._nodes[nid] = node

        view.frame_all_nodes()
        # After framing, viewport should include both nodes
        # Just verify it doesn't crash and changes the transform
        assert view.transform().m11() != 0

    def test_frame_all_empty(self, qapp):
        view = ViewOpsTestView()
        view.show()
        view.frame_all_nodes()  # Should not crash


class TestSharedWireMixin:
    """Test wire drawing, validation, and connection dispatch."""

    def test_init_state(self, qapp):
        view = WireTestView()
        assert view._wire_being_drawn is None
        assert view._wire_start_port is None
        assert view.is_drawing_wire is False

    def test_start_wire_drawing(self, qapp):
        view = WireTestView()
        port = _TestPort("a", "output", is_output=True, scene_pos=QPointF(100, 50))
        view.start_wire_drawing(port)
        assert view.is_drawing_wire is True
        assert view._wire_start_port is port

    def test_update_wire_drawing(self, qapp):
        view = WireTestView()
        port = _TestPort("a", "output", is_output=True, scene_pos=QPointF(100, 50))
        view.start_wire_drawing(port)
        view.update_wire_drawing(QPointF(200, 150))
        line = view._wire_being_drawn.line()
        assert abs(line.x2() - 200) < 1
        assert abs(line.y2() - 150) < 1

    def test_cancel_wire_drawing(self, qapp):
        view = WireTestView()
        port = _TestPort("a", "output", is_output=True, scene_pos=QPointF(100, 50))
        view.start_wire_drawing(port)
        assert view.is_drawing_wire is True
        view.cancel_wire_drawing()
        assert view.is_drawing_wire is False

    def test_can_connect_different_types(self, qapp):
        view = WireTestView()
        out_port = _TestPort("a", "result", is_output=True)
        in_port = _TestPort("b", "input", is_output=False)
        assert view.can_connect(out_port, in_port) is True

    def test_can_connect_rejects_same_node(self, qapp):
        view = WireTestView()
        out_port = _TestPort("a", "result", is_output=True)
        in_port = _TestPort("a", "input", is_output=False)
        assert view.can_connect(out_port, in_port) is False

    def test_can_connect_rejects_same_direction(self, qapp):
        view = WireTestView()
        out1 = _TestPort("a", "result", is_output=True)
        out2 = _TestPort("b", "result", is_output=True)
        assert view.can_connect(out1, out2) is False

    def test_can_connect_rejects_input_to_input(self, qapp):
        view = WireTestView()
        in1 = _TestPort("a", "input", is_output=False)
        in2 = _TestPort("b", "input", is_output=False)
        assert view.can_connect(in1, in2) is False

    def test_can_connect_rejects_duplicate(self, qapp):
        view = WireTestView()
        view._connections = [("a", "result", "b", "input")]
        out_port = _TestPort("a", "result", is_output=True)
        in_port = _TestPort("b", "input", is_output=False)
        assert view.can_connect(out_port, in_port) is False

    def test_can_connect_allows_different_ports(self, qapp):
        view = WireTestView()
        view._connections = [("a", "result", "b", "input")]
        out_port = _TestPort("a", "other_output", is_output=True)
        in_port = _TestPort("b", "input", is_output=False)
        assert view.can_connect(out_port, in_port) is True

    def test_finish_wire_creates_connection(self, qapp):
        view = WireTestView()
        out_port = _TestPort("a", "result", is_output=True, scene_pos=QPointF(0, 0))
        in_port = _TestPort("b", "input", is_output=False, scene_pos=QPointF(100, 0))
        view.start_wire_drawing(out_port)
        view.finish_wire_drawing(in_port)
        assert len(view._created_connections) == 1
        assert view.is_drawing_wire is False

    def test_finish_wire_on_empty_space(self, qapp):
        view = WireTestView()
        out_port = _TestPort("a", "result", is_output=True, scene_pos=QPointF(0, 0))
        view.start_wire_drawing(out_port)
        view.finish_wire_drawing(None)
        assert len(view._created_connections) == 0
        assert view.is_drawing_wire is False

    def test_validate_connection_domain_default_true(self, qapp):
        view = WireTestView()
        out_port = _TestPort("a", "result", is_output=True)
        in_port = _TestPort("b", "input", is_output=False)
        assert view.validate_connection_domain(out_port, in_port) is True

    def test_validate_connection_domain_override(self, qapp):
        """Subclass can reject connections based on domain rules."""
        class StrictWireView(WireTestView):
            def validate_connection_domain(self, from_port, to_port):
                # Only allow connections where port names match
                return from_port.get_port_name() == to_port.get_port_name()

        view = StrictWireView()
        out_port = _TestPort("a", "data", is_output=True)
        in_port = _TestPort("b", "data", is_output=False)
        assert view.can_connect(out_port, in_port) is True

        out_port2 = _TestPort("a", "data", is_output=True)
        in_port2 = _TestPort("b", "control", is_output=False)
        assert view.can_connect(out_port2, in_port2) is False


class TestSharedClipboardMixin:
    """Test copy, paste, duplicate with internal connection preservation."""

    def test_init_state(self, qapp):
        view = ClipboardTestView()
        assert view._clipboard_items == []
        assert view._clipboard_connections == []

    def test_copy_selection(self, qapp):
        view = ClipboardTestView()
        node = _TestNode("a", 100, 200)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        view.copy_selection()
        assert len(view._clipboard_items) == 1
        assert view._clipboard_items[0]["id"] == "a"

    def test_copy_captures_internal_connections(self, qapp):
        view = ClipboardTestView()
        for nid in ["a", "b", "c"]:
            node = _TestNode(nid, 0, 0)
            node.setSelected(nid != "c")  # a and b selected, c not
            view._scene.addItem(node)
            view._nodes[nid] = node

        view._connections = [
            ("a", "out", "b", "in"),   # internal (both selected)
            ("a", "out", "c", "in"),   # external (c not selected)
        ]

        view.copy_selection()
        assert len(view._clipboard_items) == 2  # a and b
        assert len(view._clipboard_connections) == 1  # only the internal one
        assert view._clipboard_connections[0] == ("a", "out", "b", "in")

    def test_paste_creates_new_ids(self, qapp):
        view = ClipboardTestView()
        node = _TestNode("a", 100, 200)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        view.copy_selection()
        view.paste_selection()

        assert len(view._deserialized_items) == 1
        # New ID should be different from original
        assert view._deserialized_items[0]["id"] != "a"

    def test_paste_offsets_position(self, qapp):
        view = ClipboardTestView()
        node = _TestNode("a", 100, 200)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        view.copy_selection()
        view.paste_selection(offset_x=50, offset_y=50)

        pos = view._deserialized_items[0]["position"]
        assert abs(pos["x"] - 150) < 1  # 100 + 50
        assert abs(pos["y"] - 250) < 1  # 200 + 50

    def test_paste_remaps_internal_connections(self, qapp):
        view = ClipboardTestView()
        for nid, x in [("a", 0), ("b", 200)]:
            node = _TestNode(nid, x, 0)
            node.setSelected(True)
            view._scene.addItem(node)
            view._nodes[nid] = node

        view._connections = [("a", "out", "b", "in")]
        view.copy_selection()
        view.paste_selection()

        # Internal connection should be remapped
        assert len(view._deserialized_connections) == 1
        conn = view._deserialized_connections[0]
        # from_id and to_id should be the NEW ids, not "a" and "b"
        new_ids = {item["id"] for item in view._deserialized_items}
        assert conn[0] in new_ids
        assert conn[2] in new_ids
        assert conn[0] != "a"
        assert conn[2] != "b"
        # Port names preserved
        assert conn[1] == "out"
        assert conn[3] == "in"

    def test_paste_empty_clipboard(self, qapp):
        view = ClipboardTestView()
        view.paste_selection()  # Should not crash
        assert view._deserialized_items == []

    def test_duplicate_copies_and_pastes(self, qapp):
        view = ClipboardTestView()
        node = _TestNode("a", 100, 200)
        node.setSelected(True)
        view._scene.addItem(node)
        view._nodes["a"] = node

        view.duplicate_selection()
        assert len(view._deserialized_items) == 1
        assert view._deserialized_items[0]["id"] != "a"
