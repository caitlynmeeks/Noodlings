"""
Neural Canvas View - Visual rendering of neural network graph.

Handles node rendering, wire routing, pan/zoom, and interactions.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

from PyQt6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsItem, QMenu, QInputDialog
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont,
    QWheelEvent, QMouseEvent, QPainterPath, QAction
)

from ...core.neural_canvas.neural_graph import NeuralGraph
from ...core.neural_canvas.neural_node import NeuralNode, NodeType, Connection
from ...core.neural_canvas.node_definitions import create_node_from_type, get_node_color


class PortGraphicsItem(QGraphicsItem):
    """Graphics item representing an input or output port."""

    def __init__(self, port_name: str, port, is_output: bool, index: int, is_connected: bool = False, parent=None):
        super().__init__(parent)
        self.port_name = port_name
        self.port = port  # Port object with data_type, shape, etc.
        self.is_output = is_output
        self.is_connected = is_connected
        self.index = index
        self.radius = 5

        # Vertical spacing for ports
        port_spacing = 20
        # Y position depends on whether we're showing params + whether ports come after params
        # The parent NodeGraphicsItem will need to calculate this properly
        port_start_y = 35  # Will be adjusted by parent

        # Position relative to parent node
        if is_output:
            self.setPos(180, port_start_y + index * port_spacing)  # Right edge (updated for wider nodes)
        else:
            self.setPos(0, port_start_y + index * port_spacing)  # Left edge

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setZValue(10)  # Draw on top
        self.setAcceptHoverEvents(True)

    def boundingRect(self) -> QRectF:
        # Expand bounding rect to include label
        if self.is_output:
            # Output: label to the left of circle
            return QRectF(-80, -self.radius, 80 + self.radius, self.radius * 2)
        else:
            # Input: label to the right of circle
            return QRectF(-self.radius, -self.radius, 80 + self.radius, self.radius * 2)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the port as a circle with label."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Port color based on data type (coffee shop palette - darker, saturated)
        data_type_colors = {
            'AFFECT': '#8A7A4A',       # Muted gold (affective warmth)
            'HIDDEN_STATE': '#4A5A6A', # Slate blue (recurrent state)
            'CELL_STATE': '#5A6A7A',   # Steel blue (LSTM cell)
            'PHENOMENAL_STATE': '#6A4A6A', # Deep mauve (consciousness)
            'TENSOR': '#5A5A5A',       # Medium gray
            'SCALAR': '#6A6A6A'        # Light gray
        }

        color = QColor(data_type_colors.get(self.port.data_type.value, '#555'))

        # Draw circle
        painter.setPen(QPen(QColor("#333"), 1.5))
        painter.setBrush(QBrush(color))
        circle_rect = QRectF(-self.radius, -self.radius, self.radius * 2, self.radius * 2)
        painter.drawEllipse(circle_rect)

        # Draw label (warm white, uniform brightness)
        painter.setPen(QColor("#e8e8e0"))  # Warm white, matches node text
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Use display label (human-readable)
        display_label = self.port.get_display_label()

        if self.is_output:
            # Output: label on left of circle
            label_rect = QRectF(-75, -self.radius, 70, self.radius * 2)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, display_label)
        else:
            # Input: label on right of circle
            label_rect = QRectF(self.radius + 5, -self.radius, 70, self.radius * 2)
            painter.drawText(label_rect, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, display_label)

    def hoverEnterEvent(self, event):
        """Show tooltip on hover."""
        # Format tooltip like Blender
        shape_str = f"{self.port.shape}" if self.port.shape else "dynamic"
        tooltip = f"{self.port.get_display_label()}\nTechnical: {self.port_name}\nType: {self.port.data_type.value}\nShape: {shape_str}"
        self.setToolTip(tooltip)
        super().hoverEnterEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Double-click to rename port label."""
        new_label, ok = QInputDialog.getText(
            None,
            "Rename Port",
            f"Enter label for '{self.port_name}':",
            text=self.port.get_display_label()
        )

        if ok and new_label:
            self.port.label = new_label
            self.update()  # Redraw
            print(f"[Neural Canvas] Renamed port '{self.port_name}' to '{new_label}'")

            # Emit modification signal
            if self.scene():
                for view in self.scene().views():
                    if isinstance(view, NeuralCanvasView):
                        view.graph_modified.emit()
                        break

        super().mouseDoubleClickEvent(event)

    def get_connection_point(self) -> QPointF:
        """Get the point where wires should connect (in scene coordinates)."""
        return self.scenePos()


class NodeGraphicsItem(QGraphicsItem):
    """Graphics item representing a neural network node."""

    def __init__(self, node: NeuralNode, parent=None):
        super().__init__(parent)
        self.node = node
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Visual properties (wider to fit labels, taller for ports + params)
        self.width = 180

        # Calculate height: header + params + separator + ports
        header_height = 24
        # Estimate params count (will calculate properly in _get_display_params)
        num_params = len(node.params) if node.params else 0
        params_height = num_params * 14 + (8 if num_params > 0 else 0)  # 14px per param + separator
        max_ports = max(len(node.inputs), len(node.outputs), 1)
        ports_height = max_ports * 20 + 20  # 20px per port + margin

        self.height = header_height + params_height + ports_height
        self.setPos(node.position[0], node.position[1])

        # Calculate where ports start (after header + params)
        display_params = self._get_display_params()
        self.params_height = len(display_params) * 14 + (8 if display_params else 0)
        self.port_start_y = header_height + self.params_height + 8

        # Create port items
        self.input_ports: dict[str, PortGraphicsItem] = {}
        self.output_ports: dict[str, PortGraphicsItem] = {}

        for i, (port_name, port) in enumerate(node.inputs.items()):
            port_item = PortGraphicsItem(port_name, port, is_output=False, index=i, is_connected=False, parent=self)
            # Adjust position to account for params
            port_item.setPos(0, self.port_start_y + i * 20)
            self.input_ports[port_name] = port_item

        for i, (port_name, port) in enumerate(node.outputs.items()):
            port_item = PortGraphicsItem(port_name, port, is_output=True, index=i, is_connected=False, parent=self)
            # Adjust position to account for params
            port_item.setPos(self.width, self.port_start_y + i * 20)
            self.output_ports[port_name] = port_item

    def _get_display_params(self) -> dict:
        """Get key parameters to display inline on the node (moved up for init access)."""
        display = {}

        if self.node.type in (NodeType.LSTM, NodeType.GRU, NodeType.RNN):
            if 'hidden_dim' in self.node.params:
                display['hidden'] = self.node.params['hidden_dim']
            if 'dropout' in self.node.params and self.node.params['dropout'] > 0:
                display['dropout'] = self.node.params['dropout']

        elif self.node.type == NodeType.LINEAR:
            if 'out_features' in self.node.params:
                display['out'] = self.node.params['out_features']

        elif self.node.type == NodeType.DROPOUT:
            if 'p' in self.node.params:
                display['p'] = self.node.params['p']

        elif self.node.type == NodeType.AFFECT_HEAD:
            if 'hidden_dim' in self.node.params:
                display['hidden'] = self.node.params['hidden_dim']

        elif self.node.type == NodeType.QUANTUM_MICROTUBULE:
            if 'collapse_threshold' in self.node.params:
                display['collapse'] = self.node.params['collapse_threshold']
            if 'noise_scale' in self.node.params:
                display['noise'] = self.node.params['noise_scale']

        elif self.node.type == NodeType.IBM_QUANTUM:
            if 'num_qubits' in self.node.params:
                display['qubits'] = self.node.params['num_qubits']
            if 'shots' in self.node.params:
                display['shots'] = self.node.params['shots']

        # Show total parameters if node has weights
        total_params = self.node.compute_num_parameters()
        if total_params > 0:
            display['params'] = f"{total_params:,}"

        return display

    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, self.width, self.height)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the node (Blender-style: colored header, gray body)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.boundingRect()

        # Main body background (uniform dark gray) - rounded corners
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.drawRoundedRect(rect, 4, 4)

        # Header bar with taxonomic color (sharp edges, clipped by node outline)
        header_height = 24
        header_rect = QRectF(0, 0, self.width, header_height)

        # Get node type color (rich, earthy tones)
        header_color = QColor(get_node_color(self.node.type))
        painter.setBrush(QBrush(header_color))
        painter.drawRect(header_rect)  # Sharp rectangle, not rounded

        # Selection highlight (white outline with padding, drawn LAST on top)
        if self.isSelected():
            padding = 3  # Pixels between node and selection box
            selection_rect = rect.adjusted(-padding, -padding, padding, padding)
            painter.setPen(QPen(QColor("#FFFFFF"), 3))  # 3px border (matches Facets)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(selection_rect, 4, 4)

        # Node name in header (warm white, uniform brightness)
        painter.setPen(QColor("#e8e8e0"))  # Warm white
        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            header_rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.node.name
        )

        # Node type (same warm white, smaller)
        painter.setPen(QColor("#e8e8e0"))  # Same brightness
        font = QFont("Arial", 7)
        painter.setFont(font)
        painter.drawText(
            header_rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            self.node.type.value
        )

        # Draw inline parameters (between header and ports)
        self._paint_parameters(painter, header_height)

    def _paint_parameters(self, painter: QPainter, y_offset: float):
        """
        Paint inline parameter values (Blender-style).

        Shows key parameters like hidden_dim, dropout, etc. directly on node.
        """
        font = QFont("Arial", 8)
        painter.setFont(font)

        current_y = y_offset + 6

        # Key parameters to display inline
        display_params = self._get_display_params()

        # Uniform warm white for all text
        painter.setPen(QColor("#e8e8e0"))

        for param_name, param_value in display_params.items():
            # Parameter name
            param_rect = QRectF(10, current_y, 70, 12)
            painter.drawText(param_rect, Qt.AlignmentFlag.AlignLeft, param_name + ":")

            # Parameter value (same color, uniform brightness)
            value_rect = QRectF(85, current_y, 85, 12)

            # Format value
            if isinstance(param_value, float):
                value_str = f"{param_value:.3f}"
            elif isinstance(param_value, bool):
                value_str = "✓" if param_value else "✗"
            else:
                value_str = str(param_value)

            painter.drawText(value_rect, Qt.AlignmentFlag.AlignRight, value_str)

            current_y += 14

        # Draw separator line before ports
        if display_params:
            painter.setPen(QPen(QColor("#555"), 1))
            painter.drawLine(8, int(current_y + 2), int(self.width - 8), int(current_y + 2))

    def itemChange(self, change, value):
        """Handle item changes (e.g., position)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update node position
            new_pos = self.pos()
            self.node.position = (int(new_pos.x()), int(new_pos.y()))

            # Force redraw of all wires connected to this node
            if self.scene():
                self.scene().update()

        return super().itemChange(change, value)

    def get_port_item(self, port_name: str, is_output: bool) -> PortGraphicsItem:
        """Get port graphics item by name."""
        if is_output:
            return self.output_ports.get(port_name)
        else:
            return self.input_ports.get(port_name)

    def mouseDoubleClickEvent(self, event):
        """Handle double-click to rename node."""
        # Check if double-clicking on header
        header_height = 24
        local_pos = event.pos()

        if local_pos.y() <= header_height:
            # Double-clicked header - rename node
            new_name, ok = QInputDialog.getText(
                None,
                "Rename Node",
                "Enter new name:",
                text=self.node.name
            )

            if ok and new_name and new_name != self.node.name:
                self.node.name = new_name
                self.update()  # Redraw
                print(f"[Neural Canvas] Renamed node to: {new_name}")

                # Emit modification signal (need to get parent view)
                if self.scene():
                    for view in self.scene().views():
                        if isinstance(view, NeuralCanvasView):
                            view.graph_modified.emit()
                            break

        super().mouseDoubleClickEvent(event)


class ConnectionGraphicsItem(QGraphicsItem):
    """Graphics item representing a connection between nodes."""

    def __init__(self, connection: Connection, from_item: NodeGraphicsItem, to_item: NodeGraphicsItem, parent=None):
        super().__init__(parent)
        self.connection = connection
        self.from_item = from_item
        self.to_item = to_item
        self.setZValue(-1)  # Draw behind nodes

    def boundingRect(self) -> QRectF:
        """Bounding rectangle for the connection."""
        from_port = self.from_item.get_port_item(self.connection.from_port, is_output=True)
        to_port = self.to_item.get_port_item(self.connection.to_port, is_output=False)

        if from_port and to_port:
            from_pos = from_port.get_connection_point()
            to_pos = to_port.get_connection_point()
        else:
            # Fallback to node centers
            from_pos = self.from_item.scenePos() + QPointF(self.from_item.width, self.from_item.height / 2)
            to_pos = self.to_item.scenePos() + QPointF(0, self.to_item.height / 2)

        return QRectF(from_pos, to_pos).normalized().adjusted(-10, -10, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the connection wire (orthogonal routing, color-coded by type)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)  # Sharp lines

        # Get port positions
        from_port = self.from_item.get_port_item(self.connection.from_port, is_output=True)
        to_port = self.to_item.get_port_item(self.connection.to_port, is_output=False)

        if from_port and to_port:
            from_pos = from_port.get_connection_point()
            to_pos = to_port.get_connection_point()
            # Wire color matches output port data type
            wire_color = self._get_wire_color(from_port.port.data_type.value)
        else:
            # Fallback
            from_pos = self.from_item.scenePos() + QPointF(self.from_item.width, self.from_item.height / 2)
            to_pos = self.to_item.scenePos() + QPointF(0, self.to_item.height / 2)
            wire_color = "#888"

        # Wire color based on data type (matches port colors)
        pen = QPen(QColor(wire_color), 2.5)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)  # Smooth curves

        # Bezier curve routing (Blender-style)
        # Control points extend horizontally from ports
        distance = abs(to_pos.x() - from_pos.x())
        handle_distance = min(distance * 0.5, 100)  # Adaptive handle length

        control1 = QPointF(from_pos.x() + handle_distance, from_pos.y())  # Horizontal right
        control2 = QPointF(to_pos.x() - handle_distance, to_pos.y())      # Horizontal left

        path = QPainterPath()
        path.moveTo(from_pos)
        path.cubicTo(control1, control2, to_pos)  # Smooth Bezier curve

        painter.drawPath(path)

        # Arrowhead at target (no fill, just outline to match Blender)
        arrow_size = 6
        painter.setBrush(QBrush(QColor(wire_color)))
        arrow = QPainterPath()
        arrow.moveTo(to_pos)
        arrow.lineTo(to_pos.x() - arrow_size, to_pos.y() - arrow_size / 2)
        arrow.lineTo(to_pos.x() - arrow_size, to_pos.y() + arrow_size / 2)
        arrow.closeSubpath()
        painter.drawPath(arrow)

    def _get_wire_color(self, data_type: str) -> str:
        """Get wire color based on data type (coffee shop palette - matches ports)."""
        data_type_colors = {
            'AFFECT': '#8A7A4A',       # Muted gold
            'HIDDEN_STATE': '#4A5A6A', # Slate blue
            'CELL_STATE': '#5A6A7A',   # Steel blue
            'PHENOMENAL_STATE': '#6A4A6A', # Deep mauve
            'TENSOR': '#5A5A5A',       # Medium gray
            'SCALAR': '#6A6A6A'        # Light gray
        }
        return data_type_colors.get(data_type, '#555')


class TemporaryWireItem(QGraphicsItem):
    """Temporary wire shown while dragging to create a connection."""

    def __init__(self, start_pos: QPointF, parent=None):
        super().__init__(parent)
        self.start_pos = start_pos
        self.end_pos = start_pos
        self.setZValue(5)  # Above connections, below ports

    def set_end_pos(self, pos: QPointF):
        """Update the end position of the wire."""
        self.prepareGeometryChange()
        self.end_pos = pos

    def boundingRect(self) -> QRectF:
        return QRectF(self.start_pos, self.end_pos).normalized().adjusted(-10, -10, 10, 10)

    def paint(self, painter: QPainter, option, widget=None):
        """Render temporary wire."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Dashed line
        pen = QPen(QColor("#FFA500"), 2, Qt.PenStyle.DashLine)  # Orange dashed
        painter.setPen(pen)

        # Simple straight line for now
        painter.drawLine(self.start_pos, self.end_pos)


class NeuralCanvasView(QGraphicsView):
    """
    Graphics view for rendering and interacting with neural network graph.

    Features:
    - Pan (middle mouse drag)
    - Zoom (scroll wheel)
    - Node dragging
    - Node selection
    - Grid (optional)
    """

    # Signals
    node_selected = pyqtSignal(str)  # node_id
    graph_modified = pyqtSignal()

    def __init__(self, graph: NeuralGraph, parent=None):
        super().__init__(parent)

        self.graph = graph
        self.scene = QGraphicsScene()
        self.setScene(self.scene)

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Enable rectangle selection
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # Style (lighter background for better visibility)
        self.setStyleSheet("""
            QGraphicsView {
                background: #2a2a2a;
                border: none;
            }
        """)

        # Interaction state
        self.panning = False
        self.last_pan_pos = QPointF()
        self.space_pressed = False

        # Add node mode
        self.add_node_mode = False
        self.add_node_type = None

        # Wire dragging mode
        self.wire_drag_mode = False
        self.wire_drag_start_port: PortGraphicsItem = None
        self.wire_drag_start_node: NodeGraphicsItem = None
        self.temp_wire: TemporaryWireItem = None

        # Focus state (F key toggle)
        self.is_focused = False
        self.focused_node_id: str = None
        self.pre_focus_transform = None

        # Graphics items cache
        self.node_items: dict[str, NodeGraphicsItem] = {}
        self.connection_items: list[ConnectionGraphicsItem] = []

        # Initial render
        self._render_graph()

    def set_graph(self, graph: NeuralGraph):
        """Set a new graph to display."""
        self.graph = graph
        self._render_graph()

    def _render_graph(self):
        """Render the graph as graphics items."""
        self.scene.clear()
        self.node_items.clear()
        self.connection_items.clear()

        # Create node items
        for node_id, node in self.graph.nodes.items():
            item = NodeGraphicsItem(node)
            self.scene.addItem(item)
            self.node_items[node_id] = item

        # Create connection items
        for connection in self.graph.connections:
            from_item = self.node_items.get(connection.from_node)
            to_item = self.node_items.get(connection.to_node)

            if from_item and to_item:
                conn_item = ConnectionGraphicsItem(connection, from_item, to_item)
                self.scene.addItem(conn_item)
                self.connection_items.append(conn_item)

    def start_add_node_mode(self, node_type: NodeType):
        """Enter mode to add a node of the given type."""
        self.add_node_mode = True
        self.add_node_type = node_type
        self.setCursor(Qt.CursorShape.CrossCursor)

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self.scale(zoom_factor, zoom_factor)
        else:
            self.scale(1 / zoom_factor, 1 / zoom_factor)

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Space and not self.space_pressed:
            # Space pressed - switch to pan mode
            self.space_pressed = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() == Qt.Key.Key_F:
            # F key - Focus on selected node (toggle)
            self.focus_selection()
        elif event.key() == Qt.Key.Key_A:
            # A key - Frame all nodes
            self.frame_all_nodes()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.key() == Qt.Key.Key_Space and self.space_pressed:
            # Space released - back to selection mode
            self.space_pressed = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            # Start panning
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton:
            if self.add_node_mode:
                # Add node at click position
                scene_pos = self.mapToScene(event.pos())
                self._add_node_at_position(scene_pos)
                self.add_node_mode = False
                self.add_node_type = None
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
            else:
                # Check if clicking on a port
                scene_pos = self.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                port_clicked = None
                parent_node = None

                for item in items:
                    if isinstance(item, PortGraphicsItem):
                        port_clicked = item
                        parent_node = item.parentItem()
                        break

                if port_clicked and parent_node:
                    # Start wire drag
                    self.wire_drag_mode = True
                    self.wire_drag_start_port = port_clicked
                    self.wire_drag_start_node = parent_node

                    # Create temporary wire
                    start_pos = port_clicked.get_connection_point()
                    self.temp_wire = TemporaryWireItem(start_pos)
                    self.scene.addItem(self.temp_wire)

                    event.accept()
                else:
                    # Normal selection
                    super().mousePressEvent(event)
                    # Emit selection signal
                    selected_items = self.scene.selectedItems()
                    if selected_items and isinstance(selected_items[0], NodeGraphicsItem):
                        self.node_selected.emit(selected_items[0].node.id)
        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click should NOT change selection (context menu preserves selection)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self.panning:
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
        elif self.wire_drag_mode and self.temp_wire:
            # Update temporary wire end position
            scene_pos = self.mapToScene(event.pos())
            self.temp_wire.set_end_pos(scene_pos)
            self.scene.update()
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        elif event.button() == Qt.MouseButton.LeftButton and self.wire_drag_mode:
            # Complete wire drag
            scene_pos = self.mapToScene(event.pos())
            items = self.scene.items(scene_pos)

            target_port = None
            target_node = None

            for item in items:
                if isinstance(item, PortGraphicsItem):
                    target_port = item
                    target_node = item.parentItem()
                    break

            # Validate connection
            if target_port and target_node and self.wire_drag_start_port:
                # Check if connecting output → input (or input → output)
                if self.wire_drag_start_port.is_output != target_port.is_output:
                    # Determine from/to
                    if self.wire_drag_start_port.is_output:
                        from_node = self.wire_drag_start_node
                        from_port = self.wire_drag_start_port.port_name
                        to_node = target_node
                        to_port = target_port.port_name
                    else:
                        from_node = target_node
                        from_port = target_port.port_name
                        to_node = self.wire_drag_start_node
                        to_port = self.wire_drag_start_port.port_name

                    # Create connection
                    connection = Connection(
                        from_node=from_node.node.id,
                        from_port=from_port,
                        to_node=to_node.node.id,
                        to_port=to_port
                    )

                    try:
                        self.graph.add_connection(connection)

                        # Create graphics item
                        conn_item = ConnectionGraphicsItem(connection, from_node, to_node)
                        self.scene.addItem(conn_item)
                        self.connection_items.append(conn_item)

                        self.graph_modified.emit()

                    except ValueError as e:
                        print(f"[Neural Canvas] Connection error: {e}")

            # Clean up temporary wire
            if self.temp_wire:
                self.scene.removeItem(self.temp_wire)
                self.temp_wire = None

            self.wire_drag_mode = False
            self.wire_drag_start_port = None
            self.wire_drag_start_node = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)

            # Check if node position changed
            if event.button() == Qt.MouseButton.LeftButton:
                self.graph_modified.emit()

    def contextMenuEvent(self, event):
        """Handle right-click context menu (preserves selection)."""
        scene_pos = self.mapToScene(event.pos())
        items = self.scene.items(scene_pos)

        # Check if clicking on a node
        clicked_node = None
        for item in items:
            if isinstance(item, NodeGraphicsItem):
                clicked_node = item
                break

        # Build context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #2a2a2a;
                color: #ddd;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background: #4a4a4a;
            }
        """)

        # If clicking on empty space OR on a node, show full menu
        # (Blender shows menu everywhere, keeps selection intact)

        # Add Node submenu (only if clicking empty space)
        if not clicked_node:
            add_menu = menu.addMenu("Add")

            groups = {
                "Recurrent": [NodeType.LSTM, NodeType.GRU, NodeType.RNN],
                "Feedforward": [NodeType.LINEAR, NodeType.CONV1D],
                "Activation": [NodeType.TANH, NodeType.RELU, NodeType.SIGMOID],
                "Utility": [NodeType.STATE_CONCAT, NodeType.AFFECT_HEAD],
                "Quantum": [NodeType.QUANTUM_MICROTUBULE, NodeType.IBM_QUANTUM],
                "Assets": [NodeType.CHECKPOINT]
            }

            for group_name, node_types in groups.items():
                group_menu = add_menu.addMenu(f"{group_name}")
                for node_type in node_types:
                    action = QAction(node_type.value, self)
                    action.triggered.connect(lambda checked, nt=node_type, pos=scene_pos: self._add_node_from_menu(nt, pos))
                    group_menu.addAction(action)

            menu.addSeparator()

        # Layout submenu (always available)
        layout_menu = menu.addMenu("Layout")

        auto_arrange = QAction("Auto-Arrange (Topological)", self)
        auto_arrange.triggered.connect(self.auto_arrange_nodes)
        layout_menu.addAction(auto_arrange)

        layout_menu.addSeparator()

        align_h = QAction("Align Horizontally", self)
        align_h.triggered.connect(self.align_selected_horizontally)
        layout_menu.addAction(align_h)

        align_v = QAction("Align Vertically", self)
        align_v.triggered.connect(self.align_selected_vertically)
        layout_menu.addAction(align_v)

        # If clicking on selected node(s), add node-specific actions
        selected_nodes = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if selected_nodes:
            menu.addSeparator()
            delete_action = QAction(f"Delete ({len(selected_nodes)} node{'s' if len(selected_nodes) > 1 else ''})", self)
            delete_action.triggered.connect(self._delete_selected_nodes)
            menu.addAction(delete_action)

        menu.exec(event.globalPos())
        event.accept()  # Don't propagate to base class (prevents deselection)

    def _add_node_from_menu(self, node_type: NodeType, scene_pos: QPointF):
        """Add node from context menu."""
        self.add_node_type = node_type
        self._add_node_at_position(scene_pos)

    def _delete_selected_nodes(self):
        """Delete selected nodes (called from context menu)."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if not selected:
            return

        # Delete from graph
        for node_item in selected:
            self.graph.remove_node(node_item.node.id)

        # Re-render
        self._render_graph()
        self.graph_modified.emit()
        print(f"[Neural Canvas] Deleted {len(selected)} node(s)")

    def _add_node_at_position(self, scene_pos: QPointF):
        """Add a new node at the given position."""
        if not self.add_node_type:
            return

        # Create node from template
        node = create_node_from_type(self.add_node_type)
        node.position = (int(scene_pos.x()), int(scene_pos.y()))

        # Add to graph
        self.graph.add_node(node)

        # Create graphics item
        item = NodeGraphicsItem(node)
        self.scene.addItem(item)
        self.node_items[node.id] = item

        self.graph_modified.emit()

    def focus_selection(self):
        """
        Toggle focus on selected nodes (F key).

        Supports multi-selection: frames all selected nodes as a unit.
        First press: Zooms to selection, saves view state
        Second press: Restores pre-focus view state
        """
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, NodeGraphicsItem)
        ]

        if not selected_nodes:
            print("[Neural Canvas] No nodes selected to focus")
            return

        # Create selection ID (for multi-node focus tracking)
        selection_ids = tuple(sorted(node.node.id for node in selected_nodes))

        # Check if toggling focus on same selection
        if self.is_focused and self.focused_node_id == selection_ids:
            # RESTORE: Pop back to pre-focus view
            if self.pre_focus_transform:
                self.setTransform(self.pre_focus_transform)
                print(f"[Neural Canvas] Restored pre-focus view")
            self.is_focused = False
            self.focused_node_id = None
            self.pre_focus_transform = None
        else:
            # FOCUS: Save current view and zoom to selection
            self.pre_focus_transform = self.transform()
            self.focused_node_id = selection_ids
            self.is_focused = True

            # Frame selected nodes as a unit
            self._frame_nodes(selected_nodes, padding_factor=0.1)

            if len(selected_nodes) == 1:
                print(f"[Neural Canvas] Focused on {selected_nodes[0].node.name} (press F again to restore)")
            else:
                print(f"[Neural Canvas] Focused on {len(selected_nodes)} nodes (press F again to restore)")

    def frame_all_nodes(self):
        """Frame all nodes in view (A key)."""
        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, NodeGraphicsItem)
        ]

        if not all_nodes:
            return

        self._frame_nodes(all_nodes, padding_factor=0.15)
        print(f"[Neural Canvas] Framed all {len(all_nodes)} nodes")

    def _frame_nodes(self, nodes: list, padding_factor: float = 0.1):
        """
        Frame given nodes in view with padding (centers in viewport).

        Args:
            nodes: List of NodeGraphicsItem
            padding_factor: Extra space around nodes (0.1 = 10%)
        """
        if not nodes:
            return

        # Get bounding rect of all nodes
        scene_rect = nodes[0].sceneBoundingRect()
        for node in nodes[1:]:
            scene_rect = scene_rect.united(node.sceneBoundingRect())

        # Add padding
        padding = max(scene_rect.width(), scene_rect.height()) * padding_factor
        scene_rect.adjust(-padding, -padding, padding, padding)

        # Fit in view and center
        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(scene_rect.center())

    def auto_arrange_nodes(self):
        """Auto-arrange nodes using topological layering."""
        print("[Neural Canvas View] Starting auto-arrange...")

        if not self.graph or len(self.graph.nodes) == 0:
            print("[Neural Canvas View] No nodes to arrange")
            return

        try:
            # Get topological order
            node_order = self.graph.topological_sort()

            # Build layer assignment (same layer = same depth in DAG)
            layers = {}  # layer_index -> [node_ids]
            node_layer = {}  # node_id -> layer_index

            # Assign layers based on max dependency depth
            for node_id in node_order:
                # Find max layer of dependencies
                deps = self.graph.get_connections_to_node(node_id)
                if not deps:
                    layer = 0
                else:
                    max_dep_layer = max(
                        node_layer.get(conn.from_node, 0)
                        for conn in deps
                    )
                    layer = max_dep_layer + 1

                node_layer[node_id] = layer

                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node_id)

            # Layout parameters (horizontal flow, left to right)
            layer_spacing = 300  # Horizontal spacing between layers
            node_spacing = 180   # Vertical spacing within layer
            start_x = 100
            start_y = 100

            # Position nodes layer by layer (horizontal flow)
            for layer_idx in sorted(layers.keys()):
                layer_nodes = layers[layer_idx]
                x = start_x + (layer_idx * layer_spacing)  # Horizontal progression

                for node_idx, node_id in enumerate(sorted(layer_nodes)):
                    y = start_y + (node_idx * node_spacing)  # Vertical stacking within layer

                    # Update node position
                    node = self.graph.nodes[node_id]
                    node.position = (x, y)

                    print(f"[Neural Canvas View] {node.name}: ({x}, {y}) [Layer {layer_idx}]")

            # Re-render graph
            self._render_graph()
            self.graph_modified.emit()

            print(f"[Neural Canvas View] Auto-arrange complete! {len(layers)} layers")

            # Frame all nodes to show result
            self.frame_all_nodes()

        except Exception as e:
            print(f"[Neural Canvas View] Auto-arrange error: {e}")
            import traceback
            traceback.print_exc()

    def align_selected_horizontally(self):
        """Align selected nodes to same Y coordinate."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if len(selected) < 2:
            print("[Neural Canvas] Select at least 2 nodes to align")
            return

        # Use average Y position
        avg_y = sum(node.pos().y() for node in selected) / len(selected)

        for node in selected:
            node.setPos(node.pos().x(), avg_y)

        # Force full scene redraw (prevents smearing)
        self.scene.update()
        self.graph_modified.emit()
        print(f"[Neural Canvas] Aligned {len(selected)} nodes horizontally at y={avg_y:.0f}")

    def align_selected_vertically(self):
        """Align selected nodes to same X coordinate."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if len(selected) < 2:
            print("[Neural Canvas] Select at least 2 nodes to align")
            return

        # Use average X position
        avg_x = sum(node.pos().x() for node in selected) / len(selected)

        for node in selected:
            node.setPos(avg_x, node.pos().y())

        # Force full scene redraw (prevents smearing)
        self.scene.update()
        self.graph_modified.emit()
        print(f"[Neural Canvas] Aligned {len(selected)} nodes vertically at x={avg_x:.0f}")
