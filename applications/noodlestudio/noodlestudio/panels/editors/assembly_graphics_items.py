"""Graphics items for the Assembly Editor view.

- Port items implement shared_wire_mixin duck-typing contract
  (get_parent_node_id, get_port_name, get_scene_position, is_output)
- Grid snap reads from the owning view
- Execution animation: processing pulse, complete flash, error flash,
  quantum collapse, wire data packets
"""

from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath
)
from typing import Optional, List, Dict, Tuple, Any

from ...core.facet_system import Facet, FacetPad, PadType


def get_facet_header_color(facet: Facet) -> str:
    """Get coffee shop palette color for facet type header.

    Colors match Neural Canvas wire palette (saturated, easy on eyes).
    """
    if hasattr(facet, 'custom_color') and facet.custom_color:
        return facet.custom_color

    if facet.name in ["INCOMING", "OUTGOING"]:
        return "#5A7A5A"

    facet_type = facet.facet_type

    if "LLMFacet" in facet_type or "LLM" in facet_type:
        return "#6A4A6A"
    elif "ScriptedFacet" in facet_type or "Scripted" in facet_type:
        return "#8A7A4A"
    elif "ContextIntelligence" in facet_type:
        return "#4A5A6A"
    elif "Convergence" in facet_type:
        return "#5A6A7A"
    elif "CharmNetwork" in facet_type:
        return "#6A4A6A"
    elif "NeuralCanvas" in facet_type:
        return "#7B1FA2"
    elif "Transformer" in facet_type:
        return "#6A4A8A"
    else:
        return "#5A5A5A"


# ============================================================================
# FacetPortItem
# ============================================================================

class FacetPortItem(QGraphicsEllipseItem):
    """Connection point on a facet node.

    Implements the duck-typing contract for SharedWireMixin:
    - get_parent_node_id() -> str
    - get_port_name() -> str
    - get_scene_position() -> QPointF
    - is_output (property) -> bool
    """

    PAD_RADIUS = 5

    def __init__(self, pad: FacetPad, facet_node: 'FacetNodeItem', parent=None):
        super().__init__(
            -self.PAD_RADIUS, -self.PAD_RADIUS,
            self.PAD_RADIUS * 2, self.PAD_RADIUS * 2, parent
        )
        self.pad = pad
        self.facet_node = facet_node

        # Color: output pads match parent header, input pads start neutral
        if pad.pad_type == PadType.OUTPUT:
            pad_color = get_facet_header_color(facet_node.facet)
        else:
            pad_color = "#666666"

        self.default_brush = QBrush(QColor(pad_color))
        self.hover_brush = QBrush(QColor(pad_color).lighter(130))
        self.setBrush(self.default_brush)
        self.setPen(QPen(QColor("#333"), 1.5))
        self.setAcceptHoverEvents(True)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(10)

        self.connections: List['FacetConnectionItem'] = []

    # -- SharedWireMixin duck-typing contract --

    @property
    def is_output(self) -> bool:
        return self.pad.pad_type == PadType.OUTPUT

    def get_parent_node_id(self) -> str:
        return self.facet_node.facet.id

    def get_port_name(self) -> str:
        return self.pad.name

    def get_scene_position(self) -> QPointF:
        return self.scenePos()

    # -- Visual feedback --

    def hoverEnterEvent(self, event):
        self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        self.setBrush(self.default_brush)
        super().hoverLeaveEvent(event)

    def update_color_from_connection(self):
        """Update input pad color to match incoming wire source."""
        if self.pad.pad_type == PadType.OUTPUT:
            return

        if self.connections:
            wire = self.connections[0]
            source_facet = wire.from_port.facet_node.facet
            pad_color = get_facet_header_color(source_facet)
        else:
            pad_color = "#666666"

        self.default_brush = QBrush(QColor(pad_color))
        self.hover_brush = QBrush(QColor(pad_color).lighter(130))
        self.setBrush(self.default_brush)

    def mousePressEvent(self, event):
        """Start wire drawing via the owning view."""
        if event.button() == Qt.MouseButton.LeftButton:
            scene = self.scene()
            if scene and scene.views():
                view = scene.views()[0]
                if hasattr(view, 'start_wire_drawing'):
                    view.start_wire_drawing(self)
                    event.accept()
                    return
        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        if self.facet_node:
            self.facet_node.contextMenuEvent(event)
        else:
            event.accept()


# ============================================================================
# FacetNodeItem
# ============================================================================

class FacetNodeItem(QGraphicsRectItem):
    """Visual representation of a facet node in the assembly editor."""

    NODE_WIDTH = 200
    NODE_HEIGHT = 120
    NODE_HEIGHT_COMPACT = 35
    PAD_SPACING = 25

    def __init__(self, facet: Facet, editor_view=None, parent=None):
        is_special = facet.name in ["INCOMING", "OUTGOING"]

        header_height = 24
        port_start_y = header_height + 15
        port_spacing = 20
        max_pads = max(len(facet.input_pads), len(facet.output_pads), 1)
        min_height = self.NODE_HEIGHT_COMPACT if is_special else self.NODE_HEIGHT
        calculated_height = port_start_y + (max_pads * port_spacing) + 15
        initial_height = max(min_height, calculated_height)

        super().__init__(0, 0, self.NODE_WIDTH, initial_height, parent)
        self.facet = facet
        self.is_special_node = is_special
        self._editor_view = editor_view  # Weak ref to AssemblyEditorView for grid snap

        self.setBrush(QBrush(QColor("#3a3a3a")))
        self.setPen(QPen(QColor("#555555"), 2))

        # Save defaults for animation reset
        self.base_brush = QBrush(QColor("#3a3a3a"))
        self.default_pen = QPen(QColor("#555555"), 2)
        self.selected_pen = QPen(QColor("#FFFFFF"), 3)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        # Status indicator
        self.status_indicator = QGraphicsEllipseItem(self)
        self.status_indicator.setRect(self.NODE_WIDTH - 20, 10, 10, 10)
        self.status_indicator.setBrush(QBrush(QColor("#666666")))
        self.status_indicator.setPen(QPen(QColor("#888888"), 1))

        # Lock icon
        lock_text = "[L]" if facet.locked else ""
        self.lock_label = QGraphicsTextItem(lock_text, self)
        self.lock_label.setPos(self.NODE_WIDTH - 30, 5)
        self.lock_label.setDefaultTextColor(
            QColor("#CCAA00" if facet.locked else "#888888")
        )
        self.lock_label.setFont(QFont("Courier", 10))
        self.lock_label.setZValue(15)

        # Pads
        self.input_pads: Dict[str, FacetPortItem] = {}
        self.output_pads: Dict[str, FacetPortItem] = {}

        # Execution state and animation
        self.execution_state = "idle"
        self.animation_timer: Optional[QTimer] = None
        self.pulse_phase: float = 0.0
        self.error_flash_count: int = 0
        self.collapse_flash_alpha: float = 0.0

        # Cycle tracking for multi-cycle overlap display
        self.active_cycles: List[Tuple[str, QColor, Optional[Dict]]] = []
        self.cycle_data: Dict[str, Dict[str, Any]] = {}
        self.last_inputs: Optional[Dict] = None
        self.last_outputs: Optional[Dict] = None

        # Drag tracking for undo
        self.drag_start_pos: Optional[Tuple[float, float]] = None

        self._create_pads()
        self.setPos(facet.position.get('x', 0), facet.position.get('y', 0))

    def get_node_id(self) -> str:
        return self.facet.id

    def _create_pads(self):
        """Create port items on left (inputs) and right (outputs) edges."""
        port_start_y = 39
        port_spacing = 20

        for i, pad in enumerate(self.facet.input_pads):
            port = FacetPortItem(pad, self, parent=self)
            y = port_start_y + (i * port_spacing)
            port.setPos(0, y)
            self.input_pads[pad.name] = port

            # Label
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(15, y - 8)
            label.setFont(QFont("Arial", 8))
            label.setDefaultTextColor(QColor("#AAAAAA"))

        for i, pad in enumerate(self.facet.output_pads):
            port = FacetPortItem(pad, self, parent=self)
            y = port_start_y + (i * port_spacing)
            port.setPos(self.NODE_WIDTH, y)
            self.output_pads[pad.name] = port

            # Label (right-aligned)
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(self.NODE_WIDTH - 70, y - 8)
            label.setFont(QFont("Arial", 8))
            label.setDefaultTextColor(QColor("#AAAAAA"))

    def boundingRect(self) -> QRectF:
        """Expanded rect for selection highlight rendering."""
        rect = super().boundingRect()
        return rect.adjusted(-6, -6, 6, 6)

    def paint(self, painter: QPainter, option, widget=None):
        """Render Blender-style node: gray body + colored header + cycle badges."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Gray body with rounded corners
        body_path = QPainterPath()
        body_path.addRoundedRect(rect, 4, 4)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.drawPath(body_path)

        # Colored header bar (sharp top, rounded only at top corners)
        header_color = get_facet_header_color(self.facet)
        header_rect = QRectF(rect.x(), rect.y(), rect.width(), 24)
        painter.setBrush(QBrush(QColor(header_color)))
        painter.drawRect(header_rect)

        # Selection highlight
        if self.isSelected():
            highlight_rect = rect.adjusted(-3, -3, 3, 3)
            highlight_path = QPainterPath()
            highlight_path.addRoundedRect(highlight_rect, 3, 3)
            painter.setPen(QPen(QColor("#FFFFFF"), 3))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(highlight_path)

        # Node name in header
        painter.setPen(QPen(QColor("#e8e8e0")))
        painter.setFont(QFont("Arial", 9, QFont.Weight.Bold))
        name_rect = QRectF(rect.x() + 8, rect.y() + 4, rect.width() - 60, 18)
        painter.drawText(name_rect, Qt.AlignmentFlag.AlignLeft, self.facet.name)

        # Facet type (right side of header)
        painter.setFont(QFont("Arial", 7))
        type_rect = QRectF(rect.x() + 80, rect.y() + 6, rect.width() - 90, 14)
        painter.drawText(type_rect, Qt.AlignmentFlag.AlignRight, self.facet.facet_type)

        # Cycle ID badges (bottom-right, stacked vertically)
        if self.active_cycles:
            badge_height = 14
            badge_spacing = 2
            cycles_copy = list(self.active_cycles)

            for i, cycle_data in enumerate(cycles_copy):
                if not cycle_data or len(cycle_data) < 2:
                    continue

                cycle_id, cycle_color = cycle_data[0], cycle_data[1]
                if not cycle_id:
                    continue

                badge_text = str(cycle_id)[:8]

                # Stack upward from bottom
                stack_offset = (
                    (len(cycles_copy) - 1 - i) * (badge_height + badge_spacing)
                )
                badge_y = rect.height() - 16 - stack_offset

                badge_rect = QRectF(
                    rect.width() - 60, badge_y, 56, badge_height
                )
                painter.setBrush(QBrush(cycle_color or QColor("#00BFFF")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(badge_rect, 3, 3)

                painter.setPen(QColor("#000000"))
                painter.setFont(QFont("Arial", 7))
                painter.drawText(
                    badge_rect, Qt.AlignmentFlag.AlignCenter, badge_text
                )

    def itemChange(self, change, value):
        """Handle position changes for grid snap and wire updates."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Grid snap via editor view
            if self._editor_view and hasattr(self._editor_view, '_snap_to_grid'):
                if self._editor_view._snap_to_grid:
                    gs = self._editor_view._grid_size
                    x = round(value.x() / gs) * gs
                    y = round(value.y() / gs) * gs
                    return QPointF(x, y)

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update connected wires
            for port in list(self.input_pads.values()) + list(self.output_pads.values()):
                for wire in port.connections:
                    wire.update_path()
            scene = self.scene()
            if scene:
                scene.update()

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.prepareGeometryChange()
            self.update()

        return super().itemChange(change, value)

    # ================================================================
    # Execution state machine
    # ================================================================

    def set_execution_state(self, state: str):
        """Set execution visualization state with animation.

        States:
            idle: Default appearance, no animation.
            processing: Pulsing border (60ms tick, square wave).
            complete: Brief brightness flash (200ms), auto-return to idle.
            error: Flashing red border (100ms, 5 cycles), auto-return to idle.
            quantum_collapse: Purple flash fading over 200ms.
        """
        if self.execution_state == state:
            return

        self.execution_state = state

        # Stop any running animation timer
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        if state == "idle":
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen)
            self.update()

        elif state == "processing":
            self.setBrush(self.base_brush)
            self.pulse_phase = 0.0
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._pulse_border)
            self.animation_timer.start(60)

        elif state == "complete":
            self.setBrush(QBrush(QColor("#5A5A5A")))
            self.update()
            QTimer.singleShot(200, lambda: self.set_execution_state("idle"))

        elif state == "error":
            self.error_flash_count = 0
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._flash_error_border)
            self.animation_timer.start(100)

        elif state == "quantum_collapse":
            self.collapse_flash_alpha = 1.0
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._fade_quantum_flash)
            self.animation_timer.start(20)
            QTimer.singleShot(200, lambda: self.set_execution_state("idle"))

    def _pulse_border(self):
        """Geometric border pulse (square wave, not sine)."""
        self.pulse_phase += 0.2
        if self.pulse_phase >= 1.0:
            self.pulse_phase = 0.0

        # Square wave brightness
        brightness = 255 if self.pulse_phase >= 0.5 else 170

        pen_width = 3 if self.isSelected() else 2
        self.setPen(QPen(QColor(brightness, brightness, brightness), pen_width))
        self.update()

    def _flash_error_border(self):
        """Flashing red border -- 5 cycles (10 flashes) then return to idle."""
        self.error_flash_count += 1

        if self.error_flash_count % 2 == 0:
            self.setBrush(QBrush(QColor("#FF4444")))
            self.setPen(QPen(QColor("#FF0000"), 3))
        else:
            self.setBrush(QBrush(QColor("#8B0000")))
            self.setPen(QPen(QColor("#660000"), 2))

        self.update()

        if self.error_flash_count >= 10:
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None
            self.error_flash_count = 0
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen)
            self.execution_state = "idle"
            self.update()

    def _fade_quantum_flash(self):
        """Fade quantum collapse flash (linear over 200ms)."""
        self.collapse_flash_alpha -= 0.1

        if self.collapse_flash_alpha <= 0.0:
            self.collapse_flash_alpha = 0.0
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen)
        else:
            a = self.collapse_flash_alpha
            r = int(147 * a + 58 * (1 - a))
            g = int(112 * a + 58 * (1 - a))
            b = int(219 * a + 58 * (1 - a))
            self.setBrush(QBrush(QColor(r, g, b)))

            border = int(255 * a)
            self.setPen(QPen(QColor(border, border, 255), 3))

        self.update()

    def stop_animation(self):
        """Stop any running animation timer (cleanup before removal)."""
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None
        self.execution_state = "idle"

    def set_status(self, status: str):
        """Update status indicator color."""
        color_map = {
            'inactive': '#666666', 'ready': '#999999',
            'processing': '#CCCCCC', 'waiting': '#555555',
            'cached': '#AAAAAA'
        }
        color = color_map.get(status, '#666666')
        self.status_indicator.setBrush(QBrush(QColor(color)))

    def mousePressEvent(self, event):
        """Track drag start position for undo."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_pos = (self.pos().x(), self.pos().y())
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Notify view of drag completion for undo."""
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton and self.drag_start_pos:
            current = (self.pos().x(), self.pos().y())
            dx = abs(current[0] - self.drag_start_pos[0])
            dy = abs(current[1] - self.drag_start_pos[1])
            if dx > 1 or dy > 1:
                # Moved significantly -- view will handle undo
                if self._editor_view and hasattr(self._editor_view, '_on_node_drag_finished'):
                    self._editor_view._on_node_drag_finished()
            self.drag_start_pos = None

    def mouseDoubleClickEvent(self, event):
        """Double-click on NeuralCanvasFacet opens depth navigation."""
        if (self.facet.facet_type == "NeuralCanvasFacet" and
                getattr(self.facet, 'nncanvas_path', None)):
            if self._editor_view and hasattr(self._editor_view, 'containerDoubleClicked'):
                self._editor_view.containerDoubleClicked.emit(
                    self.facet.facet_type,
                    self.facet.nncanvas_path,
                    self.facet.name
                )
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        """Delegate context menu to the view."""
        event.accept()
        scene = self.scene()
        if scene and scene.views():
            view = scene.views()[0]
            if hasattr(view, '_show_context_menu'):
                view_pos = view.mapFromScene(event.scenePos())
                view._show_context_menu(view_pos)


# ============================================================================
# FacetConnectionItem
# ============================================================================

class FacetConnectionItem(QGraphicsItem):
    """Bezier connection wire between two ports."""

    def __init__(self, from_port: FacetPortItem, to_port: FacetPortItem, parent=None):
        super().__init__(parent)
        self.from_port = from_port
        self.to_port = to_port

        # Register with ports
        from_port.connections.append(self)
        to_port.connections.append(self)
        to_port.update_color_from_connection()

        self.setZValue(-1)

        # Packet animation
        self.packet_progress: float = 0.0
        self.packet_animating: bool = False
        self.packet_timer: Optional[QTimer] = None

    def boundingRect(self) -> QRectF:
        start = self.from_port.get_scene_position()
        end = self.to_port.get_scene_position()
        rect = QRectF(start, end).normalized()
        return rect.adjusted(-50, -50, 50, 50)

    def paint(self, painter: QPainter, option, widget=None):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        start = self.from_port.get_scene_position()
        end = self.to_port.get_scene_position()

        # Bezier control points
        distance = abs(end.x() - start.x())
        handle = min(distance * 0.5, 100)
        c1 = QPointF(start.x() + handle, start.y())
        c2 = QPointF(end.x() - handle, end.y())

        # Wire color from source node
        wire_color = QColor(get_facet_header_color(self.from_port.facet_node.facet))
        if self.packet_animating:
            wire_color = wire_color.lighter(150)

        path = QPainterPath()
        path.moveTo(start)
        path.cubicTo(c1, c2, end)

        painter.setPen(QPen(wire_color, 2.5))
        painter.drawPath(path)

        # Arrowhead at target
        t = 0.95
        arrow_pos = path.pointAtPercent(t)
        tangent_pos = path.pointAtPercent(t - 0.02)
        dx = arrow_pos.x() - tangent_pos.x()
        dy = arrow_pos.y() - tangent_pos.y()
        length = (dx * dx + dy * dy) ** 0.5
        if length > 0:
            dx /= length
            dy /= length
            size = 6
            p1 = QPointF(
                arrow_pos.x() - size * dx - size * 0.5 * dy,
                arrow_pos.y() - size * dy + size * 0.5 * dx
            )
            p2 = QPointF(
                arrow_pos.x() - size * dx + size * 0.5 * dy,
                arrow_pos.y() - size * dy - size * 0.5 * dx
            )
            arrow_path = QPainterPath()
            arrow_path.moveTo(arrow_pos)
            arrow_path.lineTo(p1)
            arrow_path.lineTo(p2)
            arrow_path.closeSubpath()
            painter.setBrush(QBrush(wire_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawPath(arrow_path)

        # Data packet (geometric white square, Kraftwerk style)
        if self.packet_animating and 0.0 <= self.packet_progress <= 1.0:
            packet_pos = path.pointAtPercent(self.packet_progress)
            packet_size = 12
            painter.setBrush(QBrush(QColor("#FFFFFF")))
            painter.setPen(QPen(QColor("#CCAA00"), 2))
            painter.drawRect(QRectF(
                packet_pos.x() - packet_size / 2,
                packet_pos.y() - packet_size / 2,
                packet_size,
                packet_size
            ))

    def update_path(self):
        """Force repaint after port positions change."""
        self.prepareGeometryChange()
        self.update()

    def animate_data_flow(self):
        """Animate data packet flowing through connection wire.

        Kraftwerk style: linear motion, geometric square packet, 300ms duration.
        """
        if self.packet_animating:
            return  # Already animating

        self.packet_animating = True
        self.packet_progress = 0.0

        self.packet_timer = QTimer()
        self.packet_timer.timeout.connect(self._advance_packet)
        self.packet_timer.start(16)  # ~60fps

    def _advance_packet(self):
        """Advance packet along Bezier curve (linear motion)."""
        self.packet_progress += 0.05  # 5% per frame = ~300ms total

        if self.packet_progress >= 1.0:
            self.packet_progress = 1.0
            self.packet_animating = False
            if self.packet_timer:
                self.packet_timer.stop()
                self.packet_timer = None
            QTimer.singleShot(100, lambda: self.update())

        self.prepareGeometryChange()
        self.update()

    def stop_animation(self):
        """Stop any running packet animation (cleanup)."""
        if self.packet_timer:
            self.packet_timer.stop()
            self.packet_timer = None
        self.packet_animating = False
        self.packet_progress = 0.0
