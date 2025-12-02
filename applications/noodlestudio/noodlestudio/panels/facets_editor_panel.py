"""
Facets Editor Panel - Node-based cognitive architecture editor

Visual node graph editor for designing facet assemblies.
Unity-style node editor with drag-and-drop, connection wires, and right-click menus.

Author: Commander Spock + Cadet Caity
Date: November 28, 2025
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem,
    QGraphicsLineItem, QPushButton, QLabel, QMenu, QMessageBox, QFileDialog,
    QGraphicsProxyWidget, QTextEdit
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF, QTimer, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath, QCursor, QKeySequence, QShortcut
)
from typing import Optional, List, Dict, Tuple
import sys
import os
import requests
import json
import asyncio
from asyncio import QueueEmpty
try:
    import websockets
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False

# Import facet system
from ..core.facet_system import (
    Facet, FacetAssembly, FacetConnection, FacetPad, PadType
)
from .floating_text_editor import FloatingTextEditor


class ClickableTextItem(QGraphicsTextItem):
    """Clickable text item (for pencil icons)."""

    def __init__(self, text: str, parent_node, field_data: dict, callback):
        super().__init__(text, parent_node)
        self.parent_node = parent_node
        self.field_data = field_data
        self.callback = callback

        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

    def hoverEnterEvent(self, event):
        """Highlight on hover."""
        self.setDefaultTextColor(QColor("#FFFFFF"))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Restore color."""
        self.setDefaultTextColor(QColor("#888888"))
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event):
        """Handle click - open field editor."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.callback(self.field_data)
            event.accept()
        else:
            super().mousePressEvent(event)


class FacetPadGraphics(QGraphicsEllipseItem):
    """Visual representation of a facet pad (connection point)."""

    PAD_RADIUS = 8

    def __init__(self, pad: FacetPad, facet_node: 'FacetNodeGraphics', parent=None):
        super().__init__(-self.PAD_RADIUS, -self.PAD_RADIUS,
                         self.PAD_RADIUS * 2, self.PAD_RADIUS * 2, parent)
        self.pad = pad
        self.facet_node = facet_node

        # Monochromatic styling
        self.default_brush = QBrush(QColor("#888888"))  # Gray for all pads
        self.hover_brush = QBrush(QColor("#FFFFFF"))    # White on hover
        self.setBrush(self.default_brush)
        self.setPen(QPen(QColor("#AAAAAA"), 2))
        self.setAcceptHoverEvents(True)

        # Make pad independently clickable (don't propagate to parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        # Connection tracking
        self.connections: List['ConnectionWire'] = []

    def hoverEnterEvent(self, event):
        """Highlight pad on hover."""
        self.setBrush(self.hover_brush)
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event):
        """Restore pad color on hover exit."""
        self.setBrush(self.default_brush)
        super().hoverLeaveEvent(event)

    def get_scene_position(self) -> QPointF:
        """Get pad position in scene coordinates."""
        return self.scenePos()

    def mousePressEvent(self, event):
        """Start wire drawing on pad click."""
        if event.button() == Qt.MouseButton.LeftButton:
            # Get parent editor panel to handle wire drawing
            scene = self.scene()
            if scene and hasattr(scene, 'views') and scene.views():
                view = scene.views()[0]
                parent = view.parent()
                if isinstance(parent, FacetsEditorPanel):
                    parent.start_wire_drawing(self)
                    event.accept()
                    return
        super().mousePressEvent(event)


class FacetNodeGraphics(QGraphicsRectItem):
    """Visual representation of a facet node."""

    NODE_WIDTH = 200
    NODE_HEIGHT = 120
    NODE_HEIGHT_COMPACT = 35  # For INCOMING/OUTGOING special nodes (tight vertical)
    NODE_HEIGHT_EXPANDED = 400  # Height when expanded for editing
    PAD_SPACING = 25

    def __init__(self, facet: Facet, editor_panel=None, parent=None):
        # Check if this is a special node (INCOMING/OUTGOING) for compact size
        is_special = facet.name in ["INCOMING", "OUTGOING"]
        initial_height = self.NODE_HEIGHT_COMPACT if is_special else self.NODE_HEIGHT

        super().__init__(0, 0, self.NODE_WIDTH, initial_height, parent)
        self.facet = facet
        self.is_special_node = is_special
        self.editor_panel = editor_panel  # Reference to FacetsEditorPanel for pause state

        # Monochromatic styling (scriptable color overrides possible)
        # Check if facet has custom color override
        custom_color = getattr(facet, 'custom_color', None)
        if custom_color:
            # Scriptable color override
            self.setBrush(QBrush(QColor(custom_color)))
        else:
            # Default monochromatic based on type
            if facet.name == "INCOMING":
                self.setBrush(QBrush(QColor("#2A2A2A")))  # Darker - distinct from facets
            elif facet.name == "OUTGOING":
                self.setBrush(QBrush(QColor("#2A2A2A")))  # Darker - distinct from facets
            elif "Convergence" in facet.facet_type:
                self.setBrush(QBrush(QColor("#4A4A4A")))  # Darker gray
            else:
                self.setBrush(QBrush(QColor("#3E3E3E")))  # Default dark gray

        # Default border (will change on selection)
        self.default_pen = QPen(QColor("#666666"), 2)
        self.selected_pen = QPen(QColor("#FFFFFF"), 3)  # White border when selected
        self.setPen(self.default_pen)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Accept both Shift and Cmd for multi-selection
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        # Title text (brighter/bolder for special nodes)
        self.title = QGraphicsTextItem(facet.name, self)
        font = QFont("Arial", 11, QFont.Weight.Bold)

        if is_special:
            # Special nodes: larger, bold, center-aligned, symmetric padding
            font = QFont("Arial", 14, QFont.Weight.Bold)
            self.title.setFont(font)
            self.title.setDefaultTextColor(QColor("#FFFFFF"))

            # Calculate center position for text
            text_width = self.title.boundingRect().width()
            text_height = self.title.boundingRect().height()
            x_center = (self.NODE_WIDTH - text_width) / 2
            y_center = (self.NODE_HEIGHT_COMPACT - text_height) / 2
            self.title.setPos(x_center, y_center)
        else:
            # Regular nodes: left-aligned
            self.title.setFont(font)
            self.title.setDefaultTextColor(QColor("#FFFFFF"))
            self.title.setPos(10, 5)

        # Type label (hidden for special nodes)
        if not is_special:
            self.type_label = QGraphicsTextItem(facet.facet_type, self)
            self.type_label.setPos(10, 25)
            self.type_label.setDefaultTextColor(QColor("#AAAAAA"))
            type_font = QFont("Arial", 9)
            self.type_label.setFont(type_font)
        else:
            self.type_label = None  # No type label for special nodes

        # Field display widgets (created when zoomed in enough)
        self.field_widgets: List[QGraphicsItem] = []
        self.min_zoom_for_fields = 1.5  # Show fields when zoom > 1.5x

        # Status indicator (always visible)
        self.status_indicator = QGraphicsEllipseItem(self)
        self.status_indicator.setRect(self.NODE_WIDTH - 20, 10, 10, 10)
        self.status_indicator.setBrush(QBrush(QColor("#666666")))  # Default gray (inactive)
        self.status_indicator.setPen(QPen(QColor("#888888"), 1))
        self.status_color = "#666666"

        # Lock icon (top-right corner, clickable)
        lock_icon = "[L]" if facet.locked else ""
        self.lock_icon = ClickableTextItem(lock_icon, self, {}, self.toggle_lock)
        self.lock_icon.setPos(self.NODE_WIDTH - 30, 5)
        self.lock_icon.setDefaultTextColor(QColor("#CCAA00" if facet.locked else "#888888"))
        self.lock_icon.setFont(QFont("Courier", 10))
        self.lock_icon.setToolTip("Click to lock/unlock facet")
        self.lock_icon.setZValue(15)  # Above other elements

        # Create pad graphics
        self.input_pads: Dict[str, FacetPadGraphics] = {}
        self.output_pads: Dict[str, FacetPadGraphics] = {}

        # Animation state (Kraftwerk style - industrial precision)
        self.execution_state = "idle"  # idle, processing, complete, error, quantum_collapse
        self.animation_timer: Optional[QTimer] = None
        self.pulse_phase = 0.0  # 0.0 to 1.0 for border pulse
        self.base_brush = self.brush()  # Store original brush for restoration
        self.collapse_flash_alpha = 0.0  # For quantum collapse flash effect

        self._create_pads()

        # Set initial position from facet metadata
        self.setPos(facet.position['x'], facet.position['y'])

    def _create_pads(self):
        """Create visual representations of input/output pads (vertical layout)."""
        # Determine height based on node type
        node_height = self.NODE_HEIGHT_COMPACT if self.is_special_node else self.NODE_HEIGHT

        # Input pads on top
        num_inputs = len(self.facet.input_pads)
        if num_inputs > 0:
            spacing = self.NODE_WIDTH / (num_inputs + 1)
            for i, pad in enumerate(self.facet.input_pads):
                pad_graphics = FacetPadGraphics(pad, self, self)
                x_pos = spacing * (i + 1)
                pad_graphics.setPos(x_pos, 0)
                self.input_pads[pad.name] = pad_graphics

                # Pad label (above pad)
                label = QGraphicsTextItem(pad.name, self)
                label.setPos(x_pos - 20, -20)
                label.setDefaultTextColor(QColor("#AAAAAA"))
                label.setFont(QFont("Arial", 8))

        # Output pads on bottom
        num_outputs = len(self.facet.output_pads)
        if num_outputs > 0:
            spacing = self.NODE_WIDTH / (num_outputs + 1)
            for i, pad in enumerate(self.facet.output_pads):
                pad_graphics = FacetPadGraphics(pad, self, self)
                x_pos = spacing * (i + 1)
                pad_graphics.setPos(x_pos, node_height)
                self.output_pads[pad.name] = pad_graphics

                # Pad label (below pad)
                label = QGraphicsTextItem(pad.name, self)
                label.setPos(x_pos - 20, node_height + 5)
                label.setDefaultTextColor(QColor("#AAAAAA"))
                label.setFont(QFont("Arial", 8))

    def itemChange(self, change, value):
        """Handle item changes (e.g., position updates, selection)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Apply grid snapping if enabled
            new_pos = value
            if self.scene() and hasattr(self.scene().views()[0].parent(), 'snap_to_grid'):
                editor = self.scene().views()[0].parent()
                if editor.snap_to_grid:
                    grid = editor.grid_size
                    snapped_x = round(new_pos.x() / grid) * grid
                    snapped_y = round(new_pos.y() / grid) * grid
                    return QPointF(snapped_x, snapped_y)
            return new_pos

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged:
            # Update facet metadata
            pos = self.pos()
            self.facet.position['x'] = pos.x()
            self.facet.position['y'] = pos.y()

            # Update connected wires
            for pad_dict in [self.input_pads, self.output_pads]:
                for pad_graphics in pad_dict.values():
                    for wire in pad_graphics.connections:
                        wire.update_path()

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # Update border on selection
            if self.isSelected():
                self.setPen(self.selected_pen)
            else:
                self.setPen(self.default_pen)

        return super().itemChange(change, value)

    # ========== KRAFTWERK ANIMATION SYSTEM ==========
    # Industrial precision. No bounce. Function over flourish.

    def set_execution_state(self, state: str):
        """
        Set execution state and trigger appropriate animation.

        States: idle, processing, complete, error

        Kraftwerk style:
        - Sharp transitions (80ms linear)
        - Monochromatic palette
        - Border pulse for processing (geometric, not organic)
        - Subtle color shifts only
        """
        if self.execution_state == state:
            return  # No redundant animations

        self.execution_state = state

        # Stop any existing animation
        if self.animation_timer:
            self.animation_timer.stop()
            self.animation_timer = None

        if state == "idle":
            # Restore to neutral
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen if not self.isSelected() else self.selected_pen)

        elif state == "processing":
            # Industrial yellow - warning light
            self.setBrush(QBrush(QColor("#CCAA00")))
            # Start border pulse (60ms tick, geometric)
            self.pulse_phase = 0.0
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._pulse_border)
            self.animation_timer.start(60)  # 16fps - industrial machinery cadence

        elif state == "complete":
            # Slight brightness increase (satisfaction feedback)
            self.setBrush(QBrush(QColor("#5A5A5A")))
            # Hold for 200ms, then fade to idle
            QTimer.singleShot(200, lambda: self.set_execution_state("idle"))

        elif state == "error":
            # Dark red (emergency indicator)
            self.setBrush(QBrush(QColor("#8B0000")))
            # Hold for 500ms, then return to idle
            QTimer.singleShot(500, lambda: self.set_execution_state("idle"))

        elif state == "quantum_collapse":
            # QUANTUM COLLAPSE - Purple/blue flash
            # Orchestrated objective reduction event
            self.collapse_flash_alpha = 1.0
            self._quantum_flash()
            # Return to idle after flash completes
            QTimer.singleShot(200, lambda: self.set_execution_state("idle"))

    def _pulse_border(self):
        """
        Geometric border pulse for processing state.

        No organic curves. Square wave, not sine.
        """
        self.pulse_phase += 0.2  # Fast square wave
        if self.pulse_phase >= 1.0:
            self.pulse_phase = 0.0

        # Square wave brightness (sharp transitions)
        if self.pulse_phase < 0.5:
            brightness = 170  # Dim
        else:
            brightness = 255  # Bright

        # Update pen
        if self.isSelected():
            pen = QPen(QColor(brightness, brightness, brightness), 3)
        else:
            pen = QPen(QColor(brightness, brightness, brightness), 2)

        self.setPen(pen)

    def _quantum_flash(self):
        """
        Quantum collapse flash animation.

        Sharp purple/blue flash that fades out linearly over 200ms.
        Represents orchestrated objective reduction (Penrose-Hameroff).
        """
        # Start with bright purple/blue
        self.collapse_flash_alpha = 1.0
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._fade_quantum_flash)
        self.animation_timer.start(20)  # 50fps fade

    def _fade_quantum_flash(self):
        """Fade quantum collapse flash."""
        self.collapse_flash_alpha -= 0.1  # Linear fade over 200ms

        if self.collapse_flash_alpha <= 0.0:
            # Flash complete
            self.collapse_flash_alpha = 0.0
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen if not self.isSelected() else self.selected_pen)
        else:
            # Compute flash color (purple/blue with decreasing alpha)
            # Base: #9370DB (medium purple) fading to background
            r = int(147 * self.collapse_flash_alpha + 42 * (1 - self.collapse_flash_alpha))
            g = int(112 * self.collapse_flash_alpha + 42 * (1 - self.collapse_flash_alpha))
            b = int(219 * self.collapse_flash_alpha + 42 * (1 - self.collapse_flash_alpha))
            flash_color = QColor(r, g, b)
            self.setBrush(QBrush(flash_color))

            # Bright border during flash
            border_intensity = int(255 * self.collapse_flash_alpha)
            self.setPen(QPen(QColor(border_intensity, border_intensity, 255), 3))

    def mousePressEvent(self, event):
        """Handle mouse press - support Shift for additive selection."""
        # On macOS, Qt uses Cmd for multi-select by default
        # Make Shift also work for additive selection
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift pressed - toggle selection
            self.setSelected(not self.isSelected())
            event.accept()
        else:
            # Normal click
            super().mousePressEvent(event)

    def show_fields(self, force: bool = False):
        """
        Show field displays (triggered by F key focus).

        Args:
            force: If True, show regardless of zoom level
        """
        # Special nodes (INCOMING/OUTGOING) have no editable fields
        if self.is_special_node:
            return

        # Clear existing field widgets
        try:
            for widget in self.field_widgets:
                if widget and widget.scene():
                    self.scene().removeItem(widget)
        except:
            pass
        self.field_widgets.clear()

        # Get fields from facet (pass cognition pause state)
        cognition_paused = self.editor_panel.cognition_paused if self.editor_panel else False
        fields = self.facet.get_editable_fields(cognition_paused=cognition_paused)

        # Calculate required height for all fields
        field_height = 30  # Height per field
        total_field_height = len(fields) * field_height
        required_height = 50 + total_field_height + 30  # Header + fields + pad space

        # Get current node height
        current_height = self.NODE_HEIGHT_COMPACT if self.is_special_node else self.NODE_HEIGHT

        # Expand node if needed to fit fields
        if required_height > current_height:
            self.setRect(0, 0, self.NODE_WIDTH, required_height)
            # Reposition output pads to new bottom
            num_outputs = len(self.output_pads)
            if num_outputs > 0:
                spacing = self.NODE_WIDTH / (num_outputs + 1)
                for i, (name, pad) in enumerate(self.output_pads.items()):
                    x_pos = spacing * (i + 1)
                    pad.setPos(x_pos, required_height)

        y_offset = 45  # Start below title/type
        for field in fields:
            # Single line: "FIELD NAME: preview text..." ✎
            field_line = QGraphicsTextItem(self)

            # Build display text
            label = field['name'].upper() + ":"
            preview_italic = f"<i>{field['preview']}</i>"
            full_text = f"<span style='color: #888888; font-weight: bold;'>{label}</span> <span style='color: #AAAAAA;'>{preview_italic}</span>"

            field_line.setHtml(full_text)
            field_line.setPos(10, y_offset)
            field_line.setFont(QFont("Arial", 8))
            field_line.setTextWidth(self.NODE_WIDTH - 40)  # Leave room for pencil
            field_line.setZValue(10)  # Ensure fields render above node background
            field_line.setVisible(True)
            self.field_widgets.append(field_line)

            # Pencil button (clickable to open editor)
            pencil = ClickableTextItem("✎", self, field, self.open_field_editor)
            pencil.setPos(self.NODE_WIDTH - 20, y_offset)
            pencil.setDefaultTextColor(QColor("#666666"))
            pencil.setFont(QFont("Arial", 10))
            pencil.setZValue(10)  # Ensure pencil renders above node background
            pencil.setVisible(True)
            self.field_widgets.append(pencil)

            y_offset += field_height

    def hide_fields(self):
        """Hide field displays and restore normal node size."""
        # Safety check - don't crash if widgets are invalid
        try:
            for widget in self.field_widgets:
                if widget and widget.scene():
                    self.scene().removeItem(widget)
            self.field_widgets.clear()

            # Restore normal size
            self.setRect(0, 0, self.NODE_WIDTH, self.NODE_HEIGHT)

            # Restore output pad positions
            num_outputs = len(self.output_pads)
            if num_outputs > 0:
                spacing = self.NODE_WIDTH / (num_outputs + 1)
                for i, (name, pad) in enumerate(self.output_pads.items()):
                    x_pos = spacing * (i + 1)
                    pad.setPos(x_pos, self.NODE_HEIGHT)
        except Exception as e:
            print(f"[Node] Error hiding fields: {e}")
            self.field_widgets.clear()

    def _reposition_pads_expanded(self):
        """Reposition pads for expanded state (at new bottom)."""
        num_outputs = len(self.output_pads)
        if num_outputs > 0:
            spacing = self.NODE_WIDTH / (num_outputs + 1)
            for i, (name, pad) in enumerate(self.output_pads.items()):
                x_pos = spacing * (i + 1)
                pad.setPos(x_pos, self.NODE_HEIGHT_EXPANDED)

    def _reposition_pads_normal(self):
        """Reposition pads for normal state."""
        num_outputs = len(self.output_pads)
        if num_outputs > 0:
            spacing = self.NODE_WIDTH / (num_outputs + 1)
            for i, (name, pad) in enumerate(self.output_pads.items()):
                x_pos = spacing * (i + 1)
                pad.setPos(x_pos, self.NODE_HEIGHT)

    def update_prompt(self, new_prompt: str):
        """Update facet prompt from embedded editor."""
        self.facet.prompt = new_prompt

    def set_status(self, status: str):
        """
        Set execution status indicator.

        Args:
            status: 'inactive', 'ready', 'processing', 'waiting', 'cached'
        """
        color_map = {
            'inactive': '#666666',   # Gray - disabled/not running
            'ready': '#76AF6A',      # Green - ready to execute
            'processing': '#FFA726', # Yellow - LLM call in flight
            'waiting': '#EF5350',    # Red - waiting for upstream inputs
            'cached': '#64B5F6'      # Blue - using cached output
        }

        self.status_color = color_map.get(status, '#666666')
        self.status_indicator.setBrush(QBrush(QColor(self.status_color)))

    def open_field_editor(self, field_data: dict):
        """Open floating editor for a specific field."""
        print(f"[Node] Opening editor for field: {field_data['name']}")

        # Get parent panel
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            panel = view.parent()
            if isinstance(panel, FacetsEditorPanel):
                panel.show_floating_editor(self.facet, field_data)

    def toggle_lock(self, _field_data: dict):
        """Toggle lock state for this facet."""
        self.facet.locked = not self.facet.locked

        # Update lock icon appearance
        lock_icon_text = "[L]" if self.facet.locked else ""
        self.lock_icon.setPlainText(lock_icon_text)
        self.lock_icon.setDefaultTextColor(QColor("#CCAA00" if self.facet.locked else "#888888"))

        # Disable movement if locked
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, not self.facet.locked)

        # Emit modification signal if editor panel available
        if self.editor_panel:
            self.editor_panel.assemblyModified.emit()

        print(f"[Node] Facet '{self.facet.name}' {'locked' if self.facet.locked else 'unlocked'}")


class ConnectionWire(QGraphicsItem):
    """Visual representation of a connection between facet pads."""

    def __init__(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics, parent=None):
        super().__init__(parent)
        self.from_pad = from_pad
        self.to_pad = to_pad

        # Register with pads
        self.from_pad.connections.append(self)
        self.to_pad.connections.append(self)

        # Visual styling
        self.pen = QPen(QColor("#888888"), 3)
        self.active_pen = QPen(QColor("#CCAA00"), 4)  # Bright when data flows
        self.setZValue(-1)  # Draw behind nodes

        # Data packet animation (Kraftwerk style)
        self.packet_progress = 0.0  # 0.0 to 1.0 along bezier curve
        self.packet_animating = False
        self.packet_timer: Optional[QTimer] = None

    def boundingRect(self) -> QRectF:
        """Define bounding rectangle for drawing."""
        start = self.from_pad.get_scene_position()
        end = self.to_pad.get_scene_position()

        # Add padding for bezier curve
        return QRectF(start, end).normalized().adjusted(-50, -50, 50, 50)

    def paint(self, painter: QPainter, option, widget=None):
        """Draw the connection wire as a bezier curve."""
        start = self.from_pad.get_scene_position()
        end = self.to_pad.get_scene_position()

        # Create bezier curve path with vertical tangents
        path = QPainterPath()
        path.moveTo(start)

        # Control points for vertical flow (tangents exit/enter vertically)
        dy = end.y() - start.y()
        tangent_length = abs(dy) * 0.4  # 40% of vertical distance

        # First control point - extends DOWN from output pad
        ctrl1 = QPointF(start.x(), start.y() + tangent_length)

        # Second control point - extends UP to input pad
        ctrl2 = QPointF(end.x(), end.y() - tangent_length)

        path.cubicTo(ctrl1, ctrl2, end)

        # Draw wire (brighter if packet animating)
        if self.packet_animating:
            painter.setPen(self.active_pen)
        else:
            painter.setPen(self.pen)
        painter.drawPath(path)

        # Draw data packet (geometric square - no organic circles)
        if self.packet_animating and 0.0 <= self.packet_progress <= 1.0:
            # Calculate position along bezier curve
            t = self.packet_progress
            packet_pos = path.pointAtPercent(t)

            # Draw square packet (Kraftwerk geometric aesthetic)
            packet_size = 12
            painter.setBrush(QBrush(QColor("#FFFFFF")))  # Bright white square
            painter.setPen(QPen(QColor("#CCAA00"), 2))
            painter.drawRect(
                packet_pos.x() - packet_size/2,
                packet_pos.y() - packet_size/2,
                packet_size,
                packet_size
            )

    def update_path(self):
        """Update path when nodes move."""
        self.prepareGeometryChange()
        self.update()

    def animate_data_flow(self):
        """
        Animate data packet flowing through connection.

        Kraftwerk style: Linear motion, geometric packet (square), 300ms duration.
        """
        if self.packet_animating:
            return  # Already animating

        self.packet_animating = True
        self.packet_progress = 0.0

        # Linear animation - 300ms duration, 60fps
        self.packet_timer = QTimer()
        self.packet_timer.timeout.connect(self._advance_packet)
        self.packet_timer.start(16)  # ~60fps

    def _advance_packet(self):
        """Advance packet along curve (linear motion)."""
        self.packet_progress += 0.05  # 5% per frame = ~300ms total

        if self.packet_progress >= 1.0:
            # Animation complete
            self.packet_progress = 1.0
            self.packet_animating = False
            if self.packet_timer:
                self.packet_timer.stop()
                self.packet_timer = None
            # Brief wire highlight, then return to normal
            QTimer.singleShot(100, lambda: self.update())

        self.update()  # Trigger repaint


class FacetsEditorPanel(QWidget):
    """
    Main facets editor panel with node graph.

    Provides visual editing of facet assemblies with drag-and-drop,
    connection wires, and right-click menus.
    """

    # Signal emitted when assembly is modified
    assemblyModified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_assembly: Optional[FacetAssembly] = None
        self.current_assembly_name: Optional[str] = None  # Track loaded assembly
        self.node_graphics: Dict[str, FacetNodeGraphics] = {}
        self.wire_graphics: List[ConnectionWire] = []

        # CRITICAL: Lock to prevent event processing during scene transitions
        self.scene_transition_lock = False

        # Clipboard for copy/paste
        self.clipboard: List[Facet] = []

        # Undo/redo stacks (simplified - store assembly snapshots)
        self.undo_stack: List[str] = []  # YAML snapshots
        self.redo_stack: List[str] = []

        # Space-drag navigation
        self.space_pressed = False

        # Wire drawing state
        self.wire_being_drawn: Optional[QGraphicsLineItem] = None
        self.wire_start_pad: Optional[FacetPadGraphics] = None

        # Grid snapping settings
        self.snap_to_grid = True
        self.grid_size = 20  # Snap to 20px grid

        # Cognition pause state
        self.current_agent_id: Optional[str] = None
        self.cognition_paused: bool = False
        self.api_base = "http://localhost:8081/api"

        # Empty state message
        self.empty_state_label: Optional[QGraphicsTextItem] = None

        # WebSocket connection for execution event streaming (AUTOBAHN!)
        self.ws_connection = None
        self.ws_task = None
        self.ws_connected = False
        self.event_queue = asyncio.Queue() if WEBSOCKETS_AVAILABLE else None

        self.init_ui()

        # Start WebSocket connection if available
        if WEBSOCKETS_AVAILABLE:
            self._start_websocket_connection()

    def init_ui(self):
        """Initialize user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        toolbar = QHBoxLayout()

        # Assembly info
        self.assembly_label = QLabel("No assembly loaded")
        self.assembly_label.setStyleSheet("color: #CCCCCC; font-size: 11pt; font-weight: bold;")
        toolbar.addWidget(self.assembly_label)

        toolbar.addStretch()

        # Pause/Resume cognition button
        self.pause_button = QPushButton("⏸ Pause Cognition")
        self.pause_button.setFixedWidth(140)
        self.pause_button.setCheckable(True)
        self.pause_button.setEnabled(False)  # Disabled until agent loaded
        self.pause_button.clicked.connect(self.toggle_pause_cognition)
        self.pause_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QPushButton:hover:enabled {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
        """)
        toolbar.addWidget(self.pause_button)

        # Save/Load buttons
        save_btn = QPushButton("Save")
        save_btn.setFixedWidth(60)
        save_btn.clicked.connect(self.save_assembly)
        toolbar.addWidget(save_btn)

        load_btn = QPushButton("Load")
        load_btn.setFixedWidth(60)
        load_btn.clicked.connect(self.load_assembly)
        toolbar.addWidget(load_btn)

        validate_btn = QPushButton("Validate")
        validate_btn.setFixedWidth(80)
        validate_btn.clicked.connect(self.validate_assembly)
        toolbar.addWidget(validate_btn)

        layout.addLayout(toolbar)

        # Graphics view for node graph
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(-2000, -2000, 4000, 4000)

        # Draw grid background
        self.scene.setBackgroundBrush(QBrush(QColor("#2A2A2A")))
        self._draw_grid_background()

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)  # Enable rubber band selection
        self.view.setStyleSheet("border: none;")

        # Enable mouse wheel zoom
        self.view.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.view.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # Override wheelEvent for zoom
        self.view.wheelEvent = self.zoom_wheel_event

        # Enable rubber band selection with modifier key support
        self.view.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)

        # Enable multi-selection (Ctrl/Cmd click, Shift-drag)
        self.scene.setSelectionArea = self.scene.setSelectionArea  # Allow additive selection

        # Enable context menu
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self.show_context_menu)

        # Install event filter for wire drawing
        self.view.viewport().installEventFilter(self)

        layout.addWidget(self.view)

        # Bottom-right floating control panel (overlay on view)
        self.control_panel = QWidget(self.view)
        control_layout = QVBoxLayout(self.control_panel)
        control_layout.setContentsMargins(8, 8, 8, 8)
        control_layout.setSpacing(6)

        # Pause button (duplicate, more accessible)
        self.bottom_pause_btn = QPushButton("⏸")
        self.bottom_pause_btn.setFixedSize(40, 40)
        self.bottom_pause_btn.setCheckable(True)
        self.bottom_pause_btn.setEnabled(False)
        self.bottom_pause_btn.clicked.connect(self.toggle_pause_cognition)
        self.bottom_pause_btn.setToolTip("Pause/Resume Cognition")
        self.bottom_pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 20px;
                font-size: 16pt;
            }
            QPushButton:hover:enabled {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
            QPushButton:disabled {
                background-color: #2A2A2A;
                color: #666666;
            }
        """)
        control_layout.addWidget(self.bottom_pause_btn)

        self.control_panel.setStyleSheet("""
            QWidget {
                background-color: rgba(42, 42, 42, 200);
                border-radius: 6px;
                border: 1px solid #555555;
            }
        """)
        self.control_panel.setFixedSize(56, 56)  # Just one button now

        # Position control panel bottom-right (will be repositioned on resize)
        self.position_control_panel()

        # Enable key event handling
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        # Keyboard shortcuts
        self.setup_shortcuts()

        # Show empty state initially
        self.show_empty_state()

    def show_empty_state(self):
        """Show 'select a noodling' message when no assembly loaded."""
        # CRITICAL: Lock scene during transition to prevent event processing
        self.scene_transition_lock = True

        # CRITICAL: Stop all animations before clearing scene to prevent segfault
        for node_gfx in self.node_graphics.values():
            if hasattr(node_gfx, 'animation_timer') and node_gfx.animation_timer:
                node_gfx.animation_timer.stop()
                node_gfx.animation_timer = None

        # Clear any existing assembly
        self.scene.clear()
        self.node_graphics.clear()
        self.wire_graphics.clear()

        # Add centered text message
        self.empty_state_label = QGraphicsTextItem("Select a noodling to edit its facets")
        self.empty_state_label.setDefaultTextColor(QColor("#CCCCCC"))
        self.empty_state_label.setFont(QFont("Arial", 14))

        # Center the text
        text_rect = self.empty_state_label.boundingRect()
        self.empty_state_label.setPos(-text_rect.width() / 2, -text_rect.height() / 2)
        self.scene.addItem(self.empty_state_label)

        # Update label
        self.assembly_label.setText("No assembly loaded")

        # Clear agent reference
        self.current_assembly = None
        self.current_agent_id = None

        # Unlock scene - safe to process events now
        self.scene_transition_lock = False

    def clear_editor(self):
        """Clear editor when nothing is selected (alias for show_empty_state)."""
        self.show_empty_state()
        self.current_assembly_name = None

    def hide_empty_state(self):
        """Remove empty state message."""
        if self.empty_state_label and self.empty_state_label.scene():
            self.scene.removeItem(self.empty_state_label)
        self.empty_state_label = None

    def position_control_panel(self):
        """Position the floating control panel in bottom-right corner."""
        view_width = self.view.width()
        view_height = self.view.height()
        panel_width = self.control_panel.width()
        panel_height = self.control_panel.height()

        # Position 20px from bottom-right corner
        x = view_width - panel_width - 20
        y = view_height - panel_height - 20
        self.control_panel.move(x, y)
        self.control_panel.raise_()  # Ensure it's on top

    def resizeEvent(self, event):
        """Reposition control panel on window resize."""
        super().resizeEvent(event)
        if hasattr(self, 'control_panel'):
            self.position_control_panel()

    def load_assembly_from_data(self, assembly: FacetAssembly, force_reload: bool = False):
        """
        Load a facet assembly into the editor.

        Args:
            assembly: FacetAssembly to load
            force_reload: If True, reload even if same assembly already loaded
        """
        # Check if this assembly is already loaded
        if not force_reload and self.current_assembly_name == assembly.name:
            print(f"[Facets Editor] Assembly '{assembly.name}' already loaded, skipping reload")
            return

        print(f"[Facets Editor] Loading assembly: {assembly.name}")

        # CRITICAL: Lock scene during transition to prevent event processing
        self.scene_transition_lock = True

        # Hide empty state message if showing
        self.hide_empty_state()

        self.current_assembly = assembly
        self.current_assembly_name = assembly.name
        self.assembly_label.setText(f"{assembly.name} [REF]")

        # CRITICAL: Stop all animations before clearing scene to prevent segfault
        for node_gfx in self.node_graphics.values():
            if hasattr(node_gfx, 'animation_timer') and node_gfx.animation_timer:
                node_gfx.animation_timer.stop()
                node_gfx.animation_timer = None

        # Clear existing graphics
        self.scene.clear()
        self.node_graphics.clear()
        self.wire_graphics.clear()

        # Create node graphics for each facet
        for facet in assembly.facets:
            node = FacetNodeGraphics(facet, editor_panel=self)
            self.scene.addItem(node)
            self.node_graphics[facet.id] = node

        # Create connection wires
        for conn in assembly.connections:
            from_node = self.node_graphics.get(conn.from_facet)
            to_node = self.node_graphics.get(conn.to_facet)

            if from_node and to_node:
                from_pad = from_node.output_pads.get(conn.from_pad)
                to_pad = to_node.input_pads.get(conn.to_pad)

                if from_pad and to_pad:
                    wire = ConnectionWire(from_pad, to_pad)
                    self.scene.addItem(wire)
                    self.wire_graphics.append(wire)

        # Center view on content
        self.view.centerOn(500, 350)
        print(f"[Facets Editor] Assembly loaded successfully with {len(assembly.facets)} facets")

        # Unlock scene - safe to process events now
        self.scene_transition_lock = False

    def show_context_menu(self, position):
        """Show right-click context menu for adding facets."""
        menu = QMenu(self)

        # Add facet submenu
        add_menu = menu.addMenu("Add Facet")

        facet_types = [
            ("Intuition Facet", "IntuitionFacet"),
            ("Emotion Facet", "EmotionFacet"),
            ("Social Context Facet", "SocialFacet"),
            ("Memory Recall Facet", "MemoryFacet"),
            ("Response Planning Facet", "PlanningFacet"),
            ("Convergence Facet", "ConvergenceFacet"),
        ]

        for display_name, facet_type in facet_types:
            action = add_menu.addAction(display_name)
            action.triggered.connect(lambda checked, ft=facet_type, dn=display_name:
                                    self.add_facet(ft, dn, position))

        # Separator
        add_menu.addSeparator()

        # Custom/Empty facet at bottom
        custom_action = add_menu.addAction("Create empty facet")
        custom_action.triggered.connect(lambda: self.add_facet("CustomFacet", "Custom Facet", position))

        menu.exec(self.view.mapToGlobal(position))

    def add_facet(self, facet_type: str, display_name: str, position):
        """Add a new facet to the assembly."""
        print(f"[Facets Editor] add_facet called: type={facet_type}, name={display_name}")
        print(f"[Facets Editor] current_assembly exists: {self.current_assembly is not None}")

        if not self.current_assembly:
            print("[Facets Editor] ERROR: No assembly loaded - cannot add facet")
            return

        # Convert view position to scene position
        scene_pos = self.view.mapToScene(position)
        print(f"[Facets Editor] Position - view: {position}, scene: ({scene_pos.x()}, {scene_pos.y()})")

        # Create new facet with UUID
        facet_id = Facet.generate_uuid()
        facet = Facet(
            id=facet_id,
            name=display_name,
            facet_type=facet_type,
            prompt=f"TODO: Define prompt for {display_name}",
            position={'x': scene_pos.x(), 'y': scene_pos.y()}
        )
        print(f"[Facets Editor] Created facet: {facet_id}")

        # Add default pads based on type
        if facet_type == "ConvergenceFacet":
            facet.add_input_pad("input1", "First input")
            facet.add_input_pad("input2", "Second input")
            facet.add_output_pad("output", "Merged output")
        else:
            facet.add_input_pad("in", "Input")
            facet.add_output_pad("out", "Output")
        print(f"[Facets Editor] Added {len(facet.input_pads)} inputs, {len(facet.output_pads)} outputs")

        # Add to assembly
        self.current_assembly.facets.append(facet)
        print(f"[Facets Editor] Assembly now has {len(self.current_assembly.facets)} facets")

        # Create graphics
        node = FacetNodeGraphics(facet, editor_panel=self)
        self.scene.addItem(node)
        self.node_graphics[facet.id] = node
        print(f"[Facets Editor] Added node graphics to scene at ({node.pos().x()}, {node.pos().y()})")

        self.assemblyModified.emit()

    def save_assembly(self):
        """Save current assembly to YAML file."""
        if not self.current_assembly:
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self,
            "Save Facet Assembly",
            f"../facet_assemblies/{self.current_assembly.name}.yaml",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                self.current_assembly.save_yaml(filepath)
                QMessageBox.information(self, "Success", f"Assembly saved to {filepath}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save assembly: {e}")

    def load_assembly(self):
        """Load assembly from YAML file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Load Facet Assembly",
            "../facet_assemblies/",
            "YAML Files (*.yaml *.yml)"
        )

        if filepath:
            try:
                assembly = FacetAssembly.load_yaml(filepath)
                self.load_assembly_from_data(assembly)
                QMessageBox.information(self, "Success", f"Loaded assembly: {assembly.name}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load assembly: {e}")

    def validate_assembly(self):
        """Validate current assembly and show errors."""
        if not self.current_assembly:
            return

        errors = self.current_assembly.validate()

        if errors:
            error_text = "\n".join(f"- {e}" for e in errors)
            QMessageBox.warning(self, "Validation Errors", f"Assembly has errors:\n\n{error_text}")
        else:
            QMessageBox.information(self, "Validation Success", "Assembly is valid!")

    def setup_shortcuts(self):
        """Setup keyboard shortcuts for viewport navigation."""
        # F key - Tight focus with field display
        frame_shortcut = QShortcut(QKeySequence("F"), self)
        frame_shortcut.activated.connect(self.focus_selection_tight)

        # A key - Frame all (entire assembly)
        frame_all_shortcut = QShortcut(QKeySequence("A"), self)
        frame_all_shortcut.activated.connect(self.frame_all)

        # E key - Expand selected node for inline editing
        expand_shortcut = QShortcut(QKeySequence("E"), self)
        expand_shortcut.activated.connect(self.toggle_node_expansion)

        # Plus/Minus - Zoom in/out
        zoom_in_shortcut = QShortcut(QKeySequence.StandardKey.ZoomIn, self)
        zoom_in_shortcut.activated.connect(lambda: self.zoom_view(1.2))

        zoom_out_shortcut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        zoom_out_shortcut.activated.connect(lambda: self.zoom_view(1/1.2))

        # Home - Reset zoom and center
        home_shortcut = QShortcut(QKeySequence("Home"), self)
        home_shortcut.activated.connect(self.reset_view)

        # Copy/Paste/Duplicate/Delete
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self)
        copy_shortcut.activated.connect(self.copy_selection)

        paste_shortcut = QShortcut(QKeySequence.StandardKey.Paste, self)
        paste_shortcut.activated.connect(self.paste_selection)

        # Cmd-D for duplicate (copy + paste in one step)
        duplicate_shortcut = QShortcut(QKeySequence("Ctrl+D"), self)
        duplicate_shortcut.activated.connect(self.duplicate_selection)

        delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        delete_shortcut.activated.connect(self.delete_selection)

        # Undo/Redo
        undo_shortcut = QShortcut(QKeySequence.StandardKey.Undo, self)
        undo_shortcut.activated.connect(self.undo)

        redo_shortcut = QShortcut(QKeySequence.StandardKey.Redo, self)
        redo_shortcut.activated.connect(self.redo)

    def zoom_wheel_event(self, event):
        """Handle mouse wheel for zooming."""
        # Get zoom factor based on wheel delta
        delta = event.angleDelta().y()
        zoom_factor = 1.15 if delta > 0 else 1/1.15

        self.zoom_view(zoom_factor)

    def zoom_view(self, factor: float):
        """
        Zoom the view by given factor.

        Args:
            factor: Zoom multiplier (>1 = zoom in, <1 = zoom out)
        """
        # Limit zoom range
        current_scale = self.view.transform().m11()
        new_scale = current_scale * factor

        # Calculate max zoom based on "frame all" zoom level
        # Get all nodes to determine comfortable max zoom
        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, FacetNodeGraphics)
        ]

        max_zoom = 2.0  # Default max if no nodes

        if all_nodes:
            # Calculate bounding rect
            bounding_rect = all_nodes[0].sceneBoundingRect()
            for node in all_nodes[1:]:
                bounding_rect = bounding_rect.united(node.sceneBoundingRect())

            # Calculate what zoom would frame all nodes
            view_rect = self.view.viewport().rect()
            if bounding_rect.width() > 0 and view_rect.width() > 0:
                frame_all_scale = view_rect.width() / bounding_rect.width()
                max_zoom = frame_all_scale * 2.0  # 2x the frame-all zoom

        # Clamp between 0.5x (reasonable minimum) and calculated max
        if new_scale < 0.5 or new_scale > max_zoom:
            return

        self.view.scale(factor, factor)

    def frame_selection(self):
        """Frame selected node in view, or all nodes if none selected."""
        selected_items = self.scene.selectedItems()

        # Get selected nodes (filter to FacetNodeGraphics only)
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if selected_nodes:
            # Frame selected nodes
            self.frame_nodes(selected_nodes)
        else:
            # Frame all nodes
            all_nodes = [
                item for item in self.scene.items()
                if isinstance(item, FacetNodeGraphics)
            ]
            self.frame_nodes(all_nodes)

    def frame_nodes(self, nodes: List[FacetNodeGraphics], padding_factor: float = 0.2):
        """
        Frame given nodes in view with padding.

        Args:
            nodes: List of FacetNodeGraphics to frame
            padding_factor: Padding as fraction of bounding box (0.2 = 20%)
        """
        if not nodes:
            return

        # Calculate bounding rect of all nodes
        bounding_rect = nodes[0].sceneBoundingRect()
        for node in nodes[1:]:
            bounding_rect = bounding_rect.united(node.sceneBoundingRect())

        # Add padding
        padding = max(bounding_rect.width(), bounding_rect.height()) * padding_factor
        bounding_rect.adjust(-padding, -padding, padding, padding)

        # Fit in view
        self.view.fitInView(bounding_rect, Qt.AspectRatioMode.KeepAspectRatio)

    def reset_view(self):
        """Reset zoom to 100% and center on origin."""
        self.view.resetTransform()
        self.view.centerOn(500, 350)

    def frame_all(self):
        """Frame entire assembly (A key shortcut)."""
        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, FacetNodeGraphics)
        ]
        self.frame_nodes(all_nodes)

    def focus_selection_tight(self):
        """
        Tight focus on selected node with field display (F key).

        Zooms way in to show fields and enable inline editing.
        """
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if selected_nodes:
            # Frame selected with minimal padding
            self.frame_nodes(selected_nodes, padding_factor=0.05)

            # Force field display on selected nodes
            for node in selected_nodes:
                node.show_fields(force=True)
        else:
            print("[Facets Editor] No facet selected to focus")

    def toggle_node_expansion(self):
        """Open field editor for selected node (E key - edits Processing Prompt)."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if len(selected_nodes) == 1:
            node = selected_nodes[0]
            # Open editor for Processing Prompt field (most common edit)
            fields = node.facet.get_editable_fields()
            prompt_field = next((f for f in fields if f['key'] == 'prompt'), None)
            if prompt_field:
                self.show_floating_editor(node.facet, prompt_field)
            else:
                print("[Facets Editor] No prompt field available for this facet type")
        elif len(selected_nodes) == 0:
            print("[Facets Editor] No facet selected")
        else:
            print("[Facets Editor] Select only one facet to edit")

    def copy_selection(self):
        """Copy selected facets to clipboard."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            print("[Facets Editor] No facets selected to copy")
            return

        # Copy facet data (deep copy)
        self.clipboard = []
        for node in selected_nodes:
            # Skip special nodes
            if node.facet.facet_type == "SpecialNode":
                continue

            # Store facet copy
            import copy
            facet_copy = copy.deepcopy(node.facet)
            self.clipboard.append(facet_copy)

        print(f"[Facets Editor] Copied {len(self.clipboard)} facets")

    def paste_selection(self):
        """Paste facets from clipboard with internal connections preserved."""
        if not self.clipboard or not self.current_assembly:
            print("[Facets Editor] Nothing to paste or no assembly loaded")
            return

        # Get current mouse position in scene coords
        cursor_pos = self.view.mapToScene(self.view.mapFromGlobal(QCursor.pos()))

        # Map old IDs to new IDs for connection rewiring
        id_mapping = {}

        # Calculate offset for group (preserve relative positions)
        if self.clipboard:
            # Find top-left corner of clipboard group
            min_x = min(f.position['x'] for f in self.clipboard)
            min_y = min(f.position['y'] for f in self.clipboard)

            # Offset entire group to cursor position
            group_offset_x = cursor_pos.x() - min_x
            group_offset_y = cursor_pos.y() - min_y

        # Paste each facet with new UUID and preserved relative position
        import copy
        for facet_template in self.clipboard:
            # Deep copy and generate new UUID
            new_facet = copy.deepcopy(facet_template)
            old_id = new_facet.id
            new_facet.id = Facet.generate_uuid()
            id_mapping[old_id] = new_facet.id

            # Preserve relative position within group
            new_facet.position = {
                'x': facet_template.position['x'] + group_offset_x + 50,  # +50 for slight offset
                'y': facet_template.position['y'] + group_offset_y + 50
            }

            # Add to assembly
            self.current_assembly.facets.append(new_facet)

            # Create graphics
            node = FacetNodeGraphics(new_facet, editor_panel=self)
            self.scene.addItem(node)
            self.node_graphics[new_facet.id] = node

        # Duplicate internal connections (connections between pasted nodes)
        clipboard_ids = set(f.id for f in self.clipboard)
        for conn in self.current_assembly.connections:
            # Check if this connection is entirely within the copied set
            if conn.from_facet in clipboard_ids and conn.to_facet in clipboard_ids:
                # Duplicate this connection with new IDs
                new_conn = FacetConnection(
                    from_facet=id_mapping[conn.from_facet],
                    from_pad=conn.from_pad,
                    to_facet=id_mapping[conn.to_facet],
                    to_pad=conn.to_pad
                )
                self.current_assembly.connections.append(new_conn)

                # Create visual wire
                from_node = self.node_graphics.get(new_conn.from_facet)
                to_node = self.node_graphics.get(new_conn.to_facet)
                if from_node and to_node:
                    from_pad = from_node.output_pads.get(new_conn.from_pad)
                    to_pad = to_node.input_pads.get(new_conn.to_pad)
                    if from_pad and to_pad:
                        wire = ConnectionWire(from_pad, to_pad)
                        self.scene.addItem(wire)
                        self.wire_graphics.append(wire)

        print(f"[Facets Editor] Pasted {len(self.clipboard)} facets with internal connections")
        self.assemblyModified.emit()

    def duplicate_selection(self):
        """Duplicate selected facets in place (Cmd-D)."""
        # Copy selection to clipboard
        self.copy_selection()
        # Immediately paste
        if self.clipboard:
            self.paste_selection()

    def delete_selection(self):
        """Delete selected facets."""
        if not self.current_assembly:
            print("[Facets Editor] No assembly loaded")
            return

        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            print("[Facets Editor] No facets selected to delete")
            return

        # Filter out special nodes (can't delete)
        deletable_nodes = []
        for node in selected_nodes:
            if node.is_special_node or node.facet.name in ["INCOMING", "OUTGOING"]:
                print(f"[Facets Editor] Cannot delete special node: {node.facet.name}")
            else:
                deletable_nodes.append(node)

        if not deletable_nodes:
            print("[Facets Editor] No deletable facets in selection")
            return

        # Remove from assembly and scene
        for node in deletable_nodes:
            # Remove facet from assembly
            self.current_assembly.facets = [
                f for f in self.current_assembly.facets
                if f.id != node.facet.id
            ]

            # Remove connections involving this facet
            self.current_assembly.connections = [
                c for c in self.current_assembly.connections
                if c.from_facet != node.facet.id and c.to_facet != node.facet.id
            ]

            # Remove from scene
            self.scene.removeItem(node)
            del self.node_graphics[node.facet.id]

        print(f"[Facets Editor] Deleted {len(deletable_nodes)} facets")
        self.assemblyModified.emit()

    def undo(self):
        """Undo last operation."""
        # TODO: Implement undo from snapshot stack
        print("[Facets Editor] Undo not yet implemented")

    def redo(self):
        """Redo last undone operation."""
        # TODO: Implement redo from snapshot stack
        print("[Facets Editor] Redo not yet implemented")

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Space and not self.space_pressed:
            # Space pressed - switch to pan mode
            self.space_pressed = True
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.key() == Qt.Key.Key_Space and self.space_pressed:
            # Space released - back to selection mode
            self.space_pressed = False
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def start_wire_drawing(self, start_pad: FacetPadGraphics):
        """Start drawing a connection wire from a pad."""
        print(f"[Facets Editor] Starting wire from {start_pad.facet_node.facet.name}.{start_pad.pad.name}")
        self.wire_start_pad = start_pad

        # Create temporary line for visual feedback
        start_pos = start_pad.get_scene_position()
        self.wire_being_drawn = QGraphicsLineItem(
            start_pos.x(), start_pos.y(),
            start_pos.x(), start_pos.y()
        )
        self.wire_being_drawn.setPen(QPen(QColor("#FFFFFF"), 2, Qt.PenStyle.DashLine))
        self.scene.addItem(self.wire_being_drawn)

    def eventFilter(self, obj, event):
        """Handle viewport events for wire drawing and selection."""
        if obj == self.view.viewport():
            # Handle clicks on background
            if event.type() == event.Type.MouseButtonPress:
                scene_pos = self.view.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                # Filter to only facet nodes
                clicked_nodes = [
                    item for item in items
                    if isinstance(item, FacetNodeGraphics)
                ]

                if not clicked_nodes:
                    # Clicked empty background
                    if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                        # Cmd-click - invert selection
                        self.invert_selection()
                        return True
                    else:
                        # Regular click - collapse any expanded nodes
                        self.collapse_all_nodes()
                        return False  # Allow default behavior (clear selection)

            if event.type() == event.Type.MouseMove and self.wire_being_drawn:
                # Update wire endpoint to follow mouse
                scene_pos = self.view.mapToScene(event.pos())
                line = self.wire_being_drawn.line()
                self.wire_being_drawn.setLine(
                    line.x1(), line.y1(),
                    scene_pos.x(), scene_pos.y()
                )
                return True

            elif event.type() == event.Type.MouseButtonRelease and self.wire_being_drawn:
                # Check if released over a pad
                scene_pos = self.view.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                end_pad = None
                for item in items:
                    if isinstance(item, FacetPadGraphics):
                        end_pad = item
                        break

                if end_pad and self.wire_start_pad:
                    # Validate connection
                    if self.can_connect(self.wire_start_pad, end_pad):
                        self.create_connection(self.wire_start_pad, end_pad)
                    else:
                        print("[Facets Editor] Invalid connection")

                # Clean up temporary wire
                self.scene.removeItem(self.wire_being_drawn)
                self.wire_being_drawn = None
                self.wire_start_pad = None
                return True

        return super().eventFilter(obj, event)

    def can_connect(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics) -> bool:
        """Check if connection is valid."""
        # Can't connect to self
        if from_pad.facet_node == to_pad.facet_node:
            return False

        # Must be output -> input
        if from_pad.pad.pad_type != PadType.OUTPUT:
            return False
        if to_pad.pad.pad_type != PadType.INPUT:
            return False

        # Check if connection already exists
        for conn in self.current_assembly.connections:
            if (conn.from_facet == from_pad.facet_node.facet.id and
                conn.from_pad == from_pad.pad.name and
                conn.to_facet == to_pad.facet_node.facet.id and
                conn.to_pad == to_pad.pad.name):
                return False  # Already connected

        return True

    def create_connection(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics):
        """Create a connection between two pads."""
        if not self.current_assembly:
            return

        # Create connection data
        connection = FacetConnection(
            from_facet=from_pad.facet_node.facet.id,
            from_pad=from_pad.pad.name,
            to_facet=to_pad.facet_node.facet.id,
            to_pad=to_pad.pad.name
        )
        self.current_assembly.connections.append(connection)

        # Create visual wire
        wire = ConnectionWire(from_pad, to_pad)
        self.scene.addItem(wire)
        self.wire_graphics.append(wire)

        print(f"[Facets Editor] Connected {from_pad.facet_node.facet.name}.{from_pad.pad.name} -> {to_pad.facet_node.facet.name}.{to_pad.pad.name}")
        self.assemblyModified.emit()

    def invert_selection(self):
        """Invert current selection (ZBrush-style mask inverter)."""
        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, FacetNodeGraphics)
        ]

        for node in all_nodes:
            node.setSelected(not node.isSelected())

        selected_count = sum(1 for n in all_nodes if n.isSelected())
        print(f"[Facets Editor] Inverted selection - {selected_count} facets now selected")

    def _draw_grid_background(self):
        """Draw subtle grid lines on background."""
        grid_size = self.grid_size
        scene_rect = self.scene.sceneRect()

        # Faint gray for grid lines
        grid_pen = QPen(QColor("#333333"), 1, Qt.PenStyle.DotLine)

        # Draw vertical lines
        x = scene_rect.left()
        while x <= scene_rect.right():
            if x % grid_size == 0:
                line = self.scene.addLine(
                    x, scene_rect.top(),
                    x, scene_rect.bottom(),
                    grid_pen
                )
                line.setZValue(-100)  # Behind everything
            x += grid_size

        # Draw horizontal lines
        y = scene_rect.top()
        while y <= scene_rect.bottom():
            if y % grid_size == 0:
                line = self.scene.addLine(
                    scene_rect.left(), y,
                    scene_rect.right(), y,
                    grid_pen
                )
                line.setZValue(-100)  # Behind everything
            y += grid_size

    def toggle_grid_snap(self, enabled: bool):
        """Toggle grid snapping on/off."""
        self.snap_to_grid = enabled
        print(f"[Facets Editor] Grid snapping: {'ON' if enabled else 'OFF'}")

    def set_grid_size(self, size: int):
        """Set grid snap size in pixels."""
        self.grid_size = size
        print(f"[Facets Editor] Grid size: {size}px")

    def collapse_all_nodes(self):
        """Collapse all expanded nodes (hide fields on all nodes)."""
        for item in self.scene.items():
            if isinstance(item, FacetNodeGraphics):
                item.hide_fields()

    def show_floating_editor(self, facet: Facet, field_data: dict):
        """
        Show floating text editor for a facet field.

        Args:
            facet: Facet being edited
            field_data: Field definition dict
        """
        editor = FloatingTextEditor(
            field_name=field_data['name'],
            field_key=field_data['key'],
            initial_value=field_data['value'],
            read_only=field_data['read_only'],
            parent=self
        )

        # Connect apply signal
        def on_applied(key, value):
            # Update facet field
            if key == 'prompt':
                facet.prompt = value
                print(f"[Facets Editor] Updated {facet.name}.prompt")
            # Refresh field display if node currently showing fields
            for item in self.scene.items():
                if isinstance(item, FacetNodeGraphics) and item.facet.id == facet.id:
                    if item.field_widgets:  # If fields currently visible, refresh them
                        item.show_fields(force=True)

        editor.textApplied.connect(on_applied)

        # Position centered on screen
        editor.move(
            self.mapToGlobal(self.rect().center()).x() - 250,
            self.mapToGlobal(self.rect().center()).y() - 200
        )

        # Show as modal dialog
        editor.exec()

    def toggle_pause_cognition(self, checked: bool):
        """Toggle cognitive processing pause for the current agent."""
        if not self.current_agent_id:
            QMessageBox.warning(self, "No Agent", "No agent is currently loaded in the Facets Editor.")
            self.pause_button.setChecked(False)
            return

        try:
            if checked:
                # PAUSING: Request pause and wait for cycle completion
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={'paused': True, 'agent_id': self.current_agent_id}, timeout=35)

                if response.status_code == 200:
                    self.cognition_paused = True
                    self.pause_button.setText("▶ Resume Cognition")
                    self.bottom_pause_btn.setText("▶")
                    self.bottom_pause_btn.setChecked(True)

                    # Update Stage panel to reflect pause state
                    self._update_stage_pause_state(True)

                    # Refresh all visible fields to show output as editable
                    for node in self.node_graphics.values():
                        if node.field_widgets:  # If fields currently visible, refresh them
                            node.show_fields(force=True)
                else:
                    QMessageBox.warning(self, "Pause Failed", f"Failed to pause cognition: {response.status_code}")
                    self.pause_button.setChecked(False)
                    self.bottom_pause_btn.setChecked(False)

            else:
                # RESUMING: Apply edits and resume cognition
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={'paused': False, 'agent_id': self.current_agent_id}, timeout=2)

                if response.status_code == 200:
                    self.cognition_paused = False
                    self.pause_button.setText("⏸ Pause Cognition")
                    self.bottom_pause_btn.setText("⏸")
                    self.bottom_pause_btn.setChecked(False)

                    # Update Stage panel to reflect resume state
                    self._update_stage_pause_state(False)

                    # Refresh all visible fields to show output as read-only again
                    for node in self.node_graphics.values():
                        if node.field_widgets:  # If fields currently visible, refresh them
                            node.show_fields(force=True)
                else:
                    QMessageBox.warning(self, "Resume Failed", f"Failed to resume cognition: {response.status_code}")
                    self.pause_button.setChecked(True)
                    self.bottom_pause_btn.setChecked(True)

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Pause/resume error: {str(e)}")
            self.pause_button.setChecked(not checked)  # Revert button state

    def set_current_agent(self, agent_id: str):
        """
        Set the current agent whose facets are being edited.

        Fetches and loads the agent's facet assembly from the API.
        """
        self.current_agent_id = agent_id
        enabled = True if agent_id else False
        self.pause_button.setEnabled(enabled)
        self.bottom_pause_btn.setEnabled(enabled)

        # Reset pause state when switching agents
        if self.cognition_paused:
            self.pause_button.setChecked(False)
            self.bottom_pause_btn.setChecked(False)
            self.cognition_paused = False

        # Fetch and load the agent's facet assembly
        if agent_id:
            try:
                import requests
                response = requests.get(f"http://localhost:8081/api/agents/{agent_id}", timeout=2)
                if response.status_code == 200:
                    agent_data = response.json()
                    config = agent_data.get('config', {})
                    facet_assembly_ref = config.get('facet_assembly')

                    if facet_assembly_ref:
                        # Load assembly from YAML file
                        import os
                        assembly_path = os.path.join(
                            os.path.dirname(__file__),
                            '../facet_assemblies',
                            f"{facet_assembly_ref}.yaml"
                        )

                        if os.path.exists(assembly_path):
                            from ..core.facet_system import FacetAssembly
                            assembly = FacetAssembly.load_yaml(assembly_path)
                            self.load_assembly_from_data(assembly, force_reload=True)
                            print(f"[Facets Editor] Loaded assembly: {assembly.name} for agent {agent_id}")
                        else:
                            print(f"[Facets Editor] Assembly file not found: {assembly_path}")
                    else:
                        print(f"[Facets Editor] Agent {agent_id} has no facet_assembly")
            except Exception as e:
                print(f"[Facets Editor] Failed to load agent assembly: {e}")

    def _update_stage_pause_state(self, paused: bool):
        """Notify Stage panel to update pause state for current agent."""
        if not self.current_agent_id:
            return

        # Find main window and update Stage panel
        widget = self.parent()
        while widget and not hasattr(widget, 'hierarchy'):
            widget = widget.parent() if hasattr(widget, 'parent') else None

        if widget and hasattr(widget, 'hierarchy'):
            stage_panel = widget.hierarchy
            # Update tracked pause state
            stage_panel.agent_pause_states[self.current_agent_id] = paused
            # Refresh Stage to update icon
            stage_panel.refresh_scene()

    # ========== WEBSOCKET AUTOBAHN - EXECUTION EVENT STREAMING ==========
    # Trans-Europa-Facet-Express: Real-time telemetry from port 8081

    def _start_websocket_connection(self):
        """Start WebSocket connection to execution event stream."""
        if not WEBSOCKETS_AVAILABLE:
            print("[Facets Editor] websockets module not available - animations disabled")
            return

        # Start event processing timer (polls queue from Qt thread)
        self.event_timer = QTimer()
        self.event_timer.timeout.connect(self._process_event_queue)
        self.event_timer.start(16)  # 60fps event processing

        # Start WebSocket task in separate thread
        import threading
        ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        ws_thread.start()

        print("[Facets Editor] WebSocket connection initiated to ws://localhost:8081/ws/execution_events")

    def _run_websocket_loop(self):
        """Run WebSocket event loop in separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_handler())
        except Exception as e:
            print(f"[Facets Editor] WebSocket loop error: {e}")

    async def _websocket_handler(self):
        """Handle WebSocket connection and message receiving."""
        uri = "ws://localhost:8081/ws/execution_events"

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    self.ws_connection = websocket
                    self.ws_connected = True
                    print("[Facets Editor] WebSocket connected")

                    async for message in websocket:
                        try:
                            event_data = json.loads(message)
                            # Add to queue for Qt thread processing
                            if self.event_queue:
                                await self.event_queue.put(event_data)
                        except json.JSONDecodeError as e:
                            print(f"[Facets Editor] Invalid JSON: {e}")

            except Exception as e:
                self.ws_connected = False
                print(f"[Facets Editor] WebSocket error: {e}, reconnecting in 5s...")
                await asyncio.sleep(5)  # Reconnect delay

    def _process_event_queue(self):
        """Process execution events from queue (called from Qt timer)."""
        if not self.event_queue:
            return

        # Process up to 10 events per frame (prevent UI blocking)
        for _ in range(10):
            try:
                event = self.event_queue.get_nowait()
                self._handle_execution_event(event)
            except:
                break  # Queue empty

    def _handle_execution_event(self, event: dict):
        """
        Handle execution event and trigger appropriate animation.

        Event types:
        - facet_start: Node begins executing (yellow + pulse)
        - facet_complete: Node finishes (brief bright, then idle)
        - data_flow: Animate packet along connection wire
        - convergence_wait: (future: show waiting state)
        """
        # CRITICAL: Skip event processing during scene transitions
        if self.scene_transition_lock:
            return  # Scene is being cleared/rebuilt, ignore all events

        event_type = event.get('type')
        event_subtype = event.get('subtype')

        if event_type != 'facet_execution':
            return  # Ignore non-execution events

        facet_id = event.get('source_id')
        if not facet_id or facet_id not in self.node_graphics:
            return  # Facet not in current assembly

        node = self.node_graphics.get(facet_id)
        if not node:
            return  # Node was deleted (race condition during scene transition)

        # CRITICAL: Check if node is still in scene (not deleted)
        if not node.scene():
            return  # Node removed from scene, skip event

        # KRAFTWERK CLICK - Play terminal keypress sound for every event
        self._play_pachinko_sound()

        if event_subtype == 'facet_start':
            # KRAFTWERK: Node begins processing
            node.set_execution_state('processing')

        elif event_subtype == 'facet_complete':
            # KRAFTWERK: Node completes (brief satisfaction, then idle)
            node.set_execution_state('complete')

        elif event_subtype == 'data_flow':
            # AUTOBAHN: Animate data packet along wire
            from_facet = event.get('data', {}).get('from_facet')
            to_facet = event.get('data', {}).get('to_facet')

            if from_facet and to_facet:
                # Find connection wire between these facets
                # CRITICAL: Check if wire still in scene (race condition protection)
                for wire in list(self.wire_graphics):  # Copy list to avoid modification issues
                    if not wire.scene():
                        continue  # Wire was deleted, skip
                    if (wire.from_pad.facet_node.facet.id == from_facet and
                        wire.to_pad.facet_node.facet.id == to_facet):
                        wire.animate_data_flow()
                        break

        elif event_subtype == 'quantum_collapse':
            # QUANTUM: Orchestrated objective reduction event
            # Purple/blue flash + higher pitch sound
            node.set_execution_state('quantum_collapse')
            self._play_quantum_collapse_sound()

    def _play_pachinko_sound(self):
        """Play termkeypress.ogg sound (Kraftwerk pachinko click)."""
        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Get sound file path
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')
            sound_path = os.path.join(resources_dir, 'termkeypress.ogg')

            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create sound effect if not already created
            if not hasattr(self, '_pachinko_sound'):
                self._pachinko_sound = QSoundEffect()
                self._pachinko_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._pachinko_sound.setVolume(0.3)  # 30% volume (not too loud!)

            # Play (non-blocking)
            self._pachinko_sound.play()

        except Exception as e:
            # Silent fail - don't break execution visualization if sound fails
            pass

    def _play_quantum_collapse_sound(self):
        """
        Play quantum collapse sound effect.

        Higher pitch than normal pachinko click to indicate quantum event.
        Uses terminal beep at higher frequency.
        """
        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Get sound file path (use termstart.ogg for higher pitch)
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')
            sound_path = os.path.join(resources_dir, 'termstart.ogg')  # Higher pitch than keypress

            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create sound effect if not already created
            if not hasattr(self, '_quantum_sound'):
                self._quantum_sound = QSoundEffect()
                self._quantum_sound.setSource(QUrl.fromLocalFile(sound_path))
                self._quantum_sound.setVolume(0.4)  # Slightly louder than pachinko

            # Play (non-blocking)
            self._quantum_sound.play()

        except Exception as e:
            # Silent fail - don't break execution visualization if sound fails
            pass


if __name__ == "__main__":
    """Test the facets editor panel."""
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)

    # Load test assembly
    test_assembly_path = "../facet_assemblies/anklebiter_default.yaml"
    if os.path.exists(test_assembly_path):
        assembly = FacetAssembly.load_yaml(test_assembly_path)
    else:
        # Create simple test assembly
        from ..core.facet_system import create_default_assembly
        assembly = create_default_assembly()

    # Create and show editor
    editor = FacetsEditorPanel()
    editor.load_assembly_from_data(assembly)
    editor.setWindowTitle("Facets Editor - Test")
    editor.resize(1200, 800)
    editor.show()

    sys.exit(app.exec())
