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
    QGraphicsProxyWidget, QTextEdit, QSpinBox
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QLineF, QTimer, QPropertyAnimation, QEasingCurve, QVariantAnimation
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath, QCursor, QKeySequence, QShortcut
)
from typing import Optional, List, Dict, Tuple, Any
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


def get_facet_header_color(facet: Facet) -> str:
    """
    Get coffee shop palette color for facet type header.

    Colors match Neural Canvas wire palette (saturated, easy on eyes).
    """
    # Check for custom color override first
    if hasattr(facet, 'custom_color') and facet.custom_color:
        return facet.custom_color

    # Special I/O nodes
    if facet.name in ["INCOMING", "OUTGOING"]:
        return "#5A7A5A"  # Forest green (lighter, like Neural Canvas wires)

    # Facet type taxonomy (lighter palette matching Neural Canvas wires)
    facet_type = facet.facet_type

    if "LLMFacet" in facet_type or "LLM" in facet_type:
        return "#6A4A6A"  # Deep mauve (reasoning - matches PHENOMENAL_STATE)
    elif "ScriptedFacet" in facet_type or "Scripted" in facet_type:
        return "#8A7A4A"  # Muted gold (custom logic - matches AFFECT)
    elif "ContextIntelligence" in facet_type:
        return "#4A5A6A"  # Slate blue (utility - matches HIDDEN_STATE)
    elif "Convergence" in facet_type:
        return "#5A6A7A"  # Steel blue (synthesis - matches CELL_STATE)
    elif "CharmNetwork" in facet_type:
        return "#6A4A6A"  # Deep mauve (neural processing)
    else:
        return "#5A5A5A"  # Medium gray (default - matches TENSOR)


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

    def contextMenuEvent(self, event):
        """Pass context menu to parent node."""
        if self.parent_node:
            self.parent_node.contextMenuEvent(event)
        else:
            event.accept()  # Prevent default behavior


class FacetPadGraphics(QGraphicsEllipseItem):
    """Visual representation of a facet pad (connection point) - Neural Canvas style."""

    PAD_RADIUS = 5  # Match Neural Canvas

    def __init__(self, pad: FacetPad, facet_node: 'FacetNodeGraphics', parent=None):
        super().__init__(-self.PAD_RADIUS, -self.PAD_RADIUS,
                         self.PAD_RADIUS * 2, self.PAD_RADIUS * 2, parent)
        self.pad = pad
        self.facet_node = facet_node

        # Pad color logic:
        # - OUTPUT pads: Always use parent facet's header color (they're the source)
        # - INPUT pads: Start neutral gray, adopt wire color when connected
        if pad.pad_type == PadType.OUTPUT:
            pad_color = get_facet_header_color(facet_node.facet)
        else:  # INPUT pad
            pad_color = "#666666"  # Neutral gray until connected

        self.default_brush = QBrush(QColor(pad_color))
        self.hover_brush = QBrush(QColor(pad_color).lighter(130))  # Brighter on hover
        self.setBrush(self.default_brush)
        self.setPen(QPen(QColor("#333"), 1.5))
        self.setAcceptHoverEvents(True)

        # Make pad independently clickable (don't propagate to parent)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, False)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setZValue(10)  # Draw on top like Neural Canvas

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

    def update_color_from_connection(self):
        """
        Update input pad color to match incoming wire.

        Called when connections are added/removed.
        Output pads always keep their parent node's color.
        Input pads adopt the color of the wire feeding them.
        """
        if self.pad.pad_type == PadType.OUTPUT:
            # Output pads never change - always match parent node
            return

        # Input pad: adopt color from first connection, or neutral gray if none
        if self.connections:
            # Get color from source facet (the facet feeding this input)
            wire = self.connections[0]  # Use first connection
            source_facet = wire.from_pad.facet_node.facet
            pad_color = get_facet_header_color(source_facet)
        else:
            # No connections - neutral gray
            pad_color = "#666666"

        # Update brushes
        self.default_brush = QBrush(QColor(pad_color))
        self.hover_brush = QBrush(QColor(pad_color).lighter(130))
        self.setBrush(self.default_brush)

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

    def contextMenuEvent(self, event):
        """Pass context menu to parent node."""
        # Let the parent node handle it
        if self.facet_node:
            self.facet_node.contextMenuEvent(event)
        else:
            event.accept()  # Prevent default behavior


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

        # Calculate height based on number of pads (horizontal flow needs vertical space)
        header_height = 24
        port_start_y = header_height + 15
        port_spacing = 20
        max_pads = max(len(facet.input_pads), len(facet.output_pads), 1)
        min_height = self.NODE_HEIGHT_COMPACT if is_special else self.NODE_HEIGHT
        calculated_height = port_start_y + (max_pads * port_spacing) + 15  # +15 for bottom padding
        initial_height = max(min_height, calculated_height)

        super().__init__(0, 0, self.NODE_WIDTH, initial_height, parent)
        self.facet = facet
        self.is_special_node = is_special
        self.editor_panel = editor_panel  # Reference to FacetsEditorPanel for pause state

        # Set initial brush/pen (will be overridden in paint())
        self.setBrush(QBrush(QColor("#3a3a3a")))
        self.setPen(QPen(QColor("#555555"), 2))

        # Store pens for animation system
        self.default_pen = QPen(QColor("#555555"), 2)
        self.selected_pen = QPen(QColor("#FFFFFF"), 3)

        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)

        # Accept both Shift and Cmd for multi-selection
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        # Text now painted directly in paint() method (Neural Canvas style)
        # No QGraphicsTextItem children needed for title/type

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
        self.error_flash_count = 0  # Counter for error flash animation

        # Cycle tracking (for async cycle visualization - supports MULTIPLE active cycles)
        # List of (cycle_id, cycle_color, inputs) tuples for stacked display
        self.active_cycles: List[Tuple[str, QColor, Optional[Dict]]] = []

        # Input inspection (for debugging when paused) - keep most recent for quick access
        self.last_inputs: Optional[Dict[str, Any]] = None  # Last inputs received during execution
        self.last_outputs: Optional[Dict[str, Any]] = None  # Last outputs produced

        # Per-cycle input/output storage for inspection
        self.cycle_data: Dict[str, Dict[str, Any]] = {}  # cycle_id -> {'inputs': ..., 'outputs': ...}

        # Drag tracking for undo
        self.drag_start_pos: Optional[tuple] = None  # Position when drag started
        self.is_being_dragged = False

        self._create_pads()

        # Set initial position from facet metadata
        self.setPos(facet.position['x'], facet.position['y'])

    def _create_pads(self):
        """Create visual representations of input/output pads (horizontal layout)."""
        # Calculate port start Y (after header + some padding)
        header_height = 24
        port_start_y = header_height + 15  # Start below header with padding
        port_spacing = 20  # Vertical spacing between ports

        # Input pads on LEFT edge (horizontal flow like Neural Canvas)
        for i, pad in enumerate(self.facet.input_pads):
            pad_graphics = FacetPadGraphics(pad, self, self)
            y_pos = port_start_y + (i * port_spacing)
            pad_graphics.setPos(0, y_pos)  # x=0 (left edge), y varies vertically
            self.input_pads[pad.name] = pad_graphics

            # Pad label (to the right of pad)
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(15, y_pos - 8)
            label.setDefaultTextColor(QColor("#AAAAAA"))
            label.setFont(QFont("Arial", 8))

        # Output pads on RIGHT edge (horizontal flow like Neural Canvas)
        for i, pad in enumerate(self.facet.output_pads):
            pad_graphics = FacetPadGraphics(pad, self, self)
            y_pos = port_start_y + (i * port_spacing)
            pad_graphics.setPos(self.NODE_WIDTH, y_pos)  # x=NODE_WIDTH (right edge), y varies
            self.output_pads[pad.name] = pad_graphics

            # Pad label (to the left of pad)
            label = QGraphicsTextItem(pad.name, self)
            label.setPos(self.NODE_WIDTH - 70, y_pos - 8)  # Right-aligned
            label.setDefaultTextColor(QColor("#AAAAAA"))
            label.setFont(QFont("Arial", 8))

    def boundingRect(self) -> QRectF:
        """
        Return bounding rect that includes selection highlight area.

        The selection highlight extends 3px padding + 1.5px (half of 3px pen width)
        beyond the node rect. We use 6px margin to be safe and prevent artifacts.
        """
        rect = self.rect()
        margin = 6  # Account for selection highlight (3px padding + 3px/2 pen width + safety)
        return rect.adjusted(-margin, -margin, margin, margin)

    def paint(self, painter: QPainter, option, widget=None):
        """Render the node (Blender-style: colored header, uniform gray body)."""
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        rect = self.rect()

        # Main body background (uniform dark gray) - rounded corners
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor("#3a3a3a")))
        painter.drawRoundedRect(rect, 4, 4)

        # Header bar with taxonomic color (sharp edges, clipped by node outline)
        header_height = 24
        header_rect = QRectF(0, 0, rect.width(), header_height)

        # Get facet type color (coffee shop palette)
        header_color = QColor(get_facet_header_color(self.facet))
        painter.setBrush(QBrush(header_color))
        painter.drawRect(header_rect)  # Sharp rectangle, not rounded

        # Selection highlight (white outline with padding - Neural Canvas style)
        if self.isSelected():
            padding = 3  # Pixels between node and selection box
            selection_rect = rect.adjusted(-padding, -padding, padding, padding)
            painter.setPen(QPen(QColor("#FFFFFF"), 3))  # 3px border
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(selection_rect, 4, 4)

        # Node name in header (warm white, uniform brightness - Neural Canvas style)
        painter.setPen(QColor("#e8e8e0"))  # Warm white
        font = QFont("Arial", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.drawText(
            header_rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            self.facet.name
        )

        # Facet type (same warm white, smaller - Neural Canvas style)
        painter.setPen(QColor("#e8e8e0"))  # Same brightness
        font = QFont("Arial", 7)
        painter.setFont(font)
        painter.drawText(
            header_rect.adjusted(8, 0, -8, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
            self.facet.facet_type
        )

        # Cycle ID badges (bottom-right) - stacked for multiple active cycles
        try:
            if self.active_cycles and self.execution_state in ("processing", "complete"):
                badge_font = QFont("Monospace", 7)
                badge_font.setBold(True)
                painter.setFont(badge_font)

                # Stack badges vertically (newest at bottom, oldest at top)
                badge_height = 14
                badge_spacing = 2
                # Copy list to avoid modification during iteration
                cycles_copy = list(self.active_cycles)
                for i, cycle_data in enumerate(cycles_copy):
                    if not cycle_data or len(cycle_data) < 2:
                        continue
                    cycle_id, cycle_color = cycle_data[0], cycle_data[1]
                    if not cycle_id:
                        continue
                    badge_text = str(cycle_id)[:8]  # First 8 chars of UUID

                    # Calculate Y position (stack upward from bottom)
                    stack_offset = (len(cycles_copy) - 1 - i) * (badge_height + badge_spacing)
                    badge_y = rect.height() - 16 - stack_offset

                    # Badge background (cycle color or default cyan)
                    badge_rect = QRectF(rect.width() - 60, badge_y, 56, badge_height)
                    painter.setBrush(QBrush(cycle_color or QColor("#00BFFF")))
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.drawRoundedRect(badge_rect, 3, 3)

                    # Badge text (black for contrast)
                    painter.setPen(QColor("#000000"))
                    painter.drawText(
                        badge_rect,
                        Qt.AlignmentFlag.AlignCenter,
                        badge_text
                    )
        except Exception as e:
            pass  # Don't crash paint on badge errors

    # Note: Move undo is handled via eventFilter in FacetsEditorPanel
    # The mouse press/release handlers on items don't reliably fire during Qt drag operations

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
            # Update connected wires during drag
            for pad_dict in [self.input_pads, self.output_pads]:
                for pad_graphics in pad_dict.values():
                    for wire in pad_graphics.connections:
                        wire.update_path()

            # Force scene repaint to prevent drag residue artifacts
            if self.scene():
                self.scene().update()

            # Note: Don't save to disk here - that happens in mouseReleaseEvent
            # when the move command is pushed

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # Trigger repaint to update selection highlight
            # Use prepareGeometryChange to ensure old selection area is invalidated
            self.prepareGeometryChange()
            self.update()

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
            # Monochrome aesthetic - keep base gray fill, pulse border only
            self.setBrush(self.base_brush)  # Keep normal gray fill
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
            # FLASHING red (emergency indicator) - industrial alarm style
            self.error_flash_count = 0
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._flash_error_border)
            self.animation_timer.start(100)  # 10Hz flash rate

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

    def _flash_error_border(self):
        """
        Flashing red border for error state.

        Industrial alarm style - 5 flashes then return to idle.
        """
        self.error_flash_count += 1

        # Alternate between dark red and bright red
        if self.error_flash_count % 2 == 0:
            # Bright flash
            self.setBrush(QBrush(QColor("#FF4444")))
            self.setPen(QPen(QColor("#FF0000"), 3))
        else:
            # Dark
            self.setBrush(QBrush(QColor("#8B0000")))
            self.setPen(QPen(QColor("#660000"), 2))

        # After 10 flashes (5 cycles), return to idle
        if self.error_flash_count >= 10:
            if self.animation_timer:
                self.animation_timer.stop()
                self.animation_timer = None
            self.error_flash_count = 0
            self.setBrush(self.base_brush)
            self.setPen(self.default_pen if not self.isSelected() else self.selected_pen)
            self.execution_state = "idle"

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
        # Right-click: don't handle here, let view's context menu handle it
        if event.button() == Qt.MouseButton.RightButton:
            event.ignore()  # Pass to parent/view for context menu
            return

        # On macOS, Qt uses Cmd for multi-select by default
        # Make Shift also work for additive selection
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
            # Shift pressed - toggle selection
            self.setSelected(not self.isSelected())
            event.accept()
        else:
            # Normal click
            super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        """
        Handle context menu events on the node itself.

        Accept the event to prevent Qt default behavior (which can cause deselection),
        then trigger the parent view's context menu.
        """
        # Accept to prevent Qt default handling that causes deselection
        event.accept()

        # Trigger the view's context menu at this position
        if self.scene() and self.scene().views():
            view = self.scene().views()[0]
            parent = view.parent()
            if hasattr(parent, 'show_context_menu'):
                # Convert scene position to view position
                view_pos = view.mapFromScene(event.scenePos())
                parent.show_context_menu(view_pos.toPoint())

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
            # Pads stay on left/right edges (no repositioning needed)

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

            # Restore normal size (calculated based on pads)
            header_height = 24
            port_start_y = header_height + 15
            port_spacing = 20
            max_pads = max(len(self.facet.input_pads), len(self.facet.output_pads), 1)
            min_height = self.NODE_HEIGHT_COMPACT if self.is_special_node else self.NODE_HEIGHT
            calculated_height = port_start_y + (max_pads * port_spacing) + 15
            target_height = max(min_height, calculated_height)
            self.setRect(0, 0, self.NODE_WIDTH, target_height)

            # Pads stay on left/right edges (no repositioning needed)
        except Exception as e:
            print(f"[Node] Error hiding fields: {e}")
            self.field_widgets.clear()

    def _reposition_pads_expanded(self):
        """No-op: Pads stay on left/right edges in horizontal flow."""
        pass

    def _reposition_pads_normal(self):
        """No-op: Pads stay on left/right edges in horizontal flow."""
        pass

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
            'inactive': '#666666',   # Dark gray - disabled/not running
            'ready': '#999999',      # Medium gray - ready to execute
            'processing': '#CCCCCC', # Light gray - LLM call in flight
            'waiting': '#555555',    # Darker gray - waiting for upstream inputs
            'cached': '#AAAAAA'      # Medium-light gray - using cached output
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

        # Update input pad color to match this wire
        self.to_pad.update_color_from_connection()

        # Visual styling (match Neural Canvas)
        self.pen = QPen(QColor("#888888"), 2.5)  # 2.5px like Neural Canvas
        self.active_pen = QPen(QColor("#CCAA00"), 2.5)  # Same width when animating
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
        """Draw the connection wire with Bezier curves (Blender-style)."""
        start = self.from_pad.get_scene_position()
        end = self.to_pad.get_scene_position()

        # Bezier curve routing (Blender-style)
        # Control points extend horizontally from pads
        distance = abs(end.x() - start.x())
        handle_distance = min(distance * 0.5, 100)  # Adaptive handle length

        control1 = QPointF(start.x() + handle_distance, start.y())  # Horizontal right
        control2 = QPointF(end.x() - handle_distance, end.y())      # Horizontal left

        path = QPainterPath()
        path.moveTo(start)
        path.cubicTo(control1, control2, end)  # Smooth Bezier curve

        # Wire color matches source facet's header color (coffee shop flow!)
        source_facet = self.from_pad.facet_node.facet
        wire_color = get_facet_header_color(source_facet)

        # Draw wire with antialiasing for smooth curves
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.packet_animating:
            # Brighten color when animating
            painter.setPen(QPen(QColor(wire_color).lighter(150), 2.5))
        else:
            painter.setPen(QPen(QColor(wire_color), 2.5))
        painter.drawPath(path)

        # Arrowhead at target (matches wire color)
        arrow_size = 6
        arrow_color = wire_color if not self.packet_animating else QColor(wire_color).lighter(150)
        painter.setBrush(QBrush(QColor(arrow_color)))
        arrow = QPainterPath()
        arrow.moveTo(end)
        arrow.lineTo(end.x() - arrow_size, end.y() - arrow_size / 2)
        arrow.lineTo(end.x() - arrow_size, end.y() + arrow_size / 2)
        arrow.closeSubpath()
        painter.drawPath(arrow)

        # Draw data packet (geometric square)
        if self.packet_animating and 0.0 <= self.packet_progress <= 1.0:
            # Calculate position along Bezier curve
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


def _log_facet(facet_name: str, event_type: str, cycle_id: str = "", details: str = ""):
    """
    Log a facet execution event to the FACETS console.

    Uses [FACET] marker for routing. Includes timestamp and cycle ID.

    Args:
        facet_name: Name of the facet
        event_type: Type of event (START, COMPLETE, ERROR, etc.)
        cycle_id: Cognitive cycle UUID (first 8 chars shown)
        details: Optional additional details
    """
    from datetime import datetime
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    cycle_str = f"[{cycle_id[:8]}]" if cycle_id else ""
    detail_str = f" - {details}" if details else ""
    print(f"[FACET] {timestamp} {cycle_str} {facet_name}: {event_type}{detail_str}")


class FacetsEditorPanel(QWidget):
    """
    Main facets editor panel with node graph.

    Provides visual editing of facet assemblies with drag-and-drop,
    connection wires, and right-click menus.
    """

    # Signal emitted when assembly is modified
    assemblyModified = pyqtSignal()

    # Signal emitted when a facet is selected (for Inspector)
    facetSelected = pyqtSignal(object)  # Emits Facet object

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_assembly: Optional[FacetAssembly] = None
        self.current_assembly_name: Optional[str] = None  # Track loaded assembly
        self.current_assembly_path: Optional[str] = None  # Track loaded assembly filepath
        self.node_graphics: Dict[str, FacetNodeGraphics] = {}
        self.wire_graphics: List[ConnectionWire] = []

        # CRITICAL: Lock to prevent event processing during scene transitions
        self.scene_transition_lock = False

        # Clipboard for copy/paste
        self.clipboard: List[Facet] = []

        # Track drag start positions for undo (facet_id -> (x, y))
        self.drag_start_positions: Dict[str, tuple] = {}

        # Space-drag navigation
        self.space_pressed = False

        # Wire drawing state
        self.wire_being_drawn: Optional[QGraphicsLineItem] = None
        self.wire_start_pad: Optional[FacetPadGraphics] = None

        # Grid snapping settings (load from persistent settings)
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'FacetsEditor')
        self.snap_to_grid = settings.value('grid/snap_enabled', False, type=bool)
        self.grid_size = settings.value('grid/size', 20, type=int)
        self.grid_visible = self.snap_to_grid  # Grid visible when snapping enabled
        self.grid_lines: List[QGraphicsLineItem] = []  # Track grid lines for removal

        # Cognition pause state
        self.current_agent_id: Optional[str] = None
        self.cognition_paused: bool = False

        # Right-click timestamp guard (prevents trackpad zoom quirk)
        self._last_right_click_time: float = 0.0
        self._in_right_click: bool = False  # Prevents selection changes during right-click
        self._selection_signal_connected: bool = True  # Track signal connection state
        self.api_base = "http://localhost:8081/api"

        # Focus state tracking (for F key toggle)
        self.is_focused = False
        self.pre_focus_transform = None
        self.focused_node_id = None

        # Empty state message
        self.empty_state_label: Optional[QGraphicsTextItem] = None

        # WebSocket connection for execution event streaming (AUTOBAHN!)
        self.ws_connection = None
        self.ws_task = None
        self.ws_connected = False
        self.event_queue = asyncio.Queue() if WEBSOCKETS_AVAILABLE else None

        # Cycle color tracking (for async cycle visualization)
        # Maps execution_id -> QColor for consistent coloring
        self.cycle_colors: Dict[str, QColor] = {}
        self.cycle_color_palette = [
            QColor("#00BFFF"),  # Deep sky blue
            QColor("#32CD32"),  # Lime green
            QColor("#FFD700"),  # Gold
            QColor("#FF69B4"),  # Hot pink
            QColor("#00CED1"),  # Dark turquoise
            QColor("#FFA500"),  # Orange
            QColor("#9370DB"),  # Medium purple
            QColor("#20B2AA"),  # Light sea green
        ]
        self.next_cycle_color_index = 0

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

        # Sound toggle button
        self.sound_enabled = True  # Sound on by default
        self.sound_button = QPushButton("🔊")
        self.sound_button.setFixedWidth(40)
        self.sound_button.setCheckable(True)
        self.sound_button.setChecked(True)  # On by default
        self.sound_button.setToolTip("Toggle execution sounds")
        self.sound_button.clicked.connect(self.toggle_sound)
        self.sound_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
        """)
        toolbar.addWidget(self.sound_button)

        # Grid snap toggle button
        self.grid_button = QPushButton("⊞")  # Grid icon
        self.grid_button.setFixedWidth(40)
        self.grid_button.setCheckable(True)
        self.grid_button.setChecked(self.snap_to_grid)  # Load from settings
        self.grid_button.setToolTip("Toggle grid snapping")
        self.grid_button.clicked.connect(self.toggle_grid_snap_button)
        self.grid_button.setStyleSheet("""
            QPushButton {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
            QPushButton:checked {
                background-color: #555555;
                color: #FFFFFF;
                border: 1px solid #888888;
            }
        """)
        toolbar.addWidget(self.grid_button)

        # Grid size input
        self.grid_size_input = QSpinBox()
        self.grid_size_input.setRange(5, 100)  # 5px to 100px
        self.grid_size_input.setValue(self.grid_size)  # Load from settings
        self.grid_size_input.setSuffix("px")
        self.grid_size_input.setFixedWidth(70)
        self.grid_size_input.setToolTip("Grid size in pixels")
        self.grid_size_input.valueChanged.connect(self.on_grid_size_changed)
        self.grid_size_input.setStyleSheet("""
            QSpinBox {
                background-color: #3A3A3A;
                color: #CCCCCC;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 2px;
            }
            QSpinBox:hover {
                background-color: #4A4A4A;
                border: 1px solid #777777;
            }
        """)
        toolbar.addWidget(self.grid_size_input)

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

        # Connect selection changes to inspector
        self.scene.selectionChanged.connect(self.on_selection_changed)

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

        # Restore grid if it was enabled
        if self.grid_visible:
            self._draw_grid_background()

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

    def load_assembly_from_data(self, assembly: FacetAssembly, force_reload: bool = False, source_path: Optional[str] = None):
        """
        Load a facet assembly into the editor.

        Args:
            assembly: FacetAssembly to load
            force_reload: If True, reload even if same assembly already loaded
            source_path: Optional path to source YAML file (for direct saves)
        """
        # CRITICAL: Prevent re-entrant calls during scene transition
        if self.scene_transition_lock:
            return

        # Check if this assembly is already loaded
        if not force_reload and self.current_assembly_name == assembly.name:
            return

        # CRITICAL: Lock BEFORE auto-save to prevent re-entrancy during YAML loading
        self.scene_transition_lock = True

        # Auto-save previous assembly before switching (if positions changed)
        if self.current_assembly and self.current_assembly_name:
            try:
                import os
                assembly_dir = os.path.join(os.path.dirname(__file__), '../facet_assemblies')
                for filename in os.listdir(assembly_dir):
                    if filename.endswith('.yaml'):
                        try:
                            from ..core.facet_system import FacetAssembly
                            test_path = os.path.join(assembly_dir, filename)
                            test_assembly = FacetAssembly.load_yaml(test_path)
                            if test_assembly.name == self.current_assembly_name:
                                self.current_assembly.save_yaml(test_path)
                                break
                        except:
                            pass
            except:
                pass  # Silent auto-save failure

        # Hide empty state message if showing
        self.hide_empty_state()

        self.current_assembly = assembly
        self.current_assembly_name = assembly.name
        self.current_assembly_path = source_path
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
        self.grid_lines.clear()  # Grid lines are also cleared by scene.clear()

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

        # Force scene update and ensure all items are visible
        self.scene.update()
        for node in self.node_graphics.values():
            node.update()

        # Restore grid if it was enabled
        if self.grid_visible:
            self._draw_grid_background()

        # Center view on content
        self.view.centerOn(500, 350)

        # Unlock scene - safe to process events now
        self.scene_transition_lock = False

    def show_context_menu(self, position):
        """Show right-click context menu for adding facets."""
        # Set flag to prevent selection changes during context menu
        self._in_right_click = True

        # Temporarily disconnect selection changed to prevent crashes
        if self._selection_signal_connected:
            try:
                self.scene.selectionChanged.disconnect(self.on_selection_changed)
                self._selection_signal_connected = False
            except:
                pass  # Already disconnected

        try:
            menu = QMenu(self)

            # Add facet submenu (excluding INCOMING/OUTGOING - those are auto-created)
            add_menu = menu.addMenu("Add Facet")

            facet_types = [
                ("Intuition Facet", "IntuitionFacet"),
                ("Emotion Facet", "EmotionFacet"),
                ("Social Context Facet", "SocialFacet"),
                ("Memory Recall Facet", "MemoryFacet"),
                ("Response Planning Facet", "PlanningFacet"),
                ("Convergence Facet", "ConvergenceFacet"),
                # NOTE: INCOMING/OUTGOING (SpecialNode) not shown - they're special
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

            # Layout menu
            menu.addSeparator()
            layout_menu = menu.addMenu("Layout")

            auto_arrange_action = layout_menu.addAction("Auto-Arrange (Topological)")
            auto_arrange_action.triggered.connect(self.auto_arrange_facets)

            layout_menu.addSeparator()

            # Alignment (requires selection)
            selected_nodes = self.scene.selectedItems()
            selected_facets = [item for item in selected_nodes if isinstance(item, FacetNodeGraphics)]

            align_h_action = layout_menu.addAction(f"Align Horizontally ({len(selected_facets)} selected)")
            align_h_action.setEnabled(len(selected_facets) > 1)
            align_h_action.triggered.connect(self.align_selected_horizontally)

            align_v_action = layout_menu.addAction(f"Align Vertically ({len(selected_facets)} selected)")
            align_v_action.setEnabled(len(selected_facets) > 1)
            align_v_action.triggered.connect(self.align_selected_vertically)

            layout_menu.addSeparator()

            # Zoom (use zoom_view to respect limits)
            zoom_in_action = layout_menu.addAction("Zoom In (+)")
            zoom_in_action.triggered.connect(lambda: self.zoom_view(1.2))

            zoom_out_action = layout_menu.addAction("Zoom Out (-)")
            zoom_out_action.triggered.connect(lambda: self.zoom_view(1/1.2))

            reset_zoom_action = layout_menu.addAction("Reset View")
            reset_zoom_action.triggered.connect(self.reset_view)

            # Delete (requires selection)
            if selected_facets:
                menu.addSeparator()
                delete_action = menu.addAction(f"Delete {len(selected_facets)} facet(s)")
                delete_action.triggered.connect(self.delete_selected_facets)

            menu.exec(self.view.mapToGlobal(position))

        except Exception as e:
            pass  # Silent context menu errors
        finally:
            # Always clear the flag when menu closes
            self._in_right_click = False
            # Reconnect selection changed signal
            if not self._selection_signal_connected:
                try:
                    self.scene.selectionChanged.connect(self.on_selection_changed)
                    self._selection_signal_connected = True
                except:
                    pass

    def add_facet(self, facet_type: str, display_name: str, position):
        """Add a new facet to the assembly (with undo support)."""
        if not self.current_assembly:
            return

        # Convert view position to scene position
        scene_pos = self.view.mapToScene(position)

        # Create new facet data (not added to assembly yet - command will do that)
        facet_id = Facet.generate_uuid()
        facet = Facet(
            id=facet_id,
            name=display_name,
            facet_type=facet_type,
            prompt=f"TODO: Define prompt for {display_name}",
            position={'x': scene_pos.x(), 'y': scene_pos.y()}
        )

        # Add default pads based on type
        if facet_type == "ConvergenceFacet":
            facet.add_input_pad("input1", "First input")
            facet.add_input_pad("input2", "Second input")
            facet.add_output_pad("output", "Merged output")
        else:
            facet.add_input_pad("in", "Input")
            facet.add_output_pad("out", "Output")

        # Push create command via UndoManager (command will create the facet)
        from ..core.undo_manager import undo_manager
        from ..core.commands import CreateFacetCommand

        cmd = CreateFacetCommand(
            editor=self,
            facet_data=facet.to_dict(),
            facet_name=display_name
        )
        undo_manager.push(cmd)

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

        # Additional zoom shortcuts (+/- keys)
        zoom_in_plus = QShortcut(QKeySequence("+"), self)
        zoom_in_plus.activated.connect(lambda: self.zoom_view(1.2))

        zoom_in_equals = QShortcut(QKeySequence("="), self)
        zoom_in_equals.activated.connect(lambda: self.zoom_view(1.2))

        zoom_out_shortcut = QShortcut(QKeySequence.StandardKey.ZoomOut, self)
        zoom_out_shortcut.activated.connect(lambda: self.zoom_view(1/1.2))

        zoom_out_minus = QShortcut(QKeySequence("-"), self)
        zoom_out_minus.activated.connect(lambda: self.zoom_view(1/1.2))

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
        import time

        # Guard: Ignore wheel events within 500ms of right-click (trackpad quirk)
        # On macOS, two-finger tap can trigger both right-click AND scroll simultaneously
        if time.time() - self._last_right_click_time < 0.5:
            event.ignore()
            return

        # Also ignore if we're in right-click mode
        if self._in_right_click:
            event.ignore()
            return

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
        self.frame_nodes(all_nodes, padding_factor=0.05)  # Tight framing like F key

    def focus_selection_tight(self):
        """
        Toggle tight focus on selected node (F key).

        First press: Zooms to selected node, saves view state
        Second press: Restores exact pre-focus view state

        NOTE: No longer shows inline field editors - use Inspector panel instead
        """
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            pass  # No facet selected
            return

        selected_node = selected_nodes[0]
        selected_node_id = selected_node.facet.id

        # Check if we're toggling focus on the same node
        if self.is_focused and self.focused_node_id == selected_node_id:
            # RESTORE: Pop back to pre-focus view
            if self.pre_focus_transform:
                self.view.setTransform(self.pre_focus_transform)
                pass  # Restored pre-focus view
            self.is_focused = False
            self.focused_node_id = None
            self.pre_focus_transform = None
        else:
            # FOCUS: Save current view and zoom to node
            self.pre_focus_transform = self.view.transform()
            self.focused_node_id = selected_node_id
            self.is_focused = True

            # Frame selected with minimal padding (no field display)
            self.frame_nodes(selected_nodes, padding_factor=0.05)

            pass  # Focused on node

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

    def copy_selection(self):
        """Copy selected facets to clipboard."""
        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
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

    def paste_selection(self):
        """Paste facets from clipboard with internal connections preserved."""
        if not self.clipboard or not self.current_assembly:
            pass  # Nothing to paste
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

        pass  # Paste complete
        self.assemblyModified.emit()

    def duplicate_selection(self):
        """Duplicate selected facets in place (Cmd-D)."""
        # Copy selection to clipboard
        self.copy_selection()
        # Immediately paste
        if self.clipboard:
            self.paste_selection()

    def delete_selection(self):
        """Delete selected facets (with undo support)."""
        if not self.current_assembly:
            return

        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, FacetNodeGraphics)
        ]

        if not selected_nodes:
            return

        # Filter out special nodes (can't delete)
        deletable_nodes = [
            node for node in selected_nodes
            if not node.is_special_node and node.facet.name not in ["INCOMING", "OUTGOING"]
        ]

        if not deletable_nodes:
            return

        # Push delete commands via UndoManager
        from ..core.undo_manager import undo_manager
        from ..core.commands import DeleteFacetCommand

        # Use macro for multiple deletions (single undo)
        if len(deletable_nodes) > 1:
            undo_manager.begin_group(f"Delete {len(deletable_nodes)} Facets")

        for node in deletable_nodes:
            # Collect connections involving this facet (for restoration on undo)
            connections_data = [
                c.to_dict() for c in self.current_assembly.connections
                if c.from_facet == node.facet.id or c.to_facet == node.facet.id
            ]

            # Push delete command
            cmd = DeleteFacetCommand(
                editor=self,
                facet_data=node.facet.to_dict(),
                connections_data=connections_data,
                facet_name=node.facet.name
            )
            undo_manager.push(cmd)

        if len(deletable_nodes) > 1:
            undo_manager.end_group()

    def undo(self):
        """Undo last operation via UndoManager."""
        from ..core.undo_manager import undo_manager
        undo_manager.undo()

    def redo(self):
        """Redo last undone operation via UndoManager."""
        from ..core.undo_manager import undo_manager
        undo_manager.redo()

    # ========== INTERNAL METHODS FOR UNDO COMMANDS ==========
    # These methods perform direct state changes without pushing commands.
    # They are called by command classes in undo/redo operations.

    def _set_facet_position_internal(self, facet_id: str, position: tuple):
        """
        Set facet position without pushing undo command.

        Called by MoveFacetCommand during undo/redo.
        """
        # Update data model
        facet = self.current_assembly.get_facet(facet_id) if self.current_assembly else None
        if facet:
            facet.position = {'x': position[0], 'y': position[1]}

        # Update graphics
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx:
            # Block signals to prevent recursive position saving
            node_gfx.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
            node_gfx.setPos(position[0], position[1])
            node_gfx.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

            # Update connected wires
            for pad_dict in [node_gfx.input_pads, node_gfx.output_pads]:
                for pad_graphics in pad_dict.values():
                    for wire in pad_graphics.connections:
                        wire.update_path()

        # Save to disk
        self._save_assembly_to_disk()

    def _create_facet_internal(self, facet_data: dict):
        """
        Create facet from serialized data without pushing undo command.

        Called by CreateFacetCommand.redo() and DeleteFacetCommand.undo().
        """
        if not self.current_assembly:
            return

        # Deserialize facet
        facet = Facet.from_dict(facet_data)

        # Add to assembly
        self.current_assembly.facets.append(facet)

        # Create graphics
        node = FacetNodeGraphics(facet, editor_panel=self)
        self.scene.addItem(node)
        self.node_graphics[facet.id] = node

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _delete_facet_internal(self, facet_id: str):
        """
        Delete facet by ID without pushing undo command.

        Called by DeleteFacetCommand.redo() and CreateFacetCommand.undo().
        """
        if not self.current_assembly:
            return

        # Remove from assembly
        self.current_assembly.facets = [
            f for f in self.current_assembly.facets if f.id != facet_id
        ]

        # Remove connections involving this facet
        self.current_assembly.connections = [
            c for c in self.current_assembly.connections
            if c.from_facet != facet_id and c.to_facet != facet_id
        ]

        # Remove wire graphics involving this facet
        wires_to_remove = []
        for wire in self.wire_graphics:
            if wire.from_pad.facet_node.facet.id == facet_id or \
               wire.to_pad.facet_node.facet.id == facet_id:
                wires_to_remove.append(wire)

        for wire in wires_to_remove:
            self.scene.removeItem(wire)
            self.wire_graphics.remove(wire)

        # Remove from scene
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx:
            self.scene.removeItem(node_gfx)
            del self.node_graphics[facet_id]

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _set_facet_property_internal(self, facet_id: str, prop_name: str, value):
        """
        Set facet property without pushing undo command.

        Called by EditFacetPropertyCommand and ToggleLockCommand.
        """
        facet = self.current_assembly.get_facet(facet_id) if self.current_assembly else None
        if not facet:
            return

        setattr(facet, prop_name, value)

        # Update graphics if needed (e.g., lock icon)
        node_gfx = self.node_graphics.get(facet_id)
        if node_gfx and prop_name == 'locked':
            node_gfx.lock_icon.setPlainText("[L]" if value else "")
            node_gfx.lock_icon.setDefaultTextColor(
                QColor("#CCAA00" if value else "#888888")
            )

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _create_connection_internal(self, conn_data: dict):
        """
        Create connection from serialized data without pushing undo command.

        Called by CreateConnectionCommand.redo() and DeleteConnectionCommand.undo().
        """
        if not self.current_assembly:
            return

        # Parse connection data
        from_parts = conn_data['from'].split('.')
        to_parts = conn_data['to'].split('.')
        from_facet = from_parts[0]
        from_pad = '.'.join(from_parts[1:])
        to_facet = to_parts[0]
        to_pad = '.'.join(to_parts[1:])

        # Create connection object
        conn = FacetConnection(from_facet, from_pad, to_facet, to_pad)

        # Add to assembly
        self.current_assembly.connections.append(conn)

        # Create wire graphics
        from_node = self.node_graphics.get(from_facet)
        to_node = self.node_graphics.get(to_facet)

        if from_node and to_node:
            from_pad_gfx = from_node.output_pads.get(from_pad)
            to_pad_gfx = to_node.input_pads.get(to_pad)

            if from_pad_gfx and to_pad_gfx:
                wire = ConnectionWire(from_pad_gfx, to_pad_gfx)
                self.scene.addItem(wire)
                self.wire_graphics.append(wire)

                # Register connections on pads
                from_pad_gfx.connections.append(wire)
                to_pad_gfx.connections.append(wire)
                to_pad_gfx.update_color_from_connection()

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _delete_connection_internal(self, from_facet: str, from_pad: str,
                                    to_facet: str, to_pad: str):
        """
        Delete connection without pushing undo command.

        Called by DeleteConnectionCommand.redo() and CreateConnectionCommand.undo().
        """
        if not self.current_assembly:
            return

        # Remove from assembly
        self.current_assembly.connections = [
            c for c in self.current_assembly.connections
            if not (c.from_facet == from_facet and c.from_pad == from_pad and
                    c.to_facet == to_facet and c.to_pad == to_pad)
        ]

        # Remove wire graphics
        wire_to_remove = None
        for wire in self.wire_graphics:
            if (wire.from_pad.facet_node.facet.id == from_facet and
                wire.from_pad.pad.name == from_pad and
                wire.to_pad.facet_node.facet.id == to_facet and
                wire.to_pad.pad.name == to_pad):
                wire_to_remove = wire
                break

        if wire_to_remove:
            # Unregister from pads
            if wire_to_remove in wire_to_remove.from_pad.connections:
                wire_to_remove.from_pad.connections.remove(wire_to_remove)
            if wire_to_remove in wire_to_remove.to_pad.connections:
                wire_to_remove.to_pad.connections.remove(wire_to_remove)
                wire_to_remove.to_pad.update_color_from_connection()

            self.scene.removeItem(wire_to_remove)
            self.wire_graphics.remove(wire_to_remove)

        # Save to disk
        self._save_assembly_to_disk()
        self.assemblyModified.emit()

    def _save_assembly_to_disk(self):
        """Save current assembly to disk (called by internal methods)."""
        if not self.current_assembly or not self.current_assembly_path:
            return

        try:
            import os
            if os.path.exists(self.current_assembly_path):
                self.current_assembly.save_yaml(self.current_assembly_path)
        except Exception:
            pass  # Silent save errors

    def _push_move_commands_if_needed(self):
        """
        Push move commands for nodes that have moved since drag started.

        Called from eventFilter on mouse release.
        """
        if not self.drag_start_positions:
            return

        from ..core.undo_manager import undo_manager
        from ..core.commands import MoveFacetCommand

        moved_nodes = []

        # Check which nodes actually moved
        for facet_id, old_pos in self.drag_start_positions.items():
            node_gfx = self.node_graphics.get(facet_id)
            if node_gfx:
                new_pos = (node_gfx.pos().x(), node_gfx.pos().y())

                # Only count as moved if position changed significantly
                if abs(new_pos[0] - old_pos[0]) > 1 or abs(new_pos[1] - old_pos[1]) > 1:
                    moved_nodes.append((facet_id, old_pos, new_pos, node_gfx.facet.name))

                    # Update facet data model
                    node_gfx.facet.position = {'x': new_pos[0], 'y': new_pos[1]}

        if not moved_nodes:
            return

        # Use macro for multiple moves
        if len(moved_nodes) > 1:
            undo_manager.begin_group(f"Move {len(moved_nodes)} Facets")

        for facet_id, old_pos, new_pos, facet_name in moved_nodes:
            cmd = MoveFacetCommand(
                editor=self,
                facet_id=facet_id,
                old_pos=old_pos,
                new_pos=new_pos,
                facet_name=facet_name
            )
            undo_manager.push(cmd)

        if len(moved_nodes) > 1:
            undo_manager.end_group()

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
        """Handle viewport events for wire drawing, selection, and undo tracking."""
        if obj == self.view.viewport():
            # Handle LEFT BUTTON clicks only for drag/selection
            if event.type() == event.Type.MouseButtonPress:
                # Skip right-click - let context menu handle it
                if event.button() == Qt.MouseButton.RightButton:
                    try:
                        import time
                        self._last_right_click_time = time.time()  # Record for wheel guard
                        # Set flag to prevent selection changes during right-click
                        self._in_right_click = True
                        # CRITICAL: Disconnect selection signal BEFORE Qt processes right-click
                        # This prevents crashes from selection changes during context menu
                        if self._selection_signal_connected:
                            try:
                                self.scene.selectionChanged.disconnect(self.on_selection_changed)
                                self._selection_signal_connected = False
                            except:
                                pass
                        # Reconnect after a delay (context menu uses exec which blocks)
                        QTimer.singleShot(500, self._reconnect_selection_signal)
                    except Exception as e:
                        pass  # Silent right-click errors
                    return False  # Pass through to context menu system

                # Only handle left button for dragging/selection
                if event.button() != Qt.MouseButton.LeftButton:
                    return False

                scene_pos = self.view.mapToScene(event.pos())
                items = self.scene.items(scene_pos)

                # Filter to only facet nodes
                clicked_nodes = [
                    item for item in items
                    if isinstance(item, FacetNodeGraphics)
                ]

                if clicked_nodes:
                    # Clicked on a node - record start positions for undo
                    # Record positions of all selected nodes (they may all move together)
                    self.drag_start_positions = {}
                    for item in self.scene.selectedItems():
                        if isinstance(item, FacetNodeGraphics):
                            self.drag_start_positions[item.facet.id] = (
                                item.pos().x(), item.pos().y()
                            )
                    # Also record clicked node if not already selected
                    for node in clicked_nodes:
                        if node.facet.id not in self.drag_start_positions:
                            self.drag_start_positions[node.facet.id] = (
                                node.pos().x(), node.pos().y()
                            )
                else:
                    # Clicked empty background
                    try:
                        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                            # Cmd-click - invert selection
                            self.invert_selection()
                            return True
                        else:
                            # Regular click - collapse any expanded nodes
                            self.collapse_all_nodes()
                            return False  # Allow default behavior (clear selection)
                    except Exception:
                        return False

            # Handle mouse release to push move commands
            elif event.type() == event.Type.MouseButtonRelease and not self.wire_being_drawn:
                if self.drag_start_positions:
                    self._push_move_commands_if_needed()
                    self.drag_start_positions = {}

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
                        pass  # Invalid connection

                # Clean up temporary wire
                self.scene.removeItem(self.wire_being_drawn)
                self.wire_being_drawn = None
                self.wire_start_pad = None
                return True

        return super().eventFilter(obj, event)

    def _clear_right_click_flag(self):
        """Clear the right-click flag after context menu closes."""
        self._in_right_click = False

    def _reconnect_selection_signal(self):
        """Reconnect selection changed signal after right-click handling."""
        self._in_right_click = False
        if not self._selection_signal_connected:
            try:
                self.scene.selectionChanged.connect(self.on_selection_changed)
                self._selection_signal_connected = True
            except Exception as e:
                pass  # Silent reconnection errors

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
        """Create a connection between two pads (with undo support)."""
        if not self.current_assembly:
            return

        # Push create connection command
        from ..core.undo_manager import undo_manager
        from ..core.commands import CreateConnectionCommand

        cmd = CreateConnectionCommand(
            editor=self,
            from_facet=from_pad.facet_node.facet.id,
            from_pad=from_pad.pad.name,
            to_facet=to_pad.facet_node.facet.id,
            to_pad=to_pad.pad.name
        )
        undo_manager.push(cmd)

        pass  # Connection created

    def invert_selection(self):
        """Invert current selection (ZBrush-style mask inverter)."""
        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, FacetNodeGraphics)
        ]

        for node in all_nodes:
            node.setSelected(not node.isSelected())

        selected_count = sum(1 for n in all_nodes if n.isSelected())
        pass  # Selection inverted

    def _draw_grid_background(self):
        """Draw subtle grid lines on background."""
        if not self.scene:
            return

        # Clear any existing grid first
        self._clear_grid_background()

        grid_size = self.grid_size
        scene_rect = self.scene.sceneRect()

        # Faint gray for grid lines
        grid_pen = QPen(QColor("#333333"), 1, Qt.PenStyle.DotLine)

        # Draw vertical lines
        x = scene_rect.left()
        while x <= scene_rect.right():
            if x % grid_size == 0:
                try:
                    line = self.scene.addLine(
                        x, scene_rect.top(),
                        x, scene_rect.bottom(),
                        grid_pen
                    )
                    line.setZValue(-100)  # Behind everything
                    self.grid_lines.append(line)
                except Exception:
                    break
            x += grid_size

        # Draw horizontal lines
        y = scene_rect.top()
        while y <= scene_rect.bottom():
            if y % grid_size == 0:
                try:
                    line = self.scene.addLine(
                        scene_rect.left(), y,
                        scene_rect.right(), y,
                        grid_pen
                    )
                    line.setZValue(-100)  # Behind everything
                    self.grid_lines.append(line)
                except Exception:
                    break
            y += grid_size

    def _clear_grid_background(self):
        """Remove grid lines from scene."""
        if not self.scene:
            self.grid_lines.clear()
            return

        # Safely remove each line
        for line in list(self.grid_lines):
            try:
                if line.scene() == self.scene:
                    self.scene.removeItem(line)
            except Exception:
                pass

        self.grid_lines.clear()

    def toggle_grid_snap_button(self):
        """Toggle grid snapping from toolbar button."""
        enabled = self.grid_button.isChecked()
        self.snap_to_grid = enabled
        self.grid_visible = enabled

        # Save to settings
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'FacetsEditor')
        settings.setValue('grid/snap_enabled', enabled)

        # Redraw grid
        if enabled:
            self._draw_grid_background()
        else:
            self._clear_grid_background()

    def toggle_grid_snap(self, enabled: bool):
        """Toggle grid snapping on/off (programmatic API)."""
        self.snap_to_grid = enabled
        self.grid_visible = enabled
        if hasattr(self, 'grid_button'):
            self.grid_button.setChecked(enabled)

    def on_grid_size_changed(self, value: int):
        """Handle grid size spinbox change."""
        self.grid_size = value

        # Save to settings
        from PyQt6.QtCore import QSettings
        settings = QSettings('Noodlings', 'FacetsEditor')
        settings.setValue('grid/size', value)

        # Redraw grid if visible
        if self.grid_visible:
            self._clear_grid_background()
            self._draw_grid_background()

    def set_grid_size(self, size: int):
        """Set grid snap size in pixels."""
        self.grid_size = size

    def collapse_all_nodes(self):
        """Collapse all expanded nodes (hide fields on all nodes)."""
        try:
            if not self.scene:
                return
            # Copy items list to avoid iteration issues
            items = list(self.scene.items())
            for item in items:
                if isinstance(item, FacetNodeGraphics):
                    try:
                        item.hide_fields()
                    except Exception:
                        pass
        except Exception:
            pass

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
                # PAUSING: Request immediate freeze (mid-cycle pause for debugging)
                url = f"{self.api_base}/cognition/pause"
                response = requests.post(url, json={
                    'paused': True,
                    'agent_id': self.current_agent_id,
                    'freeze_mode': 'immediate'  # Freeze mid-cycle for inspection
                }, timeout=5)

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

    def toggle_sound(self, checked: bool):
        """Toggle execution sound effects."""
        self.sound_enabled = checked
        if checked:
            self.sound_button.setText("🔊")
        else:
            self.sound_button.setText("🔇")

    def play_sound(self, sound_type: str):
        """
        Play sound effect for execution events.

        Args:
            sound_type: 'cycle_start', 'data_flow', 'cycle_complete'
        """
        if not self.sound_enabled:
            return

        try:
            from PyQt6.QtMultimedia import QSoundEffect
            from PyQt6.QtCore import QUrl
            import os

            # Map sound types to terminal beep files (Kraftwerk aesthetic!)
            resources_dir = os.path.join(os.path.dirname(__file__), '..', 'resources', 'terminal_beeps_hq')

            sound_files = {
                'cycle_start': 'termstart.ogg',          # High pitch - cycle begins!
                'data_flow': 'termkeypress.ogg',         # Quick click - data packet
                'cycle_complete': 'bell_vt100_250ms.ogg' # Bell chime - cycle ends
            }

            sound_file = sound_files.get(sound_type)
            if not sound_file:
                return

            sound_path = os.path.join(resources_dir, sound_file)
            if not os.path.exists(sound_path):
                return  # Sound file not found, silent fail

            # Create cached sound effect for this type
            cache_attr = f'_sound_{sound_type}'
            if not hasattr(self, cache_attr):
                sound_effect = QSoundEffect()
                sound_effect.setSource(QUrl.fromLocalFile(sound_path))

                # Volume settings - industrial precision
                volumes = {
                    'cycle_start': 0.5,    # Clear attention signal
                    'data_flow': 0.2,      # Quiet clicks (many events)
                    'cycle_complete': 0.4  # Satisfying closure
                }
                sound_effect.setVolume(volumes.get(sound_type, 0.3))
                setattr(self, cache_attr, sound_effect)

            # Play (non-blocking)
            sound_effect = getattr(self, cache_attr)
            sound_effect.play()

        except Exception as e:
            # Silent fail - don't break visualization if sound fails
            pass

    def auto_arrange_facets(self):
        """
        Auto-arrange facets using topological layering (circuit schematic style).

        Algorithm:
        1. Build dependency graph from connections
        2. Compute layers using topological sort (execution order)
        3. Position INCOMING at top, OUTGOING at bottom
        4. Distribute intermediate facets in layers
        5. Minimize wire crossings within each layer
        """
        if not self.current_assembly:
            return

        print("[Auto-Arrange] Starting topological layout...")

        # Build adjacency lists (who depends on whom)
        dependencies = {}  # facet_id -> list of facets it depends on (inputs from)
        dependents = {}    # facet_id -> list of facets that depend on it (outputs to)

        for facet in self.current_assembly.facets:
            dependencies[facet.id] = []
            dependents[facet.id] = []

        # Parse connections to build graph
        for conn in self.current_assembly.connections:
            from_id = conn.from_facet  # Source facet ID
            to_id = conn.to_facet      # Destination facet ID

            if from_id in dependencies and to_id in dependencies:
                if from_id not in dependencies[to_id]:  # Avoid duplicates
                    dependencies[to_id].append(from_id)
                if to_id not in dependents[from_id]:
                    dependents[from_id].append(to_id)

        # Topological sort to determine layers (Kahn's algorithm)
        layers = []
        in_degree = {fid: len(deps) for fid, deps in dependencies.items()}

        # Layer 0: Nodes with no dependencies (usually INCOMING)
        current_layer = [fid for fid, deg in in_degree.items() if deg == 0]

        while current_layer:
            layers.append(current_layer[:])
            next_layer = []

            for node_id in current_layer:
                # Remove this node from dependents' in-degree
                for dependent in dependents.get(node_id, []):
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        next_layer.append(dependent)

            current_layer = next_layer

        # Handle cycles (shouldn't happen in well-formed assemblies)
        remaining = [fid for fid, deg in in_degree.items() if deg > 0]
        if remaining:
            layers.append(remaining)
            print(f"[Auto-Arrange] Warning: Circular dependencies detected: {remaining}")

        print(f"[Auto-Arrange] Computed {len(layers)} layers: {[len(l) for l in layers]} facets")

        # Layout parameters (HORIZONTAL FLOW - left to right like Neural Canvas)
        layer_spacing = 300  # Horizontal spacing between layers
        node_spacing = 180   # Vertical spacing within layer
        start_x = 100        # Left margin
        start_y = 100        # Top margin

        # Position facets layer by layer (HORIZONTAL FLOW)
        for layer_idx, layer_facets in enumerate(layers):
            x = start_x + (layer_idx * layer_spacing)  # Horizontal progression

            for facet_idx, facet_id in enumerate(sorted(layer_facets)):
                y = start_y + (facet_idx * node_spacing)  # Vertical stacking within layer

                # Find graphics node and move it
                if facet_id in self.node_graphics:
                    node_gfx = self.node_graphics[facet_id]
                    node_gfx.setPos(x, y)
                    print(f"[Auto-Arrange] {facet_id}: ({x}, {y}) [Layer {layer_idx}]")

        # Update all wire paths
        try:
            for wire in self.wire_graphics:
                wire.update_path()
        except Exception as e:
            print(f"[Auto-Arrange] Warning: Could not update wires: {e}")

        # Save new positions
        try:
            self.save_current_assembly_positions()
        except Exception as e:
            print(f"[Auto-Arrange] Warning: Could not save positions: {e}")

        print("[Auto-Arrange] Layout complete!")

        # Frame all nodes to show the result
        self.frame_all()

    def align_selected_horizontally(self):
        """Align selected facets to same Y coordinate (horizontal line)."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if len(selected) < 2:
            return

        # Use average Y position
        avg_y = sum(node.pos().y() for node in selected) / len(selected)

        for node in selected:
            node.setPos(node.pos().x(), avg_y)

        # Update wires
        for wire in self.wire_graphics:
            wire.update_path()

        self.save_current_assembly_positions()
        print(f"[Align] Aligned {len(selected)} facets horizontally at y={avg_y:.0f}")

    def align_selected_vertically(self):
        """Align selected facets to same X coordinate (vertical line)."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if len(selected) < 2:
            return

        # Use average X position
        avg_x = sum(node.pos().x() for node in selected) / len(selected)

        for node in selected:
            node.setPos(avg_x, node.pos().y())

        # Update wires
        for wire in self.wire_graphics:
            wire.update_path()

        self.save_current_assembly_positions()
        print(f"[Align] Aligned {len(selected)} facets vertically at x={avg_x:.0f}")

    def delete_selected_facets(self):
        """Delete selected facets from the assembly."""
        selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]
        if not selected:
            return

        # Confirm deletion
        from PyQt6.QtWidgets import QMessageBox
        facet_names = [node.facet.name for node in selected]
        reply = QMessageBox.question(
            self,
            "Delete Facets",
            f"Delete {len(selected)} facet(s)?\n\n" + "\n".join(f"- {name}" for name in facet_names),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply != QMessageBox.StandardButton.Yes:
            return

        # Remove from assembly and scene
        for node in selected:
            facet_id = node.facet.id

            # Remove connected wires
            wires_to_remove = [w for w in self.wire_graphics if w.from_pad.facet_node == node or w.to_pad.facet_node == node]
            for wire in wires_to_remove:
                # Remove wire from pad connection lists
                if wire in wire.from_pad.connections:
                    wire.from_pad.connections.remove(wire)
                if wire in wire.to_pad.connections:
                    wire.to_pad.connections.remove(wire)

                # Update input pad color (will revert to neutral or another connection's color)
                wire.to_pad.update_color_from_connection()

                # Remove from scene and list
                self.scene.removeItem(wire)
                self.wire_graphics.remove(wire)

            # Remove connections from assembly
            self.current_assembly.connections = [
                conn for conn in self.current_assembly.connections
                if conn.from_facet != facet_id and conn.to_facet != facet_id
            ]

            # Remove facet from assembly
            self.current_assembly.facets = [f for f in self.current_assembly.facets if f.id != facet_id]

            # Remove from graphics
            self.scene.removeItem(node)
            if facet_id in self.node_graphics:
                del self.node_graphics[facet_id]

        self.save_current_assembly_positions()
        print(f"[Delete] Removed {len(selected)} facet(s)")

    def on_selection_changed(self):
        """Handle facet selection changes - emit signal for Inspector."""
        # Guard: Don't process selection changes during scene transitions
        if self.scene_transition_lock:
            return

        # Guard: Don't process during right-click (causes crashes)
        if hasattr(self, '_in_right_click') and self._in_right_click:
            return

        try:
            selected = [item for item in self.scene.selectedItems() if isinstance(item, FacetNodeGraphics)]

            if len(selected) == 1:
                # Single facet selected - send to Inspector
                facet = selected[0].facet
                self.facetSelected.emit(facet)
            elif len(selected) == 0:
                # No selection - clear Inspector
                self.facetSelected.emit(None)
            else:
                # Multiple selection - Inspector shows count
                self.facetSelected.emit(None)
        except Exception as e:
            print(f"[Facets Editor] Selection error: {e}")

    def save_current_assembly_positions(self):
        """Save current assembly node positions to disk."""
        if not self.current_assembly or not self.current_assembly_name:
            return

        try:
            import os
            # Update facet positions from graphics
            updated_count = 0
            for facet_id, node_gfx in self.node_graphics.items():
                pos = node_gfx.pos()
                facet = next((f for f in self.current_assembly.facets if f.id == facet_id), None)
                if facet:
                    old_pos = facet.position
                    facet.position = {'x': pos.x(), 'y': pos.y()}
                    if old_pos != facet.position:
                        updated_count += 1

            if updated_count == 0:
                return

            # Try direct path first (fastest, most reliable)
            if self.current_assembly_path and os.path.exists(self.current_assembly_path):
                # Get file mtime before save for verification
                import time
                mtime_before = os.path.getmtime(self.current_assembly_path)
                self.current_assembly.save_yaml(self.current_assembly_path)
                mtime_after = os.path.getmtime(self.current_assembly_path)
                if mtime_after <= mtime_before:
                    print(f"[Facets Editor] Warning: File mtime unchanged - save may have failed!")
                return

            # Fallback: Search by assembly name
            assembly_dir = os.path.join(os.path.dirname(__file__), '../facet_assemblies')

            for filename in os.listdir(assembly_dir):
                if filename.endswith('.yaml'):
                    try:
                        from ..core.facet_system import FacetAssembly
                        test_path = os.path.join(assembly_dir, filename)
                        test_assembly = FacetAssembly.load_yaml(test_path)
                        if test_assembly.name == self.current_assembly_name:
                            # Found it! Save positions
                            self.current_assembly.save_yaml(test_path)
                            return
                    except Exception as e:
                        pass
            print(f"[Facets Editor] Could not find YAML file for assembly '{self.current_assembly_name}'")
        except Exception as e:
            print(f"[Facets Editor] Error saving positions: {e}")

    def set_current_agent(self, agent_id: str):
        """
        Set the current agent whose facets are being edited.

        Fetches and loads the agent's facet assembly from the API.
        """
        # Save previous agent's positions before switching
        if self.current_agent_id and self.current_agent_id != agent_id:
            self.save_current_assembly_positions()

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

                    # Handle both string and dict formats
                    if isinstance(facet_assembly_ref, dict):
                        facet_assembly_ref = facet_assembly_ref.get('ref')

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
                        else:
                            print(f"[Facets Editor] Assembly file not found: {assembly_path}")
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
            return

        # Start event processing timer (polls queue from Qt thread)
        self.event_timer = QTimer()
        self.event_timer.timeout.connect(self._process_event_queue)
        self.event_timer.start(16)  # 60fps event processing

        # Start WebSocket task in separate thread
        import threading
        ws_thread = threading.Thread(target=self._run_websocket_loop, daemon=True)
        ws_thread.start()

    def _run_websocket_loop(self):
        """Run WebSocket event loop in separate thread."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._websocket_handler())
        except Exception:
            pass  # WebSocket errors handled in _websocket_handler

    async def _websocket_handler(self):
        """Handle WebSocket connection and message receiving."""
        uri = "ws://localhost:8081/ws/execution_events"

        while True:
            try:
                async with websockets.connect(uri) as websocket:
                    self.ws_connection = websocket
                    self.ws_connected = True

                    async for message in websocket:
                        try:
                            event_data = json.loads(message)
                            # Add to queue for Qt thread processing
                            if self.event_queue:
                                await self.event_queue.put(event_data)
                        except json.JSONDecodeError:
                            pass  # Ignore malformed messages

            except Exception:
                self.ws_connected = False
                await asyncio.sleep(5)  # Reconnect delay

    def _process_event_queue(self):
        """Process execution events from queue (called from Qt timer)."""
        if not self.event_queue:
            return

        # Process up to 10 events per frame (prevent UI blocking)
        import asyncio
        for _ in range(10):
            try:
                event = self.event_queue.get_nowait()
                self._handle_execution_event(event)
            except asyncio.QueueEmpty:
                break  # Queue empty - expected
            except Exception as e:
                _log_facet("EventProcessor", "ERROR", "", str(e))
                break

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

        # Handle cycle-level events (no facet_id)
        if event_subtype == 'cycle_start':
            self.play_sound('cycle_start')
            return
        elif event_subtype == 'cycle_complete':
            self.play_sound('cycle_complete')
            # Clean up cycle color for completed cycle (prevent memory leak)
            execution_id = event.get('data', {}).get('execution_id', '')
            if not execution_id:
                execution_id = event.get('execution_id', '')
            if execution_id and execution_id in self.cycle_colors:
                del self.cycle_colors[execution_id]
            return

        # Handle data_flow events separately (they have from_facet/to_facet, not source_id)
        if event_subtype == 'data_flow':
            from_facet = event.get('from_facet')
            to_facet = event.get('to_facet')

            if from_facet and to_facet:
                # Play data flow sound
                self.play_sound('data_flow')

                # Find connection wire between these facets
                try:
                    for wire in list(self.wire_graphics):
                        if not wire or not wire.scene():
                            continue  # Wire was deleted, skip
                        if not hasattr(wire, 'from_pad') or not hasattr(wire, 'to_pad'):
                            continue  # Invalid wire state
                        if not wire.from_pad or not wire.to_pad:
                            continue
                        if not hasattr(wire.from_pad, 'facet_node') or not hasattr(wire.to_pad, 'facet_node'):
                            continue
                        if (wire.from_pad.facet_node.facet.id == from_facet and
                            wire.to_pad.facet_node.facet.id == to_facet):
                            wire.animate_data_flow()
                            break
                except Exception:
                    pass  # Silent data flow animation errors
            return  # data_flow handled, exit

        facet_id = event.get('source_id')
        if not facet_id or facet_id not in self.node_graphics:
            return  # Facet not in current assembly (normal during transitions)

        node = self.node_graphics.get(facet_id)
        if not node:
            return  # Node was deleted (race condition during scene transition)

        # CRITICAL: Check if node is still in scene (not deleted)
        if not node.scene():
            return  # Node removed from scene, skip event

        # KRAFTWERK CLICK - Play terminal keypress sound for every event
        self._play_pachinko_sound()

        # Extract execution_id for cycle tracking
        execution_id = event.get('data', {}).get('execution_id', '')
        if not execution_id:
            execution_id = event.get('execution_id', '')

        try:
            # Get facet name for logging
            facet_name = node.facet.name if node.facet else facet_id

            if event_subtype == 'facet_start':
                # KRAFTWERK: Node begins processing
                # Assign cycle color if not already assigned
                if execution_id and execution_id not in self.cycle_colors:
                    self.cycle_colors[execution_id] = self.cycle_color_palette[
                        self.next_cycle_color_index % len(self.cycle_color_palette)
                    ]
                    self.next_cycle_color_index += 1

                # Capture inputs for inspection (debugging feature)
                event_data = event.get('data', {})
                inputs = event_data.get('inputs')
                if inputs:
                    node.last_inputs = inputs
                    # Store per-cycle inputs for inspection
                    if execution_id:
                        if execution_id not in node.cycle_data:
                            node.cycle_data[execution_id] = {}
                        node.cycle_data[execution_id]['inputs'] = inputs

                # Add this cycle to active_cycles list (supports stacking!)
                if execution_id:
                    cycle_color = self.cycle_colors.get(execution_id, QColor("#00BFFF"))
                    # Check if this cycle is already active on this node (avoid duplicates)
                    existing_ids = [c[0] for c in node.active_cycles]
                    if execution_id not in existing_ids:
                        node.active_cycles.append((execution_id, cycle_color, inputs))

                # Log to FACETS console
                input_keys = list(inputs.keys()) if inputs else []
                _log_facet(facet_name, "START", execution_id, f"inputs: {input_keys}")

                node.set_execution_state('processing')
                node.update()  # Force repaint to show cycle badge

            elif event_subtype == 'facet_complete':
                # KRAFTWERK: Node completes (brief satisfaction, then idle)
                # Capture outputs for inspection (debugging feature)
                event_data = event.get('data', {})
                outputs = event_data.get('outputs')
                if outputs:
                    node.last_outputs = outputs
                    # Store per-cycle outputs for inspection
                    if execution_id:
                        if execution_id not in node.cycle_data:
                            node.cycle_data[execution_id] = {}
                        node.cycle_data[execution_id]['outputs'] = outputs

                # Log to FACETS console
                output_keys = list(outputs.keys()) if outputs else []
                _log_facet(facet_name, "COMPLETE", execution_id, f"outputs: {output_keys}")

                node.set_execution_state('complete')
                node.update()

                # Remove this cycle from active_cycles after animation completes
                captured_exec_id = execution_id  # Capture for closure
                def clear_cycle_from_list():
                    if node and node.scene():
                        # Remove only the completed cycle from the list
                        node.active_cycles = [c for c in node.active_cycles if c[0] != captured_exec_id]
                        # Clean up cycle_data after a delay (keep for inspection during pause)
                        if not self.cognition_paused:
                            if captured_exec_id in node.cycle_data:
                                del node.cycle_data[captured_exec_id]
                        node.update()
                QTimer.singleShot(300, clear_cycle_from_list)

            elif event_subtype == 'facet_error':
                # ERROR: Something went wrong - flash red
                error_msg = event.get('data', {}).get('error', 'Unknown error')
                _log_facet(facet_name, "ERROR", execution_id, error_msg)
                node.set_execution_state('error')
                node.update()

            elif event_subtype == 'quantum_collapse':
                # QUANTUM: Orchestrated objective reduction event
                _log_facet(facet_name, "QUANTUM_COLLAPSE", execution_id)
                node.set_execution_state('quantum_collapse')
                self._play_quantum_collapse_sound()

        except Exception as e:
            _log_facet(facet_name if 'facet_name' in dir() else facet_id, "ANIMATION_ERROR", "", str(e))

    def _play_pachinko_sound(self):
        """Play termkeypress.ogg sound (Kraftwerk pachinko click)."""
        if not self.sound_enabled:
            return

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
        if not self.sound_enabled:
            return

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
