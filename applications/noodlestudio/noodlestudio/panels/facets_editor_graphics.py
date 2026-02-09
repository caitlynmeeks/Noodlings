# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Facets Editor Graphics - Visual components for node graph editor
#
#   Graphics item classes for the facets editor: - ClickableT...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.facets_editor_graphics
# PURPOSE:  facets editor graphics facet implementation
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ClickableTextItem, FacetPadGraphics, FacetNodeGraphics, ConnectionWire, get_facet_header_color()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QGraphicsEllipseItem
)
from PyQt6.QtCore import Qt, QRectF, QPointF, QTimer
from PyQt6.QtGui import (
    QPen, QBrush, QColor, QPainter, QFont, QPainterPath
)
from typing import Optional, List, Dict, Tuple, Any

from ..core.facet_system import Facet, FacetPad, PadType


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
    elif "NeuralCanvas" in facet_type:
        return "#7B1FA2"  # Purple (visual neural network from NNCanvas)
    elif "Transformer" in facet_type:
        return "#6A4A8A"  # Purple-mauve (attention-based neural)
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
                # Import here to avoid circular import
                from .facets_editor_panel import FacetsEditorPanel
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
        except Exception:
            pass  # Don't crash paint on badge errors

    def itemChange(self, change, value):
        """Handle item changes (e.g., position updates, selection)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Apply grid snapping if enabled
            new_pos = value
            views = self.scene().views() if self.scene() else []
            if views and hasattr(views[0].parent(), 'snap_to_grid'):
                editor = views[0].parent()
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

        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # Trigger repaint to update selection highlight
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

        y_offset = 45  # Start below title/type
        for field in fields:
            # Single line: "FIELD NAME: preview text..." pencil
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
            pencil = ClickableTextItem("\u270E", self, field, self.open_field_editor)
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
            # Import here to avoid circular import
            from .facets_editor_panel import FacetsEditorPanel
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
            from PyQt6.QtCore import QRectF
            painter.drawRect(QRectF(
                packet_pos.x() - packet_size / 2,
                packet_pos.y() - packet_size / 2,
                packet_size,
                packet_size
            ))

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
