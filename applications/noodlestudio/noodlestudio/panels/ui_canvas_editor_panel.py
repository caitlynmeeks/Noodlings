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
#   UI Canvas Editor Panel - Visual UI Designer (Delphi-style)
#
#   The center pane design surface for building application i...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.ui_canvas_editor_panel
# PURPOSE:  ui canvas editor panel panel UI
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   ResizeHandle, ComponentGraphicsItem, UICanvasView, UICanvasEditorPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Dict, List, Optional, Set
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsRectItem, QGraphicsTextItem, QLabel,
    QToolBar, QToolButton, QSpinBox, QCheckBox, QSizePolicy
)
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal, QMimeData
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QDragEnterEvent,
    QDropEvent, QWheelEvent, QMouseEvent, QKeyEvent, QTransform
)

from ..runtime.ui.component import UIComponent, get_component_class, list_component_types
from ..runtime.ui.loader import UILoader


# Grid snap size in pixels
GRID_SIZE = 8

# Handle size for selection
HANDLE_SIZE = 8

# Component colors (monochromatic)
COMPONENT_COLORS = {
    "Panel": "#2a2a2a",
    "Label": "#3a3a3a",
    "Button": "#4a4a4a",
    "TextInput": "#3a3a3a",
    "ChatHistory": "#333333",
    "ChatInput": "#3a3a3a",
    "RadianceViewport": "#1a1a1a",
    "FacetAssembly": "#3a3a4a",  # Slight blue tint for logic components
}


class ResizeHandle(QGraphicsRectItem):
    """
    A resize handle for selected components.

    8 handles around the component: corners + midpoints
    """

    # Handle positions
    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM = 5
    BOTTOM_LEFT = 6
    LEFT = 7

    def __init__(self, position: int, parent: 'ComponentGraphicsItem'):
        super().__init__(parent)
        self.position = position
        self.setRect(-HANDLE_SIZE/2, -HANDLE_SIZE/2, HANDLE_SIZE, HANDLE_SIZE)
        self.setBrush(QBrush(QColor("#ffffff")))
        self.setPen(QPen(QColor("#333333"), 1))
        self.setZValue(100)
        self.setCursor(self._get_cursor())
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)

    def _get_cursor(self) -> Qt.CursorShape:
        """Get appropriate cursor for this handle position."""
        cursors = {
            self.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            self.TOP: Qt.CursorShape.SizeVerCursor,
            self.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            self.RIGHT: Qt.CursorShape.SizeHorCursor,
            self.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            self.BOTTOM: Qt.CursorShape.SizeVerCursor,
            self.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            self.LEFT: Qt.CursorShape.SizeHorCursor,
        }
        return cursors.get(self.position, Qt.CursorShape.ArrowCursor)

    def update_position(self, rect: QRectF):
        """Update handle position based on parent rect."""
        positions = {
            self.TOP_LEFT: QPointF(rect.left(), rect.top()),
            self.TOP: QPointF(rect.center().x(), rect.top()),
            self.TOP_RIGHT: QPointF(rect.right(), rect.top()),
            self.RIGHT: QPointF(rect.right(), rect.center().y()),
            self.BOTTOM_RIGHT: QPointF(rect.right(), rect.bottom()),
            self.BOTTOM: QPointF(rect.center().x(), rect.bottom()),
            self.BOTTOM_LEFT: QPointF(rect.left(), rect.bottom()),
            self.LEFT: QPointF(rect.left(), rect.center().y()),
        }
        pos = positions.get(self.position, QPointF())
        self.setPos(pos)


class ComponentGraphicsItem(QGraphicsRectItem):
    """
    Visual representation of a UIComponent on the canvas.

    Features:
    - Displays component type and name
    - 8-point resize handles when selected
    - Drag to move, handles to resize
    - Snaps to grid
    """

    def __init__(self, component: UIComponent, parent: Optional[QGraphicsItem] = None):
        super().__init__(parent)
        self.component = component
        self.handles: List[ResizeHandle] = []
        self._resize_start_rect: Optional[QRectF] = None
        self._resize_handle: Optional[ResizeHandle] = None
        self._move_start_pos: Optional[QPointF] = None

        # Setup item
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)

        # Visual setup
        color = COMPONENT_COLORS.get(component.component_type, "#3a3a3a")
        self.setBrush(QBrush(QColor(color)))
        self.setPen(QPen(QColor("#666666"), 1))

        # Label
        self.label = QGraphicsTextItem(self)
        self.label.setDefaultTextColor(QColor("#cccccc"))
        font = QFont("Arial", 10)
        self.label.setFont(font)
        self._update_label()

        # Create resize handles (hidden by default)
        self._create_handles()
        self._update_handles_visibility()

        # Sync geometry from component
        self.sync_from_component()

    def _create_handles(self):
        """Create the 8 resize handles."""
        for i in range(8):
            handle = ResizeHandle(i, self)
            handle.hide()
            self.handles.append(handle)

    def _update_handles_visibility(self):
        """Show/hide handles based on selection state."""
        visible = self.isSelected()
        for handle in self.handles:
            handle.setVisible(visible)

    def _update_handles_positions(self):
        """Update handle positions to match current rect."""
        rect = self.rect()
        for handle in self.handles:
            handle.update_position(rect)

    def _update_label(self):
        """Update the label text."""
        text = f"{self.component.component_type}"
        if self.component.name:
            text += f": {self.component.name}"
        self.label.setPlainText(text)
        # Center label in component
        label_rect = self.label.boundingRect()
        self.label.setPos(
            (self.rect().width() - label_rect.width()) / 2,
            (self.rect().height() - label_rect.height()) / 2
        )

    def sync_from_component(self):
        """Sync graphics item geometry from UIComponent."""
        geom = self.component.geometry
        self.setPos(geom.x, geom.y)
        self.setRect(0, 0, geom.width, geom.height)
        self._update_label()
        self._update_handles_positions()

    def sync_to_component(self):
        """Sync UIComponent geometry from graphics item."""
        pos = self.pos()
        rect = self.rect()
        self.component.geometry.x = int(pos.x())
        self.component.geometry.y = int(pos.y())
        self.component.geometry.width = int(rect.width())
        self.component.geometry.height = int(rect.height())

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value):
        """Handle item changes (selection, position)."""
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self._update_handles_visibility()
            # Update selection pen
            if value:
                self.setPen(QPen(QColor("#4a9eff"), 2))
            else:
                self.setPen(QPen(QColor("#666666"), 1))

        elif change == QGraphicsItem.GraphicsItemChange.ItemPositionChange:
            # Grid snap
            new_pos = value
            if isinstance(new_pos, QPointF):
                snapped_x = round(new_pos.x() / GRID_SIZE) * GRID_SIZE
                snapped_y = round(new_pos.y() / GRID_SIZE) * GRID_SIZE
                return QPointF(snapped_x, snapped_y)

        return super().itemChange(change, value)

    def mousePressEvent(self, event):
        """Handle mouse press for resize detection."""
        # Check if clicking on a handle
        for handle in self.handles:
            if handle.isVisible() and handle.contains(handle.mapFromParent(event.pos())):
                self._resize_handle = handle
                self._resize_start_rect = self.rect()
                self._move_start_pos = event.scenePos()
                event.accept()
                return

        # Normal move
        self._move_start_pos = self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for resize."""
        if self._resize_handle and self._resize_start_rect:
            delta = event.scenePos() - self._move_start_pos
            self._do_resize(delta)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if self._resize_handle:
            self._resize_handle = None
            self._resize_start_rect = None
            self.sync_to_component()
            # Notify scene of change
            if self.scene():
                view = self.scene().views()[0] if self.scene().views() else None
                if view and hasattr(view, 'canvas_modified'):
                    view.canvas_modified.emit()
        else:
            super().mouseReleaseEvent(event)
            self.sync_to_component()
            if self.scene():
                view = self.scene().views()[0] if self.scene().views() else None
                if view and hasattr(view, 'canvas_modified'):
                    view.canvas_modified.emit()

    def _do_resize(self, delta: QPointF):
        """Perform resize based on handle being dragged."""
        if not self._resize_start_rect:
            return

        rect = QRectF(self._resize_start_rect)
        pos = self.pos()
        handle_pos = self._resize_handle.position

        # Snap delta to grid
        dx = round(delta.x() / GRID_SIZE) * GRID_SIZE
        dy = round(delta.y() / GRID_SIZE) * GRID_SIZE

        # Minimum size
        min_size = GRID_SIZE * 2

        # Apply resize based on handle position
        if handle_pos in (ResizeHandle.TOP_LEFT, ResizeHandle.LEFT, ResizeHandle.BOTTOM_LEFT):
            new_width = rect.width() - dx
            if new_width >= min_size:
                rect.setWidth(new_width)
                self.setPos(pos.x() + dx, pos.y())

        if handle_pos in (ResizeHandle.TOP_RIGHT, ResizeHandle.RIGHT, ResizeHandle.BOTTOM_RIGHT):
            new_width = rect.width() + dx
            if new_width >= min_size:
                rect.setWidth(new_width)

        if handle_pos in (ResizeHandle.TOP_LEFT, ResizeHandle.TOP, ResizeHandle.TOP_RIGHT):
            new_height = rect.height() - dy
            if new_height >= min_size:
                rect.setHeight(new_height)
                self.setPos(self.pos().x(), pos.y() + dy)

        if handle_pos in (ResizeHandle.BOTTOM_LEFT, ResizeHandle.BOTTOM, ResizeHandle.BOTTOM_RIGHT):
            new_height = rect.height() + dy
            if new_height >= min_size:
                rect.setHeight(new_height)

        self.setRect(0, 0, rect.width(), rect.height())
        self._update_label()
        self._update_handles_positions()


class UICanvasView(QGraphicsView):
    """
    The design surface view.

    Features:
    - Grid background
    - Zoom with wheel
    - Pan with middle mouse or space+drag
    - Rubber-band multi-select
    - Drop from component palette
    """

    # Signals
    component_selected = pyqtSignal(object)  # UIComponent or None
    components_selected = pyqtSignal(list)   # List of UIComponents (multi-select)
    canvas_modified = pyqtSignal()           # UI changed, needs save

    def __init__(self, parent=None):
        super().__init__(parent)

        # Scene
        self.canvas_scene = QGraphicsScene(self)
        self.canvas_scene.setSceneRect(-2000, -2000, 4000, 4000)
        self.setScene(self.canvas_scene)

        # View settings
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setBackgroundBrush(QBrush(QColor("#1e1e1e")))

        # Accept drops
        self.setAcceptDrops(True)

        # State
        self.space_pressed = False
        self.panning = False
        self.last_pan_pos = QPointF()

        # Focus state (F key toggle)
        self.is_focused = False
        self.pre_focus_transform: Optional[QTransform] = None
        self.focused_component_ids: Optional[tuple] = None

        # Component tracking
        self.component_items: Dict[str, ComponentGraphicsItem] = {}
        self.root_component: Optional[UIComponent] = None
        self.ui_file_path: Optional[Path] = None

        # Connect selection change
        self.canvas_scene.selectionChanged.connect(self._on_selection_changed)

    def drawBackground(self, painter: QPainter, rect: QRectF):
        """Draw grid background."""
        super().drawBackground(painter, rect)

        # Draw grid
        grid_pen = QPen(QColor("#2a2a2a"), 1)
        painter.setPen(grid_pen)

        left = int(rect.left()) - (int(rect.left()) % GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % GRID_SIZE)

        # Limit grid lines to avoid performance issues
        visible_rect = self.mapToScene(self.viewport().rect()).boundingRect()

        x = left
        while x < rect.right():
            if visible_rect.left() <= x <= visible_rect.right():
                painter.drawLine(int(x), int(rect.top()), int(x), int(rect.bottom()))
            x += GRID_SIZE

        y = top
        while y < rect.bottom():
            if visible_rect.top() <= y <= visible_rect.bottom():
                painter.drawLine(int(rect.left()), int(y), int(rect.right()), int(y))
            y += GRID_SIZE

    def load_ui(self, path: Path):
        """Load a ui.yaml file into the canvas."""
        self.ui_file_path = path
        loader = UILoader()
        self.root_component = loader.load_file(path)
        self._rebuild_scene()

    def load_component(self, root: UIComponent):
        """Load a component tree directly."""
        self.root_component = root
        self._rebuild_scene()

    def _rebuild_scene(self):
        """Rebuild scene from component tree."""
        self.canvas_scene.clear()
        self.component_items.clear()

        if not self.root_component:
            return

        # Add all components recursively
        self._add_component_to_scene(self.root_component)

        # Frame all
        self.frame_all()

    def _add_component_to_scene(self, component: UIComponent, parent_item: Optional[ComponentGraphicsItem] = None):
        """Add a component and its children to the scene."""
        item = ComponentGraphicsItem(component)

        if parent_item:
            # Child components are positioned relative to parent
            item.setParentItem(parent_item)
        else:
            self.canvas_scene.addItem(item)

        self.component_items[component.name] = item

        # Add children
        for child in component.children:
            self._add_component_to_scene(child, item)

    def add_component(self, component_type: str, pos: QPointF) -> Optional[UIComponent]:
        """Add a new component at the given position."""
        component_class = get_component_class(component_type)
        if not component_class:
            return None

        # Generate unique name
        base_name = component_type.lower()
        counter = 1
        while f"{base_name}{counter}" in self.component_items:
            counter += 1
        name = f"{base_name}{counter}"

        # Create component
        component = component_class(name=name)
        component.geometry.x = int(pos.x())
        component.geometry.y = int(pos.y())

        # Default sizes based on type
        default_sizes = {
            "Panel": (200, 150),
            "Label": (100, 24),
            "Button": (80, 32),
            "TextInput": (200, 32),
            "ChatHistory": (300, 200),
            "ChatInput": (300, 48),
            "RadianceViewport": (400, 300),
            "FacetAssembly": (32, 32),  # Small - invisible at runtime
        }
        w, h = default_sizes.get(component_type, (100, 32))
        component.geometry.width = w
        component.geometry.height = h

        # Add to root or scene
        if self.root_component:
            self.root_component.add_child(component)
        else:
            self.root_component = component

        # Add to scene
        item = ComponentGraphicsItem(component)
        self.canvas_scene.addItem(item)
        self.component_items[name] = item

        # Select it
        self.canvas_scene.clearSelection()
        item.setSelected(True)

        self.canvas_modified.emit()
        return component

    def delete_selected(self):
        """Delete selected components."""
        selected = self.canvas_scene.selectedItems()
        for item in selected:
            if isinstance(item, ComponentGraphicsItem):
                # Remove from parent
                if item.component.parent:
                    item.component.parent.remove_child(item.component)
                elif item.component == self.root_component:
                    self.root_component = None

                # Remove from tracking
                if item.component.name in self.component_items:
                    del self.component_items[item.component.name]

                # Remove from scene
                self.canvas_scene.removeItem(item)

        if selected:
            self.canvas_modified.emit()

    def save_ui(self):
        """Save current UI to file."""
        if not self.ui_file_path or not self.root_component:
            return

        loader = UILoader()
        loader.save_file(self.root_component, self.ui_file_path)

    def frame_all(self):
        """Fit all components in view (A key)."""
        if not self.component_items:
            # No components - reset to origin
            self.resetTransform()
            self.centerOn(0, 0)
            return

        # Get bounding rect of all items
        rect = None
        for item in self.component_items.values():
            item_rect = item.sceneBoundingRect()
            if rect is None:
                rect = item_rect
            else:
                rect = rect.united(item_rect)

        if rect and rect.width() > 0 and rect.height() > 0:
            # Reset transform first to avoid accumulation issues
            self.resetTransform()
            # Add generous padding
            padding = max(rect.width(), rect.height()) * 0.2
            rect = rect.adjusted(-padding, -padding, padding, padding)
            self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        else:
            # Fallback: reset to origin
            self.resetTransform()
            self.centerOn(0, 0)

    def reset_view(self):
        """Reset view to default zoom and center (Home key)."""
        self.resetTransform()
        self.centerOn(0, 0)
        # Clear focus state
        self.is_focused = False
        self.pre_focus_transform = None
        self.focused_component_ids = None

    def focus_selection(self):
        """
        Toggle focus on selected components (F key).

        First press: Zooms to selection, saves view state
        Second press: Restores pre-focus view state
        """
        selected_items = self.canvas_scene.selectedItems()
        selected_components = [
            item for item in selected_items
            if isinstance(item, ComponentGraphicsItem)
        ]

        if not selected_components:
            return

        # Create selection ID for toggle tracking
        selection_ids = tuple(sorted(item.component.name for item in selected_components))

        # Check if toggling focus on same selection
        if self.is_focused and self.focused_component_ids == selection_ids:
            # RESTORE: Pop back to pre-focus view
            if self.pre_focus_transform:
                self.setTransform(self.pre_focus_transform)
            self.is_focused = False
            self.focused_component_ids = None
            self.pre_focus_transform = None
        else:
            # FOCUS: Save current view and zoom to selection
            self.pre_focus_transform = self.transform()
            self.focused_component_ids = selection_ids
            self.is_focused = True

            # Frame selected components
            self._frame_items(selected_components, padding_factor=0.1)

    def _frame_items(self, items: List[ComponentGraphicsItem], padding_factor: float = 0.1):
        """Frame given items in view with padding."""
        if not items:
            return

        # Get bounding rect of all items
        rect = items[0].sceneBoundingRect()
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())

        # Add padding
        padding = max(rect.width(), rect.height()) * padding_factor
        rect = rect.adjusted(-padding, -padding, padding, padding)

        # Fit in view
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

    def _on_selection_changed(self):
        """Handle selection changes."""
        selected = self.canvas_scene.selectedItems()
        components = [
            item.component for item in selected
            if isinstance(item, ComponentGraphicsItem)
        ]

        if len(components) == 1:
            self.component_selected.emit(components[0])
        elif len(components) > 1:
            self.components_selected.emit(components)
        else:
            self.component_selected.emit(None)

    # --- Input handling ---

    def wheelEvent(self, event: QWheelEvent):
        """Zoom with mouse wheel (with limits)."""
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom_view(zoom_factor)
        else:
            self._zoom_view(1 / zoom_factor)

    def _zoom_view(self, factor: float):
        """Zoom the view by given factor with limits."""
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        # Calculate max zoom based on content
        max_zoom = 3.0  # Default max
        min_zoom = 0.1  # Reasonable minimum

        all_items = list(self.component_items.values())
        if all_items:
            # Calculate bounding rect of all items
            bounding_rect = all_items[0].sceneBoundingRect()
            for item in all_items[1:]:
                bounding_rect = bounding_rect.united(item.sceneBoundingRect())

            # Calculate what zoom would frame all items
            view_rect = self.viewport().rect()
            if bounding_rect.width() > 0 and view_rect.width() > 0:
                frame_all_scale = view_rect.width() / bounding_rect.width()
                max_zoom = max(frame_all_scale * 3.0, 3.0)

        # Clamp to limits
        if new_scale < min_zoom or new_scale > max_zoom:
            return

        self.scale(factor, factor)

    def keyPressEvent(self, event: QKeyEvent):
        """Handle key press."""
        if event.key() == Qt.Key.Key_Space and not self.space_pressed:
            self.space_pressed = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        elif event.key() == Qt.Key.Key_Delete or event.key() == Qt.Key.Key_Backspace:
            self.delete_selected()
        elif event.key() == Qt.Key.Key_A:
            self.frame_all()
        elif event.key() == Qt.Key.Key_F:
            self.focus_selection()
        elif event.key() == Qt.Key.Key_Home:
            self.reset_view()
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent):
        """Handle key release."""
        if event.key() == Qt.Key.Key_Space and self.space_pressed:
            self.space_pressed = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self.panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move."""
        if self.panning:
            delta = event.pos() - self.last_pan_pos
            self.last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handle mouse release."""
        if event.button() == Qt.MouseButton.MiddleButton and self.panning:
            self.panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    # --- Drag and drop ---

    def dragEnterEvent(self, event: QDragEnterEvent):
        """Accept component drops."""
        if event.mimeData().hasFormat("application/x-noodlestudio-component"):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """Handle drag move."""
        if event.mimeData().hasFormat("application/x-noodlestudio-component"):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        """Handle component drop."""
        if event.mimeData().hasFormat("application/x-noodlestudio-component"):
            component_type = event.mimeData().data(
                "application/x-noodlestudio-component"
            ).data().decode()

            # Convert drop position to scene coordinates with grid snap
            scene_pos = self.mapToScene(event.position().toPoint())
            snapped_x = round(scene_pos.x() / GRID_SIZE) * GRID_SIZE
            snapped_y = round(scene_pos.y() / GRID_SIZE) * GRID_SIZE

            self.add_component(component_type, QPointF(snapped_x, snapped_y))
            event.acceptProposedAction()


class UICanvasEditorPanel(QWidget):
    """
    The UI Canvas Editor panel - container for the design surface.

    Includes toolbar for common operations.
    """

    # Signals (forwarded from view)
    component_selected = pyqtSignal(object)
    components_selected = pyqtSignal(list)
    canvas_modified = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._project_manager = None
        self._setup_ui()

    def _setup_ui(self):
        """Setup the panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toolbar
        toolbar = QToolBar()
        toolbar.setStyleSheet("""
            QToolBar {
                background: #2d2d2d;
                border: none;
                padding: 4px;
                spacing: 4px;
            }
            QToolButton {
                background: transparent;
                color: #cccccc;
                border: none;
                padding: 4px 8px;
            }
            QToolButton:hover {
                background: #3d3d3d;
            }
            QToolButton:pressed {
                background: #4d4d4d;
            }
        """)

        # Grid snap toggle
        self.grid_snap_cb = QCheckBox("Grid Snap")
        self.grid_snap_cb.setChecked(True)
        self.grid_snap_cb.setStyleSheet("color: #cccccc;")
        toolbar.addWidget(self.grid_snap_cb)

        toolbar.addSeparator()

        # Status label
        self.status_label = QLabel("No ui.yaml loaded")
        self.status_label.setStyleSheet("color: #888888; padding-left: 8px;")
        toolbar.addWidget(self.status_label)

        # Spacer
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        layout.addWidget(toolbar)

        # Empty state message (shown when no canvas loaded)
        self.empty_state = QLabel("No UI Canvas loaded\n\nRight-click in Stage tab and select:\nRez > New UI Canvas")
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setStyleSheet("""
            QLabel {
                color: #666666;
                font-size: 14px;
                padding: 40px;
                background: #1e1e1e;
            }
        """)

        # Canvas view
        self.view = UICanvasView()
        self.view.component_selected.connect(self.component_selected.emit)
        self.view.components_selected.connect(self.components_selected.emit)
        self.view.canvas_modified.connect(self._on_canvas_modified)
        self.view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.view.customContextMenuRequested.connect(self._show_context_menu)

        # Stack empty state and view
        layout.addWidget(self.empty_state)
        layout.addWidget(self.view)
        self.view.hide()  # Start with empty state visible

    def set_project_manager(self, pm):
        """Set the project manager for file operations."""
        self._project_manager = pm
        # Start with empty state - UI loaded when user selects a UI canvas
        self._show_empty_state()

    def _on_canvas_modified(self):
        """Handle canvas modification - auto-save."""
        self.view.save_ui()
        self.canvas_modified.emit()

    def reload_ui(self):
        """Reload ui.yaml from disk."""
        if self.view.ui_file_path and self.view.ui_file_path.exists():
            self.view.load_ui(self.view.ui_file_path)
            self._show_canvas()

    def load_ui_file(self, path: Path):
        """Load a specific ui.yaml file into the editor."""
        if path and path.exists():
            self.view.load_ui(path)
            self.status_label.setText(f"Editing: {path.name}")
            self._show_canvas()
        else:
            self._show_empty_state()

    def _show_canvas(self):
        """Show canvas view, hide empty state."""
        self.empty_state.hide()
        self.view.show()

    def _show_empty_state(self):
        """Show empty state, hide canvas view."""
        self.view.hide()
        self.empty_state.show()
        self.status_label.setText("No ui.yaml loaded")

    def _show_context_menu(self, pos):
        """Show context menu for adding components."""
        from PyQt6.QtWidgets import QMenu

        menu = QMenu(self)

        # Add component submenu - organized by category
        add_menu = menu.addMenu("Add")

        # Basic components
        add_menu.addAction("Panel", lambda: self._add_component_at_cursor("Panel", pos))
        add_menu.addAction("Label", lambda: self._add_component_at_cursor("Label", pos))
        add_menu.addAction("Button", lambda: self._add_component_at_cursor("Button", pos))
        add_menu.addAction("TextInput", lambda: self._add_component_at_cursor("TextInput", pos))
        add_menu.addSeparator()

        # Form controls
        add_menu.addAction("Checkbox", lambda: self._add_component_at_cursor("Checkbox", pos))
        add_menu.addAction("Dropdown", lambda: self._add_component_at_cursor("Dropdown", pos))
        add_menu.addAction("Slider", lambda: self._add_component_at_cursor("Slider", pos))
        add_menu.addAction("RadioGroup", lambda: self._add_component_at_cursor("RadioGroup", pos))
        add_menu.addSeparator()

        # Advanced components
        add_menu.addAction("ChatHistory", lambda: self._add_component_at_cursor("ChatHistory", pos))
        add_menu.addAction("ChatInput", lambda: self._add_component_at_cursor("ChatInput", pos))
        add_menu.addAction("RadianceViewport", lambda: self._add_component_at_cursor("RadianceViewport", pos))
        add_menu.addAction("WebView", lambda: self._add_component_at_cursor("WebView", pos))
        add_menu.addSeparator()

        # Logic components
        add_menu.addAction("FacetAssembly", lambda: self._add_component_at_cursor("FacetAssembly", pos))

        menu.addSeparator()

        # Delete selected
        selected = self.view.canvas_scene.selectedItems()
        if selected:
            menu.addAction(f"Delete {len(selected)} Selected", self.view.delete_selected)

        menu.addSeparator()
        menu.addAction("Frame All (A)", self.view.frame_all)

        menu.exec(self.view.mapToGlobal(pos))

    def _add_component_at_cursor(self, component_type: str, pos):
        """Add a component at the cursor position."""
        # Convert widget position to scene coordinates
        scene_pos = self.view.mapToScene(pos)
        snapped_x = round(scene_pos.x() / GRID_SIZE) * GRID_SIZE
        snapped_y = round(scene_pos.y() / GRID_SIZE) * GRID_SIZE

        # Ensure we have a root component
        if not self.view.root_component:
            # Create root panel first
            from ..runtime.ui.loader import create_default_ui
            self.view.root_component = create_default_ui()
            self.view._rebuild_scene()

        self.view.add_component(component_type, QPointF(snapped_x, snapped_y))

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
