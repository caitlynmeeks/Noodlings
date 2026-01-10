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
#   Facets Editor View Mixin - View navigation, zoom, grid, and shortcuts
#
#   Contains view/navigation operations: - setup_shortcuts: K...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.facets_editor_view_mixin
# PURPOSE:  facets editor view mixin facet implementation
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   FacetsEditorViewMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import List
from PyQt6.QtWidgets import QGraphicsView
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QKeySequence, QShortcut, QPen, QColor

from .facets_editor_graphics import FacetNodeGraphics


class FacetsEditorViewMixin:
    """Mixin providing view navigation for FacetsEditorPanel."""

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
            return

        selected_node = selected_nodes[0]
        selected_node_id = selected_node.facet.id

        # Check if we're toggling focus on the same node
        if self.is_focused and self.focused_node_id == selected_node_id:
            # RESTORE: Pop back to pre-focus view
            if self.pre_focus_transform:
                self.view.setTransform(self.pre_focus_transform)
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

    # ========== GRID OPERATIONS ==========

    def _draw_grid_background(self):
        """Draw grid lines on the scene background."""
        if not self.grid_visible:
            return

        # Clear existing grid lines first
        self._clear_grid_background()

        # Grid parameters
        scene_rect = self.scene.sceneRect()
        grid_pen = QPen(QColor("#3A3A3A"), 1)  # Subtle grid color

        # Draw vertical lines
        x = int(scene_rect.left() / self.grid_size) * self.grid_size
        while x < scene_rect.right():
            line = self.scene.addLine(x, scene_rect.top(), x, scene_rect.bottom(), grid_pen)
            line.setZValue(-10)  # Behind everything
            self.grid_lines.append(line)
            x += self.grid_size

        # Draw horizontal lines
        y = int(scene_rect.top() / self.grid_size) * self.grid_size
        while y < scene_rect.bottom():
            line = self.scene.addLine(scene_rect.left(), y, scene_rect.right(), y, grid_pen)
            line.setZValue(-10)  # Behind everything
            self.grid_lines.append(line)
            y += self.grid_size

    def _clear_grid_background(self):
        """Remove grid lines from scene."""
        if not self.grid_lines:
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
        settings = QSettings('Noodlings', 'FacetsEditor')
        settings.setValue('grid/size', value)

        # Redraw grid if visible
        if self.grid_visible:
            self._clear_grid_background()
            self._draw_grid_background()

    def set_grid_size(self, size: int):
        """Set grid snap size in pixels."""
        self.grid_size = size

    # ========== KEYBOARD EVENTS ==========

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key.Key_Space and not self.space_pressed:
            # Space pressed - switch to pan mode
            self.space_pressed = True
            self.view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.view.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            # Safely call parent - MRO may not include keyPressEvent in next mixin
            parent_method = getattr(super(), 'keyPressEvent', None)
            if parent_method:
                parent_method(event)

    def keyReleaseEvent(self, event):
        """Handle key release events."""
        if event.key() == Qt.Key.Key_Space and self.space_pressed:
            # Space released - back to selection mode
            self.space_pressed = False
            self.view.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.view.viewport().setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # Safely call parent - MRO may not include keyReleaseEvent in next mixin
            parent_method = getattr(super(), 'keyReleaseEvent', None)
            if parent_method:
                parent_method(event)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
