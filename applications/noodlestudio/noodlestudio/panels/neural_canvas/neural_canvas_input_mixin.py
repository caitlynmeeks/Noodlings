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
#   Neural Canvas Input Mixin - Mouse, keyboard, and wheel event handling
#
#   Contains: - wheelEvent: Zoom with mouse wheel - _zoom_vie...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.neural_canvas_input_mixin
# PURPOSE:  Neural Canvas Input Mixin
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralCanvasInputMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtWidgets import QGraphicsView, QGraphicsItem
from PyQt6.QtGui import QWheelEvent, QMouseEvent

from ...core.neural_canvas.neural_node import Connection


class NeuralCanvasInputMixin:
    """Mixin providing input handling for NeuralCanvasView."""

    def wheelEvent(self, event: QWheelEvent):
        """Handle zoom with mouse wheel (with limits)."""
        import time

        # Guard: Ignore wheel events within 200ms of right-click (trackpad quirk)
        if time.time() - self._last_right_click_time < 0.2:
            event.ignore()
            return

        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            self._zoom_view(zoom_factor)
        else:
            self._zoom_view(1 / zoom_factor)

    def _zoom_view(self, factor: float):
        """
        Zoom the view by given factor with limits.

        Args:
            factor: Zoom multiplier (>1 = zoom in, <1 = zoom out)
        """
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        # Calculate max zoom based on content
        max_zoom = 3.0  # Default max
        min_zoom = 0.1  # Reasonable minimum to see everything

        all_nodes = list(self.node_items.values())
        if all_nodes:
            # Calculate bounding rect of all nodes
            bounding_rect = all_nodes[0].sceneBoundingRect()
            for node in all_nodes[1:]:
                bounding_rect = bounding_rect.united(node.sceneBoundingRect())

            # Calculate what zoom would frame all nodes
            view_rect = self.viewport().rect()
            if bounding_rect.width() > 0 and view_rect.width() > 0:
                frame_all_scale = view_rect.width() / bounding_rect.width()
                max_zoom = max(frame_all_scale * 3.0, 3.0)  # 3x the frame-all zoom or 3.0

        # Clamp to limits
        if new_scale < min_zoom or new_scale > max_zoom:
            return

        self.scale(factor, factor)

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
        # Import here to avoid circular imports
        from .neural_canvas_view import PortGraphicsItem, NodeGraphicsItem, TemporaryWireItem

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
                    else:
                        # Nothing selected - emit empty string to clear inspector
                        self.node_selected.emit("")
        elif event.button() == Qt.MouseButton.RightButton:
            # Right-click should NOT change selection (context menu preserves selection)
            import time
            self._last_right_click_time = time.time()  # Record for wheel guard
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
        # Import here to avoid circular imports
        from .neural_canvas_view import PortGraphicsItem, ConnectionGraphicsItem

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
                # Check if connecting output -> input (or input -> output)
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
