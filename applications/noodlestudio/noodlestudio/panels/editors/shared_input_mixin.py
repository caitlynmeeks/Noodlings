"""Shared input handling mixin for editor views.

Provides wheel zoom, space-drag pan, and middle-mouse pan.
Assumes self is a QGraphicsView subclass.
"""

import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QGraphicsView


class SharedInputMixin:
    """Zoom, pan, and basic input handling for canvas editor views.

    Configure behavior via class attributes on the concrete view:
        ZOOM_FACTOR: Wheel zoom step (default 1.15)
        MIN_ZOOM: Minimum scale (default 0.1)
        MAX_ZOOM: Default maximum scale (default 3.0)
        MAX_ZOOM_MULTIPLIER: How far past frame-all zoom to allow (default 3.0)
        RIGHT_CLICK_GUARD_SEC: Ignore wheel events this long after right-click (default 0.2)
    """

    ZOOM_FACTOR = 1.15
    MIN_ZOOM = 0.1
    MAX_ZOOM = 3.0
    MAX_ZOOM_MULTIPLIER = 3.0
    RIGHT_CLICK_GUARD_SEC = 0.2

    def _init_input_state(self):
        """Initialize input mixin state. Call from concrete view __init__."""
        self._last_right_click_time = 0.0
        self._space_pressed = False
        self._middle_panning = False
        self._last_pan_pos = None

    def wheel_zoom(self, event):
        """Handle zoom with mouse wheel, guarded against trackpad right-click quirk."""
        if time.time() - self._last_right_click_time < self.RIGHT_CLICK_GUARD_SEC:
            event.ignore()
            return

        if event.angleDelta().y() > 0:
            self._zoom_view(self.ZOOM_FACTOR)
        else:
            self._zoom_view(1.0 / self.ZOOM_FACTOR)

    def _zoom_view(self, factor: float):
        """Zoom the view by factor with dynamic limits."""
        current_scale = self.transform().m11()
        new_scale = current_scale * factor

        max_zoom = self.MAX_ZOOM
        node_items = self.get_node_items()
        if node_items:
            items = list(node_items.values())
            bounding = items[0].sceneBoundingRect()
            for item in items[1:]:
                bounding = bounding.united(item.sceneBoundingRect())

            view_rect = self.viewport().rect()
            if bounding.width() > 0 and view_rect.width() > 0:
                frame_all_scale = view_rect.width() / bounding.width()
                max_zoom = max(frame_all_scale * self.MAX_ZOOM_MULTIPLIER, self.MAX_ZOOM)

        if new_scale < self.MIN_ZOOM or new_scale > max_zoom:
            return

        self.scale(factor, factor)

    def handle_key_press_input(self, event):
        """Handle space-to-pan key press. Returns True if consumed."""
        if event.key() == Qt.Key.Key_Space and not self._space_pressed:
            self._space_pressed = True
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return True
        return False

    def handle_key_release_input(self, event):
        """Handle space-to-pan key release. Returns True if consumed."""
        if event.key() == Qt.Key.Key_Space and self._space_pressed:
            self._space_pressed = False
            self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
            self.viewport().setCursor(Qt.CursorShape.ArrowCursor)
            return True
        return False

    def handle_middle_press(self, event):
        """Start middle-mouse pan. Returns True if consumed."""
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return True
        return False

    def handle_middle_move(self, event):
        """Continue middle-mouse pan. Returns True if consumed."""
        if self._middle_panning and self._last_pan_pos is not None:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return True
        return False

    def handle_middle_release(self, event):
        """End middle-mouse pan. Returns True if consumed."""
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_panning:
            self._middle_panning = False
            self._last_pan_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return True
        return False

    def record_right_click_time(self):
        """Record right-click timestamp for wheel guard."""
        self._last_right_click_time = time.time()

    # -- Abstract: concrete view must implement --

    def get_node_items(self) -> dict:
        """Override: return dict of node_id -> QGraphicsItem."""
        raise NotImplementedError
