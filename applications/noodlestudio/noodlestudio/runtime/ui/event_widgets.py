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
#   Event-Emitting Widget Mixins
#
#   Qt widget mixins that capture user interactions and dispa...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.event_widgets
# PURPOSE:  Event-Emitting Widget Mixins
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EventEmittingMixin, EventEmittingFrame, EventEmittingButton, EventEmittingLineEdit, EventEmittingWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Any, Callable, Dict, Optional, TYPE_CHECKING
from PyQt6.QtWidgets import QWidget, QFrame, QPushButton, QLineEdit
from PyQt6.QtCore import Qt, QEvent
from PyQt6.QtGui import QMouseEvent, QKeyEvent, QWheelEvent, QFocusEvent, QEnterEvent

from .event_data import (
    UIEventData,
    MouseButton,
    Modifiers,
    EVENT_CLICK,
    EVENT_DOUBLE_CLICK,
    EVENT_MOUSE_DOWN,
    EVENT_MOUSE_UP,
    EVENT_MOUSE_MOVE,
    EVENT_MOUSE_ENTER,
    EVENT_MOUSE_LEAVE,
    EVENT_MOUSE_WHEEL,
    EVENT_CONTEXT_MENU,
    EVENT_KEY_DOWN,
    EVENT_KEY_UP,
    EVENT_FOCUS,
    EVENT_BLUR,
)

if TYPE_CHECKING:
    from .component import UIComponent
    from .renderer import QtWidgetRenderer


class EventEmittingMixin:
    """
    Mixin that adds comprehensive event emission to any QWidget.

    Captures mouse, keyboard, and focus events and dispatches them
    through the UI event system with full UIEventData context.

    Attributes:
        _ui_component: The UIComponent this widget represents
        _ui_renderer: The renderer for event dispatch
        _track_mouse_move: Whether to emit onMouseMove events
    """

    _ui_component: 'UIComponent'
    _ui_renderer: 'QtWidgetRenderer'
    _track_mouse_move: bool = False

    def _init_event_emitting(
        self,
        component: 'UIComponent',
        renderer: 'QtWidgetRenderer',
        track_mouse_move: bool = False
    ) -> None:
        """
        Initialize event emitting for this widget.

        Args:
            component: The UIComponent this widget represents
            renderer: The renderer for dispatching events
            track_mouse_move: If True, emit onMouseMove events (can be noisy)
        """
        self._ui_component = component
        self._ui_renderer = renderer
        self._track_mouse_move = track_mouse_move

        # Enable mouse tracking if we want move events
        if track_mouse_move and hasattr(self, 'setMouseTracking'):
            self.setMouseTracking(True)

    def _dispatch_event(self, event_type: str, event_data: UIEventData) -> None:
        """Dispatch an event through the renderer's dispatcher."""
        if not hasattr(self, '_ui_renderer') or not self._ui_renderer:
            return

        dispatcher = self._ui_renderer._event_dispatcher
        if not dispatcher:
            return

        component = self._ui_component
        if not component:
            return

        # Check if this event type has a binding
        if event_type in component.events:
            binding = component.events[event_type]
            dispatcher(event_type, component, binding, event_data)

    # --- Mouse Events ---

    def mousePressEvent(self, event: QMouseEvent) -> None:
        """Handle mouse press - emits onMouseDown."""
        event_data = UIEventData.from_qt_mouse_event(
            EVENT_MOUSE_DOWN,
            self._ui_component.name if hasattr(self, '_ui_component') else "",
            event
        )
        self._dispatch_event(EVENT_MOUSE_DOWN, event_data)

        # Call parent implementation
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Handle mouse release - emits onMouseUp and onClick."""
        component_name = self._ui_component.name if hasattr(self, '_ui_component') else ""

        # Emit onMouseUp
        event_data = UIEventData.from_qt_mouse_event(
            EVENT_MOUSE_UP,
            component_name,
            event
        )
        self._dispatch_event(EVENT_MOUSE_UP, event_data)

        # Emit onClick (only for left button release within widget bounds)
        if event.button() == Qt.MouseButton.LeftButton:
            # Check if release is within widget bounds
            if hasattr(self, 'rect') and self.rect().contains(event.position().toPoint()):
                click_data = UIEventData.from_qt_mouse_event(
                    EVENT_CLICK,
                    component_name,
                    event
                )
                self._dispatch_event(EVENT_CLICK, click_data)

        # Call parent implementation
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        """Handle double click - emits onDoubleClick."""
        event_data = UIEventData.from_qt_mouse_event(
            EVENT_DOUBLE_CLICK,
            self._ui_component.name if hasattr(self, '_ui_component') else "",
            event
        )
        self._dispatch_event(EVENT_DOUBLE_CLICK, event_data)

        # Call parent implementation
        super().mouseDoubleClickEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        """Handle mouse move - emits onMouseMove if tracking enabled."""
        if self._track_mouse_move:
            event_data = UIEventData.from_qt_mouse_event(
                EVENT_MOUSE_MOVE,
                self._ui_component.name if hasattr(self, '_ui_component') else "",
                event
            )
            self._dispatch_event(EVENT_MOUSE_MOVE, event_data)

        # Call parent implementation
        super().mouseMoveEvent(event)

    def enterEvent(self, event: QEnterEvent) -> None:
        """Handle mouse enter - emits onMouseEnter."""
        event_data = UIEventData(
            type=EVENT_MOUSE_ENTER,
            source=self._ui_component.name if hasattr(self, '_ui_component') else "",
            x=int(event.position().x()),
            y=int(event.position().y()),
        )
        self._dispatch_event(EVENT_MOUSE_ENTER, event_data)

        # Call parent implementation
        super().enterEvent(event)

    def leaveEvent(self, event: QEvent) -> None:
        """Handle mouse leave - emits onMouseLeave."""
        event_data = UIEventData(
            type=EVENT_MOUSE_LEAVE,
            source=self._ui_component.name if hasattr(self, '_ui_component') else "",
        )
        self._dispatch_event(EVENT_MOUSE_LEAVE, event_data)

        # Call parent implementation
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Handle mouse wheel - emits onMouseWheel."""
        event_data = UIEventData.from_qt_wheel_event(
            self._ui_component.name if hasattr(self, '_ui_component') else "",
            event
        )
        self._dispatch_event(EVENT_MOUSE_WHEEL, event_data)

        # Call parent implementation
        super().wheelEvent(event)

    def contextMenuEvent(self, event) -> None:
        """Handle context menu (right-click) - emits onContextMenu."""
        event_data = UIEventData(
            type=EVENT_CONTEXT_MENU,
            source=self._ui_component.name if hasattr(self, '_ui_component') else "",
            x=event.pos().x(),
            y=event.pos().y(),
            global_x=event.globalPos().x(),
            global_y=event.globalPos().y(),
            button=MouseButton.RIGHT,
        )
        self._dispatch_event(EVENT_CONTEXT_MENU, event_data)

        # Don't call parent - we handle context menu ourselves
        # super().contextMenuEvent(event)

    # --- Keyboard Events ---

    def keyPressEvent(self, event: QKeyEvent) -> None:
        """Handle key press - emits onKeyDown."""
        event_data = UIEventData.from_qt_key_event(
            EVENT_KEY_DOWN,
            self._ui_component.name if hasattr(self, '_ui_component') else "",
            event
        )
        self._dispatch_event(EVENT_KEY_DOWN, event_data)

        # Call parent implementation
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        """Handle key release - emits onKeyUp."""
        event_data = UIEventData.from_qt_key_event(
            EVENT_KEY_UP,
            self._ui_component.name if hasattr(self, '_ui_component') else "",
            event
        )
        self._dispatch_event(EVENT_KEY_UP, event_data)

        # Call parent implementation
        super().keyReleaseEvent(event)

    # --- Focus Events ---

    def focusInEvent(self, event: QFocusEvent) -> None:
        """Handle focus gain - emits onFocus."""
        event_data = UIEventData.focus(
            EVENT_FOCUS,
            self._ui_component.name if hasattr(self, '_ui_component') else ""
        )
        self._dispatch_event(EVENT_FOCUS, event_data)

        # Call parent implementation
        super().focusInEvent(event)

    def focusOutEvent(self, event: QFocusEvent) -> None:
        """Handle focus loss - emits onBlur."""
        event_data = UIEventData.focus(
            EVENT_BLUR,
            self._ui_component.name if hasattr(self, '_ui_component') else ""
        )
        self._dispatch_event(EVENT_BLUR, event_data)

        # Call parent implementation
        super().focusOutEvent(event)


# --- Concrete Event-Emitting Widget Classes ---

class EventEmittingFrame(EventEmittingMixin, QFrame):
    """QFrame with full event emission."""

    def __init__(
        self,
        component: 'UIComponent',
        renderer: 'QtWidgetRenderer',
        parent: Optional[QWidget] = None,
        track_mouse_move: bool = False
    ):
        super().__init__(parent)
        self._init_event_emitting(component, renderer, track_mouse_move)


class EventEmittingButton(EventEmittingMixin, QPushButton):
    """QPushButton with full event emission."""

    def __init__(
        self,
        component: 'UIComponent',
        renderer: 'QtWidgetRenderer',
        text: str = "",
        parent: Optional[QWidget] = None
    ):
        super().__init__(text, parent)
        self._init_event_emitting(component, renderer)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        """Override to not double-emit click (QPushButton has its own clicked signal)."""
        component_name = self._ui_component.name if hasattr(self, '_ui_component') else ""

        # Emit onMouseUp
        event_data = UIEventData.from_qt_mouse_event(
            EVENT_MOUSE_UP,
            component_name,
            event
        )
        self._dispatch_event(EVENT_MOUSE_UP, event_data)

        # Let QPushButton handle click emission via its clicked signal
        # We wire that in the renderer
        QPushButton.mouseReleaseEvent(self, event)


class EventEmittingLineEdit(EventEmittingMixin, QLineEdit):
    """QLineEdit with full event emission."""

    def __init__(
        self,
        component: 'UIComponent',
        renderer: 'QtWidgetRenderer',
        parent: Optional[QWidget] = None
    ):
        super().__init__(parent)
        self._init_event_emitting(component, renderer)


class EventEmittingWidget(EventEmittingMixin, QWidget):
    """Generic QWidget with full event emission."""

    def __init__(
        self,
        component: 'UIComponent',
        renderer: 'QtWidgetRenderer',
        parent: Optional[QWidget] = None,
        track_mouse_move: bool = False
    ):
        super().__init__(parent)
        self._init_event_emitting(component, renderer, track_mouse_move)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
