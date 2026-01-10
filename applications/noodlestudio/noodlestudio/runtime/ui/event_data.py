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
#   UI Event Data
#
#   Rich event metadata for the UI event system. Inspired by ...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.event_data
# PURPOSE:  UI Event Data
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MouseButton, Modifiers, UIEventData
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
from enum import Enum
import time


class MouseButton(Enum):
    """Mouse button identifiers."""
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class Modifiers:
    """Keyboard modifier state during an event."""
    shift: bool = False
    ctrl: bool = False
    alt: bool = False
    meta: bool = False  # Cmd on macOS, Win on Windows

    def to_dict(self) -> Dict[str, bool]:
        """Convert to dictionary for script access."""
        return {
            "shift": self.shift,
            "ctrl": self.ctrl,
            "alt": self.alt,
            "meta": self.meta,
        }

    @classmethod
    def from_qt(cls, modifiers) -> 'Modifiers':
        """Create from Qt keyboard modifiers."""
        from PyQt6.QtCore import Qt
        return cls(
            shift=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
            ctrl=bool(modifiers & Qt.KeyboardModifier.ControlModifier),
            alt=bool(modifiers & Qt.KeyboardModifier.AltModifier),
            meta=bool(modifiers & Qt.KeyboardModifier.MetaModifier),
        )


@dataclass
class UIEventData:
    """
    Rich event metadata for UI events.

    Carries all context needed by event handlers, scripts, and noodlings
    to respond appropriately to user interactions.

    Attributes:
        type: Event type name (onClick, onKeyDown, onMouseMove, etc.)
        source: Name of the component that triggered the event
        timestamp: Unix timestamp when event occurred

        # Mouse events
        x: Mouse X position relative to component
        y: Mouse Y position relative to component
        global_x: Mouse X position relative to window
        global_y: Mouse Y position relative to window
        button: Which mouse button (left, right, middle)
        modifiers: Keyboard modifiers held during event

        # Keyboard events
        key: Key name (e.g., "Enter", "a", "Escape")
        key_code: Numeric key code
        text: Text input from key press (handles shift, etc.)

        # Value events
        value: Current value (for onChange, onSubmit)
        previous_value: Previous value before change

        # Drag events
        drag_data: Arbitrary data being dragged
        drop_effect: Allowed drop effect (copy, move, link)

        # 3D events (RadianceViewport)
        hit_position: World-space position of raycast hit
        hit_entity: Entity ID that was hit
        hit_semantics: Semantic metadata of hit Gaussian (body_part, etc.)

        # Scroll/wheel events
        delta_x: Horizontal scroll amount
        delta_y: Vertical scroll amount

        # Propagation control
        stopped: If True, event should not propagate further
        prevented: If True, default action should be prevented
    """

    # Core fields
    type: str
    source: str
    timestamp: float = field(default_factory=time.time)

    # Mouse
    x: Optional[int] = None
    y: Optional[int] = None
    global_x: Optional[int] = None
    global_y: Optional[int] = None
    button: MouseButton = MouseButton.NONE
    modifiers: Modifiers = field(default_factory=Modifiers)

    # Keyboard
    key: Optional[str] = None
    key_code: Optional[int] = None
    text: Optional[str] = None

    # Value
    value: Any = None
    previous_value: Any = None

    # Drag
    drag_data: Any = None
    drop_effect: Optional[str] = None

    # 3D (RadianceViewport)
    hit_position: Optional[Tuple[float, float, float]] = None
    hit_entity: Optional[str] = None
    hit_semantics: Optional[Dict[str, Any]] = None

    # Scroll
    delta_x: float = 0.0
    delta_y: float = 0.0

    # Propagation
    stopped: bool = False
    prevented: bool = False

    def stop_propagation(self) -> None:
        """Prevent event from bubbling to parent components."""
        self.stopped = True

    def prevent_default(self) -> None:
        """Prevent the default action for this event."""
        self.prevented = True

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for script access.

        Scripts receive this as the `event` object:
            event.type, event.source, event.x, event.y, etc.
        """
        data = {
            "type": self.type,
            "source": self.source,
            "timestamp": self.timestamp,
        }

        # Mouse
        if self.x is not None:
            data["x"] = self.x
        if self.y is not None:
            data["y"] = self.y
        if self.global_x is not None:
            data["globalX"] = self.global_x
        if self.global_y is not None:
            data["globalY"] = self.global_y
        if self.button != MouseButton.NONE:
            data["button"] = self.button.value
        if self.modifiers:
            data["modifiers"] = self.modifiers.to_dict()

        # Keyboard
        if self.key is not None:
            data["key"] = self.key
        if self.key_code is not None:
            data["keyCode"] = self.key_code
        if self.text is not None:
            data["text"] = self.text

        # Value
        if self.value is not None:
            data["value"] = self.value
        if self.previous_value is not None:
            data["previousValue"] = self.previous_value

        # Drag
        if self.drag_data is not None:
            data["dragData"] = self.drag_data
        if self.drop_effect is not None:
            data["dropEffect"] = self.drop_effect

        # 3D
        if self.hit_position is not None:
            data["hitPosition"] = {
                "x": self.hit_position[0],
                "y": self.hit_position[1],
                "z": self.hit_position[2],
            }
        if self.hit_entity is not None:
            data["hitEntity"] = self.hit_entity
        if self.hit_semantics is not None:
            data["hitSemantics"] = self.hit_semantics

        # Scroll
        if self.delta_x != 0.0 or self.delta_y != 0.0:
            data["deltaX"] = self.delta_x
            data["deltaY"] = self.delta_y

        return data

    @classmethod
    def from_qt_mouse_event(
        cls,
        event_type: str,
        source: str,
        qt_event,
        component_pos: Tuple[int, int] = (0, 0)
    ) -> 'UIEventData':
        """
        Create UIEventData from a Qt mouse event.

        Args:
            event_type: Event name (onClick, onMouseMove, etc.)
            source: Component name
            qt_event: QMouseEvent instance
            component_pos: Component's position for local coordinates
        """
        from PyQt6.QtCore import Qt

        # Determine button
        button = MouseButton.NONE
        if qt_event.button() == Qt.MouseButton.LeftButton:
            button = MouseButton.LEFT
        elif qt_event.button() == Qt.MouseButton.RightButton:
            button = MouseButton.RIGHT
        elif qt_event.button() == Qt.MouseButton.MiddleButton:
            button = MouseButton.MIDDLE

        pos = qt_event.position()
        global_pos = qt_event.globalPosition()

        return cls(
            type=event_type,
            source=source,
            x=int(pos.x()),
            y=int(pos.y()),
            global_x=int(global_pos.x()),
            global_y=int(global_pos.y()),
            button=button,
            modifiers=Modifiers.from_qt(qt_event.modifiers()),
        )

    @classmethod
    def from_qt_key_event(
        cls,
        event_type: str,
        source: str,
        qt_event
    ) -> 'UIEventData':
        """
        Create UIEventData from a Qt keyboard event.

        Args:
            event_type: Event name (onKeyDown, onKeyUp, onKeyPress)
            source: Component name
            qt_event: QKeyEvent instance
        """
        from PyQt6.QtCore import Qt

        # Map Qt key to readable name
        key_name = cls._qt_key_to_name(qt_event.key())

        return cls(
            type=event_type,
            source=source,
            key=key_name,
            key_code=qt_event.key(),
            text=qt_event.text() if qt_event.text() else None,
            modifiers=Modifiers.from_qt(qt_event.modifiers()),
        )

    @classmethod
    def from_qt_wheel_event(
        cls,
        source: str,
        qt_event
    ) -> 'UIEventData':
        """
        Create UIEventData from a Qt wheel event.

        Args:
            source: Component name
            qt_event: QWheelEvent instance
        """
        pos = qt_event.position()
        global_pos = qt_event.globalPosition()
        delta = qt_event.angleDelta()

        return cls(
            type="onMouseWheel",
            source=source,
            x=int(pos.x()),
            y=int(pos.y()),
            global_x=int(global_pos.x()),
            global_y=int(global_pos.y()),
            delta_x=delta.x() / 120.0,  # Normalize to "lines"
            delta_y=delta.y() / 120.0,
            modifiers=Modifiers.from_qt(qt_event.modifiers()),
        )

    @classmethod
    def value_change(
        cls,
        source: str,
        value: Any,
        previous_value: Any = None
    ) -> 'UIEventData':
        """
        Create UIEventData for a value change event.

        Args:
            source: Component name
            value: New value
            previous_value: Previous value (if known)
        """
        return cls(
            type="onChange",
            source=source,
            value=value,
            previous_value=previous_value,
        )

    @classmethod
    def submit(cls, source: str, value: Any) -> 'UIEventData':
        """
        Create UIEventData for a submit event.

        Args:
            source: Component name
            value: Submitted value
        """
        return cls(
            type="onSubmit",
            source=source,
            value=value,
        )

    @classmethod
    def click(cls, source: str) -> 'UIEventData':
        """
        Create a simple click event (no mouse position).

        Args:
            source: Component name
        """
        return cls(
            type="onClick",
            source=source,
            button=MouseButton.LEFT,
        )

    @classmethod
    def focus(cls, event_type: str, source: str) -> 'UIEventData':
        """
        Create UIEventData for focus/blur events.

        Args:
            event_type: "onFocus" or "onBlur"
            source: Component name
        """
        return cls(
            type=event_type,
            source=source,
        )

    @staticmethod
    def _qt_key_to_name(key: int) -> str:
        """Convert Qt key code to readable name."""
        from PyQt6.QtCore import Qt

        # Common keys mapping
        key_map = {
            Qt.Key.Key_Return: "Enter",
            Qt.Key.Key_Enter: "Enter",
            Qt.Key.Key_Escape: "Escape",
            Qt.Key.Key_Tab: "Tab",
            Qt.Key.Key_Backspace: "Backspace",
            Qt.Key.Key_Delete: "Delete",
            Qt.Key.Key_Insert: "Insert",
            Qt.Key.Key_Home: "Home",
            Qt.Key.Key_End: "End",
            Qt.Key.Key_PageUp: "PageUp",
            Qt.Key.Key_PageDown: "PageDown",
            Qt.Key.Key_Left: "ArrowLeft",
            Qt.Key.Key_Right: "ArrowRight",
            Qt.Key.Key_Up: "ArrowUp",
            Qt.Key.Key_Down: "ArrowDown",
            Qt.Key.Key_Space: "Space",
            Qt.Key.Key_Shift: "Shift",
            Qt.Key.Key_Control: "Control",
            Qt.Key.Key_Alt: "Alt",
            Qt.Key.Key_Meta: "Meta",
            Qt.Key.Key_CapsLock: "CapsLock",
            Qt.Key.Key_F1: "F1",
            Qt.Key.Key_F2: "F2",
            Qt.Key.Key_F3: "F3",
            Qt.Key.Key_F4: "F4",
            Qt.Key.Key_F5: "F5",
            Qt.Key.Key_F6: "F6",
            Qt.Key.Key_F7: "F7",
            Qt.Key.Key_F8: "F8",
            Qt.Key.Key_F9: "F9",
            Qt.Key.Key_F10: "F10",
            Qt.Key.Key_F11: "F11",
            Qt.Key.Key_F12: "F12",
        }

        if key in key_map:
            return key_map[key]

        # For letter keys, return the character
        if Qt.Key.Key_A <= key <= Qt.Key.Key_Z:
            return chr(key)

        # For number keys
        if Qt.Key.Key_0 <= key <= Qt.Key.Key_9:
            return chr(key)

        # Fallback to key code string
        return f"Key{key}"


# --- Event type constants ---

# Mouse events
EVENT_CLICK = "onClick"
EVENT_DOUBLE_CLICK = "onDoubleClick"
EVENT_MOUSE_DOWN = "onMouseDown"
EVENT_MOUSE_UP = "onMouseUp"
EVENT_MOUSE_MOVE = "onMouseMove"
EVENT_MOUSE_ENTER = "onMouseEnter"
EVENT_MOUSE_LEAVE = "onMouseLeave"
EVENT_MOUSE_WHEEL = "onMouseWheel"
EVENT_CONTEXT_MENU = "onContextMenu"

# Drag events
EVENT_DRAG_START = "onDragStart"
EVENT_DRAG = "onDrag"
EVENT_DRAG_ENTER = "onDragEnter"
EVENT_DRAG_OVER = "onDragOver"
EVENT_DRAG_LEAVE = "onDragLeave"
EVENT_DROP = "onDrop"
EVENT_DRAG_END = "onDragEnd"

# Keyboard events
EVENT_KEY_DOWN = "onKeyDown"
EVENT_KEY_UP = "onKeyUp"
EVENT_KEY_PRESS = "onKeyPress"

# Focus events
EVENT_FOCUS = "onFocus"
EVENT_BLUR = "onBlur"

# Value events
EVENT_CHANGE = "onChange"
EVENT_SUBMIT = "onSubmit"
EVENT_SELECT = "onSelect"
EVENT_CHECK = "onCheck"
EVENT_TOGGLE = "onToggle"

# Lifecycle events
EVENT_CREATE = "onCreate"
EVENT_DESTROY = "onDestroy"
EVENT_SHOW = "onShow"
EVENT_HIDE = "onHide"
EVENT_RESIZE = "onResize"
EVENT_MOVE = "onMove"

# Validation events
EVENT_VALIDATE = "onValidate"
EVENT_ERROR = "onError"

# RadianceViewport events
EVENT_LOAD = "onLoad"
EVENT_CAMERA_MOVE = "onCameraMove"
EVENT_GAUSSIAN_CLICK = "onGaussianClick"
EVENT_GAUSSIAN_HOVER = "onGaussianHover"

# All event types (for validation)
ALL_EVENT_TYPES = {
    EVENT_CLICK, EVENT_DOUBLE_CLICK, EVENT_MOUSE_DOWN, EVENT_MOUSE_UP,
    EVENT_MOUSE_MOVE, EVENT_MOUSE_ENTER, EVENT_MOUSE_LEAVE, EVENT_MOUSE_WHEEL,
    EVENT_CONTEXT_MENU,
    EVENT_DRAG_START, EVENT_DRAG, EVENT_DRAG_ENTER, EVENT_DRAG_OVER,
    EVENT_DRAG_LEAVE, EVENT_DROP, EVENT_DRAG_END,
    EVENT_KEY_DOWN, EVENT_KEY_UP, EVENT_KEY_PRESS,
    EVENT_FOCUS, EVENT_BLUR,
    EVENT_CHANGE, EVENT_SUBMIT, EVENT_SELECT, EVENT_CHECK, EVENT_TOGGLE,
    EVENT_CREATE, EVENT_DESTROY, EVENT_SHOW, EVENT_HIDE, EVENT_RESIZE, EVENT_MOVE,
    EVENT_VALIDATE, EVENT_ERROR,
    EVENT_LOAD, EVENT_CAMERA_MOVE, EVENT_GAUSSIAN_CLICK, EVENT_GAUSSIAN_HOVER,
}

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
