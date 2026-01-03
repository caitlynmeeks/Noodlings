"""
UI Action Recorder - Captures UI events for crash debugging and test replay

Records mouse clicks, key presses, and widget interactions in a ring buffer.
Output format is designed to be qtbot-compatible for test replay.

Usage:
    recorder = get_ui_action_recorder()
    recorder.install(main_window)  # Start recording

    # On crash:
    actions = recorder.get_recent_actions(50)
    qtbot_script = recorder.to_qtbot_script(actions)

Author: Caitlyn + Claude
Date: January 2, 2026
"""

import time
import json
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Deque
from pathlib import Path
from enum import Enum

from PyQt6.QtCore import QObject, QEvent, Qt, QTimer
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtGui import QMouseEvent, QKeyEvent


class ActionType(Enum):
    """Types of UI actions we record."""
    MOUSE_CLICK = "mouse_click"
    MOUSE_DOUBLE_CLICK = "mouse_double_click"
    MOUSE_PRESS = "mouse_press"
    MOUSE_RELEASE = "mouse_release"
    KEY_PRESS = "key_press"
    KEY_RELEASE = "key_release"
    FOCUS_IN = "focus_in"
    FOCUS_OUT = "focus_out"
    WHEEL = "wheel"
    CONTEXT_MENU = "context_menu"
    DROP = "drop"


@dataclass
class UIAction:
    """A single recorded UI action."""
    timestamp: float  # Unix timestamp with milliseconds
    action_type: str
    widget_class: str
    widget_name: str  # objectName if set
    widget_path: str  # Hierarchical path for identification

    # Mouse events
    button: Optional[str] = None
    pos_x: Optional[int] = None
    pos_y: Optional[int] = None
    global_x: Optional[int] = None
    global_y: Optional[int] = None

    # Key events
    key: Optional[str] = None
    key_code: Optional[int] = None
    text: Optional[str] = None
    modifiers: Optional[List[str]] = None

    # Wheel events
    delta_x: Optional[int] = None
    delta_y: Optional[int] = None

    # Extra context
    widget_text: Optional[str] = None  # Button text, label text, etc.
    parent_class: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_qtbot_code(self) -> str:
        """Generate qtbot-compatible Python code for this action."""
        lines = []

        # Comment with timestamp and widget info
        dt = datetime.fromtimestamp(self.timestamp)
        lines.append(f"# {dt.strftime('%H:%M:%S.%f')[:-3]} - {self.widget_class}")
        if self.widget_text:
            lines.append(f"# Widget text: {self.widget_text[:50]}")

        # Find widget code
        if self.widget_name:
            widget_finder = f"window.findChild(QWidget, '{self.widget_name}')"
        else:
            widget_finder = f"# TODO: Find widget by path: {self.widget_path}"

        # Generate qtbot call
        if self.action_type == ActionType.MOUSE_CLICK.value:
            btn = f"Qt.MouseButton.{self.button}" if self.button else "Qt.MouseButton.LeftButton"
            if self.pos_x is not None:
                lines.append(f"qtbot.mouseClick({widget_finder}, {btn}, pos=QPoint({self.pos_x}, {self.pos_y}))")
            else:
                lines.append(f"qtbot.mouseClick({widget_finder}, {btn})")

        elif self.action_type == ActionType.MOUSE_DOUBLE_CLICK.value:
            btn = f"Qt.MouseButton.{self.button}" if self.button else "Qt.MouseButton.LeftButton"
            lines.append(f"qtbot.mouseDClick({widget_finder}, {btn})")

        elif self.action_type == ActionType.KEY_PRESS.value:
            if self.text and len(self.text) == 1 and self.text.isprintable():
                lines.append(f"qtbot.keyClick({widget_finder}, '{self.text}')")
            elif self.key:
                lines.append(f"qtbot.keyClick({widget_finder}, Qt.Key.{self.key})")

        elif self.action_type == ActionType.WHEEL.value:
            lines.append(f"# Wheel: delta=({self.delta_x}, {self.delta_y})")

        return "\n".join(lines)


class UIActionRecorder(QObject):
    """
    Global event filter that records UI actions.

    Uses a ring buffer to keep memory bounded.
    """

    # Singleton instance
    _instance: Optional['UIActionRecorder'] = None

    # Buffer size (number of actions to keep)
    BUFFER_SIZE = 500

    # Debounce settings (avoid recording every mouse move)
    _last_move_time: float = 0
    MOVE_DEBOUNCE_MS = 100

    def __init__(self):
        super().__init__()
        self._buffer: Deque[UIAction] = deque(maxlen=self.BUFFER_SIZE)
        self._recording = False
        self._app: Optional[QApplication] = None
        self._click_pending: Optional[Dict] = None
        self._click_timer = QTimer()
        self._click_timer.setSingleShot(True)
        self._click_timer.timeout.connect(self._flush_pending_click)

        # Track key sequences for text input
        self._key_sequence: List[str] = []
        self._key_sequence_widget: Optional[str] = None

    @classmethod
    def instance(cls) -> 'UIActionRecorder':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = UIActionRecorder()
        return cls._instance

    def install(self, app: QApplication):
        """Install as global event filter."""
        self._app = app
        app.installEventFilter(self)
        self._recording = True
        print("[UIActionRecorder] Installed - recording UI actions")

    def uninstall(self):
        """Remove event filter."""
        if self._app:
            self._app.removeEventFilter(self)
            self._recording = False
            print("[UIActionRecorder] Uninstalled")

    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    def get_recent_actions(self, count: int = 50) -> List[UIAction]:
        """Get the N most recent actions."""
        actions = list(self._buffer)
        return actions[-count:] if len(actions) > count else actions

    def get_all_actions(self) -> List[UIAction]:
        """Get all recorded actions."""
        return list(self._buffer)

    def clear(self):
        """Clear the buffer."""
        self._buffer.clear()

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        """Filter and record relevant events."""
        if not self._recording:
            return False

        # Only process widget events
        if not isinstance(obj, QWidget):
            return False

        event_type = event.type()

        try:
            if event_type == QEvent.Type.MouseButtonPress:
                self._record_mouse_event(obj, event, ActionType.MOUSE_PRESS)

            elif event_type == QEvent.Type.MouseButtonRelease:
                self._handle_click(obj, event)

            elif event_type == QEvent.Type.MouseButtonDblClick:
                self._cancel_pending_click()
                self._record_mouse_event(obj, event, ActionType.MOUSE_DOUBLE_CLICK)

            elif event_type == QEvent.Type.KeyPress:
                self._record_key_event(obj, event, ActionType.KEY_PRESS)

            elif event_type == QEvent.Type.KeyRelease:
                self._record_key_event(obj, event, ActionType.KEY_RELEASE)

            elif event_type == QEvent.Type.Wheel:
                self._record_wheel_event(obj, event)

            elif event_type == QEvent.Type.FocusIn:
                self._record_focus_event(obj, ActionType.FOCUS_IN)

            elif event_type == QEvent.Type.FocusOut:
                self._record_focus_event(obj, ActionType.FOCUS_OUT)

            elif event_type == QEvent.Type.ContextMenu:
                self._record_context_menu(obj, event)

        except Exception as e:
            # Never let recording crash the app
            print(f"[UIActionRecorder] Error recording event: {e}")

        # Never consume events
        return False

    def _handle_click(self, widget: QWidget, event: QMouseEvent):
        """Handle click with double-click detection."""
        # Cancel any pending single click
        self._cancel_pending_click()

        # Store this click as pending
        self._click_pending = {
            'widget': widget,
            'event_data': self._extract_mouse_data(widget, event),
            'time': time.time()
        }

        # Wait a bit to see if double-click follows
        self._click_timer.start(QApplication.doubleClickInterval() + 50)

    def _flush_pending_click(self):
        """Record a pending single click."""
        if self._click_pending:
            data = self._click_pending['event_data']
            action = UIAction(
                timestamp=self._click_pending['time'],
                action_type=ActionType.MOUSE_CLICK.value,
                **data
            )
            self._buffer.append(action)
            self._click_pending = None

    def _cancel_pending_click(self):
        """Cancel pending single click (double-click detected)."""
        self._click_timer.stop()
        self._click_pending = None

    def _record_mouse_event(self, widget: QWidget, event: QMouseEvent, action_type: ActionType):
        """Record a mouse event."""
        data = self._extract_mouse_data(widget, event)
        action = UIAction(
            timestamp=time.time(),
            action_type=action_type.value,
            **data
        )
        self._buffer.append(action)

    def _extract_mouse_data(self, widget: QWidget, event: QMouseEvent) -> Dict:
        """Extract mouse event data."""
        button_map = {
            Qt.MouseButton.LeftButton: "LeftButton",
            Qt.MouseButton.RightButton: "RightButton",
            Qt.MouseButton.MiddleButton: "MiddleButton",
        }

        pos = event.position()
        global_pos = event.globalPosition()

        return {
            'widget_class': widget.__class__.__name__,
            'widget_name': widget.objectName() or "",
            'widget_path': self._get_widget_path(widget),
            'widget_text': self._get_widget_text(widget),
            'parent_class': widget.parent().__class__.__name__ if widget.parent() else None,
            'button': button_map.get(event.button(), "Unknown"),
            'pos_x': int(pos.x()),
            'pos_y': int(pos.y()),
            'global_x': int(global_pos.x()),
            'global_y': int(global_pos.y()),
        }

    def _record_key_event(self, widget: QWidget, event: QKeyEvent, action_type: ActionType):
        """Record a key event."""
        # Get key name
        key = event.key()
        key_name = Qt.Key(key).name if hasattr(Qt.Key(key), 'name') else f"Key_{key}"

        # Get modifiers
        mods = event.modifiers()
        modifiers = []
        if mods & Qt.KeyboardModifier.ShiftModifier:
            modifiers.append("Shift")
        if mods & Qt.KeyboardModifier.ControlModifier:
            modifiers.append("Ctrl")
        if mods & Qt.KeyboardModifier.AltModifier:
            modifiers.append("Alt")
        if mods & Qt.KeyboardModifier.MetaModifier:
            modifiers.append("Meta")

        action = UIAction(
            timestamp=time.time(),
            action_type=action_type.value,
            widget_class=widget.__class__.__name__,
            widget_name=widget.objectName() or "",
            widget_path=self._get_widget_path(widget),
            widget_text=self._get_widget_text(widget),
            key=key_name,
            key_code=key,
            text=event.text() if event.text().isprintable() else None,
            modifiers=modifiers if modifiers else None,
        )
        self._buffer.append(action)

    def _record_wheel_event(self, widget: QWidget, event):
        """Record a wheel event."""
        delta = event.angleDelta()

        action = UIAction(
            timestamp=time.time(),
            action_type=ActionType.WHEEL.value,
            widget_class=widget.__class__.__name__,
            widget_name=widget.objectName() or "",
            widget_path=self._get_widget_path(widget),
            delta_x=delta.x(),
            delta_y=delta.y(),
        )
        self._buffer.append(action)

    def _record_focus_event(self, widget: QWidget, action_type: ActionType):
        """Record a focus event."""
        action = UIAction(
            timestamp=time.time(),
            action_type=action_type.value,
            widget_class=widget.__class__.__name__,
            widget_name=widget.objectName() or "",
            widget_path=self._get_widget_path(widget),
            widget_text=self._get_widget_text(widget),
        )
        self._buffer.append(action)

    def _record_context_menu(self, widget: QWidget, event):
        """Record a context menu event."""
        pos = event.pos()
        global_pos = event.globalPos()

        action = UIAction(
            timestamp=time.time(),
            action_type=ActionType.CONTEXT_MENU.value,
            widget_class=widget.__class__.__name__,
            widget_name=widget.objectName() or "",
            widget_path=self._get_widget_path(widget),
            pos_x=pos.x(),
            pos_y=pos.y(),
            global_x=global_pos.x(),
            global_y=global_pos.y(),
        )
        self._buffer.append(action)

    def _get_widget_path(self, widget: QWidget) -> str:
        """Get hierarchical path to widget for identification."""
        parts = []
        current = widget
        while current:
            name = current.objectName() or current.__class__.__name__
            parts.append(name)
            current = current.parent()
            if len(parts) > 10:  # Prevent infinite loops
                break
        return "/".join(reversed(parts))

    def _get_widget_text(self, widget: QWidget) -> Optional[str]:
        """Extract meaningful text from widget if available."""
        try:
            # Try common text methods
            if hasattr(widget, 'text') and callable(widget.text):
                text = widget.text()
                if text and isinstance(text, str):
                    return text[:100]  # Limit length
            if hasattr(widget, 'currentText') and callable(widget.currentText):
                return widget.currentText()[:100]
            if hasattr(widget, 'windowTitle') and callable(widget.windowTitle):
                title = widget.windowTitle()
                if title:
                    return title[:100]
        except Exception:
            pass
        return None

    def to_json_lines(self, actions: Optional[List[UIAction]] = None) -> str:
        """Export actions as JSON Lines format."""
        if actions is None:
            actions = self.get_all_actions()

        lines = []
        for action in actions:
            lines.append(json.dumps(action.to_dict()))
        return "\n".join(lines)

    def to_qtbot_script(self, actions: Optional[List[UIAction]] = None) -> str:
        """Generate a qtbot test script from recorded actions."""
        if actions is None:
            actions = self.get_all_actions()

        lines = [
            '"""',
            'Auto-generated qtbot test from UI action recording.',
            f'Generated: {datetime.now().isoformat()}',
            f'Actions: {len(actions)}',
            '"""',
            '',
            'import pytest',
            'from PyQt6.QtCore import Qt, QPoint',
            'from PyQt6.QtWidgets import QWidget',
            '',
            '',
            'def test_replay_ui_actions(qtbot, main_window):',
            '    """Replay recorded UI actions."""',
            '    window = main_window',
            '    ',
        ]

        for action in actions:
            code = action.to_qtbot_code()
            for line in code.split('\n'):
                lines.append(f'    {line}')
            lines.append('    ')

        lines.append('    # End of recorded actions')
        return '\n'.join(lines)

    def save_to_file(self, filepath: Path, format: str = 'jsonl'):
        """Save recorded actions to file."""
        actions = self.get_all_actions()

        if format == 'jsonl':
            content = self.to_json_lines(actions)
        elif format == 'qtbot':
            content = self.to_qtbot_script(actions)
        else:
            raise ValueError(f"Unknown format: {format}")

        filepath.write_text(content)
        print(f"[UIActionRecorder] Saved {len(actions)} actions to {filepath}")

    def get_crash_report_data(self) -> Dict[str, Any]:
        """Get data formatted for crash reports."""
        actions = self.get_recent_actions(50)

        return {
            'action_count': len(actions),
            'buffer_size': len(self._buffer),
            'actions': [a.to_dict() for a in actions],
            'summary': self._summarize_actions(actions),
        }

    def _summarize_actions(self, actions: List[UIAction]) -> str:
        """Create human-readable summary of recent actions."""
        if not actions:
            return "No recent actions"

        lines = []
        for action in actions[-10:]:  # Last 10 for summary
            dt = datetime.fromtimestamp(action.timestamp)
            time_str = dt.strftime('%H:%M:%S.%f')[:-3]

            if action.action_type == ActionType.MOUSE_CLICK.value:
                text = f"[{time_str}] Click on {action.widget_class}"
                if action.widget_text:
                    text += f" '{action.widget_text[:30]}'"
            elif action.action_type == ActionType.MOUSE_DOUBLE_CLICK.value:
                text = f"[{time_str}] Double-click on {action.widget_class}"
            elif action.action_type == ActionType.KEY_PRESS.value:
                key_desc = action.text or action.key
                text = f"[{time_str}] Key '{key_desc}' in {action.widget_class}"
            elif action.action_type == ActionType.CONTEXT_MENU.value:
                text = f"[{time_str}] Right-click menu on {action.widget_class}"
            elif action.action_type == ActionType.WHEEL.value:
                text = f"[{time_str}] Scroll ({action.delta_y}) on {action.widget_class}"
            else:
                text = f"[{time_str}] {action.action_type} on {action.widget_class}"

            lines.append(text)

        return "\n".join(lines)


# Singleton accessor
def get_ui_action_recorder() -> UIActionRecorder:
    """Get the global UI action recorder instance."""
    return UIActionRecorder.instance()
