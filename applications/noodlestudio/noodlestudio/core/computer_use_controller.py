"""
Computer Use Controller - Enable Claude to see and interact with NoodleStudio

Provides screenshot capture and input injection for Claude's Computer Use capability.
Claude can see the UI, move the mouse, click, type, and press keys.

THREAD SAFETY:
    All Qt GUI operations run on the main thread via QMetaObject.invokeMethod.
    This allows safe calls from AsyncWorker threads.

Architecture:
    Screenshot: QWidget.grab() -> QBuffer -> base64 PNG
    Input: QTest synthetic events injected into Qt event loop
    Coordinates: Relative to main window (0,0 = top-left of window)

Author: Caitlyn + Claude (NinaK edition)
Date: January 2, 2026
"""

import base64
import threading
from io import BytesIO
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from PyQt6.QtCore import (
    Qt, QPoint, QPointF, QBuffer, QIODevice, QTimer, QByteArray,
    QObject, pyqtSignal, pyqtSlot, QThread, QMetaObject,
    Q_ARG, QGenericArgument
)
from PyQt6.QtGui import QCursor, QKeySequence, QMouseEvent, QKeyEvent, QWheelEvent, QPainter, QFont, QPen, QColor
from PyQt6.QtWidgets import QApplication, QWidget
from PyQt6.QtTest import QTest


class MouseButton(Enum):
    """Mouse button types."""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass
class ComputerUseAction:
    """Record of a computer use action."""
    action: str
    timestamp: datetime = field(default_factory=datetime.now)
    coordinate: Optional[Tuple[int, int]] = None
    text: Optional[str] = None
    key: Optional[str] = None
    button: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


class ComputerUseController(QObject):
    """
    Controller for Claude Computer Use integration.

    THREAD SAFE: All GUI operations are marshaled to main thread.

    Enables Claude to:
    - Take screenshots of NoodleStudio
    - Move the mouse cursor
    - Click (left, right, middle, double)
    - Type text
    - Press key combinations
    - Scroll

    All coordinates are relative to the main window's top-left corner.
    """

    # Singleton
    _instance: Optional['ComputerUseController'] = None

    # Signals for thread-safe operations
    _screenshotRequested = pyqtSignal()
    _clickRequested = pyqtSignal(int, int, str)
    _doubleClickRequested = pyqtSignal(int, int, str)
    _typeRequested = pyqtSignal(str)
    _keyRequested = pyqtSignal(str)
    _scrollRequested = pyqtSignal(int, int, int, int)
    _mouseMoveRequested = pyqtSignal(int, int)

    # Key name mapping for common keys
    KEY_MAP = {
        'return': Qt.Key.Key_Return,
        'enter': Qt.Key.Key_Return,
        'tab': Qt.Key.Key_Tab,
        'escape': Qt.Key.Key_Escape,
        'esc': Qt.Key.Key_Escape,
        'backspace': Qt.Key.Key_Backspace,
        'delete': Qt.Key.Key_Delete,
        'del': Qt.Key.Key_Delete,
        'space': Qt.Key.Key_Space,
        'up': Qt.Key.Key_Up,
        'down': Qt.Key.Key_Down,
        'left': Qt.Key.Key_Left,
        'right': Qt.Key.Key_Right,
        'home': Qt.Key.Key_Home,
        'end': Qt.Key.Key_End,
        'pageup': Qt.Key.Key_PageUp,
        'pagedown': Qt.Key.Key_PageDown,
        'f1': Qt.Key.Key_F1,
        'f2': Qt.Key.Key_F2,
        'f3': Qt.Key.Key_F3,
        'f4': Qt.Key.Key_F4,
        'f5': Qt.Key.Key_F5,
        'f6': Qt.Key.Key_F6,
        'f7': Qt.Key.Key_F7,
        'f8': Qt.Key.Key_F8,
        'f9': Qt.Key.Key_F9,
        'f10': Qt.Key.Key_F10,
        'f11': Qt.Key.Key_F11,
        'f12': Qt.Key.Key_F12,
    }

    def __init__(self):
        super().__init__()
        self._main_window: Optional[QWidget] = None
        self._action_history: List[ComputerUseAction] = []
        self._max_history = 100

        # Demo mode - shows ghost cursor with beautiful animations
        self._demo_mode = False
        self._ghost_controller = None

        # Thread synchronization for results
        self._result_lock = threading.Lock()
        self._screenshot_result: Optional[Tuple[str, int, int]] = None
        self._operation_result: Optional[bool] = None
        self._operation_error: Optional[str] = None

        # Connect signals to slots (runs on main thread)
        self._screenshotRequested.connect(self._do_screenshot)
        self._clickRequested.connect(self._do_click)
        self._doubleClickRequested.connect(self._do_double_click)
        self._typeRequested.connect(self._do_type)
        self._keyRequested.connect(self._do_key)
        self._scrollRequested.connect(self._do_scroll)
        self._mouseMoveRequested.connect(self._do_mouse_move)

    @classmethod
    def instance(cls) -> 'ComputerUseController':
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = ComputerUseController()
        return cls._instance

    def set_main_window(self, window: QWidget):
        """Set the main window to control."""
        self._main_window = window
        # Move controller to main thread to ensure signal delivery
        self.moveToThread(QApplication.instance().thread())

        # Set up ghost cursor for demo mode
        from .ghost_cursor import setup_ghost_cursor
        self._ghost_controller = setup_ghost_cursor(window)

        print("[ComputerUse] Main window set")

    @property
    def main_window(self) -> Optional[QWidget]:
        """Get the main window."""
        return self._main_window

    @property
    def demo_mode(self) -> bool:
        """Check if demo mode (visible ghost cursor) is enabled."""
        return self._demo_mode

    @demo_mode.setter
    def demo_mode(self, enabled: bool):
        """Enable or disable demo mode with visible ghost cursor."""
        self._demo_mode = enabled
        if self._ghost_controller:
            self._ghost_controller.set_demo_mode(enabled)
        print(f"[ComputerUse] Demo mode: {'enabled' if enabled else 'disabled'}")

    def _is_main_thread(self) -> bool:
        """Check if we're on the main thread."""
        app = QApplication.instance()
        if not app:
            return True
        return QThread.currentThread() == app.thread()

    def _record_action(self, action: ComputerUseAction):
        """Record an action to history."""
        self._action_history.append(action)
        if len(self._action_history) > self._max_history:
            self._action_history = self._action_history[-self._max_history:]

    # =========================================================================
    # SCREENSHOT - Thread safe
    # =========================================================================

    def screenshot(self, scale: float = 1.0, add_rulers: bool = None) -> Tuple[str, int, int]:
        """
        Capture screenshot of NoodleStudio window. THREAD SAFE.

        Args:
            scale: Scale factor (1.0 = full size, 0.5 = half size)
            add_rulers: If True, draw coordinate rulers. Default: True in demo mode.

        Returns:
            Tuple of (base64_png, width, height)
        """
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # Default to rulers in demo mode to help with coordinate targeting
        if add_rulers is None:
            add_rulers = self._demo_mode

        if self._is_main_thread():
            return self._screenshot_impl(scale, add_rulers)

        # Call from worker thread - marshal to main thread using invokeMethod
        # with BlockingQueuedConnection for synchronous execution
        with self._result_lock:
            self._screenshot_result = None
            self._operation_error = None
            self._screenshot_scale = scale
            self._screenshot_rulers = add_rulers

        # BlockingQueuedConnection waits for the slot to complete
        QMetaObject.invokeMethod(
            self,
            "_do_screenshot_sync",
            Qt.ConnectionType.BlockingQueuedConnection
        )

        with self._result_lock:
            if self._operation_error:
                raise RuntimeError(self._operation_error)
            if self._screenshot_result is None:
                raise RuntimeError("Screenshot failed - no result")
            return self._screenshot_result

    def _screenshot_impl(self, scale: float = 1.0, add_rulers: bool = False) -> Tuple[str, int, int]:
        """Internal screenshot implementation (must run on main thread).

        Args:
            scale: Scale factor for the screenshot
            add_rulers: If True, draw coordinate rulers on edges to help with click targeting
        """
        # Grab the window
        pixmap = self._main_window.grab()

        # Get device pixel ratio for HiDPI/Retina displays
        # The pixmap is in device pixels, but click coordinates need logical pixels
        device_ratio = pixmap.devicePixelRatio()

        # IMPORTANT: Scale to logical pixels so coordinates match click targets
        # On Retina (2x), a 2000px wide screenshot becomes 1000 logical pixels
        # This ensures that pixel coordinates in the image == click coordinates
        logical_width = int(pixmap.width() / device_ratio)
        logical_height = int(pixmap.height() / device_ratio)

        # Always scale to logical coordinates for consistent click targeting
        if device_ratio != 1.0:
            pixmap = pixmap.scaled(
                logical_width, logical_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        # Apply additional user-requested scaling
        if scale != 1.0:
            new_width = int(logical_width * scale)
            new_height = int(logical_height * scale)
            pixmap = pixmap.scaled(
                new_width, new_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            logical_width = new_width
            logical_height = new_height

        # Draw coordinate rulers if requested (helps debug click targeting)
        if add_rulers:
            pixmap = self._add_coordinate_rulers(pixmap)

        # Convert to PNG bytes
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()

        # Encode to base64
        png_bytes = buffer.data().data()
        b64_data = base64.b64encode(png_bytes).decode('utf-8')

        action = ComputerUseAction(
            action="screenshot",
            coordinate=(logical_width, logical_height)
        )
        self._record_action(action)

        print(f"[ComputerUse] Screenshot: {logical_width}x{logical_height} "
              f"(coordinates match click targets)")

        return b64_data, logical_width, logical_height

    def get_ui_element_map(self) -> List[Dict[str, Any]]:
        """
        Query Qt widget tree and return a map of clickable UI elements.

        Returns list of elements with:
        - name: Human-readable name
        - type: Widget type (tab, button, menu, etc.)
        - x, y: Center coordinates for clicking
        - bounds: (x, y, width, height) bounding box
        """
        if not self._main_window:
            return []

        elements = []

        def add_element(name: str, elem_type: str, widget: QWidget):
            """Add an element to the map with its coordinates."""
            if not widget.isVisible():
                return

            # Get global position and convert to window-relative
            global_pos = widget.mapToGlobal(QPoint(0, 0))
            window_pos = self._main_window.mapFromGlobal(global_pos)

            # Center point for clicking
            center_x = window_pos.x() + widget.width() // 2
            center_y = window_pos.y() + widget.height() // 2

            elements.append({
                "name": name,
                "type": elem_type,
                "x": center_x,
                "y": center_y,
                "bounds": (window_pos.x(), window_pos.y(), widget.width(), widget.height())
            })

        def scan_widget(widget: QWidget, depth: int = 0):
            """Recursively scan widget tree for clickable elements."""
            if depth > 10 or not widget.isVisible():
                return

            from PyQt6.QtWidgets import (
                QPushButton, QToolButton, QTabBar, QMenuBar,
                QComboBox, QCheckBox, QRadioButton, QLineEdit,
                QTextEdit, QPlainTextEdit, QSpinBox, QSlider
            )

            widget_name = widget.objectName() or widget.__class__.__name__

            # Tab bars - get individual tabs
            if isinstance(widget, QTabBar):
                for i in range(widget.count()):
                    tab_rect = widget.tabRect(i)
                    tab_text = widget.tabText(i)
                    if tab_text:
                        # Get tab center in window coordinates
                        global_pos = widget.mapToGlobal(QPoint(
                            tab_rect.x() + tab_rect.width() // 2,
                            tab_rect.y() + tab_rect.height() // 2
                        ))
                        window_pos = self._main_window.mapFromGlobal(global_pos)
                        elements.append({
                            "name": f"Tab: {tab_text}",
                            "type": "tab",
                            "x": window_pos.x(),
                            "y": window_pos.y(),
                            "bounds": (
                                window_pos.x() - tab_rect.width() // 2,
                                window_pos.y() - tab_rect.height() // 2,
                                tab_rect.width(),
                                tab_rect.height()
                            )
                        })

            # Buttons
            elif isinstance(widget, (QPushButton, QToolButton)):
                text = widget.text() or widget.toolTip() or widget_name
                if text:
                    add_element(f"Button: {text}", "button", widget)

            # Input fields
            elif isinstance(widget, QLineEdit):
                placeholder = widget.placeholderText() or widget_name
                add_element(f"Input: {placeholder}", "input", widget)

            # Combo boxes
            elif isinstance(widget, QComboBox):
                add_element(f"Dropdown: {widget_name}", "dropdown", widget)

            # Checkboxes
            elif isinstance(widget, QCheckBox):
                text = widget.text() or widget_name
                add_element(f"Checkbox: {text}", "checkbox", widget)

            # Recurse into children
            for child in widget.children():
                if isinstance(child, QWidget):
                    scan_widget(child, depth + 1)

        # Start scanning from main window
        scan_widget(self._main_window)

        # Sort by position (top-to-bottom, left-to-right)
        elements.sort(key=lambda e: (e["y"], e["x"]))

        return elements

    def get_ui_summary(self) -> str:
        """Get a text summary of clickable UI elements with coordinates."""
        elements = self.get_ui_element_map()

        if not elements:
            return "No UI elements found."

        lines = ["CLICKABLE UI ELEMENTS (name -> click at x,y):"]

        # Group by type
        tabs = [e for e in elements if e["type"] == "tab"]
        buttons = [e for e in elements if e["type"] == "button"]
        inputs = [e for e in elements if e["type"] == "input"]
        others = [e for e in elements if e["type"] not in ("tab", "button", "input")]

        if tabs:
            lines.append("\nTABS:")
            for e in tabs:
                lines.append(f"  {e['name']} -> ({e['x']}, {e['y']})")

        if buttons:
            lines.append("\nBUTTONS:")
            for e in buttons[:20]:  # Limit to avoid overwhelming
                lines.append(f"  {e['name']} -> ({e['x']}, {e['y']})")
            if len(buttons) > 20:
                lines.append(f"  ... and {len(buttons) - 20} more buttons")

        if inputs:
            lines.append("\nINPUT FIELDS:")
            for e in inputs:
                lines.append(f"  {e['name']} -> ({e['x']}, {e['y']})")

        if others:
            lines.append("\nOTHER:")
            for e in others[:10]:
                lines.append(f"  {e['name']} -> ({e['x']}, {e['y']})")

        return "\n".join(lines)

    def _add_calibration_pattern(self, pixmap):
        """Draw calibration crosshairs at known coordinates.

        Draws crosshairs at specific positions so the model can verify
        its coordinate reading accuracy.
        """
        from PyQt6.QtGui import QPixmap

        result = QPixmap(pixmap)
        painter = QPainter(result)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = result.width()
        height = result.height()

        # Define calibration points
        calibration_points = [
            (100, 100, "A"),
            (width // 2, 100, "B"),
            (width - 100, 100, "C"),
            (100, height // 2, "D"),
            (width // 2, height // 2, "E"),  # Center
            (width - 100, height // 2, "F"),
            (100, height - 100, "G"),
            (width // 2, height - 100, "H"),
            (width - 100, height - 100, "I"),
        ]

        font = QFont("SF Mono", 12)
        font.setBold(True)
        painter.setFont(font)

        for x, y, label in calibration_points:
            # Draw crosshair
            painter.setPen(QPen(QColor(255, 0, 0, 255), 2))  # Red
            arm_length = 15
            painter.drawLine(x - arm_length, y, x + arm_length, y)  # Horizontal
            painter.drawLine(x, y - arm_length, x, y + arm_length)  # Vertical

            # Draw circle at center
            painter.setPen(QPen(QColor(255, 255, 0, 255), 2))  # Yellow
            painter.drawEllipse(QPointF(x, y), 5, 5)

            # Draw label with coordinates
            painter.setPen(QColor(255, 255, 255, 255))
            label_text = f"{label}({x},{y})"
            painter.drawText(x + 10, y - 10, label_text)

        painter.end()
        return result

    def screenshot_with_calibration(self) -> Tuple[str, int, int, list]:
        """Take screenshot with calibration pattern overlaid.

        Returns:
            Tuple of (base64_png, width, height, calibration_points)
            calibration_points is a list of (x, y, label) tuples
        """
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # Get base screenshot at logical coordinates
        pixmap = self._main_window.grab()
        device_ratio = pixmap.devicePixelRatio()
        logical_width = int(pixmap.width() / device_ratio)
        logical_height = int(pixmap.height() / device_ratio)

        if device_ratio != 1.0:
            pixmap = pixmap.scaled(
                logical_width, logical_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

        # Define calibration points
        calibration_points = [
            (100, 100, "A"),
            (logical_width // 2, 100, "B"),
            (logical_width - 100, 100, "C"),
            (100, logical_height // 2, "D"),
            (logical_width // 2, logical_height // 2, "E"),
            (logical_width - 100, logical_height // 2, "F"),
            (100, logical_height - 100, "G"),
            (logical_width // 2, logical_height - 100, "H"),
            (logical_width - 100, logical_height - 100, "I"),
        ]

        # Add calibration pattern
        pixmap = self._add_calibration_pattern(pixmap)

        # Also add rulers
        pixmap = self._add_coordinate_rulers(pixmap)

        # Convert to base64
        buffer = QBuffer()
        buffer.open(QIODevice.OpenModeFlag.WriteOnly)
        pixmap.save(buffer, "PNG")
        buffer.close()

        b64_data = base64.b64encode(buffer.data().data()).decode('utf-8')

        print(f"[ComputerUse] Calibration screenshot: {logical_width}x{logical_height}")
        return b64_data, logical_width, logical_height, calibration_points

    def _add_coordinate_rulers(self, pixmap):
        """Draw coordinate rulers on the screenshot edges.

        Draws prominent tick marks and labels to help the vision model
        accurately identify pixel coordinates for clicking.
        """
        from PyQt6.QtGui import QPixmap

        # Create a mutable copy
        result = QPixmap(pixmap)
        painter = QPainter(result)

        # Set up drawing
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("SF Mono", 11)  # Larger font for visibility
        font.setBold(True)
        painter.setFont(font)

        width = result.width()
        height = result.height()

        # Ruler dimensions - larger for visibility
        ruler_height = 24  # Taller horizontal ruler
        ruler_width = 32   # Wider vertical ruler

        # Colors
        ruler_bg = QColor(0, 0, 0, 200)  # More opaque
        tick_color = QColor(255, 255, 0, 255)  # Bright yellow ticks
        text_color = QColor(255, 255, 255, 255)  # Bright white text

        # Draw top ruler background
        painter.fillRect(0, 0, width, ruler_height, ruler_bg)

        # Draw left ruler background
        painter.fillRect(0, 0, ruler_width, height, ruler_bg)

        # Draw horizontal ticks and labels (every 50px)
        for x in range(0, width, 50):
            painter.setPen(QPen(tick_color, 2))
            # Tick mark
            painter.drawLine(x, ruler_height - 6, x, ruler_height)
            if x % 100 == 0:
                # Longer tick at 100px intervals
                painter.drawLine(x, ruler_height - 10, x, ruler_height)
                # Label
                painter.setPen(text_color)
                painter.drawText(x + 3, 16, str(x))

        # Draw vertical ticks and labels (every 50px)
        for y in range(0, height, 50):
            painter.setPen(QPen(tick_color, 2))
            # Tick mark
            painter.drawLine(ruler_width - 6, y, ruler_width, y)
            if y % 100 == 0:
                # Longer tick at 100px intervals
                painter.drawLine(ruler_width - 10, y, ruler_width, y)
                # Label
                painter.setPen(text_color)
                painter.drawText(2, y + 14, str(y))

        # Draw corner with "X,Y" label
        painter.fillRect(0, 0, ruler_width, ruler_height, QColor(40, 40, 40, 220))
        painter.setPen(QColor(150, 150, 150))
        painter.drawText(4, 16, "X→")
        painter.drawText(4, ruler_height + 14, "Y↓")

        painter.end()
        return result

    @pyqtSlot()
    def _do_screenshot(self):
        """Slot for screenshot signal."""
        try:
            result = self._screenshot_impl(1.0)
            with self._result_lock:
                self._screenshot_result = result
        except Exception as e:
            with self._result_lock:
                self._operation_error = str(e)

    @pyqtSlot()
    def _do_screenshot_sync(self):
        """Slot for synchronous screenshot via BlockingQueuedConnection."""
        try:
            scale = getattr(self, '_screenshot_scale', 1.0)
            add_rulers = getattr(self, '_screenshot_rulers', False)
            result = self._screenshot_impl(scale, add_rulers)
            with self._result_lock:
                self._screenshot_result = result
        except Exception as e:
            with self._result_lock:
                self._operation_error = str(e)

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _get_widget_at(self, x: int, y: int) -> Optional[QWidget]:
        """Get the widget at window-relative coordinates."""
        if not self._main_window:
            return None
        global_pos = self._main_window.mapToGlobal(QPoint(x, y))
        widget = QApplication.widgetAt(global_pos)
        return widget

    def _qt_button(self, button: str) -> Qt.MouseButton:
        """Convert button string to Qt enum."""
        buttons = {
            'left': Qt.MouseButton.LeftButton,
            'right': Qt.MouseButton.RightButton,
            'middle': Qt.MouseButton.MiddleButton,
        }
        return buttons.get(button.lower(), Qt.MouseButton.LeftButton)

    def _run_on_main_thread(self, func) -> bool:
        """Run a function on main thread and return success."""
        if self._is_main_thread():
            try:
                func()
                return True
            except Exception as e:
                print(f"[ComputerUse] Error: {e}")
                return False

        event = threading.Event()
        success = [False]
        error = [None]

        def on_main_thread():
            try:
                func()
                success[0] = True
            except Exception as e:
                error[0] = str(e)
            finally:
                event.set()

        # Use invokeMethod to properly marshal to main thread
        QMetaObject.invokeMethod(
            self, "_execute_on_main",
            Qt.ConnectionType.QueuedConnection,
            Q_ARG(object, on_main_thread)
        )
        event.wait(timeout=5.0)

        if error[0]:
            print(f"[ComputerUse] Error: {error[0]}")
        return success[0]

    @pyqtSlot(object)
    def _execute_on_main(self, func):
        """Execute a function on the main thread. Slot for cross-thread calls."""
        if func:
            func()

    # =========================================================================
    # MOUSE MOVE - Thread safe
    # =========================================================================

    def _run_animation_on_main(self, animation_func, timeout: float = 2.0) -> bool:
        """Run a ghost cursor animation on main thread and wait for completion."""
        if not self._ghost_controller:
            return True

        animation_done = threading.Event()

        def on_complete():
            animation_done.set()

        def start_animation():
            animation_func(on_complete)

        # Marshal to main thread properly
        if self._is_main_thread():
            start_animation()
        else:
            QMetaObject.invokeMethod(
                self, "_execute_on_main",
                Qt.ConnectionType.QueuedConnection,
                Q_ARG(object, start_animation)
            )

        return animation_done.wait(timeout=timeout)

    def mouse_move(self, x: int, y: int) -> bool:
        """Move mouse cursor to coordinates. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # If demo mode, animate ghost cursor first
        if self._demo_mode and self._ghost_controller:
            self._run_animation_on_main(
                lambda cb: self._ghost_controller.visualize_move(x, y, cb),
                timeout=2.0
            )

        def do_move():
            global_pos = self._main_window.mapToGlobal(QPoint(x, y))
            QCursor.setPos(global_pos)
            action = ComputerUseAction(action="mouse_move", coordinate=(x, y))
            self._record_action(action)
            print(f"[ComputerUse] Mouse move to ({x}, {y})")

        return self._run_on_main_thread(do_move)

    @pyqtSlot(int, int)
    def _do_mouse_move(self, x: int, y: int):
        """Slot for mouse move signal."""
        self.mouse_move(x, y)

    # =========================================================================
    # CLICK - Thread safe
    # =========================================================================

    def click(self, x: int, y: int, button: str = "left") -> bool:
        """Click at coordinates. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # If demo mode, animate ghost cursor first (moves AND shows ripple)
        if self._demo_mode and self._ghost_controller:
            self._run_animation_on_main(
                lambda cb: self._ghost_controller.visualize_click(x, y, button, cb),
                timeout=3.0
            )

        def do_click():
            # Move cursor
            global_pos = self._main_window.mapToGlobal(QPoint(x, y))
            QCursor.setPos(global_pos)
            QApplication.processEvents()

            # Find widget
            widget = self._get_widget_at(x, y)
            if not widget:
                widget = self._main_window

            # Calculate local position
            local_pos = widget.mapFromGlobal(global_pos)

            # Click
            qt_button = self._qt_button(button)
            QTest.mouseClick(widget, qt_button, Qt.KeyboardModifier.NoModifier, local_pos)

            action = ComputerUseAction(action="click", coordinate=(x, y), button=button)
            self._record_action(action)
            print(f"[ComputerUse] Click ({button}) at ({x}, {y}) on {widget.__class__.__name__}")

        return self._run_on_main_thread(do_click)

    @pyqtSlot(int, int, str)
    def _do_click(self, x: int, y: int, button: str):
        """Slot for click signal."""
        self.click(x, y, button)

    # =========================================================================
    # DOUBLE CLICK - Thread safe
    # =========================================================================

    def double_click(self, x: int, y: int, button: str = "left") -> bool:
        """Double-click at coordinates. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # If demo mode, animate ghost cursor first (with double ripple)
        if self._demo_mode and self._ghost_controller:
            self._run_animation_on_main(
                lambda cb: self._ghost_controller.visualize_double_click(x, y, cb),
                timeout=3.0
            )

        def do_double_click():
            global_pos = self._main_window.mapToGlobal(QPoint(x, y))
            QCursor.setPos(global_pos)
            QApplication.processEvents()

            widget = self._get_widget_at(x, y)
            if not widget:
                widget = self._main_window

            local_pos = widget.mapFromGlobal(global_pos)
            qt_button = self._qt_button(button)
            QTest.mouseDClick(widget, qt_button, Qt.KeyboardModifier.NoModifier, local_pos)

            action = ComputerUseAction(action="double_click", coordinate=(x, y), button=button)
            self._record_action(action)
            print(f"[ComputerUse] Double-click ({button}) at ({x}, {y})")

        return self._run_on_main_thread(do_double_click)

    @pyqtSlot(int, int, str)
    def _do_double_click(self, x: int, y: int, button: str):
        """Slot for double click signal."""
        self.double_click(x, y, button)

    # =========================================================================
    # DRAG - Thread safe
    # =========================================================================

    def drag(self, start_x: int, start_y: int, end_x: int, end_y: int, button: str = "left") -> bool:
        """Drag from start to end coordinates. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        # If demo mode, animate the entire drag sequence
        if self._demo_mode and self._ghost_controller:
            self._run_animation_on_main(
                lambda cb: self._ghost_controller.visualize_drag(
                    start_x, start_y, end_x, end_y, cb
                ),
                timeout=5.0
            )

        def do_drag():
            # Move to start
            global_start = self._main_window.mapToGlobal(QPoint(start_x, start_y))
            QCursor.setPos(global_start)
            QApplication.processEvents()

            widget = self._get_widget_at(start_x, start_y)
            if not widget:
                widget = self._main_window

            local_start = widget.mapFromGlobal(global_start)
            qt_button = self._qt_button(button)

            # Press
            QTest.mousePress(widget, qt_button, Qt.KeyboardModifier.NoModifier, local_start)
            QApplication.processEvents()

            # Move in steps
            steps = 10
            for i in range(1, steps + 1):
                t = i / steps
                ix = int(start_x + (end_x - start_x) * t)
                iy = int(start_y + (end_y - start_y) * t)
                gp = self._main_window.mapToGlobal(QPoint(ix, iy))
                QCursor.setPos(gp)
                QApplication.processEvents()

            # Release
            global_end = self._main_window.mapToGlobal(QPoint(end_x, end_y))
            end_widget = self._get_widget_at(end_x, end_y) or widget
            local_end = end_widget.mapFromGlobal(global_end)
            QTest.mouseRelease(end_widget, qt_button, Qt.KeyboardModifier.NoModifier, local_end)

            action = ComputerUseAction(
                action="drag",
                coordinate=(start_x, start_y),
                text=f"to ({end_x}, {end_y})",
                button=button
            )
            self._record_action(action)
            print(f"[ComputerUse] Drag from ({start_x}, {start_y}) to ({end_x}, {end_y})")

        return self._run_on_main_thread(do_drag)

    # =========================================================================
    # TYPE TEXT - Thread safe
    # =========================================================================

    def type_text(self, text: str) -> bool:
        """Type text into the currently focused widget. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        def do_type():
            widget = QApplication.focusWidget()
            if not widget:
                widget = self._main_window
            QTest.keyClicks(widget, text)

            action = ComputerUseAction(action="type", text=text)
            self._record_action(action)
            display_text = text[:50] + "..." if len(text) > 50 else text
            print(f"[ComputerUse] Type: '{display_text}'")

        return self._run_on_main_thread(do_type)

    @pyqtSlot(str)
    def _do_type(self, text: str):
        """Slot for type signal."""
        self.type_text(text)

    # =========================================================================
    # KEY PRESS - Thread safe
    # =========================================================================

    def _parse_key_combo(self, key_string: str) -> Tuple[Qt.Key, Qt.KeyboardModifiers]:
        """Parse a key combination string like 'ctrl+shift+s'."""
        parts = key_string.lower().split('+')
        modifiers = Qt.KeyboardModifier.NoModifier
        key = None

        for part in parts:
            part = part.strip()
            if part in ('ctrl', 'control'):
                modifiers |= Qt.KeyboardModifier.ControlModifier
            elif part in ('shift',):
                modifiers |= Qt.KeyboardModifier.ShiftModifier
            elif part in ('alt', 'option'):
                modifiers |= Qt.KeyboardModifier.AltModifier
            elif part in ('meta', 'cmd', 'command', 'super'):
                modifiers |= Qt.KeyboardModifier.MetaModifier
            elif part in self.KEY_MAP:
                key = self.KEY_MAP[part]
            elif len(part) == 1:
                key = Qt.Key(ord(part.upper()))
            else:
                print(f"[ComputerUse] Unknown key: {part}")

        if key is None:
            key = Qt.Key.Key_unknown

        return key, modifiers

    def key(self, key_combo: str) -> bool:
        """Press a key or key combination. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        def do_key():
            widget = QApplication.focusWidget()
            if not widget:
                widget = self._main_window
            qt_key, modifiers = self._parse_key_combo(key_combo)
            QTest.keyClick(widget, qt_key, modifiers)

            action = ComputerUseAction(action="key", key=key_combo)
            self._record_action(action)
            print(f"[ComputerUse] Key: {key_combo}")

        return self._run_on_main_thread(do_key)

    @pyqtSlot(str)
    def _do_key(self, key_combo: str):
        """Slot for key signal."""
        self.key(key_combo)

    # =========================================================================
    # SCROLL - Thread safe
    # =========================================================================

    def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = -120) -> bool:
        """Scroll at coordinates. THREAD SAFE."""
        if not self._main_window:
            raise RuntimeError("Main window not set")

        def do_scroll():
            global_pos = self._main_window.mapToGlobal(QPoint(x, y))
            QCursor.setPos(global_pos)
            QApplication.processEvents()

            widget = self._get_widget_at(x, y)
            if not widget:
                widget = self._main_window

            local_pos = widget.mapFromGlobal(global_pos)

            from PyQt6.QtCore import QPointF
            event = QWheelEvent(
                QPointF(local_pos),
                QPointF(global_pos),
                QPoint(delta_x, delta_y),
                QPoint(delta_x, delta_y),
                Qt.MouseButton.NoButton,
                Qt.KeyboardModifier.NoModifier,
                Qt.ScrollPhase.NoScrollPhase,
                False
            )
            QApplication.sendEvent(widget, event)

            action = ComputerUseAction(
                action="scroll",
                coordinate=(x, y),
                text=f"delta=({delta_x}, {delta_y})"
            )
            self._record_action(action)
            print(f"[ComputerUse] Scroll at ({x}, {y}) delta=({delta_x}, {delta_y})")

        return self._run_on_main_thread(do_scroll)

    @pyqtSlot(int, int, int, int)
    def _do_scroll(self, x: int, y: int, delta_x: int, delta_y: int):
        """Slot for scroll signal."""
        self.scroll(x, y, delta_x, delta_y)

    # =========================================================================
    # UTILITY
    # =========================================================================

    def get_action_history(self) -> List[Dict[str, Any]]:
        """Get recent action history."""
        return [
            {
                'action': a.action,
                'timestamp': a.timestamp.isoformat(),
                'coordinate': a.coordinate,
                'text': a.text,
                'key': a.key,
                'button': a.button,
                'success': a.success,
                'error': a.error,
            }
            for a in self._action_history
        ]

    def get_window_size(self) -> Tuple[int, int]:
        """Get main window dimensions."""
        if not self._main_window:
            return (0, 0)
        return (self._main_window.width(), self._main_window.height())


# Singleton accessor
def get_computer_use_controller() -> ComputerUseController:
    """Get the global ComputerUseController instance."""
    return ComputerUseController.instance()
