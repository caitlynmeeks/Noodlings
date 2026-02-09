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
#   Guide Performance Window
#
#   Pure renderer for a noodling's performance. Displays VRM
#   character, receives text and affect from a facet assembly.
#   Does NOT make LLM calls. Does NOT contain personality prompts.
#   All cognition happens in the assembly, orchestrated by
#   GuidePerformanceManager.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.runtime.ui.guide_performance_window
# PURPOSE:  Floating Guide Dialogue and VRM Panel
# LAYER:    Studio / UI Runtime
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   GuidePerformanceWindow
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QFrame
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal
    from PyQt6.QtGui import (
        QTextCursor, QColor, QTextCharFormat, QMouseEvent
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# =============================================================================
# Draggable Header
# =============================================================================

if QT_AVAILABLE:

    class _DraggableHeader(QLabel):
        """Header label that supports window dragging.

        Coordinates with GuidePerformanceWindow to pause follow-parent
        tracking during drag and recalculate offset on release so the
        window stays at the user-chosen position relative to the parent.
        """

        closeClicked = pyqtSignal()
        dragStarted = pyqtSignal()
        dragFinished = pyqtSignal()

        def __init__(self, text: str = "", parent=None):
            super().__init__(text, parent)
            self.drag_position = None

        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self.drag_position = (
                    event.globalPosition().toPoint()
                    - self.window().frameGeometry().topLeft()
                )
                self.dragStarted.emit()
            super().mousePressEvent(event)

        def mouseMoveEvent(self, event: QMouseEvent):
            if (event.buttons() == Qt.MouseButton.LeftButton
                    and self.drag_position is not None):
                self.window().move(
                    event.globalPosition().toPoint() - self.drag_position
                )
            super().mouseMoveEvent(event)

        def mouseReleaseEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.LeftButton:
                self.drag_position = None
                self.dragFinished.emit()
            super().mouseReleaseEvent(event)


# =============================================================================
# Thinking Indicator (minimal variant)
# =============================================================================

if QT_AVAILABLE:

    class _ThinkingIndicator(QFrame):
        """Compact thinking indicator with pulsing dot."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFrameStyle(QFrame.Shape.NoFrame)
            self._pulse_state = 0

            layout = QHBoxLayout(self)
            layout.setContentsMargins(8, 4, 8, 4)
            layout.setSpacing(6)

            self.dot = QLabel()
            self.dot.setFixedSize(6, 6)
            self._update_dot()
            layout.addWidget(self.dot)

            self.status_label = QLabel("")
            self.status_label.setStyleSheet(
                "color: #888888; font-family: 'SF Mono', monospace; font-size: 10px;"
            )
            layout.addWidget(self.status_label)
            layout.addStretch()

            self.setStyleSheet(
                "_ThinkingIndicator { background-color: #1A1A1A; }"
            )

            self._timer = QTimer(self)
            self._timer.timeout.connect(self._pulse)
            self._timer.setInterval(400)
            self.hide()

        def _update_dot(self):
            colors = ['#555555', '#777777', '#999999', '#777777']
            color = colors[self._pulse_state % len(colors)]
            self.dot.setStyleSheet(
                f"background-color: {color}; border-radius: 3px;"
            )

        def _pulse(self):
            self._pulse_state = (self._pulse_state + 1) % 4
            self._update_dot()

        def set_status(self, text: str):
            self.status_label.setText(text)
            if not self.isVisible():
                self.show()
                self._timer.start()

        def clear(self):
            self._timer.stop()
            self.hide()


# =============================================================================
# Guide Performance Window
# =============================================================================

if QT_AVAILABLE:

    class GuidePerformanceWindow(QMainWindow):
        """
        Pure renderer for a noodling's performance.

        Displays VRM character, receives text and affect from a facet
        assembly. Does NOT make LLM calls. Does NOT contain personality
        prompts. All cognition happens in the assembly, orchestrated by
        GuidePerformanceManager.

        Combines:
        - VRM character rendering (top)
        - Dialogue text display (middle)
        - User text input (bottom)

        Floats alongside the main window, always visible regardless
        of which center tab is active.

        Usage:
            window = GuidePerformanceWindow(parent_window=main_window)
            window.set_vrm("/path/to/ajo.vrm")
            window.show_play_header("Let's Consciousness!")
            window.show()
        """

        # Signal: user submitted a message for assembly execution
        messageSubmitted = pyqtSignal(str)

        # Signal: user sent a message (for channel bus forwarding)
        messageSent = pyqtSignal(str)

        def __init__(
            self,
            parent_window: QMainWindow,
            size: Tuple[int, int] = (350, 600),
            offset: Tuple[int, int] = (10, 60),
        ):
            """
            Initialize the guide performance window.

            Args:
                parent_window: The main window to follow
                size: (width, height) of this window
                offset: (x, y) offset from right edge of parent
            """
            super().__init__()
            self.parent_window = parent_window
            self._size = size
            self._offset = offset

            # Frameless, stays on top, no taskbar entry
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )

            self.setFixedSize(*size)
            self._build_ui()

            # Position once at the right edge of the parent, then stay put.
            # The user can drag the window anywhere (including a second
            # monitor) and it will remain there independently.
            if parent_window:
                geo = parent_window.geometry()
                x = geo.right() - size[0] - offset[0]
                y = geo.top() + offset[1]
                self.move(x, y)

            # VRM viewport reference (created lazily when VRM is loaded)
            self._vrm_viewport = None

            logger.info("GuidePerformanceWindow created")

        # =====================================================================
        # UI CONSTRUCTION
        # =====================================================================

        def _build_ui(self):
            """Build the window layout."""
            container = QWidget()
            container.setStyleSheet("background-color: #020204;")
            main_layout = QVBoxLayout(container)
            main_layout.setContentsMargins(0, 0, 0, 0)
            main_layout.setSpacing(0)

            # --- Header ---
            header_frame = QFrame()
            header_frame.setStyleSheet("""
                QFrame {
                    background-color: #252525;
                    border-bottom: 1px solid #333333;
                }
            """)
            header_layout = QHBoxLayout(header_frame)
            header_layout.setContentsMargins(10, 6, 6, 6)
            header_layout.setSpacing(0)

            self.header_label = _DraggableHeader("Performance")
            self.header_label.setStyleSheet("""
                color: #B0B0B0;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 12px;
                font-weight: bold;
                background: transparent;
            """)
            header_layout.addWidget(self.header_label, stretch=1)

            close_btn = QPushButton("x")
            close_btn.setFixedSize(22, 22)
            close_btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #888888;
                    font-size: 13px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #CCCCCC;
                    background-color: #3A3A3A;
                    border-radius: 3px;
                }
            """)
            close_btn.clicked.connect(self.close)
            header_layout.addWidget(close_btn)

            main_layout.addWidget(header_frame)

            # --- VRM Viewport Container ---
            self.vrm_container = QFrame()
            self.vrm_container.setFixedHeight(250)
            self.vrm_container.setStyleSheet("""
                QFrame {
                    background-color: #020204;
                    border: none;
                }
            """)
            self.vrm_container_layout = QVBoxLayout(self.vrm_container)
            self.vrm_container_layout.setContentsMargins(0, 0, 0, 0)

            # Placeholder label (replaced when VRM loads)
            self._vrm_placeholder = QLabel("No character loaded")
            self._vrm_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._vrm_placeholder.setStyleSheet(
                "color: #555555; font-size: 11px; background: transparent;"
            )
            self.vrm_container_layout.addWidget(self._vrm_placeholder)

            main_layout.addWidget(self.vrm_container)

            # --- Thinking Indicator ---
            self.thinking_indicator = _ThinkingIndicator()
            main_layout.addWidget(self.thinking_indicator)

            # --- Dialogue Display ---
            self.dialogue_view = QTextEdit()
            self.dialogue_view.setReadOnly(True)
            self.dialogue_view.setStyleSheet("""
                QTextEdit {
                    background-color: #1A1A1A;
                    border: none;
                    color: #B0B0B0;
                    font-family: 'SF Mono', 'Source Code Pro', monospace;
                    font-size: 12px;
                    padding: 8px;
                    selection-background-color: #3A3A3A;
                }
                QScrollBar:vertical {
                    background: #1A1A1A;
                    width: 6px;
                }
                QScrollBar::handle:vertical {
                    background: #3A3A3A;
                    border-radius: 3px;
                }
                QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                    height: 0px;
                }
            """)
            main_layout.addWidget(self.dialogue_view, stretch=1)

            # --- Input Area ---
            input_frame = QFrame()
            input_frame.setStyleSheet("""
                QFrame {
                    background-color: #2A2A2A;
                    border-top: 1px solid #3A3A3A;
                }
            """)
            input_layout = QHBoxLayout(input_frame)
            input_layout.setContentsMargins(6, 6, 6, 6)
            input_layout.setSpacing(6)

            self.input_field = QLineEdit()
            self.input_field.setPlaceholderText("Talk to Guide...")
            self.input_field.setStyleSheet("""
                QLineEdit {
                    background-color: #1E1E1E;
                    border: 1px solid #3A3A3A;
                    border-radius: 4px;
                    color: #D2D2D2;
                    padding: 6px 10px;
                    font-family: 'SF Mono', 'Source Code Pro', monospace;
                    font-size: 12px;
                }
                QLineEdit:focus {
                    border: 1px solid #4FC3F7;
                }
            """)
            self.input_field.returnPressed.connect(self._on_send)
            input_layout.addWidget(self.input_field)

            self.send_button = QPushButton("Send")
            self.send_button.setStyleSheet("""
                QPushButton {
                    background-color: #4FC3F7;
                    border: none;
                    border-radius: 4px;
                    color: #1A1A1A;
                    padding: 6px 12px;
                    font-weight: bold;
                    font-size: 11px;
                }
                QPushButton:hover { background-color: #67D3FF; }
                QPushButton:pressed { background-color: #3AA3D7; }
                QPushButton:disabled {
                    background-color: #3A3A3A;
                    color: #666;
                }
            """)
            self.send_button.clicked.connect(self._on_send)
            input_layout.addWidget(self.send_button)

            main_layout.addWidget(input_frame)

            self.setCentralWidget(container)

        # =====================================================================
        # VRM
        # =====================================================================

        def set_vrm(self, vrm_path: str):
            """
            Load a VRM character model into the viewport.

            Args:
                vrm_path: Path to .vrm file
            """
            try:
                from .components.vrm_viewport import VRMViewport, VRMViewportWidget

                # Create VRMViewport component (opaque background for this window)
                component = VRMViewport("guide_character")
                component.transparent = False
                component.background = "#020204"
                component.vrm_path = vrm_path
                component.show_grid = False
                component.show_skeleton = False
                component.interactive = False

                # Portrait camera (head/upper body)
                component.camera.distance = 2.0
                component.camera.elevation = 5
                component.camera.azimuth = 175
                component.camera.target = (0.0, 0.85, 0.0)

                # Remove placeholder
                if self._vrm_placeholder:
                    self._vrm_placeholder.setParent(None)
                    self._vrm_placeholder = None

                # Remove old viewport if any
                if self._vrm_viewport:
                    self._vrm_viewport.setParent(None)

                # Create and add the viewport widget
                self._vrm_viewport = VRMViewportWidget(component, self.vrm_container)
                self.vrm_container_layout.addWidget(self._vrm_viewport)

                logger.info(f"GuidePerformanceWindow: VRM loaded from {vrm_path}")

            except Exception as e:
                print(f"[GuidePerformance] VRM load failed: {e}", flush=True)
                logger.error(f"GuidePerformanceWindow: Failed to load VRM: {e}")
                if self._vrm_placeholder:
                    self._vrm_placeholder.setText(f"VRM load failed: {e}")

        def set_muscles(self, muscles: Dict[str, float]):
            """Apply muscle values to the VRM character."""
            if self._vrm_viewport:
                self._vrm_viewport.set_muscles(muscles)

        def set_blend_shapes(self, shapes: Dict[str, float]):
            """Apply blend shape weights to the VRM character."""
            if self._vrm_viewport:
                self._vrm_viewport.set_blend_shapes(shapes)

        # =====================================================================
        # DIALOGUE DISPLAY
        # =====================================================================

        def show_play_header(self, title: str):
            """
            Set the header text to the play title.

            Args:
                title: Title of the play being performed
            """
            self.header_label.setText(title)

        def clear_dialogue(self):
            """Clear the dialogue display."""
            self.dialogue_view.clear()

        def append_guide_text(self, text: str):
            """
            Append guide (assistant) text to the dialogue.

            Args:
                text: Guide's message text (from assembly OUTGOING)
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            fmt = QTextCharFormat()
            fmt.setForeground(QColor(180, 180, 180))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"\ua69c {text}\n\n")

            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        def append_user_text(self, text: str):
            """
            Append user text to the dialogue.

            Args:
                text: User's message text
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            fmt = QTextCharFormat()
            fmt.setBackground(QColor(60, 60, 60))
            fmt.setForeground(QColor(200, 200, 200))
            cursor.setCharFormat(fmt)

            lines = text.split('\n')
            for i, line in enumerate(lines):
                if i > 0:
                    cursor.insertText('\n')
                cursor.insertText(f"\u2b44 {line}" if line.strip() else "\u2b44")

            cursor.setCharFormat(QTextCharFormat())
            cursor.insertText('\n\n')

            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        def _scroll_to_bottom(self):
            """Scroll dialogue to bottom."""
            QTimer.singleShot(10, self._do_scroll)

        def _do_scroll(self):
            scrollbar = self.dialogue_view.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        # =====================================================================
        # SENDING MESSAGES
        # =====================================================================

        def _on_send(self):
            """Handle user pressing Enter or clicking Send."""
            message = self.input_field.text().strip()
            if not message:
                return

            self.input_field.clear()

            # Display user message
            self.append_user_text(message)

            # Signal to manager for assembly execution
            self.messageSubmitted.emit(message)

            # Signal for channel bus forwarding
            self.messageSent.emit(message)

        def set_busy(self, busy: bool):
            """
            Toggle busy state (thinking indicator and input).

            Called by GuidePerformanceManager during assembly execution.

            Args:
                busy: True when assembly is executing, False when done
            """
            self.input_field.setEnabled(not busy)
            self.send_button.setEnabled(not busy)
            if busy:
                self.thinking_indicator.set_status("Thinking...")
            else:
                self.thinking_indicator.clear()
                self.input_field.setFocus()

        def _show_error(self, text: str):
            """Show an error in the dialogue."""
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(200, 100, 100))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"Error: {text}\n\n")
            self.dialogue_view.setTextCursor(cursor)
            self._scroll_to_bottom()

        # =====================================================================
        # LIFECYCLE
        # =====================================================================

        def closeEvent(self, event):
            """Clean up on close."""
            super().closeEvent(event)


# =============================================================================
# Fallback when Qt not available
# =============================================================================

if not QT_AVAILABLE:

    class GuidePerformanceWindow:
        """Stub when PyQt6 is not available."""

        messageSent = None
        messageSubmitted = None

        def __init__(self, *args, **kwargs):
            logger.warning("GuidePerformanceWindow requires PyQt6")

        def show(self): pass
        def hide(self): pass
        def close(self): pass
        def set_vrm(self, vrm_path): pass
        def set_muscles(self, muscles): pass
        def set_blend_shapes(self, shapes): pass
        def set_busy(self, busy): pass
        def show_play_header(self, title): pass
        def clear_dialogue(self): pass
        def append_guide_text(self, text): pass
        def append_user_text(self, text): pass


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
