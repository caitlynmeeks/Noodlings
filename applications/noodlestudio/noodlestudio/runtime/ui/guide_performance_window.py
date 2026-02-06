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
#   Floating combined panel for guided play performances.
#   Combines VRM character rendering, dialogue display, and user
#   text input in a single always-on-top window that floats
#   alongside the main NoodleSTUDIO window.
#
#   Designed to free NoodleCode for developer use while Guide
#   (Ajo Majo) gets his own dedicated interaction surface.
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

import asyncio
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from PyQt6.QtWidgets import (
        QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QTextEdit, QLineEdit, QPushButton, QLabel, QFrame
    )
    from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, pyqtSlot
    from PyQt6.QtGui import (
        QFont, QTextCursor, QColor, QTextCharFormat, QMouseEvent
    )
    QT_AVAILABLE = True
except ImportError:
    QT_AVAILABLE = False


# =============================================================================
# Async Worker (same pattern as NoodleCodePanel)
# =============================================================================

if QT_AVAILABLE:

    class _GuideAsyncWorker(QThread):
        """Worker thread for streaming LLM responses in the guide window."""
        chunk_received = pyqtSignal(dict)
        finished_signal = pyqtSignal()

        def __init__(self, engine, message: str, system_addition: str = ""):
            super().__init__()
            self.engine = engine
            self.message = message
            self.system_addition = system_addition
            self._running = True

        def run(self):
            """Run the async message in a thread."""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(self._process_message())
            finally:
                loop.close()

        async def _process_message(self):
            """Process message and emit chunks."""
            try:
                # Prepend guide direction as system context
                effective_message = self.message
                if self.system_addition:
                    effective_message = (
                        f"[SYSTEM CONTEXT - Guide Direction]\n"
                        f"{self.system_addition}\n"
                        f"[END SYSTEM CONTEXT]\n\n"
                        f"{self.message}"
                    )

                async for chunk in self.engine.send_message(effective_message):
                    if not self._running:
                        break
                    self.chunk_received.emit({
                        'type': chunk.type,
                        'content': chunk.content,
                        'tool_name': chunk.tool_name,
                        'tool_id': chunk.tool_id,
                        'tool_input': chunk.tool_input
                    })
            except Exception as e:
                self.chunk_received.emit({
                    'type': 'error',
                    'content': str(e)
                })
            finally:
                self.finished_signal.emit()

        def stop(self):
            """Stop the worker."""
            self._running = False


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
        Floating combined panel for guided play performances.

        Combines:
        - VRM character rendering (top)
        - Dialogue text display (middle)
        - User text input (bottom)

        Floats alongside the main window, always visible regardless
        of which center tab is active. Guide gets his own interaction
        surface independent of NoodleCode.

        Usage:
            window = GuidePerformanceWindow(parent_window=main_window)
            window.set_engine(noodle_code_engine)
            window.set_vrm("/path/to/ajo.vrm")
            window.show_play_header("Let's Consciousness!")
            window.show()
        """

        # Signal emitted when user sends a message
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

            self.engine = None
            self.worker = None
            self._guide_cue_handler = None

            # Drag state -- pauses follow-parent during user drag
            self._user_dragging = False

            # Streaming state
            self._current_response = ""
            self._response_started = False

            # Frameless, stays on top, no taskbar entry
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint |
                Qt.WindowType.WindowStaysOnTopHint |
                Qt.WindowType.Tool
            )

            self.setFixedSize(*size)
            self._build_ui()

            # Timer to follow parent window position
            self._follow_timer = QTimer()
            self._follow_timer.timeout.connect(self._follow_parent)
            self._follow_timer.start(50)

            # VRM viewport reference (created lazily when VRM is loaded)
            self._vrm_viewport = None

            logger.info("GuidePerformanceWindow created")

        # =====================================================================
        # UI CONSTRUCTION
        # =====================================================================

        def _build_ui(self):
            """Build the window layout."""
            container = QWidget()
            container.setStyleSheet("background-color: #1A1A1A;")
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
            self.header_label.dragStarted.connect(self._on_drag_start)
            self.header_label.dragFinished.connect(self._on_drag_finish)
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
                    background-color: #1A1A1A;
                    border-bottom: 1px solid #2A2A2A;
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
            self._is_stop_mode = False
            self._update_send_button_style()
            self.send_button.clicked.connect(self._on_send_or_stop)
            input_layout.addWidget(self.send_button)

            main_layout.addWidget(input_frame)

            self.setCentralWidget(container)

        def _update_send_button_style(self):
            """Update send/stop button appearance."""
            if self._is_stop_mode:
                self.send_button.setText("Stop")
                self.send_button.setStyleSheet("""
                    QPushButton {
                        background-color: #E57373;
                        border: none;
                        border-radius: 4px;
                        color: #1A1A1A;
                        padding: 6px 12px;
                        font-weight: bold;
                        font-size: 11px;
                    }
                    QPushButton:hover { background-color: #EF9A9A; }
                    QPushButton:pressed { background-color: #C55050; }
                """)
            else:
                self.send_button.setText("Send")
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

        def _set_stop_mode(self, stop_mode: bool):
            """Switch between Send and Stop button modes."""
            self._is_stop_mode = stop_mode
            self._update_send_button_style()

        # =====================================================================
        # ENGINE AND HANDLER WIRING
        # =====================================================================

        def set_engine(self, engine):
            """
            Set the NoodleCode engine for LLM communication.

            Args:
                engine: NoodleCodeEngine instance (shared with NoodleCode panel)
            """
            self.engine = engine

        def set_guide_cue_handler(self, handler):
            """
            Set the GuideCueHandler for direction context.

            Args:
                handler: GuideCueHandler instance
            """
            self._guide_cue_handler = handler

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
                component.background = "#1A1A1A"
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
                text: Guide's message text
            """
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            fmt = QTextCharFormat()
            fmt.setForeground(QColor(180, 180, 180))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"꩜ {text}\n\n")

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

        def _append_streaming_text(self, text: str):
            """Append streaming text (continues current message, no prefix)."""
            cursor = self.dialogue_view.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)

            fmt = QTextCharFormat()
            fmt.setForeground(QColor(180, 180, 180))
            cursor.setCharFormat(fmt)
            cursor.insertText(text)

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

            if not self.engine:
                self._show_error("Engine not initialized.")
                return

            self.input_field.clear()
            self.input_field.setEnabled(False)
            self._set_stop_mode(True)

            self.thinking_indicator.set_status("Considering...")

            # Display user message and track for feedback
            self.append_user_text(message)
            self._last_user_message = message

            # Reset streaming state
            self._current_response = ""
            self._response_started = False

            # Get guide direction for system prompt
            system_addition = ""
            if self._guide_cue_handler:
                system_addition = self._guide_cue_handler.build_system_prompt_addition()

            # Start async worker
            self.worker = _GuideAsyncWorker(self.engine, message, system_addition)
            self.worker.chunk_received.connect(self._on_chunk)
            self.worker.finished_signal.connect(self._on_finished)
            self.worker.start()

            self.messageSent.emit(message)

        def _on_send_or_stop(self):
            """Handle send/stop button click."""
            if self._is_stop_mode:
                self._stop_generation()
            else:
                self._on_send()

        def _stop_generation(self):
            """Stop current generation."""
            if self.worker:
                self.worker.stop()
                self.thinking_indicator.set_status("Stopping...")
                self._append_streaming_text("\n[interrupted]\n\n")

        @pyqtSlot(dict)
        def _on_chunk(self, chunk: dict):
            """Handle a streamed chunk from the LLM."""
            chunk_type = chunk['type']
            content = chunk.get('content', '')

            if chunk_type == 'text':
                self.thinking_indicator.set_status("Speaking...")

                if not self._response_started:
                    # Start new guide message with prefix
                    cursor = self.dialogue_view.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    fmt = QTextCharFormat()
                    fmt.setForeground(QColor(180, 180, 180))
                    cursor.setCharFormat(fmt)
                    cursor.insertText("꩜ ")
                    self.dialogue_view.setTextCursor(cursor)
                    self._response_started = True

                self._append_streaming_text(content)
                self._current_response += content

            elif chunk_type == 'tool_use_start':
                tool_name = chunk.get('tool_name', 'unknown')
                self.thinking_indicator.set_status(f"Using {tool_name}...")

            elif chunk_type == 'error':
                self._append_streaming_text(f"\nError: {content}")

            elif chunk_type == 'done':
                self._append_streaming_text("\n\n")
                self.thinking_indicator.clear()

        def _on_finished(self):
            """Handle worker finished."""
            # Report response to GuideCueHandler for Brenda feedback
            if (self._guide_cue_handler
                    and self._current_response.strip()):
                self._guide_cue_handler.report_response(
                    self._current_response.strip(),
                    getattr(self, '_last_user_message', "")
                )

            self._current_response = ""
            self.thinking_indicator.clear()
            self.input_field.setEnabled(True)
            self._set_stop_mode(False)
            self.input_field.setFocus()
            self.worker = None

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
        # POSITION TRACKING
        # =====================================================================

        def _on_drag_start(self):
            """Pause follow-parent while user drags the window."""
            self._user_dragging = True

        def _on_drag_finish(self):
            """Recalculate offset from parent after user drag, then resume tracking."""
            if self.parent_window:
                geo = self.parent_window.geometry()
                pos = self.pos()
                # Recalculate offset so window stays at user-chosen position
                self._offset = (
                    geo.right() - pos.x() - self._size[0],
                    pos.y() - geo.top()
                )
            self._user_dragging = False

        def _follow_parent(self):
            """Update position to follow parent window (anchored inside right edge)."""
            if self._user_dragging:
                return

            if self.parent_window and self.parent_window.isVisible():
                geo = self.parent_window.geometry()

                # Anchor inside the right edge of the parent window
                # Inset from the right so it's visible even when maximized
                x = geo.right() - self._size[0] - self._offset[0]
                y = geo.top() + self._offset[1]
                self.move(x, y)

                if not self.isVisible():
                    self.show()
            else:
                if self.isVisible():
                    self.hide()

        # =====================================================================
        # LIFECYCLE
        # =====================================================================

        def closeEvent(self, event):
            """Clean up on close."""
            self._follow_timer.stop()
            if self.worker:
                self.worker.stop()
            super().closeEvent(event)


# =============================================================================
# Fallback when Qt not available
# =============================================================================

if not QT_AVAILABLE:

    class GuidePerformanceWindow:
        """Stub when PyQt6 is not available."""

        messageSent = None

        def __init__(self, *args, **kwargs):
            logger.warning("GuidePerformanceWindow requires PyQt6")

        def show(self): pass
        def hide(self): pass
        def close(self): pass
        def set_engine(self, engine): pass
        def set_guide_cue_handler(self, handler): pass
        def set_vrm(self, vrm_path): pass
        def set_muscles(self, muscles): pass
        def set_blend_shapes(self, shapes): pass
        def show_play_header(self, title): pass
        def clear_dialogue(self): pass
        def append_guide_text(self, text): pass
        def append_user_text(self, text): pass


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
