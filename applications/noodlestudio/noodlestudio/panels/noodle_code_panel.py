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
#   Noodle Code Panel - AI coding assistant chat interface
#
#   Provides a chat interface for the Noodle Code AI assistan...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.noodle_code_panel
# PURPOSE:  noodle code panel panel UI
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AsyncWorker, ThinkingIndicator, NoodleCodePanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import asyncio
import json
from typing import Optional, List
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QScrollArea,
    QLabel, QFrame, QSizePolicy, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, pyqtSlot, QSettings, QEvent, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QColor, QTextCharFormat, QKeyEvent

import sys
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock


class AsyncWorker(QThread):
    """Worker thread for async operations."""
    chunk_received = pyqtSignal(dict)  # Emits chunk data
    finished_signal = pyqtSignal()

    def __init__(self, engine, message: str):
        super().__init__()
        self.engine = engine
        self.message = message
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
            async for chunk in self.engine.send_message(self.message):
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


class ThinkingIndicator(QFrame):
    """
    Monochromatic thinking indicator with subtle animation.

    Shows current activity status with a pulsing dot.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self._visible = False
        self._pulse_state = 0
        self._status_text = ""

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 6, 12, 6)
        layout.setSpacing(8)

        # Pulsing dot indicator
        self.dot = QLabel()
        self.dot.setFixedSize(8, 8)
        self._update_dot_style()
        layout.addWidget(self.dot)

        # Status text
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("""
            color: #888888;
            font-family: 'SF Mono', 'Source Code Pro', monospace;
            font-size: 11px;
        """)
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.setStyleSheet("""
            ThinkingIndicator {
                background-color: #252525;
                border-bottom: 1px solid #333333;
            }
        """)

        # Pulse timer
        self._pulse_timer = QTimer(self)
        self._pulse_timer.timeout.connect(self._pulse)
        self._pulse_timer.setInterval(400)

        self.hide()

    def _update_dot_style(self):
        """Update dot appearance based on pulse state."""
        colors = ['#555555', '#777777', '#999999', '#777777']
        color = colors[self._pulse_state % len(colors)]
        self.dot.setStyleSheet(f"""
            background-color: {color};
            border-radius: 4px;
        """)

    def _pulse(self):
        """Animate the pulse."""
        self._pulse_state = (self._pulse_state + 1) % 4
        self._update_dot_style()

    def set_status(self, text: str):
        """Set the current status text and show indicator."""
        self._status_text = text
        self.status_label.setText(text)
        if not self._visible:
            self._visible = True
            self.show()
            self._pulse_timer.start()

    def clear(self):
        """Hide the indicator."""
        self._visible = False
        self._pulse_timer.stop()
        self.hide()



class NoodleCodePanel(MaximizableDock):
    """
    Noodle Code AI coding assistant panel.

    Provides a chat interface for interacting with an AI that can
    read, write, and modify project files.
    """

    # Signal emitted when a file is modified (for refresh)
    fileModified = pyqtSignal(str)  # path

    # Chat history file location
    HISTORY_FILE = Path.home() / ".noodlestudio" / "noodlecode_history.json"
    MAX_HISTORY_MESSAGES = 200  # Limit stored messages

    def __init__(self, parent: QWidget = None):
        super().__init__("NOODLE CODE", parent)

        self.engine = None
        self.worker = None

        # Font size with persistence (range: 8-36)
        self.settings = QSettings("Noodlings", "NoodleCodePanel")
        self.font_size = self.settings.value("font_size", 12, type=int)

        # Input history for up/down arrow navigation
        self.input_history = []
        self.history_index = -1  # -1 means not navigating history

        # Chat history data (for persistence)
        self._chat_history: List[dict] = []
        self._current_response = ""  # Buffer for streaming response
        self._response_started = False  # Track if we've started the current response

        # Allow moving and floating
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._setup_ui()

    def _setup_ui(self):
        """Build UI components."""
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Thinking indicator (at top)
        self.thinking_indicator = ThinkingIndicator()
        main_layout.addWidget(self.thinking_indicator)

        # Chat text view - single selectable text area
        self.chat_view = QTextEdit()
        self.chat_view.setReadOnly(True)
        self.chat_view.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1A1A1A;
                border: none;
                color: #B0B0B0;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: {self.font_size}px;
                padding: 12px;
                selection-background-color: #3A3A3A;
            }}
            QScrollBar:vertical {{
                background: #1A1A1A;
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: #3A3A3A;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        main_layout.addWidget(self.chat_view, stretch=1)

        # Welcome message
        self._add_welcome_message()

        # Input area
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background-color: #2A2A2A;
                border-top: 1px solid #3A3A3A;
            }
        """)
        input_layout = QHBoxLayout(input_frame)
        input_layout.setContentsMargins(8, 8, 8, 8)
        input_layout.setSpacing(8)

        # Button style
        btn_style = """
            QPushButton {
                background-color: #3A3A3A;
                border: none;
                border-radius: 3px;
                color: #B0B0B0;
                padding: 4px 8px;
                font-weight: bold;
                font-size: 11px;
                min-width: 24px;
            }
            QPushButton:hover {
                background-color: #4A4A4A;
                color: #D2D2D2;
            }
            QPushButton:pressed {
                background-color: #2A2A2A;
            }
            QPushButton:checked {
                background-color: #2A4A4A;
                color: #88CCCC;
            }
        """

        # Profile selector
        self.profile_combo = QComboBox()
        self.profile_combo.setStyleSheet("""
            QComboBox {
                background-color: #3A3A3A;
                border: none;
                border-radius: 3px;
                color: #B0B0B0;
                padding: 4px 8px;
                font-size: 11px;
                min-width: 70px;
                max-width: 100px;
            }
            QComboBox:hover {
                background-color: #4A4A4A;
                color: #D2D2D2;
            }
            QComboBox::drop-down {
                border: none;
                width: 16px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888888;
                margin-right: 4px;
            }
            QComboBox QAbstractItemView {
                background-color: #2A2A2A;
                border: 1px solid #3A3A3A;
                color: #D2D2D2;
                selection-background-color: #4A4A4A;
            }
        """)
        self.profile_combo.setToolTip("Select coding assistant personality")
        self._populate_profiles()
        self.profile_combo.currentTextChanged.connect(self._on_profile_changed)
        input_layout.addWidget(self.profile_combo)

        # Demo mode toggle (ghost cursor visualization)
        self.demo_mode_btn = QPushButton("D")
        self.demo_mode_btn.setStyleSheet(btn_style)
        self.demo_mode_btn.setCheckable(True)
        self.demo_mode_btn.setToolTip("Demo mode: show ghost cursor during Computer Use")
        self.demo_mode_btn.clicked.connect(self._toggle_demo_mode)
        input_layout.addWidget(self.demo_mode_btn)

        # Copy chat history button
        self.copy_chat_btn = QPushButton("C")
        self.copy_chat_btn.setStyleSheet(btn_style)
        self.copy_chat_btn.setToolTip("Copy chat history to clipboard")
        self.copy_chat_btn.clicked.connect(self._copy_chat_to_clipboard)
        input_layout.addWidget(self.copy_chat_btn)

        # Font size controls (compact)
        decrease_btn = QPushButton("A-")
        decrease_btn.setStyleSheet(btn_style)
        decrease_btn.setToolTip("Decrease font size")
        decrease_btn.clicked.connect(self.decrease_font_size)
        input_layout.addWidget(decrease_btn)

        self.font_size_label = QLabel(f"{self.font_size}")
        self.font_size_label.setStyleSheet("""
            color: #808080;
            font-size: 10px;
            min-width: 20px;
        """)
        self.font_size_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        input_layout.addWidget(self.font_size_label)

        increase_btn = QPushButton("A+")
        increase_btn.setStyleSheet(btn_style)
        increase_btn.setToolTip("Increase font size")
        increase_btn.clicked.connect(self.increase_font_size)
        input_layout.addWidget(increase_btn)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Ask Noodle Code...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: #1E1E1E;
                border: 1px solid #3A3A3A;
                border-radius: 4px;
                color: #D2D2D2;
                padding: 8px 12px;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #4FC3F7;
            }
        """)
        self.input_field.returnPressed.connect(self._on_send)
        self.input_field.installEventFilter(self)  # For up/down history navigation
        input_layout.addWidget(self.input_field)

        self.send_button = QPushButton("Send")
        self._is_stop_mode = False
        self._update_send_button_style()
        self.send_button.clicked.connect(self._on_send_or_stop)
        input_layout.addWidget(self.send_button)

        main_layout.addWidget(input_frame)

        container.setStyleSheet("background-color: #1A1A1A;")
        self.setWidget(container)

        # Load persisted chat history
        self._load_history()

    def _populate_profiles(self):
        """Populate the profile dropdown with available profiles."""
        try:
            from noodlestudio.core.noodle_code_profiles import get_profile_manager
            manager = get_profile_manager()

            self.profile_combo.blockSignals(True)
            self.profile_combo.clear()

            for name in manager.get_profile_names():
                profile = manager.get_profile(name)
                display_name = name.replace('_', ' ').title()
                self.profile_combo.addItem(display_name, name)

            # Select current profile
            current = manager.current_profile_name
            index = self.profile_combo.findData(current)
            if index >= 0:
                self.profile_combo.setCurrentIndex(index)

            self.profile_combo.blockSignals(False)
        except Exception as e:
            print(f"[NoodleCode] Error loading profiles: {e}")
            self.profile_combo.addItem("Default", "default")

    def _on_profile_changed(self, display_name: str):
        """Handle profile selection change."""
        try:
            from noodlestudio.core.noodle_code_profiles import get_profile_manager
            manager = get_profile_manager()

            # Get the actual profile name from combo data
            index = self.profile_combo.currentIndex()
            profile_name = self.profile_combo.itemData(index)

            if profile_name:
                manager.set_current_profile(profile_name)

                # Notify engine if we have one
                if hasattr(self, 'engine') and self.engine:
                    self.engine.set_profile(profile_name)
        except Exception as e:
            print(f"[NoodleCode] Error changing profile: {e}")

    def _toggle_demo_mode(self):
        """Toggle demo mode for Computer Use visualization."""
        enabled = self.demo_mode_btn.isChecked()

        # If a performance manager is active, route through it
        # so it can coordinate the floating window lifecycle
        manager = getattr(self, '_guide_performance_manager', None)
        if manager and manager.is_active and not enabled:
            # Turning off demo mode while performance is active stops the performance
            manager.stop_performance()
            return

        try:
            from noodlestudio.core.computer_use_controller import get_computer_use_controller
            controller = get_computer_use_controller()
            controller.demo_mode = enabled
        except Exception as e:
            print(f"[NoodleCode] Failed to toggle demo mode: {e}")

    def _update_send_button_style(self):
        """Update send/stop button appearance based on mode."""
        if self._is_stop_mode:
            self.send_button.setText("Stop")
            self.send_button.setStyleSheet("""
                QPushButton {
                    background-color: #E57373;
                    border: none;
                    border-radius: 4px;
                    color: #1A1A1A;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #EF9A9A;
                }
                QPushButton:pressed {
                    background-color: #C55050;
                }
            """)
        else:
            self.send_button.setText("Send")
            self.send_button.setStyleSheet("""
                QPushButton {
                    background-color: #4FC3F7;
                    border: none;
                    border-radius: 4px;
                    color: #1A1A1A;
                    padding: 8px 16px;
                    font-weight: bold;
                    font-size: 12px;
                }
                QPushButton:hover {
                    background-color: #67D3FF;
                }
                QPushButton:pressed {
                    background-color: #3AA3D7;
                }
                QPushButton:disabled {
                    background-color: #3A3A3A;
                    color: #666;
                }
            """)

    def _set_stop_mode(self, stop_mode: bool):
        """Switch between Send and Stop button modes."""
        self._is_stop_mode = stop_mode
        self._update_send_button_style()

    def _on_send_or_stop(self):
        """Handle send/stop button click."""
        if self._is_stop_mode:
            self.stop_generation()
        else:
            self._on_send()

    def _add_welcome_message(self):
        """Add welcome message to chat."""
        welcome = ("Noodle Code ready.\n"
                   "- Read and edit project files\n"
                   "- Search the codebase\n"
                   "- Run shell commands\n"
                   "- Explain code")
        self._append_message(welcome, is_user=False)
        self._append_text("\n\n")

    def set_engine(self, engine):
        """Set the NoodleCodeEngine instance."""
        self.engine = engine

    def set_guide_performance_manager(self, manager):
        """
        Set the GuidePerformanceManager for coordinated performance control.

        Args:
            manager: GuidePerformanceManager instance
        """
        self._guide_performance_manager = manager

    def set_project_path(self, path: Path):
        """Set the project path for the engine."""
        if self.engine:
            self.engine.set_project_path(path)

    def eventFilter(self, obj, event):
        """Handle up/down arrow keys for input history navigation."""
        if not hasattr(self, 'input_field'):
            return super().eventFilter(obj, event)

        if obj == self.input_field and event.type() == QEvent.Type.KeyPress:
            key = event.key()

            if key == Qt.Key.Key_Up:
                # Navigate to older message
                if self.input_history:
                    if self.history_index == -1:
                        # Starting navigation - save current input
                        self._temp_input = self.input_field.text()
                        self.history_index = len(self.input_history) - 1
                    elif self.history_index > 0:
                        self.history_index -= 1

                    if 0 <= self.history_index < len(self.input_history):
                        self.input_field.setText(self.input_history[self.history_index])
                return True

            elif key == Qt.Key.Key_Down:
                # Navigate to newer message
                if self.history_index != -1:
                    self.history_index += 1
                    if self.history_index >= len(self.input_history):
                        # Back to current input
                        self.history_index = -1
                        self.input_field.setText(getattr(self, '_temp_input', ''))
                    else:
                        self.input_field.setText(self.input_history[self.history_index])
                return True

            elif key == Qt.Key.Key_Escape:
                # Stop generation if running
                if self._is_stop_mode:
                    self.stop_generation()
                    return True

        return super().eventFilter(obj, event)

    def execute_command(self, message: str):
        """
        Execute a command programmatically (e.g., from CLI --execute parameter).

        This is the public API for injecting commands into NoodleCode from
        external sources like CLI parameters or automated tests.

        Args:
            message: The command/message to execute
        """
        if not message or not message.strip():
            return

        # Set the input field and trigger send
        self.input_field.setText(message.strip())
        self._on_send()

    def _on_send(self):
        """Handle send button click or enter key."""
        message = self.input_field.text().strip()
        if not message:
            return

        if not self.engine:
            self._add_error_message("Engine not initialized. Please wait for project to load.")
            return

        # Save to history (avoid duplicates of last entry)
        if not self.input_history or self.input_history[-1] != message:
            self.input_history.append(message)
        self.history_index = -1  # Reset navigation

        # Clear input and switch to Stop mode
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self._set_stop_mode(True)

        # Show thinking indicator
        self.thinking_indicator.set_status("Considering your request...")

        # Add user message (with > prefix and background)
        self._append_message(message, is_user=True)
        self._add_message_to_history("user", message)

        # Reset for new response
        self._current_response = ""
        self._response_started = False

        # Start async worker
        self.worker = AsyncWorker(self.engine, message)
        self.worker.chunk_received.connect(self._on_chunk)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.start()

    @pyqtSlot(dict)
    def _on_chunk(self, chunk: dict):
        """Handle a streamed chunk."""
        chunk_type = chunk['type']
        content = chunk.get('content', '')

        if chunk_type == 'text':
            # Streaming text response
            self.thinking_indicator.set_status("Writing response...")

            # First chunk gets the dot prefix
            if not self._response_started:
                self._append_message("", is_user=False)  # Adds "● "
                self._response_started = True

            self._append_text(content)
            self._current_response += content

        elif chunk_type == 'tool_use_start':
            # Tool starting - just update indicator, don't clutter chat
            tool_name = chunk.get('tool_name', 'unknown')
            status_text = self._get_tool_status_text(tool_name)
            self.thinking_indicator.set_status(status_text)

        elif chunk_type == 'tool_result':
            # Emit file modified signal if it was a write/edit
            tool_name = chunk.get('tool_name', '')
            if tool_name in ['write_file', 'edit_file']:
                self.fileModified.emit('')

        elif chunk_type == 'error':
            self._append_text(f"\nError: {content}")

        elif chunk_type == 'done':
            # Response complete - add spacing
            self._append_text("\n\n")
            self.thinking_indicator.clear()

    def _get_tool_status_text(self, tool_name: str) -> str:
        """Get human-readable status text for a tool."""
        status_map = {
            'read_file': 'Reading file...',
            'write_file': 'Writing file...',
            'edit_file': 'Editing file...',
            'glob': 'Searching files...',
            'grep': 'Searching content...',
            'bash': 'Running command...',
            'list_directory': 'Listing directory...',
            'computer_use': 'Interacting with UI...',
            'github': 'Querying GitHub...',
            'hot_reload': 'Hot reloading...',
            'soft_restart': 'Restarting...',
        }
        return status_map.get(tool_name, f'Executing {tool_name}...')

    def _on_finished(self):
        """Handle worker finished."""
        # Save assistant response to history
        if hasattr(self, '_current_response') and self._current_response.strip():
            self._add_message_to_history("assistant", self._current_response.strip())
            self._current_response = ""

        self.thinking_indicator.clear()
        self.input_field.setEnabled(True)
        self._set_stop_mode(False)  # Switch back to Send button
        self.input_field.setFocus()
        self.worker = None

    def _add_error_message(self, error: str):
        """Add an error message to chat."""
        self._append_message(f"Error: {error}", is_user=False)
        self._append_text("\n\n")

    def _scroll_to_bottom(self):
        """Scroll chat to bottom."""
        QTimer.singleShot(10, self._do_scroll_to_bottom)

    def _do_scroll_to_bottom(self):
        """Actually perform the scroll."""
        scrollbar = self.chat_view.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear_chat(self):
        """Clear chat history."""
        # Clear the text view
        self.chat_view.clear()

        # Clear persistent history
        self._chat_history.clear()
        self._save_history()

        # Add welcome message back
        self._add_welcome_message()

        # Clear engine history
        if self.engine:
            self.engine.clear_history()

    def stop_generation(self):
        """Stop current generation."""
        if self.worker:
            self.worker.stop()
            self.thinking_indicator.set_status("Stopping...")
            self._append_text("\n[interrupted]\n\n")
            # Clean up will happen in _on_finished when worker emits finished_signal

        # Hide ghost cursor overlay if visible
        try:
            from noodlestudio.core.ghost_cursor import get_ghost_overlay
            overlay = get_ghost_overlay()
            if overlay:
                overlay.hide_cursor()
                overlay.hide()
        except Exception:
            pass  # Ghost cursor may not be initialized

    def increase_font_size(self):
        """Increase font size (max 36)."""
        if self.font_size < 36:
            self.font_size = min(36, self.font_size + 2)
            self._apply_font_size()

    def decrease_font_size(self):
        """Decrease font size (min 8)."""
        if self.font_size > 8:
            self.font_size = max(8, self.font_size - 2)
            self._apply_font_size()

    def _apply_font_size(self):
        """Apply current font size to chat view and save setting."""
        self.font_size_label.setText(f"{self.font_size}")
        self.chat_view.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1A1A1A;
                border: none;
                color: #B0B0B0;
                font-family: 'SF Mono', 'Source Code Pro', monospace;
                font-size: {self.font_size}px;
                padding: 12px;
                selection-background-color: #3A3A3A;
            }}
            QScrollBar:vertical {{
                background: #1A1A1A;
                width: 8px;
            }}
            QScrollBar::handle:vertical {{
                background: #3A3A3A;
                border-radius: 4px;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
        """)
        self.settings.setValue("font_size", self.font_size)

    def _append_message(self, text: str, is_user: bool = False):
        """Append a message to chat view.

        User: ➟ prefix with selection-style background
        Assistant: ⚮ (gear with handles) prefix, regular text
        """
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        if is_user:
            # User message: > prefix with highlighted background
            fmt = QTextCharFormat()
            fmt.setBackground(QColor(60, 60, 60))  # Selection-style background
            fmt.setForeground(QColor(200, 200, 200))
            cursor.setCharFormat(fmt)

            # Add > prefix to each line
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if i > 0:
                    cursor.insertText('\n')
                cursor.insertText(f"⭄ {line}" if line.strip() else "⭄")

            # Reset format and add spacing
            cursor.setCharFormat(QTextCharFormat())
            cursor.insertText('\n\n')
        else:
            # Assistant message: dot prefix
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(180, 180, 180))
            cursor.setCharFormat(fmt)
            cursor.insertText(f"꩜ {text}")

        self.chat_view.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _append_text(self, text: str, is_user: bool = False):
        """Append streaming text (no prefix, continues current message)."""
        cursor = self.chat_view.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)

        fmt = QTextCharFormat()
        fmt.setForeground(QColor(180, 180, 180))
        cursor.setCharFormat(fmt)
        cursor.insertText(text)

        self.chat_view.setTextCursor(cursor)
        self._scroll_to_bottom()

    def _copy_chat_to_clipboard(self):
        """Copy full chat history to clipboard."""
        text = self.chat_view.toPlainText()
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        # Brief visual feedback
        original_text = self.copy_chat_btn.text()
        self.copy_chat_btn.setText("OK")
        QTimer.singleShot(800, lambda: self.copy_chat_btn.setText(original_text))

    def _load_history(self):
        """Load chat history from disk."""
        try:
            if not self.HISTORY_FILE.exists():
                return

            with open(self.HISTORY_FILE, 'r') as f:
                data = json.load(f)

            self._chat_history = data.get('messages', [])
            self.input_history = data.get('input_history', [])

            # Clear and rebuild from history
            self.chat_view.clear()

            # Add history messages
            for msg in self._chat_history:
                role = msg.get('role', 'assistant')
                content = msg.get('content', '')
                if role and content:
                    is_user = (role == 'user')
                    self._append_message(content, is_user=is_user)
                    if not is_user:
                        self._append_text("\n\n")

            # If no history, add welcome message
            if not self._chat_history:
                self._add_welcome_message()

            print(f"[NoodleCode] Loaded {len(self._chat_history)} messages from history")

        except Exception as e:
            print(f"[NoodleCode] Error loading history: {e}")
            self._add_welcome_message()

    def _save_history(self):
        """Save chat history to disk."""
        try:
            # Ensure directory exists
            self.HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)

            # Trim to max messages
            if len(self._chat_history) > self.MAX_HISTORY_MESSAGES:
                self._chat_history = self._chat_history[-self.MAX_HISTORY_MESSAGES:]

            # Trim input history too
            if len(self.input_history) > 100:
                self.input_history = self.input_history[-100:]

            data = {
                'messages': self._chat_history,
                'input_history': self.input_history,
                'saved_at': datetime.now().isoformat()
            }

            with open(self.HISTORY_FILE, 'w') as f:
                json.dump(data, f, indent=2)

            print(f"[NoodleCode] Saved {len(self._chat_history)} messages to history")

        except Exception as e:
            print(f"[NoodleCode] Error saving history: {e}")

    def _add_message_to_history(self, role: str, content: str):
        """Add a message to the persistent history."""
        self._chat_history.append({
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        })
        # Save after each message for reliability
        self._save_history()

    def closeEvent(self, event):
        """Save history when panel is closed."""
        self._save_history()
        super().closeEvent(event)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
