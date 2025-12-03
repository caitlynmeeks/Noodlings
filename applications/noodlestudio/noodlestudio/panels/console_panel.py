"""
Console Panel - Live log viewer from noodleMUSH

Connects to noodleMUSH WebSocket and displays real-time logs.
Like Unity's Console panel.

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox, QCheckBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor, QFontMetrics
import json
import asyncio
import websockets
from threading import Thread
import sys
import re
sys.path.append('..')



class WebSocketWorker(QThread):
    """Background thread for WebSocket log streaming."""

    logReceived = pyqtSignal(str, str, str)  # level, module, message
    connected = pyqtSignal(bool)  # Connection status

    def __init__(self, ws_url: str):
        super().__init__()
        self.ws_url = ws_url
        self.running = True

    def run(self):
        """Connect to WebSocket and stream logs."""
        asyncio.run(self.connect_and_stream())

    async def connect_and_stream(self):
        """Async WebSocket connection."""
        try:
            print(f"Connecting to {self.ws_url}...")
            async with websockets.connect(self.ws_url) as ws:
                print("WebSocket connected! Subscribing to logs...")

                # Studio operations don't require authentication
                # Subscribe to logs immediately
                await ws.send(json.dumps({'type': 'subscribe_logs'}))

                self.logReceived.emit('INFO', 'Console', 'Connected to noodleMUSH')
                self.connected.emit(True)

                # Stream logs
                while self.running:
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                        data = json.loads(message)

                        if data.get('type') == 'log':
                            level = data.get('level', 'INFO')
                            module = data.get('name', 'unknown')
                            msg = data.get('message', '')
                            self.logReceived.emit(level, module, msg)

                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        if self.running:
                            print(f"Error in log stream: {e}")
                            self.logReceived.emit('ERROR', 'Console', f"Stream error: {e}")
                        break

        except Exception as e:
            print(f"WebSocket connection error: {e}")
            import traceback
            traceback.print_exc()
            self.logReceived.emit('ERROR', 'Console', f"Connection failed: {e}")

    def stop(self):
        """Stop the worker thread."""
        self.running = False


class ConsolePanel(QWidget):
    """
    Console panel showing live logs from noodleMUSH.

    Connects to WebSocket and streams log messages.
    Unity-style message collapsing for repeated logs.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.ws_url = "ws://localhost:8765"
        self.connected = False
        self.log_buffer = []  # Formatted HTML logs (MUSH)
        self.log_buffer_raw = []  # Raw log data for filtering (MUSH)
        self.studio_log_buffer = []  # Formatted logs (STUDIO)
        self.studio_log_buffer_raw = []  # Raw logs (STUDIO)
        self.console_mode = 'mush'  # 'mush' or 'studio'
        self.last_message = None  # Track last message for collapsing
        self.repeat_count = 0
        self.font_size = 11  # Default font size for console

        # Allow panel to shrink to small sizes (height only, full width)
        self.setMinimumHeight(50)

        # Search/regex filter settings
        self.search_text = ""
        self.use_regex = False
        self.case_sensitive = False

        # Redirect Python stdout/stderr to capture STUDIO logs
        self._setup_stdout_capture()

        # Initialize UI directly on this widget
        self.init_ui(self)

        # Start WebSocket connection in background thread
        self.start_log_stream()

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Toolbar
        from PyQt6.QtWidgets import QCheckBox
        toolbar = QHBoxLayout()
        toolbar.setSpacing(15)  # More spacing for readability

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(50)
        clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(clear_btn)

        toolbar.addWidget(QLabel("|"))  # Separator

        # MUSH/STUDIO mode toggle buttons
        self.mush_btn = QPushButton("MUSH")
        self.mush_btn.setCheckable(True)
        self.mush_btn.setChecked(True)
        self.mush_btn.setFixedWidth(60)
        self.mush_btn.clicked.connect(lambda: self.set_console_mode('mush'))
        self.mush_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5c8f;
                color: #FFFFFF;
                border: 1px solid #4a7cba;
                padding: 4px;
                font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #3a3a3a;
                color: #888888;
                border: 1px solid #555;
            }
        """)
        toolbar.addWidget(self.mush_btn)

        self.studio_btn = QPushButton("STUDIO")
        self.studio_btn.setCheckable(True)
        self.studio_btn.setChecked(False)
        self.studio_btn.setFixedWidth(70)
        self.studio_btn.clicked.connect(lambda: self.set_console_mode('studio'))
        self.studio_btn.setStyleSheet("""
            QPushButton {
                background-color: #2d5c8f;
                color: #FFFFFF;
                border: 1px solid #4a7cba;
                padding: 4px;
                font-weight: bold;
            }
            QPushButton:!checked {
                background-color: #3a3a3a;
                color: #888888;
                border: 1px solid #555;
            }
        """)
        toolbar.addWidget(self.studio_btn)

        toolbar.addWidget(QLabel("|"))  # Separator

        # Search field with regex support
        filter_label = QLabel("Filter:")
        filter_label.setStyleSheet("color: #888888; font-size: 11pt; padding-right: 2px;")
        toolbar.addWidget(filter_label)

        self.search_field = QLineEdit()
        self.search_field.setPlaceholderText("search text or regex pattern...")
        self.search_field.setFixedWidth(250)
        self.search_field.setStyleSheet("""
            QLineEdit {
                background-color: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 2px;
            }
            QLineEdit:focus {
                border: 1px solid #4a7cba;
            }
        """)
        self.search_field.textChanged.connect(self.on_search_changed)
        toolbar.addWidget(self.search_field)

        # Clear search button
        self.clear_search_btn = QPushButton("✕")
        self.clear_search_btn.setFixedSize(24, 24)
        self.clear_search_btn.setToolTip("Clear search filter")
        self.clear_search_btn.setStyleSheet("""
            QPushButton {
                background-color: #3a3a3a;
                color: #888888;
                border: 1px solid #555;
                border-radius: 2px;
                font-size: 14pt;
                padding: 0px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
                color: #AAAAAA;
            }
            QPushButton:pressed {
                background-color: #2a2a2a;
            }
        """)
        self.clear_search_btn.clicked.connect(self.clear_search)
        toolbar.addWidget(self.clear_search_btn)

        # Regex toggle checkbox
        self.cb_regex = QCheckBox("Regex")
        self.cb_regex.setChecked(False)
        self.cb_regex.setToolTip("Enable regular expression matching")
        self.cb_regex.setStyleSheet("color: #D2D2D2; font-size: 11pt; padding: 2px;")
        self.cb_regex.setFixedWidth(65)
        self.cb_regex.toggled.connect(self.on_search_mode_changed)
        toolbar.addWidget(self.cb_regex)

        # Case sensitive toggle
        self.cb_case = QCheckBox("Aa")
        self.cb_case.setChecked(False)
        self.cb_case.setToolTip("Case sensitive search")
        self.cb_case.setStyleSheet("color: #D2D2D2; font-size: 11pt; padding: 2px;")
        self.cb_case.setFixedWidth(45)
        self.cb_case.toggled.connect(self.on_search_mode_changed)
        toolbar.addWidget(self.cb_case)

        toolbar.addStretch()

        # Font size controls
        font_label = QLabel("Font:")
        font_label.setStyleSheet("color: #888888; font-size: 9pt;")
        toolbar.addWidget(font_label)

        decrease_btn = QPushButton("A-")
        decrease_btn.setMaximumWidth(40)
        decrease_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 2px;")
        decrease_btn.clicked.connect(self.decrease_font_size)
        toolbar.addWidget(decrease_btn)

        self.font_size_label = QLabel(f"{self.font_size}pt")
        self.font_size_label.setStyleSheet("color: #CCCCCC; font-size: 9pt; min-width: 30px;")
        toolbar.addWidget(self.font_size_label)

        increase_btn = QPushButton("A+")
        increase_btn.setMaximumWidth(40)
        increase_btn.setStyleSheet("background-color: #3E3E3E; color: #CCCCCC; border: 1px solid #555; padding: 2px;")
        increase_btn.clicked.connect(self.increase_font_size)
        toolbar.addWidget(increase_btn)

        layout.addLayout(toolbar)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monaco", self.font_size))
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: none;
                font-size: {self.font_size}pt;
            }}
        """)
        layout.addWidget(self.log_text)

    def start_log_stream(self):
        """Start WebSocket connection to noodleMUSH logs."""
        self.log_text.append("[Console] Connecting to noodleMUSH logs...")
        self.log_text.append(f"[Console] WebSocket: {self.ws_url}")

        # Start WebSocket worker thread
        self.ws_worker = WebSocketWorker(self.ws_url)
        self.ws_worker.logReceived.connect(self.on_log_received)
        self.ws_worker.connected.connect(self.on_connection_changed)
        self.ws_worker.start()

        self.log_text.append("[Console] Log streaming started")

    def reconnect(self):
        """Reconnect to noodleMUSH logs (when server restarts)."""
        # Stop existing worker if any
        if hasattr(self, 'ws_worker'):
            if self.ws_worker.isRunning():
                self.ws_worker.stop()
                self.ws_worker.wait(1000)  # Wait up to 1 second
            # Disconnect old signals
            try:
                self.ws_worker.logReceived.disconnect()
                self.ws_worker.connected.disconnect()
            except:
                pass  # Signals might not be connected

        # Start new connection
        self.connected = False
        self.log_text.append("[Console] Reconnecting to noodleMUSH...")
        self.start_log_stream()

    @pyqtSlot(str, str, str)
    def on_log_received(self, level: str, module: str, message: str):
        """Handle incoming log from WebSocket."""
        self.add_log(level, module, message)

    @pyqtSlot(bool)
    def on_connection_changed(self, is_connected: bool):
        """Handle connection status change."""
        self.connected = is_connected
        if is_connected:
            self.log_text.append("[Console] <span style='color: #76AF6A;'>Connected and streaming logs</span>")
        else:
            self.log_text.append("[Console] <span style='color: #999;'>Disconnected</span>")

    def add_log(self, level: str, module: str, message: str):
        """Add log entry to console with Unity-style collapsing."""
        # Store raw log data for filtering
        raw_entry = f"[{level}] [{module}] {message}"
        self.log_buffer_raw.append(raw_entry)
        if len(self.log_buffer_raw) > 1000:
            self.log_buffer_raw.pop(0)

        # Create message signature for comparison (ignore timestamps)
        msg_signature = f"{level}:{module}:{message[:100]}"

        # Format message
        color_map = {
            'INFO': '#D2D2D2',
            'WARNING': '#FFA726',
            'ERROR': '#EF5350',
            'DEBUG': '#999999'
        }
        color = color_map.get(level, '#D2D2D2')

        # Check if this is a repeat of the last message
        if self.last_message == msg_signature:
            self.repeat_count += 1

            formatted = (
                f'<span style="color: #666;">[{level}]</span> '
                f'<span style="color: #64B5F6;">[{module}]</span> '
                f'<span style="color: {color};">{message}</span> '
                f'<span style="color: #999; font-weight: bold;">(x{self.repeat_count + 1})</span>'
            )

            # Update buffer
            if self.log_buffer:
                self.log_buffer[-1] = formatted

            # Only update display if matches search filter (or no filter active)
            if not self.search_text:
                # No filter - update the last line with repeat count
                cursor = self.log_text.textCursor()
                cursor.movePosition(QTextCursor.MoveOperation.End)
                cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                cursor.removeSelectedText()
                cursor.deletePreviousChar()  # Remove newline
                self.log_text.append(formatted)
            else:
                # Filter active - check if entry matches
                search_target = raw_entry
                search_term = self.search_text if self.case_sensitive else self.search_text.lower()
                search_haystack = search_target if self.case_sensitive else search_target.lower()

                matches = False
                if self.use_regex:
                    import re
                    try:
                        flags = 0 if self.case_sensitive else re.IGNORECASE
                        matches = bool(re.search(search_term, search_haystack, flags))
                    except re.error:
                        matches = True
                else:
                    matches = search_term in search_haystack

                if matches:
                    # Update last line with repeat count
                    cursor = self.log_text.textCursor()
                    cursor.movePosition(QTextCursor.MoveOperation.End)
                    cursor.select(QTextCursor.SelectionType.LineUnderCursor)
                    cursor.removeSelectedText()
                    cursor.deletePreviousChar()
                    self.log_text.append(formatted)
        else:
            # New message - reset counter
            self.last_message = msg_signature
            self.repeat_count = 0

            formatted = (
                f'<span style="color: #666;">[{level}]</span> '
                f'<span style="color: #64B5F6;">[{module}]</span> '
                f'<span style="color: {color};">{message}</span>'
            )

            # Store formatted log
            self.log_buffer.append(formatted)
            if len(self.log_buffer) > 1000:
                self.log_buffer.pop(0)

            # Only append to display if it matches current search filter
            if self.search_text:
                # Check if this entry matches the search
                search_target = raw_entry if not self.case_sensitive else raw_entry
                search_term = self.search_text if self.case_sensitive else self.search_text.lower()
                search_haystack = search_target if self.case_sensitive else search_target.lower()

                if self.use_regex:
                    import re
                    try:
                        flags = 0 if self.case_sensitive else re.IGNORECASE
                        if re.search(search_term, search_haystack, flags):
                            self.log_text.append(formatted)
                    except re.error:
                        # Invalid regex - show anyway
                        self.log_text.append(formatted)
                else:
                    # Simple substring match
                    if search_term in search_haystack:
                        self.log_text.append(formatted)
            else:
                # No filter active - show all logs
                self.log_text.append(formatted)

        # Auto-scroll to bottom
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def set_selected_entities(self, entity_ids: list):
        """
        Update selected entities (legacy interface - now handled by regex filtering).

        Kept as no-op stub for backwards compatibility with main_window.py
        """
        pass

    def clear_logs(self):
        """Clear log display."""
        self.log_text.clear()
        self.log_text.append("[Console] Logs cleared")

    def clear_search(self):
        """Clear the search filter field."""
        self.search_field.clear()

    def on_search_changed(self, text):
        """Handle search text change - update filter in real-time."""
        self.search_text = text
        self.apply_search_filter()

    def on_search_mode_changed(self, checked):
        """Handle regex or case sensitivity toggle."""
        self.use_regex = self.cb_regex.isChecked()
        self.case_sensitive = self.cb_case.isChecked()
        self.apply_search_filter()

    def apply_search_filter(self):
        """Apply search filter to log display with highlighting."""
        if not self.search_text:
            # No filter - restore full display
            self.refresh_display()
            return

        # Build regex pattern
        try:
            if self.use_regex:
                # User provided regex pattern
                pattern = self.search_text
            else:
                # Escape special chars for literal matching
                pattern = re.escape(self.search_text)

            # Compile with appropriate flags
            flags = 0 if self.case_sensitive else re.IGNORECASE
            regex = re.compile(pattern, flags)

            # Filter and highlight matching logs
            self.log_text.clear()

            # Choose appropriate buffers based on mode
            if self.console_mode == 'studio':
                raw_buffer = self.studio_log_buffer_raw
                formatted_buffer = self.studio_log_buffer
            else:
                raw_buffer = self.log_buffer_raw
                formatted_buffer = self.log_buffer

            match_count = 0
            for i, raw_entry in enumerate(raw_buffer):
                # Check if raw log matches pattern
                if regex.search(raw_entry):
                    match_count += 1
                    # Get formatted version
                    if i < len(formatted_buffer):
                        formatted = formatted_buffer[i]
                    else:
                        # Fallback to raw if formatted not available
                        formatted = f'<span style="color: #D2D2D2;">{raw_entry}</span>'

                    # Highlight matches in formatted HTML
                    # Need to highlight in the raw text parts, not in HTML tags
                    highlighted = regex.sub(
                        lambda m: f'<span style="background-color: #FFA726; color: #000;">{m.group(0)}</span>',
                        raw_entry
                    )
                    # Wrap in basic formatting if not already formatted
                    if not highlighted.startswith('<span'):
                        highlighted = f'<span style="color: #D2D2D2;">{highlighted}</span>'

                    self.log_text.append(highlighted)

            # Show filter status
            if match_count == 0:
                self.log_text.append(f"[Console] <span style='color: #FFA726;'>No matches found for: {self.search_text}</span>")
            else:
                self.log_text.append(f"[Console] <span style='color: #76AF6A;'>Found {match_count} matches</span>")

        except re.error as e:
            # Invalid regex pattern
            self.log_text.clear()
            self.log_text.append(f"[Console] <span style='color: #EF5350;'>Invalid regex pattern: {e}</span>")

    def refresh_display(self):
        """Refresh log display without filter."""
        self.log_text.clear()
        buffer = self.studio_log_buffer if self.console_mode == 'studio' else self.log_buffer
        for log_entry in buffer:
            self.log_text.append(log_entry)

    def filter_logs(self):
        """Filter logs by search text and level."""
        # Implemented via on_search_changed and apply_search_filter
        pass

    def increase_font_size(self):
        """Increase console font size."""
        self.font_size = min(24, self.font_size + 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self.log_text.setFont(QFont("Monaco", self.font_size))
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: none;
                font-size: {self.font_size}pt;
            }}
        """)

    def decrease_font_size(self):
        """Decrease console font size."""
        self.font_size = max(8, self.font_size - 2)
        self.font_size_label.setText(f"{self.font_size}pt")
        self.log_text.setFont(QFont("Monaco", self.font_size))
        self.log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: none;
                font-size: {self.font_size}pt;
            }}
        """)

    def _setup_stdout_capture(self):
        """Capture Python stdout/stderr for STUDIO mode."""
        import sys

        class StdoutCapture:
            def __init__(self, console_panel):
                self.console_panel = console_panel
                self.original_stdout = sys.stdout
                self.original_stderr = sys.stderr

            def write(self, text):
                # Write to original stdout (terminal)
                self.original_stdout.write(text)
                # Also capture to STUDIO log buffer
                if text.strip():
                    self.console_panel.add_studio_log(text.strip())

            def flush(self):
                self.original_stdout.flush()

        # Install stdout/stderr capture
        self.stdout_capture = StdoutCapture(self)
        sys.stdout = self.stdout_capture
        sys.stderr = self.stdout_capture

    def add_studio_log(self, message):
        """Add message to STUDIO log buffer."""
        # Store raw log
        self.studio_log_buffer_raw.append(message)
        if len(self.studio_log_buffer_raw) > 1000:
            self.studio_log_buffer_raw.pop(0)

        # Store formatted log
        formatted = f'<span style="color: #D2D2D2;">{message}</span>'
        self.studio_log_buffer.append(formatted)
        if len(self.studio_log_buffer) > 1000:
            self.studio_log_buffer.pop(0)

        # If in STUDIO mode, update display (respecting search filter)
        if self.console_mode == 'studio':
            # Only append if matches search filter
            if self.search_text:
                search_term = self.search_text if self.case_sensitive else self.search_text.lower()
                search_haystack = message if self.case_sensitive else message.lower()

                matches = False
                if self.use_regex:
                    import re
                    try:
                        flags = 0 if self.case_sensitive else re.IGNORECASE
                        matches = bool(re.search(search_term, search_haystack, flags))
                    except re.error:
                        matches = True
                else:
                    matches = search_term in search_haystack

                if matches:
                    self.log_text.append(formatted)
            else:
                # No filter - show all logs
                self.log_text.append(formatted)

    def set_console_mode(self, mode):
        """Switch between MUSH and STUDIO console modes."""
        self.console_mode = mode

        # Update button states
        self.mush_btn.setChecked(mode == 'mush')
        self.studio_btn.setChecked(mode == 'studio')

        # Clear and refresh display with search filter applied
        self.apply_search_filter()
