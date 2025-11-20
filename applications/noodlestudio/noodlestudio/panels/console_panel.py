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
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock


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


class ConsolePanel(MaximizableDock):
    """
    Console panel showing live logs from noodleMUSH.

    Connects to WebSocket and streams log messages.
    Unity-style message collapsing for repeated logs.
    """

    def __init__(self, parent=None):
        super().__init__("Console", parent)
        self.ws_url = "ws://localhost:8765"
        self.connected = False
        self.log_buffer = []
        self.last_message = None  # Track last message for collapsing
        self.repeat_count = 0
        self.selected_entities = []  # Track selected entities for filtering

        # Filter settings
        self.filter_selected_only = False
        self.show_warnings = True
        self.show_info = True
        self.show_llm = True
        self.show_ruminations = True

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

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

        # Compact checkboxes for filtering
        def create_filter_checkbox(label, tooltip, checked=True):
            cb = QCheckBox(label)
            cb.setChecked(checked)
            cb.setToolTip(tooltip)
            cb.setStyleSheet("color: #D2D2D2; font-size: 13px; padding: 2px;")

            # Calculate exact width needed using QFontMetrics
            font = QFont("Arial", 13)
            metrics = QFontMetrics(font)
            text_width = metrics.horizontalAdvance(label)
            # Add space for checkbox indicator (13px) + padding (20px) + margin (10px)
            total_width = text_width + 13 + 20 + 10
            cb.setFixedWidth(total_width)

            return cb

        self.cb_selected = create_filter_checkbox("Sel", "Show only selected entities", False)
        self.cb_selected.toggled.connect(self.on_filter_changed)
        toolbar.addWidget(self.cb_selected)

        self.cb_warnings = create_filter_checkbox("W", "Show warnings", True)
        self.cb_warnings.toggled.connect(self.on_filter_changed)
        toolbar.addWidget(self.cb_warnings)

        self.cb_info = create_filter_checkbox("I", "Show info messages", True)
        self.cb_info.toggled.connect(self.on_filter_changed)
        toolbar.addWidget(self.cb_info)

        self.cb_llm = create_filter_checkbox("LLM", "Show LLM traffic", True)
        self.cb_llm.toggled.connect(self.on_filter_changed)
        toolbar.addWidget(self.cb_llm)

        self.cb_ruminations = create_filter_checkbox("Rum", "Show ruminations", True)
        self.cb_ruminations.toggled.connect(self.on_filter_changed)
        toolbar.addWidget(self.cb_ruminations)

        toolbar.addStretch()

        layout.addLayout(toolbar)

        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Monaco", 10))
        self.log_text.setStyleSheet("""
            QTextEdit {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: none;
            }
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
        # Apply filters
        # 1. Level filters
        if level == 'WARNING' and not self.show_warnings:
            return
        if level == 'INFO' and not self.show_info:
            return

        # 2. Content filters
        if not self.show_llm and ('LLM REQUEST' in message or '🤖' in message or 'llm_interface' in module):
            return
        if not self.show_ruminations and ('thinking' in message.lower() or 'rumination' in message.lower()):
            return

        # 3. Selected entities filter
        if self.filter_selected_only and self.selected_entities:
            # Check if message contains any of the selected entity IDs
            message_text = message.lower() + module.lower()
            matches = any(entity.lower() in message_text for entity in self.selected_entities)
            if not matches:
                return

        # Create message signature for comparison (ignore timestamps)
        msg_signature = f"{level}:{module}:{message[:100]}"

        # Check if this is a repeat of the last message
        if self.last_message == msg_signature:
            self.repeat_count += 1

            # Update the last line to show count
            cursor = self.log_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            cursor.select(QTextCursor.SelectionType.LineUnderCursor)
            cursor.removeSelectedText()
            cursor.deletePreviousChar()  # Remove newline

            # Re-add with count
            color_map = {
                'INFO': '#D2D2D2',
                'WARNING': '#FFA726',
                'ERROR': '#EF5350',
                'DEBUG': '#999999'
            }
            color = color_map.get(level, '#D2D2D2')

            self.log_text.append(
                f'<span style="color: #666;">[{level}]</span> '
                f'<span style="color: #64B5F6;">[{module}]</span> '
                f'<span style="color: {color};">{message}</span> '
                f'<span style="color: #999; font-weight: bold;">(x{self.repeat_count + 1})</span>'
            )
        else:
            # New message - reset counter
            self.last_message = msg_signature
            self.repeat_count = 0

            # Color code by level
            color_map = {
                'INFO': '#D2D2D2',
                'WARNING': '#FFA726',
                'ERROR': '#EF5350',
                'DEBUG': '#999999'
            }
            color = color_map.get(level, '#D2D2D2')

            self.log_text.append(
                f'<span style="color: #666;">[{level}]</span> '
                f'<span style="color: #64B5F6;">[{module}]</span> '
                f'<span style="color: {color};">{message}</span>'
            )

        # Auto-scroll to bottom
        self.log_text.moveCursor(QTextCursor.MoveOperation.End)

    def set_selected_entities(self, entity_ids: list):
        """Update selected entities for filtering."""
        self.selected_entities = entity_ids
        if self.filter_selected_only and entity_ids:
            entity_names = [e.replace('agent_', '').replace('obj_', '') for e in entity_ids]
            self.log_text.append(f"[Console] <span style='color: #64B5F6;'>Filtering to: {', '.join(entity_names)}</span>")

    def on_filter_changed(self, checked: bool):
        """Handle filter checkbox toggle - update filter flags."""
        # Update filter flags from checkboxes
        self.filter_selected_only = self.cb_selected.isChecked()
        self.show_warnings = self.cb_warnings.isChecked()
        self.show_info = self.cb_info.isChecked()
        self.show_llm = self.cb_llm.isChecked()
        self.show_ruminations = self.cb_ruminations.isChecked()

        # Show filter status
        filters = []
        if self.filter_selected_only:
            filters.append("selected only")
        if not self.show_warnings:
            filters.append("no warnings")
        if not self.show_info:
            filters.append("no info")
        if not self.show_llm:
            filters.append("no LLM")
        if not self.show_ruminations:
            filters.append("no ruminations")

        if filters:
            self.log_text.append(f"[Console] <span style='color: #FFA726;'>Filters: {', '.join(filters)}</span>")

    def clear_logs(self):
        """Clear log display."""
        self.log_text.clear()
        self.log_text.append("[Console] Logs cleared")

    def filter_logs(self):
        """Filter logs by search text and level."""
        # TODO: Implement actual filtering
        pass
