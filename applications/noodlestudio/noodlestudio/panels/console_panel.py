"""
Console Panel - Live log viewer from noodleMUSH

Connects to noodleMUSH WebSocket and displays real-time logs.
Like Unity's Console panel.

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                             QTextEdit, QLineEdit, QPushButton, QLabel, QComboBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSlot, QThread, pyqtSignal
from PyQt6.QtGui import QFont, QTextCursor
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
                print("WebSocket connected, authenticating...")

                # Authenticate (use caity's credentials)
                await ws.send(json.dumps({
                    'type': 'login',
                    'username': 'caity',
                    'password': 'caity'
                }))

                # Wait for auth response
                response = await ws.recv()
                data = json.loads(response)
                print(f"Auth response: {data}")

                if not data.get('success'):
                    print(f"Failed to authenticate for logs: {data.get('message')}")
                    self.logReceived.emit('ERROR', 'Console', f"Auth failed: {data.get('message')}")
                    return

                print("Authenticated! Subscribing to logs...")
                self.logReceived.emit('INFO', 'Console', 'Connected to noodleMUSH')

                # Subscribe to logs
                await ws.send(json.dumps({'type': 'subscribe_logs'}))

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
        toolbar = QHBoxLayout()

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.clicked.connect(self.clear_logs)
        toolbar.addWidget(clear_btn)

        # Filter
        toolbar.addWidget(QLabel("Filter:"))
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Search logs...")
        self.filter_input.textChanged.connect(self.filter_logs)
        toolbar.addWidget(self.filter_input)

        # Log level filter
        self.level_filter = QComboBox()
        self.level_filter.addItems(["All", "INFO", "WARNING", "ERROR", "DEBUG"])
        self.level_filter.currentTextChanged.connect(self.filter_logs)
        self.level_filter.setFixedWidth(100)
        toolbar.addWidget(self.level_filter)

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
        self.ws_worker.start()

        self.log_text.append("[Console] Log streaming started")

    @pyqtSlot(str, str, str)
    def on_log_received(self, level: str, module: str, message: str):
        """Handle incoming log from WebSocket."""
        self.add_log(level, module, message)

    def add_log(self, level: str, module: str, message: str):
        """Add log entry to console with Unity-style collapsing."""
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

    def clear_logs(self):
        """Clear log display."""
        self.log_text.clear()
        self.log_text.append("[Console] Logs cleared")

    def filter_logs(self):
        """Filter logs by search text and level."""
        # TODO: Implement actual filtering
        pass
