"""
Chat Panel - Embeds noodleMUSH web interface.
"""

from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QFont
import sys
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock

# Try to import WebEngine, fallback to placeholder if not available
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False


class ChatPanel(MaximizableDock):
    """
    Chat panel displaying noodleMUSH web interface.

    Embeds the existing web/index.html via QWebEngineView.
    If WebEngine is not available, shows instructions to install it.
    """

    def __init__(self, parent: QWidget = None):
        super().__init__("Chat View", parent)

        # Allow moving and floating
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        self._setup_ui()

    def _setup_ui(self):
        """Build UI components."""
        if WEBENGINE_AVAILABLE:
            self._setup_web_view()
        else:
            self._setup_fallback()

    def _setup_web_view(self):
        """Setup QWebEngineView to show noodleMUSH."""
        from PyQt6.QtWidgets import QStackedWidget, QComboBox, QHBoxLayout

        # Main container
        container = QWidget()
        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Renderer selector toolbar (left-aligned, compact)
        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(4, 2, 4, 2)
        toolbar.setSpacing(6)

        self.renderer_selector = QComboBox()
        self.renderer_selector.addItem("Chat")
        self.renderer_selector.insertSeparator(1)
        self.renderer_selector.addItem("Manage Renderers...")
        self.renderer_selector.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #444;
                padding: 3px 8px;
                border-radius: 2px;
                font-size: 11px;
            }
            QComboBox:hover {
                background: #3a3a3a;
                border: 1px solid #555;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #555;
                selection-background-color: #4a4a4a;
            }
        """)
        self.renderer_selector.currentTextChanged.connect(self.on_renderer_changed)

        # Disable "Manage Renderers..." option (grayed out)
        model = self.renderer_selector.model()
        item = model.item(2)
        if item:
            item.setEnabled(False)

        toolbar.addWidget(self.renderer_selector)
        toolbar.addStretch()

        main_layout.addLayout(toolbar)

        # Web view directly (fills remaining space)
        self.web_view = QWebEngineView()

        # Enable console message forwarding
        page = self.web_view.page()
        page.javaScriptConsoleMessage = self._on_console_message

        # Create error overlay (shown when server is down)
        stack = QStackedWidget()

        self.error_overlay = self._create_error_overlay()

        stack.addWidget(self.web_view)
        stack.addWidget(self.error_overlay)
        stack.setCurrentWidget(self.web_view)

        self.stack = stack

        # Try to load noodleMUSH
        self.web_view.setUrl(QUrl("http://localhost:8080"))

        # Check if page loads successfully
        self.web_view.loadFinished.connect(self._on_load_finished)

        main_layout.addWidget(stack, stretch=1)  # Stretch to fill

        self.setWidget(container)

    def _create_error_overlay(self):
        """Create error overlay for when server is down."""
        overlay = QWidget()
        overlay_layout = QVBoxLayout()
        overlay_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Error icon
        error_icon = QLabel("🔌")
        error_icon.setFont(QFont("", 72))
        error_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(error_icon)

        # Title
        title = QLabel("noodleMUSH Server Offline")
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffa726; margin: 20px;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overlay_layout.addWidget(title)

        # Message
        message = QLabel(
            "The noodleMUSH server is not running.\n\n"
            "Use the toggle switch in the status bar (bottom-right)\n"
            "to start the server, or run manually:\n\n"
            "cd applications/cmush && ./start.sh"
        )
        message.setFont(QFont("Arial", 16))
        message.setStyleSheet("color: #b0b0b0; padding: 30px;")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        message.setWordWrap(True)
        overlay_layout.addWidget(message)

        # Retry button
        retry_btn = QPushButton("Retry Connection")
        retry_btn.setStyleSheet("""
            QPushButton {
                background: #4CAF50;
                color: white;
                padding: 10px 20px;
                border-radius: 4px;
                font-size: 14px;
            }
            QPushButton:hover {
                background: #45a049;
            }
        """)
        retry_btn.clicked.connect(self.reload)
        overlay_layout.addWidget(retry_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        overlay.setLayout(overlay_layout)
        overlay.setStyleSheet("background: #2a2a2a;")
        return overlay

    def _on_load_finished(self, ok: bool):
        """Handle page load completion."""
        if not ok:
            # Page failed to load - show error overlay
            self.stack.setCurrentWidget(self.error_overlay)
        else:
            # Page loaded successfully - show web view
            self.stack.setCurrentWidget(self.web_view)

    def _setup_fallback(self):
        """Setup fallback UI when WebEngine is not available."""
        container = QWidget()
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Error message
        error_icon = QLabel("⚠️")
        error_icon.setFont(QFont("", 48))
        error_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(error_icon)

        title = QLabel("PyQt6-WebEngine Not Installed")
        title.setFont(QFont("Roboto", 16, QFont.Weight.Bold))
        title.setStyleSheet("color: #ffa726;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        message = QLabel(
            "To use the Chat View, install PyQt6-WebEngine:\n\n"
            "pip install PyQt6-WebEngine\n\n"
            "Then restart NoodleSTUDIO."
        )
        message.setFont(QFont("Source Code Pro", 12))
        message.setStyleSheet("color: #b0b0b0; padding: 20px;")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(message)

        # Manual alternative
        manual = QLabel(
            "Or open noodleMUSH manually:\n"
            "http://localhost:8080"
        )
        manual.setFont(QFont("Roboto", 11))
        manual.setStyleSheet("color: #808080;")
        manual.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(manual)

        container.setLayout(layout)
        self.setWidget(container)

    def reload(self):
        """Reload the web page."""
        if WEBENGINE_AVAILABLE and hasattr(self, 'web_view'):
            self.web_view.reload()

    def _on_console_message(self, level, message, line, source):
        """Forward browser console messages to our console."""
        level_str = {0: "INFO", 1: "WARNING", 2: "ERROR"}.get(level, "LOG")
        print(f"[Browser {level_str}] {source}:{line} - {message}")

    def on_renderer_changed(self, renderer_name: str):
        """Handle renderer selection change."""
        if renderer_name == "Manage Renderers...":
            # Reset to Chat
            self.renderer_selector.setCurrentText("Chat")
            # TODO: Open renderer management dialog
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Manage Renderers",
                "Renderer management coming soon!\n\n"
                "Future renderers:\n"
                "- 3D View (USD scene)\n"
                "- Graph View (relationship network)\n"
                "- Timeline (affect over time)\n"
                "- Custom (user-created)"
            )
        elif renderer_name == "Chat":
            # Default chat renderer (already active)
            pass
