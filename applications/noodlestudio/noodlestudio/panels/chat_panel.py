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
        super().__init__("World View", parent)

        # Allow moving and floating
        self.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        # Set dock background to black and ZERO margins
        self.setStyleSheet("""
            QDockWidget {
                background-color: #000000;
            }
            QDockWidget::widget {
                padding: 0px;
                margin: 0px;
            }
        """)

        # Force zero content margins
        self.setContentsMargins(0, 0, 0, 0)

        # Server state tracking
        self.server_running = False

        # Create custom title bar with renderer selector
        self._create_custom_title_bar()

        self._setup_ui()

    def _create_custom_title_bar(self):
        """Create custom title bar with renderer selector."""
        from PyQt6.QtWidgets import QComboBox, QHBoxLayout

        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(8, 4, 8, 4)
        title_layout.setSpacing(8)

        # Left side - World View title
        title_label = QLabel("World View")
        title_label.setStyleSheet("color: #D2D2D2; font-size: 11px; font-weight: bold;")
        title_layout.addWidget(title_label)

        # Push to right
        title_layout.addStretch()

        # Right side - Renderer label + dropdown
        renderer_label = QLabel("Renderer")
        renderer_label.setStyleSheet("color: #888; font-size: 11px;")
        title_layout.addWidget(renderer_label)

        self.renderer_selector = QComboBox()
        self.renderer_selector.addItem("Reductive Text")
        self.renderer_selector.insertSeparator(1)
        self.renderer_selector.addItem("Generative Renderers")
        self.renderer_selector.insertSeparator(3)
        self.renderer_selector.addItem("Manage Renderers...")
        self.renderer_selector.setStyleSheet("""
            QComboBox {
                background: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #444;
                padding: 0px;
                border-radius: 2px;
                font-size: 11px;
                min-height: 20px;
            }
            QComboBox:hover {
                background: #3a3a3a;
                border: 1px solid #555;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 5px;
            }
            QComboBox QAbstractItemView {
                background: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #555;
                selection-background-color: #4a4a4a;
                padding-top: 8px;
                padding-bottom: 4px;
            }
            QComboBox QAbstractItemView::item {
                padding: 4px 8px;
                min-height: 20px;
            }
        """)
        self.renderer_selector.currentTextChanged.connect(self.on_renderer_changed)

        # Disable future options (grayed out placeholders)
        model = self.renderer_selector.model()
        item = model.item(2)
        if item:
            item.setEnabled(False)
        item = model.item(4)
        if item:
            item.setEnabled(False)

        title_layout.addWidget(self.renderer_selector)

        # Set as custom title bar
        self.setTitleBarWidget(title_widget)

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

        # Force container to expand fully
        from PyQt6.QtWidgets import QSizePolicy
        container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        container.setMinimumHeight(0)

        main_layout = QVBoxLayout(container)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Web view directly (fills all space - renderer is in title bar)
        self.web_view = QWebEngineView()

        # Force web view to expand fully
        from PyQt6.QtWidgets import QSizePolicy, QWIDGETSIZE_MAX
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.web_view.setMinimumSize(0, 0)
        self.web_view.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)

        # Ensure web view background is black
        self.web_view.setStyleSheet("background-color: #000000;")

        # Enable console message forwarding
        page = self.web_view.page()
        page.javaScriptConsoleMessage = self._on_console_message

        # Fix viewport height issue (100vh doesn't work in QWebEngineView)
        def fix_viewport_height(ok):
            if ok:
                self.web_view.page().runJavaScript("""
                    // Force html and body to use 100% height instead of 100vh
                    document.documentElement.style.height = '100%';
                    document.body.style.height = '100%';
                    // Also fix any containers using 100vh
                    document.querySelectorAll('[style*="100vh"]').forEach(el => {
                        el.style.height = '100%';
                    });

                    // DEBUG: Log all WebSocket messages
                    const originalOnMessage = ws.onmessage;
                    ws.onmessage = function(event) {
                        console.log('[STUDIO DEBUG] WebSocket message received:', event.data);
                        if (originalOnMessage) {
                            originalOnMessage.call(this, event);
                        }
                    };
                    console.log('[STUDIO DEBUG] Message logging enabled');
                """)

        # Inject fix after page loads
        self.web_view.loadFinished.connect(fix_viewport_height)

        # Try to load noodleMUSH (with studio=true parameter)
        self.web_view.setUrl(QUrl("http://localhost:8080?studio=true"))

        # Add web view directly without stack (simpler = better)
        main_layout.addWidget(self.web_view, stretch=1)

        # Set background to black (no gray areas)
        container.setStyleSheet("background-color: #000000;")

        self.setWidget(container)

        # Store container reference for resize events
        self.container = container

        # Debug: Log sizes after a short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(500, lambda: self._debug_sizes(container))

    def resizeEvent(self, event):
        """Handle resize to force web view to fill space."""
        super().resizeEvent(event)
        if hasattr(self, 'web_view') and hasattr(self, 'container'):
            # Force web view to match container size exactly
            self.web_view.setFixedSize(self.container.size())
            # Then immediately unfix it so it can resize again
            from PyQt6.QtWidgets import QWIDGETSIZE_MAX
            self.web_view.setMaximumSize(QWIDGETSIZE_MAX, QWIDGETSIZE_MAX)

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

    def _debug_sizes(self, container):
        """Debug size information to identify layout issues."""
        debug_info = [
            "\n=== CHAT PANEL SIZE DEBUG ===",
            f"Dock widget size: {self.size().width()} x {self.size().height()}",
            f"Container size: {container.size().width()} x {container.size().height()}",
            f"Web view size: {self.web_view.size().width()} x {self.web_view.size().height()}",
            f"Container min height: {container.minimumHeight()}",
            f"Web view min height: {self.web_view.minimumHeight()}",
            f"Container size policy: {container.sizePolicy().verticalPolicy().name}",
            f"Web view size policy: {self.web_view.sizePolicy().verticalPolicy().name}",
            "============================\n"
        ]

        # Print to console
        for line in debug_info:
            print(line)

        # Also write to file for app launch debugging
        try:
            with open('/tmp/noodlestudio_chat_debug.log', 'w') as f:
                f.write('\n'.join(debug_info))
        except Exception as e:
            print(f"Failed to write debug log: {e}")

    def _on_console_message(self, level, message, line, source):
        """Forward browser console messages to our console."""
        level_str = {0: "INFO", 1: "WARNING", 2: "ERROR"}.get(level, "LOG")
        print(f"[Browser {level_str}] {source}:{line} - {message}")

    def on_renderer_changed(self, renderer_name: str):
        """Handle renderer selection change."""
        if renderer_name == "Manage Renderers...":
            # Reset to Reductive Text
            self.renderer_selector.setCurrentText("Reductive Text")
            # Show development dialog
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(
                self,
                "Manage Renderers",
                "Feature in development\n\n"
                "Planned generative renderers:\n"
                "- Runway AI (video generation)\n"
                "- Luma AI (real-time 3D)\n"
                "- Sora (cinematic video)\n"
                "- Custom renderer pipeline"
            )
        elif renderer_name == "Reductive Text":
            # Default text renderer (already active)
            pass
        elif renderer_name == "Generative Renderers":
            # This is a header, reset to current renderer
            self.renderer_selector.setCurrentText("Reductive Text")

    def show_offline_card(self):
        """Show offline card when server is not running."""
        if hasattr(self, 'web_view'):
            self.web_view.setHtml("""
                <html>
                <head>
                    <style>
                        body {
                            background: #000;
                            color: #00ff00;
                            font-family: 'Courier New', monospace;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            height: 100vh;
                            margin: 0;
                        }
                        .card {
                            text-align: center;
                            border: 2px solid #00ff00;
                            padding: 40px;
                            max-width: 500px;
                        }
                        h1 { font-size: 32px; margin-bottom: 20px; }
                        p { font-size: 16px; line-height: 1.6; }
                    </style>
                </head>
                <body>
                    <div class="card">
                        <h1>SERVER OFFLINE</h1>
                        <p>The noodleMUSH server is not running.</p>
                        <p>Use the toggle switch in the status bar (bottom-right) to start the server.</p>
                    </div>
                </body>
                </html>
            """)
            self.server_running = False

    def show_world_view(self):
        """Show the normal world view (server is running)."""
        if hasattr(self, 'web_view'):
            self.web_view.setUrl(QUrl("http://localhost:8080?studio=true"))
            self.server_running = True

    def set_server_state(self, running: bool):
        """Update UI based on server state."""
        if running:
            self.show_world_view()
        else:
            self.show_offline_card()
