"""
Chat Panel - Embeds noodleMUSH web interface.
"""

from PyQt6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import QUrl, Qt
from PyQt6.QtGui import QFont

# Try to import WebEngine, fallback to placeholder if not available
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    WEBENGINE_AVAILABLE = True
except ImportError:
    WEBENGINE_AVAILABLE = False


class ChatPanel(QDockWidget):
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
        container = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # Create web view
        self.web_view = QWebEngineView()

        # Try to load noodleMUSH
        # If server is running, this will work
        # If not, will show error page
        self.web_view.setUrl(QUrl("http://localhost:8080"))

        layout.addWidget(self.web_view)
        container.setLayout(layout)
        self.setWidget(container)

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
