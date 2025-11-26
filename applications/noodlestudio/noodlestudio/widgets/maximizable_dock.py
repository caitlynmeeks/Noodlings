"""
Maximizable Dock Widget

Double-click title bar to toggle fullscreen (within app).

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import QDockWidget, QPushButton, QWidget, QHBoxLayout
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QMouseEvent


class MaximizableDock(QDockWidget):
    """
    DockWidget that maximizes on double-click of title bar.

    Like professional tools - double-click header to go fullscreen.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.is_maximized = False
        self.saved_state = None
        self.installEventFilter(self)
        self._setup_title_bar_button()

    def _setup_title_bar_button(self):
        """Add maximize button to title bar."""
        # Create custom title bar widget with maximize button
        title_widget = QWidget()
        title_layout = QHBoxLayout(title_widget)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.setSpacing(0)

        # Spacer to push button to the right
        title_layout.addStretch()

        # Maximize/restore button
        self.maximize_button = QPushButton()
        self.maximize_button.setFixedSize(20, 20)
        self.maximize_button.clicked.connect(self.toggle_maximize)
        self._update_maximize_button_icon()
        self.maximize_button.setStyleSheet("""
            QPushButton {
                background: transparent;
                border: none;
                color: #AAAAAA;
                font-size: 14px;
                padding: 0px;
            }
            QPushButton:hover {
                background: #505050;
                color: #FFFFFF;
            }
        """)
        title_layout.addWidget(self.maximize_button)

        # Set the custom title bar
        self.setTitleBarWidget(title_widget)

    def _update_maximize_button_icon(self):
        """Update maximize button icon based on state."""
        if self.is_maximized:
            self.maximize_button.setText("◱")  # Restore icon
        else:
            self.maximize_button.setText("⬜")  # Maximize icon

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """Handle double-click on title bar to maximize."""
        # Check if click is on title bar area (top ~30px)
        if event.position().y() < 30:
            self.toggle_maximize()
        else:
            super().mouseDoubleClickEvent(event)

    def toggle_maximize(self):
        """Toggle between maximized and normal."""
        if not self.parent():
            return

        main_window = self.parent()

        if not self.is_maximized:
            # MAXIMIZE: Hide all other docks
            self.saved_docks = []

            for dock in main_window.findChildren(QDockWidget):
                if dock != self and dock.isVisible():
                    self.saved_docks.append(dock)
                    dock.hide()

            self.is_maximized = True

        else:
            # RESTORE: Show previously visible docks
            for dock in self.saved_docks:
                dock.show()

            self.saved_docks = []
            self.is_maximized = False

        # Update button icon
        self._update_maximize_button_icon()
