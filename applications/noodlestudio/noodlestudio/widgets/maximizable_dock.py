"""
Maximizable Dock Widget

Double-click title bar to toggle fullscreen (within app).

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import QDockWidget
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
