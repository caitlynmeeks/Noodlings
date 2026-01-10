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
#   Maximizable Dock Widget
#
#   Double-click title bar to toggle fullscreen (within app).
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.maximizable_dock
# PURPOSE:  Maximizable Dock Widget
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MaximizableDock
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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
        # Don't replace title bar - keep native Qt one for docking to work
        # Just enable double-click maximize via event filter
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
