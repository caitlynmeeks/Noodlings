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
#   Account Status Widget for Status Bar
#
#   ==================================== Shows login state in...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.widgets.account_status_widget
# PURPOSE:  Account Status Widget for Status Bar
# LAYER:    Studio / Widgets
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AccountStatusWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QMenu
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QCursor

from ..core.account_manager import AccountManager, UserProfile


class AccountStatusWidget(QWidget):
    """
    Status bar widget showing account state.

    When logged out: Shows "Sign In" - click to open login dialog
    When logged in: Shows name - click for menu with Sign Out
    """

    sign_in_clicked = pyqtSignal()  # Emitted when sign in requested

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._setup_ui()
        self._connect_signals()
        self._update_display()

    def _setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(0)

        # User info label
        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #D2D2D2; font-size: 12px;")
        layout.addWidget(self.info_label)

        # Make clickable - styled like Enter World button
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setStyleSheet("""
            QWidget {
                background-color: #4a4a4a;
                border: 1px solid #555;
                border-radius: 3px;
            }
            QWidget:hover {
                background-color: #555;
            }
        """)

    def _connect_signals(self):
        """Connect to account manager signals."""
        account_manager = AccountManager.instance()
        account_manager.logged_in.connect(self._on_logged_in)
        account_manager.logged_out.connect(self._on_logged_out)

    def _update_display(self):
        """Update display based on current account state."""
        account_manager = AccountManager.instance()

        if account_manager.is_logged_in:
            user = account_manager.user
            self._show_logged_in(user)
        else:
            self._show_logged_out()

    def _show_logged_in(self, user: UserProfile):
        """Show logged-in state."""
        display = user.display_name or user.email.split('@')[0]
        if len(display) > 20:
            display = display[:17] + "..."
        self.info_label.setText(display)

    def _show_logged_out(self):
        """Show logged-out state."""
        self.info_label.setText("Sign In")

    def _on_logged_in(self, user: UserProfile):
        """Handle login event."""
        self._show_logged_in(user)

    def _on_logged_out(self):
        """Handle logout event."""
        self._show_logged_out()

    def mousePressEvent(self, event):
        """Handle click."""
        if event.button() == Qt.MouseButton.LeftButton:
            account_manager = AccountManager.instance()

            if account_manager.is_logged_in:
                # Show menu with sign out option
                menu = QMenu(self)
                menu.setStyleSheet("""
                    QMenu {
                        background-color: #2a2a2a;
                        border: 1px solid #404040;
                        padding: 4px;
                    }
                    QMenu::item {
                        color: #d2d2d2;
                        padding: 6px 20px;
                    }
                    QMenu::item:selected {
                        background-color: #404040;
                    }
                """)
                sign_out_action = menu.addAction("Sign Out")
                sign_out_action.triggered.connect(self._sign_out)
                menu.exec(self.mapToGlobal(event.pos()))
            else:
                # Emit signal to open login dialog
                self.sign_in_clicked.emit()

        super().mousePressEvent(event)

    def _sign_out(self):
        """Sign out the current user."""
        AccountManager.instance().logout()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
