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
#   Main Window Status Bar Mixin - Status bar setup and avatar dropdown
#
#   Contains: - _setup_status_bar: Create status bar with acc...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_statusbar_mixin
# PURPOSE:  Main Window Statusbar Mixin
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowStatusBarMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout, QComboBox, QPushButton, QMessageBox
from PyQt6.QtCore import Qt


class MainWindowStatusBarMixin:
    """Mixin providing status bar setup for MainWindow."""

    def _setup_status_bar(self):
        """Create status bar: [Username] [Avatar dropdown] [Enter World] [Server toggle]."""
        from ..widgets.toggle_switch import ToggleSwitch
        from ..widgets.account_status_widget import AccountStatusWidget
        from .account_manager import AccountManager

        status_bar = self.statusBar()

        # Connection status (temporary messages - left side, non-permanent)
        # Styled as pill to match annotation overlay
        self.connection_label = QLabel()
        self.connection_label.setStyleSheet("""
            QLabel {
                background-color: #282828;
                color: #76AF6A;
                border-radius: 4px;
                padding: 3px 8px;
                font-size: 12px;
            }
        """)
        status_bar.addWidget(self.connection_label)

        # === RIGHT SIDE PERMANENT WIDGETS (in order) ===

        # 1. Account status (username)
        self.account_status_widget = AccountStatusWidget()
        self.account_status_widget.sign_in_clicked.connect(self.show_login_dialog)
        status_bar.addPermanentWidget(self.account_status_widget)

        # 2. Avatar dropdown (fixed width, truncates long names)
        self.avatar_dropdown = QComboBox()
        self.avatar_dropdown.setFixedWidth(140)
        self.avatar_dropdown.setStyleSheet("""
            QComboBox {
                background-color: #2a2a2a;
                color: #D2D2D2;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px 8px;
                font-size: 12px;
            }
            QComboBox:hover {
                border: 1px solid #666;
            }
            QComboBox:disabled {
                background-color: #252525;
                color: #666;
                border: 1px solid #3a3a3a;
            }
            QComboBox::drop-down {
                border: none;
                width: 20px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 5px solid #888;
                margin-right: 5px;
            }
            QComboBox::down-arrow:disabled {
                border-top: 5px solid #555;
            }
            QComboBox QAbstractItemView {
                background-color: #2a2a2a;
                color: #D2D2D2;
                selection-background-color: #555;
                border: 1px solid #555;
            }
        """)
        self._refresh_avatar_dropdown()
        status_bar.addPermanentWidget(self.avatar_dropdown)

        # 3. Enter World button (monochromatic, disabled until server is on)
        self.enter_world_btn = QPushButton("Enter World")
        self.enter_world_btn.setFixedWidth(90)
        self.enter_world_btn.setStyleSheet("""
            QPushButton {
                background-color: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555;
                border-radius: 3px;
                padding: 5px 12px;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #555;
                color: #fff;
            }
            QPushButton:pressed {
                background-color: #3a3a3a;
            }
            QPushButton:disabled {
                background-color: #333;
                color: #555;
            }
        """)
        self.enter_world_btn.clicked.connect(self._on_enter_world_clicked)
        self.enter_world_btn.setEnabled(self.is_server_running())
        status_bar.addPermanentWidget(self.enter_world_btn)

        # 4. Server toggle section
        server_container = QWidget()
        server_container.setFixedWidth(100)
        server_layout = QHBoxLayout()
        server_layout.setContentsMargins(8, 0, 8, 0)
        server_layout.setSpacing(6)

        self.server_status_label = QLabel("Server:")
        self.server_status_label.setStyleSheet("color: #888; font-size: 12px;")
        server_layout.addWidget(self.server_status_label)

        self.server_toggle = ToggleSwitch()
        self.server_toggle.setChecked(self.is_server_running())
        self.server_toggle.toggled.connect(self.on_server_toggled)
        server_layout.addWidget(self.server_toggle)

        server_container.setLayout(server_layout)
        status_bar.addPermanentWidget(server_container)

        # Now update connection status (after server_toggle exists)
        self.update_connection_status()

        # Connect to account changes to refresh avatar dropdown
        AccountManager.instance().avatars_changed.connect(self._refresh_avatar_dropdown)
        AccountManager.instance().logged_in.connect(self._refresh_avatar_dropdown)
        AccountManager.instance().logged_out.connect(self._refresh_avatar_dropdown)

    def _refresh_avatar_dropdown(self):
        """Refresh avatar dropdown with current user's avatars."""
        from .account_manager import AccountManager

        self.avatar_dropdown.clear()

        account = AccountManager.instance()
        if not account.is_logged_in:
            self.avatar_dropdown.addItem("(Sign in first)")
            self.avatar_dropdown.setEnabled(False)
            self.avatar_dropdown.setToolTip("Sign in to select an avatar")
            return

        avatars = account.avatars
        if not avatars:
            self.avatar_dropdown.addItem("(No avatars)")
            self.avatar_dropdown.setEnabled(True)
            self.avatar_dropdown.setToolTip("Use Account > Manage Avatars to create one")
            return

        self.avatar_dropdown.setEnabled(True)

        # Add avatars, marking default with tooltip for full name
        default_idx = 0
        for i, avatar in enumerate(avatars):
            full_name = avatar.display_name or "Unnamed"
            display = full_name
            if avatar.is_default:
                display = f"{display} *"
                default_idx = i
            self.avatar_dropdown.addItem(display, avatar)
            self.avatar_dropdown.setItemData(i, full_name, Qt.ItemDataRole.ToolTipRole)

        self.avatar_dropdown.setCurrentIndex(default_idx)
        self._update_avatar_tooltip()
        self.avatar_dropdown.currentIndexChanged.connect(self._update_avatar_tooltip)

    def _update_avatar_tooltip(self):
        """Update avatar dropdown tooltip to show full name of selected avatar."""
        avatar = self.avatar_dropdown.currentData()
        if avatar and hasattr(avatar, 'display_name'):
            self.avatar_dropdown.setToolTip(avatar.display_name or "Unnamed")

    def _on_enter_world_clicked(self):
        """Handle Enter World button click."""
        from .account_manager import AccountManager
        from ..dialogs.avatar_picker_dialog import AvatarPickerDialog

        # Button should be disabled if server is off, but double-check
        if not self.is_server_running():
            return

        account = AccountManager.instance()

        # Get selected avatar
        avatar = self.avatar_dropdown.currentData()

        if avatar is None:
            if not account.is_logged_in:
                # Not logged in - offer to enter as guest
                dialog = AvatarPickerDialog(self)
                dialog.set_avatars([])
                if dialog.exec():
                    avatar = dialog.get_selected_avatar()
                else:
                    return
            else:
                # Logged in but no avatars - prompt to create
                reply = QMessageBox.question(
                    self,
                    "No Avatar",
                    "You don't have any avatars yet. Create one now?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.Yes:
                    self.manage_avatars()
                return

        # Connect to world
        self._connect_to_world(avatar, account)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
