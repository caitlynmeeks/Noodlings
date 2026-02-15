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
#   Main Window Account Mixin - Account and authentication operations
#
#   Contains: - show_login_dialog, show_account_info: Login U...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_account_mixin
# PURPOSE:  Main Window Account Mixin
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowAccountMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QMessageBox
from PyQt6.QtCore import QUrl


class MainWindowAccountMixin:
    """Mixin providing account management for MainWindow."""

    def show_login_dialog(self):
        """Show the OAuth login dialog."""
        try:
            from ..dialogs.login_dialog import LoginDialog
            from .account_manager import AccountManager

            if AccountManager.instance().is_logged_in:
                self.show_account_info()
                return

            dialog = LoginDialog(self)
            dialog.login_successful.connect(self._on_login_successful)
            dialog.exec()
        except Exception as e:
            import traceback
            print(f"[MainWindow] Error in show_login_dialog: {e}")
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to show login dialog: {e}")

    def show_account_info(self):
        """Show account information dialog."""
        try:
            from ..dialogs.login_dialog import AccountInfoDialog
            from .account_manager import AccountManager

            if not AccountManager.instance().is_logged_in:
                self.show_login_dialog()
                return

            dialog = AccountInfoDialog(self)
            dialog.exec()
        except Exception as e:
            import traceback
            print(f"[MainWindow] Error in show_account_info: {e}")
            traceback.print_exc()
            QMessageBox.warning(self, "Error", f"Failed to show account info: {e}")

    def show_cloud_noodlings(self):
        """Show cloud noodlings browser."""
        from .account_manager import AccountManager

        if not AccountManager.instance().is_logged_in:
            QMessageBox.information(
                self, "Sign In Required",
                "Please sign in to view your cloud Noodlings."
            )
            self.show_login_dialog()
            return

        QMessageBox.information(
            self, "Coming Soon",
            "Cloud Noodlings browser coming soon!"
        )

    def show_buy_credits(self):
        """Open credits purchase page."""
        import webbrowser
        from .account_manager import AccountManager

        if not AccountManager.instance().is_logged_in:
            QMessageBox.information(
                self, "Sign In Required",
                "Please sign in to purchase credits."
            )
            self.show_login_dialog()
            return

        webbrowser.open("https://noodlings.ai/credits")

    def sign_out(self):
        """Sign out of cloud account."""
        from .account_manager import AccountManager

        if not AccountManager.instance().is_logged_in:
            return

        reply = QMessageBox.question(
            self, "Sign Out",
            "Are you sure you want to sign out?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            AccountManager.instance().logout()
            self.statusBar().showMessage("Signed out", 3000)

    def enter_world(self):
        """Enter noodleMUSH world with avatar selection."""
        from .account_manager import AccountManager
        from ..dialogs.avatar_picker_dialog import AvatarPickerDialog

        account = AccountManager.instance()

        if not self.is_server_running():
            QMessageBox.warning(
                self, "Server Offline",
                "noodleMUSH server is not running.\n\n"
                "Turn on the server using the toggle in the status bar, "
                "then try again."
            )
            return

        dialog = AvatarPickerDialog(self)

        if account.is_logged_in:
            dialog.set_avatars(account.avatars)
        else:
            dialog.set_avatars([])

        if dialog.exec():
            avatar = dialog.get_selected_avatar()
            if avatar:
                self._connect_to_world(avatar, account)

    def _connect_to_world(self, avatar, account):
        """Connect to noodleMUSH with selected avatar via URL parameters."""
        import json
        from urllib.parse import quote
        from PyQt6.QtCore import QSettings, QTimer

        if not hasattr(self, 'web_view') or not self.web_view:
            QMessageBox.warning(self, "Error", "Text View not available")
            return

        avatar_data = avatar.to_dict() if hasattr(avatar, 'to_dict') else {
            'display_name': avatar.display_name
        }

        base_url = "http://localhost:8080"
        token = account.session_token
        avatar_json = quote(json.dumps(avatar_data))

        auth_url = f"{base_url}?token={token}&avatar={avatar_json}"

        self.statusBar().showMessage(f"Entering world as {avatar.display_name}...", 0)

        # Switch to Text View tab
        settings = QSettings("Noodlings", "NoodleStudio")
        auto_show_chat = settings.value("auto_show_chat", True, type=bool)

        if auto_show_chat and hasattr(self, 'center_tabs'):
            self.center_tabs.setCurrentIndex(0)

        self.web_view.setUrl(QUrl(auth_url))

        QTimer.singleShot(2000, lambda: self.statusBar().showMessage(
            f"Entered world as {avatar.display_name}", 5000
        ))

    def _on_connect_finished(self, success: bool):
        """Handle connection result."""
        if hasattr(self, 'enter_world_btn') and self.enter_world_btn:
            self.enter_world_btn.setEnabled(True)
            self.enter_world_btn.setText("Enter World")

        avatar = getattr(self, '_pending_avatar', None)
        avatar_name = avatar.display_name if avatar and hasattr(avatar, 'display_name') else "avatar"

        if success:
            self.statusBar().showMessage(f"Entered world as {avatar_name}", 5000)

            from PyQt6.QtCore import QSettings
            settings = QSettings("Noodlings", "NoodleStudio")
            auto_show_chat = settings.value("auto_show_chat", True, type=bool)

            if auto_show_chat and hasattr(self, 'center_tabs'):
                self.center_tabs.setCurrentIndex(0)

            if hasattr(self, 'web_view') and self.web_view:
                self.web_view.reload()
        else:
            self.statusBar().showMessage("Connection failed", 3000)
            QMessageBox.warning(
                self, "Connection Failed",
                "Failed to connect to noodleMUSH.\n\n"
                "Check that the server is running and try again."
            )

    def manage_avatars(self):
        """Open avatar management dialog."""
        from .account_manager import AccountManager, AvatarMetadata, AvatarAssetLocation
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem,
            QPushButton, QInputDialog
        )
        from PyQt6.QtCore import Qt

        account = AccountManager.instance()

        if not account.is_logged_in:
            QMessageBox.information(
                self, "Sign In Required",
                "Please sign in to manage your avatars."
            )
            self.show_login_dialog()
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Avatars")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        list_layout = QHBoxLayout()

        avatar_list = QListWidget()
        for av in account.avatars:
            item = QListWidgetItem(av.display_name or "Unnamed")
            if av.is_default:
                item.setText(f"{av.display_name} (default)")
            item.setData(Qt.ItemDataRole.UserRole, av)
            avatar_list.addItem(item)
        list_layout.addWidget(avatar_list)

        btn_layout = QVBoxLayout()

        add_btn = QPushButton("New Avatar")
        def on_add():
            name, ok = QInputDialog.getText(dialog, "New Avatar", "Display name:")
            if ok and name:
                new_avatar = AvatarMetadata(
                    display_name=name,
                    description="",
                    asset_location=AvatarAssetLocation.LOCAL,
                    asset_ref=""
                )
                account.add_avatar(new_avatar)
                item = QListWidgetItem(name)
                item.setData(Qt.ItemDataRole.UserRole, new_avatar)
                avatar_list.addItem(item)
        add_btn.clicked.connect(on_add)
        btn_layout.addWidget(add_btn)

        edit_btn = QPushButton("Edit...")
        def on_edit():
            current = avatar_list.currentItem()
            if not current:
                return
            av = current.data(Qt.ItemDataRole.UserRole)
            name, ok = QInputDialog.getText(
                dialog, "Edit Avatar", "Display name:", text=av.display_name
            )
            if ok and name:
                av.display_name = name
                account.update_avatar(av)
                current.setText(f"{name} (default)" if av.is_default else name)
        edit_btn.clicked.connect(on_edit)
        btn_layout.addWidget(edit_btn)

        default_btn = QPushButton("Set Default")
        def on_default():
            current = avatar_list.currentItem()
            if not current:
                return
            av = current.data(Qt.ItemDataRole.UserRole)
            account.set_default_avatar(av.id)
            for i in range(avatar_list.count()):
                item = avatar_list.item(i)
                item_av = item.data(Qt.ItemDataRole.UserRole)
                if item_av.id == av.id:
                    item.setText(f"{item_av.display_name} (default)")
                else:
                    item.setText(item_av.display_name)
        default_btn.clicked.connect(on_default)
        btn_layout.addWidget(default_btn)

        delete_btn = QPushButton("Delete")
        def on_delete():
            current = avatar_list.currentItem()
            if not current:
                return
            av = current.data(Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                dialog, "Delete Avatar",
                f"Delete avatar '{av.display_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                account.delete_avatar(av.id)
                avatar_list.takeItem(avatar_list.row(current))
        delete_btn.clicked.connect(on_delete)
        btn_layout.addWidget(delete_btn)

        btn_layout.addStretch()
        list_layout.addLayout(btn_layout)

        layout.addLayout(list_layout)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.accept)
        layout.addWidget(close_btn)

        dialog.setStyleSheet("""
            QDialog { background-color: #2D2D2D; }
            QListWidget {
                background-color: #2A2A2A; color: #D2D2D2; border: 1px solid #555;
            }
            QListWidget::item { padding: 8px; }
            QListWidget::item:selected { background-color: #4A7CBA; }
            QPushButton {
                background-color: #3A3A3A; color: #D2D2D2;
                border: 1px solid #555; padding: 8px; min-width: 80px;
            }
            QPushButton:hover { background-color: #4A4A4A; }
        """)

        dialog.exec()

    def _on_login_successful(self):
        """Handle successful login."""
        from .account_manager import AccountManager
        user = AccountManager.instance().user
        if user:
            self.statusBar().showMessage(f"Signed in as {user.email}", 5000)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
