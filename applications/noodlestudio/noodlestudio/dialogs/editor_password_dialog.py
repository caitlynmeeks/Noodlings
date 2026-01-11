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
#   Editor Password Dialog
#
#   Password prompt for accessing the editor in password-protected builds.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.dialogs.editor_password_dialog
# PURPOSE:  Password verification for protected editor access
# LAYER:    Studio / Dialogs
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   EditorPasswordDialog
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import hashlib
import logging
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)


def hash_password(password: str) -> str:
    """
    Hash a password for storage.

    Uses SHA-256 for simplicity. For production, consider bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hex-encoded hash string
    """
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def verify_password(password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored hash.

    Args:
        password: Plain text password to verify
        stored_hash: Previously stored hash

    Returns:
        True if password matches
    """
    return hash_password(password) == stored_hash


class EditorPasswordDialog(QDialog):
    """
    Password dialog for accessing the editor.

    Shows when user tries to unfold a password-protected build.

    Usage:
        dialog = EditorPasswordDialog(stored_hash, parent=self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Password verified, allow unfold
            self.unfold_panels()
    """

    def __init__(
        self,
        stored_hash: str,
        parent: Optional[QWidget] = None,
        max_attempts: int = 3
    ):
        """
        Initialize the password dialog.

        Args:
            stored_hash: The stored password hash to verify against
            parent: Parent widget
            max_attempts: Maximum failed attempts before lockout
        """
        super().__init__(parent)
        self._stored_hash = stored_hash
        self._max_attempts = max_attempts
        self._attempts = 0

        self.setWindowTitle("Editor Access")
        self.setModal(True)
        self.setFixedSize(350, 180)

        # Remove question mark button on Windows
        self.setWindowFlags(
            self.windowFlags() & ~Qt.WindowType.WindowContextHelpButtonHint
        )

        self._build_ui()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # Header
        header = QLabel("Enter password to access the editor")
        header.setStyleSheet("color: #cccccc; font-size: 13px;")
        layout.addWidget(header)

        # Password field
        self._password_field = QLineEdit()
        self._password_field.setEchoMode(QLineEdit.EchoMode.Password)
        self._password_field.setPlaceholderText("Password")
        self._password_field.setStyleSheet("""
            QLineEdit {
                padding: 8px 12px;
                background: #2a2a2a;
                border: 1px solid #444444;
                border-radius: 4px;
                color: #ffffff;
                font-size: 13px;
            }
            QLineEdit:focus {
                border-color: #666666;
            }
        """)
        self._password_field.returnPressed.connect(self._on_submit)
        layout.addWidget(self._password_field)

        # Error label (hidden by default)
        self._error_label = QLabel("")
        self._error_label.setStyleSheet("color: #ff6666; font-size: 11px;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #cccccc;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        self._submit_btn = QPushButton("Unlock")
        self._submit_btn.setDefault(True)
        self._submit_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 16px;
                background: #4a4a4a;
                border: 1px solid #666666;
                border-radius: 4px;
                color: #ffffff;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #5a5a5a;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
            }
        """)
        self._submit_btn.clicked.connect(self._on_submit)
        button_layout.addWidget(self._submit_btn)

        layout.addLayout(button_layout)

        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
            }
        """)

    def _on_submit(self):
        """Handle password submission."""
        password = self._password_field.text()

        if not password:
            self._show_error("Please enter a password")
            return

        if verify_password(password, self._stored_hash):
            logger.info("Editor password verified")
            self.accept()
        else:
            self._attempts += 1
            remaining = self._max_attempts - self._attempts

            if remaining <= 0:
                logger.warning("Editor password attempts exhausted")
                self._show_error("Too many failed attempts")
                self._password_field.setEnabled(False)
                self._submit_btn.setEnabled(False)
            else:
                self._show_error(f"Incorrect password ({remaining} attempts remaining)")
                self._password_field.clear()
                self._password_field.setFocus()

    def _show_error(self, message: str):
        """Show an error message."""
        self._error_label.setText(message)
        self._error_label.show()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
