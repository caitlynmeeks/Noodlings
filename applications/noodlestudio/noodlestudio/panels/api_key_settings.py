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
#   API Key Settings Widget
#
#   Settings panel for managing NoodleROUTER API keys.
#   Zero friction for users - key auto-generates and auto-configures.
#   They only see it if they go looking in Settings.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.api_key_settings
# PURPOSE:  API Key Settings Widget
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   APIKeySettingsWidget
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import json
import logging
import os
import subprocess
import urllib.request
import urllib.error
from typing import Optional

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QGroupBox, QFrame, QProgressBar,
    QApplication
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = os.environ.get('NOODLINGS_API_URL', 'https://noodlings-api.caitsters.workers.dev')

# Keychain identifiers
KEYCHAIN_ACCOUNT = 'NoodleStudio-APIKey'
KEYCHAIN_SERVICE = 'noodlings.ai-apikey'


class APIKeySettingsWidget(QWidget):
    """
    API Key management panel in Settings.

    Features:
    - Display key (monospace, selectable)
    - Copy to clipboard with feedback
    - Regenerate key flow with confirmation dialog
    - Secure storage in OS keychain
    - Error states (no internet, invalid key)
    - Usage display (when available)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._api_key: Optional[str] = None
        self._usage_data: Optional[dict] = None
        self._build_ui()
        self._load_key()

    def _build_ui(self):
        """Build the API key settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("API Key")
        title.setStyleSheet("color: #D2D2D2; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Explanation
        explanation = QLabel(
            "This key lets your noodlings talk to AI models through "
            "NoodleROUTER. It's already configured - you only need "
            "this if you're using NoodleStudio on another device."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        layout.addWidget(explanation)

        # Key display group
        key_group = QGroupBox("Your API Key")
        key_group.setStyleSheet("""
            QGroupBox {
                background: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                color: #D2D2D2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
            }
        """)
        key_layout = QVBoxLayout()

        # Key display field
        self._key_display = QLineEdit()
        self._key_display.setReadOnly(True)
        self._key_display.setFont(QFont("SF Mono, Menlo, Monaco, Consolas", 11))
        self._key_display.setPlaceholderText("Loading...")
        self._key_display.setStyleSheet("""
            QLineEdit {
                background: #3e3e3e;
                color: #76AF6A;
                border: 1px solid #555555;
                padding: 10px;
                border-radius: 3px;
                selection-background-color: #555555;
            }
        """)
        # Select all on click
        self._key_display.mousePressEvent = self._select_all_on_click
        key_layout.addWidget(self._key_display)

        # Copy button
        copy_layout = QHBoxLayout()
        self._copy_btn = QPushButton("Copy")
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #555555;
            }
            QPushButton:disabled {
                background: #3a3a3a;
                color: #666666;
            }
        """)
        self._copy_btn.clicked.connect(self._copy_key)
        self._copy_btn.setEnabled(False)

        # Timer for resetting copy button feedback (parented to self so it
        # auto-destroys when the widget is deleted -- prevents accessing a
        # deleted QPushButton if the timer fires during teardown).
        self._copy_reset_timer = QTimer(self)
        self._copy_reset_timer.setSingleShot(True)
        self._copy_reset_timer.timeout.connect(self._reset_copy_button)
        copy_layout.addWidget(self._copy_btn)
        copy_layout.addStretch()
        key_layout.addLayout(copy_layout)

        key_group.setLayout(key_layout)
        layout.addWidget(key_group)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background: #3e3e3e;")
        layout.addWidget(separator)

        # Regenerate section
        regen_group = QGroupBox("Security")
        regen_group.setStyleSheet("""
            QGroupBox {
                background: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                color: #D2D2D2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
            }
        """)
        regen_layout = QVBoxLayout()

        regen_explanation = QLabel(
            "If you think your key was exposed, generate a new one. "
            "Your old key will stop working immediately."
        )
        regen_explanation.setWordWrap(True)
        regen_explanation.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        regen_layout.addWidget(regen_explanation)

        self._regen_btn = QPushButton("Regenerate Key")
        self._regen_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #555555;
            }
            QPushButton:disabled {
                background: #3a3a3a;
                color: #666666;
            }
        """)
        self._regen_btn.clicked.connect(self._regenerate_key)
        self._regen_btn.setEnabled(False)
        regen_layout.addWidget(self._regen_btn)

        regen_group.setLayout(regen_layout)
        layout.addWidget(regen_group)

        # Usage section (optional)
        self._usage_group = QGroupBox("This Month's Usage")
        self._usage_group.setStyleSheet("""
            QGroupBox {
                background: #2e2e2e;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                color: #D2D2D2;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
            }
        """)
        usage_layout = QVBoxLayout()

        self._usage_label = QLabel("Requests: --")
        self._usage_label.setStyleSheet("color: #D2D2D2;")
        usage_layout.addWidget(self._usage_label)

        self._usage_bar = QProgressBar()
        self._usage_bar.setStyleSheet("""
            QProgressBar {
                background: #3e3e3e;
                border: none;
                border-radius: 3px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #76AF6A;
                border-radius: 3px;
            }
        """)
        self._usage_bar.setTextVisible(False)
        self._usage_bar.setValue(0)
        usage_layout.addWidget(self._usage_bar)

        self._usage_group.setLayout(usage_layout)
        self._usage_group.hide()  # Hidden until we have usage data
        layout.addWidget(self._usage_group)

        # Error display
        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #E57373; font-size: 11px;")
        self._error_label.hide()
        layout.addWidget(self._error_label)

        layout.addStretch()

    def _select_all_on_click(self, event):
        """Select all text when clicking the key field."""
        self._key_display.selectAll()
        QLineEdit.mousePressEvent(self._key_display, event)

    def _load_key(self):
        """Load API key from keychain or environment."""
        # First try keychain
        key = self._load_from_keychain()

        # Then try environment
        if not key:
            key = os.environ.get('NOODLEROUTER_API_KEY', '')

        if key:
            self._display_key(key)
        else:
            # Show "not configured" state
            self._key_display.setText("")
            self._key_display.setPlaceholderText("No API key configured")
            self._show_error("No API key found. Log in to your Noodlings account to get one.")

        # Enable buttons regardless (user might want to configure)
        self._copy_btn.setEnabled(bool(key))
        self._regen_btn.setEnabled(bool(key))

    def _display_key(self, key: str):
        """Display the key in the UI."""
        self._api_key = key
        self._key_display.setText(key)
        self._key_display.setPlaceholderText("")
        self._copy_btn.setEnabled(True)
        self._regen_btn.setEnabled(True)
        self._error_label.hide()

        # Try to load usage data
        self._load_usage()

    def _show_error(self, message: str):
        """Display an error message."""
        self._error_label.setText(message)
        self._error_label.show()

    def _copy_key(self):
        """Copy key to clipboard with feedback."""
        if self._api_key:
            clipboard = QApplication.clipboard()
            clipboard.setText(self._api_key)

            # Visual feedback
            self._copy_btn.setText("Copied!")
            self._copy_btn.setStyleSheet("""
                QPushButton {
                    background: #76AF6A;
                    color: #1e1e1e;
                    border: 1px solid #76AF6A;
                    padding: 8px 16px;
                    border-radius: 3px;
                }
            """)

            # Reset after 2 seconds
            self._copy_reset_timer.start(2000)

    def _reset_copy_button(self):
        """Reset copy button to original state."""
        self._copy_btn.setText("Copy")
        self._copy_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #555555;
            }
        """)

    def _regenerate_key(self):
        """Regenerate API key with confirmation."""
        reply = QMessageBox.question(
            self,
            "Regenerate API Key?",
            "This will:\n\n"
            "  - Create a new key\n"
            "  - Immediately disable your old key\n"
            "  - Update NoodleStudio on this device automatically\n\n"
            "Any other devices or apps using your old key will need "
            "to be updated with the new one.\n\n"
            "Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel
        )

        if reply == QMessageBox.StandardButton.Yes:
            self._do_regenerate()

    def _do_regenerate(self):
        """Actually regenerate the key."""
        # This would call the backend API to regenerate
        # For now, show a message that this requires being logged in
        QMessageBox.information(
            self,
            "Regenerate Key",
            "To regenerate your API key, please:\n\n"
            "1. Log in to your account at noodlings.ai\n"
            "2. Go to Account Settings\n"
            "3. Click 'Regenerate API Key'\n\n"
            "Your new key will be automatically synced."
        )

    def _load_usage(self):
        """Load usage statistics (if available)."""
        # This would fetch from the API
        # For now, hide the usage section
        self._usage_group.hide()

    # -------------------------------------------------------------------------
    # Keychain Integration
    # -------------------------------------------------------------------------

    def _save_to_keychain(self, key: str) -> bool:
        """Save API key to macOS keychain."""
        try:
            result = subprocess.run([
                'security', 'add-generic-password',
                '-a', KEYCHAIN_ACCOUNT,
                '-s', KEYCHAIN_SERVICE,
                '-w', key,
                '-U'  # Update if exists
            ], capture_output=True)
            return result.returncode == 0
        except Exception as e:
            logger.warning(f"Failed to save to keychain: {e}")
            return False

    def _load_from_keychain(self) -> Optional[str]:
        """Load API key from macOS keychain."""
        try:
            result = subprocess.run([
                'security', 'find-generic-password',
                '-a', KEYCHAIN_ACCOUNT,
                '-s', KEYCHAIN_SERVICE,
                '-w'
            ], capture_output=True, text=True)
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception as e:
            logger.debug(f"No API key in keychain: {e}")
        return None

    def _delete_from_keychain(self):
        """Delete API key from macOS keychain."""
        try:
            subprocess.run([
                'security', 'delete-generic-password',
                '-a', KEYCHAIN_ACCOUNT,
                '-s', KEYCHAIN_SERVICE
            ], capture_output=True)
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def set_api_key(self, key: str):
        """
        Set API key (called after user gets key from backend).

        Args:
            key: The API key to store
        """
        # Save to keychain
        if self._save_to_keychain(key):
            logger.info("API key saved to keychain")
        else:
            logger.warning("Failed to save API key to keychain")

        # Display
        self._display_key(key)

    def get_api_key(self) -> Optional[str]:
        """Get the current API key."""
        return self._api_key


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
