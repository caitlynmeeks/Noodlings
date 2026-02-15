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

from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout


class MainWindowStatusBarMixin:
    """Mixin providing status bar setup for MainWindow."""

    def _setup_status_bar(self):
        """Create status bar: [Server connection status ... Server: [toggle]]."""
        from ..widgets.toggle_switch import ToggleSwitch

        status_bar = self.statusBar()

        # Connection status (left side)
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

        # Server toggle (right side)
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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
