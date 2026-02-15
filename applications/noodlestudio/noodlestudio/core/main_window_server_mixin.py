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
#   Main Window Server Mixin - Server management and status
#
#   Contains: - is_server_running: Check if noodleMUSH server...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_server_mixin
# PURPOSE:  Main Window Server Mixin
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowServerMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class MainWindowServerMixin:
    """Mixin providing server management for MainWindow."""

    @staticmethod
    def _cmush_dir() -> str:
        """Return absolute path to the cmush application directory."""
        return os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', '..', '..', 'cmush'
        ))

    def is_server_running(self) -> bool:
        """Check if noodleMUSH server is running."""
        result = subprocess.run(
            ['pgrep', '-f', 'python.*cmush.*server.py'], capture_output=True
        )
        return result.returncode == 0

    def on_server_toggled(self, enabled: bool):
        """Handle server toggle switch."""
        # Disable Enter World while server state is changing
        self.enter_world_btn.setEnabled(False)

        if enabled:
            cmush_dir = self._cmush_dir()
            start_script = os.path.join(cmush_dir, 'start.sh')

            if not os.path.exists(start_script):
                logger.error(f"start.sh not found at {start_script}")
                self.connection_label.setText("Server script not found")
                self.enter_world_btn.setEnabled(False)
                return

            # Build environment with project path if one is open
            env = os.environ.copy()
            if self.project_manager.is_project_open():
                env["PROJECT_PATH"] = self.project_manager.current_project_path
                self.connection_label.setText(
                    f"Starting server for {self.project_manager.current_project_name}..."
                )
            else:
                self.connection_label.setText("Starting server (legacy mode)...")

            # Redirect output to log file for diagnostics
            log_dir = os.path.join(cmush_dir, 'logs')
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, 'startup.log')
            log_file = open(log_path, 'w')

            subprocess.Popen(
                [start_script],
                cwd=cmush_dir,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT
            )
            logger.info(f"Server launch output: {log_path}")
        else:
            # Stop server — use specific pattern to avoid killing unrelated processes
            subprocess.run(['pkill', '-f', 'python.*cmush.*server.py'])
            self.connection_label.setText("Server stopped")

        # Update status after a delay (5 seconds for server startup)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(5000, self.update_connection_status)

    def update_connection_status(self):
        """Update connection status label and UI state."""
        running = self.is_server_running()

        if running:
            if hasattr(self, 'connection_label'):
                self.connection_label.setText("Server running on :8765")
                self.connection_label.setStyleSheet("""
                    QLabel {
                        background-color: #282828;
                        color: #76AF6A;
                        border-radius: 4px;
                        padding: 3px 8px;
                        font-size: 12px;
                    }
                """)
            if hasattr(self, 'server_toggle'):
                self.server_toggle.blockSignals(True)
                self.server_toggle.setChecked(True)
                self.server_toggle.blockSignals(False)
        else:
            if hasattr(self, 'connection_label'):
                self.connection_label.setText("Server offline")
                self.connection_label.setStyleSheet("""
                    QLabel {
                        background-color: #282828;
                        color: #666;
                        border-radius: 4px;
                        padding: 3px 8px;
                        font-size: 12px;
                    }
                """)
            if hasattr(self, 'server_toggle'):
                self.server_toggle.blockSignals(True)
                self.server_toggle.setChecked(False)
                self.server_toggle.blockSignals(False)

        # Update Enter World button state
        if hasattr(self, 'enter_world_btn'):
            self.enter_world_btn.setEnabled(running)

        # Load MUSH web client when server comes up
        if hasattr(self, 'web_view') and self.web_view:
            if running:
                from PyQt6.QtCore import QUrl
                current_url = self.web_view.url().toString()
                if 'localhost:8080' not in current_url:
                    self.web_view.setUrl(QUrl("http://localhost:8080"))

        # Update World View
        if hasattr(self, 'world_view'):
            self.world_view.set_server_state(running)

        # Update Hierarchy (gray out if offline)
        if hasattr(self, 'hierarchy'):
            self.hierarchy.set_server_state(running)

        # Update Console (reconnect if server just started)
        if hasattr(self, 'console'):
            if running and not self.console.connected:
                self.console.reconnect()

        # Update server-dependent menu actions
        if hasattr(self, '_server_dependent_actions'):
            for action in self._server_dependent_actions:
                action.setEnabled(running)

    def _check_autostart_mush(self):
        """Check if MUSH server should auto-start based on settings."""
        from PyQt6.QtCore import QSettings
        settings = QSettings("Noodlings", "NoodleStudio")
        autostart = settings.value("autostart_mush", False, type=bool)

        if autostart and not self.is_server_running():
            self.server_toggle.setChecked(True)

    def _start_activity_bridge(self):
        """Start the cmush activity bridge for real-time LLM activity visualization."""
        try:
            from .model_activity_tracker import start_activity_bridge
            self._activity_bridge = start_activity_bridge()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to start activity bridge: {e}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
