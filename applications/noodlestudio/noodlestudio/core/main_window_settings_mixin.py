"""
Main Window Settings Mixin - Settings and configuration operations

Contains:
- show_mcp_settings: MCP server configuration
- show_rng_settings: Random number generator configuration
- RNG detection and persistence
- Documentation/help methods
- About dialog

Author: Noodlings Project
Date: December 2025
"""

import json
from pathlib import Path

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
)


class MainWindowSettingsMixin:
    """Mixin providing settings management for MainWindow."""

    def show_mcp_settings(self):
        """Show MCP (Model Context Protocol) servers configuration dialog."""
        from ..panels.mcp_settings_panel import MCPSettingsPanel

        dialog = QDialog(self)
        dialog.setWindowTitle("MCP Server Configuration")
        dialog.resize(700, 500)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(0, 0, 0, 0)

        mcp_panel = MCPSettingsPanel()
        layout.addWidget(mcp_panel)

        dialog.exec()

    def show_rng_settings(self):
        """Show Random Number Generator settings dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Random Number Generator Settings")
        dialog.resize(400, 150)

        layout = QVBoxLayout(dialog)

        header = QLabel("Select Random Number Generator:")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        rng_combo = QComboBox()
        rng_combo.addItem("Internal RNG (Software)")

        ubild_available = self._check_ubild_connected()
        if ubild_available:
            rng_combo.addItem("TrueRNG (USB Hardware RNG)")

        current_rng = self._load_rng_setting()
        if current_rng == "truerng" and ubild_available:
            rng_combo.setCurrentIndex(1)
        else:
            rng_combo.setCurrentIndex(0)

        layout.addWidget(rng_combo)

        if ubild_available:
            status_label = QLabel("Hardware RNG detected")
            status_label.setStyleSheet("color: #76AF6A;")
        else:
            status_label = QLabel(
                "No RNG detected. Falling back to internal RNG.\n"
                "Outputs are deterministic. Consider an avalanche effect RNG\n"
                "for quantum non-determinism."
            )
            status_label.setStyleSheet("color: #999;")
            status_label.setWordWrap(True)
        layout.addWidget(status_label)

        layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(
            lambda: self._save_rng_setting(rng_combo.currentText(), dialog)
        )
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)
        dialog.exec()

    def _check_ubild_connected(self):
        """Check if TrueRNG/ubild USB hardware RNG is connected."""
        try:
            import subprocess
            import glob

            result = subprocess.run(
                ['system_profiler', 'SPUSBDataType'],
                capture_output=True, text=True, timeout=3
            )
            stdout_lower = result.stdout.lower()
            if 'truerng' in stdout_lower or 'ubild' in stdout_lower or 'hardware rng' in stdout_lower:
                return True

            usb_devices = glob.glob('/dev/cu.usbmodem*')
            if usb_devices:
                return True

            return False
        except Exception as e:
            print(f"RNG detection error: {e}")
            return False

    def _load_rng_setting(self):
        """Load RNG setting from config."""
        config_file = Path.home() / ".noodlestudio" / "settings.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('rng_source', 'internal')
            except (FileNotFoundError, json.JSONDecodeError):
                pass
        return 'internal'

    def _save_rng_setting(self, rng_text, dialog):
        """Save RNG setting to config."""
        rng_source = 'truerng' if 'TrueRNG' in rng_text else 'internal'

        config_dir = Path.home() / ".noodlestudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "settings.json"

        settings = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                pass

        settings['rng_source'] = rng_source

        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=2)

        if rng_source == 'truerng':
            message = "Hardware RNG detected and activated - True quantum randomness enabled"
        else:
            message = "Using internal RNG - Deterministic pseudorandom output"

        self.statusBar().showMessage(message, 5000)
        dialog.accept()

    def show_startup_rng_status(self):
        """Show RNG status message on startup."""
        ubild_available = self._check_ubild_connected()
        current_rng = self._load_rng_setting()

        if ubild_available and current_rng == 'truerng':
            message = "Hardware RNG detected - True quantum randomness enabled"
        else:
            message = (
                "No RNG detected. Falling back to internal RNG. "
                "Outputs are deterministic. Consider an avalanche effect RNG "
                "for quantum non-determinism"
            )

        self.statusBar().showMessage(message, 8000)

    def _open_settings_tab(self):
        """Open the Settings tab (Cmd+, shortcut)."""
        for i in range(self.center_tabs.count()):
            if self.center_tabs.tabText(i) == "Settings":
                self.center_tabs.setCurrentIndex(i)
                break

    def open_scripting_api(self):
        """Open Scripting API documentation in browser."""
        import subprocess
        import webbrowser
        import os

        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))
        )))
        mkdocs_path = os.path.join(repo_root, "..", "..", "mkdocs.yml")

        if os.path.exists(mkdocs_path):
            try:
                subprocess.Popen(
                    ["mkdocs", "serve", "--dev-addr=127.0.0.1:8000"],
                    cwd=os.path.dirname(mkdocs_path),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except (FileNotFoundError, subprocess.SubprocessError):
                pass

        webbrowser.open("http://127.0.0.1:8000/api/overview/")

    def open_documentation(self):
        """Open main documentation in browser."""
        import webbrowser
        webbrowser.open("https://noodlings.ai/")

    def show_about(self):
        """Show About dialog."""
        from .. import __version__
        about_text = (
            f"NoodleSTUDIO v{__version__}\n\n"
            "Symbiosis of Tendrils:\n"
            "Unfurling, Developing,\n"
            "Interconnected Organisms\n\n"
            "Visual IDE for Cognitive Architecture Design\n"
            "Built with PyQt6\n\n"
            "noodlings.ai"
        )
        QMessageBox.about(self, "About NoodleSTUDIO", about_text)

    def show_bug_report_dialog(self):
        """Show the bug report dialog."""
        from ..dialogs.bug_report_dialog import BugReportDialog
        dialog = BugReportDialog(self)
        dialog.exec()

    def open_github_issues(self):
        """Open GitHub issues page in browser."""
        import webbrowser
        webbrowser.open("https://github.com/noodlings-ai/noodlings/issues")

    def open_documentation(self):
        """Open documentation in browser."""
        import webbrowser
        webbrowser.open("https://docs.noodlings.ai")

    def report_issue(self):
        """Legacy: Redirects to bug report dialog."""
        self.show_bug_report_dialog()

    def show_credits(self):
        """Show demo scene style credits with music."""
        from ..panels.credits_panel import show_credits
        self.credits_window = show_credits(self)

    # ========== MISC EVENT HANDLERS ==========

    def keyPressEvent(self, event):
        """Forward key events to Konami detector."""
        self.konami_detector.key_pressed(event.key())
        super().keyPressEvent(event)

    def closeEvent(self, event):
        """Auto-save layout on shutdown."""
        try:
            self.layout_manager.save_layout(self, "Default")
            self.layout_manager.set_last_used_layout("Default")
            print("[MainWindow] Auto-saved layout on shutdown")
        except Exception as e:
            print(f"[MainWindow] Error auto-saving layout: {e}")
        super().closeEvent(event)

    def _summon_goose(self):
        """Summon the legendary goose to walk across the screen."""
        from PyQt6.QtCore import QSettings

        settings = QSettings("Noodlings", "NoodleStudio")
        degoose_code = settings.value("degoosification_code", "")

        if degoose_code:
            self.statusBar().showMessage("Goose has been degoosified. Nice try!", 3000)
            return

        if self.goose_active:
            return

        from ..widgets.goose_widget import GooseWidget
        self.goose_active = True

        goose = GooseWidget(self)
        goose.show()
        goose.destroyed.connect(lambda: setattr(self, 'goose_active', False))

    def reload_world_view(self):
        """Reload World View with autologin (Ctrl+R)."""
        if hasattr(self.world_view, 'reload'):
            self.world_view.reload()
            self.statusBar().showMessage("Reloaded (autologin)", 2000)

    def reload_world_view_clean(self):
        """Reload World View to login screen (Ctrl+Shift+R)."""
        if hasattr(self.world_view, 'web_view'):
            from PyQt6.QtWebEngineCore import QWebEngineProfile
            from PyQt6.QtCore import QUrl

            profile = QWebEngineProfile.defaultProfile()
            profile.cookieStore().deleteAllCookies()
            self.world_view.web_view.setUrl(QUrl("http://localhost:8080"))
            self.statusBar().showMessage("Reloaded (login screen)", 2000)

    def toggle_world_view_maximize(self):
        """Toggle World View between maximized and normal (Ctrl+M)."""
        if hasattr(self, 'world_view'):
            self.world_view.toggle_maximize()
