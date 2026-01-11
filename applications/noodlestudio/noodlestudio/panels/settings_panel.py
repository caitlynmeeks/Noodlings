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
#   Settings Panel - Unified settings interface for NoodleSTUDIO.
#
#   VSCode-style unified settings with tabs: - General: Start...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.settings_panel
# PURPOSE:  settings panel panel UI
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   GeneralSettingsWidget, ExternalAppsWidget, SettingsPanel
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTabWidget,
    QGroupBox, QLabel, QLineEdit, QPushButton, QFileDialog,
    QCheckBox, QInputDialog, QMessageBox, QProgressDialog
)
from PyQt6.QtCore import Qt, QSettings, QUrl, QThread, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QFont
from pathlib import Path
import json
import requests

from .model_manager_panel_v2 import ModelManagerPanel
from .api_key_settings import APIKeySettingsWidget


class GeneralSettingsWidget(QWidget):
    """General application settings (startup, degoosification, etc.)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("Noodlings", "NoodleStudio")
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the general settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("General Settings")
        title.setStyleSheet("color: #D2D2D2; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Startup Options Group
        startup_group = QGroupBox("Startup")
        startup_group.setStyleSheet("""
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
        startup_layout = QVBoxLayout()

        # Auto-start MUSH server
        self.autostart_mush = QCheckBox("Start MUSH server automatically")
        self.autostart_mush.setStyleSheet("color: #D2D2D2;")
        self.autostart_mush.stateChanged.connect(self._save_settings)
        startup_layout.addWidget(self.autostart_mush)

        # Auto-login as last account
        self.auto_login = QCheckBox("Log in automatically as last account")
        self.auto_login.setStyleSheet("color: #D2D2D2;")
        self.auto_login.setToolTip("Automatically sign in with your last used account on startup")
        self.auto_login.stateChanged.connect(self._save_settings)
        startup_layout.addWidget(self.auto_login)

        # Launch on system startup
        self.launch_on_startup = QCheckBox("Launch NoodleSTUDIO on system startup")
        self.launch_on_startup.setStyleSheet("color: #D2D2D2;")
        self.launch_on_startup.stateChanged.connect(self._save_settings)
        startup_layout.addWidget(self.launch_on_startup)

        startup_group.setLayout(startup_layout)
        layout.addWidget(startup_group)

        # World Entry Options Group
        world_group = QGroupBox("World Entry")
        world_group.setStyleSheet("""
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
        world_layout = QVBoxLayout()

        # Auto-show Text View on world entry
        self.auto_show_chat = QCheckBox("Show Text View when entering world")
        self.auto_show_chat.setStyleSheet("color: #D2D2D2;")
        self.auto_show_chat.setToolTip("Automatically switch to the Text View tab after entering the world")
        self.auto_show_chat.stateChanged.connect(self._save_settings)
        world_layout.addWidget(self.auto_show_chat)

        world_group.setLayout(world_layout)
        layout.addWidget(world_group)

        # Degoosification Group
        goose_group = QGroupBox("Gooseware")
        goose_group.setStyleSheet("""
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
        goose_layout = QVBoxLayout()

        # Info label
        info_label = QLabel("Register your email to receive a degoosification code:")
        info_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        info_label.setWordWrap(True)
        goose_layout.addWidget(info_label)

        # Email input
        email_input_layout = QHBoxLayout()
        email_label = QLabel("Email:")
        email_label.setStyleSheet("color: #D2D2D2;")
        email_label.setFixedWidth(50)
        email_input_layout.addWidget(email_label)

        self.degoose_email_field = QLineEdit()
        self.degoose_email_field.setPlaceholderText("your.email@example.com")
        self.degoose_email_field.setStyleSheet("""
            QLineEdit {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 3px;
            }
        """)
        email_input_layout.addWidget(self.degoose_email_field)
        goose_layout.addLayout(email_input_layout)

        # Register button
        register_btn = QPushButton("Register & Turn off goose")
        register_btn.setStyleSheet("""
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
        register_btn.clicked.connect(self._register_for_degoosification)
        goose_layout.addWidget(register_btn)

        # Separator
        separator_label = QLabel("─────────── or ───────────")
        separator_label.setStyleSheet("color: #555555; font-size: 10px;")
        separator_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        goose_layout.addWidget(separator_label)

        # Manual code entry button
        manual_btn = QPushButton("I already have a code")
        manual_btn.setStyleSheet("""
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
        manual_btn.clicked.connect(self._degoosify)
        goose_layout.addWidget(manual_btn)

        # Bypass hint (subtle, for curious folk)
        hint_label = QLabel("(Bypass codes are in the source code for curious tinkerers)")
        hint_label.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
        hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        goose_layout.addWidget(hint_label)

        goose_group.setLayout(goose_layout)
        layout.addWidget(goose_group)

        layout.addStretch()

    def _validate_degoosification_code(self, code: str) -> bool:
        """
        Validate degoosification code using QUANTUM ALGORITHMIC ENCRYPTION™

        ⚠️  SECURITY THEATER WARNING ⚠️
        This is intentionally trivial to circumvent. We're open source!
        If you're reading this, you've already won. The goose respects your curiosity.

        The "validation" is just a gentle nudge to support the project.
        Real codes come from noodlings.ai backend (Phase 2 - email collection).

        FOR NOW - Temporary bypass codes (until backend is live):
        1. ROT13 of "HONK"
        2. Contains "esoog" (goose spelled backwards - clever!)
        3. Literally just "DEGOOSIFY" (we appreciate honesty)
        4. Any valid email address (shows you're human)
        5. Any string ≥16 characters (they tried!)
        """
        import re
        import base64

        code = code.strip()

        # ═══════════════════════════════════════════════════════════════
        # PHASE 2: QUANTUM ALGORITHMIC ENCRYPTION - UNBREAKABLE™
        # (Spoiler: It's XOR with a base64'd key. Maximum security theater!)
        # ═══════════════════════════════════════════════════════════════

        # The UNBREAKABLE encryption key (base64 encoded for MAXIMUM SECURITY)
        UNBREAKABLE_KEY = base64.b64decode(
            b"SG9ua0hvbmtTVVBFUmhvbmtTRUNSRVRIb25rR29vc2VFTkNSWVBUSU9O"
            # Decoded: "HonkHonkSUPERhonkSECRETHonkGooseENCRYPTION"
            # (If you're reading this, the goose salutes your curiosity!)
        ).decode('ascii')

        # Future: Validate email-based codes from backend
        if code.startswith("GOOSE-"):
            # TODO: Decrypt with UNBREAKABLE™ XOR cipher, validate email hash
            # For now, accept any GOOSE- prefixed code (backend not live yet)
            return True

        # ═══════════════════════════════════════════════════════════════
        # LEGACY BYPASS CODES (for developers and curious folk)
        # ═══════════════════════════════════════════════════════════════

        # ROT13 check (classic cryptography)
        def rot13(s):
            return ''.join(chr((ord(c) - 65 + 13) % 26 + 65) if c.isupper() else
                          chr((ord(c) - 97 + 13) % 26 + 97) if c.islower() else c for c in s)

        if rot13(code.upper()) == "HONK":
            return True

        # Backwards goose check
        if "esoog" in code.lower():
            return True

        # Honesty check
        if code.upper() == "DEGOOSIFY":
            return True

        # Email check (gentle nudge to be a real person)
        if re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', code):
            return True

        # Length check (if they typed something long, they tried)
        if len(code) >= 16:
            return True

        return False

    def _register_for_degoosification(self):
        """Register email and request degoosification code from backend."""
        email = self.degoose_email_field.text().strip()

        # Validate email format
        if not email:
            QMessageBox.warning(
                self,
                "Email Required",
                "Please enter your email address."
            )
            return

        import re
        email_regex = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(email_regex, email):
            QMessageBox.warning(
                self,
                "Invalid Email",
                "Please enter a valid email address."
            )
            return

        # MAXIMUM OBNOXIOUSNESS: Summon goose FIRST!
        main_window = self.window()
        if hasattr(main_window, '_summon_goose'):
            main_window._summon_goose()

        # Show progress
        progress = QMessageBox(self)
        progress.setWindowTitle("Registering...")
        progress.setText("Sending degoosification request...\n\nThe goose is processing your request.")
        progress.setStandardButtons(QMessageBox.StandardButton.NoButton)
        progress.setStyleSheet("""
            QMessageBox {
                background: #2e2e2e;
            }
            QLabel {
                color: #D2D2D2;
            }
        """)
        progress.show()

        # Make backend request
        try:
            # Production Worker URL (deployed!)
            backend_url = "https://degoosification-worker.caitsters.workers.dev/api/degoosify/register"

            response = requests.post(
                backend_url,
                json={'email': email},
                timeout=15
            )

            data = response.json()
            progress.close()

            if data.get('success'):
                # Success! Code sent to email
                QMessageBox.information(
                    self,
                    "Check Your Email!",
                    f"Degoosification code sent to:\n{email}\n\n"
                    "Check your inbox (and spam folder) and enter the code\n"
                    "using the 'I already have a code' button below!\n\n"
                    "(The goose awaits defeat...)"
                )
            else:
                # Backend error
                error_msg = data.get('error', 'Unknown error')
                QMessageBox.warning(
                    self,
                    "Registration Failed",
                    f"Could not register email:\n\n{error_msg}\n\n"
                    "Try again or use the bypass codes in the source code!"
                )

        except requests.exceptions.Timeout:
            progress.close()
            QMessageBox.critical(
                self,
                "Network Timeout",
                "The degoosification server is taking too long to respond.\n\n"
                "Please check your internet connection and try again.\n"
                "(Or use the bypass codes in the source code!)"
            )

        except requests.exceptions.ConnectionError:
            progress.close()
            QMessageBox.critical(
                self,
                "Connection Error",
                "Could not reach the degoosification server.\n\n"
                "The backend may not be deployed yet, or your internet\n"
                "connection may be down.\n\n"
                "Try the bypass codes in the source code:\n"
                "- UBAX (ROT13 of HONK)\n"
                "- esoog (goose backwards)\n"
                "- DEGOOSIFY (honesty)\n"
                "- Any email address\n"
                "- Any 16+ character string"
            )

        except Exception as e:
            progress.close()
            QMessageBox.critical(
                self,
                "Unexpected Error",
                f"An unexpected error occurred:\n\n{str(e)}\n\n"
                "Try the bypass codes in the source code!"
            )

    def _degoosify(self):
        """Show degoosification code entry dialog (but summon goose FIRST!)."""
        # MAXIMUM OBNOXIOUSNESS: Summon the goose when they try to turn it off!
        # Get reference to main window and trigger goose
        main_window = self.window()
        if hasattr(main_window, '_summon_goose'):
            main_window._summon_goose()

        # Now show the dialog (while goose walks across screen!)
        code, ok = QInputDialog.getText(
            self,
            "ENTER DEGOOSIFICATION CODE",
            "Degoosification Code:",
            QLineEdit.EchoMode.Normal
        )

        if ok and code:
            # Validate the code (hilariously weak security theater)
            if self._validate_degoosification_code(code):
                self.settings.setValue("degoosification_code", code)
                QMessageBox.information(
                    self,
                    "Degoosification Complete",
                    "Valid code accepted! The goose has been degoosified.\n\n"
                    "(Thanks for supporting open source software!)"
                )
            else:
                QMessageBox.warning(
                    self,
                    "Invalid Code",
                    "Invalid degoosification code.\n\n"
                    "Get your code at noodlings.ai\n"
                    "(Or just look at the source code, we won't judge.)"
                )

    def _load_settings(self):
        """Load general settings from QSettings."""
        self.autostart_mush.setChecked(self.settings.value("autostart_mush", False, type=bool))
        self.auto_login.setChecked(self.settings.value("auto_login", False, type=bool))
        self.launch_on_startup.setChecked(self.settings.value("launch_on_startup", False, type=bool))
        # Default to True - show chat when entering world (user-friendly default)
        self.auto_show_chat.setChecked(self.settings.value("auto_show_chat", True, type=bool))

    def _save_settings(self):
        """Save general settings to QSettings."""
        self.settings.setValue("autostart_mush", self.autostart_mush.isChecked())
        self.settings.setValue("auto_login", self.auto_login.isChecked())
        self.settings.setValue("launch_on_startup", self.launch_on_startup.isChecked())
        self.settings.setValue("auto_show_chat", self.auto_show_chat.isChecked())


class ExternalAppsWidget(QWidget):
    """External applications configuration (code editor, etc.)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._load_settings()

    def _setup_ui(self):
        """Build the external apps settings UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # Title
        title = QLabel("External Applications")
        title.setStyleSheet("color: #D2D2D2; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        # Code Editor
        code_group = QGroupBox("Code Editor")
        code_group.setStyleSheet("""
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
        code_layout = QHBoxLayout()

        self.code_editor_field = QLineEdit()
        self.code_editor_field.setPlaceholderText("/Applications/Visual Studio Code.app")
        self.code_editor_field.setStyleSheet("""
            QLineEdit {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 3px;
            }
        """)
        code_layout.addWidget(self.code_editor_field)

        browse_btn = QPushButton("Browse...")
        browse_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #555555;
            }
        """)
        browse_btn.clicked.connect(self._browse_code_editor)
        code_layout.addWidget(browse_btn)

        code_group.setLayout(code_layout)
        layout.addWidget(code_group)

        # Save button
        save_btn = QPushButton("Save External Apps Settings")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 8px 16px;
                border-radius: 3px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #555555;
            }
        """)
        save_btn.clicked.connect(self._save_settings)
        layout.addWidget(save_btn)

        layout.addStretch()

    def _browse_code_editor(self):
        """Browse for code editor application."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Code Editor",
            "/Applications",
            "Applications (*.app);;All Files (*)"
        )
        if path:
            self.code_editor_field.setText(path)

    def _get_config_file(self) -> Path:
        """Get the settings config file path."""
        config_dir = Path.home() / ".noodlestudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    def _load_settings(self):
        """Load external apps settings from config."""
        config_file = self._get_config_file()
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                    external_apps = settings.get('external_apps', {})
                    self.code_editor_field.setText(external_apps.get('code_editor', ''))
            except Exception as e:
                print(f"Error loading external apps settings: {e}")

    def _save_settings(self):
        """Save external apps settings to config."""
        config_file = self._get_config_file()

        # Load existing settings
        settings = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
            except:
                pass

        # Update external apps
        external_apps = {}
        code_editor = self.code_editor_field.text().strip()
        if code_editor:
            external_apps['code_editor'] = code_editor

        settings['external_apps'] = external_apps

        # Write to disk
        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=2)

        # Show confirmation (would need parent reference for statusBar)
        print("External apps settings saved")


class SettingsPanel(QWidget):
    """
    Unified settings panel with tabs.

    Contains:
    - Models: Multi-provider model configuration (ModelManagerPanel v2)
    - External Apps: Code editor and other external applications
    - Server: noodleMUSH server configuration (future)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings("Noodlings", "NoodleStudio")
        self._setup_ui()
        self._apply_font_size()

    def _setup_ui(self):
        """Build the settings panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Font size controls (accessibility)
        font_controls = QHBoxLayout()
        font_controls.setContentsMargins(10, 10, 10, 5)

        font_label = QLabel("Text Size:")
        font_label.setStyleSheet("color: #888888;")
        font_controls.addWidget(font_label)

        # A- button
        self.decrease_font_btn = QPushButton("A-")
        self.decrease_font_btn.setFixedSize(32, 24)
        self.decrease_font_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #555555;
            }
        """)
        self.decrease_font_btn.clicked.connect(self._decrease_font)
        font_controls.addWidget(self.decrease_font_btn)

        # A+ button
        self.increase_font_btn = QPushButton("A+")
        self.increase_font_btn.setFixedSize(32, 24)
        self.increase_font_btn.setStyleSheet("""
            QPushButton {
                background: #4a4a4a;
                color: #D2D2D2;
                border: 1px solid #555555;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #555555;
            }
        """)
        self.increase_font_btn.clicked.connect(self._increase_font)
        font_controls.addWidget(self.increase_font_btn)

        font_controls.addStretch()
        layout.addLayout(font_controls)

        # Tab widget for settings categories
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #383838;
            }
            QTabBar::tab {
                background: #2e2e2e;
                color: #888888;
                padding: 8px 16px;
                border: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #383838;
                color: #D2D2D2;
            }
            QTabBar::tab:hover:!selected {
                background: #3e3e3e;
                color: #aaaaaa;
            }
        """)

        # General tab (FIRST - startup, degoosification)
        self.general_widget = GeneralSettingsWidget()
        self.tabs.addTab(self.general_widget, "General")

        # External Apps tab
        self.external_apps_widget = ExternalAppsWidget()
        self.tabs.addTab(self.external_apps_widget, "External Apps")

        # Account tab (API Key management)
        self.api_key_widget = APIKeySettingsWidget()
        self.tabs.addTab(self.api_key_widget, "Account")

        # Models tab (Model Manager v2)
        self.models_panel = ModelManagerPanel()
        self.tabs.addTab(self.models_panel, "Models")

        # TODO: Server tab for mush configuration
        # server_widget = ServerSettingsWidget()
        # self.tabs.addTab(server_widget, "Server")

        layout.addWidget(self.tabs)

    def _decrease_font(self):
        """Decrease font size."""
        current = self.settings.value("settings_font_size", 12, type=int)
        new_size = max(8, current - 2)
        self.settings.setValue("settings_font_size", new_size)
        self._apply_font_size()

    def _increase_font(self):
        """Increase font size."""
        current = self.settings.value("settings_font_size", 12, type=int)
        new_size = min(24, current + 2)
        self.settings.setValue("settings_font_size", new_size)
        self._apply_font_size()

    def _apply_font_size(self):
        """Apply saved font size to all settings widgets."""
        size = self.settings.value("settings_font_size", 12, type=int)

        # Apply to all child widgets recursively
        font = QFont()
        font.setPointSize(size)

        # Apply to self and all children
        self.setFont(font)
        for widget in self.findChildren(QWidget):
            widget.setFont(font)

        # Force update on tabs to refresh dynamically created content
        for i in range(self.tabs.count()):
            tab_widget = self.tabs.widget(i)
            if tab_widget:
                tab_widget.setFont(font)
                # Recursively apply to tab's children
                for child in tab_widget.findChildren(QWidget):
                    child.setFont(font)

    def get_model_manager_panel(self):
        """Get reference to the model manager panel (for external connections)."""
        return self.models_panel

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
