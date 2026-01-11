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
#   Build Settings Dialog - Unity-style Build Settings
#
#   File > Build Settings... (Ctrl+Shift+B)
#   Configures all build options and saves to build.yaml
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.dialogs.build_settings_dialog
# PURPOSE:  Build Settings Dialog
# LAYER:    Studio / Dialogs
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   BuildSettingsDialog
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox, QRadioButton,
    QButtonGroup, QGroupBox, QScrollArea, QWidget,
    QFileDialog, QMessageBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QFrame
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

from ..core.build_config import BuildConfig
from ..widgets.collapsible_section import CollapsibleSection
from ..widgets.color_picker_widget import ColorFieldWidget

logger = logging.getLogger(__name__)


class BuildSettingsDialog(QDialog):
    """
    Unity-style Build Settings dialog.

    Provides comprehensive configuration for building standalone applications:
    - Target platform selection
    - App identity (name, bundle ID, version, icon)
    - Splash screen configuration
    - Editor access settings
    - LLM provider configuration
    - Content inclusion options
    - Distribution and signing settings
    - Advanced options

    Settings are saved to build.yaml in the project root.
    """

    def __init__(self, project_path: Path, parent=None):
        """
        Initialize the Build Settings dialog.

        Args:
            project_path: Path to the project directory
            parent: Parent widget
        """
        super().__init__(parent)
        self.project_path = Path(project_path)
        self.config = self._load_or_create_config()

        self.setWindowTitle("Build Settings")
        self.setMinimumSize(650, 700)
        self.resize(700, 800)

        self._setup_ui()
        self._load_values_from_config()

    def _load_or_create_config(self) -> BuildConfig:
        """Load existing build.yaml or create default config."""
        yaml_path = self.project_path / "build.yaml"
        if yaml_path.exists():
            try:
                return BuildConfig.from_yaml(yaml_path)
            except Exception as e:
                logger.warning(f"Failed to load build.yaml: {e}")

        # Create default config with project name
        return BuildConfig.default(
            name=self.project_path.name,
            bundle_id=f"ai.noodlings.{self.project_path.name.lower().replace(' ', '').replace('-', '')}"
        )

    def _setup_ui(self):
        """Build the dialog UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # Header
        header = QLabel("Build Settings")
        header_font = QFont()
        header_font.setPointSize(16)
        header_font.setBold(True)
        header.setFont(header_font)
        main_layout.addWidget(header)

        # Scroll area for sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(8)
        scroll_layout.setContentsMargins(0, 0, 8, 0)

        # ===== TARGET PLATFORM =====
        platform_section = self._create_platform_section()
        scroll_layout.addWidget(platform_section)

        # ===== APP IDENTITY =====
        identity_section = self._create_identity_section()
        scroll_layout.addWidget(identity_section)

        # ===== SPLASH SCREEN (collapsible) =====
        self.splash_section = self._create_splash_section()
        scroll_layout.addWidget(self.splash_section)

        # ===== EDITOR ACCESS (collapsible) =====
        self.editor_section = self._create_editor_section()
        scroll_layout.addWidget(self.editor_section)

        # ===== LLM PROVIDER (collapsible) =====
        self.llm_section = self._create_llm_section()
        scroll_layout.addWidget(self.llm_section)

        # ===== INCLUDED CONTENT (collapsible) =====
        self.content_section = self._create_content_section()
        scroll_layout.addWidget(self.content_section)

        # ===== DISTRIBUTION (collapsible) =====
        self.distribution_section = self._create_distribution_section()
        scroll_layout.addWidget(self.distribution_section)

        # ===== ADVANCED (collapsible) =====
        self.advanced_section = self._create_advanced_section()
        scroll_layout.addWidget(self.advanced_section)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        # ===== OUTPUT DIRECTORY =====
        output_layout = QHBoxLayout()
        output_label = QLabel("Output Directory:")
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("~/Desktop/builds")
        output_browse = QPushButton("Browse...")
        output_browse.clicked.connect(self._browse_output_directory)
        output_layout.addWidget(output_label)
        output_layout.addWidget(self.output_edit, 1)
        output_layout.addWidget(output_browse)
        main_layout.addLayout(output_layout)

        # ===== BUTTONS =====
        main_layout.addWidget(self._create_separator())

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(cancel_btn)

        build_run_btn = QPushButton("Build and Run")
        build_run_btn.clicked.connect(self._build_and_run)
        button_layout.addWidget(build_run_btn)

        build_btn = QPushButton("Build")
        build_btn.setDefault(True)
        build_btn.setStyleSheet("""
            QPushButton {
                background: #4a9eff;
                color: white;
                padding: 8px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background: #3a8eef;
            }
        """)
        build_btn.clicked.connect(self._build)
        button_layout.addWidget(build_btn)

        main_layout.addLayout(button_layout)

    def _create_separator(self) -> QFrame:
        """Create a horizontal separator line."""
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3d3d3d;")
        sep.setFixedHeight(1)
        return sep

    # -------------------------------------------------------------------------
    # TARGET PLATFORM
    # -------------------------------------------------------------------------
    def _create_platform_section(self) -> QGroupBox:
        """Create the Target Platform section."""
        group = QGroupBox("Target Platform")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)

        self.platform_group = QButtonGroup(self)

        self.platform_macos = QRadioButton("macOS Application (.app)")
        self.platform_windows = QRadioButton("Windows Application (.exe)")
        self.platform_linux = QRadioButton("Linux Application")
        self.platform_web = QRadioButton("Web (coming soon)")
        self.platform_web.setEnabled(False)

        self.platform_group.addButton(self.platform_macos, 0)
        self.platform_group.addButton(self.platform_windows, 1)
        self.platform_group.addButton(self.platform_linux, 2)
        self.platform_group.addButton(self.platform_web, 3)

        layout.addWidget(self.platform_macos)
        layout.addWidget(self.platform_windows)
        layout.addWidget(self.platform_linux)
        layout.addWidget(self.platform_web)

        return group

    # -------------------------------------------------------------------------
    # APP IDENTITY
    # -------------------------------------------------------------------------
    def _create_identity_section(self) -> QGroupBox:
        """Create the App Identity section."""
        group = QGroupBox("App Identity")
        layout = QFormLayout(group)
        layout.setSpacing(8)

        self.identity_name = QLineEdit()
        self.identity_name.setPlaceholderText("My App")
        layout.addRow("App Name:", self.identity_name)

        self.identity_bundle_id = QLineEdit()
        self.identity_bundle_id.setPlaceholderText("ai.noodlings.myapp")
        layout.addRow("Bundle ID:", self.identity_bundle_id)

        self.identity_version = QLineEdit()
        self.identity_version.setPlaceholderText("1.0.0")
        layout.addRow("Version:", self.identity_version)

        # Icon picker
        icon_layout = QHBoxLayout()
        self.identity_icon = QLineEdit()
        self.identity_icon.setPlaceholderText("assets/icon.png")
        icon_browse = QPushButton("Browse...")
        icon_browse.clicked.connect(self._browse_icon)
        icon_layout.addWidget(self.identity_icon)
        icon_layout.addWidget(icon_browse)

        # Icon preview
        self.icon_preview = QLabel()
        self.icon_preview.setFixedSize(48, 48)
        self.icon_preview.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        icon_layout.addWidget(self.icon_preview)

        layout.addRow("App Icon:", icon_layout)

        return group

    # -------------------------------------------------------------------------
    # SPLASH SCREEN
    # -------------------------------------------------------------------------
    def _create_splash_section(self) -> CollapsibleSection:
        """Create the Splash Screen section (collapsible)."""
        section = CollapsibleSection("Splash Screen")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Enable checkbox
        self.splash_enabled = QCheckBox("Show splash screen on launch")
        layout.addWidget(self.splash_enabled)

        # Form for splash settings
        form = QFormLayout()
        form.setSpacing(6)

        # Image picker
        image_layout = QHBoxLayout()
        self.splash_image = QLineEdit()
        self.splash_image.setPlaceholderText("assets/splash.png")
        image_browse = QPushButton("Browse...")
        image_browse.clicked.connect(self._browse_splash_image)
        image_layout.addWidget(self.splash_image)
        image_layout.addWidget(image_browse)
        form.addRow("Splash Image:", image_layout)

        # Image preview
        self.splash_preview = QLabel()
        self.splash_preview.setFixedSize(200, 120)
        self.splash_preview.setStyleSheet("""
            QLabel {
                background-color: #2d2d2d;
                border: 1px solid #3d3d3d;
                border-radius: 4px;
            }
        """)
        self.splash_preview.setScaledContents(True)
        form.addRow("", self.splash_preview)

        # Duration
        self.splash_duration = QDoubleSpinBox()
        self.splash_duration.setRange(0.5, 30.0)
        self.splash_duration.setSingleStep(0.5)
        self.splash_duration.setSuffix(" seconds")
        form.addRow("Duration:", self.splash_duration)

        # Click to dismiss
        self.splash_click_dismiss = QCheckBox("Click/keypress to dismiss")
        form.addRow("", self.splash_click_dismiss)

        # Background color
        self.splash_background = ColorFieldWidget()
        form.addRow("Background:", self.splash_background)

        # Fade timings
        fade_layout = QHBoxLayout()
        self.splash_fade_in = QDoubleSpinBox()
        self.splash_fade_in.setRange(0.0, 5.0)
        self.splash_fade_in.setSingleStep(0.1)
        self.splash_fade_in.setSuffix(" s")
        fade_layout.addWidget(QLabel("Fade In:"))
        fade_layout.addWidget(self.splash_fade_in)
        fade_layout.addSpacing(16)

        self.splash_fade_out = QDoubleSpinBox()
        self.splash_fade_out.setRange(0.0, 5.0)
        self.splash_fade_out.setSingleStep(0.1)
        self.splash_fade_out.setSuffix(" s")
        fade_layout.addWidget(QLabel("Fade Out:"))
        fade_layout.addWidget(self.splash_fade_out)
        fade_layout.addStretch()
        form.addRow("", fade_layout)

        layout.addLayout(form)

        # Attribution section (locked)
        layout.addWidget(self._create_separator())

        attr_label = QLabel("REQUIRED ATTRIBUTION (cannot be disabled)")
        attr_label.setStyleSheet("color: #888888; font-size: 11px; font-weight: bold;")
        layout.addWidget(attr_label)

        # Locked checkboxes
        self.attr_badge = QCheckBox('"Made with NoodleSTUDIO" badge')
        self.attr_badge.setChecked(True)
        self.attr_badge.setEnabled(False)
        layout.addWidget(self.attr_badge)

        self.attr_nec = QCheckBox("Link to Noodling Ethical Covenant")
        self.attr_nec.setChecked(True)
        self.attr_nec.setEnabled(False)
        layout.addWidget(self.attr_nec)

        # Position dropdown
        pos_layout = QHBoxLayout()
        pos_layout.addWidget(QLabel("Position:"))
        self.attr_position = QComboBox()
        self.attr_position.addItems(["Bottom Right", "Bottom Left", "Bottom Center"])
        pos_layout.addWidget(self.attr_position)
        pos_layout.addStretch()
        layout.addLayout(pos_layout)

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # EDITOR ACCESS
    # -------------------------------------------------------------------------
    def _create_editor_section(self) -> CollapsibleSection:
        """Create the Editor Access section (collapsible)."""
        section = CollapsibleSection("Editor Access")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.editor_group = QButtonGroup(self)

        # Allow unfold
        self.editor_allow = QRadioButton('Allow "View Project" (unfold to editor)')
        self.editor_group.addButton(self.editor_allow, 0)
        layout.addWidget(self.editor_allow)

        # Keyboard shortcut
        shortcut_layout = QHBoxLayout()
        shortcut_layout.addSpacing(24)
        shortcut_layout.addWidget(QLabel("Keyboard shortcut:"))
        self.editor_shortcut = QLineEdit()
        self.editor_shortcut.setPlaceholderText("Ctrl+Shift+U")
        self.editor_shortcut.setFixedWidth(150)
        shortcut_layout.addWidget(self.editor_shortcut)
        shortcut_layout.addStretch()
        layout.addLayout(shortcut_layout)

        # Password protected
        self.editor_password = QRadioButton("Require password to unfold")
        self.editor_group.addButton(self.editor_password, 1)
        layout.addWidget(self.editor_password)

        # Password field
        pw_layout = QHBoxLayout()
        pw_layout.addSpacing(24)
        pw_layout.addWidget(QLabel("Password:"))
        self.editor_pw_field = QLineEdit()
        self.editor_pw_field.setEchoMode(QLineEdit.EchoMode.Password)
        self.editor_pw_field.setFixedWidth(200)
        self.editor_pw_field.setEnabled(False)
        pw_layout.addWidget(self.editor_pw_field)
        pw_layout.addStretch()
        layout.addLayout(pw_layout)

        # Hidden completely
        self.editor_hidden = QRadioButton("Hide editor completely (app-only mode)")
        self.editor_group.addButton(self.editor_hidden, 2)
        layout.addWidget(self.editor_hidden)

        warning = QLabel("Users cannot inspect or modify the project")
        warning.setStyleSheet("color: #ffaa00; font-size: 11px; margin-left: 24px;")
        layout.addWidget(warning)

        # Connect signals
        self.editor_password.toggled.connect(lambda checked: self.editor_pw_field.setEnabled(checked))

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # LLM PROVIDER
    # -------------------------------------------------------------------------
    def _create_llm_section(self) -> CollapsibleSection:
        """Create the LLM Provider section (collapsible)."""
        section = CollapsibleSection("LLM Provider")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.llm_group = QButtonGroup(self)

        # NoodleROUTER
        self.llm_noodlerouter = QRadioButton("NoodleROUTER (recommended)")
        self.llm_group.addButton(self.llm_noodlerouter, 0)
        layout.addWidget(self.llm_noodlerouter)

        desc1 = QLabel("Uses noodlings.ai API. Users need account.\nCost: Provider rate + 20% margin")
        desc1.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(desc1)

        # User keys
        self.llm_user_keys = QRadioButton("User provides own API keys")
        self.llm_group.addButton(self.llm_user_keys, 1)
        layout.addWidget(self.llm_user_keys)

        desc2 = QLabel("Users enter their own Anthropic/OpenAI keys.\nSettings panel will prompt for keys on first run.")
        desc2.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(desc2)

        # Ollama
        self.llm_ollama = QRadioButton("Local models only (Ollama)")
        self.llm_group.addButton(self.llm_ollama, 2)
        layout.addWidget(self.llm_ollama)

        desc3 = QLabel("Requires Ollama installed. No cloud dependency.")
        desc3.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(desc3)

        warn3 = QLabel("Limited model selection, requires user setup")
        warn3.setStyleSheet("color: #ffaa00; font-size: 11px; margin-left: 24px;")
        layout.addWidget(warn3)

        # Bundled key
        self.llm_bundled = QRadioButton("Bundled API key (not recommended)")
        self.llm_group.addButton(self.llm_bundled, 3)
        layout.addWidget(self.llm_bundled)

        warn4 = QLabel("Your key is embedded in the app. You pay for all usage.")
        warn4.setStyleSheet("color: #ff6666; font-size: 11px; margin-left: 24px;")
        layout.addWidget(warn4)

        # Bundled key field
        key_layout = QHBoxLayout()
        key_layout.addSpacing(24)
        key_layout.addWidget(QLabel("Key:"))
        self.llm_bundled_key = QLineEdit()
        self.llm_bundled_key.setPlaceholderText("sk-...")
        self.llm_bundled_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.llm_bundled_key.setEnabled(False)
        key_layout.addWidget(self.llm_bundled_key)
        key_layout.addStretch()
        layout.addLayout(key_layout)

        # Connect signals
        self.llm_bundled.toggled.connect(lambda checked: self.llm_bundled_key.setEnabled(checked))

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # INCLUDED CONTENT
    # -------------------------------------------------------------------------
    def _create_content_section(self) -> CollapsibleSection:
        """Create the Included Content section (collapsible)."""
        section = CollapsibleSection("Included Content")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.content_stages = QCheckBox("All stages")
        self.content_noodlings = QCheckBox("All noodlings")
        self.content_ui = QCheckBox("All UI layouts")
        self.content_assemblies = QCheckBox("All facet assemblies")
        self.content_plays = QCheckBox("All plays (.play.yaml)")

        layout.addWidget(self.content_stages)
        layout.addWidget(self.content_noodlings)
        layout.addWidget(self.content_ui)
        layout.addWidget(self.content_assemblies)
        layout.addWidget(self.content_plays)

        layout.addSpacing(8)

        self.content_unused = QCheckBox("Include unused assets")
        self.content_source = QCheckBox("Include source facet code")
        layout.addWidget(self.content_unused)
        layout.addWidget(self.content_source)

        layout.addSpacing(8)

        # Size estimate (placeholder)
        self.size_estimate = QLabel("Estimated size: calculating...")
        self.size_estimate.setStyleSheet("color: #888888; font-size: 11px;")
        layout.addWidget(self.size_estimate)

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # DISTRIBUTION
    # -------------------------------------------------------------------------
    def _create_distribution_section(self) -> CollapsibleSection:
        """Create the Distribution section (collapsible)."""
        section = CollapsibleSection("Distribution")
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        sign_label = QLabel("Signing:")
        sign_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(sign_label)

        self.dist_group = QButtonGroup(self)

        # NoodleStudio Signed
        self.dist_noodlestudio = QRadioButton("NoodleStudio Signed (recommended)")
        self.dist_group.addButton(self.dist_noodlestudio, 0)
        layout.addWidget(self.dist_noodlestudio)

        desc1 = QLabel("Free signed distribution. Requires attribution.")
        desc1.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(desc1)

        # Own certificate
        self.dist_own_cert = QRadioButton("Your Own Certificate")
        self.dist_group.addButton(self.dist_own_cert, 1)
        layout.addWidget(self.dist_own_cert)

        desc2 = QLabel("Use your Apple Developer ID. You handle signing.")
        desc2.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(desc2)

        # Certificate field
        cert_layout = QHBoxLayout()
        cert_layout.addSpacing(24)
        cert_layout.addWidget(QLabel("Certificate:"))
        self.dist_certificate = QLineEdit()
        self.dist_certificate.setPlaceholderText("Developer ID Application: ...")
        self.dist_certificate.setEnabled(False)
        cert_layout.addWidget(self.dist_certificate)
        cert_select = QPushButton("Select...")
        cert_select.setEnabled(False)
        self.dist_cert_btn = cert_select
        cert_layout.addWidget(cert_select)
        layout.addLayout(cert_layout)

        # Unsigned
        self.dist_unsigned = QRadioButton("Unsigned")
        self.dist_group.addButton(self.dist_unsigned, 2)
        layout.addWidget(self.dist_unsigned)

        warn = QLabel("Users will see security warnings on macOS")
        warn.setStyleSheet("color: #ffaa00; font-size: 11px; margin-left: 24px;")
        layout.addWidget(warn)

        layout.addWidget(self._create_separator())

        # Notarization
        self.dist_notarize = QCheckBox("Submit for notarization after build")
        layout.addWidget(self.dist_notarize)

        notarize_desc = QLabel("Required for NoodleStudio Signed distribution.")
        notarize_desc.setStyleSheet("color: #888888; font-size: 11px; margin-left: 24px;")
        layout.addWidget(notarize_desc)

        # Connect signals
        self.dist_own_cert.toggled.connect(lambda checked: self.dist_certificate.setEnabled(checked))
        self.dist_own_cert.toggled.connect(lambda checked: self.dist_cert_btn.setEnabled(checked))

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # ADVANCED
    # -------------------------------------------------------------------------
    def _create_advanced_section(self) -> CollapsibleSection:
        """Create the Advanced section (collapsible)."""
        section = CollapsibleSection("Advanced")
        layout = QFormLayout()
        layout.setSpacing(8)

        # Python version
        self.adv_python = QComboBox()
        self.adv_python.addItems(["3.11 (bundled)", "3.12 (bundled)"])
        layout.addRow("Python Version:", self.adv_python)

        # Qt version
        self.adv_qt = QComboBox()
        self.adv_qt.addItems(["6.6.1 (bundled)", "6.7.0 (bundled)"])
        layout.addRow("Qt Version:", self.adv_qt)

        # Strip debug
        self.adv_strip_debug = QCheckBox("Strip debug symbols")
        layout.addRow("", self.adv_strip_debug)

        # Build hooks
        layout.addRow(QLabel("Build script hooks:"))

        pre_layout = QHBoxLayout()
        self.adv_pre_build = QLineEdit()
        self.adv_pre_build.setPlaceholderText("scripts/pre_build.py")
        pre_browse = QPushButton("Browse...")
        pre_browse.clicked.connect(self._browse_pre_build)
        pre_layout.addWidget(self.adv_pre_build)
        pre_layout.addWidget(pre_browse)
        layout.addRow("Pre-build:", pre_layout)

        post_layout = QHBoxLayout()
        self.adv_post_build = QLineEdit()
        self.adv_post_build.setPlaceholderText("scripts/post_build.py")
        post_browse = QPushButton("Browse...")
        post_browse.clicked.connect(self._browse_post_build)
        post_layout.addWidget(self.adv_post_build)
        post_layout.addWidget(post_browse)
        layout.addRow("Post-build:", post_layout)

        section.set_content_layout(layout)
        section.set_expanded(False)
        return section

    # -------------------------------------------------------------------------
    # FILE BROWSERS
    # -------------------------------------------------------------------------
    def _browse_icon(self):
        """Browse for app icon file."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select App Icon",
            str(self.project_path),
            "Images (*.png *.icns *.ico);;All Files (*)"
        )
        if path:
            rel_path = self._make_relative(path)
            self.identity_icon.setText(rel_path)
            self._update_icon_preview(path)

    def _browse_splash_image(self):
        """Browse for splash screen image."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Splash Image",
            str(self.project_path),
            "Images (*.png *.jpg *.jpeg);;All Files (*)"
        )
        if path:
            rel_path = self._make_relative(path)
            self.splash_image.setText(rel_path)
            self._update_splash_preview(path)

    def _browse_output_directory(self):
        """Browse for output directory."""
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory",
            str(Path(self.output_edit.text()).expanduser()) if self.output_edit.text() else str(Path.home() / "Desktop")
        )
        if path:
            self.output_edit.setText(path)

    def _browse_pre_build(self):
        """Browse for pre-build script."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Pre-build Script",
            str(self.project_path),
            "Python Scripts (*.py);;All Files (*)"
        )
        if path:
            self.adv_pre_build.setText(self._make_relative(path))

    def _browse_post_build(self):
        """Browse for post-build script."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Post-build Script",
            str(self.project_path),
            "Python Scripts (*.py);;All Files (*)"
        )
        if path:
            self.adv_post_build.setText(self._make_relative(path))

    def _make_relative(self, path: str) -> str:
        """Convert absolute path to relative path from project."""
        try:
            return str(Path(path).relative_to(self.project_path))
        except ValueError:
            return path

    def _update_icon_preview(self, path: str):
        """Update the icon preview image."""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.icon_preview.setPixmap(pixmap.scaled(
                48, 48,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    def _update_splash_preview(self, path: str):
        """Update the splash preview image."""
        pixmap = QPixmap(path)
        if not pixmap.isNull():
            self.splash_preview.setPixmap(pixmap.scaled(
                200, 120,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            ))

    # -------------------------------------------------------------------------
    # CONFIG LOAD/SAVE
    # -------------------------------------------------------------------------
    def _load_values_from_config(self):
        """Populate UI fields from config."""
        c = self.config

        # Target platform
        platform_map = {"macos": 0, "windows": 1, "linux": 2}
        btn = self.platform_group.button(platform_map.get(c.target, 0))
        if btn:
            btn.setChecked(True)

        # Identity
        self.identity_name.setText(c.identity.name)
        self.identity_bundle_id.setText(c.identity.bundle_id)
        self.identity_version.setText(c.identity.version)
        self.identity_icon.setText(c.identity.icon)
        if c.identity.icon:
            icon_path = self.project_path / c.identity.icon
            if icon_path.exists():
                self._update_icon_preview(str(icon_path))

        # Splash
        self.splash_enabled.setChecked(c.splash.enabled)
        self.splash_image.setText(c.splash.image)
        if c.splash.image:
            splash_path = self.project_path / c.splash.image
            if splash_path.exists():
                self._update_splash_preview(str(splash_path))
        self.splash_duration.setValue(c.splash.duration)
        self.splash_click_dismiss.setChecked(c.splash.click_to_dismiss)
        self.splash_background.setColor(c.splash.background)
        self.splash_fade_in.setValue(c.splash.fade_in)
        self.splash_fade_out.setValue(c.splash.fade_out)

        # Attribution position
        pos_map = {"bottom_right": 0, "bottom_left": 1, "bottom_center": 2}
        self.attr_position.setCurrentIndex(pos_map.get(c.splash.attribution_position, 0))

        # Editor access
        access_map = {"allow": 0, "password": 1, "hidden": 2}
        btn = self.editor_group.button(access_map.get(c.editor.access, 0))
        if btn:
            btn.setChecked(True)
        self.editor_shortcut.setText(c.editor.keyboard_shortcut)

        # LLM provider
        llm_map = {"noodlerouter": 0, "user_keys": 1, "ollama": 2, "bundled": 3}
        btn = self.llm_group.button(llm_map.get(c.llm.provider, 0))
        if btn:
            btn.setChecked(True)
        if c.llm.bundled_key:
            self.llm_bundled_key.setText(c.llm.bundled_key)

        # Content
        self.content_stages.setChecked(c.content.include_stages)
        self.content_noodlings.setChecked(c.content.include_noodlings)
        self.content_ui.setChecked(c.content.include_ui_layouts)
        self.content_assemblies.setChecked(c.content.include_assemblies)
        self.content_plays.setChecked(c.content.include_plays)
        self.content_unused.setChecked(c.content.include_unused)
        self.content_source.setChecked(c.content.include_source)

        # Distribution
        dist_map = {"noodlestudio": 0, "own_cert": 1, "unsigned": 2}
        btn = self.dist_group.button(dist_map.get(c.distribution.signing, 0))
        if btn:
            btn.setChecked(True)
        if c.distribution.certificate:
            self.dist_certificate.setText(c.distribution.certificate)
        self.dist_notarize.setChecked(c.distribution.notarize)

        # Advanced
        self.adv_strip_debug.setChecked(c.advanced.strip_debug)
        if c.advanced.hooks.pre_build:
            self.adv_pre_build.setText(c.advanced.hooks.pre_build)
        if c.advanced.hooks.post_build:
            self.adv_post_build.setText(c.advanced.hooks.post_build)

        # Output
        self.output_edit.setText(c.output_directory)

    def _save_values_to_config(self):
        """Save UI field values to config."""
        c = self.config

        # Target platform
        platform_map = {0: "macos", 1: "windows", 2: "linux"}
        c.target = platform_map.get(self.platform_group.checkedId(), "macos")

        # Identity
        c.identity.name = self.identity_name.text() or "Untitled"
        c.identity.bundle_id = self.identity_bundle_id.text() or "ai.noodlings.untitled"
        c.identity.version = self.identity_version.text() or "1.0.0"
        c.identity.icon = self.identity_icon.text()

        # Splash
        c.splash.enabled = self.splash_enabled.isChecked()
        c.splash.image = self.splash_image.text()
        c.splash.duration = self.splash_duration.value()
        c.splash.click_to_dismiss = self.splash_click_dismiss.isChecked()
        c.splash.background = self.splash_background.color().name()
        c.splash.fade_in = self.splash_fade_in.value()
        c.splash.fade_out = self.splash_fade_out.value()

        # Attribution position
        pos_map = {0: "bottom_right", 1: "bottom_left", 2: "bottom_center"}
        c.splash.attribution_position = pos_map.get(self.attr_position.currentIndex(), "bottom_right")

        # Editor access
        access_map = {0: "allow", 1: "password", 2: "hidden"}
        c.editor.access = access_map.get(self.editor_group.checkedId(), "allow")
        c.editor.keyboard_shortcut = self.editor_shortcut.text() or "Ctrl+Shift+U"

        # Password handling - use SHA-256 hash
        if c.editor.access == "password" and self.editor_pw_field.text():
            from .editor_password_dialog import hash_password
            c.editor.password_hash = hash_password(self.editor_pw_field.text())

        # LLM provider
        llm_map = {0: "noodlerouter", 1: "user_keys", 2: "ollama", 3: "bundled"}
        c.llm.provider = llm_map.get(self.llm_group.checkedId(), "noodlerouter")
        if c.llm.provider == "bundled":
            c.llm.bundled_key = self.llm_bundled_key.text() or None

        # Content
        c.content.include_stages = self.content_stages.isChecked()
        c.content.include_noodlings = self.content_noodlings.isChecked()
        c.content.include_ui_layouts = self.content_ui.isChecked()
        c.content.include_assemblies = self.content_assemblies.isChecked()
        c.content.include_plays = self.content_plays.isChecked()
        c.content.include_unused = self.content_unused.isChecked()
        c.content.include_source = self.content_source.isChecked()

        # Distribution
        dist_map = {0: "noodlestudio", 1: "own_cert", 2: "unsigned"}
        c.distribution.signing = dist_map.get(self.dist_group.checkedId(), "noodlestudio")
        if c.distribution.signing == "own_cert":
            c.distribution.certificate = self.dist_certificate.text() or None
        c.distribution.notarize = self.dist_notarize.isChecked()

        # Advanced
        c.advanced.strip_debug = self.adv_strip_debug.isChecked()
        c.advanced.hooks.pre_build = self.adv_pre_build.text() or None
        c.advanced.hooks.post_build = self.adv_post_build.text() or None

        # Output
        c.output_directory = self.output_edit.text() or "~/Desktop/builds"

    def _save_config(self) -> bool:
        """Save config to build.yaml."""
        try:
            self._save_values_to_config()
            yaml_path = self.project_path / "build.yaml"
            self.config.to_yaml(yaml_path)
            logger.info(f"Saved build settings to {yaml_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to save build settings: {e}")
            QMessageBox.critical(self, "Save Error", f"Failed to save build settings:\n{e}")
            return False

    # -------------------------------------------------------------------------
    # BUILD ACTIONS
    # -------------------------------------------------------------------------
    def _build(self):
        """Save settings and initiate build."""
        if self._save_config():
            self.accept()
            # TODO: Trigger actual build process in Phase 7

    def _build_and_run(self):
        """Save settings, build, and run the result."""
        if self._save_config():
            self.accept()
            # TODO: Trigger build + run in Phase 7

    def accept(self):
        """Override accept to save config before closing."""
        self._save_config()
        super().accept()


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
