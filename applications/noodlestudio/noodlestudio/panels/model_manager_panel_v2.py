"""
Model Manager Panel V2 - Multi-provider model configuration.

Features:
- Provider selector (Ollama, Anthropic, OpenAI, OpenRouter, LM Studio, custom)
- Provider configuration (API keys, endpoints)
- Model browser with search
- Label assignment (SMALL/MEDIUM/LARGE/custom to any provider's models)
- Download progress tracking (Ollama)
- Cross-provider label overview
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame, QMessageBox, QComboBox,
    QLineEdit, QDialog, QFormLayout, QSpinBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont

from ..core.model_label_manager import get_model_label_manager
from ..core.provider_manager import get_provider_manager, ProviderConfig


class ProviderConfigDialog(QDialog):
    """Dialog for configuring provider settings (API keys, endpoints)."""

    def __init__(self, provider_config: ProviderConfig, parent=None):
        super().__init__(parent)
        self.provider_config = provider_config
        self.setWindowTitle(f"Configure {provider_config.name}")
        self.setModal(True)
        self.resize(450, 300)

        self._setup_ui()

    def _setup_ui(self):
        """Build dialog UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        # Title
        title = QLabel(f"Configure {self.provider_config.name}")
        title.setStyleSheet("color: #D2D2D2; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        # Form
        form = QFormLayout()
        form.setSpacing(10)

        # Provider type (read-only)
        type_label = QLabel(self.provider_config.type)
        type_label.setStyleSheet("color: #888888;")
        form.addRow("Type:", type_label)

        # API Key (if needed)
        if self.provider_config.type in ["anthropic", "openai", "openrouter", "custom"]:
            self.api_key_input = QLineEdit()
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.api_key_input.setPlaceholderText("Enter API key...")
            self.api_key_input.setText(self.provider_config.api_key or "")
            self.api_key_input.setStyleSheet("""
                QLineEdit {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 6px;
                    border-radius: 3px;
                }
            """)
            form.addRow("API Key:", self.api_key_input)

        # Base URL (if needed)
        if self.provider_config.type in ["ollama", "lmstudio", "custom", "openrouter"]:
            self.base_url_input = QLineEdit()
            self.base_url_input.setPlaceholderText("e.g., http://localhost:11434")
            self.base_url_input.setText(self.provider_config.base_url or "")
            self.base_url_input.setStyleSheet("""
                QLineEdit {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 6px;
                    border-radius: 3px;
                }
            """)
            form.addRow("Base URL:", self.base_url_input)

        # Port (if needed)
        if self.provider_config.type in ["lmstudio"]:
            self.port_input = QSpinBox()
            self.port_input.setRange(1, 65535)
            self.port_input.setValue(self.provider_config.port or 1234)
            self.port_input.setStyleSheet("""
                QSpinBox {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 6px;
                    border-radius: 3px;
                }
            """)
            form.addRow("Port:", self.port_input)

        layout.addLayout(form)
        layout.addStretch()

        # Buttons
        button_row = QHBoxLayout()
        button_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #4e4e4e;
            }
        """)
        cancel_btn.clicked.connect(self.reject)
        button_row.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setStyleSheet("""
            QPushButton {
                background: #555555;
                color: #D2D2D2;
                border: 1px solid #666666;
                padding: 6px 16px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #666666;
            }
        """)
        save_btn.clicked.connect(self._save)
        button_row.addWidget(save_btn)

        layout.addLayout(button_row)

    def _save(self):
        """Save configuration and close."""
        # Update provider config
        if hasattr(self, 'api_key_input'):
            self.provider_config.api_key = self.api_key_input.text() or None

        if hasattr(self, 'base_url_input'):
            self.provider_config.base_url = self.base_url_input.text() or None

        if hasattr(self, 'port_input'):
            self.provider_config.port = self.port_input.value()

        self.accept()


class ModelRow(QFrame):
    """Single model row showing model name, label dropdown, and actions."""

    labelChanged = pyqtSignal(str, str, str)  # provider_id, model_name, label
    deleteRequested = pyqtSignal(str)  # model_name

    def __init__(self, provider_id: str, model_name: str, parent=None):
        super().__init__(parent)
        self.provider_id = provider_id
        self.model_name = model_name

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ModelRow {
                background: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                padding: 6px;
            }
            ModelRow:hover {
                border: 1px solid #555555;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Build row UI."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Model name
        name_label = QLabel(self.model_name)
        name_label.setStyleSheet("color: #D2D2D2; font-size: 11px;")
        layout.addWidget(name_label)

        layout.addStretch()

        # "Use as" dropdown
        self.label_combo = QComboBox()
        self.label_combo.setStyleSheet("""
            QComboBox {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 4px 8px 4px 8px;
                padding-right: 24px;
                border-radius: 3px;
                min-width: 80px;
            }
            QComboBox:hover {
                border: 1px solid #666666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: 1px solid #555555;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #888888;
                width: 0px;
                height: 0px;
                margin-right: 6px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #aaaaaa;
            }
            QComboBox QAbstractItemView {
                background: #3e3e3e;
                color: #D2D2D2;
                selection-background-color: #555555;
                selection-color: #FFFFFF;
                border: 1px solid #555555;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 12px;
                border: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #4e4e4e;
                color: #FFFFFF;
            }
            QComboBox QAbstractItemView::item:selected {
                background: #555555;
                color: #FFFFFF;
            }
            QComboBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)
        self.label_combo.currentTextChanged.connect(self._on_label_changed)
        layout.addWidget(self.label_combo)

        # Delete button (only for ollama)
        if self.provider_id == "ollama":
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet("""
                QPushButton {
                    background: #3e3e3e;
                    color: #D2D2D2;
                    border: 1px solid #555555;
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 10px;
                }
                QPushButton:hover {
                    background: #4e4e4e;
                }
            """)
            delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self.model_name))
            layout.addWidget(delete_btn)

    def update_labels(self, all_labels: List[str], current_label: Optional[str]):
        """Update dropdown with available labels."""
        print(f"DEBUG update_labels: model={self.model_name}, current_label='{current_label}' (type: {type(current_label).__name__})")
        print(f"DEBUG: Signals currently blocked? {self.label_combo.signalsBlocked()}")

        was_blocked = self.label_combo.signalsBlocked()
        self.label_combo.blockSignals(True)

        try:
            self.label_combo.clear()
            self.label_combo.addItem("(None)")
            self.label_combo.addItem("(Apply to All Labels)")
            self.label_combo.addItems(all_labels)

            # Set the selection based on current_label (from database)
            # Treat string "None" as unassigned (legacy data)
            if current_label and current_label != "None":
                index = self.label_combo.findText(current_label)
                print(f"DEBUG: Found '{current_label}' at index {index}")
                if index >= 0:
                    self.label_combo.setCurrentIndex(index)
                    print(f"DEBUG: Set index to {index}, currentText is now '{self.label_combo.currentText()}'")
                else:
                    # Fallback to (None) if label not found
                    print(f"DEBUG: Label '{current_label}' not found in dropdown, setting to (None)")
                    self.label_combo.setCurrentIndex(0)
            else:
                # No label assigned - show (None)
                print(f"DEBUG: No label assigned (or legacy 'None'), setting to (None)")
                self.label_combo.setCurrentIndex(0)
        finally:
            # ALWAYS unblock signals, even if there was an error
            self.label_combo.blockSignals(was_blocked)
            print(f"DEBUG: Signals unblocked, now blocked? {self.label_combo.signalsBlocked()}")

        # Force visual update
        self.label_combo.update()
        print(f"DEBUG: Final currentText = '{self.label_combo.currentText()}'")

    def _on_label_changed(self, label_text: str):
        """Handle label selection."""
        print(f"DEBUG ModelRow._on_label_changed: Dropdown changed to '{label_text}' for {self.provider_id}/{self.model_name}")

        # Check for special options
        if label_text == "(Apply to All Labels)":
            # Signal to apply this model to all labels
            print(f"DEBUG: Emitting __APPLY_TO_ALL__")
            self.labelChanged.emit(self.provider_id, self.model_name, "__APPLY_TO_ALL__")
        elif label_text == "(None)":
            # Clear assignment
            print(f"DEBUG: Emitting clear (empty string)")
            self.labelChanged.emit(self.provider_id, self.model_name, "")
        else:
            # Regular label assignment
            print(f"DEBUG: Emitting label '{label_text}'")
            self.labelChanged.emit(self.provider_id, self.model_name, label_text)


class DownloadProgressRow(QFrame):
    """Download progress row for Ollama models."""

    cancelRequested = pyqtSignal(str)  # model_name

    def __init__(self, model_name: str, parent=None):
        super().__init__(parent)
        self.model_name = model_name

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            DownloadProgressRow {
                background: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 3px;
                padding: 6px;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Build progress UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Top row: name + cancel button
        top_row = QHBoxLayout()

        name_label = QLabel(self.model_name)
        name_label.setStyleSheet("color: #D2D2D2; font-size: 11px; font-weight: bold;")
        top_row.addWidget(name_label)

        top_row.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet("""
            QPushButton {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 10px;
            }
            QPushButton:hover {
                background: #4e4e4e;
            }
        """)
        cancel_btn.clicked.connect(lambda: self.cancelRequested.emit(self.model_name))
        top_row.addWidget(cancel_btn)

        layout.addLayout(top_row)

        # Progress label
        self.progress_label = QLabel("Starting download...")
        self.progress_label.setStyleSheet("color: #888888; font-size: 10px;")
        layout.addWidget(self.progress_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #3e3e3e;
                border-radius: 2px;
                background: #1a1a1a;
                height: 14px;
            }
            QProgressBar::chunk {
                background: #555555;
                border-radius: 1px;
            }
        """)
        self.progress_bar.setTextVisible(False)
        layout.addWidget(self.progress_bar)

    def update_progress(self, downloaded_mb: float, total_mb: float, percent: float, speed_mbps: float):
        """Update progress display."""
        total_gb = total_mb / 1024
        progress_text = f"{downloaded_mb:.2f} MB / {total_gb:.2f} GB ({percent:.1f}%) | {speed_mbps:.2f} MB/s"
        self.progress_label.setText(progress_text)
        self.progress_bar.setValue(int(percent))


class ModelManagerPanel(QWidget):
    """
    Model Manager panel - multi-provider model configuration.

    Layout:
    - Provider selector + configure
    - Search field
    - Model browser (provider-specific)
    - Label assignments overview
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.provider_manager = get_provider_manager()
        self.label_manager = get_model_label_manager()

        self.model_rows: Dict[str, ModelRow] = {}
        self.download_rows: Dict[str, DownloadProgressRow] = {}

        self._setup_ui()

        # Connect signals
        self.provider_manager.providersChanged.connect(self._refresh_provider_dropdown)
        self.label_manager.mappingsChanged.connect(self._refresh_label_assignments)

        # Refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self._refresh_models)
        self.refresh_timer.start(2000)  # Every 2 seconds

        # Initial load
        QTimer.singleShot(100, self._refresh_models)

    def _setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # === Top section: Provider selector ===
        provider_row = QHBoxLayout()

        provider_label = QLabel("Provider:")
        provider_label.setStyleSheet("color: #D2D2D2; font-weight: bold;")
        provider_row.addWidget(provider_label)

        self.provider_combo = QComboBox()
        self.provider_combo.setStyleSheet("""
            QComboBox {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px 12px 6px 12px;
                padding-right: 32px;
                border-radius: 3px;
                min-width: 180px;
            }
            QComboBox:hover {
                border: 1px solid #666666;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 24px;
                border-left: 1px solid #555555;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #888888;
                width: 0px;
                height: 0px;
                margin-right: 8px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #aaaaaa;
            }
            QComboBox QAbstractItemView {
                background: #3e3e3e;
                color: #D2D2D2;
                selection-background-color: #555555;
                selection-color: #FFFFFF;
                border: 1px solid #555555;
                outline: none;
            }
            QComboBox QAbstractItemView::item {
                padding: 6px 12px;
                border: none;
            }
            QComboBox QAbstractItemView::item:hover {
                background: #4e4e4e;
                color: #FFFFFF;
            }
            QComboBox QAbstractItemView::item:selected {
                background: #555555;
                color: #FFFFFF;
            }
            QComboBox::indicator {
                width: 0px;
                height: 0px;
            }
        """)
        self.provider_combo.currentTextChanged.connect(self._on_provider_changed)
        provider_row.addWidget(self.provider_combo)

        configure_btn = QPushButton("Configure")
        configure_btn.setStyleSheet("""
            QPushButton {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #4e4e4e;
            }
        """)
        configure_btn.clicked.connect(self._configure_provider)
        provider_row.addWidget(configure_btn)

        provider_row.addStretch()

        # Disk space (for Ollama)
        self.disk_space_label = QLabel()
        self.disk_space_label.setStyleSheet("color: #888888; font-size: 11px;")
        provider_row.addWidget(self.disk_space_label)

        layout.addLayout(provider_row)

        # === Search field ===
        search_row = QHBoxLayout()

        search_label = QLabel("Search:")
        search_label.setStyleSheet("color: #888888;")
        search_row.addWidget(search_label)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Filter models...")
        self.search_input.setStyleSheet("""
            QLineEdit {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px;
                border-radius: 3px;
            }
        """)
        self.search_input.textChanged.connect(self._filter_models)
        search_row.addWidget(self.search_input)

        refresh_btn = QPushButton("Refresh Models")
        refresh_btn.setStyleSheet("""
            QPushButton {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 6px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #4e4e4e;
            }
        """)
        refresh_btn.clicked.connect(self._force_refresh_models)
        search_row.addWidget(refresh_btn)

        layout.addLayout(search_row)

        # === Model list (scrollable) ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")

        self.models_container = QWidget()
        self.models_layout = QVBoxLayout(self.models_container)
        self.models_layout.setContentsMargins(0, 0, 0, 0)
        self.models_layout.setSpacing(4)
        self.models_layout.addStretch()

        scroll.setWidget(self.models_container)
        layout.addWidget(scroll, stretch=1)

        # === Separator ===
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("QFrame { color: #555555; background-color: #555555; height: 2px; }")
        layout.addWidget(separator)

        # === Label assignments section ===
        assignments_header = QHBoxLayout()

        assignments_label = QLabel("Label Assignments (All Providers)")
        assignments_label.setStyleSheet("color: #888888; font-size: 12px; font-weight: bold;")
        assignments_header.addWidget(assignments_label)

        assignments_header.addStretch()

        # Add custom label button
        add_label_btn = QPushButton("+ Add Label")
        add_label_btn.setStyleSheet("""
            QPushButton {
                background: #3e3e3e;
                color: #888888;
                border: 1px solid #555555;
                padding: 4px 12px;
                border-radius: 3px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #4e4e4e;
                color: #aaaaaa;
                border: 1px solid #666666;
            }
        """)
        add_label_btn.clicked.connect(self._add_custom_label)
        assignments_header.addWidget(add_label_btn)

        layout.addLayout(assignments_header)

        self.assignments_container = QWidget()
        self.assignments_layout = QVBoxLayout(self.assignments_container)
        self.assignments_layout.setContentsMargins(0, 0, 0, 0)
        self.assignments_layout.setSpacing(4)
        layout.addWidget(self.assignments_container)

        # Populate provider dropdown
        self._refresh_provider_dropdown()

    def _refresh_provider_dropdown(self):
        """Refresh provider dropdown with all configured providers."""
        self.provider_combo.blockSignals(True)

        current_provider = self.provider_combo.currentData()

        self.provider_combo.clear()

        for provider_id in self.provider_manager.get_all_provider_ids():
            provider = self.provider_manager.get_provider(provider_id)
            if provider:
                self.provider_combo.addItem(provider.name, provider.id)

        # Restore selection
        if current_provider:
            index = self.provider_combo.findData(current_provider)
            if index >= 0:
                self.provider_combo.setCurrentIndex(index)

        self.provider_combo.blockSignals(False)

    def _on_provider_changed(self):
        """Handle provider selection change."""
        self._refresh_models()

    def _configure_provider(self):
        """Open provider configuration dialog."""
        provider_id = self.provider_combo.currentData()
        if not provider_id:
            return

        provider = self.provider_manager.get_provider(provider_id)
        if not provider:
            return

        dialog = ProviderConfigDialog(provider, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Save updated config
            self.provider_manager.update_provider(provider)
            QMessageBox.information(self, "Provider Configured",
                                    f"{provider.name} configuration saved.")

    def _refresh_models(self):
        """Refresh model list for current provider."""
        provider_id = self.provider_combo.currentData()
        if not provider_id:
            return

        provider = self.provider_manager.get_provider(provider_id)
        if not provider:
            return

        # Update disk space for Ollama
        if provider_id == "ollama":
            self._update_disk_space()
        else:
            self.disk_space_label.setText("")

        # Fetch models
        models = self.provider_manager.fetch_available_models(provider_id)

        # Get all labels
        all_labels = sorted(self.label_manager.get_all_labels())

        # Check if model list has actually changed
        existing_models = set(self.model_rows.keys())
        new_models = set(models)

        # Only recreate widgets if the model list changed
        if existing_models != new_models:
            # Clear existing rows
            for row in list(self.model_rows.values()):
                self.models_layout.removeWidget(row)
                row.deleteLater()
            self.model_rows.clear()

            for row in list(self.download_rows.values()):
                self.models_layout.removeWidget(row)
                row.deleteLater()
            self.download_rows.clear()

            # Add model rows
            for model_name in models:
                row = ModelRow(provider_id, model_name)
                row.labelChanged.connect(self._on_label_changed)
                row.deleteRequested.connect(self._delete_model)

                # Find current label assignment
                current_label = self.label_manager.get_label_for_model(provider_id, model_name)
                row.update_labels(all_labels, current_label)

                self.models_layout.insertWidget(self.models_layout.count() - 1, row)
                self.model_rows[model_name] = row
        else:
            # Model list hasn't changed - just update existing dropdowns
            # BUT only if their label assignment has actually changed
            for model_name, row in self.model_rows.items():
                current_label = self.label_manager.get_label_for_model(provider_id, model_name)

                # Check if dropdown is already showing the correct value
                current_display = row.label_combo.currentText()
                expected_display = current_label if current_label else "(None)"

                # Only update if the display is wrong
                if current_display != expected_display:
                    print(f"DEBUG _refresh_models: Updating {model_name} dropdown: '{current_display}' -> '{expected_display}'")
                    row.update_labels(all_labels, current_label)

        # For Ollama: Add downloads section
        if provider_id == "ollama":
            downloads = self._check_active_downloads()
            if downloads:
                # Add separator
                sep = QFrame()
                sep.setFrameShape(QFrame.Shape.HLine)
                sep.setStyleSheet("QFrame { color: #444444; background-color: #444444; height: 1px; }")
                self.models_layout.insertWidget(self.models_layout.count() - 1, sep)

                # Add "Downloading" label
                dl_label = QLabel("<i>Downloading</i>")
                dl_label.setStyleSheet("color: #888888; font-size: 10px;")
                self.models_layout.insertWidget(self.models_layout.count() - 1, dl_label)

                # Add download rows
                for download in downloads:
                    row = DownloadProgressRow(download["name"])
                    row.cancelRequested.connect(self._cancel_download)
                    row.update_progress(
                        download.get("downloaded_mb", 0),
                        download.get("total_mb", 1),
                        download.get("progress", 0),
                        download.get("speed_mbps", 0)
                    )

                    self.models_layout.insertWidget(self.models_layout.count() - 1, row)
                    self.download_rows[download["name"]] = row

        # Apply search filter
        self._filter_models()

        # Refresh label assignments
        self._refresh_label_assignments()

    def _force_refresh_models(self):
        """Force model list refresh from API."""
        provider_id = self.provider_combo.currentData()
        if not provider_id:
            return

        # Show loading message
        self.search_input.setPlaceholderText("Fetching models from API...")

        # Fetch models (will update cache)
        models = self.provider_manager.fetch_available_models(provider_id)

        self.search_input.setPlaceholderText("Filter models...")

        # Refresh display
        self._refresh_models()

        QMessageBox.information(self, "Models Refreshed",
                                f"Found {len(models)} models from {provider_id}")

    def _filter_models(self):
        """Filter model rows based on search text."""
        search_text = self.search_input.text().lower()

        for model_name, row in self.model_rows.items():
            visible = search_text in model_name.lower()
            row.setVisible(visible)

    def _on_label_changed(self, provider_id: str, model_name: str, label: str):
        """Handle label assignment change."""
        # Check for "Apply to All Labels" special option
        if label == "__APPLY_TO_ALL__":
            # Show confirmation dialog
            reply = QMessageBox.question(
                self,
                "Apply to All Labels",
                f"This will set ALL labels (Small, Medium, Large, and custom labels) to use:\n\n"
                f"{provider_id} / {model_name}\n\n"
                f"This will affect all facets and components using these labels.\n\n"
                f"Are you sure?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Apply to all labels
                for lbl in self.label_manager.get_all_labels():
                    self.label_manager.set_model_for_label(lbl, provider_id, model_name)

                # Full refresh needed since all labels changed
                self._refresh_models()
            else:
                # User cancelled - refresh to reset the dropdown
                self._refresh_models()

            return

        if label:
            print(f"DEBUG _on_label_changed: User selected '{label}' for {provider_id}/{model_name}")

            # Check if label is already assigned to a different model
            current_provider, current_model = self.label_manager.get_model_for_label(label)
            print(f"DEBUG: Label '{label}' currently assigned to: {current_provider}/{current_model}")

            if current_provider and current_model and (current_provider != provider_id or current_model != model_name):
                # Label is being reassigned - show impact warning
                affected_facets = self._find_facets_using_label(label)

                if affected_facets:
                    facet_list = "\n".join([f"  • {f}" for f in affected_facets[:10]])
                    if len(affected_facets) > 10:
                        facet_list += f"\n  ... and {len(affected_facets) - 10} more"

                    message = (
                        f"Changing '{label}' from\n"
                        f"  {current_provider} / {current_model}\n"
                        f"to\n"
                        f"  {provider_id} / {model_name}\n\n"
                        f"This will affect:\n\n"
                        f"{facet_list}\n\n"
                        f"Continue?"
                    )

                    reply = QMessageBox.question(
                        self,
                        "Confirm Label Change",
                        message,
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.Yes
                    )

                    if reply != QMessageBox.StandardButton.Yes:
                        # User cancelled - refresh to reset dropdown
                        self._refresh_models()
                        return

            self.label_manager.set_model_for_label(label, provider_id, model_name)
            print(f"DEBUG: Saved label '{label}' for {provider_id}/{model_name}")

            # Verify immediately after save
            verify = self.label_manager.get_label_for_model(provider_id, model_name)
            print(f"DEBUG: Immediate verification -> '{verify}'")
        else:
            # Clear assignment - check if this is the last one
            current_label = self.label_manager.get_label_for_model(provider_id, model_name)
            if current_label:
                # Check if this would leave no assigned labels
                assigned_labels = [lbl for lbl in self.label_manager.get_all_labels()
                                  if self.label_manager.get_model_for_label(lbl)[0]]

                if len(assigned_labels) == 1 and current_label in assigned_labels:
                    QMessageBox.warning(
                        self,
                        "Cannot Clear",
                        f"Cannot clear '{current_label}' because it's the only label assigned to a model.\n\n"
                        f"The system must have at least one LLM configured."
                    )
                    self._refresh_models()
                    return

                self.label_manager.set_model_for_label(current_label, None, None)

        # Delay refresh slightly to allow dropdown to close naturally
        # This prevents the dropdown from closing while user is still interacting
        QTimer.singleShot(200, lambda: self._delayed_refresh_after_label_change(provider_id, model_name))

    def _delayed_refresh_after_label_change(self, provider_id: str, model_name: str):
        """Delayed refresh after label change to avoid closing dropdown prematurely."""
        # Refresh label assignments section
        self._refresh_label_assignments()

        # Update ALL model rows including the one that just changed
        # By this point (after 500ms delay), the dropdown has closed naturally
        all_labels = sorted(self.label_manager.get_all_labels())
        for row_model_name, row in self.model_rows.items():
            # Get current label for this model
            current = self.label_manager.get_label_for_model(row.provider_id, row_model_name)
            # Update the row's dropdown to show the correct selection
            row.update_labels(all_labels, current)

    def _rename_label(self, old_label: str):
        """Rename a label."""
        from PyQt6.QtWidgets import QInputDialog

        # Don't allow renaming default labels
        if old_label in ["Small", "Medium", "Large"]:
            QMessageBox.information(
                self,
                "Cannot Rename",
                f"The default label '{old_label}' cannot be renamed."
            )
            return

        new_label, ok = QInputDialog.getText(
            self,
            "Rename Label",
            f"Rename '{old_label}' to:",
            text=old_label
        )

        if ok and new_label:
            # Convert to title case and strip whitespace
            new_label = new_label.strip().title()

            # Validate: alphanumeric, spaces, and underscores only
            if not all(c.isalnum() or c in ' _' for c in new_label):
                QMessageBox.warning(
                    self,
                    "Invalid Label",
                    "Label names must contain only letters, numbers, spaces, and underscores."
                )
                return

            # Check if new name already exists
            if new_label in self.label_manager.get_all_labels() and new_label != old_label:
                QMessageBox.information(
                    self,
                    "Label Exists",
                    f"Label '{new_label}' already exists."
                )
                return

            # Get the current assignment
            provider, model = self.label_manager.get_model_for_label(old_label)

            # Delete old label and create new one with same assignment
            self.label_manager.delete_label(old_label)
            if provider and model:
                self.label_manager.set_model_for_label(new_label, provider, model)
            else:
                self.label_manager.create_label(new_label)

            # Refresh UI
            self._refresh_models()

    def _add_custom_label(self):
        """Add a custom label category."""
        from PyQt6.QtWidgets import QInputDialog

        label_name, ok = QInputDialog.getText(
            self,
            "Add Custom Label",
            "Enter label name (e.g., Reasoning, Creative, etc.):",
            text=""
        )

        if ok and label_name:
            # Convert to title case and strip whitespace
            label_name = label_name.strip().title()

            # Validate: alphanumeric, spaces, and underscores only
            if not all(c.isalnum() or c in ' _' for c in label_name):
                QMessageBox.warning(
                    self,
                    "Invalid Label",
                    "Label names must contain only letters, numbers, spaces, and underscores."
                )
                return

            # Check if already exists
            existing_labels = self.label_manager.get_all_labels()
            if label_name in existing_labels:
                QMessageBox.information(
                    self,
                    "Label Exists",
                    f"Label '{label_name}' already exists."
                )
                return

            # Add the label (unassigned initially)
            self.label_manager.create_label(label_name)

            # Refresh models to update dropdowns
            self._refresh_models()

            QMessageBox.information(
                self,
                "Label Added",
                f"Label '{label_name}' has been added.\n\nYou can now assign models to it using the dropdown menus."
            )

    def _refresh_label_assignments(self):
        """Refresh label assignments overview."""
        # Clear existing
        while self.assignments_layout.count():
            item = self.assignments_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Get all mappings
        mappings = self.label_manager.get_all_mappings()

        if not mappings:
            no_label = QLabel("No label assignments yet")
            no_label.setStyleSheet("color: #666666; font-style: italic; font-size: 11px;")
            self.assignments_layout.addWidget(no_label)
            return

        # Add assignment rows
        for label, (provider_id, model_name) in sorted(mappings.items()):
            provider = self.provider_manager.get_provider(provider_id)
            provider_name = provider.name if provider else provider_id

            row = QFrame()
            row.setStyleSheet("""
                QFrame {
                    background: #2d2d2d;
                    border: 1px solid #3e3e3e;
                    border-radius: 3px;
                    padding: 6px;
                }
            """)

            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(8, 4, 8, 4)

            # Make label clickable for renaming
            label_widget = QPushButton(f"{label}")
            label_widget.setFlat(True)
            label_widget.setCursor(Qt.CursorShape.PointingHandCursor)
            label_widget.setStyleSheet("""
                QPushButton {
                    color: #D2D2D2;
                    font-size: 11px;
                    font-weight: bold;
                    background: transparent;
                    border: none;
                    text-align: left;
                    padding: 0px;
                }
                QPushButton:hover {
                    color: #FFFFFF;
                    text-decoration: underline;
                }
            """)
            label_widget.setToolTip("Double-click to rename")
            label_widget.clicked.connect(lambda checked, lbl=label: self._rename_label(lbl))
            row_layout.addWidget(label_widget)

            arrow = QLabel("→")
            arrow.setStyleSheet("color: #666666;")
            row_layout.addWidget(arrow)

            info = QLabel(f"{provider_name} / {model_name}")
            info.setStyleSheet("color: #888888; font-size: 11px;")
            row_layout.addWidget(info)

            row_layout.addStretch()

            # Delete button (only for custom labels, not defaults)
            if label not in ["Small", "Medium", "Large"]:
                delete_btn = QPushButton("×")
                delete_btn.setFixedSize(20, 20)
                delete_btn.setCursor(Qt.CursorShape.PointingHandCursor)
                delete_btn.setStyleSheet("""
                    QPushButton {
                        background: #3e3e3e;
                        color: #888888;
                        border: 1px solid #555555;
                        border-radius: 3px;
                        font-size: 16px;
                        font-weight: bold;
                        padding: 0px;
                    }
                    QPushButton:hover {
                        background: #8b0000;
                        color: #FFFFFF;
                        border: 1px solid #aa0000;
                    }
                """)
                delete_btn.setToolTip(f"Delete label '{label}'")
                delete_btn.clicked.connect(lambda checked, lbl=label: self._delete_label(lbl))
                row_layout.addWidget(delete_btn)

            self.assignments_layout.addWidget(row)

    def _delete_label(self, label: str):
        """Delete a custom label with safety checks."""
        # Check if this would leave no assigned labels
        assigned_labels = [lbl for lbl in self.label_manager.get_all_labels()
                          if self.label_manager.get_model_for_label(lbl)[0]]  # Has provider

        if len(assigned_labels) == 1 and label in assigned_labels:
            QMessageBox.warning(
                self,
                "Cannot Delete",
                f"Cannot delete '{label}' because it's the only label assigned to a model.\n\n"
                f"The system must have at least one LLM configured.\n\n"
                f"Assign another label first, then delete this one."
            )
            return

        # Find which facets use this label
        affected_facets = self._find_facets_using_label(label)

        # Build warning message
        if affected_facets:
            facet_list = "\n".join([f"  • {f}" for f in affected_facets[:10]])  # Show max 10
            if len(affected_facets) > 10:
                facet_list += f"\n  ... and {len(affected_facets) - 10} more"

            message = (
                f"Deleting label '{label}' will affect the following:\n\n"
                f"{facet_list}\n\n"
                f"These facets will need to be updated with a different label.\n\n"
                f"Are you sure you want to delete this label?"
            )
        else:
            message = f"Delete label '{label}'?\n\nThis label is not currently used by any facets."

        reply = QMessageBox.question(
            self,
            "Delete Label",
            message,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.label_manager.delete_label(label)
            self._refresh_models()

    def _find_facets_using_label(self, label: str) -> List[str]:
        """Find all facets that reference this label."""
        affected = []

        # Search through all YAML files in facet_assemblies
        import glob
        import yaml
        from pathlib import Path

        # Check applications/noodlestudio/facet_assemblies/
        studio_path = Path(__file__).parent.parent / "facet_assemblies"
        if studio_path.exists():
            for yaml_file in studio_path.glob("**/*.yaml"):
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f)
                        if data and 'facets' in data:
                            for facet in data['facets']:
                                if facet.get('model') == label or facet.get('properties', {}).get('model') == label:
                                    facet_name = f"{yaml_file.stem}: {facet.get('name', facet.get('id'))}"
                                    affected.append(facet_name)
                except:
                    pass

        return affected

    def _delete_model(self, model_name: str):
        """Delete an Ollama model."""
        reply = QMessageBox.question(
            self, "Delete Model",
            f"Delete {model_name}?\n\nThis will free disk space but require re-download if used again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                env = os.environ.copy()
                env["OLLAMA_MODELS"] = "/Volumes/DOUBLETROUBLE/models"

                result = subprocess.run(
                    ["ollama", "rm", model_name],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    env=env
                )

                if result.returncode == 0:
                    self._refresh_models()
                else:
                    QMessageBox.warning(self, "Delete Failed", f"Error: {result.stderr}")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to delete model: {e}")

    def _cancel_download(self, model_name: str):
        """Cancel Ollama model download."""
        try:
            subprocess.run(["pkill", "-f", f"ollama pull {model_name}"], timeout=5)
            self._refresh_models()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to cancel download: {e}")

    def _check_active_downloads(self) -> List[Dict]:
        """Check for active Ollama downloads."""
        downloads = []

        try:
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=2
            )

            for line in result.stdout.split('\n'):
                if 'ollama pull' in line:
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'pull' and i + 1 < len(parts):
                            model_name = parts[i + 1]
                            downloads.append({
                                'name': model_name,
                                'downloaded_mb': 0,
                                'total_mb': 100,
                                'progress': 0,
                                'speed_mbps': 0
                            })
                            break

        except Exception as e:
            print(f"Error checking downloads: {e}")

        return downloads

    def _update_disk_space(self):
        """Update disk space label for Ollama."""
        try:
            models_path = Path("/Volumes/DOUBLETROUBLE/models")
            if models_path.exists():
                stat = shutil.disk_usage(models_path)
                free_gb = stat.free / (1024**3)
                total_gb = stat.total / (1024**3)
                used_percent = (stat.used / stat.total) * 100
                self.disk_space_label.setText(
                    f"Free: {free_gb:.1f} GB / {total_gb:.1f} GB ({used_percent:.0f}% used)"
                )
            else:
                self.disk_space_label.setText("Models volume not found")
        except:
            self.disk_space_label.setText("")
