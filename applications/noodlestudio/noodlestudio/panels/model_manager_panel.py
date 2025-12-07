"""
Model Manager Panel for NoodleSTUDIO.

Shows Ollama model status, downloads, and management controls.
"""

import os
import subprocess
import shutil
import json
from pathlib import Path
from typing import Optional, Dict, List
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QProgressBar, QScrollArea, QFrame, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class ModelCard(QFrame):
    """Card widget for a single model."""

    deleteRequested = pyqtSignal(str)  # model_name
    pauseRequested = pyqtSignal(str)  # model_name
    cancelRequested = pyqtSignal(str)  # model_name
    downloadRequested = pyqtSignal(str)  # model_name
    retryRequested = pyqtSignal(str)  # model_name

    def __init__(self, model_name: str, status: str = "downloaded", parent=None):
        super().__init__(parent)
        self.model_name = model_name
        self.status = status  # "downloaded", "downloading", "available", "failed"

        self.progress_bar = None
        self.speed_label = None
        self.button_container = None

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet("""
            ModelCard {
                background: #2d2d2d;
                border: 1px solid #3e3e3e;
                border-radius: 4px;
                padding: 8px;
            }
            ModelCard:hover {
                border: 1px solid #555555;
            }
        """)

        self._setup_ui()

    def _setup_ui(self):
        """Build card UI."""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(12, 8, 12, 8)
        self.main_layout.setSpacing(8)

        # Top row: name + buttons
        top_row = QHBoxLayout()

        # Model name
        self.name_label = QLabel(self.model_name)
        self.name_label.setStyleSheet("color: #D2D2D2; font-weight: bold;")
        top_row.addWidget(self.name_label)

        top_row.addStretch()

        # Button container that we can update
        self.button_container = QWidget()
        self.button_layout = QHBoxLayout(self.button_container)
        self.button_layout.setContentsMargins(0, 0, 0, 0)
        self.button_layout.setSpacing(4)
        top_row.addWidget(self.button_container)

        self.main_layout.addLayout(top_row)

        # Info row: size + last used / progress
        info_row = QHBoxLayout()

        self.info_label = QLabel()
        self.info_label.setStyleSheet("color: #888888; font-size: 11px;")
        info_row.addWidget(self.info_label)

        info_row.addStretch()

        self.status_label = QLabel()
        self.status_label.setStyleSheet("color: #888888; font-size: 11px;")
        info_row.addWidget(self.status_label)

        self.main_layout.addLayout(info_row)

        # Update buttons based on initial status
        self._update_buttons()

    def _update_buttons(self):
        """Update action buttons based on current status."""
        # Clear existing buttons
        while self.button_layout.count():
            item = self.button_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Button styling constant
        button_style = """
            QPushButton {
                background: #3e3e3e;
                color: #D2D2D2;
                border: 1px solid #555555;
                padding: 4px 12px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background: #4e4e4e;
                border: 1px solid #666666;
            }
        """

        # Action buttons (conditional based on status)
        if self.status == "downloaded":
            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet(button_style)
            delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self.model_name))
            self.button_layout.addWidget(delete_btn)

        elif self.status == "downloading":
            cancel_btn = QPushButton("Cancel")
            cancel_btn.setStyleSheet(button_style)
            cancel_btn.clicked.connect(lambda: self.cancelRequested.emit(self.model_name))
            self.button_layout.addWidget(cancel_btn)

        elif self.status == "failed":
            retry_btn = QPushButton("Retry")
            retry_btn.setStyleSheet(button_style)
            retry_btn.clicked.connect(lambda: self.retryRequested.emit(self.model_name))
            self.button_layout.addWidget(retry_btn)

            delete_btn = QPushButton("Delete")
            delete_btn.setStyleSheet(button_style)
            delete_btn.clicked.connect(lambda: self.deleteRequested.emit(self.model_name))
            self.button_layout.addWidget(delete_btn)

        elif self.status == "available":
            download_btn = QPushButton("Download")
            download_btn.setStyleSheet(button_style)
            download_btn.clicked.connect(lambda: self.downloadRequested.emit(self.model_name))
            self.button_layout.addWidget(download_btn)

        # Ensure progress bar and speed label exist if needed
        self._ensure_progress_widgets()

    def _ensure_progress_widgets(self):
        """Ensure progress bar and speed label exist when needed."""
        if self.status in ["downloading", "failed"]:
            # Add progress bar if not present
            if not self.progress_bar:
                self.progress_bar = QProgressBar()
                self.progress_bar.setStyleSheet("""
                    QProgressBar {
                        border: 1px solid #3e3e3e;
                        border-radius: 3px;
                        background: #1a1a1a;
                        height: 20px;
                        text-align: center;
                        color: #D2D2D2;
                    }
                    QProgressBar::chunk {
                        background: #555555;
                        border-radius: 2px;
                    }
                """)
                self.progress_bar.setTextVisible(True)
                self.main_layout.addWidget(self.progress_bar)

            # Add speed label if not present
            if not self.speed_label:
                self.speed_label = QLabel()
                self.speed_label.setStyleSheet("color: #888888; font-size: 10px;")
                self.main_layout.addWidget(self.speed_label)
        else:
            # Remove progress widgets if they exist
            if self.progress_bar:
                self.progress_bar.deleteLater()
                self.progress_bar = None
            if self.speed_label:
                self.speed_label.deleteLater()
                self.speed_label = None

    def set_status(self, new_status: str):
        """Update card status and refresh UI."""
        if self.status != new_status:
            self.status = new_status
            self._update_buttons()

    def update_info(self, size: str = "", last_used: str = ""):
        """Update size and last used info."""
        self.info_label.setText(size)
        self.status_label.setText(last_used)

    def update_progress(self, percent: int, eta: str = "", speed: str = ""):
        """Update download progress."""
        if self.progress_bar:
            self.progress_bar.setValue(percent)
            if eta:
                self.progress_bar.setFormat(f"{percent}% - ETA: {eta}")
            else:
                self.progress_bar.setFormat(f"{percent}%")

        if self.speed_label and speed:
            self.speed_label.setText(f"Speed: {speed}")

    def set_failed(self, error_msg: str = "Download failed"):
        """Mark download as failed."""
        if self.progress_bar:
            self.progress_bar.setStyleSheet("""
                QProgressBar {
                    border: 1px solid #666666;
                    border-radius: 3px;
                    background: #1a1a1a;
                    height: 20px;
                    text-align: center;
                    color: #999999;
                }
                QProgressBar::chunk {
                    background: #3e3e3e;
                    border-radius: 2px;
                }
            """)
            self.progress_bar.setFormat(error_msg)

        if self.speed_label:
            self.speed_label.setText("Download stopped")
            self.speed_label.setStyleSheet("color: #666666; font-size: 10px;")


class ModelManagerPanel(QWidget):
    """
    Model Manager panel showing Ollama status and model downloads.

    Features:
    - List of downloaded models
    - Active download progress bars
    - Download/Delete/Cancel controls
    - Free disk space indicator
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.model_cards: Dict[str, ModelCard] = {}
        self.active_downloads: Dict[str, Dict] = {}  # Track active ollama pull processes
        self.download_processes: Dict[str, subprocess.Popen] = {}  # Track process handles
        self._setup_ui()

        # Poll for updates every 1 second for smoother progress
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(1000)

        # Initial load
        QTimer.singleShot(100, self.refresh_status)

    def _setup_ui(self):
        """Build panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        # Header: title + disk space
        header = QHBoxLayout()

        title = QLabel("Model Manager")
        title.setStyleSheet("color: #D2D2D2; font-size: 16px; font-weight: bold;")
        header.addWidget(title)

        header.addStretch()

        self.disk_space_label = QLabel("Checking disk space...")
        self.disk_space_label.setStyleSheet("color: #888888; font-size: 11px;")
        header.addWidget(self.disk_space_label)

        layout.addLayout(header)

        # Scroll area for model cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea {
                background: transparent;
                border: none;
            }
        """)

        # Container for cards
        self.cards_container = QWidget()
        self.cards_layout = QVBoxLayout(self.cards_container)
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(8)
        self.cards_layout.addStretch()

        scroll.setWidget(self.cards_container)
        layout.addWidget(scroll)

    def refresh_status(self):
        """Refresh model status from Ollama."""
        try:
            # Update disk space
            self._update_disk_space()

            # Get list of models from Ollama
            models = self._get_ollama_models()

            if models is None:
                # Ollama not available, show placeholder
                models = []

            # Check for active downloads
            active_downloads = self._check_active_downloads()

            # Merge active downloads into models list
            for download_info in active_downloads:
                # Check if this model is already in the list (partially downloaded)
                existing = next((m for m in models if m['name'] == download_info['name']), None)
                if not existing:
                    # Add as downloading
                    models.append(download_info)

            # Update or create cards
            existing_models = set(self.model_cards.keys())
            current_models = set(m["name"] for m in models)

            # Remove cards for models that no longer exist
            for model_name in existing_models - current_models:
                card = self.model_cards[model_name]
                self.cards_layout.removeWidget(card)
                card.deleteLater()
                del self.model_cards[model_name]

            # Add or update cards
            for model in models:
                name = model["name"]
                if name not in self.model_cards:
                    # Create new card
                    card = ModelCard(name, model["status"])
                    card.deleteRequested.connect(self.delete_model)
                    card.pauseRequested.connect(self.pause_download)
                    card.cancelRequested.connect(self.cancel_download)
                    card.downloadRequested.connect(self.start_download)
                    card.retryRequested.connect(self.start_download)  # Retry uses same logic as download

                    # Insert before the stretch
                    self.cards_layout.insertWidget(self.cards_layout.count() - 1, card)
                    self.model_cards[name] = card

                # Update card status and info
                card = self.model_cards[name]
                card.set_status(model["status"])

                if model["status"] == "downloaded":
                    card.update_info(model["size"], model.get("last_used", ""))
                elif model["status"] == "downloading":
                    card.update_info(model["size"])
                    card.update_progress(
                        model.get("progress", 0),
                        model.get("eta", ""),
                        model.get("speed", "")
                    )
                elif model["status"] == "failed":
                    card.update_info(model["size"])
                    card.set_failed(model.get("error", "Download failed"))

        except Exception as e:
            print(f"Error refreshing model status: {e}")

    def _get_ollama_models(self):
        """Get list of models from Ollama."""
        try:
            env = os.environ.copy()
            env["OLLAMA_MODELS"] = "/Volumes/DOUBLETROUBLE/models"

            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
                env=env
            )

            if result.returncode != 0:
                return None

            # Parse ollama list output
            lines = result.stdout.strip().split('\n')
            if len(lines) < 2:
                return []

            models = []
            for line in lines[1:]:  # Skip header
                parts = line.split()
                if len(parts) >= 4:
                    name = parts[0]
                    # ID is parts[1], skip it
                    size = f"{parts[2]} {parts[3]}"
                    # Modified time would be parts[4:]

                    models.append({
                        "name": name,
                        "size": size,
                        "status": "downloaded",
                        "last_used": ""  # Could parse from modified time
                    })

            return models

        except Exception as e:
            print(f"Error getting ollama models: {e}")
            return None

    def _check_active_downloads(self):
        """Check for active ollama pull processes and their progress."""
        downloads = []

        try:
            # Use ps to find ollama pull processes
            result = subprocess.run(
                ["ps", "aux"],
                capture_output=True,
                text=True,
                timeout=2
            )

            if result.returncode != 0:
                return downloads

            # Look for "ollama pull" commands in process list
            for line in result.stdout.split('\n'):
                if 'ollama pull' in line:
                    # Extract model name from command line
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == 'pull' and i + 1 < len(parts):
                            model_name = parts[i + 1]
                            # This is actively downloading
                            downloads.append({
                                'name': model_name,
                                'status': 'downloading',
                                'size': 'Unknown',
                                'progress': 0,
                                'eta': 'Calculating...',
                                'speed': '...'
                            })
                            break

        except Exception as e:
            print(f"Error checking active downloads: {e}")

        return downloads

    def _update_disk_space(self):
        """Update disk space label."""
        try:
            # Check disk space on DOUBLETROUBLE volume
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
        except Exception as e:
            self.disk_space_label.setText(f"Disk space: Error")

    def delete_model(self, model_name: str):
        """Delete a downloaded model."""
        reply = QMessageBox.question(
            self,
            "Delete Model",
            f"Are you sure you want to delete {model_name}?\n\nThis will free up disk space but the model will need to be re-downloaded if used again.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
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
                    QMessageBox.information(self, "Delete Model", f"Model {model_name} deleted successfully.")
                    self.refresh_status()
                else:
                    QMessageBox.warning(self, "Delete Model", f"Failed to delete {model_name}:\n{result.stderr}")

            except Exception as e:
                QMessageBox.critical(self, "Delete Model", f"Error deleting model:\n{str(e)}")

    def pause_download(self, model_name: str):
        """Pause a model download."""
        # TODO: Implement pause (may need custom download manager)
        print(f"Pause download: {model_name}")
        QMessageBox.information(self, "Pause Download", "Pause not yet implemented.")

    def cancel_download(self, model_name: str):
        """Cancel a model download."""
        reply = QMessageBox.question(
            self,
            "Cancel Download",
            f"Are you sure you want to cancel downloading {model_name}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            try:
                # Kill the process if we're tracking it
                if model_name in self.download_processes:
                    process = self.download_processes[model_name]
                    process.terminate()
                    process.wait(timeout=5)
                    del self.download_processes[model_name]

                # Also try to kill any ollama pull process for this model
                subprocess.run(
                    ["pkill", "-f", f"ollama pull {model_name}"],
                    timeout=5
                )

                self.refresh_status()

            except Exception as e:
                QMessageBox.critical(self, "Cancel Download", f"Error cancelling download:\n{str(e)}")

    def start_download(self, model_name: str):
        """Start downloading a model."""
        try:
            env = os.environ.copy()
            env["OLLAMA_MODELS"] = "/Volumes/DOUBLETROUBLE/models"

            # Start download process in background
            process = subprocess.Popen(
                ["ollama", "pull", model_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env
            )

            # Track the process
            self.download_processes[model_name] = process

            # Update UI immediately
            self.refresh_status()

        except Exception as e:
            QMessageBox.critical(self, "Start Download", f"Error starting download:\n{str(e)}")
