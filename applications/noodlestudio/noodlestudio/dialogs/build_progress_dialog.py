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
#   Build Progress Dialog
#
#   Shows progress while building a standalone application.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.dialogs.build_progress_dialog
# PURPOSE:  Build progress UI with cancel support
# LAYER:    Studio / Dialogs
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   BuildProgressDialog, BuildWorker
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
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar,
    QPushButton, QWidget, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..appbuilder.builder import Builder, BuildResult
from ..core.build_config import BuildConfig

logger = logging.getLogger(__name__)


class BuildWorker(QThread):
    """
    Background worker thread for building applications.

    Emits signals for progress, completion, and errors.

    Signals:
        progress(int, str): Progress percentage (0-100) and status message
        finished(BuildResult): Build completed (check result.success)
        error(str): Fatal error during build
    """

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(object)  # BuildResult
    error = pyqtSignal(str)

    def __init__(
        self,
        config: BuildConfig,
        project_path: Path,
        parent: Optional[QThread] = None
    ):
        """
        Initialize the build worker.

        Args:
            config: Build configuration
            project_path: Path to the project directory
            parent: Parent QThread
        """
        super().__init__(parent)
        self.config = config
        self.project_path = Path(project_path)
        self._builder: Optional[Builder] = None

    def run(self):
        """Execute the build in background thread."""
        try:
            self._builder = Builder(self.config, self.project_path)
            self._builder.on_progress(self._on_progress)

            result = self._builder.build()
            self.finished.emit(result)

        except Exception as e:
            logger.exception(f"Build worker error: {e}")
            self.error.emit(str(e))

    def _on_progress(self, percent: int, message: str):
        """Forward progress to signal."""
        self.progress.emit(percent, message)

    def cancel(self):
        """Request build cancellation."""
        if self._builder:
            self._builder.cancel()


class BuildProgressDialog(QDialog):
    """
    Dialog showing build progress with cancel option.

    Usage:
        dialog = BuildProgressDialog(config, project_path, parent=self)
        dialog.build_completed.connect(self._on_build_complete)
        dialog.exec()

    Signals:
        build_completed(BuildResult): Emitted when build finishes
    """

    build_completed = pyqtSignal(object)  # BuildResult

    def __init__(
        self,
        config: BuildConfig,
        project_path: Path,
        parent: Optional[QWidget] = None,
        run_after_build: bool = False
    ):
        """
        Initialize the build progress dialog.

        Args:
            config: Build configuration
            project_path: Path to the project directory
            parent: Parent widget
            run_after_build: If True, launch the app after successful build
        """
        super().__init__(parent)
        self.config = config
        self.project_path = Path(project_path)
        self.run_after_build = run_after_build
        self._worker: Optional[BuildWorker] = None
        self._result: Optional[BuildResult] = None

        app_name = config.identity.name if hasattr(config, 'identity') else "App"
        self.setWindowTitle(f"Building: {app_name}")
        self.setModal(True)
        self.setFixedSize(450, 200)

        # Remove close button to prevent accidental close
        self.setWindowFlags(
            self.windowFlags()
            & ~Qt.WindowType.WindowContextHelpButtonHint
            & ~Qt.WindowType.WindowCloseButtonHint
        )

        self._build_ui()

    def _build_ui(self):
        """Build the dialog UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)

        # App name header
        app_name = self.config.identity.name if hasattr(self.config, 'identity') else "App"
        header = QLabel(f"Building: {app_name}")
        header.setStyleSheet("color: #ffffff; font-size: 15px; font-weight: bold;")
        layout.addWidget(header)

        # Progress bar
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setTextVisible(True)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444444;
                border-radius: 4px;
                background: #2a2a2a;
                height: 20px;
                text-align: center;
                color: #ffffff;
            }
            QProgressBar::chunk {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #4a4a4a, stop: 1 #666666
                );
                border-radius: 3px;
            }
        """)
        layout.addWidget(self._progress_bar)

        # Status label
        self._status_label = QLabel("Preparing build...")
        self._status_label.setStyleSheet("color: #cccccc; font-size: 12px;")
        layout.addWidget(self._status_label)

        # Detail label (current file being processed)
        self._detail_label = QLabel("")
        self._detail_label.setStyleSheet("color: #888888; font-size: 11px;")
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

        layout.addStretch()

        # Cancel button
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setStyleSheet("""
            QPushButton {
                padding: 8px 20px;
                background: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 4px;
                color: #cccccc;
            }
            QPushButton:hover {
                background: #4a4a4a;
            }
            QPushButton:disabled {
                background: #2a2a2a;
                color: #666666;
            }
        """)
        self._cancel_btn.clicked.connect(self._on_cancel)
        button_layout.addWidget(self._cancel_btn)

        layout.addLayout(button_layout)

        # Style the dialog
        self.setStyleSheet("""
            QDialog {
                background: #1e1e1e;
            }
        """)

    def showEvent(self, event):
        """Start build when dialog is shown."""
        super().showEvent(event)
        self._start_build()

    def _start_build(self):
        """Start the build process."""
        self._worker = BuildWorker(self.config, self.project_path)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_progress(self, percent: int, message: str):
        """Handle progress update."""
        self._progress_bar.setValue(percent)
        self._status_label.setText(message)

    def _on_finished(self, result: BuildResult):
        """Handle build completion."""
        self._result = result
        self._worker = None

        if result.success:
            self._progress_bar.setValue(100)
            self._status_label.setText("Build complete!")
            self._cancel_btn.setText("Close")
            self._cancel_btn.clicked.disconnect()
            self._cancel_btn.clicked.connect(self.accept)

            # Show success message
            size_mb = result.total_size_bytes / (1024 * 1024)
            time_s = result.build_time_seconds
            self._detail_label.setText(
                f"Output: {result.output_path}\n"
                f"Size: {size_mb:.1f} MB | Time: {time_s:.1f}s | Files: {result.total_files}"
            )

            # Launch if requested
            if self.run_after_build:
                self._launch_app(result.output_path)
                self.accept()
            else:
                # Emit and allow user to close
                self.build_completed.emit(result)

        else:
            self._status_label.setText("Build failed")
            self._status_label.setStyleSheet("color: #ff6666; font-size: 12px;")
            self._detail_label.setText("\n".join(result.errors))
            self._cancel_btn.setText("Close")
            self._cancel_btn.clicked.disconnect()
            self._cancel_btn.clicked.connect(self.reject)
            self.build_completed.emit(result)

    def _on_error(self, error_msg: str):
        """Handle fatal error."""
        self._worker = None
        self._status_label.setText("Build error")
        self._status_label.setStyleSheet("color: #ff6666; font-size: 12px;")
        self._detail_label.setText(error_msg)
        self._cancel_btn.setText("Close")
        self._cancel_btn.clicked.disconnect()
        self._cancel_btn.clicked.connect(self.reject)

        # Create a failed result
        result = BuildResult()
        result.errors = [error_msg]
        self.build_completed.emit(result)

    def _on_cancel(self):
        """Handle cancel button click."""
        if self._worker and self._worker.isRunning():
            self._cancel_btn.setEnabled(False)
            self._cancel_btn.setText("Cancelling...")
            self._worker.cancel()
            # Wait for worker to finish
            self._worker.wait(5000)  # 5 second timeout
            if self._worker.isRunning():
                self._worker.terminate()
            self._worker = None
        self.reject()

    def _launch_app(self, output_path: Path):
        """Launch the built application."""
        output_path = Path(output_path)
        if not output_path.exists():
            logger.warning(f"Cannot launch - output not found: {output_path}")
            return

        try:
            # macOS: use 'open' command
            import platform
            if platform.system() == 'Darwin':
                subprocess.Popen(['open', str(output_path)])
                logger.info(f"Launched: {output_path}")
            else:
                logger.warning(f"Auto-launch not supported on {platform.system()}")
        except Exception as e:
            logger.error(f"Failed to launch app: {e}")

    def closeEvent(self, event):
        """Handle dialog close."""
        if self._worker and self._worker.isRunning():
            # Don't allow close while building
            event.ignore()
        else:
            super().closeEvent(event)

    def get_result(self) -> Optional[BuildResult]:
        """Get the build result after dialog closes."""
        return self._result


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
