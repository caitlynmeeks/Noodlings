# ──────────────────────────────────────────────────────────────
# Build Progress Dialog Tests
# ──────────────────────────────────────────────────────────────

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PyQt6.QtCore import Qt

from noodlestudio.core.build_config import BuildConfig
from noodlestudio.appbuilder.builder import Builder, BuildResult
from noodlestudio.dialogs.build_progress_dialog import (
    BuildProgressDialog, BuildWorker
)


# ──────────────────────────────────────────────────────────────
# BuildResult Tests
# ──────────────────────────────────────────────────────────────

class TestBuildResult:
    """Tests for BuildResult dataclass."""

    def test_default_values(self):
        """Test default BuildResult values."""
        result = BuildResult()
        assert result.success is False
        assert result.output_path == Path()
        assert result.errors == []
        assert result.warnings == []
        assert result.total_files == 0
        assert result.total_size_bytes == 0
        assert result.build_time_seconds == 0.0

    def test_success_result(self):
        """Test creating a successful result."""
        result = BuildResult()
        result.success = True
        result.output_path = Path("/test/output.app")
        result.total_files = 42
        result.total_size_bytes = 1024 * 1024  # 1MB
        result.build_time_seconds = 5.5

        assert result.success
        assert result.output_path == Path("/test/output.app")
        assert result.total_files == 42
        assert result.total_size_bytes == 1024 * 1024

    def test_failed_result_with_errors(self):
        """Test creating a failed result with errors."""
        result = BuildResult()
        result.errors = ["Missing ui.yaml", "Invalid icon format"]

        assert not result.success
        assert len(result.errors) == 2
        assert "Missing ui.yaml" in result.errors


# ──────────────────────────────────────────────────────────────
# Builder Tests
# ──────────────────────────────────────────────────────────────

class TestBuilder:
    """Tests for Builder class."""

    def test_builder_init(self, tmp_path):
        """Test Builder initialization."""
        config = BuildConfig.default("Test App")
        builder = Builder(config, tmp_path)

        assert builder.config is config
        assert builder.project_path == tmp_path
        assert builder._cancelled is False

    def test_builder_cancel(self, tmp_path):
        """Test Builder cancellation."""
        config = BuildConfig.default("Test App")
        builder = Builder(config, tmp_path)

        assert builder._cancelled is False
        builder.cancel()
        assert builder._cancelled is True

    def test_builder_progress_callback(self, tmp_path):
        """Test Builder progress callback."""
        config = BuildConfig.default("Test App")
        builder = Builder(config, tmp_path)

        progress_calls = []
        builder.on_progress(lambda p, m: progress_calls.append((p, m)))

        builder._report_progress(50, "Test message")

        assert len(progress_calls) == 1
        assert progress_calls[0] == (50, "Test message")

    def test_builder_validates_project(self, tmp_path):
        """Test Builder validates project before build."""
        config = BuildConfig.default("Test App")
        config.ui = "nonexistent.yaml"

        builder = Builder(config, tmp_path)
        result = builder.build()

        assert not result.success
        assert any("ui" in e.lower() or "not found" in e.lower() for e in result.errors)


# ──────────────────────────────────────────────────────────────
# BuildWorker Tests
# ──────────────────────────────────────────────────────────────

class TestBuildWorker:
    """Tests for BuildWorker thread."""

    def test_worker_init(self, tmp_path):
        """Test BuildWorker initialization."""
        config = BuildConfig.default("Test App")
        worker = BuildWorker(config, tmp_path)

        assert worker.config is config
        assert worker.project_path == tmp_path

    def test_worker_has_signals(self, tmp_path):
        """Test BuildWorker has required signals."""
        config = BuildConfig.default("Test App")
        worker = BuildWorker(config, tmp_path)

        # Check signals exist
        assert hasattr(worker, 'progress')
        assert hasattr(worker, 'finished')
        assert hasattr(worker, 'error')


# ──────────────────────────────────────────────────────────────
# BuildProgressDialog Tests
# ──────────────────────────────────────────────────────────────

class TestBuildProgressDialog:
    """Tests for BuildProgressDialog UI."""

    def test_dialog_creates(self, qtbot, tmp_path):
        """Test dialog can be created."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        assert dialog is not None
        assert "Test App" in dialog.windowTitle()

    def test_dialog_has_progress_bar(self, qtbot, tmp_path):
        """Test dialog has progress bar."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        assert dialog._progress_bar is not None
        assert dialog._progress_bar.value() == 0
        assert dialog._progress_bar.maximum() == 100

    def test_dialog_has_cancel_button(self, qtbot, tmp_path):
        """Test dialog has cancel button."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        assert dialog._cancel_btn is not None
        assert dialog._cancel_btn.text() == "Cancel"

    def test_dialog_has_status_labels(self, qtbot, tmp_path):
        """Test dialog has status labels."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        assert dialog._status_label is not None
        assert dialog._detail_label is not None

    def test_dialog_progress_update(self, qtbot, tmp_path):
        """Test dialog updates progress."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        dialog._on_progress(50, "Packaging assets...")

        assert dialog._progress_bar.value() == 50
        assert dialog._status_label.text() == "Packaging assets..."

    def test_dialog_run_after_build_flag(self, qtbot, tmp_path):
        """Test dialog run_after_build flag."""
        config = BuildConfig.default("Test App")

        dialog1 = BuildProgressDialog(config, tmp_path, run_after_build=False)
        qtbot.addWidget(dialog1)
        assert dialog1.run_after_build is False

        dialog2 = BuildProgressDialog(config, tmp_path, run_after_build=True)
        qtbot.addWidget(dialog2)
        assert dialog2.run_after_build is True

    def test_dialog_has_build_completed_signal(self, qtbot, tmp_path):
        """Test dialog has build_completed signal."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        assert hasattr(dialog, 'build_completed')

    def test_dialog_on_finished_success(self, qtbot, tmp_path):
        """Test dialog handles successful build."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        result = BuildResult()
        result.success = True
        result.output_path = tmp_path / "output.app"
        result.total_files = 10
        result.total_size_bytes = 1024
        result.build_time_seconds = 2.5

        # Simulate completion
        dialog._on_finished(result)

        assert dialog._progress_bar.value() == 100
        assert "complete" in dialog._status_label.text().lower()
        assert dialog._cancel_btn.text() == "Close"

    def test_dialog_on_finished_failure(self, qtbot, tmp_path):
        """Test dialog handles failed build."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        result = BuildResult()
        result.success = False
        result.errors = ["Missing dependency", "Invalid config"]

        # Simulate failure
        dialog._on_finished(result)

        assert "failed" in dialog._status_label.text().lower()
        assert dialog._cancel_btn.text() == "Close"

    def test_dialog_on_error(self, qtbot, tmp_path):
        """Test dialog handles errors."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        dialog._on_error("Fatal error occurred")

        assert "error" in dialog._status_label.text().lower()
        assert "Fatal error" in dialog._detail_label.text()

    def test_dialog_get_result(self, qtbot, tmp_path):
        """Test dialog get_result method."""
        config = BuildConfig.default("Test App")
        dialog = BuildProgressDialog(config, tmp_path)
        qtbot.addWidget(dialog)

        # Initially no result
        assert dialog.get_result() is None

        # After completion
        result = BuildResult()
        result.success = True
        dialog._on_finished(result)

        assert dialog.get_result() is not None
        assert dialog.get_result().success is True


# ──────────────────────────────────────────────────────────────
# Integration Tests
# ──────────────────────────────────────────────────────────────

class TestBuildIntegration:
    """Integration tests for the build system."""

    def test_build_empty_project(self, tmp_path):
        """Test building an empty project fails gracefully."""
        # Create minimal project structure
        (tmp_path / "ui.yaml").write_text("# Empty UI\n")

        config = BuildConfig.default("Empty Project")
        config.ui = "ui.yaml"

        builder = Builder(config, tmp_path)
        result = builder.build()

        # Should fail because no real content
        # But should not crash
        assert isinstance(result, BuildResult)

    def test_build_with_ui_file(self, tmp_path):
        """Test building with a valid UI file."""
        # Create a valid project structure
        ui_content = """
root:
  type: Panel
  width: 800
  height: 600
  children: []
"""
        (tmp_path / "ui.yaml").write_text(ui_content)

        config = BuildConfig.default("UI Project")
        config.ui = "ui.yaml"

        builder = Builder(config, tmp_path)
        result = builder.build()

        # Result should be a BuildResult object
        assert isinstance(result, BuildResult)


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
