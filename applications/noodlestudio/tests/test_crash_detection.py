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
#   Tests for Crash Detection and Recovery System
#
#   Tests the sentinel file mechanism that detects crashes wh...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.tests.test_crash_detection
# PURPOSE:  Tests for Crash Detection and Recovery System
# LAYER:    Studio / Tests
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   TestSentinelFile, TestCrashDetection, TestSaveCrashInfo, TestCrashRecoveryFlow, TestRecoveryDialog
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import pytest
import json
import os
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

pytestmark = pytest.mark.slow

# Import the crash detection and single-instance functions
from noodlestudio.main import (
    SENTINEL_DIR,
    SENTINEL_FILE,
    CRASH_INFO_FILE,
    SINGLE_INSTANCE_KEY,
    create_sentinel,
    remove_sentinel,
    check_for_crash,
    save_crash_info,
    claim_single_instance,
)


@pytest.fixture
def clean_sentinel_state():
    """Ensure clean state before and after each test."""
    # Clean up before test
    if SENTINEL_FILE.exists():
        SENTINEL_FILE.unlink()
    if CRASH_INFO_FILE.exists():
        CRASH_INFO_FILE.unlink()

    yield

    # Clean up after test
    if SENTINEL_FILE.exists():
        SENTINEL_FILE.unlink()
    if CRASH_INFO_FILE.exists():
        CRASH_INFO_FILE.unlink()


class TestSentinelFile:
    """Tests for sentinel file creation and removal."""

    def test_create_sentinel(self, clean_sentinel_state):
        """Sentinel file is created with correct contents."""
        create_sentinel()

        assert SENTINEL_FILE.exists(), "Sentinel file should be created"

        content = SENTINEL_FILE.read_text()
        assert f"pid={os.getpid()}" in content, "Should contain current PID"
        assert "started=" in content, "Should contain start timestamp"
        assert "version=" in content, "Should contain version"

    def test_remove_sentinel(self, clean_sentinel_state):
        """Sentinel file is removed on clean shutdown."""
        create_sentinel()
        assert SENTINEL_FILE.exists()

        remove_sentinel()
        assert not SENTINEL_FILE.exists(), "Sentinel should be removed"

    def test_remove_sentinel_idempotent(self, clean_sentinel_state):
        """Removing non-existent sentinel doesn't raise error."""
        assert not SENTINEL_FILE.exists()
        remove_sentinel()  # Should not raise
        assert not SENTINEL_FILE.exists()

    def test_sentinel_dir_created(self, clean_sentinel_state):
        """Sentinel directory is created if it doesn't exist."""
        # The directory might already exist from other tests
        # Just verify create_sentinel works
        create_sentinel()
        assert SENTINEL_DIR.exists()
        assert SENTINEL_DIR.is_dir()


class TestCrashDetection:
    """Tests for crash detection logic."""

    def test_no_crash_on_fresh_start(self, clean_sentinel_state):
        """No crash detected when no sentinel exists."""
        assert not SENTINEL_FILE.exists()
        result = check_for_crash()
        assert result is False, "Should not detect crash on fresh start"

    def test_crash_detected_when_sentinel_exists(self, clean_sentinel_state):
        """Crash detected when sentinel file exists from previous session."""
        # Simulate previous session that crashed (left sentinel behind)
        create_sentinel()
        assert SENTINEL_FILE.exists()

        # Simulate next session startup
        result = check_for_crash()

        assert result is True, "Should detect crash when sentinel exists"
        assert not SENTINEL_FILE.exists(), "Sentinel should be cleaned up"

    def test_crash_info_saved_on_detection(self, clean_sentinel_state):
        """Crash info is saved when crash is detected."""
        create_sentinel()

        check_for_crash()

        assert CRASH_INFO_FILE.exists(), "Crash info file should be created"

        # Verify crash info contents
        import ast
        content = CRASH_INFO_FILE.read_text()
        crash_info = ast.literal_eval(content)

        assert "detected_at" in crash_info
        assert "sentinel_info" in crash_info

    def test_crash_detection_cleans_sentinel(self, clean_sentinel_state):
        """Sentinel is removed after crash detection."""
        create_sentinel()
        check_for_crash()

        assert not SENTINEL_FILE.exists(), "Sentinel should be removed after detection"


class TestSaveCrashInfo:
    """Tests for saving crash information from exceptions."""

    def test_save_crash_info_from_exception(self, clean_sentinel_state):
        """Crash info is saved correctly from exception details."""
        try:
            raise ValueError("Test error message")
        except ValueError:
            import sys
            exc_type, exc_value, exc_tb = sys.exc_info()
            save_crash_info(exc_type, exc_value, exc_tb)

        assert CRASH_INFO_FILE.exists()

        content = CRASH_INFO_FILE.read_text()
        crash_info = json.loads(content)

        assert crash_info["exception_type"] == "ValueError"
        assert "Test error message" in crash_info["exception_value"]
        assert "traceback" in crash_info
        assert "version" in crash_info
        assert "timestamp" in crash_info

    def test_save_crash_info_with_none_values(self, clean_sentinel_state):
        """Handles None exception values gracefully."""
        save_crash_info(None, None, None)

        assert CRASH_INFO_FILE.exists()

        content = CRASH_INFO_FILE.read_text()
        crash_info = json.loads(content)

        assert crash_info["exception_type"] == "Unknown"
        assert crash_info["exception_value"] == ""


class TestCrashRecoveryFlow:
    """Integration tests for the full crash recovery flow."""

    def test_full_crash_recovery_flow(self, clean_sentinel_state):
        """Test complete flow: start -> crash -> restart -> detect."""
        # Session 1: App starts
        create_sentinel()
        assert SENTINEL_FILE.exists()

        # Session 1: App crashes (sentinel remains)
        # (We just don't call remove_sentinel)

        # Session 2: App restarts and detects crash
        crash_detected = check_for_crash()

        assert crash_detected is True
        assert CRASH_INFO_FILE.exists()
        assert not SENTINEL_FILE.exists()  # Cleaned up

        # Session 2: App creates new sentinel
        create_sentinel()
        assert SENTINEL_FILE.exists()

        # Session 2: App exits cleanly
        remove_sentinel()
        assert not SENTINEL_FILE.exists()

        # Session 3: No crash detected
        crash_detected = check_for_crash()
        assert crash_detected is False

    def test_clean_shutdown_no_crash_next_session(self, clean_sentinel_state):
        """Clean shutdown means no crash detected on next start."""
        # Session 1: Normal operation
        create_sentinel()
        remove_sentinel()  # Clean shutdown

        # Session 2: No crash
        assert check_for_crash() is False


class TestRecoveryDialog:
    """GUI tests for the recovery dialog."""

    def test_recovery_dialog_creation(self, qapp, clean_sentinel_state):
        """Recovery dialog can be created and shown."""
        from noodlestudio.main import show_crash_recovery_dialog

        # Create some crash info
        create_sentinel()
        check_for_crash()  # Creates crash info

        # Create a mock parent widget
        from PyQt6.QtWidgets import QWidget
        parent = QWidget()

        # We can't easily test dialog.exec() in unit tests
        # but we can verify the function doesn't crash
        # Use a timer to close the dialog immediately
        from PyQt6.QtCore import QTimer

        def close_dialog():
            # Find and close any open dialogs
            for widget in qapp.topLevelWidgets():
                if widget.windowTitle() == "Session Recovery":
                    widget.reject()

        QTimer.singleShot(100, close_dialog)

        # This will open and immediately close
        show_crash_recovery_dialog(parent)

        # If we get here without exception, the dialog works
        parent.close()

    def test_recovery_dialog_without_crash_info(self, qapp, clean_sentinel_state):
        """Recovery dialog handles missing crash info gracefully."""
        from noodlestudio.main import show_crash_recovery_dialog
        from PyQt6.QtWidgets import QWidget
        from PyQt6.QtCore import QTimer

        # No crash info file
        assert not CRASH_INFO_FILE.exists()

        parent = QWidget()

        def close_dialog():
            for widget in qapp.topLevelWidgets():
                if widget.windowTitle() == "Session Recovery":
                    widget.reject()

        QTimer.singleShot(100, close_dialog)

        # Should not crash even without crash info
        show_crash_recovery_dialog(parent)

        parent.close()


class TestBugReportDialog:
    """Tests for the bug report dialog."""

    def test_bug_report_dialog_creation(self, qapp):
        """Bug report dialog can be created."""
        from noodlestudio.dialogs.bug_report_dialog import BugReportDialog

        dialog = BugReportDialog()

        assert dialog.windowTitle() == "Report a Bug"
        assert dialog.summary_edit is not None
        assert dialog.severity_combo is not None
        assert dialog.description_edit is not None

        dialog.close()

    def test_bug_report_dialog_with_crash_info(self, qapp):
        """Bug report dialog accepts crash info."""
        from noodlestudio.dialogs.bug_report_dialog import BugReportDialog

        crash_info = {
            "exception_type": "TestError",
            "exception_message": "Test crash",
            "traceback": "Traceback line 1\nTraceback line 2",
        }

        dialog = BugReportDialog(crash_info=crash_info)

        assert dialog.windowTitle() == "Crash Report"
        assert "TestError" in dialog.summary_edit.text()

        dialog.close()

    def test_system_info_collection(self):
        """System info collector gathers required fields."""
        from noodlestudio.dialogs.bug_report_dialog import SystemInfoCollector

        info = SystemInfoCollector.collect()

        assert "noodlestudio_version" in info
        assert "python_version" in info
        assert "platform" in info
        assert "timestamp" in info

    def test_bug_report_payload_structure(self, qapp):
        """Bug report builds correct payload structure."""
        from noodlestudio.dialogs.bug_report_dialog import BugReportDialog

        dialog = BugReportDialog()
        dialog.summary_edit.setText("Test summary")
        dialog.description_edit.setText("Test description")
        dialog.severity_combo.setCurrentIndex(2)  # Minor

        report = dialog._build_report()

        assert report["summary"] == "Test summary"
        assert report["description"] == "Test description"
        assert report["severity"] == "minor"
        assert "system_info" in report
        assert "noodlestudio_version" in report["system_info"]

        dialog.close()


# =============================================================================
# Single-Instance Enforcement Tests
# =============================================================================

class TestSingleInstance:
    """Tests for single-instance application enforcement."""

    @pytest.fixture(autouse=True)
    def clean_server(self, qapp):
        """Ensure no stale server from previous tests."""
        from PyQt6.QtNetwork import QLocalServer
        QLocalServer.removeServer(SINGLE_INSTANCE_KEY)
        yield
        QLocalServer.removeServer(SINGLE_INSTANCE_KEY)

    def test_claim_succeeds_when_none_running(self, qapp):
        """Claiming the lock succeeds when no other instance holds it."""
        assert claim_single_instance(qapp) is True

    def test_second_claim_fails(self, qapp):
        """Second claim fails when first instance holds the lock."""
        assert claim_single_instance(qapp) is True
        # A second claim in the same process should fail
        from PyQt6.QtNetwork import QLocalServer
        second_server = QLocalServer(qapp)
        assert second_server.listen(SINGLE_INSTANCE_KEY) is False

    def test_parse_args_allow_multiple(self):
        """--allow-multiple flag is parsed correctly."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--allow-multiple', '--no-splash']
        try:
            args = parse_args()
            assert args.allow_multiple is True
        finally:
            sys.argv = original_argv

    def test_parse_args_default_no_multiple(self):
        """Default: allow_multiple is False."""
        from noodlestudio.main import parse_args
        import sys

        original_argv = sys.argv
        sys.argv = ['noodlestudio', '--no-splash']
        try:
            args = parse_args()
            assert args.allow_multiple is False
        finally:
            sys.argv = original_argv


# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
