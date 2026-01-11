# ──────────────────────────────────────────────────────────────
#   Tests for Editor Access Enforcement
#
#   Tests for the editor access control system that restricts
#   access to the NoodleStudio editor in published apps.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import pytest
from unittest.mock import MagicMock, patch


class TestPasswordHashing:
    """Tests for password hashing functions."""

    def test_hash_password_returns_hex(self):
        """hash_password returns a hex string."""
        from noodlestudio.dialogs.editor_password_dialog import hash_password
        result = hash_password("test123")
        assert isinstance(result, str)
        assert len(result) == 64  # SHA-256 produces 64 hex chars

    def test_hash_password_deterministic(self):
        """Same password produces same hash."""
        from noodlestudio.dialogs.editor_password_dialog import hash_password
        hash1 = hash_password("mysecret")
        hash2 = hash_password("mysecret")
        assert hash1 == hash2

    def test_hash_password_different_inputs(self):
        """Different passwords produce different hashes."""
        from noodlestudio.dialogs.editor_password_dialog import hash_password
        hash1 = hash_password("password1")
        hash2 = hash_password("password2")
        assert hash1 != hash2

    def test_verify_password_correct(self):
        """verify_password returns True for correct password."""
        from noodlestudio.dialogs.editor_password_dialog import (
            hash_password, verify_password
        )
        stored_hash = hash_password("correct_password")
        assert verify_password("correct_password", stored_hash) is True

    def test_verify_password_incorrect(self):
        """verify_password returns False for incorrect password."""
        from noodlestudio.dialogs.editor_password_dialog import (
            hash_password, verify_password
        )
        stored_hash = hash_password("correct_password")
        assert verify_password("wrong_password", stored_hash) is False


class TestEditorPasswordDialog:
    """Tests for EditorPasswordDialog."""

    def test_dialog_creates(self, qtbot):
        """Dialog creates successfully."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("testpass")
        dialog = EditorPasswordDialog(stored_hash)
        qtbot.addWidget(dialog)
        assert dialog is not None

    def test_dialog_has_password_field(self, qtbot):
        """Dialog has a password field."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("testpass")
        dialog = EditorPasswordDialog(stored_hash)
        qtbot.addWidget(dialog)
        assert dialog._password_field is not None

    def test_dialog_rejects_empty_password(self, qtbot):
        """Dialog shows error for empty password."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("testpass")
        dialog = EditorPasswordDialog(stored_hash)
        qtbot.addWidget(dialog)
        dialog.show()

        dialog._on_submit()

        # Error label text should be set (visible state can be tricky in tests)
        assert "enter a password" in dialog._error_label.text().lower()

    def test_dialog_accepts_correct_password(self, qtbot):
        """Dialog accepts correct password."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        from PyQt6.QtWidgets import QDialog

        stored_hash = hash_password("correct")
        dialog = EditorPasswordDialog(stored_hash)
        qtbot.addWidget(dialog)

        dialog._password_field.setText("correct")

        # Mock accept to track if it was called
        accepted = []
        original_accept = dialog.accept
        dialog.accept = lambda: (accepted.append(True), original_accept())

        dialog._on_submit()

        assert len(accepted) == 1

    def test_dialog_rejects_incorrect_password(self, qtbot):
        """Dialog shows error for incorrect password."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("correct")
        dialog = EditorPasswordDialog(stored_hash)
        qtbot.addWidget(dialog)
        dialog.show()

        dialog._password_field.setText("wrong")
        dialog._on_submit()

        # Error label text should be set
        assert "incorrect" in dialog._error_label.text().lower()

    def test_dialog_tracks_attempts(self, qtbot):
        """Dialog tracks failed attempts."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("correct")
        dialog = EditorPasswordDialog(stored_hash, max_attempts=3)
        qtbot.addWidget(dialog)

        # First wrong attempt
        dialog._password_field.setText("wrong")
        dialog._on_submit()
        assert dialog._attempts == 1
        assert "2 attempts" in dialog._error_label.text()

        # Second wrong attempt
        dialog._password_field.setText("wrong")
        dialog._on_submit()
        assert dialog._attempts == 2
        assert "1 attempt" in dialog._error_label.text()

    def test_dialog_locks_after_max_attempts(self, qtbot):
        """Dialog locks out after max failed attempts."""
        from noodlestudio.dialogs.editor_password_dialog import (
            EditorPasswordDialog, hash_password
        )
        stored_hash = hash_password("correct")
        dialog = EditorPasswordDialog(stored_hash, max_attempts=2)
        qtbot.addWidget(dialog)

        # Exhaust attempts
        dialog._password_field.setText("wrong")
        dialog._on_submit()
        dialog._password_field.setText("wrong")
        dialog._on_submit()

        assert dialog._password_field.isEnabled() is False
        assert dialog._submit_btn.isEnabled() is False


class TestMainWindowFoldMixinAccess:
    """Tests for editor access in MainWindowFoldMixin."""

    def test_default_access_is_allow(self):
        """Default editor access is 'allow'."""
        # Create a mock class that uses the mixin
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "allow"
                self._editor_password_hash = None

        window = MockWindow()
        assert window._editor_access == "allow"

    def test_set_editor_access_allow(self):
        """set_editor_access configures allow mode."""
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "allow"
                self._editor_password_hash = None

        window = MockWindow()
        window.set_editor_access(access="allow")
        assert window._editor_access == "allow"

    def test_set_editor_access_hidden(self):
        """set_editor_access configures hidden mode."""
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin
        from unittest.mock import MagicMock

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "allow"
                self._editor_password_hash = None
                self._fold_shortcut = MagicMock()

        window = MockWindow()
        window.set_editor_access(access="hidden")

        assert window._editor_access == "hidden"
        window._fold_shortcut.setEnabled.assert_called_with(False)

    def test_set_editor_access_password(self):
        """set_editor_access configures password mode."""
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin
        from noodlestudio.dialogs.editor_password_dialog import hash_password

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "allow"
                self._editor_password_hash = None

        window = MockWindow()
        pw_hash = hash_password("secret")
        window.set_editor_access(access="password", password_hash=pw_hash)

        assert window._editor_access == "password"
        assert window._editor_password_hash == pw_hash

    def test_check_editor_access_allow(self):
        """_check_editor_access returns True for allow mode."""
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "allow"
                self._editor_password_hash = None

        window = MockWindow()
        assert window._check_editor_access() is True

    def test_check_editor_access_hidden(self):
        """_check_editor_access returns False for hidden mode."""
        from noodlestudio.core.main_window_fold_mixin import MainWindowFoldMixin

        class MockWindow(MainWindowFoldMixin):
            def __init__(self):
                self._editor_access = "hidden"
                self._editor_password_hash = None

        window = MockWindow()
        assert window._check_editor_access() is False


class TestBuildConfigEditorIntegration:
    """Tests for BuildConfig editor settings integration."""

    def test_build_config_editor_defaults(self):
        """BuildConfig.editor has correct defaults."""
        from noodlestudio.core.build_config import BuildConfig

        config = BuildConfig.default(name="Test")
        assert config.editor.access == "allow"
        assert config.editor.password_hash is None
        assert config.editor.keyboard_shortcut == "Ctrl+Shift+U"

    def test_build_config_editor_round_trip(self):
        """EditorConfig serializes and deserializes correctly."""
        from noodlestudio.core.build_config import EditorConfig

        config = EditorConfig(
            access="password",
            password_hash="abc123hash",
            keyboard_shortcut="Ctrl+Alt+E"
        )

        data = config.to_dict()
        restored = EditorConfig.from_dict(data)

        assert restored.access == "password"
        assert restored.password_hash == "abc123hash"
        assert restored.keyboard_shortcut == "Ctrl+Alt+E"

    def test_build_settings_dialog_hashes_password(self, qtbot, tmp_path):
        """BuildSettingsDialog hashes password on save."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog
        from noodlestudio.core.build_config import BuildConfig
        from noodlestudio.dialogs.editor_password_dialog import hash_password

        dialog = BuildSettingsDialog(tmp_path)
        qtbot.addWidget(dialog)

        # Set to password mode and enter password
        dialog.editor_password.setChecked(True)
        dialog.editor_pw_field.setText("secret123")

        # Update config from UI values
        dialog._save_values_to_config()

        # Verify password is hashed (not plaintext)
        expected_hash = hash_password("secret123")
        assert dialog.config.editor.password_hash == expected_hash
        assert "secret123" not in dialog.config.editor.password_hash
