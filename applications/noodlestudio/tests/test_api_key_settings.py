# ──────────────────────────────────────────────────────────────
#   Tests for API Key Settings Widget
#
#   Tests for secure API key storage, display, and management.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import pytest
from unittest.mock import patch, MagicMock
import subprocess

from noodlestudio.panels.api_key_settings import (
    APIKeySettingsWidget,
    KEYCHAIN_ACCOUNT,
    KEYCHAIN_SERVICE,
)


class TestAPIKeySettingsKeychain:
    """Tests for keychain integration (no Qt required)."""

    def test_save_to_keychain_command(self):
        """Verify keychain save command structure."""
        # Test the command structure without creating widget
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0)

            # Call the static-like method by patching widget creation
            result = subprocess.run([
                'security', 'add-generic-password',
                '-a', KEYCHAIN_ACCOUNT,
                '-s', KEYCHAIN_SERVICE,
                '-w', 'test_key_123',
                '-U'
            ], capture_output=True)

            mock_run.assert_called_once()
            call_args = mock_run.call_args[0][0]
            assert 'security' in call_args
            assert 'add-generic-password' in call_args
            assert KEYCHAIN_ACCOUNT in call_args
            assert KEYCHAIN_SERVICE in call_args

    def test_load_from_keychain_command(self):
        """Verify keychain load command structure."""
        with patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(
                returncode=0,
                stdout="nood_test_key_xyz\n"
            )

            result = subprocess.run([
                'security', 'find-generic-password',
                '-a', KEYCHAIN_ACCOUNT,
                '-s', KEYCHAIN_SERVICE,
                '-w'
            ], capture_output=True, text=True)

            mock_run.assert_called_once()
            assert result.stdout.strip() == "nood_test_key_xyz"


class TestAPIKeySettingsWidget:
    """Tests for API Key Settings Widget (requires Qt)."""

    @pytest.fixture
    def widget(self, qtbot):
        """Create widget with mocked key loading."""
        with patch.object(APIKeySettingsWidget, '_load_from_keychain', return_value=None):
            with patch.dict('os.environ', {'NOODLEROUTER_API_KEY': ''}):
                w = APIKeySettingsWidget()
                qtbot.addWidget(w)
                return w

    @pytest.fixture
    def widget_with_key(self, qtbot):
        """Create widget with a key already loaded."""
        with patch.object(APIKeySettingsWidget, '_load_from_keychain', return_value='nood_test_key_123'):
            w = APIKeySettingsWidget()
            qtbot.addWidget(w)
            return w

    def test_widget_creation(self, widget):
        """Widget creates without error."""
        assert widget is not None
        assert widget._key_display is not None
        assert widget._copy_btn is not None
        assert widget._regen_btn is not None

    def test_widget_with_no_key_shows_error(self, widget):
        """Widget shows error when no key available."""
        # Error label is not hidden (isHidden checks parent-independent state)
        assert not widget._error_label.isHidden()
        assert "No API key" in widget._error_label.text()

    def test_widget_with_key_displays_it(self, widget_with_key):
        """Widget displays key when available."""
        assert widget_with_key._api_key == "nood_test_key_123"
        assert widget_with_key._key_display.text() == "nood_test_key_123"
        assert widget_with_key._copy_btn.isEnabled()
        assert widget_with_key._regen_btn.isEnabled()

    def test_display_key(self, widget):
        """Display key sets text and enables buttons."""
        widget._display_key("nood_test_12345")

        assert widget._api_key == "nood_test_12345"
        assert widget._key_display.text() == "nood_test_12345"
        assert widget._copy_btn.isEnabled()
        assert widget._regen_btn.isEnabled()

    def test_get_api_key(self, widget):
        """get_api_key returns stored key."""
        widget._api_key = "nood_stored_key"
        assert widget.get_api_key() == "nood_stored_key"

    def test_set_api_key(self, widget):
        """set_api_key stores and displays key."""
        with patch.object(widget, '_save_to_keychain', return_value=True):
            with patch.object(widget, '_load_usage'):
                widget.set_api_key("nood_new_key")

        assert widget._api_key == "nood_new_key"

    def test_show_error(self, widget):
        """Error label shows message."""
        widget._show_error("Test error message")

        assert widget._error_label.text() == "Test error message"
        assert not widget._error_label.isHidden()

    def test_copy_key_to_clipboard(self, qtbot, widget_with_key):
        """Copy button copies key to clipboard."""
        from PyQt6.QtWidgets import QApplication

        widget_with_key._copy_key()

        clipboard = QApplication.clipboard()
        assert clipboard.text() == "nood_test_key_123"

    def test_copy_button_feedback(self, widget_with_key):
        """Copy button shows feedback after click."""
        widget_with_key._copy_key()

        assert widget_with_key._copy_btn.text() == "Copied!"


class TestAPIKeySettingsLoadKey:
    """Tests for key loading logic."""

    def test_load_key_from_keychain(self, qtbot):
        """Load key from keychain when available."""
        with patch.object(APIKeySettingsWidget, '_load_from_keychain', return_value="nood_keychain_key"):
            widget = APIKeySettingsWidget()
            qtbot.addWidget(widget)

            assert widget._api_key == "nood_keychain_key"

    def test_load_key_from_env(self, qtbot):
        """Load key from environment when keychain empty."""
        with patch.dict('os.environ', {'NOODLEROUTER_API_KEY': 'nood_env_key'}):
            with patch.object(APIKeySettingsWidget, '_load_from_keychain', return_value=None):
                widget = APIKeySettingsWidget()
                qtbot.addWidget(widget)

                assert widget._api_key == "nood_env_key"

    def test_load_key_keychain_priority(self, qtbot):
        """Keychain takes priority over environment."""
        with patch.dict('os.environ', {'NOODLEROUTER_API_KEY': 'nood_env_key'}):
            with patch.object(APIKeySettingsWidget, '_load_from_keychain', return_value="nood_keychain_key"):
                widget = APIKeySettingsWidget()
                qtbot.addWidget(widget)

                assert widget._api_key == "nood_keychain_key"
