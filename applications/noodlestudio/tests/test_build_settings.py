# ──────────────────────────────────────────────────────────────
#   Tests for Build Settings
#
#   Tests for BuildConfig dataclass and BuildSettingsDialog.
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# (C) 2026 Caitlyn Meeks / Noodling Technologies, LLC
# ──────────────────────────────────────────────────────────────

import pytest
import tempfile
from pathlib import Path

import yaml


# =============================================================================
# BuildConfig Tests
# =============================================================================

class TestAppIdentity:
    """Tests for AppIdentity dataclass."""

    def test_default_values(self):
        """Default identity has expected values."""
        from noodlestudio.core.build_config import AppIdentity
        identity = AppIdentity()

        assert identity.name == "Untitled"
        assert identity.bundle_id == "ai.noodlings.untitled"
        assert identity.version == "1.0.0"
        assert identity.icon == ""

    def test_to_dict_round_trip(self):
        """to_dict and from_dict preserve values."""
        from noodlestudio.core.build_config import AppIdentity
        identity = AppIdentity(
            name="My App",
            bundle_id="com.example.myapp",
            version="2.1.0",
            icon="assets/icon.png"
        )

        data = identity.to_dict()
        restored = AppIdentity.from_dict(data)

        assert restored.name == identity.name
        assert restored.bundle_id == identity.bundle_id
        assert restored.version == identity.version
        assert restored.icon == identity.icon


class TestSplashConfig:
    """Tests for SplashConfig dataclass."""

    def test_default_values(self):
        """Default splash config has expected values."""
        from noodlestudio.core.build_config import SplashConfig
        config = SplashConfig()

        assert config.enabled is True
        assert config.duration == 3.0
        assert config.click_to_dismiss is True
        assert config.background == "#1a1a1a"
        assert config.fade_in == 0.3
        assert config.fade_out == 0.3
        assert config.attribution_position == "bottom_right"

    def test_to_dict_round_trip(self):
        """to_dict and from_dict preserve values."""
        from noodlestudio.core.build_config import SplashConfig
        config = SplashConfig(
            enabled=False,
            image="splash.png",
            duration=5.0,
            click_to_dismiss=False,
            background="#ff0000",
            fade_in=1.0,
            fade_out=0.5,
            attribution_position="bottom_left"
        )

        data = config.to_dict()
        restored = SplashConfig.from_dict(data)

        assert restored.enabled == config.enabled
        assert restored.image == config.image
        assert restored.duration == config.duration
        assert restored.click_to_dismiss == config.click_to_dismiss
        assert restored.background == config.background
        assert restored.fade_in == config.fade_in
        assert restored.fade_out == config.fade_out
        assert restored.attribution_position == config.attribution_position


class TestEditorConfig:
    """Tests for EditorConfig dataclass."""

    def test_default_values(self):
        """Default editor config allows unfold."""
        from noodlestudio.core.build_config import EditorConfig
        config = EditorConfig()

        assert config.access == "allow"
        assert config.password_hash is None
        assert config.keyboard_shortcut == "Ctrl+Shift+U"

    def test_to_dict_round_trip(self):
        """to_dict and from_dict preserve values."""
        from noodlestudio.core.build_config import EditorConfig
        config = EditorConfig(
            access="password",
            password_hash="$2b$hash",
            keyboard_shortcut="Ctrl+Alt+E"
        )

        data = config.to_dict()
        restored = EditorConfig.from_dict(data)

        assert restored.access == config.access
        assert restored.password_hash == config.password_hash
        assert restored.keyboard_shortcut == config.keyboard_shortcut


class TestLLMConfig:
    """Tests for LLMConfig dataclass."""

    def test_default_values(self):
        """Default LLM config uses NoodleROUTER."""
        from noodlestudio.core.build_config import LLMConfig
        config = LLMConfig()

        assert config.provider == "noodlerouter"
        assert config.bundled_key is None

    def test_bundled_key_serialization(self):
        """Bundled key is serialized only when present."""
        from noodlestudio.core.build_config import LLMConfig
        config = LLMConfig(provider="bundled", bundled_key="sk-test123")

        data = config.to_dict()
        assert "bundled_key" in data
        assert data["bundled_key"] == "sk-test123"

        restored = LLMConfig.from_dict(data)
        assert restored.bundled_key == "sk-test123"


class TestContentConfig:
    """Tests for ContentConfig dataclass."""

    def test_default_values(self):
        """Default content config includes all content types."""
        from noodlestudio.core.build_config import ContentConfig
        config = ContentConfig()

        assert config.include_stages is True
        assert config.include_noodlings is True
        assert config.include_ui_layouts is True
        assert config.include_assemblies is True
        assert config.include_plays is True
        assert config.include_unused is False
        assert config.include_source is False


class TestDistributionConfig:
    """Tests for DistributionConfig dataclass."""

    def test_default_values(self):
        """Default distribution uses NoodleStudio signing."""
        from noodlestudio.core.build_config import DistributionConfig
        config = DistributionConfig()

        assert config.signing == "noodlestudio"
        assert config.notarize is True
        assert config.certificate is None


class TestBuildConfig:
    """Tests for BuildConfig main class."""

    def test_default_values(self):
        """Default config has expected values."""
        from noodlestudio.core.build_config import BuildConfig
        config = BuildConfig()

        assert config.target == "macos"
        assert config.ui == "ui.yaml"
        assert config.output_directory == "~/Desktop/builds"

    def test_default_factory(self):
        """default() creates config with name and bundle_id."""
        from noodlestudio.core.build_config import BuildConfig
        config = BuildConfig.default(name="Test App")

        assert config.identity.name == "Test App"
        assert config.identity.bundle_id == "ai.noodlings.testapp"

    def test_default_factory_custom_bundle_id(self):
        """default() respects custom bundle_id."""
        from noodlestudio.core.build_config import BuildConfig
        config = BuildConfig.default(
            name="Test App",
            bundle_id="com.example.test"
        )

        assert config.identity.bundle_id == "com.example.test"

    def test_to_dict_round_trip(self):
        """to_dict and from_dict preserve all values."""
        from noodlestudio.core.build_config import BuildConfig
        config = BuildConfig.default(name="Round Trip Test")
        config.target = "windows"
        config.splash.enabled = False
        config.llm.provider = "ollama"

        data = config.to_dict()
        restored = BuildConfig.from_dict(data)

        assert restored.target == "windows"
        assert restored.identity.name == "Round Trip Test"
        assert restored.splash.enabled is False
        assert restored.llm.provider == "ollama"

    def test_yaml_round_trip(self):
        """Saving to YAML and loading back preserves values."""
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "build.yaml"

            # Create and save
            config = BuildConfig.default(name="YAML Test")
            config.target = "linux"
            config.splash.duration = 5.0
            config.editor.access = "hidden"
            config.to_yaml(yaml_path)

            # Load back
            restored = BuildConfig.from_yaml(yaml_path)

            assert restored.target == "linux"
            assert restored.identity.name == "YAML Test"
            assert restored.splash.duration == 5.0
            assert restored.editor.access == "hidden"

    def test_yaml_file_has_header_comment(self):
        """Saved YAML file has descriptive header comment."""
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "build.yaml"

            config = BuildConfig.default(name="Comment Test")
            config.to_yaml(yaml_path)

            content = yaml_path.read_text()
            assert "NoodleStudio Build Configuration" in content

    def test_from_yaml_nonexistent_raises(self):
        """from_yaml raises FileNotFoundError for missing file."""
        from noodlestudio.core.build_config import BuildConfig

        with pytest.raises(FileNotFoundError):
            BuildConfig.from_yaml(Path("/nonexistent/build.yaml"))

    def test_validate_ui_file_exists(self):
        """validate() checks UI file exists."""
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)

            config = BuildConfig()
            config.ui = "ui.yaml"

            errors = config.validate(project)
            assert any("UI file not found" in e for e in errors)

            # Create the UI file
            (project / "ui.yaml").touch()
            errors = config.validate(project)
            assert not any("UI file not found" in e for e in errors)

    def test_validate_splash_image_exists(self):
        """validate() checks splash image exists when enabled."""
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / "ui.yaml").touch()

            config = BuildConfig()
            config.ui = "ui.yaml"
            config.splash.enabled = True
            config.splash.image = "assets/splash.png"

            errors = config.validate(project)
            assert any("Splash image not found" in e for e in errors)

            # Create the splash image
            (project / "assets").mkdir()
            (project / "assets" / "splash.png").touch()
            errors = config.validate(project)
            assert not any("Splash image not found" in e for e in errors)

    def test_validate_icon_exists(self):
        """validate() checks icon file exists when specified."""
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            (project / "ui.yaml").touch()

            config = BuildConfig()
            config.ui = "ui.yaml"
            config.identity.icon = "icon.png"

            errors = config.validate(project)
            assert any("Icon file not found" in e for e in errors)

            # Create the icon
            (project / "icon.png").touch()
            errors = config.validate(project)
            assert not any("Icon file not found" in e for e in errors)


# =============================================================================
# BuildSettingsDialog Tests
# =============================================================================

class TestBuildSettingsDialog:
    """Tests for BuildSettingsDialog UI."""

    def test_dialog_creates(self, qtbot):
        """Dialog creates without error."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            dialog = BuildSettingsDialog(project)
            qtbot.addWidget(dialog)

            assert dialog is not None
            assert dialog.windowTitle() == "Build Settings"

    def test_dialog_loads_existing_config(self, qtbot):
        """Dialog loads existing build.yaml."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)

            # Create a config file
            config = BuildConfig.default(name="Existing App")
            config.target = "linux"
            config.to_yaml(project / "build.yaml")

            # Open dialog
            dialog = BuildSettingsDialog(project)
            qtbot.addWidget(dialog)

            assert dialog.identity_name.text() == "Existing App"
            assert dialog.platform_linux.isChecked()

    def test_dialog_creates_default_for_new_project(self, qtbot):
        """Dialog creates default config for new project."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)
            project_name = Path(tmpdir).name

            dialog = BuildSettingsDialog(project)
            qtbot.addWidget(dialog)

            # Should have project directory name as app name
            assert dialog.config.identity.name == project_name

    def test_dialog_saves_on_accept(self, qtbot):
        """Dialog saves build.yaml when accepted."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog
        from noodlestudio.core.build_config import BuildConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            project = Path(tmpdir)

            dialog = BuildSettingsDialog(project)
            qtbot.addWidget(dialog)

            # Modify values
            dialog.identity_name.setText("Saved App")
            dialog.platform_windows.setChecked(True)

            # Accept (saves config)
            dialog._save_config()

            # Verify saved
            assert (project / "build.yaml").exists()
            saved = BuildConfig.from_yaml(project / "build.yaml")
            assert saved.identity.name == "Saved App"
            assert saved.target == "windows"

    def test_platform_radio_buttons(self, qtbot):
        """Platform radio buttons work correctly."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Default should be macOS
            assert dialog.platform_macos.isChecked()

            # Select Windows
            dialog.platform_windows.setChecked(True)
            assert not dialog.platform_macos.isChecked()
            assert dialog.platform_windows.isChecked()

            # Web should be disabled
            assert not dialog.platform_web.isEnabled()

    def test_splash_section_is_collapsible(self, qtbot):
        """Splash section can be expanded/collapsed."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Splash section starts collapsed
            assert not dialog.splash_section.is_expanded

            # Can expand
            dialog.splash_section.set_expanded(True)
            assert dialog.splash_section.is_expanded

    def test_editor_password_field_enabled_when_selected(self, qtbot):
        """Password field enables when password option selected."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Password field starts disabled
            assert not dialog.editor_pw_field.isEnabled()

            # Select password option
            dialog.editor_password.setChecked(True)
            assert dialog.editor_pw_field.isEnabled()

            # Deselect
            dialog.editor_allow.setChecked(True)
            assert not dialog.editor_pw_field.isEnabled()

    def test_llm_bundled_key_field_enabled_when_selected(self, qtbot):
        """Bundled key field enables when bundled option selected."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Key field starts disabled
            assert not dialog.llm_bundled_key.isEnabled()

            # Select bundled option
            dialog.llm_bundled.setChecked(True)
            assert dialog.llm_bundled_key.isEnabled()

    def test_attribution_checkboxes_are_locked(self, qtbot):
        """Attribution checkboxes are always checked and disabled."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Attribution checkboxes must be locked
            assert dialog.attr_badge.isChecked()
            assert not dialog.attr_badge.isEnabled()

            assert dialog.attr_nec.isChecked()
            assert not dialog.attr_nec.isEnabled()

    def test_content_checkboxes_default_state(self, qtbot):
        """Content checkboxes have correct default state."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # All content types enabled by default
            assert dialog.content_stages.isChecked()
            assert dialog.content_noodlings.isChecked()
            assert dialog.content_ui.isChecked()
            assert dialog.content_assemblies.isChecked()
            assert dialog.content_plays.isChecked()

            # Optional content disabled by default
            assert not dialog.content_unused.isChecked()
            assert not dialog.content_source.isChecked()

    def test_distribution_certificate_field_enabled_when_selected(self, qtbot):
        """Certificate field enables when own_cert option selected."""
        from noodlestudio.dialogs.build_settings_dialog import BuildSettingsDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            dialog = BuildSettingsDialog(Path(tmpdir))
            qtbot.addWidget(dialog)

            # Certificate field starts disabled
            assert not dialog.dist_certificate.isEnabled()

            # Select own cert option
            dialog.dist_own_cert.setChecked(True)
            assert dialog.dist_certificate.isEnabled()


class TestBuildConfigEnums:
    """Tests for BuildConfig enums."""

    def test_target_platform_values(self):
        """TargetPlatform enum has expected values."""
        from noodlestudio.core.build_config import TargetPlatform

        assert TargetPlatform.MACOS.value == "macos"
        assert TargetPlatform.WINDOWS.value == "windows"
        assert TargetPlatform.LINUX.value == "linux"

    def test_llm_provider_values(self):
        """LLMProvider enum has expected values."""
        from noodlestudio.core.build_config import LLMProvider

        assert LLMProvider.NOODLEROUTER.value == "noodlerouter"
        assert LLMProvider.USER_KEYS.value == "user_keys"
        assert LLMProvider.OLLAMA.value == "ollama"
        assert LLMProvider.BUNDLED.value == "bundled"

    def test_editor_access_values(self):
        """EditorAccess enum has expected values."""
        from noodlestudio.core.build_config import EditorAccess

        assert EditorAccess.ALLOW.value == "allow"
        assert EditorAccess.PASSWORD.value == "password"
        assert EditorAccess.HIDDEN.value == "hidden"

    def test_signing_option_values(self):
        """SigningOption enum has expected values."""
        from noodlestudio.core.build_config import SigningOption

        assert SigningOption.NOODLESTUDIO.value == "noodlestudio"
        assert SigningOption.OWN_CERT.value == "own_cert"
        assert SigningOption.UNSIGNED.value == "unsigned"

    def test_attribution_position_values(self):
        """AttributionPosition enum has expected values."""
        from noodlestudio.core.build_config import AttributionPosition

        assert AttributionPosition.BOTTOM_RIGHT.value == "bottom_right"
        assert AttributionPosition.BOTTOM_LEFT.value == "bottom_left"
        assert AttributionPosition.BOTTOM_CENTER.value == "bottom_center"


# =============================================================================
# Runtime LLM Provider Switching Tests
# =============================================================================

class TestApplyBuildConfigLLMSettings:
    """Tests for _apply_build_config_llm_settings helper."""

    def test_noop_when_no_build_config(self):
        """Does nothing when build_config is None."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, None)

        assert args.provider == 'ollama'

    def test_noop_when_provider_already_set(self):
        """Does nothing when CLI explicitly set provider."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig, LLMConfig

        config = BuildConfig.default("Test")
        config.llm.provider = 'noodlerouter'

        # Simulate user explicitly setting anthropic on CLI
        args = argparse.Namespace(provider='anthropic', api_key=None)
        _apply_build_config_llm_settings(args, config)

        # Should not change - CLI takes precedence
        assert args.provider == 'anthropic'

    def test_applies_noodlerouter_from_build_yaml(self):
        """Applies noodlerouter provider from build.yaml."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        config = BuildConfig.default("Test")
        config.llm.provider = 'noodlerouter'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'noodlerouter'

    def test_applies_ollama_from_build_yaml(self):
        """Keeps ollama when build.yaml specifies ollama."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        config = BuildConfig.default("Test")
        config.llm.provider = 'ollama'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'ollama'

    def test_bundled_maps_to_noodlerouter(self):
        """Bundled provider maps to noodlerouter."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        config = BuildConfig.default("Test")
        config.llm.provider = 'bundled'
        config.llm.bundled_key = 'test-key-123'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'noodlerouter'
        assert args.api_key == 'test-key-123'

    def test_bundled_does_not_override_cli_key(self):
        """Bundled key does not override CLI-provided key."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        config = BuildConfig.default("Test")
        config.llm.provider = 'bundled'
        config.llm.bundled_key = 'build-key-123'

        args = argparse.Namespace(provider='ollama', api_key='cli-key-456')
        _apply_build_config_llm_settings(args, config)

        # CLI key takes precedence
        assert args.api_key == 'cli-key-456'

    def test_user_keys_detects_anthropic(self, monkeypatch):
        """user_keys mode detects ANTHROPIC_API_KEY."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-anthropic-key')
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

        config = BuildConfig.default("Test")
        config.llm.provider = 'user_keys'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'anthropic'

    def test_user_keys_detects_openai(self, monkeypatch):
        """user_keys mode detects OPENAI_API_KEY when no Anthropic key."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.setenv('OPENAI_API_KEY', 'test-openai-key')
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

        config = BuildConfig.default("Test")
        config.llm.provider = 'user_keys'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'openai'

    def test_user_keys_detects_openrouter(self, monkeypatch):
        """user_keys mode detects OPENROUTER_API_KEY when no others."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.setenv('OPENROUTER_API_KEY', 'test-openrouter-key')

        config = BuildConfig.default("Test")
        config.llm.provider = 'user_keys'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        assert args.provider == 'openrouter'

    def test_user_keys_no_keys_warns(self, monkeypatch, capsys):
        """user_keys mode prints warning when no keys found."""
        import argparse
        from noodlestudio.runtime.cli import _apply_build_config_llm_settings
        from noodlestudio.core.build_config import BuildConfig

        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)

        config = BuildConfig.default("Test")
        config.llm.provider = 'user_keys'

        args = argparse.Namespace(provider='ollama', api_key=None)
        _apply_build_config_llm_settings(args, config)

        captured = capsys.readouterr()
        assert "user_keys mode but no API keys found" in captured.err
