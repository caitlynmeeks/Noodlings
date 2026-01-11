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
#   Build Configuration - Dataclasses for build.yaml
#
#   Unity-style Build Settings configuration with YAML serialization.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.build_config
# PURPOSE:  Build Configuration - Dataclasses for build.yaml
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   BuildConfig, AppIdentity, SplashConfig, EditorConfig,
#   LLMConfig, ContentConfig, AdvancedConfig
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional
import yaml

logger = logging.getLogger(__name__)


class TargetPlatform(Enum):
    """Build target platforms."""
    MACOS = "macos"
    WINDOWS = "windows"
    LINUX = "linux"
    # WEB = "web"  # Coming soon


class LLMProvider(Enum):
    """LLM provider options for built apps."""
    NOODLEROUTER = "noodlerouter"
    USER_KEYS = "user_keys"
    OLLAMA = "ollama"
    BUNDLED = "bundled"


class EditorAccess(Enum):
    """Editor access levels for built apps."""
    ALLOW = "allow"           # Allow unfold to editor
    PASSWORD = "password"     # Password protected unfold
    HIDDEN = "hidden"         # No editor access


class SigningOption(Enum):
    """Code signing options."""
    NOODLESTUDIO = "noodlestudio"  # Signed by noodlings.ai (free tier)
    OWN_CERT = "own_cert"          # User's own certificate
    UNSIGNED = "unsigned"           # No signing


class AttributionPosition(Enum):
    """Position for attribution badge."""
    BOTTOM_RIGHT = "bottom_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"


@dataclass
class AppIdentity:
    """App identity settings."""
    name: str = "Untitled"
    bundle_id: str = "ai.noodlings.untitled"
    version: str = "1.0.0"
    icon: str = ""  # Path relative to project

    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'bundle_id': self.bundle_id,
            'version': self.version,
            'icon': self.icon,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AppIdentity':
        return cls(
            name=data.get('name', 'Untitled'),
            bundle_id=data.get('bundle_id', 'ai.noodlings.untitled'),
            version=data.get('version', '1.0.0'),
            icon=data.get('icon', ''),
        )


@dataclass
class SplashConfig:
    """Splash screen settings."""
    enabled: bool = True
    image: str = ""  # Path relative to project
    duration: float = 3.0  # seconds
    click_to_dismiss: bool = True
    background: str = "#1a1a1a"
    fade_in: float = 0.3  # seconds
    fade_out: float = 0.3  # seconds
    # Attribution is ALWAYS enabled - these fields are read-only indicators
    attribution_position: str = "bottom_right"

    def to_dict(self) -> dict:
        return {
            'enabled': self.enabled,
            'image': self.image,
            'duration': self.duration,
            'click_to_dismiss': self.click_to_dismiss,
            'background': self.background,
            'fade_in': self.fade_in,
            'fade_out': self.fade_out,
            'attribution_position': self.attribution_position,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'SplashConfig':
        return cls(
            enabled=data.get('enabled', True),
            image=data.get('image', ''),
            duration=data.get('duration', 3.0),
            click_to_dismiss=data.get('click_to_dismiss', True),
            background=data.get('background', '#1a1a1a'),
            fade_in=data.get('fade_in', 0.3),
            fade_out=data.get('fade_out', 0.3),
            attribution_position=data.get('attribution_position', 'bottom_right'),
        )


@dataclass
class EditorConfig:
    """Editor access settings for built apps."""
    access: str = "allow"  # allow, password, hidden
    password_hash: Optional[str] = None  # bcrypt hash if password protected
    keyboard_shortcut: str = "Ctrl+Shift+U"

    def to_dict(self) -> dict:
        return {
            'access': self.access,
            'password_hash': self.password_hash,
            'keyboard_shortcut': self.keyboard_shortcut,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'EditorConfig':
        return cls(
            access=data.get('access', 'allow'),
            password_hash=data.get('password_hash'),
            keyboard_shortcut=data.get('keyboard_shortcut', 'Ctrl+Shift+U'),
        )


@dataclass
class LLMConfig:
    """LLM provider configuration."""
    provider: str = "noodlerouter"  # noodlerouter, user_keys, ollama, bundled
    bundled_key: Optional[str] = None  # Only if provider is bundled (not recommended)

    def to_dict(self) -> dict:
        d = {'provider': self.provider}
        if self.bundled_key:
            d['bundled_key'] = self.bundled_key
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'LLMConfig':
        return cls(
            provider=data.get('provider', 'noodlerouter'),
            bundled_key=data.get('bundled_key'),
        )


@dataclass
class ContentConfig:
    """Content inclusion settings."""
    include_stages: bool = True
    include_noodlings: bool = True
    include_ui_layouts: bool = True
    include_assemblies: bool = True
    include_plays: bool = True
    include_unused: bool = False
    include_source: bool = False

    def to_dict(self) -> dict:
        return {
            'include_stages': self.include_stages,
            'include_noodlings': self.include_noodlings,
            'include_ui_layouts': self.include_ui_layouts,
            'include_assemblies': self.include_assemblies,
            'include_plays': self.include_plays,
            'include_unused': self.include_unused,
            'include_source': self.include_source,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ContentConfig':
        return cls(
            include_stages=data.get('include_stages', True),
            include_noodlings=data.get('include_noodlings', True),
            include_ui_layouts=data.get('include_ui_layouts', True),
            include_assemblies=data.get('include_assemblies', True),
            include_plays=data.get('include_plays', True),
            include_unused=data.get('include_unused', False),
            include_source=data.get('include_source', False),
        )


@dataclass
class CodeSignConfig:
    """Code signing settings."""
    enabled: bool = False
    certificate: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'enabled': self.enabled,
            'certificate': self.certificate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'CodeSignConfig':
        return cls(
            enabled=data.get('enabled', False),
            certificate=data.get('certificate'),
        )


@dataclass
class NotarizeConfig:
    """Notarization settings."""
    enabled: bool = False
    apple_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'enabled': self.enabled,
            'apple_id': self.apple_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'NotarizeConfig':
        return cls(
            enabled=data.get('enabled', False),
            apple_id=data.get('apple_id'),
        )


@dataclass
class BuildHooks:
    """Build script hooks."""
    pre_build: Optional[str] = None
    post_build: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            'pre_build': self.pre_build,
            'post_build': self.post_build,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BuildHooks':
        return cls(
            pre_build=data.get('pre_build'),
            post_build=data.get('post_build'),
        )


@dataclass
class DistributionConfig:
    """Distribution and signing settings."""
    signing: str = "noodlestudio"  # noodlestudio, own_cert, unsigned
    certificate: Optional[str] = None  # For own_cert
    notarize: bool = True  # Submit for notarization

    def to_dict(self) -> dict:
        d = {
            'signing': self.signing,
            'notarize': self.notarize,
        }
        if self.certificate:
            d['certificate'] = self.certificate
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'DistributionConfig':
        return cls(
            signing=data.get('signing', 'noodlestudio'),
            certificate=data.get('certificate'),
            notarize=data.get('notarize', True),
        )


@dataclass
class AdvancedConfig:
    """Advanced build settings."""
    python_version: str = "3.11"
    strip_debug: bool = False
    codesign: CodeSignConfig = field(default_factory=CodeSignConfig)
    notarize: NotarizeConfig = field(default_factory=NotarizeConfig)
    hooks: BuildHooks = field(default_factory=BuildHooks)

    def to_dict(self) -> dict:
        return {
            'python_version': self.python_version,
            'strip_debug': self.strip_debug,
            'codesign': self.codesign.to_dict(),
            'notarize': self.notarize.to_dict(),
            'hooks': self.hooks.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'AdvancedConfig':
        return cls(
            python_version=data.get('python_version', '3.11'),
            strip_debug=data.get('strip_debug', False),
            codesign=CodeSignConfig.from_dict(data.get('codesign', {})),
            notarize=NotarizeConfig.from_dict(data.get('notarize', {})),
            hooks=BuildHooks.from_dict(data.get('hooks', {})),
        )


@dataclass
class BuildConfig:
    """
    Complete build configuration for a NoodleStudio project.

    Stored in build.yaml at project root.
    """
    target: str = "macos"  # macos, windows, linux
    identity: AppIdentity = field(default_factory=AppIdentity)
    splash: SplashConfig = field(default_factory=SplashConfig)
    editor: EditorConfig = field(default_factory=EditorConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    content: ContentConfig = field(default_factory=ContentConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    advanced: AdvancedConfig = field(default_factory=AdvancedConfig)
    output_directory: str = "~/Desktop/builds"

    # Entry point (from existing builder)
    ui: str = "ui.yaml"
    main_stage: str = ""

    # Window settings
    window_size: tuple = (1280, 720)
    window_title: str = ""  # Defaults to identity.name
    resizable: bool = True
    min_size: tuple = (640, 480)
    fullscreen: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML serialization."""
        return {
            'target': self.target,
            'identity': self.identity.to_dict(),
            'splash': self.splash.to_dict(),
            'editor': self.editor.to_dict(),
            'llm': self.llm.to_dict(),
            'content': self.content.to_dict(),
            'distribution': self.distribution.to_dict(),
            'advanced': self.advanced.to_dict(),
            'output': {
                'directory': self.output_directory,
            },
            'ui': self.ui,
            'main_stage': self.main_stage,
            'settings': {
                'window_size': list(self.window_size),
                'window_title': self.window_title or self.identity.name,
                'resizable': self.resizable,
                'min_size': list(self.min_size),
                'fullscreen': self.fullscreen,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'BuildConfig':
        """Create from dictionary (loaded from YAML)."""
        settings = data.get('settings', {})
        output = data.get('output', {})

        window_size = settings.get('window_size', [1280, 720])
        min_size = settings.get('min_size', [640, 480])

        return cls(
            target=data.get('target', 'macos'),
            identity=AppIdentity.from_dict(data.get('identity', {})),
            splash=SplashConfig.from_dict(data.get('splash', {})),
            editor=EditorConfig.from_dict(data.get('editor', {})),
            llm=LLMConfig.from_dict(data.get('llm', {})),
            content=ContentConfig.from_dict(data.get('content', {})),
            distribution=DistributionConfig.from_dict(data.get('distribution', {})),
            advanced=AdvancedConfig.from_dict(data.get('advanced', {})),
            output_directory=output.get('directory', '~/Desktop/builds'),
            ui=data.get('ui', 'ui.yaml'),
            main_stage=data.get('main_stage', ''),
            window_size=tuple(window_size) if isinstance(window_size, list) else window_size,
            window_title=settings.get('window_title', ''),
            resizable=settings.get('resizable', True),
            min_size=tuple(min_size) if isinstance(min_size, list) else min_size,
            fullscreen=settings.get('fullscreen', False),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> 'BuildConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to build.yaml

        Returns:
            BuildConfig instance
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Build config not found: {path}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f) or {}

        config = cls.from_dict(data)
        logger.info(f"Loaded build config from {path}")
        return config

    def to_yaml(self, path: Path):
        """
        Save configuration to YAML file.

        Args:
            path: Path to write build.yaml
        """
        path = Path(path)
        data = self.to_dict()

        # Add header comment
        yaml_content = "# NoodleStudio Build Configuration\n"
        yaml_content += "# Generated by NoodleStudio Build Settings\n\n"
        yaml_content += yaml.dump(data, default_flow_style=False, sort_keys=False)

        with open(path, 'w') as f:
            f.write(yaml_content)

        logger.info(f"Saved build config to {path}")

    @classmethod
    def default(cls, name: str = "Untitled", bundle_id: str = "") -> 'BuildConfig':
        """
        Create default configuration for a new project.

        Args:
            name: Project name
            bundle_id: Bundle identifier (auto-generated if empty)

        Returns:
            BuildConfig with sensible defaults
        """
        if not bundle_id:
            # Generate bundle ID from name
            safe_name = name.lower().replace(' ', '').replace('-', '')
            bundle_id = f"ai.noodlings.{safe_name}"

        config = cls()
        config.identity.name = name
        config.identity.bundle_id = bundle_id
        config.window_title = name
        return config

    def validate(self, project_path: Path) -> list[str]:
        """
        Validate the configuration against a project.

        Args:
            project_path: Path to project directory

        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        project_path = Path(project_path)

        if not project_path.exists():
            errors.append(f"Project path does not exist: {project_path}")
            return errors

        # Check UI file
        if self.ui:
            ui_path = project_path / self.ui
            if not ui_path.exists():
                errors.append(f"UI file not found: {self.ui}")

        # Check splash image
        if self.splash.enabled and self.splash.image:
            splash_path = project_path / self.splash.image
            if not splash_path.exists():
                errors.append(f"Splash image not found: {self.splash.image}")

        # Check icon
        if self.identity.icon:
            icon_path = project_path / self.identity.icon
            if not icon_path.exists():
                errors.append(f"Icon file not found: {self.identity.icon}")

        # Check main stage
        if self.main_stage:
            stage_path = project_path / self.main_stage
            if not stage_path.exists() and not (stage_path / "stage.yaml").exists():
                errors.append(f"Main stage not found: {self.main_stage}")

        return errors


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# Made with love. Use with love.
# Caitlyn Meeks 2026
