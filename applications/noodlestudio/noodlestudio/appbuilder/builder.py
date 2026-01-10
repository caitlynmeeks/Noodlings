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
#   Builder - Main orchestrator for the NoodleStudio build system
#
#   Coordinates the build process: 1. Validate project struct...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.appbuilder.builder
# PURPOSE:  Builder
# LAYER:    Studio / Build System
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   BuildConfig, BuildResult, Builder, create_default_build_yaml()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
import yaml

logger = logging.getLogger(__name__)


@dataclass
class BuildConfig:
    """Configuration loaded from build.yaml."""

    # Project identity
    name: str = "Untitled"
    version: str = "1.0.0"
    identifier: str = ""  # Bundle identifier (e.g., ai.noodlings.myapp)
    icon: str = ""  # Path to icon file (relative to project)

    # Entry point
    ui: str = ""  # Path to ui.yaml (the Delphi-style canvas)
    main_stage: str = ""  # Stage reference for RadianceViewport components

    # Window settings
    window_size: tuple = (1280, 720)
    window_title: str = ""  # Defaults to name
    resizable: bool = True
    min_size: tuple = (640, 480)
    fullscreen: bool = False

    # LLM settings
    llm_default_provider: str = "noodlings"  # noodlings, ollama, own_keys
    llm_allow_local: bool = True
    llm_allow_own_keys: bool = True

    # Build options
    include_renderer: str = "auto"  # auto, always, never
    compress_assets: bool = True

    # Source paths (computed)
    project_path: Path = field(default_factory=Path)
    build_yaml_path: Path = field(default_factory=Path)

    @staticmethod
    def load(project_path: Path) -> 'BuildConfig':
        """
        Load build configuration from project's build.yaml.

        Args:
            project_path: Path to project directory

        Returns:
            BuildConfig instance

        Raises:
            FileNotFoundError: If project or build.yaml not found
            ValueError: If build.yaml is invalid
        """
        project_path = Path(project_path)

        if not project_path.exists():
            raise FileNotFoundError(f"Project not found: {project_path}")

        if not project_path.is_dir():
            raise ValueError(f"Project path is not a directory: {project_path}")

        # Look for build.yaml
        build_yaml = project_path / "build.yaml"
        if not build_yaml.exists():
            # Fall back to project.yaml
            build_yaml = project_path / "project.yaml"
            if not build_yaml.exists():
                raise FileNotFoundError(
                    f"No build.yaml or project.yaml found in {project_path}"
                )

        # Load YAML
        with open(build_yaml, 'r') as f:
            data = yaml.safe_load(f) or {}

        config = BuildConfig()
        config.project_path = project_path
        config.build_yaml_path = build_yaml

        # Identity
        config.name = data.get('name', project_path.name)
        config.version = data.get('version', '1.0.0')
        config.identifier = data.get('identifier', f"ai.noodlings.{config.name.lower().replace(' ', '')}")
        config.icon = data.get('icon', '')

        # Entry point
        config.ui = data.get('ui', 'ui.yaml')
        config.main_stage = data.get('main_stage', '')

        # Window settings
        settings = data.get('settings', {})
        window_size = settings.get('window_size', [1280, 720])
        config.window_size = tuple(window_size) if isinstance(window_size, list) else window_size
        config.window_title = settings.get('window_title', config.name)
        config.resizable = settings.get('resizable', True)
        min_size = settings.get('min_size', [640, 480])
        config.min_size = tuple(min_size) if isinstance(min_size, list) else min_size
        config.fullscreen = settings.get('fullscreen', False)

        # LLM settings
        llm = data.get('llm', {})
        config.llm_default_provider = llm.get('default_provider', 'noodlings')
        config.llm_allow_local = llm.get('allow_local', True)
        config.llm_allow_own_keys = llm.get('allow_own_keys', True)

        # Build options
        build = data.get('build', {})
        config.include_renderer = build.get('include_renderer', 'auto')
        config.compress_assets = build.get('compress_assets', True)

        logger.info(f"Loaded build config: {config.name} v{config.version}")
        return config

    def validate(self) -> List[str]:
        """
        Validate the build configuration.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check project path
        if not self.project_path.exists():
            errors.append(f"Project path does not exist: {self.project_path}")
            return errors

        # Check UI file
        if self.ui:
            ui_path = self.project_path / self.ui
            if not ui_path.exists():
                errors.append(f"UI file not found: {self.ui}")

        # Check main stage
        if self.main_stage:
            stage_path = self.project_path / self.main_stage
            if not stage_path.exists():
                # Try with stage.yaml
                if not (stage_path / "stage.yaml").exists():
                    errors.append(f"Main stage not found: {self.main_stage}")

        # Check icon
        if self.icon:
            icon_path = self.project_path / self.icon
            if not icon_path.exists():
                errors.append(f"Icon file not found: {self.icon}")

        return errors


@dataclass
class BuildResult:
    """Result of a build operation."""

    success: bool = False
    output_path: Path = field(default_factory=Path)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Stats
    total_files: int = 0
    total_size_bytes: int = 0
    build_time_seconds: float = 0.0


class Builder:
    """
    Main build orchestrator.

    Coordinates the entire build process:
    1. Validate project and config
    2. Package assets
    3. Create platform-specific bundle
    4. Clean up temporary files

    Usage:
        config = BuildConfig.load("/path/to/project")
        builder = Builder(config)

        # Optional: register progress callback
        builder.on_progress(lambda p, msg: print(f"{p}%: {msg}"))

        result = builder.build("/path/to/output.app")
    """

    def __init__(self, config: BuildConfig):
        """
        Initialize builder with configuration.

        Args:
            config: BuildConfig loaded from project
        """
        self.config = config
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._temp_dir: Optional[Path] = None

    def on_progress(self, callback: Callable[[int, str], None]):
        """
        Register progress callback.

        Args:
            callback: Function called with (percent: int, message: str)
        """
        self._progress_callback = callback

    def _report_progress(self, percent: int, message: str):
        """Report progress to callback if registered."""
        logger.info(f"[Build {percent}%] {message}")
        if self._progress_callback:
            self._progress_callback(percent, message)

    def build(self, output_path: str, platform: str = "macos") -> BuildResult:
        """
        Build the project into a standalone application.

        Args:
            output_path: Where to create the output (e.g., ~/Desktop/MyApp.app)
            platform: Target platform ("macos" for now)

        Returns:
            BuildResult with success status and details
        """
        import time
        start_time = time.time()

        result = BuildResult()
        output_path = Path(output_path)

        try:
            # Step 1: Validate (0-10%)
            self._report_progress(0, "Validating project...")
            errors = self.config.validate()
            if errors:
                result.errors = errors
                return result

            self._report_progress(10, "Project validated")

            # Step 2: Create temp directory for staging
            self._temp_dir = Path(tempfile.mkdtemp(prefix="noodlebuild_"))
            logger.info(f"Build staging directory: {self._temp_dir}")

            # Step 3: Package assets (10-50%)
            self._report_progress(15, "Packaging assets...")
            package_result = self._package_assets()
            if not package_result['success']:
                result.errors = package_result.get('errors', ['Asset packaging failed'])
                return result

            result.total_files = package_result.get('file_count', 0)
            self._report_progress(50, f"Packaged {result.total_files} files")

            # Step 4: Create bundle (50-90%)
            self._report_progress(55, f"Creating {platform} bundle...")
            if platform == "macos":
                bundle_result = self._create_macos_bundle(output_path)
            else:
                result.errors = [f"Unsupported platform: {platform}"]
                return result

            if not bundle_result['success']:
                result.errors = bundle_result.get('errors', ['Bundle creation failed'])
                return result

            self._report_progress(90, "Bundle created")

            # Step 5: Finalize (90-100%)
            self._report_progress(95, "Finalizing...")

            # Get output size
            if output_path.exists():
                result.total_size_bytes = self._get_dir_size(output_path)

            result.success = True
            result.output_path = output_path
            result.build_time_seconds = time.time() - start_time

            self._report_progress(100, f"Build complete: {output_path}")

        except Exception as e:
            logger.exception(f"Build failed: {e}")
            result.errors.append(str(e))

        finally:
            # Clean up temp directory
            if self._temp_dir and self._temp_dir.exists():
                try:
                    shutil.rmtree(self._temp_dir)
                except Exception as e:
                    logger.warning(f"Failed to clean temp dir: {e}")

        return result

    def _package_assets(self) -> Dict[str, Any]:
        """
        Package project assets into staging directory.

        Returns:
            Dict with success status and file count
        """
        from .packager import Packager

        packager = Packager(self.config, self._temp_dir)
        packager.on_progress(lambda p, m: self._report_progress(15 + int(p * 0.35), m))

        return packager.package()

    def _create_macos_bundle(self, output_path: Path) -> Dict[str, Any]:
        """
        Create macOS .app bundle.

        Args:
            output_path: Where to create the .app

        Returns:
            Dict with success status
        """
        from .bundler_macos import MacOSBundler

        bundler = MacOSBundler(self.config, self._temp_dir)
        bundler.on_progress(lambda p, m: self._report_progress(55 + int(p * 0.35), m))

        return bundler.bundle(output_path)

    def _get_dir_size(self, path: Path) -> int:
        """Get total size of directory in bytes."""
        total = 0
        for f in path.rglob('*'):
            if f.is_file():
                total += f.stat().st_size
        return total


def create_default_build_yaml(project_path: Path, name: str = "") -> Path:
    """
    Create a default build.yaml for a project.

    Args:
        project_path: Path to project directory
        name: Project name (defaults to directory name)

    Returns:
        Path to created build.yaml
    """
    if not name:
        name = project_path.name

    build_yaml = project_path / "build.yaml"

    content = f"""# NoodleStudio Build Configuration
# Generated by NoodleStudio

name: "{name}"
version: "1.0.0"

# UI canvas definition (the application interface)
ui: "ui.yaml"

# Stage for RadianceViewport components (optional)
# main_stage: "Stages/main"

settings:
  window_size: [1280, 720]
  resizable: true

llm:
  default_provider: noodlings  # noodlings, ollama, or own_keys
  allow_local: true
  allow_own_keys: true
"""

    with open(build_yaml, 'w') as f:
        f.write(content)

    logger.info(f"Created default build.yaml: {build_yaml}")
    return build_yaml

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
