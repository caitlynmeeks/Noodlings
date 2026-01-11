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

# Use the canonical BuildConfig from core
from ..core.build_config import BuildConfig

logger = logging.getLogger(__name__)


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
        config = BuildConfig.from_yaml(project_path / "build.yaml")
        builder = Builder(config, project_path)

        # Optional: register progress callback
        builder.on_progress(lambda p, msg: print(f"{p}%: {msg}"))

        result = builder.build("/path/to/output.app")
    """

    def __init__(self, config: BuildConfig, project_path: Path):
        """
        Initialize builder with configuration.

        Args:
            config: BuildConfig loaded from project's build.yaml
            project_path: Path to the project directory
        """
        self.config = config
        self.project_path = Path(project_path)
        self._progress_callback: Optional[Callable[[int, str], None]] = None
        self._temp_dir: Optional[Path] = None
        self._cancelled = False

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

    def cancel(self):
        """Request build cancellation."""
        self._cancelled = True
        logger.info("Build cancellation requested")

    def build(self, output_path: str = "", platform: str = "") -> BuildResult:
        """
        Build the project into a standalone application.

        Args:
            output_path: Where to create the output (defaults to config.output_directory)
            platform: Target platform (defaults to config.target)

        Returns:
            BuildResult with success status and details
        """
        import time
        import os
        start_time = time.time()
        self._cancelled = False

        result = BuildResult()

        # Use config values if not provided
        if not platform:
            platform = self.config.target or "macos"

        if not output_path:
            output_dir = os.path.expanduser(self.config.output_directory or "~/Desktop/builds")
            app_name = self.config.identity.name or "Untitled"
            output_path = os.path.join(output_dir, f"{app_name}.app")

        output_path = Path(output_path)

        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Step 1: Validate (0-10%)
            self._report_progress(0, "Validating project...")
            errors = self.config.validate(self.project_path)
            if errors:
                result.errors = errors
                return result

            if self._cancelled:
                result.errors = ["Build cancelled"]
                return result

            self._report_progress(10, "Project validated")

            # Step 2: Create temp directory for staging
            self._temp_dir = Path(tempfile.mkdtemp(prefix="noodlebuild_"))
            logger.info(f"Build staging directory: {self._temp_dir}")

            # Step 3: Package assets (10-50%)
            if self._cancelled:
                result.errors = ["Build cancelled"]
                return result

            self._report_progress(15, "Packaging assets...")
            package_result = self._package_assets()
            if not package_result['success']:
                result.errors = package_result.get('errors', ['Asset packaging failed'])
                return result

            result.total_files = package_result.get('file_count', 0)
            self._report_progress(50, f"Packaged {result.total_files} files")

            # Step 4: Create bundle (50-90%)
            if self._cancelled:
                result.errors = ["Build cancelled"]
                return result

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
            if self._cancelled:
                result.errors = ["Build cancelled"]
                # Clean up partial build
                if output_path.exists():
                    shutil.rmtree(output_path)
                return result

            self._report_progress(95, "Finalizing...")

            # Get output size
            if output_path.exists():
                result.total_size_bytes = self._get_dir_size(output_path)

            result.success = True
            result.output_path = output_path
            result.build_time_seconds = time.time() - start_time

            app_name = self.config.identity.name or "App"
            self._report_progress(100, f"Build complete: {app_name}")

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

        packager = Packager(self.config, self.project_path, self._temp_dir)
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

        bundler = MacOSBundler(self.config, self.project_path, self._temp_dir)
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
    project_path = Path(project_path)
    if not name:
        name = project_path.name

    # Create config using the canonical BuildConfig.default()
    config = BuildConfig.default(name=name)
    yaml_path = project_path / "build.yaml"
    config.to_yaml(yaml_path)

    logger.info(f"Created default build.yaml: {yaml_path}")
    return yaml_path

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
