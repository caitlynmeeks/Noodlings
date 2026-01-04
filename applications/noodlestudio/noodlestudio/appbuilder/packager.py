"""
Packager - Asset collection and filtering for builds

Copies project assets to staging directory, filtering out:
- Editor-only files
- Development artifacts
- Unnecessary dependencies

Also analyzes ui.yaml to determine what's actually needed.

Author: Caitlyn + Claude
Date: January 3, 2026
"""

import logging
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
import yaml

logger = logging.getLogger(__name__)


# Directories to always exclude
EXCLUDE_DIRS = {
    '__pycache__',
    '.git',
    '.svn',
    'node_modules',
    '.pytest_cache',
    '.mypy_cache',
    'venv',
    '.venv',
    'Library',  # Unity-style library cache
    'Temp',
    'Logs',
    'logs',
    '.DS_Store',
}

# File patterns to exclude
EXCLUDE_PATTERNS = {
    '*.pyc',
    '*.pyo',
    '.DS_Store',
    'Thumbs.db',
    '*.log',
    '.gitignore',
    '.gitattributes',
    '*.bak',
    '*~',
}

# Directories that are project content (should be copied)
PROJECT_DIRS = {
    'Noodlings',
    'Stages',
    'Prims',
    'Assets',
    'Radiances',
    'Scripts',
    'facet_assemblies',
}


class Packager:
    """
    Packages project assets for building.

    Copies necessary files to a staging directory, excluding
    editor-only and development files.

    Usage:
        packager = Packager(config, staging_dir)
        result = packager.package()
    """

    def __init__(self, config: 'BuildConfig', staging_dir: Path):
        """
        Initialize packager.

        Args:
            config: Build configuration
            staging_dir: Directory to stage files into
        """
        self.config = config
        self.staging_dir = Path(staging_dir)
        self.project_dir = Path(config.project_path)

        self._progress_callback: Optional[Callable[[float, str], None]] = None
        self._copied_files: List[Path] = []
        self._referenced_assets: Set[str] = set()

    def on_progress(self, callback: Callable[[float, str], None]):
        """Register progress callback (0.0 to 1.0)."""
        self._progress_callback = callback

    def _report_progress(self, percent: float, message: str):
        """Report progress."""
        if self._progress_callback:
            self._progress_callback(percent, message)

    def package(self) -> Dict[str, Any]:
        """
        Package all project assets.

        Returns:
            Dict with:
                - success: bool
                - file_count: int
                - errors: List[str]
        """
        errors = []

        try:
            # Create staging structure
            project_staging = self.staging_dir / "project"
            project_staging.mkdir(parents=True, exist_ok=True)

            # Step 1: Analyze UI to find referenced assets (0-20%)
            self._report_progress(0.0, "Analyzing UI...")
            self._analyze_ui()
            self._report_progress(0.2, f"Found {len(self._referenced_assets)} asset references")

            # Step 2: Copy UI file (20-25%)
            self._report_progress(0.2, "Copying UI definition...")
            if self.config.ui:
                ui_src = self.project_dir / self.config.ui
                if ui_src.exists():
                    ui_dst = project_staging / self.config.ui
                    ui_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(ui_src, ui_dst)
                    self._copied_files.append(ui_dst)

            # Step 3: Copy build.yaml (25-30%)
            self._report_progress(0.25, "Copying build config...")
            build_src = self.config.build_yaml_path
            if build_src.exists():
                shutil.copy2(build_src, project_staging / "build.yaml")

            # Step 4: Copy project directories (30-80%)
            self._report_progress(0.3, "Copying project assets...")
            self._copy_project_directories(project_staging)

            # Step 5: Copy icon if specified (80-85%)
            if self.config.icon:
                self._report_progress(0.8, "Copying icon...")
                icon_src = self.project_dir / self.config.icon
                if icon_src.exists():
                    icon_dst = project_staging / self.config.icon
                    icon_dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(icon_src, icon_dst)
                    self._copied_files.append(icon_dst)

            # Step 6: Copy any referenced external assets (85-95%)
            self._report_progress(0.85, "Copying referenced assets...")
            self._copy_referenced_assets(project_staging)

            self._report_progress(1.0, f"Packaged {len(self._copied_files)} files")

            return {
                'success': True,
                'file_count': len(self._copied_files),
                'errors': []
            }

        except Exception as e:
            logger.exception(f"Packaging failed: {e}")
            return {
                'success': False,
                'file_count': len(self._copied_files),
                'errors': [str(e)]
            }

    def _analyze_ui(self):
        """
        Analyze ui.yaml to find referenced assets.

        Looks for:
        - stage references in RadianceViewport components
        - noodling references in event handlers
        - script file references
        - image assets
        """
        if not self.config.ui:
            return

        ui_path = self.project_dir / self.config.ui
        if not ui_path.exists():
            return

        try:
            with open(ui_path, 'r') as f:
                ui_data = yaml.safe_load(f) or {}

            # Recursively find asset references
            self._find_asset_refs(ui_data)

            # Also add main_stage if specified
            if self.config.main_stage:
                self._referenced_assets.add(self.config.main_stage)

        except Exception as e:
            logger.warning(f"Failed to analyze UI: {e}")

    def _find_asset_refs(self, data: Any, depth: int = 0):
        """Recursively find asset references in data structure."""
        if depth > 50:  # Prevent infinite recursion
            return

        if isinstance(data, dict):
            for key, value in data.items():
                # Check for known reference keys
                if key in ('stage', 'noodling', 'target_noodling', 'script_file', 'image', 'radiance'):
                    if isinstance(value, str) and value:
                        self._referenced_assets.add(value)

                # Recurse
                self._find_asset_refs(value, depth + 1)

        elif isinstance(data, list):
            for item in data:
                self._find_asset_refs(item, depth + 1)

    def _copy_project_directories(self, staging: Path):
        """Copy standard project directories."""
        total_dirs = len(PROJECT_DIRS)
        copied_dirs = 0

        for dir_name in PROJECT_DIRS:
            src_dir = self.project_dir / dir_name
            if src_dir.exists() and src_dir.is_dir():
                dst_dir = staging / dir_name
                self._copy_directory(src_dir, dst_dir)
                copied_dirs += 1

            # Update progress (30% to 80% range)
            progress = 0.3 + (copied_dirs / total_dirs) * 0.5
            self._report_progress(progress, f"Copied {dir_name}/")

    def _copy_directory(self, src: Path, dst: Path):
        """
        Copy directory recursively, applying filters.

        Args:
            src: Source directory
            dst: Destination directory
        """
        dst.mkdir(parents=True, exist_ok=True)

        for item in src.iterdir():
            # Skip excluded directories
            if item.is_dir():
                if item.name in EXCLUDE_DIRS:
                    continue
                self._copy_directory(item, dst / item.name)

            # Skip excluded files
            elif item.is_file():
                if self._should_exclude_file(item):
                    continue
                dst_file = dst / item.name
                shutil.copy2(item, dst_file)
                self._copied_files.append(dst_file)

    def _should_exclude_file(self, path: Path) -> bool:
        """Check if file should be excluded."""
        name = path.name

        # Check exact matches
        if name in EXCLUDE_PATTERNS:
            return True

        # Check pattern matches
        for pattern in EXCLUDE_PATTERNS:
            if pattern.startswith('*') and name.endswith(pattern[1:]):
                return True
            if pattern.endswith('*') and name.startswith(pattern[:-1]):
                return True

        return False

    def _copy_referenced_assets(self, staging: Path):
        """Copy any assets referenced in UI that aren't in standard dirs."""
        for ref in self._referenced_assets:
            # Skip if already in a project directory
            ref_path = Path(ref)
            if ref_path.parts and ref_path.parts[0] in PROJECT_DIRS:
                continue  # Already copied with directory

            # Try to find and copy the asset
            src = self.project_dir / ref
            if src.exists():
                dst = staging / ref
                dst.parent.mkdir(parents=True, exist_ok=True)

                if src.is_dir():
                    self._copy_directory(src, dst)
                else:
                    shutil.copy2(src, dst)
                    self._copied_files.append(dst)

    def get_copied_files(self) -> List[Path]:
        """Get list of all copied files."""
        return self._copied_files.copy()

    def get_referenced_assets(self) -> Set[str]:
        """Get set of referenced asset paths."""
        return self._referenced_assets.copy()


def estimate_package_size(project_path: Path) -> int:
    """
    Estimate the size of packaged assets.

    Args:
        project_path: Path to project

    Returns:
        Estimated size in bytes
    """
    total = 0

    for dir_name in PROJECT_DIRS:
        dir_path = project_path / dir_name
        if dir_path.exists():
            for f in dir_path.rglob('*'):
                if f.is_file():
                    name = f.name
                    # Skip excluded files
                    skip = False
                    for pattern in EXCLUDE_PATTERNS:
                        if pattern.startswith('*') and name.endswith(pattern[1:]):
                            skip = True
                            break
                    if not skip:
                        total += f.stat().st_size

    return total
