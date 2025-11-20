"""
Project Manager - Handles project creation, loading, and structure.

Similar to Unity's project system - each project is a folder with:
- project.noodleproj (metadata)
- Assets/ (all project assets)
- Temp/ (temporary files)
- Library/ (cached data)
"""

import os
import json
from pathlib import Path
from typing import Optional
from PyQt6.QtCore import QObject, pyqtSignal


class ProjectManager(QObject):
    """
    Manages NoodleStudio projects.

    Signals:
        projectOpened: Emitted when a project is opened (path: str)
        projectClosed: Emitted when a project is closed
    """

    projectOpened = pyqtSignal(str)
    projectClosed = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_project_path: Optional[str] = None
        self.current_project_name: Optional[str] = None

    def create_project(self, parent_dir: str, project_name: str) -> bool:
        """
        Create a new NoodleStudio project.

        Args:
            parent_dir: Parent directory where project folder will be created
            project_name: Name of the project

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create project directory
            project_path = os.path.join(parent_dir, project_name)
            if os.path.exists(project_path):
                return False  # Project already exists

            os.makedirs(project_path)

            # Create subdirectories
            os.makedirs(os.path.join(project_path, "Assets", "Noodlings"))
            os.makedirs(os.path.join(project_path, "Assets", "Ensembles"))
            os.makedirs(os.path.join(project_path, "Assets", "Prims"))
            os.makedirs(os.path.join(project_path, "Assets", "Scripts"))
            os.makedirs(os.path.join(project_path, "Assets", "Stages"))
            os.makedirs(os.path.join(project_path, "Temp"))
            os.makedirs(os.path.join(project_path, "Library"))

            # Create project metadata file
            metadata = {
                "name": project_name,
                "version": "1.0.0",
                "created": self._get_timestamp(),
                "noodlestudio_version": "0.1.0",
                "description": ""
            }

            metadata_path = os.path.join(project_path, "project.noodleproj")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Create .gitignore
            gitignore_content = """# NoodleStudio
Temp/
Library/
*.log

# OS
.DS_Store
Thumbs.db
"""
            gitignore_path = os.path.join(project_path, ".gitignore")
            with open(gitignore_path, 'w') as f:
                f.write(gitignore_content)

            # Open the new project
            self.open_project(project_path)

            return True

        except Exception as e:
            print(f"Error creating project: {e}")
            return False

    def open_project(self, project_path: str) -> bool:
        """
        Open an existing project.

        Args:
            project_path: Path to project directory

        Returns:
            True if successful, False otherwise
        """
        try:
            # Verify it's a valid project
            metadata_path = os.path.join(project_path, "project.noodleproj")
            if not os.path.exists(metadata_path):
                return False

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Close current project if any
            if self.current_project_path:
                self.close_project()

            # Set current project
            self.current_project_path = project_path
            self.current_project_name = metadata.get("name", os.path.basename(project_path))

            # Update last opened timestamp
            metadata["last_opened"] = self._get_timestamp()
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            # Emit signal
            self.projectOpened.emit(project_path)

            return True

        except Exception as e:
            print(f"Error opening project: {e}")
            return False

    def close_project(self):
        """Close the current project and shutdown the server."""
        if self.current_project_path:
            # Trigger graceful server shutdown
            try:
                import requests
                # Use a very short delay (1 second) for project switches
                requests.post('http://localhost:8081/api/shutdown', json={'delay': 1}, timeout=2)
                print("Server shutdown initiated...")
            except Exception as e:
                print(f"Warning: Could not shutdown server: {e}")

            self.current_project_path = None
            self.current_project_name = None
            self.projectClosed.emit()

    def get_assets_path(self, asset_type: str = "") -> Optional[str]:
        """
        Get path to Assets directory or specific asset type.

        Args:
            asset_type: Optional asset type (Noodlings, Ensembles, etc.)

        Returns:
            Path to assets directory or None if no project open
        """
        if not self.current_project_path:
            return None

        assets_path = os.path.join(self.current_project_path, "Assets")

        if asset_type:
            return os.path.join(assets_path, asset_type)
        return assets_path

    def import_ensemble(self, source_path: str) -> bool:
        """
        Import an ensemble file into the current project.

        Args:
            source_path: Path to .ensemble file to import

        Returns:
            True if successful, False otherwise
        """
        if not self.current_project_path:
            return False

        try:
            import shutil
            dest_dir = self.get_assets_path("Ensembles")
            filename = os.path.basename(source_path)
            dest_path = os.path.join(dest_dir, filename)

            shutil.copy2(source_path, dest_path)
            return True

        except Exception as e:
            print(f"Error importing ensemble: {e}")
            return False

    def import_noodling(self, source_path: str) -> bool:
        """
        Import a noodling recipe into the current project.

        Args:
            source_path: Path to .json recipe file to import

        Returns:
            True if successful, False otherwise
        """
        if not self.current_project_path:
            return False

        try:
            import shutil
            dest_dir = self.get_assets_path("Noodlings")
            filename = os.path.basename(source_path)
            dest_path = os.path.join(dest_dir, filename)

            shutil.copy2(source_path, dest_path)
            return True

        except Exception as e:
            print(f"Error importing noodling: {e}")
            return False

    def _get_timestamp(self) -> str:
        """Get current timestamp as ISO string."""
        from datetime import datetime
        return datetime.now().isoformat()

    def is_project_open(self) -> bool:
        """Check if a project is currently open."""
        return self.current_project_path is not None
