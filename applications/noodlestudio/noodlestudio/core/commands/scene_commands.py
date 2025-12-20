"""
Scene Hierarchy Commands - Undo commands for Scene Hierarchy CRUD operations

Commands for:
- Creating/deleting noodling instances
- Creating/deleting props
- Creating/deleting zones

Author: Commander Spock + Cadet Caity
Date: December 19, 2025
"""

import os
import shutil
import yaml
from typing import Dict, Any, Optional, TYPE_CHECKING

from .base_command import StudioCommand

if TYPE_CHECKING:
    from ...panels.scene_hierarchy import SceneHierarchy


class CreateNoodlingCommand(StudioCommand):
    """Command for creating a new noodling instance in a stage."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        instance_path: str,
        instance_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize create noodling command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            instance_path: Path to instance folder
            instance_data: Instance YAML data to write
            display_name: Display name for undo text
        """
        text = f"Create '{display_name}'" if display_name else "Create Noodling"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.instance_path = instance_path
        self.instance_data = instance_data

    def _do(self):
        """Create the noodling instance."""
        # Create folder
        os.makedirs(self.instance_path, exist_ok=True)

        # Write instance.yaml
        instance_yaml = os.path.join(self.instance_path, "instance.yaml")
        with open(instance_yaml, 'w') as f:
            yaml.dump(self.instance_data, f, default_flow_style=False)

        # Refresh hierarchy
        self.hierarchy.refresh_scene()

        # Select the new item
        instance_id = self.instance_data.get('id', '')
        self.hierarchy._select_item_by_id(f"agent_{instance_id}")

    def _undo(self):
        """Delete the created noodling instance."""
        if os.path.exists(self.instance_path):
            shutil.rmtree(self.instance_path)

        self.hierarchy.refresh_scene()


class DeleteNoodlingCommand(StudioCommand):
    """Command for deleting a noodling instance."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        instance_path: str,
        instance_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize delete noodling command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            instance_path: Path to instance folder
            instance_data: Full instance data for restoration
            display_name: Display name for undo text
        """
        text = f"Delete '{display_name}'" if display_name else "Delete Noodling"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.instance_path = instance_path
        self.instance_data = instance_data
        # Store entire folder contents for undo
        self.folder_backup = None

    def _do(self):
        """Delete the noodling instance."""
        # Backup folder contents before deletion
        if os.path.exists(self.instance_path):
            self.folder_backup = self._backup_folder(self.instance_path)
            shutil.rmtree(self.instance_path)

        self.hierarchy.refresh_scene()
        # Clear Inspector (entity was deleted)
        self.hierarchy.entitySelected.emit("", {})

    def _undo(self):
        """Restore the deleted noodling instance."""
        if self.folder_backup:
            self._restore_folder(self.instance_path, self.folder_backup)
        else:
            # Fallback: recreate from instance_data
            os.makedirs(self.instance_path, exist_ok=True)
            instance_yaml = os.path.join(self.instance_path, "instance.yaml")
            with open(instance_yaml, 'w') as f:
                yaml.dump(self.instance_data, f, default_flow_style=False)

        self.hierarchy.refresh_scene()

        # Select restored item
        instance_id = self.instance_data.get('id', '')
        self.hierarchy._select_item_by_id(f"agent_{instance_id}")

    def _backup_folder(self, folder_path: str) -> Dict[str, Any]:
        """Backup folder contents to memory."""
        backup = {'files': {}, 'dirs': []}
        for root, dirs, files in os.walk(folder_path):
            rel_root = os.path.relpath(root, folder_path)
            if rel_root != '.':
                backup['dirs'].append(rel_root)
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, folder_path)
                with open(file_path, 'rb') as f:
                    backup['files'][rel_path] = f.read()
        return backup

    def _restore_folder(self, folder_path: str, backup: Dict[str, Any]):
        """Restore folder contents from backup."""
        os.makedirs(folder_path, exist_ok=True)
        # Create subdirectories
        for dir_path in backup.get('dirs', []):
            os.makedirs(os.path.join(folder_path, dir_path), exist_ok=True)
        # Restore files
        for rel_path, contents in backup.get('files', {}).items():
            file_path = os.path.join(folder_path, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(contents)


class CreatePropCommand(StudioCommand):
    """Command for creating a new prop in a stage."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        prop_path: str,
        prop_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize create prop command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            prop_path: Path to prop folder
            prop_data: Prop YAML data to write
            display_name: Display name for undo text
        """
        text = f"Create '{display_name}'" if display_name else "Create Prop"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.prop_path = prop_path
        self.prop_data = prop_data

    def _do(self):
        """Create the prop."""
        # Create folder
        os.makedirs(self.prop_path, exist_ok=True)

        # Write prop.yaml
        prop_yaml = os.path.join(self.prop_path, "prop.yaml")
        with open(prop_yaml, 'w') as f:
            yaml.dump(self.prop_data, f, default_flow_style=False)

        # Refresh hierarchy
        self.hierarchy.refresh_scene()

        # Select the new item
        prop_id = self.prop_data.get('id', '')
        self.hierarchy._select_item_by_id(f"prop_{prop_id}")

    def _undo(self):
        """Delete the created prop."""
        if os.path.exists(self.prop_path):
            shutil.rmtree(self.prop_path)

        self.hierarchy.refresh_scene()


class DeletePropCommand(StudioCommand):
    """Command for deleting a prop."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        prop_path: str,
        prop_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize delete prop command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            prop_path: Path to prop folder
            prop_data: Full prop data for restoration
            display_name: Display name for undo text
        """
        text = f"Delete '{display_name}'" if display_name else "Delete Prop"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.prop_path = prop_path
        self.prop_data = prop_data
        self.folder_backup = None

    def _do(self):
        """Delete the prop."""
        if os.path.exists(self.prop_path):
            self.folder_backup = self._backup_folder(self.prop_path)
            shutil.rmtree(self.prop_path)

        self.hierarchy.refresh_scene()
        self.hierarchy.entitySelected.emit("", {})

    def _undo(self):
        """Restore the deleted prop."""
        if self.folder_backup:
            self._restore_folder(self.prop_path, self.folder_backup)
        else:
            os.makedirs(self.prop_path, exist_ok=True)
            prop_yaml = os.path.join(self.prop_path, "prop.yaml")
            with open(prop_yaml, 'w') as f:
                yaml.dump(self.prop_data, f, default_flow_style=False)

        self.hierarchy.refresh_scene()

        prop_id = self.prop_data.get('id', '')
        self.hierarchy._select_item_by_id(f"prop_{prop_id}")

    def _backup_folder(self, folder_path: str) -> Dict[str, Any]:
        """Backup folder contents to memory."""
        backup = {'files': {}, 'dirs': []}
        for root, dirs, files in os.walk(folder_path):
            rel_root = os.path.relpath(root, folder_path)
            if rel_root != '.':
                backup['dirs'].append(rel_root)
            for filename in files:
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, folder_path)
                with open(file_path, 'rb') as f:
                    backup['files'][rel_path] = f.read()
        return backup

    def _restore_folder(self, folder_path: str, backup: Dict[str, Any]):
        """Restore folder contents from backup."""
        os.makedirs(folder_path, exist_ok=True)
        for dir_path in backup.get('dirs', []):
            os.makedirs(os.path.join(folder_path, dir_path), exist_ok=True)
        for rel_path, contents in backup.get('files', {}).items():
            file_path = os.path.join(folder_path, rel_path)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(contents)


class CreateZoneCommand(StudioCommand):
    """Command for creating a new zone in a stage."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        zone_path: str,
        zone_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize create zone command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            zone_path: Path to zone YAML file
            zone_data: Zone YAML data to write
            display_name: Display name for undo text
        """
        text = f"Create '{display_name}'" if display_name else "Create Zone"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.zone_path = zone_path
        self.zone_data = zone_data

    def _do(self):
        """Create the zone."""
        # Ensure zones directory exists
        os.makedirs(os.path.dirname(self.zone_path), exist_ok=True)

        # Write zone YAML
        with open(self.zone_path, 'w') as f:
            yaml.dump(self.zone_data, f, default_flow_style=False)

        # Refresh hierarchy
        self.hierarchy.refresh_scene()

        # Select the new item
        zone_id = self.zone_data.get('id', '')
        self.hierarchy._select_item_by_id(zone_id)

    def _undo(self):
        """Delete the created zone."""
        if os.path.exists(self.zone_path):
            os.remove(self.zone_path)

        self.hierarchy.refresh_scene()


class DeleteZoneCommand(StudioCommand):
    """Command for deleting a zone."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        zone_path: str,
        zone_data: Dict[str, Any],
        display_name: str = ""
    ):
        """
        Initialize delete zone command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            zone_path: Path to zone YAML file
            zone_data: Full zone data for restoration
            display_name: Display name for undo text
        """
        text = f"Delete '{display_name}'" if display_name else "Delete Zone"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.zone_path = zone_path
        self.zone_data = zone_data
        self.file_backup = None

    def _do(self):
        """Delete the zone."""
        if os.path.exists(self.zone_path):
            # Backup file contents
            with open(self.zone_path, 'rb') as f:
                self.file_backup = f.read()
            os.remove(self.zone_path)

        self.hierarchy.refresh_scene()
        self.hierarchy.entitySelected.emit("", {})

    def _undo(self):
        """Restore the deleted zone."""
        # Ensure zones directory exists
        os.makedirs(os.path.dirname(self.zone_path), exist_ok=True)

        if self.file_backup:
            with open(self.zone_path, 'wb') as f:
                f.write(self.file_backup)
        else:
            with open(self.zone_path, 'w') as f:
                yaml.dump(self.zone_data, f, default_flow_style=False)

        self.hierarchy.refresh_scene()

        zone_id = self.zone_data.get('id', '')
        self.hierarchy._select_item_by_id(zone_id)
