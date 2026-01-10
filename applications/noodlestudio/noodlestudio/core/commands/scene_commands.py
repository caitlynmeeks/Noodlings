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
#   Scene Hierarchy Commands - Undo commands for Scene Hierarchy CRUD operations
#
#   Commands for: - Creating/deleting noodling instances - Cr...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.commands.scene_commands
# PURPOSE:  Scene Commands
# LAYER:    Studio / Commands
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   CreateNoodlingCommand, DeleteNoodlingCommand, CreatePropCommand, DeletePropCommand, CreateZoneCommand
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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
        instance_id = self.instance_data.get('id', '')

        # Find and store sibling index for undo (BEFORE deleting)
        # Only store on first run - preserve for redo
        if not hasattr(self, 'deleted_sibling_index'):
            self.deleted_parent_id = None
            self.deleted_sibling_index = 0
            for nid, node in self.hierarchy.scene_graph.nodes.items():
                if node.asset_path and (instance_id in node.asset_path or self.instance_path == node.asset_path):
                    self.deleted_parent_id = node.parent_id
                    if node.parent_id:
                        parent = self.hierarchy.scene_graph.get_node(node.parent_id)
                        if parent and nid in parent.children_ids:
                            self.deleted_sibling_index = parent.children_ids.index(nid)
                    else:
                        root_ids = self.hierarchy.scene_graph.root_ids
                        if nid in root_ids:
                            self.deleted_sibling_index = root_ids.index(nid)
                    break

        # Backup folder contents before deletion
        if os.path.exists(self.instance_path):
            self.folder_backup = self._backup_folder(self.instance_path)
            shutil.rmtree(self.instance_path)

        # Remove node from scene graph so hierarchy.yaml is updated
        node_to_delete = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (instance_id in node.asset_path or self.instance_path == node.asset_path):
                node_to_delete = nid
                break
        if node_to_delete:
            self.hierarchy.scene_graph.delete_node(node_to_delete)

        # Save updated hierarchy and refresh
        self.hierarchy._save_hierarchy()
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

        # Restore original position in hierarchy
        instance_id = self.instance_data.get('id', '')
        restored_node_id = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (instance_id in node.asset_path or self.instance_path == node.asset_path):
                restored_node_id = nid
                break

        if restored_node_id and hasattr(self, 'deleted_sibling_index'):
            self.hierarchy.scene_graph.reorder_child(
                self.deleted_parent_id, restored_node_id, self.deleted_sibling_index)
            self.hierarchy._save_hierarchy()
            self.hierarchy.refresh_scene()  # Refresh to show new order

        # Select restored item
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
        prop_id = self.prop_data.get('id', '')

        # Find and store sibling index for undo (BEFORE deleting)
        # Only store on first run - preserve for redo
        if not hasattr(self, 'deleted_sibling_index'):
            self.deleted_parent_id = None
            self.deleted_sibling_index = 0
            for nid, node in self.hierarchy.scene_graph.nodes.items():
                if node.asset_path and (prop_id in node.asset_path or self.prop_path == node.asset_path):
                    self.deleted_parent_id = node.parent_id
                    if node.parent_id:
                        parent = self.hierarchy.scene_graph.get_node(node.parent_id)
                        if parent and nid in parent.children_ids:
                            self.deleted_sibling_index = parent.children_ids.index(nid)
                    else:
                        root_ids = self.hierarchy.scene_graph.root_ids
                        if nid in root_ids:
                            self.deleted_sibling_index = root_ids.index(nid)
                    break

        # Backup folder contents before deletion
        if os.path.exists(self.prop_path):
            self.folder_backup = self._backup_folder(self.prop_path)
            shutil.rmtree(self.prop_path)

        # Remove node from scene graph so hierarchy.yaml is updated
        node_to_delete = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (prop_id in node.asset_path or self.prop_path == node.asset_path):
                node_to_delete = nid
                break
        if node_to_delete:
            self.hierarchy.scene_graph.delete_node(node_to_delete)

        # Save updated hierarchy and refresh
        self.hierarchy._save_hierarchy()
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

        # Restore original position in hierarchy
        prop_id = self.prop_data.get('id', '')
        restored_node_id = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (prop_id in node.asset_path or self.prop_path == node.asset_path):
                restored_node_id = nid
                break

        if restored_node_id and hasattr(self, 'deleted_sibling_index'):
            self.hierarchy.scene_graph.reorder_child(
                self.deleted_parent_id, restored_node_id, self.deleted_sibling_index)
            self.hierarchy._save_hierarchy()
            self.hierarchy.refresh_scene()  # Refresh to show new order

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
        zone_id = self.zone_data.get('id', '')

        # Find and store sibling index for undo (BEFORE deleting)
        # Only store on first run - preserve for redo
        if not hasattr(self, 'deleted_sibling_index'):
            self.deleted_parent_id = None
            self.deleted_sibling_index = 0
            for nid, node in self.hierarchy.scene_graph.nodes.items():
                if node.asset_path and (zone_id in node.asset_path or self.zone_path == node.asset_path):
                    self.deleted_parent_id = node.parent_id
                    if node.parent_id:
                        parent = self.hierarchy.scene_graph.get_node(node.parent_id)
                        if parent and nid in parent.children_ids:
                            self.deleted_sibling_index = parent.children_ids.index(nid)
                    else:
                        root_ids = self.hierarchy.scene_graph.root_ids
                        if nid in root_ids:
                            self.deleted_sibling_index = root_ids.index(nid)
                    break

        # Backup file contents before deletion
        if os.path.exists(self.zone_path):
            with open(self.zone_path, 'rb') as f:
                self.file_backup = f.read()
            os.remove(self.zone_path)

        # Remove node from scene graph so hierarchy.yaml is updated
        node_to_delete = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (zone_id in node.asset_path or self.zone_path == node.asset_path):
                node_to_delete = nid
                break
        if node_to_delete:
            self.hierarchy.scene_graph.delete_node(node_to_delete)

        # Save updated hierarchy and refresh
        self.hierarchy._save_hierarchy()
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

        # Restore original position in hierarchy
        zone_id = self.zone_data.get('id', '')
        restored_node_id = None
        for nid, node in self.hierarchy.scene_graph.nodes.items():
            if node.asset_path and (zone_id in node.asset_path or self.zone_path == node.asset_path):
                restored_node_id = nid
                break

        if restored_node_id and hasattr(self, 'deleted_sibling_index'):
            self.hierarchy.scene_graph.reorder_child(
                self.deleted_parent_id, restored_node_id, self.deleted_sibling_index)
            self.hierarchy._save_hierarchy()
            self.hierarchy.refresh_scene()  # Refresh to show new order

        self.hierarchy._select_item_by_id(zone_id)


class BatchDeleteCommand(StudioCommand):
    """Command for deleting multiple entities at once with undo support."""

    def __init__(
        self,
        hierarchy: 'SceneHierarchy',
        entities: list
    ):
        """
        Initialize batch delete command.

        Args:
            hierarchy: Reference to SceneHierarchy panel
            entities: List of (entity_data, entity_path) tuples
        """
        count = len(entities)
        text = f"Delete {count} items"
        super().__init__(text)

        self.hierarchy = hierarchy
        self.entities = entities  # [(entity_data, path), ...]
        self.backups = []  # Will store backup data for each entity

    def _do(self):
        """Delete all entities, backing up each one first."""
        # Only create backup structure on first run - preserve position info for redo
        first_run = len(self.backups) == 0

        if first_run:
            # First run: create backup entries with position info
            for entity_data, entity_path in self.entities:
                prim_type = entity_data.get('type', 'unknown')
                entity_id = entity_data.get('id', '')
                backup_entry = {
                    'entity_data': entity_data.copy(),
                    'path': entity_path,
                    'type': prim_type,
                    'folder_backup': None,
                    'file_backup': None,
                    'parent_id': None,
                    'sibling_index': 0
                }

                # Find node in scene graph to store position
                for nid, node in self.hierarchy.scene_graph.nodes.items():
                    if node.asset_path and (entity_id in node.asset_path or entity_path == node.asset_path):
                        backup_entry['parent_id'] = node.parent_id
                        if node.parent_id:
                            parent = self.hierarchy.scene_graph.get_node(node.parent_id)
                            if parent and nid in parent.children_ids:
                                backup_entry['sibling_index'] = parent.children_ids.index(nid)
                        else:
                            root_ids = self.hierarchy.scene_graph.root_ids
                            if nid in root_ids:
                                backup_entry['sibling_index'] = root_ids.index(nid)
                        break

                self.backups.append(backup_entry)

        # Backup file contents and delete (on first run AND redo)
        for backup in self.backups:
            entity_path = backup['path']
            exists = os.path.exists(entity_path) if entity_path else False
            if entity_path and exists:
                if os.path.isdir(entity_path):
                    backup['folder_backup'] = self._backup_folder(entity_path)
                    shutil.rmtree(entity_path)
                else:
                    with open(entity_path, 'rb') as f:
                        backup['file_backup'] = f.read()
                    os.remove(entity_path)

        # Remove deleted nodes from scene graph so hierarchy.yaml is updated
        for backup in self.backups:
            entity_path = backup['path']
            entity_id = backup['entity_data'].get('id', '')
            # Find and delete the node from scene graph
            node_to_delete = None
            for nid, node in self.hierarchy.scene_graph.nodes.items():
                if node.asset_path and (entity_id in node.asset_path or entity_path == node.asset_path):
                    node_to_delete = nid
                    break
            if node_to_delete:
                self.hierarchy.scene_graph.delete_node(node_to_delete)

        # Save updated hierarchy (without deleted nodes)
        self.hierarchy._save_hierarchy()

        # Now refresh scene from updated hierarchy
        self.hierarchy.refresh_scene()

        # Clear inspector
        self.hierarchy.entitySelected.emit("", {})

    def _undo(self):
        """Restore all deleted entities."""
        for backup in self.backups:
            entity_path = backup['path']
            if not entity_path:
                continue

            if backup['folder_backup']:
                self._restore_folder(entity_path, backup['folder_backup'])
            elif backup['file_backup']:
                os.makedirs(os.path.dirname(entity_path), exist_ok=True)
                with open(entity_path, 'wb') as f:
                    f.write(backup['file_backup'])

        # Single refresh after all restorations
        self.hierarchy.refresh_scene()

        # Restore positions for all entities
        for backup in self.backups:
            entity_path = backup['path']
            entity_id = backup['entity_data'].get('id', '')
            parent_id = backup.get('parent_id')
            sibling_index = backup.get('sibling_index', 0)

            if not entity_path:
                continue

            # Find the restored node
            restored_node_id = None
            for nid, node in self.hierarchy.scene_graph.nodes.items():
                if node.asset_path and (entity_id in node.asset_path or entity_path == node.asset_path):
                    restored_node_id = nid
                    break

            if restored_node_id:
                self.hierarchy.scene_graph.reorder_child(parent_id, restored_node_id, sibling_index)

        # Save and refresh to show restored order
        self.hierarchy._save_hierarchy()
        self.hierarchy.refresh_scene()

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
