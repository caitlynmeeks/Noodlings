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
#   Scene Hierarchy Create Mixin - Entity creation (noodlings, props, zones, folders)
#
#   Contains: - create_folder_under: Create folder in hierarc...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.scene_hierarchy_create_mixin
# PURPOSE:  Scene Hierarchy Create Mixin
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SceneHierarchyCreateMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os
import uuid

from PyQt6.QtWidgets import QTreeWidgetItem, QMessageBox, QInputDialog, QLineEdit
from PyQt6.QtCore import Qt

from ..core.undo_manager import UndoManager
from ..core.commands.scene_commands import (
    CreateNoodlingCommand, CreatePropCommand, CreateZoneCommand
)
from ..core.scene_node import SceneNodeType


class SceneHierarchyCreateMixin:
    """Mixin providing entity creation for SceneHierarchy."""

    # =========================================================================
    # Folder Management (Unity-style hierarchy)
    # =========================================================================

    def create_folder_under(self, parent_id: str = None):
        """Create a new folder under the specified parent (or at root if None)."""
        if not self.project_manager or not self.project_manager.is_project_open():
            self._prompt_open_project()
            return

        # Suppress refresh during folder creation to avoid recursion
        self._suppress_refresh = True
        folder_node_id = None
        try:
            # Generate unique folder name
            base_name = "New Folder"
            name = self.scene_graph.get_unique_name(base_name, parent_id)

            # Create folder node
            folder_node = self.scene_graph.create_folder(name, parent_id)
            folder_node.node_type = SceneNodeType.FOLDER  # Ensure it's marked as folder
            folder_node_id = folder_node.id

            # Save hierarchy
            self._save_hierarchy()
            print(f"Created folder: {name}")
        finally:
            self._suppress_refresh = False

        # Now refresh and select (with suppression off)
        self.refresh_scene()
        if folder_node_id:
            self._select_node_by_id(folder_node_id)

    def rename_node(self, node_id: str, item: QTreeWidgetItem = None):
        """Rename a node via inline editing or dialog."""
        node = self.scene_graph.get_node(node_id)
        if not node:
            return

        if node.is_virtual:
            QMessageBox.warning(self, "Cannot Rename", "Cannot rename skeleton bones.")
            return

        # Use dialog for rename
        new_name, ok = QInputDialog.getText(
            self,
            "Rename",
            "Enter new name:",
            QLineEdit.EchoMode.Normal,
            node.name
        )

        if ok and new_name and new_name != node.name:
            self.scene_graph.rename_node(node_id, new_name)
            # Update tree item directly (don't refresh - it destroys user hierarchy)
            if item:
                item.setText(0, new_name)
            self._save_hierarchy()
            print(f"Renamed to: {new_name}")

    def delete_folder(self, node_id: str):
        """Delete a folder and optionally its contents."""
        node = self.scene_graph.get_node(node_id)
        if not node:
            return

        if node.is_virtual:
            QMessageBox.warning(self, "Cannot Delete", "Cannot delete skeleton folders.")
            return

        # Check if folder has children
        children = self.scene_graph.get_children(node_id)
        if children:
            reply = QMessageBox.question(
                self,
                "Delete Folder",
                f"Delete folder '{node.name}' and all its contents ({len(children)} items)?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
        else:
            reply = QMessageBox.question(
                self,
                "Delete Folder",
                f"Delete empty folder '{node.name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # Delete the folder
        self.scene_graph.delete_node(node_id, recursive=True)
        self._save_hierarchy()
        self.refresh_scene()
        print(f"Deleted folder: {node.name}")

    def _select_node_by_id(self, node_id: str):
        """Select a node in the tree by its ID."""
        item = self._node_id_to_item.get(node_id)
        if item:
            self.tree.clearSelection()
            item.setSelected(True)
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item)

            # Expand parent chain to make visible
            parent = item.parent()
            while parent:
                parent.setExpanded(True)
                parent = parent.parent()

    # =========================================================================
    # Entity Creation
    # =========================================================================

    def create_empty_noodling(self):
        """Create a new Noodling instance in the current stage (no dialog)."""
        if not self.project_manager or not self.project_manager.is_project_open():
            self._prompt_open_project()
            return

        if not self.current_stage:
            print("No stage selected - cannot create noodling")
            return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            print(f"Stage path not found for {self.current_stage}")
            return

        # Get Instances directory (will be created by command if needed)
        instances_dir = os.path.join(stage_path, "Instances")

        # Generate unique name: "New Noodling", "New Noodling (2)", etc.
        # Need to create dir first so we can check existing names
        os.makedirs(instances_dir, exist_ok=True)
        display_name = self._generate_unique_name(instances_dir, "New Noodling", "instance.yaml")

        # Use UUID for folder name (the actual identifier)
        instance_id = str(uuid.uuid4())
        instance_path = os.path.join(instances_dir, instance_id)

        # Prepare instance data
        instance_data = {
            'id': instance_id,
            'noodling': 'empty_noodling',  # Reference to Library/Noodlings/empty_noodling
            'overrides': {
                'name': display_name,
                'zone': 'default',
                'position': [0, 0, 0],
                'rotation': [0, 0, 0],
            }
        }

        # Push undo command (which creates the files)
        cmd = CreateNoodlingCommand(
            hierarchy=self,
            instance_path=instance_path,
            instance_data=instance_data,
            display_name=display_name
        )
        UndoManager().push(cmd)
        print(f"Created Noodling: {display_name} ({instance_id[:8]}...)")

    def create_empty_prim(self):
        """Create a new Prop in the current stage (no dialog)."""
        if not self.project_manager or not self.project_manager.is_project_open():
            self._prompt_open_project()
            return

        if not self.current_stage:
            print("No stage selected - cannot create prop")
            return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            print(f"Stage path not found for {self.current_stage}")
            return

        # Get Props directory (will be created by command if needed)
        props_dir = os.path.join(stage_path, "Props")

        # Generate unique name: "New Prim", "New Prim (2)", etc.
        # Need to create dir first so we can check existing names
        os.makedirs(props_dir, exist_ok=True)
        display_name = self._generate_unique_name(props_dir, "New Prim", "prop.yaml")

        # Use UUID for folder name (the actual identifier)
        prop_id = str(uuid.uuid4())
        prop_path = os.path.join(props_dir, prop_id)

        # Prepare prop data with default physics properties
        prop_data = {
            'id': prop_id,
            'name': display_name,
            'description': 'A newly created prop.',
            'zone': 'default',
            'position': [0, 0, 0],
            'rotation': [0, 0, 0],
            'scale': [1, 1, 1],
            # Physics properties (SPE)
            'mass': 'medium',
            'material': 'unknown',
            'friction': 'medium',
            'elasticity': 'normal',
            'softness': 'normal',
        }

        # Push undo command (which creates the files)
        cmd = CreatePropCommand(
            hierarchy=self,
            prop_path=prop_path,
            prop_data=prop_data,
            display_name=display_name
        )
        UndoManager().push(cmd)
        print(f"Created Prop: {display_name} ({prop_id[:8]}...)")

    def create_empty_zone(self):
        """Create a new Zone in the current stage (no dialog)."""
        if not self.project_manager or not self.project_manager.is_project_open():
            self._prompt_open_project()
            return

        if not self.current_stage:
            print("No stage selected - cannot create zone")
            return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            print(f"Stage path not found for {self.current_stage}")
            return

        # Get Zones directory (will be created by command if needed)
        zones_dir = os.path.join(stage_path, "Zones")

        # Generate unique name: "New Zone", "New Zone (2)", etc.
        # Need to create dir first so we can check existing names
        os.makedirs(zones_dir, exist_ok=True)
        display_name = self._generate_unique_zone_name(zones_dir, "New Zone")

        # Use UUID for the zone file
        zone_id = str(uuid.uuid4())
        zone_filename = f"{zone_id}.zone.yaml"
        zone_path = os.path.join(zones_dir, zone_filename)

        # Prepare zone data
        zone_data = {
            'id': zone_id,
            'name': display_name,
            'description': 'A newly created zone.',
            'center': [0, 0, 0],
            'radius': 10,
            'falloff': 5,
            'shape': 'sphere',
        }

        # Push undo command (which creates the file)
        cmd = CreateZoneCommand(
            hierarchy=self,
            zone_path=zone_path,
            zone_data=zone_data,
            display_name=display_name
        )
        UndoManager().push(cmd)
        print(f"Created Zone: {display_name} ({zone_id[:8]}...)")

    def _generate_unique_name(self, directory: str, base_name: str, yaml_file: str) -> str:
        """Generate unique name like 'New Prop', 'New Prop (2)', etc."""
        import yaml

        existing_names = set()

        # Scan existing items for their display names
        if os.path.exists(directory):
            for item_name in os.listdir(directory):
                item_path = os.path.join(directory, item_name)
                if os.path.isdir(item_path):
                    yaml_path = os.path.join(item_path, yaml_file)
                    if os.path.exists(yaml_path):
                        try:
                            with open(yaml_path, 'r') as f:
                                data = yaml.safe_load(f) or {}
                            # Check both direct 'name' and 'overrides.name'
                            name = data.get('name') or data.get('overrides', {}).get('name', '')
                            if name:
                                existing_names.add(name)
                        except:
                            pass

        # Find first available name
        if base_name not in existing_names:
            return base_name

        counter = 2
        while f"{base_name} ({counter})" in existing_names:
            counter += 1

        return f"{base_name} ({counter})"

    def _generate_unique_zone_name(self, zones_dir: str, base_name: str) -> str:
        """Generate unique zone name like 'New Zone', 'New Zone (2)', etc."""
        import yaml

        existing_names = set()

        # Scan existing zone files for their names
        if os.path.exists(zones_dir):
            for filename in os.listdir(zones_dir):
                if filename.endswith('.zone.yaml'):
                    zone_path = os.path.join(zones_dir, filename)
                    try:
                        with open(zone_path, 'r') as f:
                            data = yaml.safe_load(f) or {}
                        name = data.get('name', '')
                        if name:
                            existing_names.add(name)
                    except:
                        pass

        # Find first available name
        if base_name not in existing_names:
            return base_name

        counter = 2
        while f"{base_name} ({counter})" in existing_names:
            counter += 1

        return f"{base_name} ({counter})"

    def create_empty_room(self):
        """LEGACY: Create an empty room - redirects to create_empty_zone."""
        # Legacy rooms are now zones in project mode
        self.create_empty_zone()

    def create_custom_prim(self):
        """Create a custom prim with specific type."""
        name, ok = QInputDialog.getText(
            self,
            "Create Custom Prim",
            "Prim name and type (e.g., 'MyProp:prop'):",
            text="CustomPrim"
        )
        if ok and name:
            print(f"Creating custom prim: {name}")
            # TODO: Send to noodleMUSH API
            self.refresh_scene()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
