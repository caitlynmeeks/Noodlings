"""
Scene Hierarchy Derez Mixin - Delete and derez operations

Contains:
- delete_prim_data: Legacy prim deletion
- delete_prop: Delete project-mode prop
- duplicate_prop: Duplicate prop
- delete_instance: Delete noodling instance
- duplicate_instance: Duplicate noodling instance
- delete_zone: Delete zone
- delete_selected_items: Batch deletion
- _derez_entity: Core derez logic
- _derez_entity_direct: Direct derez without undo

Author: Noodlings Project
Date: December 2025
"""

import os
import json
import shutil

from PyQt6.QtWidgets import QMessageBox, QCheckBox
from PyQt6.QtCore import Qt
from PyQt6 import sip

from ..core.undo_manager import UndoManager
from ..core.commands.scene_commands import (
    DeleteNoodlingCommand, DeletePropCommand, DeleteZoneCommand, BatchDeleteCommand
)


class SceneHierarchyDerezMixin:
    """Mixin providing delete/derez operations for SceneHierarchy."""

    def delete_prim_data(self, entity_data):
        """De-rez a prim or Noodling (delete from scene)."""
        prim_id = entity_data.get('id')
        prim_type = entity_data.get('type')

        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("De-Rez")
        msgBox.setText(f"De-rez {prim_type} '{prim_id}'?\n\nThis will remove it from the scene.")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msgBox.setIcon(QMessageBox.Icon.NoIcon)  # No icon!
        reply = msgBox.exec()

        if reply == QMessageBox.StandardButton.Yes:
            # De-rez via direct file manipulation (simple and fast)
            try:
                print(f"Derezzing {prim_type}: {prim_id}")

                # Derez by directly modifying world data files
                base_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/world"
                )
                base_path = os.path.abspath(base_path)

                if prim_type == 'noodling' or prim_type == 'agent':
                    # Remove from agents.json
                    agents_path = os.path.join(base_path, "agents.json")
                    with open(agents_path, 'r') as f:
                        agents = json.load(f)

                    if prim_id in agents:
                        del agents[prim_id]
                        with open(agents_path, 'w') as f:
                            json.dump(agents, f, indent=2)
                        print(f"Derezzed {prim_id}")

                elif prim_type == 'prim':
                    # Remove from objects.json
                    objects_path = os.path.join(base_path, "objects.json")
                    with open(objects_path, 'r') as f:
                        objects = json.load(f)

                    if prim_id in objects:
                        del objects[prim_id]
                        with open(objects_path, 'w') as f:
                            json.dump(objects, f, indent=2)
                        print(f"Derezzed {prim_id}")

                # Remove from tree immediately
                current_item = self.tree.currentItem()
                if current_item:
                    parent = current_item.parent()
                    if parent:
                        parent.removeChild(current_item)
                    else:
                        index = self.tree.indexOfTopLevelItem(current_item)
                        if index >= 0:
                            self.tree.takeTopLevelItem(index)

            except Exception as e:
                QMessageBox.warning(self, "Derez Failed", f"Error: {e}")

    # =========================================================================
    # Delete/Duplicate for Project-mode entities
    # =========================================================================

    def delete_prop(self, entity_data: dict):
        """Delete a prop from the project (with undo support)."""
        import yaml

        prop_path = entity_data.get('path', '')
        prop_name = entity_data.get('name', 'this prop')

        if not prop_path or not os.path.exists(prop_path):
            print(f"Prop path not found: {prop_path}")
            return

        reply = QMessageBox.question(
            self,
            "De-Rez Prop",
            f"Delete '{prop_name}'?\n\nCmd+Z to undo.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Load full prop data for potential undo
            prop_yaml = os.path.join(prop_path, "prop.yaml")
            prop_data = entity_data.get('data', {})
            if os.path.exists(prop_yaml):
                try:
                    import yaml
                    with open(prop_yaml, 'r') as f:
                        prop_data = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"Warning: Could not read prop.yaml: {e}")

            # Push undo command (which deletes the files)
            cmd = DeletePropCommand(
                hierarchy=self,
                prop_path=prop_path,
                prop_data=prop_data,
                display_name=prop_name
            )
            UndoManager().push(cmd)
            print(f"Deleted prop: {prop_name}")

    def duplicate_prop(self, entity_data: dict):
        """Duplicate a prop in the project."""
        import yaml
        import uuid

        prop_path = entity_data.get('path', '')
        if not prop_path or not os.path.exists(prop_path):
            print(f"Prop path not found: {prop_path}")
            return

        # Load existing prop data
        prop_yaml = os.path.join(prop_path, "prop.yaml")
        if not os.path.exists(prop_yaml):
            return

        with open(prop_yaml, 'r') as f:
            prop_data = yaml.safe_load(f) or {}

        # Get props directory
        props_dir = os.path.dirname(prop_path)

        # Generate new UUID and name
        new_id = str(uuid.uuid4())
        old_name = prop_data.get('name', 'Prop')
        new_name = self._generate_unique_name(props_dir, old_name, "prop.yaml")

        # Create new folder
        new_path = os.path.join(props_dir, new_id)
        os.makedirs(new_path, exist_ok=True)

        # Copy and update data
        new_data = prop_data.copy()
        new_data['id'] = new_id
        new_data['name'] = new_name

        with open(os.path.join(new_path, "prop.yaml"), 'w') as f:
            yaml.dump(new_data, f, default_flow_style=False)

        print(f"Duplicated prop: {new_name} ({new_id[:8]}...)")
        self.refresh_scene()

    def delete_instance(self, entity_data: dict):
        """Delete a noodling instance from the project (with undo support)."""
        import yaml

        inst_path = entity_data.get('path', '')
        inst_name = entity_data.get('name', 'this instance')

        if not inst_path or not os.path.exists(inst_path):
            print(f"Instance path not found: {inst_path}")
            return

        reply = QMessageBox.question(
            self,
            "De-Rez Instance",
            f"Delete '{inst_name}'?\n\nCmd+Z to undo.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Load full instance data for potential undo
            inst_yaml = os.path.join(inst_path, "instance.yaml")
            inst_data = entity_data.get('data', {})
            if os.path.exists(inst_yaml):
                try:
                    with open(inst_yaml, 'r') as f:
                        inst_data = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"Warning: Could not read instance.yaml: {e}")

            # Push undo command (which deletes the files)
            cmd = DeleteNoodlingCommand(
                hierarchy=self,
                instance_path=inst_path,
                instance_data=inst_data,
                display_name=inst_name
            )
            UndoManager().push(cmd)
            print(f"Deleted instance: {inst_name}")

    def duplicate_instance(self, entity_data: dict):
        """Duplicate a noodling instance in the project."""
        import yaml
        import uuid

        inst_path = entity_data.get('path', '')
        if not inst_path or not os.path.exists(inst_path):
            print(f"Instance path not found: {inst_path}")
            return

        # Load existing instance data
        inst_yaml = os.path.join(inst_path, "instance.yaml")
        if not os.path.exists(inst_yaml):
            return

        with open(inst_yaml, 'r') as f:
            inst_data = yaml.safe_load(f) or {}

        # Get instances directory
        instances_dir = os.path.dirname(inst_path)

        # Generate new UUID and name
        new_id = str(uuid.uuid4())
        old_name = inst_data.get('overrides', {}).get('name', 'Noodling')
        new_name = self._generate_unique_name(instances_dir, old_name, "instance.yaml")

        # Create new folder
        new_path = os.path.join(instances_dir, new_id)
        os.makedirs(new_path, exist_ok=True)

        # Copy and update data
        new_data = inst_data.copy()
        new_data['id'] = new_id
        if 'overrides' not in new_data:
            new_data['overrides'] = {}
        new_data['overrides']['name'] = new_name

        with open(os.path.join(new_path, "instance.yaml"), 'w') as f:
            yaml.dump(new_data, f, default_flow_style=False)

        print(f"Duplicated instance: {new_name} ({new_id[:8]}...)")
        self.refresh_scene()

    def delete_zone(self, entity_data: dict):
        """Delete a zone from the project (with undo support)."""
        import yaml

        zone_path = entity_data.get('path', '')
        zone_name = entity_data.get('name') or entity_data.get('data', {}).get('name', 'this zone')

        if not zone_path or not os.path.exists(zone_path):
            print(f"Zone path not found: {zone_path}")
            return

        reply = QMessageBox.question(
            self,
            "Delete Zone",
            f"Delete zone '{zone_name}'?\n\nCmd+Z to undo.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Load full zone data for potential undo
            zone_data = entity_data.get('data', {})
            if os.path.exists(zone_path):
                try:
                    with open(zone_path, 'r') as f:
                        zone_data = yaml.safe_load(f) or {}
                except Exception as e:
                    print(f"Warning: Could not read zone file: {e}")

            # Push undo command (which deletes the file)
            cmd = DeleteZoneCommand(
                hierarchy=self,
                zone_path=zone_path,
                zone_data=zone_data,
                display_name=zone_name
            )
            UndoManager().push(cmd)
            print(f"Deleted zone: {zone_name}")

    # =========================================================================
    # Batch Deletion
    # =========================================================================

    def delete_selected_items(self):
        """De-rez all selected items (supports multi-selection with undo)."""
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [delete_selected_items] Called")
        selected_items = self.tree.selectedItems()
        if not selected_items:
            print(f"[{timestamp}] [delete_selected_items] No items selected, returning")
            return

        # Collect entities to derez (copy data to avoid modification issues)
        entities_to_derez = []
        for item in selected_items:
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if entity_data and isinstance(entity_data, dict):
                entity_path = entity_data.get('path', '')
                entities_to_derez.append((entity_data.copy(), entity_path))

        if not entities_to_derez:
            return

        # Show confirmation if enabled
        if self.derez_confirm:
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Derez")

            if len(entities_to_derez) == 1:
                prim_type = entities_to_derez[0][0].get('type', 'item')
                prim_name = entities_to_derez[0][0].get('name', entities_to_derez[0][0].get('id', 'unknown'))
                msgBox.setText(f"Derez {prim_type} '{prim_name}'?\n\nThis will remove it from the scene.")
            else:
                msgBox.setText(f"Derez {len(entities_to_derez)} items?\n\nThis will remove them from the scene.")

            msgBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            msgBox.setIcon(QMessageBox.Icon.NoIcon)

            # Add "Don't ask again" checkbox
            dont_ask = QCheckBox("Don't ask again")
            msgBox.setCheckBox(dont_ask)

            reply = msgBox.exec()

            # Save preference
            if dont_ask.isChecked():
                self.derez_confirm = False

            if reply != QMessageBox.StandardButton.Yes:
                return

        # Check if all entities have paths (project-mode, supports undo)
        all_have_paths = all(path for _, path in entities_to_derez)

        if all_have_paths:
            # Use BatchDeleteCommand for proper undo support
            cmd = BatchDeleteCommand(self, entities_to_derez)
            UndoManager().push(cmd)
            print(f"Derezzed {len(entities_to_derez)} items via BatchDeleteCommand")
        else:
            # Legacy entities without paths - direct deletion, no undo
            self.tree.blockSignals(True)
            try:
                for entity_data, entity_path in entities_to_derez:
                    try:
                        self._derez_entity_direct(entity_data, entity_path)
                    except Exception as e:
                        print(f"Error derezzing entity: {e}")
                        import traceback
                        traceback.print_exc()
            finally:
                self.tree.blockSignals(False)

            # Refresh after legacy deletions
            self.refresh_scene()
            self.entitySelected.emit("", {})

    def _derez_entity(self, entity_data, tree_item, use_undo: bool = True):
        """Derez a single entity (helper for batch operations).

        Args:
            entity_data: Entity data dict
            tree_item: QTreeWidgetItem to remove
            use_undo: If True, use undo commands (only for single-item deletes)

        Uses undo commands for project-mode entities when use_undo=True.
        Direct deletion for legacy entities or batch operations.
        """
        prim_id = entity_data.get('id', entity_data.get('name', 'unknown'))
        prim_type = entity_data.get('type', 'unknown')
        entity_path = entity_data.get('path', '')
        display_name = entity_data.get('name', prim_id)

        # Check if tree_item is still valid before proceeding
        if sip.isdeleted(tree_item):
            print(f"Tree item already deleted for {prim_type}: {prim_id}")
            return

        try:
            print(f"Derezzing {prim_type}: {prim_id} (undo={use_undo})")

            # Project-mode entities with paths use undo commands (single-item only)
            if use_undo and entity_path and prim_type == 'prop':
                # Use DeletePropCommand for undo support
                cmd = DeletePropCommand(
                    self,
                    entity_path,
                    entity_data.get('data', entity_data),
                    display_name=display_name
                )
                UndoManager().push(cmd)
                print(f"Derezzed prop via command: {display_name}")
                return  # Command handles tree/graph cleanup

            elif use_undo and entity_path and prim_type == 'zone':
                # Use DeleteZoneCommand for undo support
                cmd = DeleteZoneCommand(
                    self,
                    entity_path,
                    entity_data.get('data', entity_data),
                    display_name=display_name
                )
                UndoManager().push(cmd)
                print(f"Derezzed zone via command: {display_name}")
                return  # Command handles tree/graph cleanup

            elif use_undo and entity_path and prim_type in ('noodling', 'agent'):
                # Use DeleteNoodlingCommand for undo support
                cmd = DeleteNoodlingCommand(
                    self,
                    entity_path,
                    entity_data.get('data', entity_data),
                    display_name=display_name
                )
                UndoManager().push(cmd)
                print(f"Derezzed noodling via command: {display_name}")
                return  # Command handles tree/graph cleanup

            # =========================================================
            # Direct deletion (batch operations or legacy cmush entities)
            # =========================================================

            # For project-mode entities without undo, still delete from filesystem
            if entity_path:
                if prim_type == 'prop':
                    if os.path.isdir(entity_path):
                        shutil.rmtree(entity_path)
                    elif os.path.exists(entity_path):
                        os.remove(entity_path)
                    print(f"Derezzed prop (no undo): {display_name}")
                elif prim_type == 'zone':
                    if os.path.exists(entity_path):
                        if os.path.isdir(entity_path):
                            shutil.rmtree(entity_path)
                        else:
                            os.remove(entity_path)
                    print(f"Derezzed zone (no undo): {display_name}")
                elif prim_type in ('noodling', 'agent'):
                    if os.path.isdir(entity_path):
                        shutil.rmtree(entity_path)
                    print(f"Derezzed noodling (no undo): {display_name}")

            base_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world"
            )
            base_path = os.path.abspath(base_path)

            if prim_type in ('noodling', 'agent'):
                # Remove via API (removes from running server + agents.json)
                import requests
                try:
                    response = requests.delete(f"http://localhost:8081/api/agents/{prim_id}", timeout=2)
                    if response.status_code == 200:
                        print(f"Derezzed {prim_id} from server (no undo)")
                    else:
                        print(f"API error: {response.text}")
                except Exception as e:
                    print(f"Failed to derez via API: {e}")
                    # Fallback: remove from file only
                    agents_path = os.path.join(base_path, "agents.json")
                    if os.path.exists(agents_path):
                        with open(agents_path, 'r') as f:
                            agents = json.load(f)
                        if prim_id in agents:
                            del agents[prim_id]
                            with open(agents_path, 'w') as f:
                                json.dump(agents, f, indent=2)
                            print(f"Derezzed {prim_id} from file (server may still have it)")

            elif prim_type == 'prim':
                # Remove from objects.json (legacy prim)
                objects_path = os.path.join(base_path, "objects.json")
                if os.path.exists(objects_path):
                    with open(objects_path, 'r') as f:
                        objects = json.load(f)

                    if prim_id in objects:
                        del objects[prim_id]
                        with open(objects_path, 'w') as f:
                            json.dump(objects, f, indent=2)
                        print(f"Derezzed {prim_id} (no undo)")

            else:
                print(f"Unknown entity type '{prim_type}' - removing from tree only")

            # Clean up internal mappings before removing from tree
            item_id = id(tree_item)
            node_id = self._item_id_to_node_id.pop(item_id, None)
            if node_id:
                self._node_id_to_item.pop(node_id, None)
                # Also remove from scene graph
                try:
                    self.scene_graph.delete_node(node_id, recursive=True)
                except Exception as e:
                    print(f"Warning: Could not delete node {node_id} from scene graph: {e}")

            # Remove from tree (safe even if item was already removed)
            try:
                if not sip.isdeleted(tree_item):
                    parent = tree_item.parent()
                    if parent:
                        parent.removeChild(tree_item)
                    else:
                        index = self.tree.indexOfTopLevelItem(tree_item)
                        if index >= 0:
                            self.tree.takeTopLevelItem(index)
            except RuntimeError:
                # Item was already deleted
                pass

        except Exception as e:
            print(f"Error derezzing {prim_id}: {e}")
            import traceback
            traceback.print_exc()

    def _derez_entity_direct(self, entity_data: dict, entity_path: str):
        """Direct deletion for legacy entities (no undo support).

        Used for cmush API-based entities that don't have filesystem paths.
        """
        prim_id = entity_data.get('id', entity_data.get('name', 'unknown'))
        prim_type = entity_data.get('type', 'unknown')

        try:
            print(f"Derezzing legacy {prim_type}: {prim_id} (no undo)")

            base_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world"
            )
            base_path = os.path.abspath(base_path)

            if prim_type in ('noodling', 'agent'):
                # Remove via API
                import requests
                try:
                    response = requests.delete(f"http://localhost:8081/api/agents/{prim_id}", timeout=2)
                    if response.status_code == 200:
                        print(f"Derezzed {prim_id} from server")
                    else:
                        print(f"API error: {response.text}")
                except Exception as e:
                    print(f"Failed to derez via API: {e}")
                    # Fallback: remove from file
                    agents_path = os.path.join(base_path, "agents.json")
                    if os.path.exists(agents_path):
                        with open(agents_path, 'r') as f:
                            agents = json.load(f)
                        if prim_id in agents:
                            del agents[prim_id]
                            with open(agents_path, 'w') as f:
                                json.dump(agents, f, indent=2)

            elif prim_type == 'prim':
                # Remove from objects.json
                objects_path = os.path.join(base_path, "objects.json")
                if os.path.exists(objects_path):
                    with open(objects_path, 'r') as f:
                        objects = json.load(f)
                    if prim_id in objects:
                        del objects[prim_id]
                        with open(objects_path, 'w') as f:
                            json.dump(objects, f, indent=2)

        except Exception as e:
            print(f"Error derezzing legacy {prim_id}: {e}")
            import traceback
            traceback.print_exc()
