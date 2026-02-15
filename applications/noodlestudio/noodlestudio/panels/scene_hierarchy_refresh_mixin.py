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
#   Scene Hierarchy Refresh Mixin - Tree building and refresh logic
#
#   Contains: - refresh_scene: Main refresh entry point - _re...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.scene_hierarchy_refresh_mixin
# PURPOSE:  Scene Hierarchy Refresh Mixin
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SceneHierarchyRefreshMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os

from PyQt6.QtWidgets import QTreeWidgetItem, QPushButton
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from ..core.scene_node import SceneNode, SceneNodeType


class SceneHierarchyRefreshMixin:
    """Mixin providing tree refresh logic for SceneHierarchy."""

    def refresh_scene(self):
        """Refresh scene hierarchy from project structure."""
        import yaml

        # Skip refresh if suppressed (during drag-drop operations)
        if self._suppress_refresh:
            return

        # Populate stage selector on first refresh
        if self.stage_selector.count() == 0:
            self.populate_stage_selector()

        try:
            # Save expanded state before clearing
            self.save_expanded_state()

            # CRITICAL: Block signals BEFORE clearing to prevent empty selection event
            self.tree.blockSignals(True)
            self.tree.clear()

            # Hide status label - will be shown if needed by message methods
            self.status_label.hide()

            # Clear item/node mappings
            self._item_id_to_node_id.clear()
            self._node_id_to_item.clear()

            # Check if project is open
            is_open = self.project_manager and self.project_manager.is_project_open()
            print(f"[SceneHierarchy] refresh_scene: project_manager={self.project_manager is not None}, is_open={is_open}, server={self._server_running}, current_stage={self.current_stage}")

            if not is_open:
                self._show_no_project_message()
            else:
                self._refresh_from_project()

            # Restore expanded state after rebuilding tree (signals still blocked)
            self.restore_expanded_state()

            # CRITICAL: Re-enable signals AFTER everything is restored
            self.tree.blockSignals(False)

        except Exception as e:
            print(f"Error refreshing scene: {e}")
            import traceback
            traceback.print_exc()
            self.tree.blockSignals(False)

    def _show_no_project_message(self):
        """Show message when no project is open."""
        self.status_label.setText("No project open\nFile > Open Project...")
        self.status_label.show()

    def _refresh_from_project(self):
        """Refresh scene from project structure (Stages/xxx/...)."""
        import yaml

        if not self.current_stage:
            stages = self.project_manager.list_stages()
            if stages:
                self.current_stage = stages[0]
            else:
                # No stages - show empty
                empty_item = QTreeWidgetItem(["No stages in project", ""])
                empty_item.setForeground(0, Qt.GlobalColor.gray)
                self.tree.addTopLevelItem(empty_item)
                return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            return

        # Try to load saved hierarchy first (preserves user organization)
        hierarchy_path = os.path.join(stage_path, "hierarchy.yaml")
        if os.path.exists(hierarchy_path):
            print(f"[SceneHierarchy] Loading saved hierarchy: {hierarchy_path}")
            try:
                if self._load_hierarchy(stage_path):
                    # Update names from disk files (in case inspector changed them)
                    self._sync_names_from_disk(stage_path)
                    # Detect any new files not in hierarchy
                    self._add_new_files_to_hierarchy(stage_path)
                    # Build tree from scene graph
                    self._build_tree_from_graph()
                    print(f"[SceneHierarchy] Tree built from hierarchy, {self.tree.topLevelItemCount()} top-level items")
                    return
            except Exception as e:
                print(f"[SceneHierarchy] Error loading hierarchy: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to build from files

        # No hierarchy file or load failed - build fresh from files
        print(f"[SceneHierarchy] Building tree from files: {stage_path}")
        try:
            self._build_tree_from_files(stage_path)
            print(f"[SceneHierarchy] Tree built from files, {self.tree.topLevelItemCount()} top-level items")
            # Save the initial hierarchy
            self._save_hierarchy()
        except Exception as e:
            print(f"[SceneHierarchy] ERROR building tree: {e}")
            import traceback
            traceback.print_exc()
            return

    def _extract_user_hierarchy(self) -> dict:
        """Extract user modifications to hierarchy (folders + reparented items)."""
        # Unity-style: Items at root level (parent_id=None) are default
        # User can create folders to organize, and move items into them

        user_folders = []
        reparented_items = []  # Items moved into user-created folders

        for node_id, node in self.scene_graph.nodes.items():
            if node.node_type == SceneNodeType.FOLDER and not node.is_virtual:
                # User-created folder - save it
                parent_node = self.scene_graph.get_node(node.parent_id) if node.parent_id else None
                user_folders.append({
                    'name': node.name,
                    'parent_name': parent_node.name if parent_node else None,
                    'is_expanded': node.is_expanded,
                })
            elif node.node_type in (SceneNodeType.ZONE, SceneNodeType.NOODLING, SceneNodeType.PROP):
                # Check if this item was moved from root to a user folder
                if node.parent_id:
                    parent_node = self.scene_graph.get_node(node.parent_id)
                    if parent_node:
                        # Item is not at root - was moved to a folder
                        reparented_items.append({
                            'name': node.name,
                            'node_type': node.node_type.value,
                            'parent_name': parent_node.name,
                        })

        return {'folders': user_folders, 'reparented': reparented_items}

    def _restore_user_folders(self, user_folders: list):
        """Restore user-created folders after tree rebuild."""
        for folder_info in user_folders:
            # Find parent by name
            parent_id = None
            if folder_info['parent_name']:
                for node_id, node in self.scene_graph.nodes.items():
                    if node.name == folder_info['parent_name']:
                        parent_id = node_id
                        break

            # Create folder in scene graph
            folder_node = self.scene_graph.create_folder(
                folder_info['name'], parent_id)
            folder_node.is_expanded = folder_info.get('is_expanded', True)

            # Create tree item
            folder_item = QTreeWidgetItem([folder_info['name'], ""])
            folder_item.setForeground(0, Qt.GlobalColor.gray)
            folder_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'folder',
                'node_id': folder_node.id,
                'name': folder_info['name']
            })

            # Find parent item and add
            if parent_id and parent_id in self._node_id_to_item:
                parent_item = self._node_id_to_item[parent_id]
                parent_item.addChild(folder_item)
            else:
                self.tree.addTopLevelItem(folder_item)

            # Track mappings
            self._item_id_to_node_id[id(folder_item)] = folder_node.id
            self._node_id_to_item[folder_node.id] = folder_item

            # Set expanded state
            folder_item.setExpanded(folder_info.get('is_expanded', True))

    def _restore_reparented_items(self, reparented_items: list):
        """Restore items that were moved from their default locations."""
        for item_info in reparented_items:
            item_name = item_info['name']
            new_parent_name = item_info['parent_name']

            # Find the item node by name
            item_node = None
            item_node_id = None
            for node_id, node in self.scene_graph.nodes.items():
                if node.name == item_name:
                    item_node = node
                    item_node_id = node_id
                    break

            if not item_node:
                print(f"[SceneHierarchy] Could not find item '{item_name}' to reparent")
                continue

            # Find new parent node by name
            new_parent_id = None
            for node_id, node in self.scene_graph.nodes.items():
                if node.name == new_parent_name:
                    new_parent_id = node_id
                    break

            if not new_parent_id:
                print(f"[SceneHierarchy] Could not find parent '{new_parent_name}' for item '{item_name}'")
                continue

            # Reparent in scene graph
            self.scene_graph.reparent(item_node_id, new_parent_id)

            # Move tree widget item
            tree_item = self._node_id_to_item.get(item_node_id)
            new_parent_item = self._node_id_to_item.get(new_parent_id)

            if tree_item and new_parent_item:
                # Remove from current parent
                old_parent = tree_item.parent()
                if old_parent:
                    old_parent.removeChild(tree_item)
                else:
                    index = self.tree.indexOfTopLevelItem(tree_item)
                    if index >= 0:
                        self.tree.takeTopLevelItem(index)

                # Add to new parent
                new_parent_item.addChild(tree_item)
                # Don't force expand - let restore_expanded_state handle it

                print(f"[SceneHierarchy] Reparented '{item_name}' to '{new_parent_name}'")

    def _build_tree_from_graph(self):
        """Build tree widget from scene graph data."""
        import yaml

        # Build tree recursively from root nodes
        for node_id in self.scene_graph.root_ids:
            node = self.scene_graph.get_node(node_id)
            if node:
                self._add_node_to_tree(node, None)

    def _add_node_to_tree(self, node: SceneNode, parent_item: QTreeWidgetItem):
        """Recursively add a node and its children to the tree."""
        import yaml

        # Create tree item with name in column 0
        item = QTreeWidgetItem([node.name, ""])

        # Style based on node type
        if node.node_type == SceneNodeType.FOLDER:
            item.setForeground(0, Qt.GlobalColor.gray)
        elif node.node_type == SceneNodeType.BONE:
            item.setForeground(0, Qt.GlobalColor.cyan)
        elif node.node_type == SceneNodeType.UI:
            item.setForeground(0, Qt.GlobalColor.lightGray)  # Monochromatic palette

        # Store node data - load full entity data from disk for inspector
        entity_data = {
            'type': node.node_type.value,
            'id': node.id,
            'name': node.name,
            'node_id': node.id,  # For scene graph operations
        }
        if node.asset_path:
            entity_data['asset_path'] = node.asset_path
            entity_data['path'] = node.asset_path

            # Load full data from disk for inspector
            self._load_entity_data_from_disk(node, entity_data)

        if node.bone_name:
            entity_data['bone_name'] = node.bone_name

        item.setData(0, Qt.ItemDataRole.UserRole, entity_data)

        # Track mappings
        self._item_id_to_node_id[id(item)] = node.id
        self._node_id_to_item[node.id] = item

        # Add to tree
        if parent_item:
            parent_item.addChild(item)
        else:
            self.tree.addTopLevelItem(item)

        # Add pause button for noodlings (must be after item is in tree)
        if node.node_type == SceneNodeType.NOODLING:
            self._add_pause_button(item, entity_data)

        # Set expanded state
        item.setExpanded(node.is_expanded)

        # For UI nodes, add component children from the loaded component tree
        if node.node_type == SceneNodeType.UI and 'component' in entity_data:
            root_component = entity_data['component']
            # Don't create graph nodes - we're rebuilding from existing graph
            self._add_ui_component_to_tree(root_component, item, node.id, node.asset_path,
                                          create_graph_nodes=False)
            item.setExpanded(True)

        # Add children recursively (skip UI_COMPONENT - handled by _add_ui_component_to_tree)
        for child_id in node.children_ids:
            child_node = self.scene_graph.get_node(child_id)
            if child_node and child_node.node_type != SceneNodeType.UI_COMPONENT:
                self._add_node_to_tree(child_node, item)

    def _load_entity_data_from_disk(self, node: SceneNode, entity_data: dict):
        """Load full entity data from disk files for inspector."""
        import yaml

        if node.node_type == SceneNodeType.PROP:
            prop_yaml = os.path.join(node.asset_path, 'prop.yaml')
            if os.path.exists(prop_yaml):
                try:
                    with open(prop_yaml, 'r') as f:
                        prop_data = yaml.safe_load(f) or {}
                    entity_data['data'] = prop_data
                except Exception as e:
                    print(f"[SceneHierarchy] Error loading prop data: {e}")

        elif node.node_type == SceneNodeType.NOODLING:
            inst_yaml = os.path.join(node.asset_path, 'instance.yaml')
            if os.path.exists(inst_yaml):
                try:
                    with open(inst_yaml, 'r') as f:
                        inst_data = yaml.safe_load(f) or {}
                    entity_data['data'] = inst_data
                    entity_data['noodling_ref'] = inst_data.get('noodling', '')
                    entity_data['zone'] = inst_data.get('overrides', {}).get('zone', 'default')
                    # Use 'noodling' type for inspector compatibility
                    entity_data['type'] = 'noodling'
                    # Build agent_id
                    entity_data['id'] = f"agent_{os.path.basename(node.asset_path)}"
                except Exception as e:
                    print(f"[SceneHierarchy] Error loading noodling data: {e}")

        elif node.node_type == SceneNodeType.ZONE:
            # Zone asset_path IS the yaml file
            if os.path.exists(node.asset_path):
                try:
                    with open(node.asset_path, 'r') as f:
                        zone_data = yaml.safe_load(f) or {}
                    entity_data['data'] = zone_data
                    entity_data['id'] = zone_data.get('id', node.id)
                    entity_data['radius'] = zone_data.get('radius', 10)
                    entity_data['falloff'] = zone_data.get('falloff', 5)
                except Exception as e:
                    print(f"[SceneHierarchy] Error loading zone data: {e}")

        elif node.node_type == SceneNodeType.UI:
            # UI canvas - load component tree
            if os.path.exists(node.asset_path):
                try:
                    from ..runtime.ui.loader import UILoader
                    loader = UILoader()
                    root_component = loader.load_file(node.asset_path)
                    entity_data['component'] = root_component
                except Exception as e:
                    print(f"[SceneHierarchy] Error loading UI canvas data: {e}")

    def _build_tree_from_files(self, stage_path: str):
        """Build tree from file structure (legacy mode)."""
        import yaml

        # Clear and populate scene graph
        self._suppress_refresh = True
        self.scene_graph.clear()

        # Load stage.yaml for zone graph info
        stage_data = {}
        stage_yaml = os.path.join(stage_path, "stage.yaml")
        if os.path.exists(stage_yaml):
            try:
                with open(stage_yaml, 'r') as f:
                    stage_data = yaml.safe_load(f) or {}
            except:
                pass

        # Unity-style: Load all items as top-level (no Stage: wrapper)
        # Stage is selected via dropdown, not shown in tree
        # User can create their own folders to organize

        # Load zones from Zones/*.zone.yaml
        zones_dir = os.path.join(stage_path, "Zones")
        if os.path.exists(zones_dir):
            for filename in os.listdir(zones_dir):
                if filename.endswith(".zone.yaml"):
                    zone_path = os.path.join(zones_dir, filename)
                    try:
                        with open(zone_path, 'r') as f:
                            zone_data = yaml.safe_load(f) or {}
                        zone_id = zone_data.get('id', filename.replace('.zone.yaml', ''))
                        zone_name = zone_data.get('name', zone_id)
                        radius = zone_data.get('radius', 10)
                        falloff = zone_data.get('falloff', 5)

                        display_text = f"{zone_name} (r={radius}, f={falloff})"
                        zone_item = QTreeWidgetItem([display_text, ""])  # Two columns

                        # Add to scene graph as root (parent_id=None)
                        zone_node = self.scene_graph.create_node(
                            zone_name, SceneNodeType.ZONE, None, zone_path)

                        zone_item.setData(0, Qt.ItemDataRole.UserRole, {
                            'type': 'zone',
                            'id': zone_id,
                            'name': zone_name,
                            'path': zone_path,
                            'data': zone_data,
                            'node_id': zone_node.id
                        })
                        self.tree.addTopLevelItem(zone_item)

                        self._item_id_to_node_id[id(zone_item)] = zone_node.id
                        self._node_id_to_item[zone_node.id] = zone_item

                    except Exception as e:
                        print(f"Error loading zone {filename}: {e}")

        # Load instances from Instances/*/instance.yaml
        instances_dir = os.path.join(stage_path, "Instances")
        if os.path.exists(instances_dir):
            for inst_name in os.listdir(instances_dir):
                inst_path = os.path.join(instances_dir, inst_name)
                if not os.path.isdir(inst_path):
                    continue

                inst_yaml = os.path.join(inst_path, "instance.yaml")
                if not os.path.exists(inst_yaml):
                    continue

                try:
                    with open(inst_yaml, 'r') as f:
                        inst_data = yaml.safe_load(f) or {}

                    overrides = inst_data.get('overrides', {})
                    display_name = overrides.get('name', inst_name)
                    noodling_ref = inst_data.get('noodling', '')
                    zone = overrides.get('zone', 'default')
                    agent_id = f"agent_{inst_name}"

                    # Add to scene graph as root (parent_id=None)
                    inst_node = self.scene_graph.create_node(
                        display_name, SceneNodeType.NOODLING, None, inst_path)

                    inst_item = QTreeWidgetItem([display_name, ""])  # Two columns
                    entity_data = {
                        'type': 'noodling',
                        'id': agent_id,
                        'name': display_name,
                        'path': inst_path,
                        'noodling_ref': noodling_ref,
                        'zone': zone,
                        'data': inst_data,
                        'node_id': inst_node.id
                    }
                    inst_item.setData(0, Qt.ItemDataRole.UserRole, entity_data)
                    self.tree.addTopLevelItem(inst_item)

                    # Add pause button after item is in tree
                    self._add_pause_button(inst_item, entity_data)

                    self._item_id_to_node_id[id(inst_item)] = inst_node.id
                    self._node_id_to_item[inst_node.id] = inst_item

                except Exception as e:
                    print(f"Error loading instance {inst_name}: {e}")

        # Load props from Props/*/prop.yaml
        props_dir = os.path.join(stage_path, "Props")
        if os.path.exists(props_dir):
            for prop_name in os.listdir(props_dir):
                prop_path = os.path.join(props_dir, prop_name)
                if not os.path.isdir(prop_path):
                    continue

                prop_yaml = os.path.join(prop_path, "prop.yaml")
                if not os.path.exists(prop_yaml):
                    continue

                try:
                    with open(prop_yaml, 'r') as f:
                        prop_data = yaml.safe_load(f) or {}

                    display_name = prop_data.get('name', prop_name)
                    prim_ref = prop_data.get('prim', '')
                    zone = prop_data.get('zone', 'default')
                    is_locked = prop_data.get('locked', False)

                    lock_icon = "[L]" if is_locked else ""

                    # Add to scene graph as root (parent_id=None)
                    prop_node = self.scene_graph.create_node(
                        display_name, SceneNodeType.PROP, None, prop_path)

                    prop_item = QTreeWidgetItem([display_name, lock_icon])  # Two columns
                    prop_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'type': 'prop',
                        'id': f"prop_{prop_name}",
                        'name': display_name,
                        'path': prop_path,
                        'prim_ref': prim_ref,
                        'zone': zone,
                        'locked': is_locked,
                        'data': prop_data,
                        'node_id': prop_node.id
                    })
                    self.tree.addTopLevelItem(prop_item)

                    self._item_id_to_node_id[id(prop_item)] = prop_node.id
                    self._node_id_to_item[prop_node.id] = prop_item

                except Exception as e:
                    print(f"Error loading prop {prop_name}: {e}")

        # Load UI canvases from *.ui.yaml files at stage root
        self._load_ui_canvases(stage_path)

        # Zone connections shown as top-level items
        zone_graph = stage_data.get('zone_graph', {})
        for from_zone, connections in zone_graph.items():
            for to_zone in connections:
                exit_item = QTreeWidgetItem([f"{from_zone} -> {to_zone}", ""])  # Two columns
                exit_item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'zone_connection',
                    'from': from_zone,
                    'to': to_zone
                })
                self.tree.addTopLevelItem(exit_item)

        # Re-enable refresh
        self._suppress_refresh = False

    def _refresh_from_legacy(self):
        """Refresh scene from legacy format (cmush/world/...)."""
        import json
        import requests

        # Try to get agents from API first
        agents_data = []
        try:
            agents_resp = requests.get(f"{self.api_base}/agents", timeout=2)
            agents_data = agents_resp.json().get('agents', [])
        except:
            pass

        # Get stage data from rooms.json
        stage_name = "Unknown Stage"
        stage_data = None
        try:
            rooms_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world/rooms.json"
            )
            with open(rooms_path, 'r') as f:
                rooms_data = json.load(f)
                if self.current_room in rooms_data:
                    stage_data = rooms_data[self.current_room]
                    stage_name = stage_data.get('name', self.current_room)
        except:
            rooms_data = {}

        room_item = QTreeWidgetItem([f"Stage: {stage_name}", ""])
        room_item.setFont(0, QFont("Arial", 12, QFont.Weight.Bold))
        room_item.setForeground(0, Qt.GlobalColor.white)
        room_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'stage',
            'id': self.current_room,
            'data': stage_data or {'name': stage_name}
        })
        self.tree.addTopLevelItem(room_item)
        room_item.setExpanded(True)

        # Connected Users folder
        users_folder = QTreeWidgetItem(["Connected Users", ""])
        users_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(users_folder)

        user_item = QTreeWidgetItem(["caity [Noodler, 9yo, she/her]", ""])
        user_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'user', 'id': 'user_caity'})
        users_folder.addChild(user_item)

        # Noodlings folder
        noodlings_folder = QTreeWidgetItem(["Noodlings", ""])
        noodlings_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(noodlings_folder)

        # Add each Noodling (filter by location)
        for agent in agents_data:
            agent_location = agent.get('location') or agent.get('current_room')
            if agent_location != self.current_room:
                continue

            name = agent.get('name', agent.get('id'))
            agent_id = agent.get('id')
            is_locked = agent.get('locked', False)

            entity_data = {
                'type': 'noodling',
                'id': agent_id,
                'name': name,
                'locked': is_locked,
                'data': agent
            }
            noodling_item = QTreeWidgetItem([name, ""])
            noodling_item.setData(0, Qt.ItemDataRole.UserRole, entity_data)
            noodlings_folder.addChild(noodling_item)

            # Add pause button after item is in tree
            self._add_pause_button(noodling_item, entity_data)

        # Prims folder
        prims_folder = QTreeWidgetItem(["Prims", ""])
        prims_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(prims_folder)

        try:
            objects_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world/objects.json"
            )
            with open(objects_path, 'r') as f:
                objects_data = json.load(f)

            for obj_id, obj_data in objects_data.items():
                if obj_data.get('location') == self.current_room:
                    prim_name = obj_data.get('name', obj_id)
                    is_locked = obj_data.get('locked', False)
                    is_disabled = obj_data.get('disabled', False)

                    status_icon = "[L]" if is_locked else ""

                    prim_item = QTreeWidgetItem([prim_name, status_icon])
                    prim_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'type': 'prim',
                        'id': obj_id,
                        'locked': is_locked,
                        'disabled': is_disabled,
                        'data': obj_data
                    })
                    prims_folder.addChild(prim_item)

        except Exception as e:
            print(f"Error loading prims: {e}")

        # Exits folder
        exits_folder = QTreeWidgetItem(["Exits", ""])
        exits_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(exits_folder)

        try:
            if self.current_room in rooms_data:
                room_data = rooms_data[self.current_room]
                exits = room_data.get('exits', {})

                for direction, dest_room_id in exits.items():
                    dest_name = rooms_data.get(dest_room_id, {}).get('name', dest_room_id)
                    exit_item = QTreeWidgetItem([f"{direction} -> {dest_name}", ""])
                    exit_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'type': 'exit',
                        'direction': direction,
                        'destination': dest_room_id
                    })
                    exits_folder.addChild(exit_item)

        except Exception as e:
            print(f"Error loading exits: {e}")
            # Make sure signals are unblocked even on error
            self.tree.blockSignals(False)

    def _add_pause_button(self, item: QTreeWidgetItem, entity_data: dict):
        """Add a pause/play toggle button to column 1 for noodlings."""
        from PyQt6.QtWidgets import QWidget, QHBoxLayout

        agent_id = entity_data.get('id', '')

        # Create container with right alignment
        container = QWidget()
        container.setStyleSheet("background: transparent;")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addStretch()  # Push button to right

        # Create small pause button
        pause_btn = QPushButton()
        pause_btn.setFixedSize(24, 20)
        pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        pause_btn.setProperty('agent_id', agent_id)
        pause_btn.setProperty('paused', False)

        # Style the button
        pause_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                color: #888;
                font-size: 11px;
                font-weight: bold;
            }
            QPushButton:hover {
                color: #FFF;
                background-color: #333;
                border-radius: 3px;
            }
        """)
        pause_btn.setText("||")  # Pause icon

        # Connect click handler
        pause_btn.clicked.connect(lambda: self._toggle_pause(pause_btn))

        layout.addWidget(pause_btn)

        # Add container to column 1
        self.tree.setItemWidget(item, 1, container)

    def _toggle_pause(self, btn: QPushButton):
        """Toggle pause state for a noodling."""
        agent_id = btn.property('agent_id')
        is_paused = btn.property('paused')

        # Toggle state
        new_paused = not is_paused
        btn.setProperty('paused', new_paused)

        # Update button appearance
        if new_paused:
            btn.setText(">")  # Play icon (currently paused, click to play)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #D9A641;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #FFF;
                    background-color: #333;
                    border-radius: 3px;
                }
            """)
        else:
            btn.setText("||")  # Pause icon (currently running, click to pause)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: transparent;
                    border: none;
                    color: #888;
                    font-size: 11px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    color: #FFF;
                    background-color: #333;
                    border-radius: 3px;
                }
            """)

        # Emit signal for facet executor to handle
        if hasattr(self, 'pauseToggled'):
            self.pauseToggled.emit(agent_id, new_paused)
        print(f"[SceneHierarchy] {'Paused' if new_paused else 'Resumed'} cognition for {agent_id}")

    def _load_ui_canvases(self, stage_path: str):
        """Load UI canvas files (ui.yaml or *.ui.yaml) into hierarchy."""
        import yaml

        # Look for ui.yaml at stage root, also any *.ui.yaml files
        ui_files = []

        # Check for main ui.yaml
        main_ui = os.path.join(stage_path, "ui.yaml")
        if os.path.exists(main_ui):
            ui_files.append(("UI", main_ui))

        # Check for additional *.ui.yaml files
        for filename in os.listdir(stage_path):
            if filename.endswith('.ui.yaml') and filename != 'ui.yaml':
                name = filename.replace('.ui.yaml', '')
                ui_files.append((f"UI: {name}", os.path.join(stage_path, filename)))

        # Load each UI canvas
        for display_name, ui_path in ui_files:
            try:
                # Load the UI component tree
                from ..runtime.ui.loader import UILoader
                loader = UILoader()
                root_component = loader.load_file(ui_path)

                # Create UI root node
                ui_node = self.scene_graph.create_node(
                    display_name, SceneNodeType.UI, None, ui_path)

                ui_item = QTreeWidgetItem([display_name, ""])
                ui_item.setForeground(0, Qt.GlobalColor.lightGray)  # Monochromatic palette
                ui_item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'ui',
                    'name': display_name,
                    'path': ui_path,
                    'node_id': ui_node.id,
                    'component': root_component
                })
                self.tree.addTopLevelItem(ui_item)

                self._item_id_to_node_id[id(ui_item)] = ui_node.id
                self._node_id_to_item[ui_node.id] = ui_item

                # Add UI components as children recursively
                self._add_ui_component_to_tree(root_component, ui_item, ui_node.id, ui_path)

                ui_item.setExpanded(True)

            except Exception as e:
                print(f"[SceneHierarchy] Error loading UI canvas {ui_path}: {e}")
                import traceback
                traceback.print_exc()

    def _add_ui_component_to_tree(self, component, parent_item: QTreeWidgetItem,
                                   parent_node_id: str, ui_path: str,
                                   create_graph_nodes: bool = True):
        """Recursively add UI component and its children to the tree.

        Args:
            component: The UIComponent to add
            parent_item: Parent tree widget item
            parent_node_id: Parent node ID in scene graph
            ui_path: Path to the ui.yaml file
            create_graph_nodes: If True, create scene graph nodes. False when
                               rebuilding from existing graph to avoid duplicates.
        """
        # Create display name with type
        display_name = f"{component.name} ({component.component_type})"

        # Create scene node for this component (only if requested)
        if create_graph_nodes:
            comp_node = self.scene_graph.create_node(
                component.name, SceneNodeType.UI_COMPONENT, parent_node_id, ui_path)
            comp_node.metadata['component_type'] = component.component_type
            node_id = comp_node.id
        else:
            # Generate a temporary ID for tree item tracking (not in scene graph)
            node_id = f"ui_comp_{component.name}_{id(component)}"

        # Create tree item
        comp_item = QTreeWidgetItem([display_name, ""])
        comp_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'ui_component',
            'name': component.name,
            'component_type': component.component_type,
            'path': ui_path,
            'node_id': node_id,
            'component': component
        })
        # Add tooltip for long names
        comp_item.setToolTip(0, display_name)
        parent_item.addChild(comp_item)

        if create_graph_nodes:
            self._item_id_to_node_id[id(comp_item)] = node_id
            self._node_id_to_item[node_id] = comp_item

        # Recursively add children
        for child in component.children:
            self._add_ui_component_to_tree(child, comp_item, node_id, ui_path, create_graph_nodes)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
