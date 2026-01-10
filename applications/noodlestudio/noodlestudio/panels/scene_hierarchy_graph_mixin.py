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
#   Scene Hierarchy Graph Mixin - SceneGraph signal handlers and hierarchy persistence
#
#   Contains: - SceneGraph signal handlers (_on_graph_changed...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.scene_hierarchy_graph_mixin
# PURPOSE:  Scene Hierarchy Graph Mixin
# LAYER:    Studio / Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   SceneHierarchyGraphMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import os

from PyQt6.QtWidgets import QTreeWidgetItem
from PyQt6.QtCore import Qt

from ..core.scene_node import SceneNodeType


class SceneHierarchyGraphMixin:
    """Mixin providing SceneGraph operations for SceneHierarchy."""

    def set_project_manager(self, project_manager):
        """Set project manager reference for loading from project structure."""
        self.project_manager = project_manager
        print(f"[SceneHierarchy] set_project_manager called, project_manager={project_manager is not None}")
        # Refresh stage selector when project changes
        self.populate_stage_selector()
        # Do initial refresh (event-driven, no polling timer)
        self.refresh_scene()

    # =========================================================================
    # SceneGraph Signal Handlers
    # =========================================================================

    def _on_graph_changed(self, node_id: str):
        """Handle node added/removed - rebuild tree and mark dirty."""
        self._set_dirty(True)
        if not self._suppress_refresh:
            self.refresh_scene()

    def _on_node_reparented(self, node_id: str, old_parent: str, new_parent: str):
        """Handle node reparented - update tree structure and mark dirty."""
        self._set_dirty(True)
        # For now, rebuild. Later: move item in tree without full rebuild.
        if not self._suppress_refresh:
            self.refresh_scene()

    def _on_node_renamed(self, node_id: str, new_name: str):
        """Handle node renamed - update tree item text and mark dirty."""
        self._set_dirty(True)
        item = self._node_id_to_item.get(node_id)
        if item:
            item.setText(0, new_name)

    def update_entity_name(self, entity_type: str, entity_id: str, new_name: str):
        """Update tree item name when changed from Inspector."""
        print(f"[DEBUG] update_entity_name: type={entity_type}, id={entity_id}, name={new_name}")

        # Find the tree item by entity_id
        found = False
        for i in range(self.tree.topLevelItemCount()):
            item = self._find_item_by_id(self.tree.topLevelItem(i), entity_id)
            if item:
                found = True
                entity_data = item.data(0, Qt.ItemDataRole.UserRole)
                if entity_data:
                    # Update display text in column 0 (name only, pause button is in column 1)
                    if entity_type == 'zone':
                        # Zones show radius/falloff in name
                        data = entity_data.get('data', {})
                        radius = data.get('radius', 10)
                        falloff = data.get('falloff', 5)
                        display_text = f"{new_name} (r={radius}, f={falloff})"
                        item.setText(0, display_text)
                    else:
                        # Noodlings, props, etc - just the name (button in column 1)
                        item.setText(0, new_name)

                    # Update entity_data
                    entity_data['name'] = new_name
                    item.setData(0, Qt.ItemDataRole.UserRole, entity_data)

                    # Update scene graph
                    node_id = entity_data.get('node_id')
                    if node_id:
                        node = self.scene_graph.get_node(node_id)
                        if node:
                            # Rename without triggering signal (already updating tree)
                            node.name = new_name

                    self._set_dirty(True)
                    self._save_hierarchy()
                break
        if not found:
            print(f"[DEBUG] Entity not found in tree! id={entity_id}")

    def _find_item_by_id(self, item, entity_id: str):
        """Recursively find tree item by entity ID."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data and entity_data.get('id') == entity_id:
            return item
        for i in range(item.childCount()):
            found = self._find_item_by_id(item.child(i), entity_id)
            if found:
                return found
        return None

    # =========================================================================
    # Drag and Drop Handling
    # =========================================================================

    def _handle_tree_drop(self, event):
        """Handle drop event for reparenting nodes."""
        # Get the item being dropped on
        drop_item = self.tree.itemAt(event.position().toPoint())

        # Get the dragged items
        dragged_items = self.tree.selectedItems()
        if not dragged_items:
            event.ignore()
            return

        dragged_item = dragged_items[0]
        dragged_node_id = self._item_id_to_node_id.get(id(dragged_item))

        if not dragged_node_id:
            event.ignore()
            return

        dragged_node = self.scene_graph.get_node(dragged_node_id)
        if not dragged_node:
            event.ignore()
            return

        # Can't move virtual nodes (bones)
        if dragged_node.is_virtual:
            event.ignore()
            return

        # Determine new parent
        new_parent_id = None
        if drop_item:
            new_parent_id = self._item_id_to_node_id.get(id(drop_item))

            # If dropping on non-folder, drop as sibling instead
            if new_parent_id:
                drop_node = self.scene_graph.get_node(new_parent_id)
                if drop_node and drop_node.node_type != SceneNodeType.FOLDER:
                    # Make sibling - use drop node's parent
                    new_parent_id = drop_node.parent_id

        # Prevent invalid reparent operations
        if new_parent_id == dragged_node_id:
            event.ignore()
            return

        # Perform reparent in scene graph
        self._suppress_refresh = True
        success = self.scene_graph.reparent(dragged_node_id, new_parent_id)
        self._suppress_refresh = False

        if success:
            # Save hierarchy
            self._save_hierarchy()
            # Rebuild tree
            self.refresh_scene()
            event.accept()
        else:
            event.ignore()

    # =========================================================================
    # Hierarchy Persistence
    # =========================================================================

    def _save_hierarchy(self):
        """Save the scene graph to hierarchy.yaml."""
        if not self.project_manager or not self.project_manager.is_project_open():
            return False

        if not self.current_stage:
            return False

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            return False

        hierarchy_path = os.path.join(stage_path, "hierarchy.yaml")
        self.scene_graph.save(hierarchy_path)
        self._dirty = False
        print(f"[SceneHierarchy] Saved hierarchy to {hierarchy_path}")
        return True

    def save_stage(self) -> bool:
        """Public method to save current stage hierarchy. Returns True if saved."""
        return self._save_hierarchy()

    def is_dirty(self) -> bool:
        """Return True if there are unsaved hierarchy changes."""
        return self._dirty

    def _set_dirty(self, dirty: bool = True):
        """Mark the hierarchy as having unsaved changes."""
        self._dirty = dirty

    def _load_hierarchy(self, stage_path: str) -> bool:
        """Load scene graph from hierarchy.yaml if it exists."""
        hierarchy_path = os.path.join(stage_path, "hierarchy.yaml")
        if os.path.exists(hierarchy_path):
            return self.scene_graph.load(hierarchy_path)
        return False

    def _sync_names_from_disk(self, stage_path: str):
        """Update node names from disk files (in case inspector changed them)."""
        import yaml

        for node_id, node in list(self.scene_graph.nodes.items()):
            if not node.asset_path:
                continue

            # Determine which YAML file to check
            yaml_file = None
            name_key = 'name'

            if node.node_type == SceneNodeType.PROP:
                yaml_file = os.path.join(node.asset_path, 'prop.yaml')
            elif node.node_type == SceneNodeType.NOODLING:
                yaml_file = os.path.join(node.asset_path, 'instance.yaml')
                name_key = 'overrides.name'
            elif node.node_type == SceneNodeType.ZONE:
                yaml_file = node.asset_path  # Zone path IS the yaml file

            if yaml_file and os.path.exists(yaml_file):
                try:
                    with open(yaml_file, 'r') as f:
                        data = yaml.safe_load(f) or {}

                    # Get name from data
                    if name_key == 'overrides.name':
                        disk_name = data.get('overrides', {}).get('name', '')
                    else:
                        disk_name = data.get('name', '')

                    # Update node if name changed
                    if disk_name and disk_name != node.name:
                        print(f"[SceneHierarchy] Syncing name from disk: {node.name} -> {disk_name}")
                        self.scene_graph.rename_node(node_id, disk_name)
                except Exception as e:
                    print(f"[SceneHierarchy] Error syncing name for {node.name}: {e}")

    def _add_new_files_to_hierarchy(self, stage_path: str):
        """Detect files on disk not in hierarchy and add them."""
        import yaml

        # CRITICAL: Suppress refresh during file detection to prevent signal cascade
        # (nodeAdded signals would otherwise trigger refresh_scene() recursively)
        self._suppress_refresh = True

        try:
            # Get existing asset paths in hierarchy
            existing_paths = set()
            for node in self.scene_graph.nodes.values():
                if node.asset_path:
                    existing_paths.add(node.asset_path)

            # New items are added at root (parent_id=None)

            # Check for new zones
            zones_dir = os.path.join(stage_path, "Zones")
            if os.path.exists(zones_dir):
                for filename in os.listdir(zones_dir):
                    if filename.endswith(".zone.yaml"):
                        zone_path = os.path.join(zones_dir, filename)
                        if zone_path not in existing_paths:
                            try:
                                with open(zone_path, 'r') as f:
                                    zone_data = yaml.safe_load(f) or {}
                                zone_name = zone_data.get('name', filename.replace('.zone.yaml', ''))
                                self.scene_graph.create_node(
                                    zone_name, SceneNodeType.ZONE, None, zone_path)
                                print(f"[SceneHierarchy] Added new zone: {zone_name}")
                            except Exception as e:
                                print(f"[SceneHierarchy] Error adding zone {filename}: {e}")

            # Check for new instances (noodlings)
            instances_dir = os.path.join(stage_path, "Instances")
            if os.path.exists(instances_dir):
                for inst_name in os.listdir(instances_dir):
                    inst_path = os.path.join(instances_dir, inst_name)
                    if not os.path.isdir(inst_path):
                        continue
                    if inst_path not in existing_paths:
                        inst_yaml = os.path.join(inst_path, "instance.yaml")
                        if os.path.exists(inst_yaml):
                            try:
                                with open(inst_yaml, 'r') as f:
                                    inst_data = yaml.safe_load(f) or {}
                                display_name = inst_data.get('overrides', {}).get('name', inst_name)
                                self.scene_graph.create_node(
                                    display_name, SceneNodeType.NOODLING, None, inst_path)
                                print(f"[SceneHierarchy] Added new noodling: {display_name}")
                            except Exception as e:
                                print(f"[SceneHierarchy] Error adding noodling {inst_name}: {e}")

            # Check for new props
            props_dir = os.path.join(stage_path, "Props")
            if os.path.exists(props_dir):
                for prop_name in os.listdir(props_dir):
                    prop_path = os.path.join(props_dir, prop_name)
                    if not os.path.isdir(prop_path):
                        continue
                    if prop_path not in existing_paths:
                        prop_yaml = os.path.join(prop_path, "prop.yaml")
                        if os.path.exists(prop_yaml):
                            try:
                                with open(prop_yaml, 'r') as f:
                                    prop_data = yaml.safe_load(f) or {}
                                display_name = prop_data.get('name', prop_name)
                                self.scene_graph.create_node(
                                    display_name, SceneNodeType.PROP, None, prop_path)
                                print(f"[SceneHierarchy] Added new prop: {display_name}")
                            except Exception as e:
                                print(f"[SceneHierarchy] Error adding prop {prop_name}: {e}")

            # Check for new UI canvases (ui.yaml and *.ui.yaml)
            for filename in os.listdir(stage_path):
                if filename == 'ui.yaml' or filename.endswith('.ui.yaml'):
                    ui_path = os.path.join(stage_path, filename)
                    if ui_path not in existing_paths:
                        # Determine display name
                        if filename == 'ui.yaml':
                            display_name = "UI"
                        else:
                            display_name = f"UI: {filename.replace('.ui.yaml', '')}"
                        self.scene_graph.create_node(
                            display_name, SceneNodeType.UI, None, ui_path)
                        print(f"[SceneHierarchy] Added new UI canvas: {display_name}", flush=True)
        finally:
            self._suppress_refresh = False

    def clear_for_project_change(self):
        """Clear all state when switching to a different project."""
        self._suppress_refresh = True
        try:
            self.scene_graph.clear()
            self.tree.clear()
            self._item_id_to_node_id.clear()
            self._node_id_to_item.clear()
            self.expanded_items.clear()
            self.selected_item_path = None
            self.current_stage = None
        finally:
            self._suppress_refresh = False

    # =========================================================================
    # Expanded State Management
    # =========================================================================

    def save_expanded_state(self):
        """Save which items are currently expanded and selected."""
        def get_expanded_paths(item, path=""):
            """Recursively collect paths of expanded items."""
            current_path = path + "/" + item.text(0) if path else item.text(0)
            if item.isExpanded():
                self.expanded_items.add(current_path)
            # Check if this is the selected item
            if item.isSelected():
                self.selected_item_path = current_path
            for i in range(item.childCount()):
                get_expanded_paths(item.child(i), current_path)

        self.expanded_items.clear()
        for i in range(self.tree.topLevelItemCount()):
            get_expanded_paths(self.tree.topLevelItem(i))

    def restore_expanded_state(self):
        """Restore expanded state and selection for items that match saved paths."""
        # NOTE: Signals should already be blocked by caller (refresh_scene)
        # This prevents the 2-second refresh from overwriting Inspector when facet is selected

        def restore_item(item, path=""):
            """Recursively restore expanded state and selection."""
            current_path = path + "/" + item.text(0) if path else item.text(0)

            # Restore expanded state from saved set
            if current_path in self.expanded_items:
                item.setExpanded(True)
            else:
                # Explicitly collapse items not in the expanded set
                item.setExpanded(False)

            # Restore selection
            if self.selected_item_path and current_path == self.selected_item_path:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
            for i in range(item.childCount()):
                restore_item(item.child(i), current_path)

        for i in range(self.tree.topLevelItemCount()):
            restore_item(self.tree.topLevelItem(i))

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
