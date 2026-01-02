"""
Scene Hierarchy Selection Mixin - Selection, editing, renaming

Contains:
- on_selection_changed: Handle entity selection
- on_item_double_clicked: Start inline rename
- _on_item_renamed: Handle rename completion
- _rename_on_disk: Update YAML files on disk
- on_item_clicked_for_expansion: Toggle expansion

Author: Noodlings Project
Date: December 2025
"""

import os

from PyQt6.QtWidgets import QTreeWidgetItem, QLineEdit
from PyQt6.QtCore import Qt, QTimer
from PyQt6 import sip


class SceneHierarchySelectionMixin:
    """Mixin providing selection handling for SceneHierarchy."""

    def on_selection_changed(self):
        """Handle entity selection (doesn't interfere with expand/collapse)."""
        # Don't interfere with inline editing
        if getattr(self, '_editing_item', None):
            return
        try:
            print("[HIERARCHY] on_selection_changed() called")
            items = self.tree.selectedItems()
            if items:
                item = items[0]
                entity_data = item.data(0, Qt.ItemDataRole.UserRole)
                if entity_data:
                    # Handle both dict (normal) and tuple (from Assets panel drag)
                    if isinstance(entity_data, tuple):
                        # Assets panel stores (asset_type, asset_name)
                        asset_type, asset_name = entity_data
                        # For now, don't emit - ensembles have their own handling
                        print("[HIERARCHY] Asset tuple detected, skipping emit")
                        return
                    elif isinstance(entity_data, dict):
                        entity_type = entity_data.get('type', 'unknown')
                        entity_id = entity_data.get('id', 'unknown')
                        print(f"[HIERARCHY] About to emit entitySelected: type={entity_type}, id={entity_id}")
                        try:
                            self.entitySelected.emit(entity_type, entity_data)
                            print(f"[HIERARCHY] emit returned successfully")
                        except Exception as emit_error:
                            print(f"[HIERARCHY] CRASH during emit: {emit_error}")
                            import traceback
                            traceback.print_exc()
                            # Re-raise to show full error
                            raise
            else:
                # Nothing selected - emit empty values to clear Inspector and Facets Editor
                # Note: Signal requires (str, dict) types, can't pass None directly
                print("[HIERARCHY] No selection, emitting empty values")
                self.entitySelected.emit("", {})
                print("[HIERARCHY] Empty emit returned successfully")
        except Exception as e:
            print(f"[HIERARCHY] EXCEPTION in on_selection_changed: {e}")
            import traceback
            traceback.print_exc()

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click - start inline rename (Unity-style)."""
        try:
            # Safety check - ensure item is valid
            if sip.isdeleted(item):
                return

            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if entity_data:
                if isinstance(entity_data, tuple):
                    # Ensemble from Assets - unpack it!
                    asset_type, asset_name = entity_data
                    if asset_type == "ensemble":
                        self.unpack_ensemble(asset_name)
                        return

                # Start inline editing for renaming (Unity-style)
                entity_type = entity_data.get('type', '') if isinstance(entity_data, dict) else ''

                # Don't allow renaming stage root or virtual nodes
                if entity_type == 'stage':
                    return

                node_id = entity_data.get('node_id') if isinstance(entity_data, dict) else None
                if node_id:
                    node = self.scene_graph.get_node(node_id)
                    if node and node.is_virtual:
                        return  # Can't rename bones

                # Make item editable and start editing
                # CRITICAL: Suppress refresh during inline editing to prevent tree rebuild
                self._suppress_refresh = True
                self._editing_item = item  # Track which item is being edited

                # Block signals while modifying flags to prevent itemChanged from firing
                self.tree.blockSignals(True)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
                self.tree.blockSignals(False)

                # Use openPersistentEditor for better text selection support
                index = self.tree.indexFromItem(item, 0)
                self.tree.openPersistentEditor(item, 0)

                # Focus the editor and select all text
                def focus_editor():
                    # Find the editor widget
                    editor = self.tree.itemWidget(item, 0)
                    if not editor:
                        # Try to find QLineEdit child
                        for child in self.tree.findChildren(QLineEdit):
                            if child.isVisible():
                                child.setFocus()
                                child.selectAll()
                                break
                QTimer.singleShot(50, focus_editor)
        except Exception as e:
            print(f"[HIERARCHY] Error in on_item_double_clicked: {e}")
            import traceback
            traceback.print_exc()
            self._suppress_refresh = False
            self._editing_item = None

    def _on_item_renamed(self, item: QTreeWidgetItem, column: int):
        """Handle inline rename completion."""
        print(f"[HIERARCHY] _on_item_renamed called! column={column}")

        # Re-enable refresh now that editing is done
        self._suppress_refresh = False
        self._editing_item = None

        # Safety check - ensure item is valid
        try:
            if sip.isdeleted(item):
                print("[HIERARCHY] Item was deleted, returning")
                return
        except RuntimeError:
            print("[HIERARCHY] RuntimeError checking item, returning")
            return

        # Close persistent editor and make item non-editable again
        # MUST defer to avoid Qt crash (can't call setFlags inside itemChanged signal handler)
        def cleanup_editor():
            try:
                if not sip.isdeleted(item):
                    self.tree.closePersistentEditor(item, 0)
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            except RuntimeError:
                pass  # Item was deleted
        QTimer.singleShot(0, cleanup_editor)

        try:
            new_name = item.text(0).strip()
            if not new_name:
                return

            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if not entity_data or not isinstance(entity_data, dict):
                return

            old_name = entity_data.get('name', '')
            if new_name == old_name:
                return  # No change

            # Always update entity_data with new name
            entity_data['name'] = new_name
            item.setData(0, Qt.ItemDataRole.UserRole, entity_data)

            node_id = entity_data.get('node_id')
            if node_id:
                # Update scene graph
                node = self.scene_graph.get_node(node_id)
                if node:
                    self.scene_graph.rename_node(node_id, new_name)
                    self._save_hierarchy()

            # Also update prop/noodling/zone on disk if it has a path
            prop_path = entity_data.get('path')
            if prop_path:
                self._rename_on_disk(entity_data, new_name)

            print(f"Renamed '{old_name}' to '{new_name}'")

            # Re-emit entitySelected to update Inspector with new name
            entity_type = entity_data.get('type', 'unknown')
            self.entitySelected.emit(entity_type, entity_data)
        except Exception as e:
            print(f"[HIERARCHY] Error in _on_item_renamed: {e}")
            import traceback
            traceback.print_exc()

    def _rename_on_disk(self, entity_data: dict, new_name: str):
        """Update the display name in the YAML file on disk."""
        import yaml

        entity_type = entity_data.get('type', '')
        prop_path = entity_data.get('path')

        if not prop_path:
            return

        yaml_file = None
        name_key = 'name'

        if entity_type == 'prop':
            yaml_file = os.path.join(prop_path, 'prop.yaml')
        elif entity_type in ('instance', 'noodling'):
            # Noodling instances store name in instance.yaml under overrides.name
            yaml_file = os.path.join(prop_path, 'instance.yaml')
            name_key = 'overrides.name'  # Nested key
        elif entity_type == 'zone':
            yaml_file = prop_path  # Zone path is the yaml file itself

        if yaml_file and os.path.exists(yaml_file):
            try:
                with open(yaml_file, 'r') as f:
                    data = yaml.safe_load(f) or {}

                if name_key == 'overrides.name':
                    if 'overrides' not in data:
                        data['overrides'] = {}
                    data['overrides']['name'] = new_name
                else:
                    data[name_key] = new_name

                with open(yaml_file, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False)
            except Exception as e:
                print(f"Error updating name on disk: {e}")

    def on_item_clicked_for_expansion(self, item: QTreeWidgetItem, column: int):
        """
        Handle item click to toggle expansion (consistent with Inspector CollapsibleSection).

        Clicking text on items with children will expand/collapse them.
        Leaf items (no children) are unaffected - only selection occurs.
        """
        # Don't interfere with inline editing
        if getattr(self, '_editing_item', None):
            return
        if item.childCount() > 0:
            # Item has children - toggle expanded state
            item.setExpanded(not item.isExpanded())

    def on_item_clicked(self, item: QTreeWidgetItem, column: int = 0):
        """Handle direct item clicks (used by context menu)."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data:
            # Handle both dict and tuple
            if isinstance(entity_data, dict):
                entity_type = entity_data.get('type', 'unknown')
                self.entitySelected.emit(entity_type, entity_data)
