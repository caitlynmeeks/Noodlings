"""
Scene Hierarchy Utils Mixin - Miscellaneous utility methods

Contains:
- unpack_ensemble: Unpack ensemble to scene
- view_ensemble_info: Show ensemble info dialog
- remove_item_from_tree: Remove tree item
- set_server_state: Update server status
- get_agent_pause_state: Get agent pause state
- toggle_cognition_pause_data: Toggle agent cognition pause
- dropEvent: Handle drops from Assets panel
- check_and_unpack_dropped_ensembles: Auto-unpack dropped ensembles

Author: Noodlings Project
Date: December 2025
"""

import os
import json

from PyQt6.QtWidgets import QTreeWidgetItem, QMessageBox
from PyQt6.QtCore import Qt, QTimer

import requests


class SceneHierarchyUtilsMixin:
    """Mixin providing utility methods for SceneHierarchy."""

    # =========================================================================
    # Ensemble Operations
    # =========================================================================

    def unpack_ensemble(self, ensemble_filename):
        """Unpack an ensemble - rez all members with shared context."""
        # Get parent window's assets panel to call its load function
        if hasattr(self.parent(), 'assets'):
            self.parent().assets._load_ensemble(ensemble_filename)
        else:
            QMessageBox.warning(self, "Cannot Unpack", "Assets panel not found.")

    def view_ensemble_info(self, ensemble_filename):
        """View ensemble information (mission, roles, dynamics)."""
        try:
            # Try to load from project first, then fall back to cmush/ensembles
            ensemble_path = None

            # Check if we have a project manager via parent
            if hasattr(self.parent(), 'assets') and hasattr(self.parent().assets, 'project_manager'):
                pm = self.parent().assets.project_manager
                if pm and pm.is_project_open():
                    ensembles_dir = pm.get_assets_path("Ensembles")
                    ensemble_path = os.path.join(ensembles_dir, ensemble_filename)

            # Fallback to cmush ensembles
            if not ensemble_path or not os.path.exists(ensemble_path):
                ensemble_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/ensembles",
                    ensemble_filename
                )

            with open(ensemble_path, 'r') as f:
                ensemble = json.load(f)

            # Format info
            info = f"{ensemble.get('name', 'Unknown Ensemble')}\n"
            info += f"Type: {ensemble.get('ensemble_type', 'unknown')}\n\n"
            info += f"SHARED MISSION:\n{ensemble.get('shared_mission', 'None')}\n\n"

            dynamics = ensemble.get('ensemble_dynamics', {})
            if dynamics:
                info += "ENSEMBLE DYNAMICS:\n"
                info += f"  Interaction: {dynamics.get('interaction_style', 'unknown')}\n"
                info += f"  Decision Making: {dynamics.get('decision_making', 'unknown')}\n\n"

                roles = dynamics.get('role_distribution', {})
                if roles:
                    info += "ROLES:\n"
                    for member, role in roles.items():
                        info += f"  {member}: {role}\n"

            info += f"\n\nMembers: {len(ensemble.get('agents', []))}"

            QMessageBox.information(self, "Ensemble Info", info)

        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load ensemble info: {e}")

    # =========================================================================
    # Tree Item Management
    # =========================================================================

    def remove_item_from_tree(self, item):
        """Remove an item from the tree (doesn't derez, just removes from view)."""
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            index = self.tree.indexOfTopLevelItem(item)
            if index >= 0:
                self.tree.takeTopLevelItem(index)

    def _remove_tree_item_by_node_id(self, node_id: str):
        """Find and remove a tree item by its scene graph node_id."""
        if not node_id:
            return

        # Search all items in tree
        def find_and_remove(parent_item):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                entity_data = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(entity_data, dict):
                    item_node_id = entity_data.get('node_id', '')
                    if item_node_id == node_id:
                        parent_item.removeChild(child)
                        return True
                # Recurse
                if find_and_remove(child):
                    return True
            return False

        # Search top-level items
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(entity_data, dict):
                item_node_id = entity_data.get('node_id', '')
                if item_node_id == node_id:
                    self.tree.takeTopLevelItem(i)
                    return
            # Search children
            if find_and_remove(item):
                return

    def _select_item_by_id(self, item_id: str):
        """Find and select an item in the tree by its ID, triggering Inspector update."""
        def find_item(parent_item, target_id):
            for i in range(parent_item.childCount()):
                child = parent_item.child(i)
                data = child.data(0, Qt.ItemDataRole.UserRole)
                if isinstance(data, dict) and data.get('id') == target_id:
                    return child
                # Recurse into children
                found = find_item(child, target_id)
                if found:
                    return found
            return None

        # Search from root
        for i in range(self.tree.topLevelItemCount()):
            top_item = self.tree.topLevelItem(i)
            data = top_item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(data, dict) and data.get('id') == item_id:
                self.tree.clearSelection()
                top_item.setSelected(True)
                self.tree.setCurrentItem(top_item)
                self.tree.scrollToItem(top_item)
                return

            found = find_item(top_item, item_id)
            if found:
                self.tree.clearSelection()
                found.setSelected(True)
                self.tree.setCurrentItem(found)
                self.tree.scrollToItem(found)
                return

    # =========================================================================
    # Server State
    # =========================================================================

    def set_server_state(self, running: bool):
        """Update hierarchy based on server state."""
        print(f"[SceneHierarchy] set_server_state({running})")
        self._server_running = running

        if running:
            # Server online - enable tree and refresh to show full hierarchy
            self.tree.setEnabled(True)
            self.tree.setStyleSheet(self.tree.styleSheet().replace("color: #666;", "color: #D2D2D2;"))
        else:
            # Server offline - show offline message instead of full tree
            self.tree.setEnabled(True)  # Keep enabled so user can see the message

        # Refresh to show appropriate content based on server state
        self.refresh_scene()

    def _gray_out_item(self, item):
        """Recursively gray out an item and its children."""
        from PyQt6.QtGui import QColor
        item.setForeground(0, QColor(100, 100, 100))  # Dark gray
        for i in range(item.childCount()):
            self._gray_out_item(item.child(i))

    # =========================================================================
    # Agent Pause State
    # =========================================================================

    def get_agent_pause_state(self, agent_id: str) -> bool:
        """Get the pause state for a specific agent."""
        return self.agent_pause_states.get(agent_id, False)

    def toggle_cognition_pause_data(self, entity_data):
        """Toggle cognition pause for a noodling (uses entity data)."""
        agent_id = entity_data.get('id')
        if not agent_id:
            QMessageBox.warning(self, "Error", "Cannot find agent ID")
            return

        try:
            # Get current pause state
            is_paused = self.get_agent_pause_state(agent_id)
            new_pause_state = not is_paused

            # Send API request
            url = f"{self.api_base}/cognition/pause"
            print(f"[Stage] Sending pause API request to {url}")
            print(f"[Stage] Payload: paused={new_pause_state}, agent_id={agent_id}")
            response = requests.post(url, json={'paused': new_pause_state, 'agent_id': agent_id}, timeout=35)
            print(f"[Stage] API response: {response.status_code}")

            if response.status_code == 200:
                # Update tracked state
                self.agent_pause_states[agent_id] = new_pause_state

                # Notify Facets Editor if it's editing this agent
                # (Need to emit signal or call parent)
                if hasattr(self, 'parent') and self.parent():
                    main_window = self.parent()
                    while main_window and not hasattr(main_window, 'facets_editor'):
                        main_window = main_window.parent() if hasattr(main_window, 'parent') else None

                    if main_window and hasattr(main_window, 'facets_editor'):
                        facets_editor = main_window.facets_editor
                        if facets_editor.current_agent_id == agent_id:
                            # Update Facets Editor pause state
                            facets_editor.cognition_paused = new_pause_state
                            facets_editor.pause_button.setChecked(new_pause_state)
                            facets_editor.bottom_pause_btn.setChecked(new_pause_state)
                            if new_pause_state:
                                facets_editor.pause_button.setText("> Resume Cognition")
                                facets_editor.bottom_pause_btn.setText(">")
                            else:
                                facets_editor.pause_button.setText("|| Pause Cognition")
                                facets_editor.bottom_pause_btn.setText("||")

                # Refresh tree to update icon
                self.refresh_scene()

                # Log success (no popup)
                state_text = "paused" if new_pause_state else "resumed"
                agent_name = entity_data.get('name', agent_id)
                print(f"[Stage] Cognition {state_text} for {agent_name}")
            else:
                QMessageBox.warning(self, "API Error", f"Failed to toggle cognition: {response.status_code}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to toggle cognition pause: {str(e)}")

    # =========================================================================
    # Drag and Drop from Assets Panel
    # =========================================================================

    def dropEvent(self, event):
        """Handle drop from Assets panel - automatically unpack ensembles."""
        # Get the mime data
        mime = event.mimeData()

        # Check if this is from our Assets panel
        if mime.hasText():
            # The dropped item should have data attached
            # For now, just accept the drop and let parent handle it
            super().dropEvent(event)

            # After drop, check if an ensemble was dropped and unpack it
            QTimer.singleShot(100, self.check_and_unpack_dropped_ensembles)

    def check_and_unpack_dropped_ensembles(self):
        """Check for dropped ensemble items and automatically unpack them."""
        # Find any ensemble items in the tree (tuples)
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            self._check_item_for_ensemble(item)

    def _check_item_for_ensemble(self, item):
        """Recursively check item and children for ensembles to unpack."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)

        if isinstance(entity_data, tuple):
            asset_type, asset_name = entity_data
            if asset_type == "ensemble":
                # Found an ensemble - unpack it!
                print(f"Auto-unpacking dropped ensemble: {asset_name}")
                self.unpack_ensemble(asset_name)

                # Remove the placeholder item
                parent = item.parent()
                if parent:
                    parent.removeChild(item)
                else:
                    index = self.tree.indexOfTopLevelItem(item)
                    if index >= 0:
                        self.tree.takeTopLevelItem(index)
                return

        # Check children
        for i in range(item.childCount()):
            self._check_item_for_ensemble(item.child(i))
