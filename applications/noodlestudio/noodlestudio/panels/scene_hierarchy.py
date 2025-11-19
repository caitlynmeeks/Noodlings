"""
Scene Hierarchy Panel - Unity-style entity tree

Shows all prims in the noodleMUSH world:
- Rooms (with exits)
- Users (Noodlers)
- Noodlings (kindled beings)
- Prims (WANTED POSTER, RADIO, etc.)

Click to select → Inspector shows editable properties

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QLabel, QPushButton, QMenu, QInputDialog)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QAction
import requests
import sys
import os
sys.path.append('..')
from noodlestudio.widgets.maximizable_dock import MaximizableDock


class SceneHierarchy(MaximizableDock):
    """
    Unity-style Scene Hierarchy panel.

    Tree structure:
    ┬ Scene: The Nexus (room_000)
    ├─┬ Users
    │ └─ caity [Noodler, 9yo, she/her]
    ├─┬ Noodlings
    │ ├─ Phi [kitten, they]
    │ ├─ Servnak [robot, they]
    │ └─ Callie [noodling, they]
    ├─┬ Objects
    │ ├─ WANTED POSTER
    │ └─ RADIO
    └─┬ Exits
      ├─ north → The Forest Path
      └─ east → The Pond
    """

    entitySelected = pyqtSignal(str, dict)  # (entity_type, entity_data)

    def __init__(self, parent=None):
        super().__init__("Stage Hierarchy", parent)
        self.api_base = "http://localhost:8081/api"
        self.current_room = "room_000"  # Start at Nexus

        # Track expanded state (survives tree rebuild)
        self.expanded_items = set()

        # Track selected item (survives tree rebuild)
        self.selected_item_path = None

        # Create central widget
        widget = QWidget()
        self.setWidget(widget)

        self.init_ui(widget)

        # Auto-refresh
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_scene)
        self.refresh_timer.start(2000)

        # Initial load
        self.refresh_scene()

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header = QLabel("STAGE HIERARCHY")
        header.setFont(QFont("Arial", 11, QFont.Weight.Bold))
        header.setStyleSheet("color: #B4B4B4; padding: 4px;")
        layout.addWidget(header)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)

        # Prevent auto-collapse: set animation to false
        self.tree.setAnimated(False)

        # Keep selection visible
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.SingleSelection)
        self.tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)

        # Use itemSelectionChanged to avoid interfering with expand/collapse
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)

        # Enable drag and drop for parenting (Unity-style)
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.InternalMove)

        # Context menu
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self.show_context_menu)

        layout.addWidget(self.tree)

        # Refresh button
        refresh_btn = QPushButton("Refresh Scene")
        refresh_btn.clicked.connect(self.refresh_scene)
        layout.addWidget(refresh_btn)

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
        # If no saved state (first load), set sensible defaults
        if not self.expanded_items:
            self.expanded_items = {
                "Scene: The Nexus",
                "Scene: The Nexus/Connected Users",
                "Scene: The Nexus/Noodlings"
            }

        def restore_item(item, path=""):
            """Recursively restore expanded state and selection."""
            current_path = path + "/" + item.text(0) if path else item.text(0)
            if current_path in self.expanded_items:
                item.setExpanded(True)
            # Restore selection
            if self.selected_item_path and current_path == self.selected_item_path:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
            for i in range(item.childCount()):
                restore_item(item.child(i), current_path)

        for i in range(self.tree.topLevelItemCount()):
            restore_item(self.tree.topLevelItem(i))

    def refresh_scene(self):
        """Refresh scene hierarchy from noodleMUSH world state."""
        try:
            # Save expanded state before clearing
            self.save_expanded_state()

            # TODO: Create proper world API endpoint
            # For now, use agents API and build structure
            agents_resp = requests.get(f"{self.api_base}/agents", timeout=2)
            agents_data = agents_resp.json().get('agents', [])

            self.tree.clear()

            # Root: Current room
            room_item = QTreeWidgetItem(["Scene: The Nexus"])
            room_item.setFont(0, QFont("Arial", 12, QFont.Weight.Bold))
            room_item.setForeground(0, Qt.GlobalColor.white)
            self.tree.addTopLevelItem(room_item)

            # Connected Users folder
            users_folder = QTreeWidgetItem(["Connected Users"])
            users_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(users_folder)

            # Add current user
            user_item = QTreeWidgetItem(["caity [Noodler, 9yo, she/her]"])
            user_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'user', 'id': 'user_caity'})
            users_folder.addChild(user_item)

            # Noodlings folder
            noodlings_folder = QTreeWidgetItem(["Noodlings"])
            noodlings_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(noodlings_folder)

            # Add each Noodling
            for agent in agents_data:
                name = agent.get('name', agent.get('id'))
                species = agent.get('species', 'unknown')
                agent_id = agent.get('id')

                noodling_item = QTreeWidgetItem([f"{name} [{species}]"])
                noodling_item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'noodling',
                    'id': agent_id,
                    'data': agent
                })
                noodlings_folder.addChild(noodling_item)

            # Prims folder (USD terminology!)
            prims_folder = QTreeWidgetItem(["Prims"])
            prims_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(prims_folder)

            # TODO: Load actual prims from world state
            poster_item = QTreeWidgetItem(["WANTED POSTER"])
            poster_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'prim',
                'id': 'obj_wanted_poster'
            })
            prims_folder.addChild(poster_item)

            radio_item = QTreeWidgetItem(["RADIO"])
            radio_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'prim',
                'id': 'obj_radio'
            })
            prims_folder.addChild(radio_item)

            # Exits folder
            exits_folder = QTreeWidgetItem(["Exits"])
            exits_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(exits_folder)

            # TODO: Load actual exits from room data
            north_exit = QTreeWidgetItem(["north → The Forest Path"])
            north_exit.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'exit',
                'direction': 'north',
                'destination': 'room_001'
            })
            exits_folder.addChild(north_exit)

            east_exit = QTreeWidgetItem(["east → The Pond"])
            east_exit.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'exit',
                'direction': 'east',
                'destination': 'room_002'
            })
            exits_folder.addChild(east_exit)

            # Restore expanded state after rebuilding tree
            self.restore_expanded_state()

        except Exception as e:
            print(f"Error refreshing scene: {e}")

    def on_selection_changed(self):
        """Handle entity selection (doesn't interfere with expand/collapse)."""
        items = self.tree.selectedItems()
        if items:
            item = items[0]
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if entity_data:
                entity_type = entity_data.get('type', 'unknown')
                self.entitySelected.emit(entity_type, entity_data)

    def on_item_clicked(self, item: QTreeWidgetItem, column: int = 0):
        """Handle direct item clicks (used by context menu)."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data:
            entity_type = entity_data.get('type', 'unknown')
            self.entitySelected.emit(entity_type, entity_data)

    def show_context_menu(self, position):
        """Show right-click context menu (Unity-style)."""
        item = self.tree.itemAt(position)

        menu = QMenu()

        if item:
            # Capture data immediately (item may be deleted after menu closes)
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            entity_type = entity_data.get('type', '') if entity_data else None

            # Context-specific actions (capture data, not item reference)
            if entity_type == 'noodling':
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addAction("Toggle Enlightenment", lambda d=entity_data: self.toggle_enlightenment_data(d))
                menu.addSeparator()
                menu.addAction("Export Noodling", lambda d=entity_data: self.export_noodling_data(d))
                menu.addSeparator()
                menu.addAction("Duplicate Noodling", lambda d=entity_data: self.duplicate_prim_data(d))
                menu.addAction("Reset State", lambda d=entity_data: self.reset_prim_state_data(d))
                menu.addSeparator()
                menu.addAction("Delete Noodling", lambda d=entity_data: self.delete_prim_data(d))

            elif entity_type == 'prim':
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addAction("Edit Description", lambda d=entity_data: self.edit_description_data(d))
                menu.addSeparator()
                menu.addAction("Duplicate Prim", lambda d=entity_data: self.duplicate_prim_data(d))
                menu.addAction("Delete Prim", lambda d=entity_data: self.delete_prim_data(d))

            elif entity_type == 'user':
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addAction("View Profile", lambda d=entity_data: self.view_user_profile_data(d))

            elif entity_type == 'exit':
                menu.addAction("Edit Exit", lambda d=entity_data: self.edit_exit_data(d))
                menu.addAction("Delete Exit", lambda d=entity_data: self.delete_prim_data(d))

            else:
                # Folder or other
                menu.addAction("Expand All", lambda: self.expand_recursive(item))
                menu.addAction("Collapse All", lambda: self.collapse_recursive(item))
        else:
            # Empty space - show create options
            create_menu = menu.addMenu("Create")
            create_menu.addAction("Empty Noodling", lambda: self.create_empty_noodling())
            create_menu.addAction("Empty Prim", lambda: self.create_empty_prim())
            create_menu.addAction("Empty Room", lambda: self.create_empty_room())
            create_menu.addSeparator()
            create_menu.addAction("Custom Prim Type...", lambda: self.create_custom_prim())

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def inspect_entity(self, entity_data):
        """Inspect entity (safe - uses data not item)."""
        entity_type = entity_data.get('type', 'unknown')
        self.entitySelected.emit(entity_type, entity_data)

    def toggle_enlightenment_data(self, entity_data):
        """Toggle enlightenment (uses data)."""
        noodling_id = entity_data.get('id')
        print(f"Toggle enlightenment for {noodling_id}")
        # TODO: Send to noodleMUSH API

    def export_noodling_data(self, entity_data):
        """Export Noodling to YAML file."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        import json
        import yaml

        noodling_id = entity_data.get('id')
        noodling_data = entity_data.get('data', {})

        # Get agent name for default filename
        agent_name = noodling_data.get('name', noodling_id.replace('agent_', ''))

        # Open file save dialog
        default_path = str(Path.home() / f"{agent_name}.yaml")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Noodling",
            default_path,
            "YAML Files (*.yaml);;All Files (*)"
        )

        if file_path:
            try:
                # Fetch full agent data from API
                import requests
                resp = requests.get(f"{self.api_base}/agents/{noodling_id}", timeout=2)
                if resp.status_code == 200:
                    full_data = resp.json()

                    # Export to YAML (recipe format)
                    with open(file_path, 'w') as f:
                        yaml.dump(full_data, f, default_flow_style=False, sort_keys=False)

                    print(f"Exported {agent_name} to {file_path}")
                else:
                    print(f"Failed to fetch agent data: {resp.status_code}")
            except Exception as e:
                print(f"Error exporting noodling: {e}")

    def duplicate_prim_data(self, entity_data):
        """Duplicate a prim (uses data)."""
        prim_type = entity_data.get('type')
        prim_id = entity_data.get('id')
        print(f"Duplicate {prim_type}: {prim_id}")
        # TODO: Send to noodleMUSH API

    def reset_prim_state_data(self, entity_data):
        """Reset prim state (uses data)."""
        prim_id = entity_data.get('id')
        print(f"Reset state for {prim_id}")
        # TODO: Send to noodleMUSH API

    def delete_prim_data(self, entity_data):
        """De-rez a prim or Noodling (delete from scene)."""
        prim_id = entity_data.get('id')
        prim_type = entity_data.get('type')

        from PyQt6.QtWidgets import QMessageBox
        msgBox = QMessageBox(self)
        msgBox.setWindowTitle("De-Rez")
        msgBox.setText(f"De-rez {prim_type} '{prim_id}'?\n\nThis will remove it from the scene.")
        msgBox.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msgBox.setIcon(QMessageBox.Icon.NoIcon)  # No icon!
        reply = msgBox.exec()

        if reply == QMessageBox.StandardButton.Yes:
            # Send @remove command to noodleMUSH
            import subprocess
            try:
                # Use the @remove command via the running server
                # For now, just remove from the scene data and refresh
                print(f"De-rezzing {prim_type}: {prim_id}")

                # De-rez by directly modifying world data AND removing from tree
                if prim_type == 'agent':
                    # Load agents.json and remove the agent
                    import json
                    agents_path = os.path.join(
                        os.path.dirname(__file__),
                        "../../../cmush/world/agents.json"
                    )
                    agents_path = os.path.abspath(agents_path)

                    try:
                        with open(agents_path, 'r') as f:
                            agents = json.load(f)

                        if prim_id in agents:
                            del agents[prim_id]

                            with open(agents_path, 'w') as f:
                                json.dump(agents, f, indent=2)

                            print(f"De-rezzed {prim_id}")

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
                        print(f"Error de-rezzing: {e}")
                else:
                    # For other types, just refresh
                    self.refresh_scene()

            except Exception as e:
                QMessageBox.warning(self, "De-Rez Failed", f"Error: {e}")

    def edit_description_data(self, entity_data):
        """Edit description (uses data)."""
        text, ok = QInputDialog.getMultiLineText(
            self,
            "Edit Description",
            "Object description:",
            ""
        )
        if ok:
            print(f"Update description: {text}")
            # TODO: Send to noodleMUSH API

    def view_user_profile_data(self, entity_data):
        """View user profile (uses data)."""
        user_id = entity_data.get('id')
        print(f"View profile for {user_id}")
        # TODO: Open profile panel

    def edit_exit_data(self, entity_data):
        """Edit exit (uses data)."""
        direction = entity_data.get('direction')
        print(f"Edit exit: {direction}")
        # TODO: Show exit editor dialog

    def expand_recursive(self, item):
        """Expand item and all children."""
        item.setExpanded(True)
        for i in range(item.childCount()):
            self.expand_recursive(item.child(i))

    def collapse_recursive(self, item):
        """Collapse item and all children."""
        item.setExpanded(False)
        for i in range(item.childCount()):
            self.collapse_recursive(item.child(i))

    def create_empty_noodling(self):
        """Create an empty Noodling with default settings."""
        name, ok = QInputDialog.getText(
            self,
            "Create Empty Noodling",
            "Noodling name:",
            text="NewNoodling"
        )
        if ok and name:
            print(f"Creating empty Noodling: {name}")
            # TODO: Send to noodleMUSH API with defaults
            self.refresh_scene()

    def create_empty_prim(self):
        """Create an empty prim."""
        name, ok = QInputDialog.getText(
            self,
            "Create Empty Prim",
            "Prim name:",
            text="NewPrim"
        )
        if ok and name:
            print(f"Creating empty prim: {name}")
            # TODO: Send to noodleMUSH API
            self.refresh_scene()

    def create_empty_room(self):
        """Create an empty room prim."""
        name, ok = QInputDialog.getText(
            self,
            "Create Empty Room",
            "Room name:",
            text="NewRoom"
        )
        if ok and name:
            print(f"Creating empty room: {name}")
            # TODO: Send to noodleMUSH API
            self.refresh_scene()

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
