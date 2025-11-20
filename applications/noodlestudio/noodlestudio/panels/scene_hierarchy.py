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

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QLabel, QPushButton, QMenu, QInputDialog, QComboBox,
                             QMessageBox)
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

        # Derez confirmation settings
        self.derez_confirm = True  # Show confirmation dialog

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

        # Stage selector dropdown
        stage_layout = QHBoxLayout()
        stage_label = QLabel("Stage:")
        stage_label.setStyleSheet("color: #D2D2D2; padding: 2px;")
        stage_layout.addWidget(stage_label)

        self.stage_selector = QComboBox()
        self.stage_selector.setStyleSheet("""
            QComboBox {
                background-color: #1E1E1E;
                color: #D2D2D2;
                border: 1px solid #555;
                padding: 4px;
                border-radius: 3px;
            }
            QComboBox:hover {
                border: 1px solid #888;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 4px solid transparent;
                border-right: 4px solid transparent;
                border-top: 6px solid #D2D2D2;
                margin-right: 8px;
            }
        """)
        self.stage_selector.currentTextChanged.connect(self.on_stage_changed)
        stage_layout.addWidget(self.stage_selector, stretch=1)

        layout.addLayout(stage_layout)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(16)

        # Prevent auto-collapse: set animation to false
        self.tree.setAnimated(False)

        # Enable multi-selection for batch derez
        self.tree.setSelectionMode(QTreeWidget.SelectionMode.ExtendedSelection)
        self.tree.setSelectionBehavior(QTreeWidget.SelectionBehavior.SelectRows)

        # Use itemSelectionChanged to avoid interfering with expand/collapse
        self.tree.itemSelectionChanged.connect(self.on_selection_changed)

        # Double-click to unpack ensembles or inspect entities
        self.tree.itemDoubleClicked.connect(self.on_item_double_clicked)

        # Enable drag and drop for parenting (Unity-style) and rezzing from Assets
        self.tree.setDragEnabled(True)
        self.tree.setAcceptDrops(True)
        self.tree.setDropIndicatorShown(True)
        self.tree.setDragDropMode(QTreeWidget.DragDropMode.DragDrop)  # Accept external drops

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
        def restore_item(item, path=""):
            """Recursively restore expanded state and selection."""
            current_path = path + "/" + item.text(0) if path else item.text(0)

            # Always expand top-level stage item
            if item.text(0).startswith("Stage:") and not path:
                item.setExpanded(True)
            elif current_path in self.expanded_items:
                item.setExpanded(True)
            # Restore selection
            if self.selected_item_path and current_path == self.selected_item_path:
                item.setSelected(True)
                self.tree.setCurrentItem(item)
            for i in range(item.childCount()):
                restore_item(item.child(i), current_path)

        for i in range(self.tree.topLevelItemCount()):
            restore_item(self.tree.topLevelItem(i))

    def populate_stage_selector(self):
        """Populate stage selector with available stages/rooms."""
        try:
            import json
            rooms_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world/rooms.json"
            )
            with open(rooms_path, 'r') as f:
                rooms_data = json.load(f)

            # Block signals during population to avoid triggering on_stage_changed
            self.stage_selector.blockSignals(True)
            self.stage_selector.clear()

            for room_id, room_data in rooms_data.items():
                room_name = room_data.get('name', room_id)
                display_text = f"{room_name} ({room_id})"
                self.stage_selector.addItem(display_text, room_id)

                # Select current room
                if room_id == self.current_room:
                    self.stage_selector.setCurrentText(display_text)

            self.stage_selector.blockSignals(False)

        except Exception as e:
            print(f"Error populating stage selector: {e}")

    def on_stage_changed(self, text):
        """Handle stage selection change."""
        if not text:
            return

        # Get room_id from combo box data
        new_room_id = self.stage_selector.currentData()
        if new_room_id == self.current_room:
            return

        # Show teleport popup
        room_name = text.split(' (')[0] if ' (' in text else text
        reply = QMessageBox.question(
            self,
            "Teleport to Stage?",
            f"Teleport your character to stage '{room_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Send teleport command to MUSH
            self.teleport_to_stage(new_room_id)

        # Update current room and refresh regardless of teleport choice
        self.current_room = new_room_id
        self.refresh_scene()

    def teleport_to_stage(self, room_id):
        """Send teleport command to noodleMUSH."""
        try:
            # TODO: Implement teleport API endpoint
            # For now, just log it
            print(f"Teleporting to stage: {room_id}")
        except Exception as e:
            print(f"Error teleporting: {e}")

    def refresh_scene(self):
        """Refresh scene hierarchy from noodleMUSH world state."""
        # Populate stage selector on first refresh
        if self.stage_selector.count() == 0:
            self.populate_stage_selector()

        try:
            # Save expanded state before clearing
            self.save_expanded_state()

            # TODO: Create proper world API endpoint
            # For now, use agents API and build structure
            agents_resp = requests.get(f"{self.api_base}/agents", timeout=2)
            agents_data = agents_resp.json().get('agents', [])

            self.tree.clear()

            # Root: Current stage - Get stage data from rooms.json
            stage_name = "Unknown Stage"
            stage_data = None
            try:
                import json
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
                pass

            room_item = QTreeWidgetItem([f"Stage: {stage_name}"])
            room_item.setFont(0, QFont("Arial", 12, QFont.Weight.Bold))
            room_item.setForeground(0, Qt.GlobalColor.white)
            # Set stage data for Inspector
            room_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'stage',
                'id': self.current_room,
                'data': stage_data or {'name': stage_name}
            })
            self.tree.addTopLevelItem(room_item)
            # Expand stage by default
            room_item.setExpanded(True)

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

            # Add each Noodling (filter by location)
            for agent in agents_data:
                # Only show agents in current stage
                agent_location = agent.get('location') or agent.get('current_room')
                if agent_location != self.current_room:
                    continue

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

            # Prims folder (USD terminology!) - Always show, load from world
            prims_folder = QTreeWidgetItem(["Prims"])
            prims_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(prims_folder)

            # Load actual prims from objects.json
            try:
                import json
                objects_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/world/objects.json"
                )
                with open(objects_path, 'r') as f:
                    objects_data = json.load(f)

                # Add each prim in current room
                for obj_id, obj_data in objects_data.items():
                    if obj_data.get('location') == self.current_room:
                        prim_name = obj_data.get('name', obj_id)
                        prim_item = QTreeWidgetItem([prim_name])
                        prim_item.setData(0, Qt.ItemDataRole.UserRole, {
                            'type': 'prim',
                            'id': obj_id,
                            'data': obj_data
                        })
                        prims_folder.addChild(prim_item)

            except Exception as e:
                print(f"Error loading prims: {e}")

            # Exits folder - Always show, load from room data
            exits_folder = QTreeWidgetItem(["Exits"])
            exits_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(exits_folder)

            # Load actual exits from rooms.json
            try:
                import json
                rooms_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/world/rooms.json"
                )
                with open(rooms_path, 'r') as f:
                    rooms_data = json.load(f)

                # Get current room's exits
                if self.current_room in rooms_data:
                    room_data = rooms_data[self.current_room]
                    exits = room_data.get('exits', {})

                    for direction, dest_room_id in exits.items():
                        # Get destination room name
                        dest_name = rooms_data.get(dest_room_id, {}).get('name', dest_room_id)
                        exit_item = QTreeWidgetItem([f"{direction} → {dest_name}"])
                        exit_item.setData(0, Qt.ItemDataRole.UserRole, {
                            'type': 'exit',
                            'direction': direction,
                            'destination': dest_room_id
                        })
                        exits_folder.addChild(exit_item)

            except Exception as e:
                print(f"Error loading exits: {e}")

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
                # Handle both dict (normal) and tuple (from Assets panel drag)
                if isinstance(entity_data, tuple):
                    # Assets panel stores (asset_type, asset_name)
                    asset_type, asset_name = entity_data
                    # For now, don't emit - ensembles have their own handling
                    return
                elif isinstance(entity_data, dict):
                    entity_type = entity_data.get('type', 'unknown')
                    self.entitySelected.emit(entity_type, entity_data)

    def on_item_double_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle double-click - unpack ensembles, inspect entities."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data:
            if isinstance(entity_data, tuple):
                # Ensemble from Assets - unpack it!
                asset_type, asset_name = entity_data
                if asset_type == "ensemble":
                    self.unpack_ensemble(asset_name)
            elif isinstance(entity_data, dict):
                # Regular entity - inspect it
                self.inspect_entity(entity_data)

    def on_item_clicked(self, item: QTreeWidgetItem, column: int = 0):
        """Handle direct item clicks (used by context menu)."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data:
            # Handle both dict and tuple
            if isinstance(entity_data, dict):
                entity_type = entity_data.get('type', 'unknown')
                self.entitySelected.emit(entity_type, entity_data)

    def show_context_menu(self, position):
        """Show right-click context menu (Unity-style)."""
        item = self.tree.itemAt(position)

        menu = QMenu()

        if item:
            # Capture data immediately (item may be deleted after menu closes)
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)

            # Check if it's an ensemble (tuple from Assets)
            if isinstance(entity_data, tuple):
                asset_type, asset_name = entity_data
                if asset_type == "ensemble":
                    # Ensemble context menu
                    menu.addAction("Unpack Ensemble", lambda: self.unpack_ensemble(asset_name))
                    menu.addAction("View Ensemble Info", lambda: self.view_ensemble_info(asset_name))
                    menu.addSeparator()
                    menu.addAction("Remove from Hierarchy", lambda: self.remove_item_from_tree(item))
                    menu.exec(self.tree.viewport().mapToGlobal(position))
                    return

            entity_type = entity_data.get('type', '') if entity_data and isinstance(entity_data, dict) else None

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
                menu.addAction("De-Rez Noodling", lambda d=entity_data: self.delete_selected_items())

            elif entity_type == 'prim':
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addAction("Edit Description", lambda d=entity_data: self.edit_description_data(d))
                menu.addSeparator()
                menu.addAction("Duplicate Prim", lambda d=entity_data: self.duplicate_prim_data(d))
                menu.addAction("De-Rez Prim", lambda d=entity_data: self.delete_selected_items())

            elif entity_type == 'user':
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addAction("View Profile", lambda d=entity_data: self.view_user_profile_data(d))

            elif entity_type == 'exit':
                menu.addAction("Edit Exit", lambda d=entity_data: self.edit_exit_data(d))
                menu.addAction("De-Rez Exit", lambda d=entity_data: self.delete_prim_data(d))

            else:
                # Folder or other
                menu.addAction("Expand All", lambda: self.expand_recursive(item))
                menu.addAction("Collapse All", lambda: self.collapse_recursive(item))
        else:
            # Empty space - show rez options
            create_menu = menu.addMenu("Rez")
            create_menu.addAction("Empty Noodling", lambda: self.create_empty_noodling())
            create_menu.addAction("Empty Prim", lambda: self.create_empty_prim())
            create_menu.addAction("Empty Room", lambda: self.create_empty_room())

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
            # De-rez via direct file manipulation (simple and fast)
            try:
                print(f"Derezzing {prim_type}: {prim_id}")

                # Derez by directly modifying world data files
                import json
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
            try:
                import requests
                response = requests.post(
                    'http://localhost:8081/api/agents',
                    json={'name': name, 'species': 'unknown', 'pronouns': 'they/them'},
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"Created Noodling: {name}")
                    self.refresh_scene()
                else:
                    print(f"Error creating Noodling: {response.text}")
            except Exception as e:
                print(f"Error creating Noodling: {e}")

    def create_empty_prim(self):
        """Create an empty prim."""
        name, ok = QInputDialog.getText(
            self,
            "Create Empty Prim",
            "Prim name:",
            text="NewPrim"
        )
        if ok and name:
            try:
                import requests
                response = requests.post(
                    'http://localhost:8081/api/objects',
                    json={'name': name, 'location': self.current_room},  # Pass current room
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"Created Prim: {name}")
                    self.refresh_scene()
                else:
                    print(f"Error creating Prim: {response.text}")
            except Exception as e:
                print(f"Error creating Prim: {e}")

    def create_empty_room(self):
        """Create an empty room."""
        name, ok = QInputDialog.getText(
            self,
            "Create Empty Room",
            "Room name:",
            text="NewRoom"
        )
        if ok and name:
            try:
                import requests
                response = requests.post(
                    'http://localhost:8081/api/rooms',
                    json={'name': name},
                    timeout=5
                )
                if response.status_code == 200:
                    print(f"Created Room: {name}")
                    self.refresh_scene()
                else:
                    print(f"Error creating Room: {response.text}")
            except Exception as e:
                print(f"Error creating Room: {e}")

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

    def unpack_ensemble(self, ensemble_filename):
        """Unpack an ensemble - rez all members with shared context."""
        # Get parent window's assets panel to call its load function
        if hasattr(self.parent(), 'assets'):
            self.parent().assets._load_ensemble(ensemble_filename)
        else:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Cannot Unpack", "Assets panel not found.")

    def view_ensemble_info(self, ensemble_filename):
        """View ensemble information (mission, roles, dynamics)."""
        try:
            import json

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

            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Ensemble Info", info)

        except Exception as e:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Error", f"Failed to load ensemble info: {e}")

    def remove_item_from_tree(self, item):
        """Remove an item from the tree (doesn't derez, just removes from view)."""
        parent = item.parent()
        if parent:
            parent.removeChild(item)
        else:
            index = self.tree.indexOfTopLevelItem(item)
            if index >= 0:
                self.tree.takeTopLevelItem(index)

    def delete_selected_items(self):
        """De-rez all selected items (supports multi-selection)."""
        selected_items = self.tree.selectedItems()
        if not selected_items:
            return

        # Collect entities to derez
        entities_to_derez = []
        for item in selected_items:
            entity_data = item.data(0, Qt.ItemDataRole.UserRole)
            if entity_data and isinstance(entity_data, dict):
                entities_to_derez.append((item, entity_data))

        if not entities_to_derez:
            return

        # Show confirmation if enabled
        if self.derez_confirm:
            from PyQt6.QtWidgets import QMessageBox, QCheckBox
            msgBox = QMessageBox(self)
            msgBox.setWindowTitle("Derez")

            if len(entities_to_derez) == 1:
                prim_type = entities_to_derez[0][1].get('type')
                prim_id = entities_to_derez[0][1].get('id')
                msgBox.setText(f"Derez {prim_type} '{prim_id}'?\n\nThis will remove it from the scene.")
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

        # Derez all selected entities
        for item, entity_data in entities_to_derez:
            self._derez_entity(entity_data, item)

    def _derez_entity(self, entity_data, tree_item):
        """Derez a single entity (helper for batch operations)."""
        prim_id = entity_data.get('id')
        prim_type = entity_data.get('type')

        try:
            print(f"Derezzing {prim_type}: {prim_id}")

            # Derez by directly modifying world data files
            import json
            base_path = os.path.join(
                os.path.dirname(__file__),
                "../../../cmush/world"
            )
            base_path = os.path.abspath(base_path)

            if prim_type == 'noodling' or prim_type == 'agent':
                # Remove via API (removes from running server + agents.json)
                import requests
                try:
                    response = requests.delete(f"http://localhost:8081/api/agents/{prim_id}", timeout=2)
                    if response.status_code == 200:
                        print(f"Derezzed {prim_id} from server")
                    else:
                        print(f"API error: {response.text}")
                except Exception as e:
                    print(f"Failed to derez via API: {e}")
                    # Fallback: remove from file only
                    agents_path = os.path.join(base_path, "agents.json")
                    with open(agents_path, 'r') as f:
                        agents = json.load(f)
                    if prim_id in agents:
                        del agents[prim_id]
                        with open(agents_path, 'w') as f:
                            json.dump(agents, f, indent=2)
                        print(f"Derezzed {prim_id} from file (server may still have it)")

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

            # Remove from tree
            parent = tree_item.parent()
            if parent:
                parent.removeChild(tree_item)
            else:
                index = self.tree.indexOfTopLevelItem(tree_item)
                if index >= 0:
                    self.tree.takeTopLevelItem(index)

        except Exception as e:
            print(f"Error derezzing {prim_id}: {e}")

    def set_server_state(self, running: bool):
        """Update hierarchy based on server state - gray out if offline."""
        if running:
            # Server online - enable tree
            self.tree.setEnabled(True)
            self.tree.setStyleSheet(self.tree.styleSheet().replace("color: #666;", "color: #D2D2D2;"))
        else:
            # Server offline - disable and gray out tree
            self.tree.setEnabled(False)
            # Make text gray
            for i in range(self.tree.topLevelItemCount()):
                self._gray_out_item(self.tree.topLevelItem(i))

    def _gray_out_item(self, item):
        """Recursively gray out an item and its children."""
        from PyQt6.QtGui import QColor
        item.setForeground(0, QColor(100, 100, 100))  # Dark gray
        for i in range(item.childCount()):
            self._gray_out_item(item.child(i))

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
