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
        super().__init__("Scene Hierarchy", parent)
        self.api_base = "http://localhost:8081/api"
        self.current_room = "room_000"  # Start at Nexus

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
        header = QLabel("SCENE HIERARCHY")
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

    def refresh_scene(self):
        """Refresh scene hierarchy from noodleMUSH world state."""
        try:
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
            room_item.setExpanded(True)

            # Users folder
            users_folder = QTreeWidgetItem(["Users"])
            users_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(users_folder)
            users_folder.setExpanded(True)

            # Add current user
            user_item = QTreeWidgetItem(["caity [Noodler, 9yo, she/her]"])
            user_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'user', 'id': 'user_caity'})
            users_folder.addChild(user_item)

            # Noodlings folder
            noodlings_folder = QTreeWidgetItem(["Noodlings"])
            noodlings_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(noodlings_folder)
            noodlings_folder.setExpanded(True)

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
        """Delete a prim (uses data)."""
        prim_id = entity_data.get('id')
        prim_type = entity_data.get('type')

        from PyQt6.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self,
            "Delete Prim",
            f"Delete {prim_type} '{prim_id}'?\n\nThis cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            print(f"Delete {prim_type}: {prim_id}")
            # TODO: Send to noodleMUSH API
            self.refresh_scene()

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
