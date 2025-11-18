"""
Scene Hierarchy Panel - Unity-style entity tree

Shows all entities in the noodleMUSH world:
- Rooms (with exits)
- Users (Noodlers)
- Noodlings (consciousness agents)
- Objects (WANTED POSTER, RADIO, etc.)

Click to select → Inspector shows editable properties

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QLabel, QPushButton)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon
import requests


class SceneHierarchy(QDockWidget):
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
        self.tree.itemClicked.connect(self.on_item_clicked)
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

            # Objects folder
            objects_folder = QTreeWidgetItem(["Objects"])
            objects_folder.setForeground(0, Qt.GlobalColor.gray)
            room_item.addChild(objects_folder)

            # TODO: Load actual objects from world state
            poster_item = QTreeWidgetItem(["WANTED POSTER"])
            poster_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'object',
                'id': 'obj_wanted_poster'
            })
            objects_folder.addChild(poster_item)

            radio_item = QTreeWidgetItem(["RADIO"])
            radio_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'object',
                'id': 'obj_radio'
            })
            objects_folder.addChild(radio_item)

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

    def on_item_clicked(self, item: QTreeWidgetItem, column: int):
        """Handle entity selection."""
        entity_data = item.data(0, Qt.ItemDataRole.UserRole)
        if entity_data:
            entity_type = entity_data.get('type', 'unknown')
            self.entitySelected.emit(entity_type, entity_data)
