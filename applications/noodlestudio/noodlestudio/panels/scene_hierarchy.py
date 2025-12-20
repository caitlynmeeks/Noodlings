"""
Scene Hierarchy Panel - Unity-style entity tree

Shows all prims in the noodleMUSH world:
- Rooms (with exits)
- Users (Noodlers)
- Noodlings
- Prims (WANTED POSTER, RADIO, etc.)

Click to select → Inspector shows editable properties

Author: Caitlyn + Claude
Date: November 17, 2025
"""

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget,
                             QTreeWidgetItem, QLabel, QPushButton, QMenu, QInputDialog, QComboBox,
                             QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QAction
import requests
import sys
import os

from ..core.undo_manager import UndoManager
from ..core.commands.scene_commands import (
    CreateNoodlingCommand, DeleteNoodlingCommand,
    CreatePropCommand, DeletePropCommand,
    CreateZoneCommand, DeleteZoneCommand
)


class SceneHierarchy(QWidget):
    """
    Unity-style Scene Hierarchy panel.

    Tree structure:
    ┬ Stage: The Nexus
    ├─┬ Zones
    │ └─ main (soft region)
    ├─┬ Users
    │ └─ caity [Noodler, 9yo, she/her]
    ├─┬ Noodlings (Instances)
    │ ├─ Red [noodling, they]
    │ └─ Servnak [robot, they]
    ├─┬ Props
    │ ├─ WANTED POSTER
    │ └─ RADIO
    └─┬ Exits
      ├─ north → The Forest Path
      └─ east → The Pond

    Supports both new project structure (Stages/xxx/Instances/) and legacy
    format (cmush/world/agents.json).
    """

    entitySelected = pyqtSignal(str, dict)  # (entity_type, entity_data)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.api_base = "http://localhost:8081/api"
        self.current_room = "room_000"  # Start at Nexus (legacy)
        self.current_stage = None  # New project format
        self.project_manager = None  # Set via set_project_manager()

        # Track expanded state (survives tree rebuild)
        self.expanded_items = set()

        # Track selected item (survives tree rebuild)
        self.selected_item_path = None

        # Derez confirmation settings
        self.derez_confirm = True  # Show confirmation dialog

        # Track agent pause states
        self.agent_pause_states = {}  # {agent_id: bool}

        # Initialize UI directly on this widget
        self.init_ui(self)

        # Auto-refresh
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_scene)
        self.refresh_timer.start(2000)

        # Initial load
        self.refresh_scene()

    def init_ui(self, widget):
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

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

        # Single-click on text to toggle expansion (consistent with Inspector CollapsibleSection)
        self.tree.itemClicked.connect(self.on_item_clicked_for_expansion)

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

    def set_project_manager(self, project_manager):
        """Set project manager reference for loading from project structure."""
        self.project_manager = project_manager
        # Refresh stage selector when project changes
        self.populate_stage_selector()

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
        import json
        import yaml

        # Block signals during population to avoid triggering on_stage_changed
        self.stage_selector.blockSignals(True)
        self.stage_selector.clear()

        try:
            # Check if project is open - use new format
            if self.project_manager and self.project_manager.is_project_open():
                stages = self.project_manager.list_stages()

                for stage_name in stages:
                    stage_path = self.project_manager.get_stage_path(stage_name)
                    if stage_path:
                        # Try to load stage.yaml for display name
                        display_name = stage_name
                        stage_yaml = os.path.join(stage_path, "stage.yaml")
                        if os.path.exists(stage_yaml):
                            try:
                                with open(stage_yaml, 'r') as f:
                                    stage_data = yaml.safe_load(f) or {}
                                    display_name = stage_data.get('name', stage_name)
                            except:
                                pass

                        display_text = f"{display_name} ({stage_name})"
                        self.stage_selector.addItem(display_text, stage_name)

                        # Select current stage
                        if stage_name == self.current_stage:
                            self.stage_selector.setCurrentText(display_text)

                # If no stage selected, select first one
                if not self.current_stage and stages:
                    self.current_stage = stages[0]
                    self.stage_selector.setCurrentIndex(0)

            else:
                # Legacy mode - load from rooms.json
                rooms_path = os.path.join(
                    os.path.dirname(__file__),
                    "../../../cmush/world/rooms.json"
                )
                if os.path.exists(rooms_path):
                    with open(rooms_path, 'r') as f:
                        rooms_data = json.load(f)

                    for room_id, room_data in rooms_data.items():
                        room_name = room_data.get('name', room_id)
                        display_text = f"{room_name} ({room_id})"
                        self.stage_selector.addItem(display_text, room_id)

                        # Select current room
                        if room_id == self.current_room:
                            self.stage_selector.setCurrentText(display_text)

        except Exception as e:
            print(f"Error populating stage selector: {e}")

        self.stage_selector.blockSignals(False)

    def on_stage_changed(self, text):
        """Handle stage selection change."""
        if not text:
            return

        # Get stage/room id from combo box data
        new_id = self.stage_selector.currentData()

        # Check if this is project mode or legacy mode
        if self.project_manager and self.project_manager.is_project_open():
            # Project mode - using stages
            if new_id == self.current_stage:
                return

            stage_name = text.split(' (')[0] if ' (' in text else text
            reply = QMessageBox.question(
                self,
                "Switch Stage?",
                f"Switch to stage '{stage_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.current_stage = new_id
                self.refresh_scene()
        else:
            # Legacy mode - using rooms
            if new_id == self.current_room:
                return

            room_name = text.split(' (')[0] if ' (' in text else text
            reply = QMessageBox.question(
                self,
                "Teleport to Stage?",
                f"Teleport your character to stage '{room_name}'?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.teleport_to_stage(new_id)

            self.current_room = new_id
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
        """Refresh scene hierarchy from project structure."""
        import yaml

        # Populate stage selector on first refresh
        if self.stage_selector.count() == 0:
            self.populate_stage_selector()

        try:
            # Save expanded state before clearing
            self.save_expanded_state()

            # CRITICAL: Block signals BEFORE clearing to prevent empty selection event
            self.tree.blockSignals(True)
            self.tree.clear()

            # Check if project is open
            if self.project_manager and self.project_manager.is_project_open():
                self._refresh_from_project()
            else:
                self._show_no_project_message()

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
        msg_item = QTreeWidgetItem(["No project open"])
        msg_item.setForeground(0, Qt.GlobalColor.gray)
        msg_item.setFlags(msg_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(msg_item)

        hint_item = QTreeWidgetItem(["File > Open Project..."])
        hint_item.setForeground(0, Qt.GlobalColor.darkGray)
        hint_item.setFlags(hint_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        self.tree.addTopLevelItem(hint_item)

    def _refresh_from_project(self):
        """Refresh scene from project structure (Stages/xxx/...)."""
        import yaml

        if not self.current_stage:
            stages = self.project_manager.list_stages()
            if stages:
                self.current_stage = stages[0]
            else:
                # No stages - show empty
                empty_item = QTreeWidgetItem(["No stages in project"])
                empty_item.setForeground(0, Qt.GlobalColor.gray)
                self.tree.addTopLevelItem(empty_item)
                return

        stage_path = self.project_manager.get_stage_path(self.current_stage)
        if not stage_path:
            return

        # Load stage.yaml
        stage_name = self.current_stage
        stage_data = {}
        stage_yaml = os.path.join(stage_path, "stage.yaml")
        if os.path.exists(stage_yaml):
            try:
                with open(stage_yaml, 'r') as f:
                    stage_data = yaml.safe_load(f) or {}
                    stage_name = stage_data.get('name', self.current_stage)
            except:
                pass

        # Root: Stage item
        stage_item = QTreeWidgetItem([f"Stage: {stage_name}"])
        stage_item.setFont(0, QFont("Arial", 12, QFont.Weight.Bold))
        stage_item.setForeground(0, Qt.GlobalColor.white)
        stage_item.setData(0, Qt.ItemDataRole.UserRole, {
            'type': 'stage',
            'id': self.current_stage,
            'path': stage_path,
            'data': stage_data
        })
        self.tree.addTopLevelItem(stage_item)
        stage_item.setExpanded(True)

        # Zones folder
        zones_folder = QTreeWidgetItem(["Zones"])
        zones_folder.setForeground(0, Qt.GlobalColor.gray)
        stage_item.addChild(zones_folder)

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
                        zone_item = QTreeWidgetItem([display_text])
                        zone_item.setData(0, Qt.ItemDataRole.UserRole, {
                            'type': 'zone',
                            'id': zone_id,
                            'path': zone_path,
                            'data': zone_data
                        })
                        zones_folder.addChild(zone_item)
                    except Exception as e:
                        print(f"Error loading zone {filename}: {e}")

        # Connected Users folder
        users_folder = QTreeWidgetItem(["Connected Users"])
        users_folder.setForeground(0, Qt.GlobalColor.gray)
        stage_item.addChild(users_folder)

        user_item = QTreeWidgetItem(["caity [Noodler, 9yo, she/her]"])
        user_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'user', 'id': 'user_caity'})
        users_folder.addChild(user_item)

        # Instances folder (Noodlings in stage)
        instances_folder = QTreeWidgetItem(["Noodlings"])
        instances_folder.setForeground(0, Qt.GlobalColor.gray)
        stage_item.addChild(instances_folder)

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

                    # Check pause state
                    agent_id = f"agent_{inst_name}"
                    is_paused = self.get_agent_pause_state(agent_id)

                    # Build status text
                    status_parts = []
                    if is_paused:
                        status_parts.append("paused")
                    status_text = f"[{', '.join(status_parts)}]" if status_parts else ""
                    pause_icon = "▶" if is_paused else "⏸"

                    display_text = f"{display_name:<20} {status_text:<20} {pause_icon}"

                    inst_item = QTreeWidgetItem([display_text])
                    inst_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'type': 'instance',
                        'id': agent_id,
                        'name': display_name,
                        'path': inst_path,
                        'noodling_ref': noodling_ref,
                        'zone': zone,
                        'data': inst_data
                    })
                    instances_folder.addChild(inst_item)

                except Exception as e:
                    print(f"Error loading instance {inst_name}: {e}")

        # Props folder
        props_folder = QTreeWidgetItem(["Props"])
        props_folder.setForeground(0, Qt.GlobalColor.gray)
        stage_item.addChild(props_folder)

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
                    display_text = f"{display_name:<20}                     {lock_icon}"

                    prop_item = QTreeWidgetItem([display_text])
                    prop_item.setData(0, Qt.ItemDataRole.UserRole, {
                        'type': 'prop',
                        'id': f"prop_{prop_name}",
                        'name': display_name,
                        'path': prop_path,
                        'prim_ref': prim_ref,
                        'zone': zone,
                        'locked': is_locked,
                        'data': prop_data
                    })
                    props_folder.addChild(prop_item)

                except Exception as e:
                    print(f"Error loading prop {prop_name}: {e}")

        # Zone connections (exits between zones)
        exits_folder = QTreeWidgetItem(["Zone Connections"])
        exits_folder.setForeground(0, Qt.GlobalColor.gray)
        stage_item.addChild(exits_folder)

        # Load from stage.yaml zone_graph
        zone_graph = stage_data.get('zone_graph', {})
        for from_zone, connections in zone_graph.items():
            for to_zone in connections:
                exit_item = QTreeWidgetItem([f"{from_zone} → {to_zone}"])
                exit_item.setData(0, Qt.ItemDataRole.UserRole, {
                    'type': 'zone_connection',
                    'from': from_zone,
                    'to': to_zone
                })
                exits_folder.addChild(exit_item)

    def _refresh_from_legacy(self):
        """Refresh scene from legacy format (cmush/world/...)."""
        import json

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

        room_item = QTreeWidgetItem([f"Stage: {stage_name}"])
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
        users_folder = QTreeWidgetItem(["Connected Users"])
        users_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(users_folder)

        user_item = QTreeWidgetItem(["caity [Noodler, 9yo, she/her]"])
        user_item.setData(0, Qt.ItemDataRole.UserRole, {'type': 'user', 'id': 'user_caity'})
        users_folder.addChild(user_item)

        # Noodlings folder
        noodlings_folder = QTreeWidgetItem(["Noodlings"])
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

            is_paused = self.get_agent_pause_state(agent_id)
            status_parts = []
            if is_paused:
                status_parts.append("paused")
            if is_locked:
                status_parts.append("locked")
            status_text = f"[{', '.join(status_parts)}]" if status_parts else ""

            pause_icon = "▶" if is_paused else "⏸"
            lock_icon = "[L]" if is_locked else ""

            display_text = f"{name:<20} {status_text:<20} {pause_icon} {lock_icon}"

            noodling_item = QTreeWidgetItem([display_text])
            noodling_item.setData(0, Qt.ItemDataRole.UserRole, {
                'type': 'noodling',
                'id': agent_id,
                'name': name,
                'locked': is_locked,
                'data': agent
            })
            noodlings_folder.addChild(noodling_item)

        # Prims folder
        prims_folder = QTreeWidgetItem(["Prims"])
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

                    status_parts = []
                    if is_disabled:
                        status_parts.append("disabled")
                    if is_locked:
                        status_parts.append("locked")
                    status_text = f"[{', '.join(status_parts)}]" if status_parts else ""

                    lock_icon = "[L]" if is_locked else ""
                    display_text = f"{prim_name:<20} {status_text:<20}    {lock_icon}"

                    prim_item = QTreeWidgetItem([display_text])
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
        exits_folder = QTreeWidgetItem(["Exits"])
        exits_folder.setForeground(0, Qt.GlobalColor.gray)
        room_item.addChild(exits_folder)

        try:
            if self.current_room in rooms_data:
                room_data = rooms_data[self.current_room]
                exits = room_data.get('exits', {})

                for direction, dest_room_id in exits.items():
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
            # Make sure signals are unblocked even on error
            self.tree.blockSignals(False)

    def on_selection_changed(self):
        """Handle entity selection (doesn't interfere with expand/collapse)."""
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

    def on_item_clicked_for_expansion(self, item: QTreeWidgetItem, column: int):
        """
        Handle item click to toggle expansion (consistent with Inspector CollapsibleSection).

        Clicking text on items with children will expand/collapse them.
        Leaf items (no children) are unaffected - only selection occurs.
        """
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

                # Check if cognition is paused for this agent
                agent_id = entity_data.get('id')
                is_paused = self.get_agent_pause_state(agent_id)
                pause_text = "Resume Cognition" if is_paused else "Pause Cognition"
                menu.addAction(pause_text, lambda d=entity_data: self.toggle_cognition_pause_data(d))

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
                menu.addAction("Export Prim", lambda d=entity_data: self.export_prim_data(d))
                menu.addSeparator()
                menu.addAction("Duplicate Prim", lambda d=entity_data: self.duplicate_prim_data(d))
                menu.addAction("De-Rez Prim", lambda d=entity_data: self.delete_selected_items())

            elif entity_type == 'prop':
                # Project-mode prop (same as prim but stored in project)
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addSeparator()
                menu.addAction("Duplicate Prop", lambda d=entity_data: self.duplicate_prop(d))
                menu.addAction("De-Rez Prop", lambda d=entity_data: self.delete_prop(d))

            elif entity_type == 'instance':
                # Project-mode noodling instance
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addSeparator()
                menu.addAction("Duplicate Instance", lambda d=entity_data: self.duplicate_instance(d))
                menu.addAction("De-Rez Instance", lambda d=entity_data: self.delete_instance(d))

            elif entity_type == 'zone':
                # Project-mode zone
                menu.addAction("Inspect Properties", lambda d=entity_data: self.inspect_entity(d))
                menu.addSeparator()
                menu.addAction("Delete Zone", lambda d=entity_data: self.delete_zone(d))

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
            # Empty space - show rez options only if project is open
            if self.project_manager and self.project_manager.is_project_open():
                create_menu = menu.addMenu("Rez")
                create_menu.addAction("New Noodling", lambda: self.create_empty_noodling())
                create_menu.addAction("New Prop", lambda: self.create_empty_prim())
                create_menu.addAction("New Zone", lambda: self.create_empty_zone())

                menu.addSeparator()
                menu.addAction("Import Prop...", lambda: self.import_prim())
            else:
                menu.addAction("Open Project...", lambda: self._prompt_open_project())

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _prompt_open_project(self):
        """Prompt user to open a project."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(
            self,
            "No Project Open",
            "Please open a project first.\n\nFile > Open Project..."
        )

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

    def export_prim_data(self, entity_data):
        """Export Prim to .prim file (USD-augmented format)."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import sys
        import os

        # Add cmush to path for imports
        cmush_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../cmush"))
        if cmush_path not in sys.path:
            sys.path.insert(0, cmush_path)

        prim_id = entity_data.get('id')
        prim_data = entity_data.get('data', {})

        # Get prim name for default filename
        prim_name = prim_data.get('name', prim_id).replace(' ', '_')

        # Open file save dialog
        default_path = str(Path.home() / f"{prim_name}.prim")
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Prim (USD Format)",
            default_path,
            "Prim Files (*.prim);;USD Files (*.usd *.usda);;All Files (*)"
        )

        if file_path:
            try:
                # Import export functionality
                from world import World
                from prim_import_export import PrimExporter

                # Load world
                world_path = os.path.join(cmush_path, "world")
                world = World(world_path)

                # Export prim
                exporter = PrimExporter(world)
                success = exporter.export_prim(prim_id, file_path)

                if success:
                    print(f"✅ Exported {prim_name} to {file_path}")
                    QMessageBox.information(
                        self,
                        "Export Successful",
                        f"Prim '{prim_name}' exported to:\n{file_path}\n\nUSD-augmented format with full physics and permissions."
                    )
                else:
                    print(f"❌ Failed to export {prim_name}")
                    QMessageBox.warning(
                        self,
                        "Export Failed",
                        f"Could not export prim '{prim_name}'.\nCheck console for details."
                    )

            except Exception as e:
                print(f"Error exporting prim: {e}")
                import traceback
                traceback.print_exc()
                QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting prim:\n{str(e)}"
                )

    def import_prim(self):
        """Import Prim from .prim file."""
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        from pathlib import Path
        import sys
        import os

        # Add cmush to path for imports
        cmush_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../cmush"))
        if cmush_path not in sys.path:
            sys.path.insert(0, cmush_path)

        # Open file open dialog
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Prim (USD Format)",
            str(Path.home()),
            "Prim Files (*.prim);;USD Files (*.usd *.usda);;All Files (*)"
        )

        if file_path:
            try:
                # Import import functionality
                from world import World
                from prim_import_export import PrimImporter

                # Load world
                world_path = os.path.join(cmush_path, "world")
                world = World(world_path)

                # Get current room (default to room_000)
                current_room = "room_000"  # TODO: Get from current selection

                # Import prim
                importer = PrimImporter(world)
                imported_id = importer.import_prim(
                    file_path,
                    room_id=current_room,
                    importer_user="studio_user"
                )

                if imported_id:
                    print(f"✅ Imported prim as {imported_id}")

                    # Refresh hierarchy (catch errors separately)
                    try:
                        self.refresh_hierarchy()
                    except Exception as refresh_error:
                        print(f"Warning: Hierarchy refresh failed: {refresh_error}")
                        # Don't show error - import succeeded

                    QMessageBox.information(
                        self,
                        "Import Successful",
                        f"Prim imported successfully as:\n{imported_id}\n\nSpawned in {current_room}"
                    )
                else:
                    print(f"❌ Failed to import prim from {file_path}")
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        f"Could not import prim from:\n{file_path}\n\nCheck console for details."
                    )

            except Exception as e:
                print(f"Error importing prim: {e}")
                import traceback
                traceback.print_exc()
                # Only show error if import actually failed
                # Check if prim was created despite error
                from world import World
                import os
                cmush_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../cmush"))
                world_path = os.path.join(cmush_path, "world")
                try:
                    world = World(world_path)
                    # Check if any object was just created
                    recent_objects = sorted(world.objects.items(), key=lambda x: x[0], reverse=True)[:3]
                    if recent_objects:
                        print(f"Recent objects: {[obj[1].get('name') for obj in recent_objects]}")
                        # Import may have succeeded - don't show error
                        print("Import may have succeeded despite error - check hierarchy")
                        return
                except:
                    pass

                # Show error only if import definitely failed
                QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Error importing prim:\n{str(e)}"
                )

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
        """Create a new Noodling instance in the current stage (no dialog)."""
        import uuid

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
            'noodling': '',  # Reference to Noodlings/xxx (empty = blank template)
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
        UndoManager.instance().push(cmd)
        print(f"Created Noodling: {display_name} ({instance_id[:8]}...)")

    def create_empty_prim(self):
        """Create a new Prop in the current stage (no dialog)."""
        import uuid

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

        # Generate unique name: "New Prop", "New Prop (2)", etc.
        # Need to create dir first so we can check existing names
        os.makedirs(props_dir, exist_ok=True)
        display_name = self._generate_unique_name(props_dir, "New Prop", "prop.yaml")

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
        UndoManager.instance().push(cmd)
        print(f"Created Prop: {display_name} ({prop_id[:8]}...)")

    def create_empty_zone(self):
        """Create a new Zone in the current stage (no dialog)."""
        import uuid

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
        UndoManager.instance().push(cmd)
        print(f"Created Zone: {display_name} ({zone_id[:8]}...)")

    def _generate_unique_name(self, directory: str, base_name: str, yaml_file: str) -> str:
        """Generate unique name like 'New Prop', 'New Prop (2)', etc."""
        import os
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
        import os
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
            UndoManager.instance().push(cmd)
            print(f"Deleted prop: {prop_name}")

    def duplicate_prop(self, entity_data: dict):
        """Duplicate a prop in the project."""
        import yaml
        import uuid
        import shutil

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
            UndoManager.instance().push(cmd)
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
            UndoManager.instance().push(cmd)
            print(f"Deleted zone: {zone_name}")

    # =========================================================================
    # Auto-select newly created items
    # =========================================================================

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

    def _REMOVED_create_prim_legacy(self, name: str):
        """REMOVED: Legacy prim creation via API. Projects are now required."""
        pass

    def _REMOVED_create_prim_legacy_actual(self, name: str):
        """Create a prim via the legacy API (objects.json). REMOVED - keeping for reference."""
        try:
            import requests
            response = requests.post(
                'http://localhost:8081/api/objects',
                json={'name': name, 'location': self.current_room},
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

            # Check if this item was selected before removing
            was_selected = tree_item.isSelected()

            # Remove from tree
            parent = tree_item.parent()
            if parent:
                parent.removeChild(tree_item)
            else:
                index = self.tree.indexOfTopLevelItem(tree_item)
                if index >= 0:
                    self.tree.takeTopLevelItem(index)

            # If the derezzed entity was selected, clear Inspector and Facets Editor
            if was_selected:
                print(f"Derezzed entity was selected, clearing Inspector/Facets Editor")
                self.entitySelected.emit("", {})

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
                                facets_editor.pause_button.setText("▶ Resume Cognition")
                                facets_editor.bottom_pause_btn.setText("▶")
                            else:
                                facets_editor.pause_button.setText("⏸ Pause Cognition")
                                facets_editor.bottom_pause_btn.setText("⏸")

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
