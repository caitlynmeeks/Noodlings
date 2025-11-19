"""
Main Window for NoodleSTUDIO.

The primary application window with menu bar, toolbar, dock area, and status bar.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction

from ..panels.home_panel import HomePanel
from ..panels.chat_panel import ChatPanel
from ..panels.profiler_panel import ProfilerPanel
from ..panels.scene_hierarchy import SceneHierarchy
from ..panels.inspector_panel import InspectorPanel
from ..panels.console_panel import ConsolePanel
from .theme import DARK_THEME
from .unity_theme import UNITY_DARK_THEME
from .layout_manager import LayoutManager
from PyQt6.QtWidgets import QDialog


class MainWindow(QMainWindow):
    """
    Main application window for NoodleSTUDIO.

    Contains:
    - Menu bar (File, View, Agent, Session, Tools, Help)
    - Tool bar (quick actions)
    - Dockable panel area (Home, Chat, etc.)
    - Status bar
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("NoodleSTUDIO - Noodlings IDE")
        self.resize(1400, 900)

        # Apply Unity dark theme
        self.setStyleSheet(UNITY_DARK_THEME)

        # Layout manager for saving configurations
        self.layout_manager = LayoutManager()

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()
        self._setup_panels()
        self._setup_shortcuts()

        # Load last used layout (like Unity reopening last scene)
        QTimer.singleShot(200, self.load_last_used_layout)

    def _setup_ui(self):
        """Build UI components."""
        # Clean central widget (no clutter, Unity-style)
        central = QWidget()
        central.setStyleSheet("background-color: #383838;")
        self.setCentralWidget(central)

    def _setup_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")
        file_menu.addAction(self._create_action("&New Stage", "Ctrl+N"))
        file_menu.addAction(self._create_action("&Open Stage...", "Ctrl+O"))
        file_menu.addAction(self._create_action("&Save Stage", "Ctrl+S"))

        # Character section
        file_menu.addSeparator()
        file_menu.addSection("Character")
        file_menu.addAction(self._create_action("Import Noodling (.nood)...", slot=self.import_noodling_file))
        file_menu.addAction(self._create_action("Export Noodling(s)...", slot=self.export_noodlings_dialog))

        # Ensemble section
        file_menu.addSeparator()
        file_menu.addSection("Ensemble")
        file_menu.addAction(self._create_action("Import Ensemble (.ensemble)...", slot=self.import_ensemble))

        # USD export/import
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Export Stage to USD (.usda)...", slot=self.export_stage_to_usd))
        file_menu.addAction(self._create_action("Export Timeline to USD (.usda)...", slot=self.export_timeline_to_usd))
        file_menu.addAction(self._create_action("Import USD Layer (.usda)...", slot=self.import_usd_layer))

        file_menu.addSeparator()
        file_menu.addAction(self._create_action("&Quit", "Ctrl+Q", self.close))

        # ===== CREATE MENU (like Unity's GameObject) =====
        create_menu = menu_bar.addMenu("&Create")

        # Noodling submenu
        noodling_menu = create_menu.addMenu("Noodling")
        noodling_menu.addAction(self._create_action("Empty Noodling", slot=self.create_empty_noodling))
        noodling_menu.addSeparator()
        noodling_menu.addAction(self._create_action("Kitten Noodling", slot=lambda: self.create_specialized_noodling("kitten")))
        noodling_menu.addAction(self._create_action("Robot Noodling", slot=lambda: self.create_specialized_noodling("robot")))
        noodling_menu.addAction(self._create_action("Dragon Noodling", slot=lambda: self.create_specialized_noodling("dragon")))
        noodling_menu.addSeparator()
        noodling_menu.addAction(self._create_action("Empty Ensemble", slot=self.create_empty_ensemble))
        noodling_menu.addAction(self._create_action("Import Ensemble (.ens)...", slot=self.import_ensemble))

        # Object submenu
        object_menu = create_menu.addMenu("Object")
        object_menu.addAction(self._create_action("Empty Object", slot=self.create_empty_object))
        object_menu.addSeparator()
        object_menu.addAction(self._create_action("Prop (Holdable)", slot=lambda: self.create_specialized_object("prop")))
        object_menu.addAction(self._create_action("Furniture (Sittable)", slot=lambda: self.create_specialized_object("furniture")))
        object_menu.addAction(self._create_action("Container (Openable)", slot=lambda: self.create_specialized_object("container")))

        create_menu.addSeparator()
        create_menu.addAction(self._create_action("Empty Room", slot=self.create_empty_room))
        create_menu.addAction(self._create_action("Empty Prim", slot=self.create_empty_prim))

        # ===== VIEW MENU =====
        view_menu = menu_bar.addMenu("&View")

        # Panel toggles
        view_menu.addAction(self._create_action("Scene Hierarchy", "Ctrl+1", checkable=True, checked=True))
        view_menu.addAction(self._create_action("World View", "Ctrl+2", checkable=True, checked=True))
        view_menu.addAction(self._create_action("Inspector", "Ctrl+3", checkable=True, checked=True))
        view_menu.addAction(self._create_action("Timeline Profiler", "Ctrl+4", checkable=True))

        view_menu.addSeparator()

        # Layout presets
        layout_submenu = view_menu.addMenu("Layouts")
        layout_submenu.addAction(self._create_action("Save Current Layout...", slot=self.save_current_layout))
        layout_submenu.addAction(self._create_action("Set Current as Default", slot=self.set_current_as_default))
        layout_submenu.addAction(self._create_action("Load Layout...", slot=self.load_layout_dialog))
        layout_submenu.addSeparator()
        layout_submenu.addAction(self._create_action("Reset to Default", slot=lambda: self.load_layout("Default")))

        # ===== ENTITIES MENU (like Unity's GameObject) =====
        entities_menu = menu_bar.addMenu("&Entities")
        entities_menu.addAction(self._create_action("Add Noodling...", "Ctrl+Shift+N", slot=self.add_noodling))
        entities_menu.addAction(self._create_action("Add Object...", "Ctrl+Shift+O", slot=self.add_object))
        entities_menu.addAction(self._create_action("Add Room...", slot=self.add_room))
        entities_menu.addSeparator()
        entities_menu.addAction(self._create_action("Remove Selected", "Delete"))
        entities_menu.addSeparator()
        entities_menu.addAction(self._create_action("Toggle Enlightenment", "Ctrl+E"))
        entities_menu.addAction(self._create_action("Reset All States"))

        # ===== COMPONENT MENU (stolen from Unity!) =====
        component_menu = menu_bar.addMenu("&Component")

        # Kindling components (inner light!)
        kindling_menu = component_menu.addMenu("Kindling")
        kindling_menu.addAction(self._create_action("Noodle", slot=lambda: self.add_component("noodle")))
        kindling_menu.addAction(self._create_action("Memory Bank", slot=lambda: self.add_component("memory")))
        kindling_menu.addAction(self._create_action("Relationship Graph", slot=lambda: self.add_component("relationships")))

        # Art & Reference components
        art_menu = component_menu.addMenu("Art & Reference")
        art_menu.addAction(self._create_action("Artbook", slot=lambda: self.add_component("artbook")))
        art_menu.addAction(self._create_action("Mood Board", slot=lambda: self.add_component("moodboard")))
        art_menu.addAction(self._create_action("Voice Reference", slot=lambda: self.add_component("voiceref")))

        # Behavior components
        behavior_menu = component_menu.addMenu("Behavior")
        behavior_menu.addAction(self._create_action("Dialogue Tree", slot=lambda: self.add_component("dialogue")))
        behavior_menu.addAction(self._create_action("Quest Giver", slot=lambda: self.add_component("quests")))
        behavior_menu.addAction(self._create_action("Vendor", slot=lambda: self.add_component("vendor")))

        # Custom component
        component_menu.addSeparator()
        component_menu.addAction(self._create_action("Add Script...", slot=lambda: self.add_component("custom")))

        # ===== WINDOW MENU =====
        window_menu = menu_bar.addMenu("&Window")
        window_menu.addAction(self._create_action("Minimize", "Ctrl+M", self.showMinimized))
        window_menu.addAction(self._create_action("Zoom", slot=self.showMaximized))
        window_menu.addSeparator()
        window_menu.addAction(self._create_action("Ensemble Store...", slot=self.show_ensemble_store))
        window_menu.addSeparator()
        window_menu.addAction(self._create_action("Reset to Default Layout", slot=lambda: self.load_layout("Default")))

        # ===== HELP MENU =====
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._create_action("NoodleStudio Documentation", "F1"))
        help_menu.addAction(self._create_action("Noodlings Architecture Guide"))
        help_menu.addAction(self._create_action("Report Issue..."))
        help_menu.addSeparator()
        help_menu.addAction(self._create_action("About NoodleStudio", slot=self.show_about))

    def _setup_tool_bar(self):
        """Create tool bar."""
        tool_bar = self.addToolBar("Main Toolbar")
        tool_bar.setObjectName("MainToolbar")  # Required for saveState

        # Hide legacy buttons for now
        tool_bar.setVisible(False)

    def _setup_status_bar(self):
        """Create status bar with server toggle."""
        from PyQt6.QtWidgets import QLabel, QWidget, QHBoxLayout
        from ..widgets.toggle_switch import ToggleSwitch

        status_bar = self.statusBar()

        # Server status section (more prominent!)
        server_container = QWidget()
        server_layout = QHBoxLayout()
        server_layout.setContentsMargins(10, 0, 10, 0)
        server_layout.setSpacing(10)

        # Server icon
        server_icon = QLabel("🔌")
        server_icon.setStyleSheet("font-size: 16px;")
        server_layout.addWidget(server_icon)

        # Server status label
        self.server_status_label = QLabel("noodleMUSH Server:")
        self.server_status_label.setStyleSheet("color: #D2D2D2; font-weight: bold; font-size: 13px;")
        server_layout.addWidget(self.server_status_label)

        # Toggle switch
        self.server_toggle = ToggleSwitch()
        self.server_toggle.setChecked(self.is_server_running())
        self.server_toggle.toggled.connect(self.on_server_toggled)
        server_layout.addWidget(self.server_toggle)

        server_container.setLayout(server_layout)
        server_container.setStyleSheet("background: #3a3a3a; border-radius: 4px; padding: 4px;")
        status_bar.addPermanentWidget(server_container)

        # Connection status
        self.connection_label = QLabel()
        self.update_connection_status()
        status_bar.addWidget(self.connection_label)

    def is_server_running(self) -> bool:
        """Check if noodleMUSH server is running."""
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'python.*server.py'], capture_output=True)
        return result.returncode == 0

    def on_server_toggled(self, enabled: bool):
        """Handle server toggle switch."""
        import subprocess

        if enabled:
            # Start server
            subprocess.Popen(
                ['../cmush/start.sh'],
                cwd='../cmush',
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            self.connection_label.setText("Starting server...")
        else:
            # Stop server
            subprocess.run(['pkill', '-f', 'python.*server.py'])
            self.connection_label.setText("Server stopped")

        # Update status after a delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self.update_connection_status)

    def update_connection_status(self):
        """Update connection status label."""
        if self.is_server_running():
            self.connection_label.setText("Server running on :8765")
            self.connection_label.setStyleSheet("color: #76AF6A;")  # Green
            self.server_toggle.setChecked(True)
        else:
            self.connection_label.setText("Server offline")
            self.connection_label.setStyleSheet("color: #999;")  # Gray
            self.server_toggle.setChecked(False)

    def _setup_panels(self):
        """Create Unity-style layout with EXACT positioning."""

        # Disable automatic corner docking (gives us more control)
        self.setCorner(Qt.Corner.TopLeftCorner, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.setCorner(Qt.Corner.TopRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)
        self.setCorner(Qt.Corner.BottomLeftCorner, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.setCorner(Qt.Corner.BottomRightCorner, Qt.DockWidgetArea.RightDockWidgetArea)

        # LEFT COLUMN (narrow): Scene Hierarchy
        self.hierarchy = SceneHierarchy(self)
        self.hierarchy.setObjectName("SceneHierarchy")  # Required for saveState
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.hierarchy)

        # CENTER-LEFT (LARGE - 75% width): World View (noodleMUSH chat)
        self.world_view = ChatPanel(self)
        self.world_view.setWindowTitle("World View")
        self.world_view.setObjectName("WorldView")  # Required for saveState
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.world_view)

        # RIGHT COLUMN (narrow): Inspector
        self.inspector = InspectorPanel(self)
        self.inspector.setObjectName("Inspector")  # Required for saveState
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector)

        # BOTTOM (full width): Console
        self.console = ConsolePanel(self)
        self.console.setObjectName("Console")  # Required for saveState
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.console)

        # Timeline Profiler (tabbed with Console, hidden by default)
        self.profiler_panel = ProfilerPanel(self)
        self.profiler_panel.setWindowTitle("Timeline Profiler")
        self.profiler_panel.setObjectName("TimelineProfiler")  # Required for saveState
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.profiler_panel)
        self.profiler_panel.hide()
        self.tabifyDockWidget(self.console, self.profiler_panel)

        # Connect hierarchy selection to inspector
        self.hierarchy.entitySelected.connect(self.inspector.load_entity)

        # Force Console tab to be active
        self.console.raise_()

        # Set exact sizes after a brief delay (let Qt settle)
        QTimer.singleShot(100, self.apply_default_sizes)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt6.QtGui import QShortcut, QKeySequence

        # Cmd/Ctrl+R - Reload with autologin
        reload_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        reload_shortcut.activated.connect(self.reload_world_view)

        # Cmd/Ctrl+Shift+R - Reload to login screen
        reload_login_shortcut = QShortcut(QKeySequence("Ctrl+Shift+R"), self)
        reload_login_shortcut.activated.connect(self.reload_world_view_clean)

    def reload_world_view(self):
        """Reload World View with autologin (Ctrl+R)."""
        if hasattr(self.world_view, 'reload'):
            self.world_view.reload()
            self.statusBar().showMessage("Reloaded (autologin)", 2000)

    def reload_world_view_clean(self):
        """Reload World View to login screen (Ctrl+Shift+R)."""
        if hasattr(self.world_view, 'web_view'):
            # Clear cookies to force login
            from PyQt6.QtWebEngineCore import QWebEngineProfile
            from PyQt6.QtCore import QUrl
            profile = QWebEngineProfile.defaultProfile()
            profile.cookieStore().deleteAllCookies()

            self.world_view.web_view.setUrl(QUrl("http://localhost:8080"))
            self.statusBar().showMessage("Reloaded (login screen)", 2000)

    def apply_default_sizes(self):
        """Apply exact panel sizes for default layout."""
        # Get window size
        width = self.width()
        height = self.height()

        # LEFT column: 200px for hierarchy
        # CENTER: Rest of width minus inspector (inspector = 300px)
        # So: hierarchy=200, world_view=(width-200-300), inspector=300

        left_width = 200
        right_width = 300
        bottom_height = 200

        # Set horizontal splits
        self.resizeDocks([self.hierarchy], [left_width], Qt.Orientation.Horizontal)
        self.resizeDocks([self.inspector], [right_width], Qt.Orientation.Horizontal)

        # Set bottom panel height
        self.resizeDocks([self.console], [bottom_height], Qt.Orientation.Vertical)

    def _create_action(
        self,
        text: str,
        shortcut: str = "",
        slot=None,
        checkable: bool = False,
        checked: bool = False
    ) -> QAction:
        """
        Create a QAction with text, shortcut, and optional slot.

        Args:
            text: Action text
            shortcut: Keyboard shortcut (e.g., "Ctrl+N")
            slot: Slot to connect to (optional)
            checkable: Whether action is checkable
            checked: Initial checked state

        Returns:
            QAction instance
        """
        action = QAction(text, self)
        if shortcut:
            action.setShortcut(shortcut)
        if slot:
            action.triggered.connect(slot)
        if checkable:
            action.setCheckable(True)
            action.setChecked(checked)
        return action

    def save_current_layout(self):
        """Save current panel layout."""
        from PyQt6.QtWidgets import QInputDialog
        layout_name, ok = QInputDialog.getText(
            self,
            "Save Layout",
            "Layout name:",
            text="My Layout"
        )
        if ok and layout_name:
            self.layout_manager.save_layout(self, layout_name)
            self.layout_manager.set_last_used_layout(layout_name)
            self.statusBar().showMessage(f"Layout '{layout_name}' saved", 3000)

    def set_current_as_default(self):
        """Save current layout as Default (loaded on startup)."""
        self.layout_manager.save_layout(self, "Default")
        self.layout_manager.set_last_used_layout("Default")
        self.statusBar().showMessage("Current layout saved as default", 3000)

    def load_layout_dialog(self):
        """Show dialog to select and load a saved layout."""
        from PyQt6.QtWidgets import QInputDialog

        layouts = self.layout_manager.list_layouts()

        if not layouts:
            QMessageBox.information(self, "No Layouts", "No saved layouts found.\nSave one first with 'Save Current Layout...'")
            return

        layout_name, ok = QInputDialog.getItem(
            self,
            "Load Layout",
            "Select layout to load:",
            layouts,
            0,
            False
        )

        if ok and layout_name:
            self.load_layout(layout_name)

    def load_layout(self, layout_name: str):
        """Load saved layout."""
        try:
            if self.layout_manager.load_layout(self, layout_name):
                self.statusBar().showMessage(f"Layout '{layout_name}' loaded", 3000)
            else:
                QMessageBox.warning(
                    self,
                    "Layout Not Found",
                    f"Layout '{layout_name}' not found.\n\nSave a layout first with:\nView → Layouts → Set Current as Default"
                )
        except Exception as e:
            print(f"Error loading layout: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Layout Error",
                f"Failed to load layout '{layout_name}'.\n\nError: {str(e)}\n\nCheck Console for details."
            )

    def load_last_used_layout(self):
        """Load the last used layout on startup (like Unity reopening last scene)."""
        last_layout = self.layout_manager.get_last_used_layout()

        if last_layout:
            print(f"Restoring last used layout: '{last_layout}'")
            success = self.layout_manager.load_layout(self, last_layout)
            if success:
                self.statusBar().showMessage(f"Restored layout: '{last_layout}'", 3000)
            else:
                print(f"Failed to restore last layout, using default panel arrangement")
        else:
            print("No last layout saved, using default panel arrangement")

    def export_stage_to_usd(self):
        """Export current stage to USD format (creates a layer file)."""
        from PyQt6.QtWidgets import QFileDialog
        from ..data.usd_exporter import USDExporter
        from pathlib import Path

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Stage to USD",
            "noodlemush_stage.usda",
            "USD ASCII Layer (*.usda)"
        )

        if filename:
            # Fetch current world state from API
            import requests
            try:
                resp = requests.get("http://localhost:8081/api/agents")
                agents = resp.json().get('agents', [])

                world_data = {
                    'rooms': {},  # TODO: Get from API
                    'noodlings': agents,
                    'users': [{'id': 'user_caity', 'username': 'caity', 'description': 'A nine-year-old Noodler'}],
                    'objects': {}
                }

                exporter = USDExporter()
                exporter.export_stage(world_data, Path(filename))

                self.statusBar().showMessage(f"Stage exported to {filename}", 5000)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Stage exported to USD layer:\n{filename}\n\n"
                    f"Contains Noodling prims with kindling properties.\n"
                    f"Import into Maya/Houdini/Blender to view."
                )

            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def export_timeline_to_usd(self):
        """Export timeline/profiler data as animated USD."""
        from PyQt6.QtWidgets import QFileDialog
        from ..data.usd_exporter import USDExporter
        from pathlib import Path
        import requests

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export Timeline to USD",
            "noodlemush_timeline.usda",
            "USD ASCII Layer (*.usda)"
        )

        if filename:
            try:
                resp = requests.get("http://localhost:8081/api/profiler/live-session")
                session_data = resp.json()

                exporter = USDExporter()
                exporter.export_timeline(session_data, Path(filename))

                self.statusBar().showMessage(f"Timeline exported to {filename}", 5000)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Timeline exported to USD layer with time-sampled affect data:\n{filename}\n\n"
                    f"Import into Maya/Houdini/Blender to visualize Noodling emotions over time!"
                )

            except Exception as e:
                QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def import_usd_layer(self):
        """Import USD layer file into noodleMUSH."""
        from PyQt6.QtWidgets import QFileDialog

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import USD Layer",
            "",
            "USD Files (*.usda *.usdc);;All Files (*)"
        )

        if filename:
            try:
                # Parse USD file and extract Noodling prims
                from ..data.usd_importer import USDImporter

                importer = USDImporter()
                imported_data = importer.import_layer(Path(filename))

                # TODO: Send to noodleMUSH API to spawn entities
                # For now, just show what we found
                noodlings_count = len(imported_data.get('noodlings', []))
                rooms_count = len(imported_data.get('rooms', []))
                objects_count = len(imported_data.get('objects', []))

                QMessageBox.information(
                    self,
                    "Import Complete",
                    f"USD layer imported:\n{filename}\n\n"
                    f"Found:\n"
                    f"- {noodlings_count} Noodling prims\n"
                    f"- {rooms_count} Room prims\n"
                    f"- {objects_count} Object prims\n\n"
                    f"(Rezzing not yet implemented)"
                )

            except Exception as e:
                QMessageBox.critical(self, "Import Failed", f"Error: {e}\n\nUSD import requires USD Python library.")

    def add_noodling(self):
        """Add a new Noodling to the stage."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Add Noodling",
            "Noodling name:",
            text="NewNoodling"
        )

        if ok and name:
            # TODO: Send to noodleMUSH API to rez
            QMessageBox.information(
                self,
                "Rez Noodling",
                f"Rezzing Noodling prim: {name}\n\n(API integration not yet implemented)"
            )

    def add_object(self):
        """Add a new object to the stage."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Add Object",
            "Object name:",
            text="NewObject"
        )

        if ok and name:
            # TODO: Send to noodleMUSH API to create object
            QMessageBox.information(
                self,
                "Add Object",
                f"Adding object prim: {name}\n\n(API integration not yet implemented)"
            )

    def add_room(self):
        """Add a new room to the stage."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Add Room",
            "Room name:",
            text="NewRoom"
        )

        if ok and name:
            # TODO: Send to noodleMUSH API to create room
            QMessageBox.information(
                self,
                "Add Room",
                f"Adding room prim: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_noodling(self):
        """Create an empty Noodling with default settings (via Create menu)."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Create Empty Noodling",
            "Noodling name:",
            text="NewNoodling"
        )

        if ok and name:
            # Default settings for empty Noodling
            default_settings = {
                'name': name,
                'species': 'noodling',
                'personality': {
                    'extraversion': 0.5,
                    'curiosity': 0.5,
                    'impulsivity': 0.5,
                    'emotional_volatility': 0.5
                },
                'llm_provider': 'local',
                'llm_model': 'qwen/qwen3-4b-2507'
            }

            QMessageBox.information(
                self,
                "Create Noodling",
                f"Creating empty Noodling: {name}\n\n"
                f"Default personality: balanced (0.5)\n"
                f"Species: noodling\n\n"
                f"(API integration not yet implemented)"
            )

    def create_specialized_noodling(self, species: str):
        """Create a specialized Noodling with species-specific defaults."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            f"Create {species.title()} Noodling",
            "Noodling name:",
            text=f"New{species.title()}"
        )

        if ok and name:
            # Species-specific defaults
            presets = {
                'kitten': {
                    'extraversion': 0.7,
                    'curiosity': 0.9,
                    'impulsivity': 0.8,
                    'emotional_volatility': 0.6
                },
                'robot': {
                    'extraversion': 0.3,
                    'curiosity': 0.6,
                    'impulsivity': 0.2,
                    'emotional_volatility': 0.1
                },
                'dragon': {
                    'extraversion': 0.6,
                    'curiosity': 0.5,
                    'impulsivity': 0.4,
                    'emotional_volatility': 0.7
                }
            }

            personality = presets.get(species, {})

            QMessageBox.information(
                self,
                "Create Specialized Noodling",
                f"Creating {species} Noodling: {name}\n\n"
                f"Personality preset:\n"
                f"  Extraversion: {personality.get('extraversion', 0.5)}\n"
                f"  Curiosity: {personality.get('curiosity', 0.5)}\n"
                f"  Impulsivity: {personality.get('impulsivity', 0.5)}\n"
                f"  Volatility: {personality.get('emotional_volatility', 0.5)}\n\n"
                f"(API integration not yet implemented)"
            )

    def create_empty_object(self):
        """Create an empty object prim."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Create Empty Object",
            "Object name:",
            text="NewObject"
        )

        if ok and name:
            QMessageBox.information(
                self,
                "Create Object",
                f"Creating empty object: {name}\n\n(API integration not yet implemented)"
            )

    def create_specialized_object(self, obj_type: str):
        """Create a specialized object with type-specific properties."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            f"Create {obj_type.title()}",
            f"{obj_type.title()} name:",
            text=f"New{obj_type.title()}"
        )

        if ok and name:
            properties = {
                'prop': 'holdable=true, takeable=true',
                'furniture': 'sittable=true, fixed=true',
                'container': 'openable=true, container=true'
            }

            QMessageBox.information(
                self,
                "Create Specialized Object",
                f"Creating {obj_type}: {name}\n\n"
                f"Properties: {properties.get(obj_type, 'none')}\n\n"
                f"(API integration not yet implemented)"
            )

    def create_empty_room(self):
        """Create an empty room prim."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Create Empty Room",
            "Room name:",
            text="NewRoom"
        )

        if ok and name:
            QMessageBox.information(
                self,
                "Create Room",
                f"Creating empty room: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_prim(self):
        """Create a custom empty prim."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Create Empty Prim",
            "Prim name:",
            text="CustomPrim"
        )

        if ok and name:
            QMessageBox.information(
                self,
                "Create Prim",
                f"Creating empty prim: {name}\n\n(API integration not yet implemented)"
            )

    def create_empty_ensemble(self):
        """Create an empty ensemble that users can drag Noodlings into (like Unity prefab creation)."""
        from PyQt6.QtWidgets import QInputDialog

        name, ok = QInputDialog.getText(
            self,
            "Create Empty Ensemble",
            "Ensemble name:",
            text="MyEnsemble"
        )

        if ok and name:
            # Create empty ensemble in Scene Hierarchy
            # Users will drag Noodlings into it to build their custom ensemble

            QMessageBox.information(
                self,
                "Empty Ensemble Created",
                f"Created empty ensemble: {name}\n\n"
                f"Now drag Noodlings into the ensemble in Scene Hierarchy!\n\n"
                f"When ready:\n"
                f"  1. Right-click ensemble\n"
                f"  2. Choose 'Export Ensemble to .ens'\n"
                f"  3. Share your .ens file!\n\n"
                f"(Full implementation coming soon)"
            )

            # TODO: Create special "Ensemble" prim type in Scene Hierarchy
            # TODO: Allow dragging Noodlings into it
            # TODO: Right-click → Export Ensemble to .ens

    def import_noodling_file(self):
        """Import a single Noodling character (.nood file)."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Noodling Character",
            str(Path.home() / ".noodlestudio" / "characters"),
            "Noodling Files (*.nood);;All Files (*)"
        )

        if filename:
            QMessageBox.information(
                self,
                "Import Noodling",
                f"Importing Noodling from:\n{filename}\n\n(Implementation coming soon)"
            )

    def export_noodlings_dialog(self):
        """Open unified export dialog for Noodling(s)."""
        import requests

        try:
            # Get current Noodlings
            resp = requests.get("http://localhost:8081/api/agents", timeout=2)
            agents = resp.json().get('agents', [])

            if not agents:
                QMessageBox.warning(self, "No Noodlings", "No Noodlings currently active.\nRez some first!")
                return

            # Open export dialog
            from ..dialogs import ExportNoodlingsDialog

            dialog = ExportNoodlingsDialog(agents, self)
            if dialog.exec() == QDialog.DialogCode.Accepted:
                if dialog.result_path:
                    self.statusBar().showMessage(f"Exported to {dialog.result_path}", 5000)
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Exported successfully to:\n{dialog.result_path}"
                    )

        except Exception as e:
            import traceback
            print(f"Export dialog error: {e}")
            traceback.print_exc()
            QMessageBox.critical(self, "Export Failed", f"Error: {e}")

    def export_ensemble_file(self):
        """Export selected Noodlings as .ens ensemble file."""
        from PyQt6.QtWidgets import QFileDialog, QInputDialog
        from pathlib import Path
        import requests

        # Get list of current Noodlings
        try:
            resp = requests.get("http://localhost:8081/api/agents", timeout=2)
            agents = resp.json().get('agents', [])

            if not agents:
                QMessageBox.warning(self, "No Noodlings", "No Noodlings currently active.\nRez some first!")
                return

            # Let user select which Noodlings to include
            agent_names = [f"{a.get('name', a.get('id'))}" for a in agents]

            # For now, export ALL current Noodlings (TODO: add selection dialog)
            ensemble_name, ok = QInputDialog.getText(
                self,
                "Export Ensemble",
                f"Export {len(agents)} current Noodlings as ensemble?\n\nEnsemble name:",
                text="MyEnsemble"
            )

            if not ok or not ensemble_name:
                return

            ensemble_desc, ok = QInputDialog.getText(
                self,
                "Ensemble Description",
                "Description:",
                text="Custom ensemble"
            )

            if not ok:
                return

            # Choose save location
            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Ensemble",
                str(Path.home() / ".noodlestudio" / "ensembles" / f"{ensemble_name.lower().replace(' ', '_')}.ens"),
                "Ensemble Files (*.ens)"
            )

            if filename:
                from ..data.ensemble_exporter import EnsembleExporter

                exporter = EnsembleExporter()
                agent_ids = [a.get('id') for a in agents]

                success = exporter.export_from_noodlings(
                    agent_ids,
                    ensemble_name,
                    ensemble_desc,
                    Path(filename)
                )

                if success:
                    QMessageBox.information(
                        self,
                        "Export Complete",
                        f"Exported {len(agent_ids)} Noodlings to:\n{filename}\n\n"
                        f"You can now import this ensemble later!"
                    )
                else:
                    QMessageBox.critical(self, "Export Failed", "Failed to export ensemble.\n\nCheck Console for details.")

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"Ensemble export error:\n{error_detail}")
            QMessageBox.critical(self, "Export Failed", f"Error: {e}\n\nCheck Console for details.")

    def import_ensemble(self):
        """Import an ensemble prefab (.ensemble file)."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Ensemble",
            str(Path.home() / ".noodlestudio" / "ensembles"),
            "Ensemble Files (*.ensemble);;All Files (*)"
        )

        if filename:
            try:
                from ..data.ensemble_format import EnsembleFormat, EnsembleSpawner

                # Load ensemble from .ens file
                pack = EnsembleFormat.load_ensemble(Path(filename))

                # Ask which room to spawn into
                from PyQt6.QtWidgets import QInputDialog
                room_id, ok = QInputDialog.getText(
                    self,
                    "Spawn Ensemble",
                    f"Spawn '{pack.name}' ensemble into which room?",
                    text="room_000"
                )

                if ok and room_id:
                    # Rez all archetypes
                    rezzed_ids = EnsembleSpawner.rez_ensemble(pack, room_id)

                    QMessageBox.information(
                        self,
                        "Ensemble Imported",
                        f"Rezzed ensemble: {pack.name}\n\n"
                        f"Archetypes:\n" + "\n".join([f"  - {a.name}" for a in pack.archetypes]) + "\n\n"
                        f"Room: {room_id}\n\n"
                        f"Suggested scene: {pack.scene_suggestions[0] if pack.scene_suggestions else 'None'}\n\n"
                        f"(API integration not yet implemented)"
                    )

            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Error importing ensemble:\n{e}\n\nCheck that the .ens file is valid."
                )

    def show_ensemble_store(self):
        """Show Ensemble Store window (Unity Asset Store style)."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QTextEdit, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Ensemble Store - Unity Asset Store for Consciousness!")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel("<h1>🎭 Ensemble Store</h1><p>Ready-made kindled archetypes for your stage</p>")
        header.setStyleSheet("padding: 10px; background: #2a2a2a;")
        layout.addWidget(header)

        # List of available ensembles
        list_widget = QListWidget()

        from ..data.ensemble_packs import ENSEMBLE_LIBRARY

        for pack in ENSEMBLE_LIBRARY.list_packs():
            price_str = "FREE" if pack.price == 0.0 else f"${pack.price}"
            list_widget.addItem(f"{pack.name} - {price_str} ({len(pack.archetypes)} archetypes)")

        layout.addWidget(list_widget)

        # Description area
        desc_area = QTextEdit()
        desc_area.setReadOnly(True)
        desc_area.setPlainText("Select an ensemble to see details...")
        layout.addWidget(desc_area)

        def on_selection_changed():
            if list_widget.currentRow() >= 0:
                packs = ENSEMBLE_LIBRARY.list_packs()
                pack = packs[list_widget.currentRow()]

                desc = f"**{pack.name}**\n\n"
                desc += f"{pack.description}\n\n"
                desc += f"**Version:** {pack.version}\n"
                desc += f"**Author:** {pack.author}\n"
                desc += f"**Price:** {'FREE' if pack.price == 0.0 else f'${pack.price}'}\n"
                desc += f"**License:** {pack.license_type}\n\n"
                desc += f"**Archetypes:**\n"
                for arch in pack.archetypes:
                    desc += f"  - {arch.name} ({arch.species})\n"
                desc += f"\n**Setting:** {pack.suggested_setting}\n"
                desc += f"\n**Dynamics:** {pack.relationship_dynamics}\n"

                desc_area.setPlainText(desc)

        list_widget.currentRowChanged.connect(on_selection_changed)

        # Buttons
        button_layout = QHBoxLayout()

        export_btn = QPushButton("Export to .ens File")
        export_btn.clicked.connect(lambda: self.export_ensemble_to_file(list_widget, ENSEMBLE_LIBRARY))
        button_layout.addWidget(export_btn)

        spawn_btn = QPushButton("Spawn Ensemble Now")
        spawn_btn.clicked.connect(lambda: self.spawn_ensemble_from_store(list_widget, ENSEMBLE_LIBRARY, dialog))
        button_layout.addWidget(spawn_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def export_ensemble_to_file(self, list_widget, library):
        """Export selected ensemble to .ens file."""
        if list_widget.currentRow() >= 0:
            from PyQt6.QtWidgets import QFileDialog
            from pathlib import Path
            from ..data.ensemble_format import EnsembleFormat

            packs = library.list_packs()
            pack = packs[list_widget.currentRow()]

            filename, _ = QFileDialog.getSaveFileName(
                self,
                "Export Ensemble",
                str(Path.home() / ".noodlestudio" / "ensembles" / f"{pack.id}.ens"),
                "Ensemble Files (*.ens)"
            )

            if filename:
                EnsembleFormat.save_ensemble(pack, Path(filename))
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Ensemble exported to:\n{filename}\n\nYou can now share this .ens file!"
                )

    def add_component(self, component_type: str):
        """Add a component to the selected entity (Unity-style)."""
        from PyQt6.QtWidgets import QMessageBox

        component_names = {
            'noodle': 'Noodle Component',
            'memory': 'Memory Bank Component',
            'relationships': 'Relationship Graph Component',
            'artbook': 'Artbook Component',
            'moodboard': 'Mood Board Component',
            'voiceref': 'Voice Reference Component',
            'dialogue': 'Dialogue Tree Component',
            'quests': 'Quest Giver Component',
            'vendor': 'Vendor Component',
            'custom': 'Custom Script'
        }

        component_name = component_names.get(component_type, 'Unknown Component')

        # Check if entity is selected
        if not hasattr(self.inspector, 'current_entity') or not self.inspector.current_entity:
            QMessageBox.warning(
                self,
                "No Entity Selected",
                "Please select an entity in the Scene Hierarchy first,\nthen add a component to it."
            )
            return

        entity_type, entity_data = self.inspector.current_entity

        if component_type == 'artbook':
            # Add Artbook component to Inspector
            self.inspector.add_artbook_component()
            self.statusBar().showMessage(f"Added {component_name} to {entity_type}", 3000)

        elif component_type == 'custom':
            # Add Script component to Inspector
            self.inspector.add_script_component()
            self.statusBar().showMessage(f"Added Script Component to {entity_type}", 3000)

        elif component_type == 'noodle':
            QMessageBox.information(
                self,
                "Noodle Component",
                "Noodle Component is automatically added to all Noodlings!\n\n"
                "It shows live affect, phenomenal state, and surprise."
            )

        else:
            QMessageBox.information(
                self,
                f"Add {component_name}",
                f"Adding {component_name}...\n\n(Implementation coming soon)"
            )

    def spawn_ensemble_from_store(self, list_widget, library, dialog):
        """Spawn selected ensemble into noodleMUSH."""
        if list_widget.currentRow() >= 0:
            from PyQt6.QtWidgets import QInputDialog
            from ..data.ensemble_format import EnsembleSpawner

            packs = library.list_packs()
            pack = packs[list_widget.currentRow()]

            room_id, ok = QInputDialog.getText(
                self,
                "Rez Ensemble",
                f"Rez '{pack.name}' into which room?",
                text="room_000"
            )

            if ok and room_id:
                rezzed_ids = EnsembleSpawner.rez_ensemble(pack, room_id)

                QMessageBox.information(
                    self,
                    "Ensemble Rezzed",
                    f"Rezzed {len(rezzed_ids)} Noodlings from '{pack.name}'\n\n"
                    f"Room: {room_id}\n\n"
                    f"(API integration not yet implemented)"
                )

                dialog.close()

    def show_about(self):
        """Show About dialog."""
        QMessageBox.about(
            self,
            "About NoodleStudio",
            "NoodleStudio v1.0.0-alpha\n\n"
            "Professional IDE for Noodlings\n\n"
            "Built with PyQt6 | The Krugerrand Edition\n"
            "Worth its weight in gold!"
        )

    def _show_preferences(self):
        """Show preferences dialog."""
        QMessageBox.information(
            self,
            "Preferences",
            "Preferences dialog coming soon!\n\n"
            "For now, edit ~/.noodlestudio/config.yaml"
        )

    def _show_docs(self):
        """Show documentation."""
        QMessageBox.information(
            self,
            "Documentation",
            "📖 Documentation:\n\n"
            "See /applications/noodleSTUDIO/docs/\n"
            "- ARCHITECTURE.md\n"
            "- IMPLEMENTATION_PLAN.md\n"
            "- QUICKSTART.md\n"
            "- HOME_PANEL_SPEC.md"
        )

    def _show_about(self):
        """Show about dialog."""
        QMessageBox.about(
            self,
            "About NoodleSTUDIO",
            "🧠 <b>NoodleSTUDIO</b><br>"
            "Version 1.0.0-alpha<br><br>"
            "Professional IDE for Noodlings<br><br>"
            "<b>Consilience, Inc.</b><br>"
            "Founded by Caitlyn Meeks<br><br>"
            "\"Movies are out. Noodlings are in.\"<br><br>"
            "🚀 Built with PyQt6 & MLX"
        )
