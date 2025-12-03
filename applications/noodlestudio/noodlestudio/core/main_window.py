"""
Main Window for NoodleSTUDIO.

The primary application window with menu bar, toolbar, dock area, and status bar.
"""

import os
import json
from pathlib import Path
from typing import Optional, List
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox, QTabWidget,
    QHBoxLayout, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QStandardPaths, QUrl
from PyQt6.QtGui import QAction, QFont

from ..panels.home_panel import HomePanel
from ..panels.chat_panel import ChatPanel
from ..panels.profiler_panel import ProfilerPanel
from ..panels.scene_hierarchy import SceneHierarchy
from ..panels.inspector_panel import InspectorPanel
from ..panels.console_panel import ConsolePanel
from ..panels.assets_panel import AssetsPanel
from ..panels.noodle_tuner_panel import NoodleTunerPanel
from .theme import DARK_THEME
from .unity_theme import UNITY_DARK_THEME
from .layout_manager import LayoutManager
from .project_manager import ProjectManager
from PyQt6.QtWidgets import QDialog, QFileDialog, QInputDialog


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

        # Apply dark theme with darker gray background (not jet black)
        # This distinguishes IDE chrome from noodleMUSH terminal content
        self.setStyleSheet(UNITY_DARK_THEME + """
            QMainWindow {
                background-color: #383838;
            }
            QWidget {
                background-color: #383838;
            }
        """)

        # Layout manager for saving configurations
        self.layout_manager = LayoutManager()

        # Project manager
        self.project_manager = ProjectManager()
        self.project_manager.projectOpened.connect(self.on_project_opened)
        self.project_manager.projectClosed.connect(self.on_project_closed)

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()
        self._setup_panels()
        self._setup_shortcuts()

        # Load last used layout (preserve workspace state)
        QTimer.singleShot(200, self.load_last_used_layout)

        # Auto-open last project (restore workspace)
        QTimer.singleShot(300, self.auto_open_last_project)

        # Show RNG status on startup
        QTimer.singleShot(500, self.show_startup_rng_status)

    def _setup_ui(self):
        """Build UI components."""
        # Central widget will be World View (main viewport)
        # Don't set it here - will be set in _setup_panels()

    def _setup_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        # Project management
        file_menu.addAction(self._create_action("&New Project...", slot=self.new_project))
        file_menu.addAction(self._create_action("&Open Project...", slot=self.open_project))

        # Recent Projects submenu
        self.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self.update_recent_projects_menu()

        file_menu.addSeparator()

        # Stage management
        file_menu.addAction(self._create_action("&New Stage", "Ctrl+N"))
        file_menu.addAction(self._create_action("&Open Stage...", "Ctrl+O"))
        file_menu.addAction(self._create_action("&Save Stage", "Ctrl+S"))

        # Import section
        file_menu.addSeparator()
        file_menu.addSection("Import")
        file_menu.addAction(self._create_action("Import Prim (.prim)...", slot=self.import_prim_menu))
        file_menu.addAction(self._create_action("Import Ensemble (.ensemble)...", slot=self.import_ensemble))
        file_menu.addAction(self._create_action("Import Noodling (.json)...", slot=self.import_noodling_file))

        # Export section
        file_menu.addSeparator()
        file_menu.addSection("Export")
        file_menu.addAction(self._create_action("Export Selected Prim(s)...", slot=self.export_selected_prims))
        file_menu.addAction(self._create_action("Export Noodling(s)...", slot=self.export_noodlings_dialog))

        # USD export/import
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Export Stage to USD (.usda)...", slot=self.export_stage_to_usd))
        file_menu.addAction(self._create_action("Export Timeline to USD (.usda)...", slot=self.export_timeline_to_usd))
        file_menu.addAction(self._create_action("Import USD Layer (.usda)...", slot=self.import_usd_layer))

        file_menu.addSeparator()
        file_menu.addAction(self._create_action("&Quit", "Ctrl+Q", self.close))

        # ===== REZ MENU (instantiate entities) =====
        create_menu = menu_bar.addMenu("&Rez")

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

        # Panel toggles - HIDDEN (layout is locked, panels always visible)
        # view_menu.addAction(self._create_action("Scene Hierarchy", "Ctrl+1", checkable=True, checked=True))
        # view_menu.addAction(self._create_action("World View", "Ctrl+2", checkable=True, checked=True))
        # view_menu.addAction(self._create_action("Inspector", "Ctrl+3", checkable=True, checked=True))
        # view_menu.addAction(self._create_action("Timeline Profiler", "Ctrl+4", checkable=True))
        # view_menu.addAction(self._create_action("Noodle Tuner", "Ctrl+5", checkable=True))
        # view_menu.addSeparator()

        # Layout presets - HIDDEN (layout is now locked down)
        # Keeping code for potential future use
        # layout_submenu = view_menu.addMenu("Layouts")
        # layout_submenu.addAction(self._create_action("Save Current Layout...", slot=self.save_current_layout))
        # layout_submenu.addAction(self._create_action("Set Current as Default", slot=self.set_current_as_default))
        # layout_submenu.addAction(self._create_action("Load Layout...", slot=self.load_layout_dialog))
        # layout_submenu.addSeparator()
        # layout_submenu.addAction(self._create_action("Reset to Default", slot=lambda: self.load_layout("Default")))
        # layout_submenu.addAction(self._create_action("Reset to Factory Default", slot=self.reset_to_factory_layout))

        # ===== ENTITIES MENU (create/manage entities) =====
        entities_menu = menu_bar.addMenu("&Entities")
        entities_menu.addAction(self._create_action("Add Noodling...", "Ctrl+Shift+N", slot=self.add_noodling))
        entities_menu.addAction(self._create_action("Add Object...", "Ctrl+Shift+O", slot=self.add_object))
        entities_menu.addAction(self._create_action("Add Room...", slot=self.add_room))
        entities_menu.addSeparator()
        entities_menu.addAction(self._create_action("Remove Selected", "Delete"))
        entities_menu.addSeparator()
        entities_menu.addAction(self._create_action("Toggle Enlightenment", "Ctrl+E"))
        entities_menu.addAction(self._create_action("Reset All States"))

        # ===== COMPONENT MENU (modular component system) =====
        component_menu = menu_bar.addMenu("&Component")

        #  components
        charm_menu = component_menu.addMenu("Charm")
        charm_menu.addAction(self._create_action("Noodle", slot=lambda: self.add_component("noodle")))
        charm_menu.addAction(self._create_action("Memory Bank", slot=lambda: self.add_component("memory")))
        charm_menu.addAction(self._create_action("Relationship Graph", slot=lambda: self.add_component("relationships")))

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
        # Panel visibility shortcuts - HIDDEN (layout is locked, panels always visible)
        # window_menu.addAction(self._create_action("Stage Hierarchy", "Cmd+1", lambda: self._toggle_panel(self.hierarchy)))
        # window_menu.addAction(self._create_action("Assets", "Cmd+2", lambda: self._toggle_panel(self.assets)))
        # window_menu.addAction(self._create_action("World View", "Cmd+3", lambda: self._toggle_panel(self.world_view)))
        # window_menu.addAction(self._create_action("Inspector", "Cmd+4", lambda: self._toggle_panel(self.inspector)))
        # window_menu.addAction(self._create_action("Noodle Tuner", "Cmd+5", lambda: self._toggle_panel(self.noodle_tuner)))
        # window_menu.addAction(self._create_action("Console", "Cmd+6", lambda: self._toggle_panel(self.console)))
        # window_menu.addAction(self._create_action("Timeline Profiler", "Cmd+7", lambda: self._toggle_panel(self.profiler_panel)))
        # window_menu.addSeparator()
        window_menu.addAction(self._create_action("Ensemble Store...", slot=self.show_ensemble_store))
        window_menu.addSeparator()
        # window_menu.addAction(self._create_action("Reset to Default Layout", slot=lambda: self.load_layout("Default")))

        # ===== SETTINGS MENU =====
        settings_menu = menu_bar.addMenu("&Settings")
        settings_menu.addAction(self._create_action("Random Number Generator...", slot=self.show_rng_settings))
        settings_menu.addAction(self._create_action("External Applications...", slot=self.show_external_apps_settings))

        # ===== HELP MENU =====
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._create_action("NoodleStudio Documentation", "F1"))
        help_menu.addAction(self._create_action("Noodlings Architecture Guide"))
        help_menu.addAction(self._create_action("Report Issue..."))
        help_menu.addSeparator()
        help_menu.addAction(self._create_action("Credits (Demo Scene Style)", slot=self.show_credits))
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

        # Update status after a delay (increased to 5 seconds for server startup)
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(5000, self.update_connection_status)

    def update_connection_status(self):
        """Update connection status label and UI state."""
        running = self.is_server_running()

        if running:
            self.connection_label.setText("Server running on :8765")
            self.connection_label.setStyleSheet("color: #76AF6A;")  # Green
            self.server_toggle.setChecked(True)
        else:
            self.connection_label.setText("Server offline")
            self.connection_label.setStyleSheet("color: #999;")  # Gray
            self.server_toggle.setChecked(False)

        # Update World View
        if hasattr(self, 'world_view'):
            self.world_view.set_server_state(running)

        # Update Hierarchy (gray out if offline)
        if hasattr(self, 'hierarchy'):
            self.hierarchy.set_server_state(running)

        # Update Console (reconnect if server just started)
        if hasattr(self, 'console'):
            if running and not self.console.connected:
                self.console.reconnect()

    def _setup_panels(self):
        """Create locked-down layout with fixed splitters (no dragging/docking)."""

        # LEFT COLUMN: Tabbed widget for Hierarchy + Assets
        left_tabs = QTabWidget()
        left_tabs.setTabPosition(QTabWidget.TabPosition.North)
        left_tabs.setMinimumWidth(150)  # Prevent collapsing
        left_tabs.setDocumentMode(True)  # Remove extra margins/backgrounds
        left_tabs.setStyleSheet("""
            QTabWidget {
                background-color: #383838;
            }
            QTabWidget::pane {
                border: none;
                background-color: #3E3E3E;
            }
            QTabWidget::tab-bar {
                background-color: #383838;
                alignment: left;
            }
            QTabBar {
                background-color: #383838;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #888888;
                padding: 6px 12px;
                border: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3E3E3E;
                color: #CCCCCC;
            }
        """)

        self.hierarchy = SceneHierarchy(None)  # Not a dock widget anymore
        self.assets = AssetsPanel(None)
        self.assets.project_manager = self.project_manager
        self.assets.agentRezzed.connect(self.hierarchy.refresh_scene)

        left_tabs.addTab(self.hierarchy, "Stage")
        left_tabs.addTab(self.assets, "Assets")

        # CENTER: Tabbed widget for World View + Facets Editor
        center_tabs = QTabWidget()
        center_tabs.setTabPosition(QTabWidget.TabPosition.North)
        center_tabs.setStyleSheet("""
            QTabWidget::pane {
                border: none;
                background: #383838;
            }
            QTabBar::tab {
                background: #3a3a3a;
                color: #888888;
                padding: 8px 16px;
                border: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3E3E3E;
                color: #D2D2D2;
            }
        """)

        # World View tab (WebView)
        world_widget = QWidget()
        world_layout = QVBoxLayout(world_widget)
        world_layout.setContentsMargins(0, 0, 0, 0)
        world_layout.setSpacing(0)

        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self.web_view = QWebEngineView()

            # Set background color to match theme (prevents white flash)
            # Don't force dark mode - noodleMUSH has its own styling
            self.web_view.setStyleSheet("background-color: #1a1a1a;")

            self.web_view.setUrl(QUrl("http://localhost:8080"))
            world_layout.addWidget(self.web_view)
        except ImportError:
            placeholder = QLabel("WebEngine not available\nInstall: pip install PyQt6-WebEngine")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #999; font-size: 14px;")
            world_layout.addWidget(placeholder)
            self.web_view = None

        center_tabs.addTab(world_widget, "World")

        # Facets Editor tab
        from ..panels.facets_editor_panel import FacetsEditorPanel
        self.facets_editor = FacetsEditorPanel()
        center_tabs.addTab(self.facets_editor, "Facets Editor")

        # Store reference to center tabs for access
        self.center_tabs = center_tabs

        # Create stub for compatibility
        class WorldViewStub:
            def __init__(self, web_view):
                self.web_view = web_view
            def show(self): pass
            def hide(self): pass
            def isVisible(self): return True
            def raise_(self): pass
            def set_server_state(self, running):
                if not running:
                    self.show_offline_card()
                else:
                    # Server is online - reload to show noodleMUSH
                    if self.web_view:
                        self.web_view.setUrl(QUrl("http://localhost:8080"))
            def reload(self):
                if self.web_view:
                    self.web_view.reload()
            def toggle_maximize(self): pass
            def show_offline_card(self):
                """Show offline placeholder when server is not running."""
                if self.web_view:
                    offline_html = """
                    <html>
                    <head>
                        <style>
                            body {
                                background: #1a1a1a;
                                color: #999;
                                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                                display: flex;
                                align-items: center;
                                justify-content: center;
                                height: 100vh;
                                margin: 0;
                            }
                            .card {
                                text-align: center;
                                padding: 40px;
                                background: #2d2d2d;
                                border-radius: 8px;
                                border: 2px solid #3e3e3e;
                            }
                            .icon {
                                font-size: 64px;
                                margin-bottom: 20px;
                            }
                            h1 {
                                color: #ccc;
                                font-size: 24px;
                                margin-bottom: 10px;
                            }
                            p {
                                color: #888;
                                font-size: 14px;
                                margin: 5px 0;
                            }
                            .hint {
                                margin-top: 20px;
                                font-size: 12px;
                                color: #666;
                            }
                        </style>
                    </head>
                    <body>
                        <div class="card">
                            <div class="icon">🔌</div>
                            <h1>noodleMUSH Server Offline</h1>
                            <p>Please start the server to view the world</p>
                            <p class="hint">Toggle the server switch in the bottom right</p>
                        </div>
                    </body>
                    </html>
                    """
                    self.web_view.setHtml(offline_html)

        self.world_view = WorldViewStub(self.web_view)

        # RIGHT COLUMN: Tabbed widget for Inspector + Noodle Tuner
        right_tabs = QTabWidget()
        right_tabs.setTabPosition(QTabWidget.TabPosition.North)
        right_tabs.setMinimumWidth(200)  # Prevent collapsing
        right_tabs.setDocumentMode(True)  # Remove extra margins/backgrounds
        right_tabs.setStyleSheet("""
            QTabWidget {
                background-color: #383838;
            }
            QTabWidget::pane {
                border: none;
                background-color: #3E3E3E;
            }
            QTabWidget::tab-bar {
                background-color: #383838;
                alignment: left;
            }
            QTabBar {
                background-color: #383838;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #888888;
                padding: 6px 12px;
                border: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3E3E3E;
                color: #CCCCCC;
            }
        """)

        self.inspector = InspectorPanel(None)
        self.noodle_tuner = NoodleTunerPanel(None)

        right_tabs.addTab(self.inspector, "Inspector")
        right_tabs.addTab(self.noodle_tuner, "Noodle Tuner")

        # BOTTOM: Tabbed widget for Console + Profiler
        bottom_tabs = QTabWidget()
        bottom_tabs.setTabPosition(QTabWidget.TabPosition.North)
        bottom_tabs.setMinimumHeight(100)  # Prevent collapsing
        bottom_tabs.setDocumentMode(True)  # Remove extra margins/backgrounds
        bottom_tabs.setStyleSheet("""
            QTabWidget {
                background-color: #383838;
            }
            QTabWidget::pane {
                border: none;
                background-color: #3E3E3E;
            }
            QTabWidget::tab-bar {
                background-color: #383838;
                alignment: left;
            }
            QTabBar {
                background-color: #383838;
            }
            QTabBar::tab {
                background-color: #2D2D2D;
                color: #888888;
                padding: 6px 12px;
                border: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #3E3E3E;
                color: #CCCCCC;
            }
        """)

        self.console = ConsolePanel(None)
        self.profiler_panel = ProfilerPanel(None)

        bottom_tabs.addTab(self.console, "Console")
        bottom_tabs.addTab(self.profiler_panel, "Timeline Profiler")

        # Create horizontal splitter for left | center | right
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(left_tabs)
        top_splitter.addWidget(center_tabs)
        top_splitter.addWidget(right_tabs)
        top_splitter.setStretchFactor(0, 0)  # Left fixed width
        top_splitter.setStretchFactor(1, 1)  # Center stretches
        top_splitter.setStretchFactor(2, 0)  # Right fixed width
        top_splitter.setSizes([250, 800, 280])
        top_splitter.setChildrenCollapsible(False)  # Prevent panels from disappearing!

        # Create vertical splitter for top | bottom
        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_tabs)
        main_splitter.setStretchFactor(0, 1)  # Top stretches
        main_splitter.setStretchFactor(1, 0)  # Bottom fixed height
        main_splitter.setSizes([600, 180])
        main_splitter.setChildrenCollapsible(False)  # Prevent panels from disappearing!

        # Style the splitter handles for visibility
        main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #383838;
            }
            QSplitter::handle:horizontal {
                width: 3px;
            }
            QSplitter::handle:vertical {
                height: 3px;
            }
        """)

        # Set as central widget
        self.setCentralWidget(main_splitter)

        # Connect signals
        self.hierarchy.entitySelected.connect(self.inspector.load_entity)
        self.hierarchy.entitySelected.connect(self.on_entity_selected_for_console)
        self.hierarchy.entitySelected.connect(self.on_entity_selected_for_noodle_tuner)
        self.hierarchy.entitySelected.connect(self.on_entity_selected_for_facets_editor)

        # Check server state
        QTimer.singleShot(200, self.update_connection_status)

    def _setup_shortcuts(self):
        """Setup keyboard shortcuts."""
        from PyQt6.QtGui import QShortcut, QKeySequence

        # Cmd/Ctrl+R - Reload with autologin
        reload_shortcut = QShortcut(QKeySequence("Ctrl+R"), self)
        reload_shortcut.activated.connect(self.reload_world_view)

        # Cmd/Ctrl+Shift+R - Reload to login screen
        reload_login_shortcut = QShortcut(QKeySequence("Ctrl+Shift+R"), self)
        reload_login_shortcut.activated.connect(self.reload_world_view_clean)

        # Cmd/Ctrl+M - Maximize/restore World View
        maximize_shortcut = QShortcut(QKeySequence("Ctrl+M"), self)
        maximize_shortcut.activated.connect(self.toggle_world_view_maximize)

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

    def toggle_world_view_maximize(self):
        """Toggle World View between maximized and normal (Ctrl+M)."""
        if hasattr(self, 'world_view'):
            self.world_view.toggle_maximize()

    def _toggle_panel(self, panel):
        """Toggle panel visibility (show/hide)."""
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()  # Bring to front if tabbed

    def _toggle_panel_maximize(self, panel):
        """Toggle any dock panel between maximized and normal state."""
        if hasattr(panel, 'toggle_maximize'):
            panel.show()  # Make sure panel is visible first
            panel.toggle_maximize()
        else:
            # Fallback for panels that don't inherit from MaximizableDock
            panel.show()


    def reset_to_factory_layout(self):
        """Reset to factory default layout (locked-down splitter layout)."""
        # Splitters are already locked down - just show message
        self.statusBar().showMessage("Layout is locked to optimal arrangement", 3000)

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
        """Load the last used layout on startup (restore workspace state)."""
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
                    f"Contains Noodling prims with charm properties.\n"
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
        """Create an empty ensemble that users can drag Noodlings into (prefab system)."""
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
        """Import an ensemble file into the project."""
        if not self.project_manager.is_project_open():
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please create or open a project first."
            )
            return

        # Default to cmush ensembles directory
        default_dir = os.path.join(
            os.path.dirname(__file__),
            "../../../cmush/ensembles"
        )
        default_dir = os.path.abspath(default_dir)

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Import Ensemble",
            default_dir,
            "Ensemble Files (*.ensemble);;All Files (*)"
        )

        if filename:
            try:
                # Import using project manager
                if self.project_manager.import_ensemble(filename):
                    basename = os.path.basename(filename)
                    self.statusBar().showMessage(f"Imported ensemble: {basename}", 3000)
                    # Refresh assets panel
                    if hasattr(self, 'assets'):
                        self.assets.refresh()
                else:
                    QMessageBox.warning(
                        self,
                        "Import Failed",
                        f"Failed to import ensemble."
                    )
            except Exception as e:
                QMessageBox.critical(
                    self,
                    "Import Failed",
                    f"Error importing ensemble:\n{e}"
                )

    def import_prim_menu(self):
        """Import prim from File menu (delegates to scene hierarchy)."""
        if hasattr(self, 'hierarchy') and self.hierarchy:
            self.hierarchy.import_prim()
        else:
            QMessageBox.warning(
                self,
                "No Scene",
                "Please create or open a scene first."
            )

    def export_selected_prims(self):
        """Export selected prims from File menu (delegates to scene hierarchy)."""
        if hasattr(self, 'hierarchy') and self.hierarchy:
            # Get selected items
            selected = self.hierarchy.tree.selectedItems()
            if not selected:
                QMessageBox.information(
                    self,
                    "No Selection",
                    "Please select one or more prims to export."
                )
                return

            # Export each selected prim
            for item in selected:
                entity_data = item.data(0, Qt.ItemDataRole.UserRole)
                if entity_data and isinstance(entity_data, dict):
                    entity_type = entity_data.get('type')
                    if entity_type == 'prim':
                        self.hierarchy.export_prim_data(entity_data)
        else:
            QMessageBox.warning(
                self,
                "No Scene",
                "Please create or open a scene first."
            )

    def show_ensemble_store(self):
        """Show Ensemble Store window (content marketplace)."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QListWidget, QTextEdit, QPushButton, QHBoxLayout

        dialog = QDialog(self)
        dialog.setWindowTitle("Ensemble Store")
        dialog.resize(800, 600)

        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel("<h1> Ensemble Store</h1><p>Ensemble archetypes for your stage</p>")
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
        """Add a component to the selected entity (modular component system)."""
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

    def new_project(self):
        """Create a new NoodleStudio project."""
        # Get project name
        project_name, ok = QInputDialog.getText(
            self,
            "New Project",
            "Project Name:",
            text="MyNoodlingProject"
        )

        if not ok or not project_name:
            return

        # Get location
        parent_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Project Location",
            os.path.expanduser("~/Documents")
        )

        if not parent_dir:
            return

        # Create project
        if self.project_manager.create_project(parent_dir, project_name):
            self.statusBar().showMessage(f"Created project: {project_name}", 3000)
        else:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to create project.\nProject may already exist at that location."
            )

    def open_project(self):
        """Open an existing NoodleStudio project."""
        project_path = QFileDialog.getExistingDirectory(
            self,
            "Open Project",
            os.path.expanduser("~/Documents")
        )

        if not project_path:
            return

        # Open project
        if self.project_manager.open_project(project_path):
            self.statusBar().showMessage(f"Opened project: {self.project_manager.current_project_name}", 3000)
        else:
            QMessageBox.warning(
                self,
                "Error",
                f"Failed to open project.\nNot a valid NoodleStudio project."
            )

    def on_project_opened(self, project_path: str):
        """Handle project opened event."""
        # Stop server when switching projects
        import subprocess
        subprocess.run(['pkill', '-f', 'python.*server.py'])

        # Update window title
        self.setWindowTitle(f"NoodleSTUDIO - {self.project_manager.current_project_name}")

        # Save to recent projects
        self.add_to_recent_projects(project_path)
        self.update_recent_projects_menu()

        # Refresh assets panel
        if hasattr(self, 'assets'):
            self.assets.refresh()

        # Show offline card (server is stopped)
        if hasattr(self, 'world_view'):
            self.world_view.show_offline_card()

        # Gray out hierarchy
        if hasattr(self, 'hierarchy'):
            self.hierarchy.set_server_state(False)

        # Update toggle
        QTimer.singleShot(500, self.update_connection_status)

        print(f"Project opened: {project_path}")

    def on_project_closed(self):
        """Handle project closed event."""
        # Reset window title
        self.setWindowTitle("NoodleSTUDIO - Noodlings IDE")

        # Refresh assets panel
        if hasattr(self, 'assets'):
            self.assets.refresh()

        print("Project closed")

    def get_settings_path(self) -> Path:
        """Get path to NoodleStudio settings file."""
        config_dir = Path(QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation))
        config_dir = config_dir / "NoodleStudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "settings.json"

    def load_recent_projects(self) -> List[str]:
        """Load recent projects list from settings."""
        settings_path = self.get_settings_path()
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
                    return settings.get('recent_projects', [])
            except:
                return []
        return []

    def save_recent_projects(self, projects: List[str]):
        """Save recent projects list to settings."""
        settings_path = self.get_settings_path()
        settings = {}
        if settings_path.exists():
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except:
                pass
        settings['recent_projects'] = projects
        with open(settings_path, 'w') as f:
            json.dump(settings, f, indent=2)

    def add_to_recent_projects(self, project_path: str):
        """Add a project to the recent projects list."""
        recent = self.load_recent_projects()
        # Remove if already in list (to move to top)
        if project_path in recent:
            recent.remove(project_path)
        # Add to front
        recent.insert(0, project_path)
        # Keep only last 10
        recent = recent[:10]
        self.save_recent_projects(recent)

    def update_recent_projects_menu(self):
        """Update the Recent Projects menu with current list."""
        self.recent_projects_menu.clear()
        recent = self.load_recent_projects()

        if not recent:
            action = self.recent_projects_menu.addAction("(No recent projects)")
            action.setEnabled(False)
            return

        for project_path in recent:
            # Check if project still exists
            if not os.path.exists(project_path):
                continue

            # Get project name from path
            project_name = os.path.basename(project_path)
            action = self.recent_projects_menu.addAction(project_name)
            # Use lambda with default argument to capture project_path
            action.triggered.connect(lambda checked, p=project_path: self.open_recent_project(p))

        # Add separator and "Clear Recent" option
        if recent:
            self.recent_projects_menu.addSeparator()
            clear_action = self.recent_projects_menu.addAction("Clear Recent Projects")
            clear_action.triggered.connect(self.clear_recent_projects)

    def open_recent_project(self, project_path: str):
        """Open a project from the recent projects list."""
        if os.path.exists(project_path):
            self.project_manager.open_project(project_path)
        else:
            QMessageBox.warning(
                self,
                "Project Not Found",
                f"Project no longer exists:\n{project_path}"
            )
            # Remove from recent list
            recent = self.load_recent_projects()
            if project_path in recent:
                recent.remove(project_path)
                self.save_recent_projects(recent)
                self.update_recent_projects_menu()

    def clear_recent_projects(self):
        """Clear the recent projects list."""
        self.save_recent_projects([])
        self.update_recent_projects_menu()

    def auto_open_last_project(self):
        """Automatically open the last opened project on startup."""
        recent = self.load_recent_projects()
        if recent and os.path.exists(recent[0]):
            print(f"Auto-opening last project: {recent[0]}")
            self.project_manager.open_project(recent[0])

    def on_entity_selected_for_console(self, entity_type: str, entity_data: dict):
        """Update Console filter when entity is selected in hierarchy."""
        if not hasattr(self, 'console'):
            return

        # Handle deselection
        if entity_type is None or entity_data is None:
            self.console.set_selected_entities([])
            return

        # Get entity ID
        entity_id = entity_data.get('id', '')

        # TODO: Support multi-selection - for now just single entity
        # In future, hierarchy should emit list of all selected entities
        if entity_id:
            self.console.set_selected_entities([entity_id])

    def on_entity_selected_for_noodle_tuner(self, entity_type: str, entity_data: dict):
        """Update Noodle Tuner when an agent is selected in hierarchy."""
        if not hasattr(self, 'noodle_tuner'):
            return

        # Handle deselection
        if entity_type is None or entity_data is None:
            # Clear noodle tuner
            self.noodle_tuner.set_agent(None)
            return

        # Only update Noodle Tuner for noodlings
        if entity_type == 'noodling':
            agent_id = entity_data.get('id', '')
            if agent_id:
                self.noodle_tuner.set_agent(agent_id)

    def on_entity_selected_for_facets_editor(self, entity_type: str, entity_data: dict):
        """Update Facets Editor when a noodling is selected in hierarchy."""
        if not hasattr(self, 'facets_editor'):
            return

        # Handle deselection (nothing selected)
        if entity_type is None or entity_data is None:
            self.facets_editor.clear_editor()
            return

        # Only load facet assemblies for noodlings
        if entity_type == 'noodling':
            agent_id = entity_data.get('id', '')
            if agent_id:
                import os
                from ..core.facet_system import FacetAssembly

                # Check if agent has facet_assembly reference
                # entity_data structure: {'type': 'noodling', 'id': 'agent_xxx', 'data': {full agent data including config}}
                agent_full_data = entity_data.get('data', {})
                config = agent_full_data.get('config', {})
                facet_assembly_config = config.get('facet_assembly')

                assembly_filename = None
                if facet_assembly_config:
                    # Handle both string and dict formats
                    if isinstance(facet_assembly_config, str):
                        ref = facet_assembly_config
                    elif isinstance(facet_assembly_config, dict):
                        ref = facet_assembly_config.get('ref')
                    else:
                        ref = None

                    if ref:
                        assembly_filename = f"{ref}.yaml"
                        print(f"[Facets Editor] Loading assembly from ref: {assembly_filename}")

                # Fallback to default if no reference
                if not assembly_filename:
                    assembly_filename = "anklebiter_default.yaml"
                    print(f"[Facets Editor] No assembly ref, using default: {assembly_filename}")

                # Build path to assembly file (up to noodlestudio/ then into facet_assemblies/)
                assembly_path = os.path.join(
                    os.path.dirname(__file__),
                    '../../facet_assemblies',
                    assembly_filename
                )
                print(f"[Facets Editor] Looking for assembly at: {assembly_path}")

                if os.path.exists(assembly_path):
                    try:
                        assembly = FacetAssembly.load_yaml(assembly_path)

                        # Always reload when switching agents (even if same assembly name)
                        # This ensures we're viewing the correct agent's instance
                        was_loaded = (self.facets_editor.current_assembly_name == assembly.name and
                                     self.facets_editor.current_agent_id == agent_id)

                        self.facets_editor.load_assembly_from_data(assembly, force_reload=not was_loaded)

                        # Set current agent (enables pause button, tracks agent for API calls)
                        self.facets_editor.set_current_agent(agent_id)

                        # Don't auto-switch tabs - let user control tab selection
                        # Just load assembly in background
                        if not was_loaded:
                            print(f"[Facets Editor] Loaded assembly '{assembly.name}' for agent {agent_id}")
                    except Exception as e:
                        print(f"[Facets Editor] Error loading facet assembly: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[Facets Editor] Assembly file not found: {assembly_path}")

    def show_credits(self):
        """Show demo scene style credits with music."""
        from ..panels.credits_panel import show_credits
        self.credits_window = show_credits(self)

    def show_about(self):
        """Show About dialog with Bjork's tentacles quote."""
        about_text = (
            "NoodleSTUDIO v1.0.0-alpha\n\n"
            "Symbiosis of Tendrils:\n"
            "Unfurling, Developing,\n"
            "Interconnected Organisms\n\n"
            "IDE for Noodlings\n"
            "Built with PyQt6"
        )

        QMessageBox.about(
            self,
            "About NoodleSTUDIO",
            about_text
        )

    def show_rng_settings(self):
        """Show Random Number Generator settings dialog."""
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QComboBox, QPushButton, QHBoxLayout
        import os

        dialog = QDialog(self)
        dialog.setWindowTitle("Random Number Generator Settings")
        dialog.resize(400, 150)

        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel("Select Random Number Generator:")
        header.setStyleSheet("font-weight: bold; font-size: 13px;")
        layout.addWidget(header)

        # RNG selection dropdown
        rng_combo = QComboBox()
        rng_combo.addItem("Internal RNG (Software)")

        # Check if TrueRNG/ubild USB RNG is connected
        ubild_available = self._check_ubild_connected()
        if ubild_available:
            rng_combo.addItem("TrueRNG (USB Hardware RNG)")

        # Load current setting
        current_rng = self._load_rng_setting()
        if current_rng == "truerng" and ubild_available:
            rng_combo.setCurrentIndex(1)
        else:
            rng_combo.setCurrentIndex(0)

        layout.addWidget(rng_combo)

        # Status label
        if ubild_available:
            status_label = QLabel("Hardware RNG detected")
            status_label.setStyleSheet("color: #76AF6A;")
        else:
            status_label = QLabel("No RNG detected. Falling back to internal RNG.\nOutputs are deterministic. Consider an avalanche effect RNG\nfor quantum non-determinism.")
            status_label.setStyleSheet("color: #999;")
            status_label.setWordWrap(True)
        layout.addWidget(status_label)

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(lambda: self._save_rng_setting(rng_combo.currentText(), dialog))
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _check_ubild_connected(self):
        """Check if TrueRNG/ubild USB hardware RNG is connected."""
        try:
            import subprocess
            # Check via system_profiler on macOS for TrueRNG device
            result = subprocess.run(['system_profiler', 'SPUSBDataType'],
                                  capture_output=True, text=True, timeout=3)
            # Look for TrueRNG, ubild, or other hardware RNG devices
            stdout_lower = result.stdout.lower()
            if 'truerng' in stdout_lower or 'ubild' in stdout_lower or 'hardware rng' in stdout_lower:
                return True

            # Also check for specific device files on macOS
            import glob
            usb_devices = glob.glob('/dev/cu.usbmodem*')
            if usb_devices:
                # Found USB modem devices - could be TrueRNG
                return True

            return False
        except Exception as e:
            print(f"RNG detection error: {e}")
            return False

    def _load_rng_setting(self):
        """Load RNG setting from config."""
        config_file = Path.home() / ".noodlestudio" / "settings.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('rng_source', 'internal')
            except:
                pass
        return 'internal'

    def _save_rng_setting(self, rng_text, dialog):
        """Save RNG setting to config."""
        rng_source = 'truerng' if 'TrueRNG' in rng_text else 'internal'

        config_dir = Path.home() / ".noodlestudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "settings.json"

        settings = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
            except:
                pass

        settings['rng_source'] = rng_source

        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=2)

        # Show appropriate status message
        if rng_source == 'truerng':
            message = "Hardware RNG detected and activated - True quantum randomness enabled"
        else:
            message = "Using internal RNG - Deterministic pseudorandom output"

        self.statusBar().showMessage(message, 5000)
        dialog.accept()

    def show_startup_rng_status(self):
        """Show RNG status message on startup."""
        ubild_available = self._check_ubild_connected()
        current_rng = self._load_rng_setting()

        if ubild_available and current_rng == 'truerng':
            message = "Hardware RNG detected - True quantum randomness enabled"
        else:
            message = "No RNG detected. Falling back to internal RNG. Outputs are deterministic. Consider an avalanche effect RNG for quantum non-determinism"

        self.statusBar().showMessage(message, 8000)

    def show_external_apps_settings(self):
        """Show External Applications settings dialog."""
        from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QLabel, QLineEdit,
                                    QPushButton, QHBoxLayout, QGroupBox, QFormLayout)

        dialog = QDialog(self)
        dialog.setWindowTitle("External Applications")
        dialog.resize(600, 400)

        layout = QVBoxLayout(dialog)

        # Header
        header = QLabel("Configure external applications for opening files:")
        header.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 10px;")
        layout.addWidget(header)

        # Load current settings
        settings = self._load_external_apps_settings()

        # Store field references
        self.app_fields = {}

        # Text Editor
        text_group = QGroupBox("Text Editor")
        text_layout = QHBoxLayout()
        text_field = QLineEdit(settings.get('text_editor', ''))
        text_field.setPlaceholderText("/Applications/Visual Studio Code.app")
        text_layout.addWidget(text_field)
        text_btn = QPushButton("Browse...")
        text_btn.clicked.connect(lambda: self._browse_application(text_field))
        text_layout.addWidget(text_btn)
        text_group.setLayout(text_layout)
        layout.addWidget(text_group)
        self.app_fields['text_editor'] = text_field

        # Image Editor
        image_group = QGroupBox("Image Editor")
        image_layout = QHBoxLayout()
        image_field = QLineEdit(settings.get('image_editor', ''))
        image_field.setPlaceholderText("/Applications/Photoshop.app")
        image_layout.addWidget(image_field)
        image_btn = QPushButton("Browse...")
        image_btn.clicked.connect(lambda: self._browse_application(image_field))
        image_layout.addWidget(image_btn)
        image_group.setLayout(image_layout)
        layout.addWidget(image_group)
        self.app_fields['image_editor'] = image_field

        # Audio Editor
        audio_group = QGroupBox("Audio Editor")
        audio_layout = QHBoxLayout()
        audio_field = QLineEdit(settings.get('audio_editor', ''))
        audio_field.setPlaceholderText("/Applications/Audacity.app")
        audio_layout.addWidget(audio_field)
        audio_btn = QPushButton("Browse...")
        audio_btn.clicked.connect(lambda: self._browse_application(audio_field))
        audio_layout.addWidget(audio_btn)
        audio_group.setLayout(audio_layout)
        layout.addWidget(audio_group)
        self.app_fields['audio_editor'] = audio_field

        # 3D Tool
        threed_group = QGroupBox("3D Tool")
        threed_layout = QHBoxLayout()
        threed_field = QLineEdit(settings.get('threed_tool', ''))
        threed_field.setPlaceholderText("/Applications/Blender.app")
        threed_layout.addWidget(threed_field)
        threed_btn = QPushButton("Browse...")
        threed_btn.clicked.connect(lambda: self._browse_application(threed_field))
        threed_layout.addWidget(threed_btn)
        threed_group.setLayout(threed_layout)
        layout.addWidget(threed_group)
        self.app_fields['threed_tool'] = threed_field

        layout.addStretch()

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(dialog.reject)
        button_layout.addWidget(cancel_btn)

        save_btn = QPushButton("Save")
        save_btn.setDefault(True)
        save_btn.clicked.connect(lambda: self._save_external_apps_settings(dialog))
        button_layout.addWidget(save_btn)

        layout.addLayout(button_layout)

        dialog.exec()

    def _browse_application(self, line_edit):
        """Browse for application file."""
        from PyQt6.QtWidgets import QFileDialog

        # Start in Applications folder on macOS
        start_dir = "/Applications" if os.path.exists("/Applications") else str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Application",
            start_dir,
            "Applications (*.app);;All Files (*)"
        )

        if file_path:
            line_edit.setText(file_path)

    def _load_external_apps_settings(self):
        """Load external apps settings from config."""
        config_file = Path.home() / ".noodlestudio" / "settings.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                    return settings.get('external_apps', {})
            except:
                pass
        return {}

    def _save_external_apps_settings(self, dialog):
        """Save external apps settings to config."""
        config_dir = Path.home() / ".noodlestudio"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_file = config_dir / "settings.json"

        settings = {}
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
            except:
                pass

        external_apps = {}
        for key, field in self.app_fields.items():
            if field.text().strip():
                external_apps[key] = field.text().strip()

        settings['external_apps'] = external_apps

        with open(config_file, 'w') as f:
            json.dump(settings, f, indent=2)

        self.statusBar().showMessage("External applications saved", 3000)
        dialog.accept()

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
