"""
Main Window Panels Mixin - Panel setup and layout management

Contains:
- _setup_panels: Create the locked-down splitter layout
- _setup_shortcuts: Keyboard shortcut setup
- Layout management methods (save, load, reset)

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QUrl


class MainWindowPanelsMixin:
    """Mixin providing panel setup for MainWindow."""

    def _setup_panels(self):
        """Create locked-down layout with fixed splitters (no dragging/docking)."""
        from ..panels.scene_hierarchy import SceneHierarchy
        from ..panels.assets_panel import AssetsPanel
        from ..panels.inspector_panel import InspectorPanel
        from ..panels.console_panel import ConsolePanel
        from ..panels.profiler_panel import ProfilerPanel
        from ..panels.cognitive_cycles_panel import CognitiveCyclesPanel
        from ..panels.gaussian_viewer_panel import GaussianViewerPanel
        from ..panels.settings_panel import SettingsPanel

        # LEFT COLUMN: Tabbed widget for Hierarchy + Assets
        left_tabs = QTabWidget()
        left_tabs.setTabPosition(QTabWidget.TabPosition.North)
        left_tabs.setMinimumWidth(150)
        left_tabs.setDocumentMode(True)
        left_tabs.setStyleSheet("""
            QTabWidget { background-color: #383838; }
            QTabWidget::pane { border: none; background-color: #3E3E3E; }
            QTabWidget::tab-bar { background-color: #383838; alignment: left; }
            QTabBar { background-color: #383838; }
            QTabBar::tab {
                background-color: #2D2D2D; color: #888888;
                padding: 6px 12px; border: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: #3E3E3E; color: #CCCCCC; }
        """)

        self.hierarchy = SceneHierarchy(None)
        self.hierarchy.set_project_manager(self.project_manager)
        self.assets = AssetsPanel(None)
        self.assets.set_project_manager(self.project_manager)
        self.assets.agentRezzed.connect(self.hierarchy.refresh_scene)

        # Connect GenerationsManager for AI-generated asset storage
        from .generations_manager import get_generations_manager
        self.assets.set_generations_manager(get_generations_manager())

        left_tabs.addTab(self.hierarchy, "Stage")
        left_tabs.addTab(self.assets, "Assets")

        # CENTER: Tabbed widget for World View + Facets Editor + etc.
        center_tabs = QTabWidget()
        center_tabs.setTabPosition(QTabWidget.TabPosition.North)
        center_tabs.setDocumentMode(True)
        center_tabs.setStyleSheet("""
            QTabWidget { background-color: #383838; }
            QTabWidget::pane { border: none; background: #383838; }
            QTabBar::tab {
                background: #3a3a3a; color: #888888;
                padding: 8px 16px; border: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background: #3E3E3E; color: #D2D2D2; }
        """)

        # World View tab (WebView)
        world_widget = QWidget()
        world_layout = QVBoxLayout(world_widget)
        world_layout.setContentsMargins(0, 0, 0, 0)
        world_layout.setSpacing(0)

        try:
            from PyQt6.QtWebEngineWidgets import QWebEngineView
            self.web_view = QWebEngineView()
            self.web_view.setStyleSheet("background-color: #1a1a1a;")
            self.web_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
            self.web_view.setUrl(QUrl("http://localhost:8080"))
            world_layout.addWidget(self.web_view)
        except ImportError:
            placeholder = QLabel("WebEngine not available\nInstall: pip install PyQt6-WebEngine")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #999; font-size: 14px;")
            world_layout.addWidget(placeholder)
            self.web_view = None

        center_tabs.addTab(world_widget, "Text View")

        # Spatial View tab
        from ..panels.spatial_view_panel import SpatialViewPanel
        self.spatial_view = SpatialViewPanel()
        self.spatial_view.set_project_manager(self.project_manager)
        self.spatial_view.zoneSelected.connect(self._on_zone_selected)
        center_tabs.addTab(self.spatial_view, "Spatial View")

        # Facets Editor tab
        from ..panels.facets_editor_panel import FacetsEditorPanel
        self.facets_editor = FacetsEditorPanel()
        center_tabs.addTab(self.facets_editor, "Facets Editor")

        # Neural Canvas tab
        from ..panels.neural_canvas import NeuralCanvasPanel
        self.neural_canvas = NeuralCanvasPanel()
        center_tabs.addTab(self.neural_canvas, "Neural Canvas")

        # Gaussian Viewer tab
        self.gaussian_viewer = GaussianViewerPanel()
        center_tabs.addTab(self.gaussian_viewer, "Gaussian Viewer")

        # Settings tab
        self.settings_panel = SettingsPanel()
        center_tabs.addTab(self.settings_panel, "Settings")

        # Keep reference to model manager for backward compatibility
        self.model_manager = self.settings_panel.get_model_manager_panel()

        # Store reference to center tabs
        self.center_tabs = center_tabs

        # Create WorldView stub for compatibility
        self.world_view = self._create_world_view_stub(self.web_view)

        # RIGHT COLUMN: Inspector
        right_tabs = QTabWidget()
        right_tabs.setTabPosition(QTabWidget.TabPosition.North)
        right_tabs.setMinimumWidth(200)
        right_tabs.setDocumentMode(True)
        right_tabs.setStyleSheet("""
            QTabWidget { background-color: #383838; }
            QTabWidget::pane { border: none; background-color: #3E3E3E; }
            QTabWidget::tab-bar { background-color: #383838; alignment: left; }
            QTabBar { background-color: #383838; }
            QTabBar::tab {
                background-color: #2D2D2D; color: #888888;
                padding: 6px 12px; border: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: #3E3E3E; color: #CCCCCC; }
        """)

        self.inspector = InspectorPanel(None)
        right_tabs.addTab(self.inspector, "Inspector")

        # BOTTOM: Console + Profiler
        bottom_tabs = QTabWidget()
        bottom_tabs.setTabPosition(QTabWidget.TabPosition.North)
        bottom_tabs.setMinimumHeight(100)
        bottom_tabs.setDocumentMode(True)
        bottom_tabs.setStyleSheet("""
            QTabWidget { background-color: #383838; }
            QTabWidget::pane { border: none; background-color: #3E3E3E; }
            QTabWidget::tab-bar { background-color: #383838; alignment: left; }
            QTabBar { background-color: #383838; }
            QTabBar::tab {
                background-color: #2D2D2D; color: #888888;
                padding: 6px 12px; border: none; margin-right: 2px;
            }
            QTabBar::tab:selected { background-color: #3E3E3E; color: #CCCCCC; }
        """)

        self.console = ConsolePanel(None)
        self.profiler_panel = ProfilerPanel(None)
        bottom_tabs.addTab(self.console, "Console")
        bottom_tabs.addTab(self.profiler_panel, "Timeline Profiler")

        # Cognitive Cycles panel
        self.cognitive_cycles = CognitiveCyclesPanel(None)
        bottom_tabs.addTab(self.cognitive_cycles, "Cognitive Cycles")

        # Create splitters
        top_splitter = QSplitter(Qt.Orientation.Horizontal)
        top_splitter.addWidget(left_tabs)
        top_splitter.addWidget(center_tabs)
        top_splitter.addWidget(right_tabs)
        top_splitter.setStretchFactor(0, 0)
        top_splitter.setStretchFactor(1, 1)
        top_splitter.setStretchFactor(2, 0)
        top_splitter.setSizes([250, 800, 280])
        top_splitter.setChildrenCollapsible(False)

        main_splitter = QSplitter(Qt.Orientation.Vertical)
        main_splitter.addWidget(top_splitter)
        main_splitter.addWidget(bottom_tabs)
        main_splitter.setStretchFactor(0, 1)
        main_splitter.setStretchFactor(1, 0)
        main_splitter.setSizes([600, 180])
        main_splitter.setChildrenCollapsible(False)

        # Style splitter handles
        splitter_style = """
            QSplitter::handle { background-color: #2a2a2a; }
            QSplitter::handle:hover { background-color: #555555; }
            QSplitter::handle:horizontal { width: 6px; }
            QSplitter::handle:vertical { height: 6px; }
        """
        main_splitter.setStyleSheet(splitter_style)
        top_splitter.setStyleSheet(splitter_style)

        # Set as central widget
        self.setCentralWidget(main_splitter)

        # Connect signals
        self._connect_panel_signals()

        # Check server state
        QTimer.singleShot(200, self.update_connection_status)

    def _create_world_view_stub(self, web_view):
        """Create a stub object for WorldView compatibility."""
        class WorldViewStub:
            def __init__(self, web_view):
                self.web_view = web_view

            def show(self):
                pass

            def hide(self):
                pass

            def isVisible(self):
                return True

            def raise_(self):
                pass

            def set_server_state(self, running):
                if not running:
                    self.show_offline_card()
                else:
                    if self.web_view:
                        self.web_view.setUrl(QUrl("http://localhost:8080"))

            def reload(self):
                if self.web_view:
                    self.web_view.reload()

            def toggle_maximize(self):
                pass

            def show_offline_card(self):
                """Show offline placeholder when server is not running."""
                if self.web_view:
                    offline_html = """
                    <html>
                    <head>
                        <style>
                            body {
                                background: #1a1a1a; color: #999;
                                font-family: -apple-system, BlinkMacSystemFont, sans-serif;
                                display: flex; align-items: center; justify-content: center;
                                height: 100vh; margin: 0;
                            }
                            .card {
                                text-align: center; padding: 40px;
                                background: #2d2d2d; border-radius: 8px; border: 2px solid #3e3e3e;
                            }
                            .icon { font-size: 64px; margin-bottom: 20px; }
                            h1 { color: #ccc; font-size: 24px; margin-bottom: 10px; }
                            p { color: #888; font-size: 14px; margin: 5px 0; }
                            .hint { margin-top: 20px; font-size: 12px; color: #666; }
                        </style>
                    </head>
                    <body>
                        <div class="card">
                            <div class="icon">-</div>
                            <h1>noodleMUSH Server Offline</h1>
                            <p>Please start the server to view the world</p>
                            <p class="hint">Toggle the server switch in the bottom right</p>
                        </div>
                    </body>
                    </html>
                    """
                    self.web_view.setHtml(offline_html)

        return WorldViewStub(web_view)

    def _connect_panel_signals(self):
        """Connect all panel signals with error handling wrappers."""
        def safe_inspector_load(entity_type, entity_data):
            try:
                self.inspector.load_entity(entity_type, entity_data)
            except Exception as e:
                import traceback
                print(f"[SAFE WRAPPER] ERROR in inspector.load_entity: {e}")
                traceback.print_exc()

        def safe_console_select(entity_type, entity_data):
            try:
                self.on_entity_selected_for_console(entity_type, entity_data)
            except Exception as e:
                import traceback
                print(f"[SAFE WRAPPER] ERROR in on_entity_selected_for_console: {e}")
                traceback.print_exc()

        def safe_facets_select(entity_type, entity_data):
            try:
                self.on_entity_selected_for_facets_editor(entity_type, entity_data)
            except Exception as e:
                import traceback
                print(f"[SAFE WRAPPER] ERROR in on_entity_selected_for_facets_editor: {e}")
                traceback.print_exc()

        self.hierarchy.entitySelected.connect(safe_inspector_load)
        self.facets_editor.facetSelected.connect(lambda facet: self.inspector.load_facet(facet))

        # Neural Canvas connections
        self.neural_canvas.node_selected.connect(self._on_neural_canvas_node_selected)
        self.neural_canvas.canvas_view.node_param_changed.connect(self._on_neural_canvas_param_changed)
        self.neural_canvas.graph_loaded.connect(self._on_neural_canvas_graph_loaded)

        self.hierarchy.entitySelected.connect(safe_console_select)
        self.hierarchy.entitySelected.connect(safe_facets_select)

        # Gaussian Viewer connections
        self.gaussian_viewer.radianceLoaded.connect(self._on_radiance_loaded)
        self.gaussian_viewer.meshImported.connect(self._on_mesh_imported)

        # Inspector connections
        self.inspector.nameChanged.connect(self._on_inspector_name_changed)

        # Assets Panel connections
        self.assets.assetRenamed.connect(self._on_asset_renamed)
        self.assets.assetSelected.connect(self._on_asset_selected)

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

        # Cmd/Ctrl+Shift+G - SUMMON THE GOOSE
        goose_shortcut = QShortcut(QKeySequence("Ctrl+Shift+G"), self)
        goose_shortcut.activated.connect(self._summon_goose)

    def _toggle_panel(self, panel):
        """Toggle panel visibility (show/hide)."""
        if panel.isVisible():
            panel.hide()
        else:
            panel.show()
            panel.raise_()

    def _toggle_panel_maximize(self, panel):
        """Toggle any dock panel between maximized and normal state."""
        if hasattr(panel, 'toggle_maximize'):
            panel.show()
            panel.toggle_maximize()
        else:
            panel.show()

    def reset_to_factory_layout(self):
        """Reset to factory default layout."""
        self.statusBar().showMessage("Layout is locked to optimal arrangement", 3000)

    def save_current_layout(self):
        """Save current panel layout."""
        from PyQt6.QtWidgets import QInputDialog
        layout_name, ok = QInputDialog.getText(
            self, "Save Layout", "Layout name:", text="New Layout"
        )
        if ok and layout_name:
            self.layout_manager.save_layout(self, layout_name)
            self.layout_manager.set_last_used_layout(layout_name)
            self.statusBar().showMessage(f"Layout '{layout_name}' saved", 3000)

    def set_current_as_default(self):
        """Save current layout as Default."""
        self.layout_manager.save_layout(self, "Default")
        self.layout_manager.set_last_used_layout("Default")
        self.statusBar().showMessage("Current layout saved as default", 3000)

    def load_layout_dialog(self):
        """Show dialog to select and load a saved layout."""
        from PyQt6.QtWidgets import QInputDialog, QMessageBox

        layouts = self.layout_manager.list_layouts()
        if not layouts:
            QMessageBox.information(
                self, "No Layouts",
                "No saved layouts found.\nSave one first with 'Save Current Layout...'"
            )
            return

        layout_name, ok = QInputDialog.getItem(
            self, "Load Layout", "Select layout to load:", layouts, 0, False
        )
        if ok and layout_name:
            self.load_layout(layout_name)

    def load_layout(self, layout_name: str):
        """Load saved layout."""
        from PyQt6.QtWidgets import QMessageBox
        try:
            if self.layout_manager.load_layout(self, layout_name):
                self.statusBar().showMessage(f"Layout '{layout_name}' loaded", 3000)
            else:
                QMessageBox.warning(
                    self, "Layout Not Found",
                    f"Layout '{layout_name}' not found."
                )
        except Exception as e:
            print(f"Error loading layout: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(
                self, "Layout Error",
                f"Failed to load layout '{layout_name}'.\n\nError: {str(e)}"
            )

    def load_last_used_layout(self):
        """Load the last used layout on startup."""
        last_layout = self.layout_manager.get_last_used_layout()
        if last_layout:
            print(f"Restoring last used layout: '{last_layout}'")
            success = self.layout_manager.load_layout(self, last_layout)
            if success:
                self.statusBar().showMessage(f"Restored layout: '{last_layout}'", 3000)
            else:
                print("Failed to restore last layout, using default panel arrangement")
        else:
            print("No last layout saved, using default panel arrangement")
