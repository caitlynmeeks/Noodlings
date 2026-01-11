# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Main Window Panels Mixin - Panel setup and layout management
#
#   Contains: - _setup_panels: Create the locked-down splitte...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_panels_mixin
# PURPOSE:  main window panels mixin panel UI
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MaximizableCenterTabs, MainWindowPanelsMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QTabWidget, QSplitter, QTabBar
)
from PyQt6.QtCore import Qt, QTimer, QUrl, QEvent


class MaximizableCenterTabs(QTabWidget):
    """
    QTabWidget with double-click header to maximize/restore the center panel.

    Double-clicking the tab bar toggles between normal view and maximized
    center pane (hides left, right, and bottom panels).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._maximized = False
        self._saved_sizes = {}
        self._main_splitter = None
        self._top_splitter = None

        # Install event filter on tab bar for double-click detection
        self.tabBar().installEventFilter(self)

    def set_splitters(self, main_splitter: QSplitter, top_splitter: QSplitter):
        """Set references to the splitters for maximize/restore."""
        self._main_splitter = main_splitter
        self._top_splitter = top_splitter

    def eventFilter(self, obj, event):
        """Detect double-click on tab bar."""
        if obj == self.tabBar() and event.type() == QEvent.Type.MouseButtonDblClick:
            self.toggle_maximize()
            return True
        return super().eventFilter(obj, event)

    def toggle_maximize(self):
        """Toggle between maximized and normal center panel view."""
        if not self._main_splitter or not self._top_splitter:
            return

        if self._maximized:
            # Restore saved sizes
            if 'main' in self._saved_sizes:
                self._main_splitter.setSizes(self._saved_sizes['main'])
            if 'top' in self._saved_sizes:
                self._top_splitter.setSizes(self._saved_sizes['top'])
            self._maximized = False
        else:
            # Save current sizes
            self._saved_sizes['main'] = self._main_splitter.sizes()
            self._saved_sizes['top'] = self._top_splitter.sizes()

            # Maximize center: give nearly all space to center, minimal to others
            total_h = sum(self._top_splitter.sizes())
            total_v = sum(self._main_splitter.sizes())

            # Left=0, Center=max, Right=0
            self._top_splitter.setSizes([0, total_h, 0])
            # Top=max, Bottom=0
            self._main_splitter.setSizes([total_v, 0])

            self._maximized = True

    @property
    def is_maximized(self) -> bool:
        return self._maximized


class MainWindowPanelsMixin:
    """Mixin providing panel setup for MainWindow."""

    def _setup_panels(self):
        """Create locked-down layout with fixed splitters (no dragging/docking)."""
        from ..panels.scene_hierarchy import SceneHierarchy
        from ..panels.assets_panel import AssetsPanel
        from ..panels.inspector_panel import InspectorPanel
        from ..panels.console_panel import ConsolePanel
        from ..panels.profiler_panel import ProfilerPanel
        from ..panels.cognitive_cycles_panel_v2 import CognitiveCyclesPanel
        from ..panels.gaussian_viewer_panel import GaussianViewerPanel
        from ..panels.settings_panel import SettingsPanel
        from ..panels.noodle_code_panel import NoodleCodePanel
        from .noodle_code_engine import NoodleCodeEngine
        from ..panels.ui_canvas_editor_panel import UICanvasEditorPanel

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
        # Note: Components palette removed - UI components now shown in Stage hierarchy
        # Users right-click on UI node to add components (Unity-style)

        # CENTER: Tabbed widget for World View + Facets Editor + etc.
        # Uses MaximizableCenterTabs for double-click maximize feature
        center_tabs = MaximizableCenterTabs()
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

        # Skip WebEngine in test environment (crashes during pytest)
        import os
        in_test_mode = "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules

        if in_test_mode:
            placeholder = QLabel("WebEngine disabled in test mode")
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setStyleSheet("color: #666; font-size: 12px;")
            world_layout.addWidget(placeholder)
            self.web_view = None
        else:
            try:
                from PyQt6.QtWebEngineWidgets import QWebEngineView
                self.web_view = QWebEngineView()
                self.web_view.setStyleSheet("background-color: #1a1a1a;")
                self.web_view.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
                self.web_view.setUrl(QUrl("http://localhost:8080"))
                world_layout.addWidget(self.web_view)
            except ImportError as e:
                print(f"[WebView] ImportError: {e}")
                placeholder = QLabel("WebEngine not available\nInstall: pip install PyQt6-WebEngine")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder.setStyleSheet("color: #999; font-size: 14px;")
                world_layout.addWidget(placeholder)
                self.web_view = None
            except Exception as e:
                print(f"[WebView] Unexpected error initializing WebEngine: {e}")
                import traceback
                traceback.print_exc()
                placeholder = QLabel(f"WebEngine error:\n{e}")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder.setStyleSheet("color: #FF6666; font-size: 12px;")
                placeholder.setWordWrap(True)
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

        # UI Canvas Editor tab
        self.ui_canvas_editor = UICanvasEditorPanel()
        self.ui_canvas_editor.set_project_manager(self.project_manager)
        center_tabs.addTab(self.ui_canvas_editor, "UI Canvas")

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

        # Noodle Code AI assistant panel - goes in CENTER pane (leftmost)
        from .model_label_manager import get_model_label_manager
        from .provider_manager import get_provider_manager
        from pathlib import Path
        self.noodle_code_panel = NoodleCodePanel(None)
        project_path = None
        if self.project_manager and self.project_manager.current_project_path:
            project_path = Path(self.project_manager.current_project_path)
        self.noodle_code_engine = NoodleCodeEngine(
            model_label_manager=get_model_label_manager(),
            provider_manager=get_provider_manager(),
            project_path=project_path
        )
        self.noodle_code_panel.set_engine(self.noodle_code_engine)
        center_tabs.insertTab(0, self.noodle_code_panel, "Noodle Code")  # Leftmost

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

        # Connect center_tabs to splitters for double-click maximize
        center_tabs.set_splitters(main_splitter, top_splitter)

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

        # Setup annotation overlay for screenshot debugging
        self._setup_annotation_overlay()

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

        # UI Canvas Editor connections
        self.ui_canvas_editor.component_selected.connect(self._on_ui_component_selected)

        # UI entity selection from Stage -> Canvas Editor
        def safe_ui_canvas_select(entity_type, entity_data):
            try:
                self._on_ui_entity_selected_for_canvas_editor(entity_type, entity_data)
            except Exception as e:
                import traceback
                print(f"[SAFE WRAPPER] ERROR in _on_ui_entity_selected_for_canvas_editor: {e}")
                traceback.print_exc()

        self.hierarchy.entitySelected.connect(safe_ui_canvas_select)

    def _setup_annotation_overlay(self):
        """Setup the annotation overlay for screenshot debugging with Claude."""
        from .annotation_overlay import AnnotationOverlay

        self.annotation_overlay = AnnotationOverlay(self)
        self.annotation_overlay.setGeometry(self.rect())
        self.annotation_overlay.show()
        self.annotation_overlay.raise_()

        # Start in passthrough mode (not intercepting clicks)
        self.annotation_overlay.toggle_edit_mode()  # Turns off edit mode

        # Install event filter to resize overlay with window
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Handle window resize to keep annotation overlay sized correctly."""
        if obj == self and event.type() == QEvent.Type.Resize:
            if hasattr(self, 'annotation_overlay'):
                self.annotation_overlay.setGeometry(self.rect())
        return super().eventFilter(obj, event)

    def _toggle_annotation_overlay(self):
        """Toggle annotation overlay visibility and edit mode (Shift+Tab)."""
        if not hasattr(self, 'annotation_overlay'):
            return

        # If annotations are hidden, show them and enter edit mode
        if not self.annotation_overlay.visible_annotations:
            self.annotation_overlay.visible_annotations = True
            self.annotation_overlay.edit_mode = True
            self.annotation_overlay.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, False
            )
            self.annotation_overlay.raise_()
            self.annotation_overlay.setFocus()
            self.statusBar().showMessage("Annotations: EDIT MODE (right-click for tools)", 3000)
        # If in edit mode, switch to view-only (passthrough)
        elif self.annotation_overlay.edit_mode:
            self.annotation_overlay.edit_mode = False
            self.annotation_overlay.setAttribute(
                Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
            )
            self.statusBar().showMessage("Annotations: VIEW ONLY (Shift+Tab to edit/hide)", 3000)
        # If view-only, hide annotations
        else:
            self.annotation_overlay.visible_annotations = False
            self.annotation_overlay.update()
            self.statusBar().showMessage("Annotations: HIDDEN (Shift+Tab to show)", 3000)

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

        # Cmd+Option+S - Screenshot for debugging with Claude
        screenshot_shortcut = QShortcut(QKeySequence("Ctrl+Alt+S"), self)
        screenshot_shortcut.activated.connect(self._take_debug_screenshot)

        # Shift+Tab - Toggle annotation overlay for screenshot markup
        annotation_shortcut = QShortcut(QKeySequence("Shift+Tab"), self)
        annotation_shortcut.activated.connect(self._toggle_annotation_overlay)

    def _take_debug_screenshot(self):
        """
        Capture a screenshot of the main window for debugging with Claude.

        Screenshots are saved to: ~/.noodlestudio/screenshots/
        Filename format: screenshot_YYYY-MM-DD_HH-MM-SS.png
        """
        from datetime import datetime
        from pathlib import Path

        try:
            # Create screenshots directory
            screenshots_dir = Path.home() / ".noodlestudio" / "screenshots"
            screenshots_dir.mkdir(parents=True, exist_ok=True)

            # Generate timestamp filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            filepath = screenshots_dir / filename

            # Capture the window using QWidget.grab() - no permissions needed
            pixmap = self.grab()

            if not pixmap.isNull() and pixmap.save(str(filepath)):
                print(f"[Screenshot] Saved: {filepath}")
                self._show_screenshot_confirmation(str(filepath), filename)
            else:
                self._show_screenshot_error("Failed to save screenshot")
                print(f"[Screenshot] ERROR: Failed to save to {filepath}")

        except Exception as e:
            self._show_screenshot_error(str(e))
            print(f"[Screenshot] ERROR: {e}")
            import traceback
            traceback.print_exc()

    def _show_screenshot_confirmation(self, filepath: str, filename: str):
        """Show a popup confirming screenshot was saved with copy button."""
        from PyQt6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
        )
        from PyQt6.QtCore import Qt

        dialog = QDialog(self)
        dialog.setWindowTitle("Screenshot Saved")
        dialog.setFixedWidth(420)
        dialog.setStyleSheet("""
            QDialog {
                background-color: #2D2D2D;
                border: 1px solid #555;
            }
            QLabel {
                color: #CCCCCC;
            }
            QLabel#title {
                color: #44FF88;
                font-size: 14px;
                font-weight: bold;
            }
            QLabel#path {
                color: #888888;
                font-size: 11px;
                font-family: monospace;
            }
            QPushButton {
                background-color: #3E3E3E;
                color: #CCCCCC;
                border: 1px solid #555;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #4E4E4E;
            }
            QPushButton#copy {
                background-color: #2D5A2D;
            }
            QPushButton#copy:hover {
                background-color: #3D6A3D;
            }
        """)

        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Title
        title = QLabel("Screenshot Saved")
        title.setObjectName("title")
        layout.addWidget(title)

        # Filename
        name_label = QLabel(f"File: {filename}")
        layout.addWidget(name_label)

        # Full path
        path_label = QLabel(filepath)
        path_label.setObjectName("path")
        path_label.setWordWrap(True)
        path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(path_label)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(8)

        copy_btn = QPushButton("Copy Path")
        copy_btn.setObjectName("copy")
        copy_btn.clicked.connect(lambda checked, btn=copy_btn: self._copy_to_clipboard(filepath, btn))
        button_layout.addWidget(copy_btn)

        open_btn = QPushButton("Open in Preview")
        open_btn.clicked.connect(lambda: self._open_in_default_viewer(filepath))
        button_layout.addWidget(open_btn)

        button_layout.addStretch()

        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(dialog.accept)
        ok_btn.setDefault(True)
        button_layout.addWidget(ok_btn)

        layout.addLayout(button_layout)

        # Auto-close after 5 seconds
        QTimer.singleShot(5000, dialog.accept)

        dialog.exec()

    def _copy_to_clipboard(self, text: str, copy_btn):
        """Copy text to clipboard and update button."""
        from PyQt6.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)
        # Update the button text
        try:
            copy_btn.setText("Copied!")
        except RuntimeError:
            pass  # Button may have been deleted

    def _open_in_default_viewer(self, filepath: str):
        """Open file in system default viewer (Preview.app on macOS)."""
        import subprocess
        import sys
        try:
            if sys.platform == 'darwin':  # macOS
                subprocess.run(['open', filepath])
            elif sys.platform == 'win32':  # Windows
                subprocess.run(['start', '', filepath], shell=True)
            else:  # Linux
                subprocess.run(['xdg-open', filepath])
        except Exception as e:
            print(f"[Screenshot] Error opening file: {e}")

    def _show_screenshot_error(self, message: str):
        """Show error popup for screenshot failure."""
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.warning(self, "Screenshot Failed", message)

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
