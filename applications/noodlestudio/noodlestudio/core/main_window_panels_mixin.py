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

import os
import sys

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
        from ..panels.settings_panel import SettingsPanel
        from ..panels.noodle_code_panel import NoodleCodePanel
        from .noodle_code_engine import NoodleCodeEngine

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

        # -- DEFERRED (MVP): Text View, Spatial View, Gaussian Viewer, UI Canvas
        self.web_view = None
        self.spatial_view = None

        self.gaussian_viewer = None

        # Unified Editor (replaces old Facets Editor + Neural Canvas tabs)
        from ..panels.editors import UnifiedEditorPanel
        self.unified_editor = UnifiedEditorPanel()
        center_tabs.addTab(self.unified_editor, "Assembly")

        # Settings tab
        self.settings_panel = SettingsPanel()
        center_tabs.addTab(self.settings_panel, "Settings")

        self.ui_canvas_editor = None

        # Keep reference to model manager for backward compatibility
        self.model_manager = self.settings_panel.get_model_manager_panel()

        # Store reference to center tabs
        self.center_tabs = center_tabs

        self.world_view = None

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

        # Default to Assembly tab (Unified Editor)
        for i in range(center_tabs.count()):
            if center_tabs.tabText(i) == "Assembly":
                center_tabs.setCurrentIndex(i)
                break

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

        self.hierarchy.entitySelected.connect(safe_console_select)
        self.hierarchy.entitySelected.connect(safe_facets_select)

        # Unified editor signal wiring (C.7)
        self.unified_editor.facetSelected.connect(
            lambda facet: self.inspector.load_facet(facet)
        )
        self.unified_editor.ncNodeSelected.connect(
            self._on_nc_depth_node_selected
        )
        self.unified_editor.ncParamChanged.connect(
            self._on_neural_canvas_param_changed
        )
        self.unified_editor.ncGraphLoaded.connect(
            self._on_neural_canvas_graph_loaded
        )

        # Inspector connections
        self.inspector.nameChanged.connect(self._on_inspector_name_changed)

        # Assets Panel connections
        self.assets.assetRenamed.connect(self._on_asset_renamed)
        self.assets.assetSelected.connect(self._on_asset_selected)

        # Hierarchy -> Performance Window sync (noodling selection)
        def safe_performance_select(entity_type, entity_data):
            try:
                self._on_entity_selected_for_performance(entity_type, entity_data)
            except Exception as e:
                import traceback
                print(f"[SAFE WRAPPER] ERROR in _on_entity_selected_for_performance: {e}")
                traceback.print_exc()

        self.hierarchy.entitySelected.connect(safe_performance_select)

    def _setup_annotation_overlay(self):
        """Setup the annotation overlay for screenshot debugging with Claude.

        The overlay is a top-level transparent window (not a child widget)
        so it can render above ALL windows including floating tool windows
        like the Guide Performance Window.

        IMPORTANT: The overlay starts HIDDEN. On macOS, top-level windows
        always receive input events from the WindowServer regardless of
        WA_TransparentForMouseEvents (that attribute only works for child
        widgets within a parent hierarchy). Showing the overlay in
        "passthrough" mode creates an invisible glass pane that blocks
        all input to the main window. The overlay is only shown when
        the user activates edit mode via Shift+Tab.
        """
        from .annotation_overlay import AnnotationOverlay

        self.annotation_overlay = AnnotationOverlay(None)  # Top-level, no parent

        # Top-level transparent window that floats above everything
        self.annotation_overlay.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool  # No taskbar entry
            | Qt.WindowType.NoDropShadowWindowHint  # No macOS outline on annotations
        )
        self.annotation_overlay.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground, True
        )
        self.annotation_overlay.setAttribute(
            Qt.WidgetAttribute.WA_NoSystemBackground, True
        )

        # Start HIDDEN -- only shown when user enters edit mode (Shift+Tab)
        self.annotation_overlay.visible_annotations = False
        self.annotation_overlay.edit_mode = False

        # Install event filter to track main window geometry changes
        self.installEventFilter(self)

    def eventFilter(self, obj, event):
        """Track main window geometry to keep annotation overlay positioned."""
        if obj == self and event.type() in (
            QEvent.Type.Resize, QEvent.Type.Move
        ):
            if hasattr(self, 'annotation_overlay') and self.annotation_overlay.isVisible():
                self.annotation_overlay.setGeometry(self.geometry())
        return super().eventFilter(obj, event)

    def _toggle_annotation_overlay(self):
        """Toggle annotation overlay edit mode (Shift+Tab).

        Two-state toggle: Hidden <-> Edit Mode.

        On macOS, top-level windows always receive input from the
        WindowServer regardless of WA_TransparentForMouseEvents, so
        a "view-only passthrough" state is not possible. The overlay
        must be hidden when not editing. Screenshots still composite
        annotations from memory even when the overlay is hidden.
        """
        if not hasattr(self, 'annotation_overlay'):
            return

        overlay = self.annotation_overlay

        if not overlay.isVisible():
            # Hidden -> Edit Mode: show overlay for annotation editing
            overlay.visible_annotations = True
            overlay.edit_mode = True
            overlay.setGeometry(self.geometry())
            overlay.show()
            overlay.raise_()
            overlay.activateWindow()
            overlay.setFocus()
            self.statusBar().showMessage(
                "Annotations: EDIT MODE (right-click for tools, Shift+Tab to close)", 3000
            )
        else:
            # Edit Mode -> Hidden: hide overlay to restore input
            overlay.edit_mode = False
            overlay.hide()
            # Keep visible_annotations True so screenshots still render them
            self.statusBar().showMessage(
                "Annotations: HIDDEN (Shift+Tab to edit)", 3000
            )

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

        # Cmd+Option+S - Screenshot for debugging with Claude
        # ApplicationShortcut so it works even when a floating window has focus
        screenshot_shortcut = QShortcut(QKeySequence("Ctrl+Alt+S"), self)
        screenshot_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        screenshot_shortcut.activated.connect(self._take_debug_screenshot)

        # Shift+Tab - Toggle annotation overlay for screenshot markup
        annotation_shortcut = QShortcut(QKeySequence("Shift+Tab"), self)
        annotation_shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
        annotation_shortcut.activated.connect(self._toggle_annotation_overlay)

    def _grab_with_floating_windows(self):
        """Grab main window and composite floating windows + annotations.

        Tool windows (like GuidePerformanceWindow) float above the main
        window as separate OS windows, so QWidget.grab() misses them.

        Compositing order:
        1. Main window (without annotation overlay)
        2. Floating tool windows at their screen-relative positions
        3. Annotation overlay ON TOP of everything (user markup)
        """
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtGui import QPainter, QPixmap
        from PyQt6.QtCore import Qt as QtConst

        overlay = getattr(self, 'annotation_overlay', None)
        overlay_was_visible = False

        # Temporarily hide annotation overlay so main grab excludes it
        if overlay and overlay.isVisible():
            overlay_was_visible = True
            overlay.hide()

        # 1. Grab main window (without annotations)
        pixmap = self.grab()
        main_geo = self.geometry()

        painter = QPainter(pixmap)

        # 2. Composite floating tool windows (skip annotation overlay — it goes last)
        for widget in QApplication.topLevelWidgets():
            if widget is self or not widget.isVisible() or widget.isMinimized():
                continue
            if widget is overlay:
                continue
            if widget.windowTitle() in ("Screenshot Saved",):
                continue

            child_geo = widget.geometry()
            rel_x = child_geo.x() - main_geo.x()
            rel_y = child_geo.y() - main_geo.y()

            child_pixmap = widget.grab()
            painter.drawPixmap(rel_x, rel_y, child_pixmap)

        # 3. Render annotations ON TOP of everything (from memory, even if
        #    the overlay is hidden -- user draws annotations then hides the
        #    overlay to work, but screenshots should still include them)
        if overlay and overlay.annotations:
            saved_visible = overlay.visible_annotations
            overlay.visible_annotations = True
            overlay.setGeometry(self.geometry())

            ann_pixmap = QPixmap(pixmap.size())
            ann_pixmap.fill(QtConst.GlobalColor.transparent)
            overlay.render(ann_pixmap)
            painter.drawPixmap(0, 0, ann_pixmap)

            overlay.visible_annotations = saved_visible

        painter.end()

        # Restore overlay visibility if it was showing (edit mode)
        if overlay_was_visible:
            overlay.show()
            overlay.raise_()

        return pixmap

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

            # Capture the main window + any floating tool windows
            pixmap = self._grab_with_floating_windows()

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

        # Always default to Assembly tab after layout restore
        center_tabs = getattr(self, 'center_tabs', None)
        if center_tabs:
            for i in range(center_tabs.count()):
                if center_tabs.tabText(i) == "Assembly":
                    center_tabs.setCurrentIndex(i)
                    break

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
