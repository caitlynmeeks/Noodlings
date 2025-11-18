"""
Main Window for NoodleSTUDIO.

The primary application window with menu bar, toolbar, dock area, and status bar.
"""

from typing import Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QLabel, QMessageBox
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction

from ..panels.home_panel import HomePanel
from ..panels.chat_panel import ChatPanel
from ..panels.profiler_panel import ProfilerPanel
from ..panels.scene_hierarchy import SceneHierarchy
from ..panels.inspector_panel import InspectorPanel
from .theme import DARK_THEME
from .unity_theme import UNITY_DARK_THEME
from .layout_manager import LayoutManager


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

        # Try to load default layout
        self.layout_manager.load_layout(self, "Default")

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
        file_menu.addAction(self._create_action("&New Recipe...", "Ctrl+N"))
        file_menu.addAction(self._create_action("&Open Recipe...", "Ctrl+O"))
        file_menu.addAction(self._create_action("&Save Recipe", "Ctrl+S"))
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("&Quit", "Ctrl+Q", self.close))

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
        layout_submenu.addAction(self._create_action("Load Layout..."))
        layout_submenu.addSeparator()
        layout_submenu.addAction(self._create_action("Default (3-Panel Unity)", slot=lambda: self.load_layout("Default")))
        layout_submenu.addAction(self._create_action("Demo Mode (Timeline Focus)", slot=lambda: self.load_layout("Demo")))
        layout_submenu.addAction(self._create_action("Dev Mode (All Panels)", slot=lambda: self.load_layout("Dev")))

        # ===== NOODLINGS MENU =====
        noodlings_menu = menu_bar.addMenu("&Noodlings")
        noodlings_menu.addAction(self._create_action("Spawn Noodling...", "Ctrl+Shift+N"))
        noodlings_menu.addAction(self._create_action("Remove Noodling...", "Ctrl+Shift+R"))
        noodlings_menu.addSeparator()
        noodlings_menu.addAction(self._create_action("Toggle Enlightenment", "Ctrl+E"))
        noodlings_menu.addAction(self._create_action("Reset All States"))

        # ===== WINDOW MENU =====
        window_menu = menu_bar.addMenu("&Window")
        window_menu.addAction(self._create_action("Minimize", "Ctrl+M", self.showMinimized))
        window_menu.addAction(self._create_action("Zoom", slot=self.showMaximized))
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

        # Add actions (no icons for now)
        tool_bar.addAction(self._create_action("New", "Ctrl+N"))
        tool_bar.addAction(self._create_action("Open", "Ctrl+O"))
        tool_bar.addAction(self._create_action("Save", "Ctrl+S"))
        tool_bar.addSeparator()
        tool_bar.addAction(self._create_action("Spawn Agent", "Ctrl+Shift+N"))

    def _setup_status_bar(self):
        """Create status bar."""
        status_bar = self.statusBar()
        status_bar.showMessage("● Not connected to noodleMUSH")

    def _setup_panels(self):
        """Create Unity-style 3-panel layout."""

        # LEFT: Scene Hierarchy (like Unity's Hierarchy panel)
        self.hierarchy = SceneHierarchy(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.hierarchy)

        # CENTER: World View (noodleMUSH chat - like Unity's Scene view)
        self.world_view = ChatPanel(self)
        self.world_view.setWindowTitle("World View")
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.world_view)

        # RIGHT: Inspector (editable properties - like Unity's Inspector)
        self.inspector = InspectorPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.inspector)

        # Connect hierarchy selection to inspector
        self.hierarchy.entitySelected.connect(self.inspector.load_entity)

        # BOTTOM: Timeline Profiler (optional - can be toggled)
        self.profiler_panel = ProfilerPanel(self)
        self.profiler_panel.setWindowTitle("Timeline Profiler")
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.profiler_panel)
        self.profiler_panel.hide()  # Hidden by default, toggle with View menu

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
            self.statusBar().showMessage(f"Layout '{layout_name}' saved", 3000)

    def load_layout(self, layout_name: str):
        """Load saved layout."""
        if self.layout_manager.load_layout(self, layout_name):
            self.statusBar().showMessage(f"Layout '{layout_name}' loaded", 3000)
        else:
            self.statusBar().showMessage(f"Layout '{layout_name}' not found", 3000)

    def show_about(self):
        """Show About dialog."""
        QMessageBox.about(
            self,
            "About NoodleStudio",
            "NoodleStudio v1.0.0-alpha\n\n"
            "Unity-style IDE for Noodlings consciousness agents\n\n"
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
            "Professional IDE for Noodlings consciousness agents<br><br>"
            "<b>Consilience, Inc.</b><br>"
            "Founded by Caitlyn Meeks<br><br>"
            "\"Movies are out. Noodlings are in.\"<br><br>"
            "🚀 Built with PyQt6 & MLX"
        )
