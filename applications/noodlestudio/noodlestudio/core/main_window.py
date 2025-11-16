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
from .theme import DARK_THEME


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

        # Apply dark theme
        self.setStyleSheet(DARK_THEME)

        self._setup_ui()
        self._setup_menu_bar()
        self._setup_tool_bar()
        self._setup_status_bar()
        self._setup_panels()

    def _setup_ui(self):
        """Build UI components."""
        # Central widget placeholder (will be removed once we add dock widgets properly)
        central = QWidget()
        layout = QVBoxLayout()

        welcome = QLabel(
            "🧠 NoodleSTUDIO v1.0.0-alpha\n\n"
            "Welcome to the Noodlings IDE!\n\n"
            "Check out the Home panel to get started →"
        )
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.setStyleSheet("font-size: 18px; color: #64b5f6;")

        layout.addWidget(welcome)
        central.setLayout(layout)
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

        # View Menu
        view_menu = menu_bar.addMenu("&View")

        # Add panel toggles
        self.home_panel_action = self._create_action("Home", "Ctrl+1", checkable=True)
        self.home_panel_action.setChecked(True)
        view_menu.addAction(self.home_panel_action)

        self.chat_panel_action = self._create_action("Chat View", "Ctrl+2", checkable=True)
        self.chat_panel_action.setChecked(True)
        view_menu.addAction(self.chat_panel_action)

        # Agent Menu
        agent_menu = menu_bar.addMenu("&Agent")
        agent_menu.addAction(self._create_action("Spawn Agent...", "Ctrl+Shift+N"))

        # Session Menu
        session_menu = menu_bar.addMenu("&Session")
        session_menu.addAction(self._create_action("Load Session..."))

        # Tools Menu
        tools_menu = menu_bar.addMenu("&Tools")
        tools_menu.addAction(self._create_action("Preferences...", slot=self._show_preferences))

        # Help Menu
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._create_action("Documentation", slot=self._show_docs))
        help_menu.addAction(self._create_action("About NoodleSTUDIO", slot=self._show_about))

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
        """Create and add dock panels."""
        # Home panel (left side)
        self.home_panel = HomePanel(self)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.home_panel)

        # Connect to toggle action
        self.home_panel_action.triggered.connect(
            lambda checked: self.home_panel.setVisible(checked)
        )
        self.home_panel.visibilityChanged.connect(
            lambda visible: self.home_panel_action.setChecked(visible)
        )

        # Chat panel (right side)
        self.chat_panel = ChatPanel(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.chat_panel)

        # Connect to toggle action
        self.chat_panel_action.triggered.connect(
            lambda checked: self.chat_panel.setVisible(checked)
        )
        self.chat_panel.visibilityChanged.connect(
            lambda visible: self.chat_panel_action.setChecked(visible)
        )

    def _create_action(
        self,
        text: str,
        shortcut: str = "",
        slot=None,
        checkable: bool = False
    ) -> QAction:
        """
        Create a QAction with text, shortcut, and optional slot.

        Args:
            text: Action text
            shortcut: Keyboard shortcut (e.g., "Ctrl+N")
            slot: Slot to connect to (optional)
            checkable: Whether action is checkable

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
        return action

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
