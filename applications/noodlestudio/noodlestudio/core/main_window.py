"""
Main Window for NoodleSTUDIO.

The primary application window with menu bar, toolbar, dock area, and status bar.

This file uses mixins for organization:
- MainWindowMenusMixin: Menu bar and toolbar setup
- MainWindowStatusBarMixin: Status bar with avatar dropdown
- MainWindowServerMixin: Server management and connection status
- MainWindowPanelsMixin: Panel layout and shortcuts
- MainWindowProjectMixin: Project management and recent projects
- MainWindowEntitiesMixin: Entity creation (noodlings, objects, etc.)
- MainWindowAccountMixin: Account management and world entry
- MainWindowSignalsMixin: Panel signal handlers
- MainWindowSettingsMixin: Settings, RNG, help, and misc handlers
"""

from typing import Optional

from PyQt6.QtWidgets import QMainWindow, QWidget
from PyQt6.QtCore import QTimer

from .unity_theme import UNITY_DARK_THEME
from .layout_manager import LayoutManager
from .project_manager import ProjectManager

# Import all mixins
from .main_window_menus_mixin import MainWindowMenusMixin
from .main_window_statusbar_mixin import MainWindowStatusBarMixin
from .main_window_server_mixin import MainWindowServerMixin
from .main_window_panels_mixin import MainWindowPanelsMixin
from .main_window_project_mixin import MainWindowProjectMixin
from .main_window_entities_mixin import MainWindowEntitiesMixin
from .main_window_account_mixin import MainWindowAccountMixin
from .main_window_signals_mixin import MainWindowSignalsMixin
from .main_window_settings_mixin import MainWindowSettingsMixin


class MainWindow(
    MainWindowMenusMixin,
    MainWindowStatusBarMixin,
    MainWindowServerMixin,
    MainWindowPanelsMixin,
    MainWindowProjectMixin,
    MainWindowEntitiesMixin,
    MainWindowAccountMixin,
    MainWindowSignalsMixin,
    MainWindowSettingsMixin,
    QMainWindow
):
    """
    Main application window for NoodleSTUDIO.

    Contains:
    - Menu bar (File, Edit, Rez, View, Entities, Component, Window, Settings, Account, Help)
    - Tool bar (quick actions)
    - Dockable panel area (Stage View, Assets, World View, Facets Editor, etc.)
    - Status bar with avatar dropdown and server toggle
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("NoodleSTUDIO - Noodlings IDE")
        self.resize(1400, 900)

        # Apply dark theme with darker gray background
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

        # Goose system (The Origin)
        from ..widgets.goose_widget import KonamiCodeDetector
        self.konami_detector = KonamiCodeDetector()
        self.konami_detector.goose_summoned.connect(self._summon_goose)
        self.goose_active = False

        # Setup UI components (from mixins)
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

        # Start cmush activity bridge for LLM visualization
        QTimer.singleShot(600, self._start_activity_bridge)

        # Auto-start MUSH server if setting is enabled
        QTimer.singleShot(700, self._check_autostart_mush)

        # Initialize Computer Use controller for Claude integration
        QTimer.singleShot(800, self._init_computer_use)

    def _init_computer_use(self):
        """Initialize Computer Use controller for Claude to see and interact with UI."""
        from .computer_use_controller import get_computer_use_controller
        controller = get_computer_use_controller()
        controller.set_main_window(self)

    def _setup_ui(self):
        """Build UI components."""
        # Central widget will be World View (main viewport)
        # Actual setup is done in _setup_panels() from mixin
        pass
