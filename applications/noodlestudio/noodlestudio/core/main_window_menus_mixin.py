"""
Main Window Menus Mixin - Menu bar setup

Contains:
- _setup_menu_bar: Complete menu bar construction
- _create_action: Helper for creating QActions
- _setup_tool_bar: Toolbar setup

Author: Noodlings Project
Date: December 2025
"""


class MainWindowMenusMixin:
    """Mixin providing menu bar setup for MainWindow."""

    def _setup_menu_bar(self):
        """Create menu bar."""
        menu_bar = self.menuBar()

        # Track server-dependent actions for enable/disable
        self._server_dependent_actions = []

        # File Menu
        file_menu = menu_bar.addMenu("&File")

        # Project management
        file_menu.addAction(self._create_action("&New Project...", slot=self.new_project))
        file_menu.addAction(self._create_action("&Open Project...", slot=self.open_project))

        # Recent Projects submenu
        self.recent_projects_menu = file_menu.addMenu("Recent Projects")
        self.update_recent_projects_menu()

        file_menu.addSeparator()

        # Create new stage (requires server)
        self.new_stage_action = self._create_action("New &Stage...", "Ctrl+Shift+N", slot=self.new_stage, enabled=False)
        file_menu.addAction(self.new_stage_action)
        self._server_dependent_actions.append(self.new_stage_action)

        file_menu.addSeparator()

        # Save
        file_menu.addAction(self._create_action("&Save Project", "Ctrl+S", slot=self.save_project))
        file_menu.addAction(self._create_action("Save Sta&ge", "Ctrl+Shift+S", slot=self.save_stage))

        # Import section
        file_menu.addSeparator()
        file_menu.addSection("Import")
        file_menu.addAction(self._create_action("Import Noodling Folder...", slot=self.import_noodling_folder))
        file_menu.addAction(self._create_action("Import USD Layer (.usda)...", slot=self.import_usd_layer))

        # Export section
        file_menu.addSeparator()
        file_menu.addSection("Export")
        file_menu.addAction(self._create_action("Export Noodling...", slot=self.export_noodling))
        file_menu.addAction(self._create_action("Export Stage to USD (.usda)...", slot=self.export_stage_to_usd))

        # Migration tool
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Migrate Legacy Data...", slot=self.migrate_legacy_data))

        # Developer tools
        file_menu.addSeparator()
        file_menu.addAction(self._create_action(
            "Soft Restart...",
            "Ctrl+Shift+R",
            slot=self._soft_restart
        ))

        file_menu.addSeparator()
        file_menu.addAction(self._create_action("&Quit", "Ctrl+Q", self.close))

        # ===== EDIT MENU (undo/redo) =====
        edit_menu = menu_bar.addMenu("&Edit")

        # Undo/Redo - created by UndoManager for auto-updating text
        from .undo_manager import undo_manager
        self.undo_action = undo_manager.create_undo_action(self, "Undo")
        self.undo_action.setShortcut("Ctrl+Z")
        edit_menu.addAction(self.undo_action)

        self.redo_action = undo_manager.create_redo_action(self, "Redo")
        self.redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addAction(self.redo_action)

        edit_menu.addSeparator()

        # Standard edit operations (for future use)
        edit_menu.addAction(self._create_action("Cu&t", "Ctrl+X"))
        edit_menu.addAction(self._create_action("&Copy", "Ctrl+C"))
        edit_menu.addAction(self._create_action("&Paste", "Ctrl+V"))
        edit_menu.addAction(self._create_action("&Delete", "Delete"))

        # ===== VIEW MENU =====
        view_menu = menu_bar.addMenu("&View")
        # Layout is locked - panels always visible

        # ===== COMPONENT MENU =====
        component_menu = menu_bar.addMenu("&Component")
        component_menu.addAction(self._create_action("Add Script...", slot=lambda: self.add_component("custom")))

        # ===== WINDOW MENU =====
        window_menu = menu_bar.addMenu("&Window")
        window_menu.addAction(self._create_action("Minimize", "Ctrl+M", self.showMinimized))
        window_menu.addAction(self._create_action("Zoom", slot=self.showMaximized))
        window_menu.addSeparator()

        # ===== ACCOUNT MENU =====
        account_menu = menu_bar.addMenu("&Account")
        account_menu.addAction(self._create_action("Sign In...", slot=self.show_login_dialog))
        account_menu.addAction(self._create_action("Account Info...", slot=self.show_account_info))
        account_menu.addSeparator()
        self.enter_world_action = self._create_action("Enter World...", "Ctrl+W", slot=self.enter_world)
        account_menu.addAction(self.enter_world_action)
        account_menu.addAction(self._create_action("Manage Avatars...", slot=self.manage_avatars))
        account_menu.addSeparator()
        account_menu.addAction(self._create_action("My Noodlings (Cloud)", slot=self.show_cloud_noodlings))
        account_menu.addSeparator()
        account_menu.addAction(self._create_action("Sign Out", slot=self.sign_out))

        # ===== HELP MENU =====
        help_menu = menu_bar.addMenu("&Help")
        help_menu.addAction(self._create_action("Scripting API Reference", "F1", slot=self.open_scripting_api))
        help_menu.addAction(self._create_action("Documentation...", slot=self.open_documentation))
        help_menu.addSeparator()
        help_menu.addAction(self._create_action("Report a Bug...", slot=self.show_bug_report_dialog))
        help_menu.addAction(self._create_action("View Known Issues...", slot=self.open_github_issues))
        help_menu.addSeparator()
        help_menu.addAction(self._create_action("About NoodleStudio", slot=self.show_about))

    def _setup_tool_bar(self):
        """Create tool bar."""
        tool_bar = self.addToolBar("Main Toolbar")
        tool_bar.setObjectName("MainToolbar")  # Required for saveState
        # Hide legacy buttons for now
        tool_bar.setVisible(False)

    def _create_action(
        self,
        text: str,
        shortcut: str = None,
        slot=None,
        checkable: bool = False,
        checked: bool = False,
        enabled: bool = True
    ):
        """
        Create a QAction with optional shortcut and slot.

        Args:
            text: Action text (menu item label)
            shortcut: Keyboard shortcut (e.g., "Ctrl+S")
            slot: Callable to connect to triggered signal
            checkable: Whether action is checkable
            checked: Initial checked state (if checkable)
            enabled: Whether action is enabled

        Returns:
            QAction configured with provided settings
        """
        from PyQt6.QtGui import QAction

        action = QAction(text, self)

        if shortcut:
            action.setShortcut(shortcut)

        if slot:
            action.triggered.connect(slot)

        action.setCheckable(checkable)
        action.setChecked(checked)
        action.setEnabled(enabled)

        return action

    def _soft_restart(self):
        """
        Perform a soft restart of NoodleStudio.

        Saves current state (project, tabs, selection) and restarts.
        Used to apply code changes that require restart.
        """
        from .soft_restart import request_soft_restart
        request_soft_restart(self, "Apply code changes")
