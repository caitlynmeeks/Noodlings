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
#   Main Window Menus Mixin - Menu bar setup
#
#   Contains: - _setup_menu_bar: Complete menu bar constructi...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_menus_mixin
# PURPOSE:  Main Window Menus Mixin - Menu bar setup
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowMenusMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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

        file_menu.addAction(self._create_action(
            "&Close Project", slot=self._close_project
        ))

        file_menu.addSeparator()

        # Create new stage (requires server)
        self.new_stage_action = self._create_action("New &Stage...", "Ctrl+Shift+N", slot=self.new_stage, enabled=False)
        file_menu.addAction(self.new_stage_action)
        self._server_dependent_actions.append(self.new_stage_action)

        file_menu.addSeparator()

        # Save
        file_menu.addAction(self._create_action("&Save Project", "Ctrl+S", slot=self.save_project))
        file_menu.addAction(self._create_action("Save Sta&ge", "Ctrl+Shift+S", slot=self.save_stage))

        # Import
        file_menu.addSeparator()
        file_menu.addAction(self._create_action("Import Noodling Folder...", slot=self.import_noodling_folder))

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

        # ===== VIEW MENU =====
        view_menu = menu_bar.addMenu("&View")
        # Layout is locked - panels always visible

        # ===== WINDOW MENU =====
        window_menu = menu_bar.addMenu("&Window")
        window_menu.addAction(self._create_action("Minimize", "Ctrl+M", self.showMinimized))
        window_menu.addAction(self._create_action("Zoom", slot=self.showMaximized))
        window_menu.addSeparator()

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
        """Create tool bar with Play/Stop button."""
        from PyQt6.QtWidgets import QPushButton
        from PyQt6.QtCore import QSize

        tool_bar = self.addToolBar("Main Toolbar")
        tool_bar.setObjectName("MainToolbar")
        tool_bar.setMovable(False)

        # Play/Stop toggle button
        self._play_button = QPushButton("Play")
        self._play_button.setCheckable(True)
        self._play_button.setFixedSize(QSize(64, 28))
        self._play_button.setStyleSheet(
            "QPushButton { background: #333; color: #ccc; border: 1px solid #555; "
            "border-radius: 4px; font-weight: bold; } "
            "QPushButton:checked { background: #4a7a4a; color: #eee; }"
        )
        self._play_button.toggled.connect(self._on_play_toggled)
        tool_bar.addWidget(self._play_button)

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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
