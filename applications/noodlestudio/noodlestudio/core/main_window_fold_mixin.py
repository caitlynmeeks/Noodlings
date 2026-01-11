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
#   Main Window Fold Mixin
#
#   Adds fold/unfold functionality to MainWindow.
#   Provides the "View Project" button and Cmd+Shift+U shortcut.
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.main_window_fold_mixin
# PURPOSE:  Fold/Unfold Integration for MainWindow
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   MainWindowFoldMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

import logging
from typing import Optional

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QDialog
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtCore import Qt

logger = logging.getLogger(__name__)


class MainWindowFoldMixin:
    """
    Mixin providing fold/unfold functionality for MainWindow.

    Adds:
    - PanelFoldManager for animated transitions
    - ViewProjectButton shown in App Mode
    - Cmd+Shift+U keyboard shortcut to toggle
    - Editor access enforcement (allow/password/hidden)

    Call _setup_fold() after _setup_panels() in MainWindow.__init__.
    """

    def _setup_fold(self):
        """
        Set up fold/unfold functionality.

        Call this after _setup_panels() so splitters exist.
        """
        from .panel_fold_manager import PanelFoldManager
        from ..widgets.view_project_button import ViewProjectButton

        # Editor access settings (defaults to allow)
        self._editor_access = "allow"
        self._editor_password_hash: Optional[str] = None

        # Get splitters from panel setup
        # centralWidget is main_splitter, first child is top_splitter
        main_splitter = self.centralWidget()
        top_splitter = main_splitter.widget(0) if main_splitter else None

        if not main_splitter or not top_splitter:
            logger.warning("Could not find splitters for fold manager")
            return

        # Create fold manager
        self._fold_manager = PanelFoldManager(main_splitter, top_splitter, self)

        # Connect signals
        self._fold_manager.fold_complete.connect(self._on_fold_complete)
        self._fold_manager.unfold_complete.connect(self._on_unfold_complete)

        # Create View Project button (hidden by default in studio mode)
        self._view_project_button = ViewProjectButton(self, "View Project")
        self._view_project_button.clicked.connect(self._on_view_project_clicked)
        self._view_project_button.hide()

        # Set up keyboard shortcut: Cmd+Shift+U (Mac) / Ctrl+Shift+U (Win/Linux)
        self._fold_shortcut = QShortcut(
            QKeySequence("Ctrl+Shift+U"),
            self
        )
        self._fold_shortcut.activated.connect(self._toggle_fold)

        logger.debug("Fold manager initialized with Ctrl+Shift+U shortcut")

    def set_editor_access(
        self,
        access: str = "allow",
        password_hash: Optional[str] = None,
        keyboard_shortcut: str = "Ctrl+Shift+U"
    ):
        """
        Configure editor access restrictions.

        Call this after _setup_fold() to apply build settings.

        Args:
            access: "allow", "password", or "hidden"
            password_hash: SHA-256 hash if access is "password"
            keyboard_shortcut: Custom keyboard shortcut (default Ctrl+Shift+U)
        """
        self._editor_access = access
        self._editor_password_hash = password_hash

        if access == "hidden":
            # Disable keyboard shortcut
            if hasattr(self, '_fold_shortcut'):
                self._fold_shortcut.setEnabled(False)
            logger.info("Editor access: hidden (shortcut disabled)")

        elif access == "password":
            logger.info("Editor access: password protected")

        else:
            # Update keyboard shortcut if custom
            if hasattr(self, '_fold_shortcut') and keyboard_shortcut != "Ctrl+Shift+U":
                self._fold_shortcut.setKey(QKeySequence(keyboard_shortcut))
                logger.info(f"Editor access: allow (shortcut: {keyboard_shortcut})")
            else:
                logger.info("Editor access: allow")

    def _check_editor_access(self) -> bool:
        """
        Check if editor access is permitted.

        For "hidden" - always returns False
        For "password" - shows password dialog and returns result
        For "allow" - always returns True

        Returns:
            True if access should be granted
        """
        if self._editor_access == "hidden":
            logger.debug("Editor access denied: hidden")
            return False

        if self._editor_access == "password":
            if not self._editor_password_hash:
                logger.warning("Password protection enabled but no hash set")
                return True  # Fail open if misconfigured

            from ..dialogs.editor_password_dialog import EditorPasswordDialog

            dialog = EditorPasswordDialog(
                stored_hash=self._editor_password_hash,
                parent=self
            )

            if dialog.exec() == QDialog.DialogCode.Accepted:
                logger.info("Editor access granted via password")
                return True
            else:
                logger.info("Editor access denied: password not provided")
                return False

        # Default: allow
        return True

    def _position_view_project_button(self):
        """Position the View Project button at bottom center."""
        if not hasattr(self, '_view_project_button'):
            return

        button = self._view_project_button
        window_width = self.width()
        window_height = self.height()

        # Position at bottom center, 20px from bottom
        x = (window_width - button.width()) // 2
        y = window_height - button.height() - 20

        button.move(x, y)
        button.raise_()  # Ensure it's on top

    def resizeEvent(self, event):
        """Reposition button on window resize."""
        super().resizeEvent(event)
        self._position_view_project_button()

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def fold_panels(self, animated: bool = True):
        """
        Fold panels away (enter App Mode).

        Args:
            animated: Whether to animate the transition
        """
        if hasattr(self, '_fold_manager'):
            self._fold_manager.fold(animated)

    def unfold_panels(self, animated: bool = True):
        """
        Unfold panels (enter Studio Mode).

        Args:
            animated: Whether to animate the transition
        """
        if hasattr(self, '_fold_manager'):
            self._fold_manager.unfold(animated)

    def toggle_fold(self, animated: bool = True):
        """
        Toggle between App Mode and Studio Mode.

        Args:
            animated: Whether to animate the transition
        """
        if hasattr(self, '_fold_manager'):
            self._fold_manager.toggle(animated)

    def is_folded(self) -> bool:
        """Check if panels are currently folded."""
        if hasattr(self, '_fold_manager'):
            return self._fold_manager.is_folded
        return False

    def set_app_mode(self, enabled: bool, animated: bool = True):
        """
        Set App Mode state.

        Args:
            enabled: True for App Mode (folded), False for Studio Mode
            animated: Whether to animate the transition
        """
        if enabled:
            self.fold_panels(animated)
        else:
            self.unfold_panels(animated)

    # =========================================================================
    # CALLBACKS
    # =========================================================================

    def _toggle_fold(self):
        """Called by keyboard shortcut."""
        # Check access before toggling
        if self.is_folded() and not self._check_editor_access():
            return
        self.toggle_fold()

    def _on_view_project_clicked(self):
        """Called when View Project button is clicked."""
        # Check access before unfolding
        if not self._check_editor_access():
            return
        logger.info("View Project clicked - unfolding studio")
        self.unfold_panels()

    def _on_fold_complete(self):
        """Called when fold animation completes."""
        # Show View Project button (unless access is hidden)
        if hasattr(self, '_view_project_button'):
            if self._editor_access == "hidden":
                # Don't show button if editor access is hidden
                self._view_project_button.hide()
            else:
                self._position_view_project_button()
                self._view_project_button.fade_in()

        # Hide menu bar and status bar in App Mode (optional)
        # self.menuBar().hide()
        # self.statusBar().hide()

        logger.info("App Mode active")

    def _on_unfold_complete(self):
        """Called when unfold animation completes."""
        # Hide View Project button
        if hasattr(self, '_view_project_button'):
            self._view_project_button.fade_out()

        # Show menu bar and status bar
        # self.menuBar().show()
        # self.statusBar().show()

        logger.info("Studio Mode active")


# ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡ ~ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
