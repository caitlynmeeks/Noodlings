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
#   Undo Manager - Global undo/redo system for NoodleStudio
#
#   Provides a centralized QUndoStack that all editors can pu...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.undo_manager
# PURPOSE:  Undo Manager
# LAYER:    Studio / Core
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   UndoManager
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from typing import Optional, Callable
from PyQt6.QtGui import QUndoStack, QUndoCommand, QAction
from PyQt6.QtCore import QObject, pyqtSignal


class UndoManager(QObject):
    """
    Singleton manager for application-wide undo/redo.

    Usage:
        from noodlestudio.core.undo_manager import undo_manager

        # Push a command
        undo_manager.push(MoveFacetCommand(...))

        # Undo/redo
        undo_manager.undo()
        undo_manager.redo()

        # Check state
        if undo_manager.can_undo():
            print(f"Can undo: {undo_manager.undo_text()}")
    """

    # Signals for UI updates
    stack_changed = pyqtSignal()  # Emitted when stack state changes
    clean_changed = pyqtSignal(bool)  # Emitted when clean state changes

    _instance: Optional['UndoManager'] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        super().__init__()
        self._initialized = True

        # The central undo stack
        self._stack = QUndoStack(self)

        # Connect stack signals to our signals
        self._stack.canUndoChanged.connect(lambda _: self.stack_changed.emit())
        self._stack.canRedoChanged.connect(lambda _: self.stack_changed.emit())
        self._stack.indexChanged.connect(lambda _: self.stack_changed.emit())
        self._stack.cleanChanged.connect(self.clean_changed.emit)

        # Undo limit (0 = unlimited)
        self._stack.setUndoLimit(100)

        # Track active undo group
        self._group_active = False

    @property
    def stack(self) -> QUndoStack:
        """Direct access to QUndoStack for advanced usage."""
        return self._stack

    # ========== Core Operations ==========

    def push(self, command: QUndoCommand):
        """
        Push a command onto the undo stack.

        The command's redo() is called immediately.
        If a command with the same id() is on top, mergeWith() is attempted.
        """
        self._stack.push(command)

    def undo(self):
        """Undo the last command."""
        if self._stack.canUndo():
            self._stack.undo()

    def redo(self):
        """Redo the last undone command."""
        if self._stack.canRedo():
            self._stack.redo()

    def clear(self):
        """Clear all undo history."""
        self._stack.clear()

    # ========== State Queries ==========

    def can_undo(self) -> bool:
        """Check if undo is available."""
        return self._stack.canUndo()

    def can_redo(self) -> bool:
        """Check if redo is available."""
        return self._stack.canRedo()

    def undo_text(self) -> str:
        """Get description of next undo operation."""
        return self._stack.undoText()

    def redo_text(self) -> str:
        """Get description of next redo operation."""
        return self._stack.redoText()

    def is_clean(self) -> bool:
        """Check if stack is in clean state (no unsaved changes)."""
        return self._stack.isClean()

    def set_clean(self):
        """Mark current state as clean (after save)."""
        self._stack.setClean()

    def count(self) -> int:
        """Get number of commands on stack."""
        return self._stack.count()

    def index(self) -> int:
        """Get current position in stack."""
        return self._stack.index()

    # ========== Undo Groups (Macro Commands) ==========

    def begin_group(self, text: str):
        """
        Begin an undo group (macro).

        All commands pushed until end_group() are combined into one undo.
        Useful for complex operations that should undo as a single unit.

        Example:
            undo_manager.begin_group("Paste Nodes")
            for node in nodes:
                undo_manager.push(CreateFacetCommand(node))
            undo_manager.end_group()
        """
        if self._group_active:
            print("[UndoManager] Warning: begin_group called while group already active")
            return

        self._stack.beginMacro(text)
        self._group_active = True

    def end_group(self):
        """End the current undo group."""
        if not self._group_active:
            print("[UndoManager] Warning: end_group called with no active group")
            return

        self._stack.endMacro()
        self._group_active = False

    def is_group_active(self) -> bool:
        """Check if an undo group is currently active."""
        return self._group_active

    # ========== Qt Action Integration ==========

    def create_undo_action(self, parent, prefix: str = "Undo") -> QAction:
        """
        Create a QAction for undo that auto-updates.

        Args:
            parent: Parent widget for the action
            prefix: Text prefix (default "Undo")

        Returns:
            QAction that shows "Undo: <command name>" and auto-enables
        """
        return self._stack.createUndoAction(parent, prefix)

    def create_redo_action(self, parent, prefix: str = "Redo") -> QAction:
        """
        Create a QAction for redo that auto-updates.

        Args:
            parent: Parent widget for the action
            prefix: Text prefix (default "Redo")

        Returns:
            QAction that shows "Redo: <command name>" and auto-enables
        """
        return self._stack.createRedoAction(parent, prefix)


# Global singleton instance
undo_manager = UndoManager()

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
