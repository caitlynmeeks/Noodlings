"""
Base Command Classes for NoodleStudio Undo System

Provides base classes for all undoable commands with:
- Consistent logging
- Command IDs for merging
- Clean state tracking

Author: Commander Spock + Cadet Caity
Date: December 15, 2025
"""

from typing import Optional
from PyQt6.QtGui import QUndoCommand


class StudioCommand(QUndoCommand):
    """
    Base class for all NoodleStudio undo commands.

    Provides:
    - Debug logging (set DEBUG = True to enable)
    - Consistent command naming
    - Reference counting for first-run detection

    Subclasses must implement:
    - undo(): Reverse the operation
    - redo(): Perform/re-perform the operation
    """

    DEBUG = False  # Set True for verbose logging

    def __init__(self, text: str, parent: Optional[QUndoCommand] = None):
        """
        Initialize command with description text.

        Args:
            text: Human-readable description (e.g., "Move 'Intuition'")
            parent: Parent command for grouping (optional)
        """
        super().__init__(text, parent)
        self._first_redo = True  # Track if this is initial execution

    def redo(self):
        """
        Perform the command.

        Note: redo() is called immediately when command is pushed.
        Override _do() in subclasses for the actual operation.
        """
        if self.DEBUG:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            action = "Execute" if self._first_redo else "Redo"
            print(f"[{timestamp}] [StudioCommand.redo] {action}: {self.text()}")

        self._do()
        self._first_redo = False

        if self.DEBUG:
            print(f"[{timestamp}] [StudioCommand.redo] {action} complete")

    def undo(self):
        """
        Reverse the command.

        Override _undo() in subclasses for the actual operation.
        """
        if self.DEBUG:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            print(f"[{timestamp}] [StudioCommand.undo] Undo: {self.text()}")

        self._undo()

        if self.DEBUG:
            print(f"[{timestamp}] [StudioCommand.undo] Undo complete")

    def _do(self):
        """
        Implement the actual operation (called by redo).

        Subclasses should override this method.
        """
        raise NotImplementedError("Subclass must implement _do()")

    def _undo(self):
        """
        Implement the undo operation.

        Subclasses should override this method.
        """
        raise NotImplementedError("Subclass must implement _undo()")

    def is_first_redo(self) -> bool:
        """Check if this is the first execution (not a redo)."""
        return self._first_redo


class MergeableCommand(StudioCommand):
    """
    Base class for commands that can merge with subsequent commands.

    Used for operations like dragging where many small moves should
    combine into a single undo operation.

    Subclasses must:
    - Set a unique merge_id in __init__
    - Implement mergeWith() to combine with compatible commands
    - Call setObsolete(True) if merge makes this command redundant
    """

    # Unique ID for command type (used by Qt's merge system)
    # Subclasses should override with their own unique int
    COMMAND_ID = -1

    def __init__(self, text: str, merge_id: str, parent: Optional[QUndoCommand] = None):
        """
        Initialize mergeable command.

        Args:
            text: Human-readable description
            merge_id: Unique identifier for merge grouping (e.g., facet_id)
            parent: Parent command for grouping
        """
        super().__init__(text, parent)
        self.merge_id = merge_id

    def id(self) -> int:
        """
        Return command type ID for merge compatibility.

        Qt only attempts to merge commands with the same id().
        Returns -1 to disable merging (override in subclass).
        """
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Attempt to merge with another command.

        Called by Qt when a new command of the same type is pushed.
        Return True if merged successfully, False otherwise.

        Subclasses should:
        1. Check if other is the same command type
        2. Check if merge_id matches
        3. Update self with combined state
        4. Return True

        Example:
            if not isinstance(other, MoveCommand):
                return False
            if other.merge_id != self.merge_id:
                return False
            self.new_pos = other.new_pos
            return True
        """
        return False


# Command IDs for different command types (used for merge compatibility)
# Each command type needs a unique ID > 0 for merging to work
class CommandID:
    """Unique command type IDs for merge system."""
    MOVE_FACET = 1
    MOVE_NEURAL_NODE = 2
    CHANGE_PROPERTY = 3
    EDIT_NEURAL_PARAM = 4
    EDIT_FACET_PROPERTY = 5
    # Add more as needed
