"""
Neural Canvas Commands - Undo commands for Neural Canvas operations

Commands for:
- Moving neural nodes (with drag merging)
- Creating/deleting neural nodes
- Creating/deleting connections

Author: Commander Spock + Cadet Caity
Date: December 15, 2025
"""

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from PyQt6.QtGui import QUndoCommand

from .base_command import StudioCommand, MergeableCommand, CommandID

if TYPE_CHECKING:
    from ...panels.neural_canvas.neural_canvas_view import NeuralCanvasView


class MoveNeuralNodeCommand(MergeableCommand):
    """
    Command for moving a neural network node.

    Supports merging consecutive moves (drag = 1 undo operation).
    """

    COMMAND_ID = CommandID.MOVE_NEURAL_NODE

    def __init__(
        self,
        view: 'NeuralCanvasView',
        node_id: str,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
        node_name: str = ""
    ):
        """
        Initialize move command.

        Args:
            view: Reference to NeuralCanvasView
            node_id: UUID of node being moved
            old_pos: Position before move (x, y)
            new_pos: Position after move (x, y)
            node_name: Display name for undo text
        """
        text = f"Move '{node_name}'" if node_name else "Move Node"
        super().__init__(text, merge_id=node_id)

        self.view = view
        self.node_id = node_id
        self.old_pos = old_pos
        self.new_pos = new_pos

    def _do(self):
        """Move node to new position."""
        # On first execution (when pushed), node is already at new_pos
        # Just save to disk. On re-redo, actually move the node.
        if self._first_redo:
            # Just emit modified signal (auto-save handles the rest)
            self.view.graph_modified.emit()
        else:
            # Re-redo: actually move the node
            self.view._set_node_position_internal(self.node_id, self.new_pos)

    def _undo(self):
        """Move node back to old position."""
        self.view._set_node_position_internal(self.node_id, self.old_pos)

    def id(self) -> int:
        """Return command ID for merge compatibility."""
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """Merge consecutive moves of the same node."""
        if not isinstance(other, MoveNeuralNodeCommand):
            return False
        if other.node_id != self.node_id:
            return False

        # Merge: keep our old_pos, take their new_pos
        self.new_pos = other.new_pos
        return True


class CreateNeuralNodeCommand(StudioCommand):
    """Command for creating a new neural network node."""

    def __init__(
        self,
        view: 'NeuralCanvasView',
        node_data: Dict[str, Any],
        node_name: str = ""
    ):
        """
        Initialize create command.

        Args:
            view: Reference to NeuralCanvasView
            node_data: Serialized node data (from node.to_dict())
            node_name: Display name for undo text
        """
        text = f"Create '{node_name}'" if node_name else "Create Node"
        super().__init__(text)

        self.view = view
        self.node_data = node_data
        self.node_id = node_data.get('id', '')

    def _do(self):
        """Create the node."""
        self.view._create_node_internal(self.node_data)

    def _undo(self):
        """Delete the created node."""
        self.view._delete_node_internal(self.node_id)


class DeleteNeuralNodeCommand(StudioCommand):
    """Command for deleting a neural network node."""

    def __init__(
        self,
        view: 'NeuralCanvasView',
        node_data: Dict[str, Any],
        connections_data: list,
        node_name: str = ""
    ):
        """
        Initialize delete command.

        Args:
            view: Reference to NeuralCanvasView
            node_data: Serialized node data (for restoration)
            connections_data: List of connection dicts involving this node
            node_name: Display name for undo text
        """
        text = f"Delete '{node_name}'" if node_name else "Delete Node"
        super().__init__(text)

        self.view = view
        self.node_data = node_data
        self.node_id = node_data.get('id', '')
        self.connections_data = connections_data

    def _do(self):
        """Delete the node and its connections."""
        self.view._delete_node_internal(self.node_id)

    def _undo(self):
        """Restore the node and its connections."""
        self.view._create_node_internal(self.node_data)
        for conn_data in self.connections_data:
            self.view._create_connection_internal(conn_data)


class CreateNeuralConnectionCommand(StudioCommand):
    """Command for creating a connection between neural nodes."""

    def __init__(
        self,
        view: 'NeuralCanvasView',
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str
    ):
        """
        Initialize connection create command.

        Args:
            view: Reference to NeuralCanvasView
            from_node: Source node ID
            from_port: Source port name
            to_node: Destination node ID
            to_port: Destination port name
        """
        super().__init__("Create Connection")

        self.view = view
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port

    def _do(self):
        """Create the connection."""
        self.view._create_connection_internal({
            'from_node': self.from_node,
            'from_port': self.from_port,
            'to_node': self.to_node,
            'to_port': self.to_port
        })

    def _undo(self):
        """Delete the connection."""
        self.view._delete_connection_internal(
            self.from_node, self.from_port,
            self.to_node, self.to_port
        )


class DeleteNeuralConnectionCommand(StudioCommand):
    """Command for deleting a connection between neural nodes."""

    def __init__(
        self,
        view: 'NeuralCanvasView',
        from_node: str,
        from_port: str,
        to_node: str,
        to_port: str
    ):
        """
        Initialize connection delete command.

        Args:
            view: Reference to NeuralCanvasView
            from_node: Source node ID
            from_port: Source port name
            to_node: Destination node ID
            to_port: Destination port name
        """
        super().__init__("Delete Connection")

        self.view = view
        self.from_node = from_node
        self.from_port = from_port
        self.to_node = to_node
        self.to_port = to_port

    def _do(self):
        """Delete the connection."""
        self.view._delete_connection_internal(
            self.from_node, self.from_port,
            self.to_node, self.to_port
        )

    def _undo(self):
        """Restore the connection."""
        self.view._create_connection_internal({
            'from_node': self.from_node,
            'from_port': self.from_port,
            'to_node': self.to_node,
            'to_port': self.to_port
        })


class EditNeuralNodeParamCommand(MergeableCommand):
    """
    Command for editing a neural node parameter.

    Supports merging consecutive edits to the same param (e.g., slider dragging).
    """

    COMMAND_ID = CommandID.EDIT_NEURAL_PARAM

    def __init__(
        self,
        view: 'NeuralCanvasView',
        node_id: str,
        param_name: str,
        old_value: Any,
        new_value: Any,
        node_name: str = ""
    ):
        """
        Initialize param edit command.

        Args:
            view: Reference to NeuralCanvasView
            node_id: UUID of node being edited
            param_name: Name of the parameter being changed
            old_value: Value before change
            new_value: Value after change
            node_name: Display name for undo text
        """
        if node_name:
            text = f"Edit '{node_name}' {param_name}"
        else:
            text = f"Edit {param_name}"
        # Merge ID combines node_id and param_name so different params don't merge
        super().__init__(text, merge_id=f"{node_id}:{param_name}")

        self.view = view
        self.node_id = node_id
        self.param_name = param_name
        self.old_value = old_value
        self.new_value = new_value

    def _do(self):
        """Apply the new value."""
        if self._first_redo:
            # Value is already set, just emit modified
            self.view.graph_modified.emit()
        else:
            # Re-redo: actually set the value
            self.view._set_node_param_internal(self.node_id, self.param_name, self.new_value)

    def _undo(self):
        """Restore the old value."""
        self.view._set_node_param_internal(self.node_id, self.param_name, self.old_value)

    def id(self) -> int:
        """Return command ID for merge compatibility."""
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """Merge consecutive edits to the same param."""
        if not isinstance(other, EditNeuralNodeParamCommand):
            return False
        if other.node_id != self.node_id:
            return False
        if other.param_name != self.param_name:
            return False

        # Merge: keep our old_value, take their new_value
        self.new_value = other.new_value
        return True


class RenameNeuralNodeCommand(StudioCommand):
    """Command for renaming a neural network node."""

    def __init__(
        self,
        view: 'NeuralCanvasView',
        node_id: str,
        old_name: str,
        new_name: str
    ):
        """
        Initialize rename command.

        Args:
            view: Reference to NeuralCanvasView
            node_id: UUID of node being renamed
            old_name: Name before rename
            new_name: Name after rename
        """
        super().__init__(f"Rename '{old_name}' to '{new_name}'")

        self.view = view
        self.node_id = node_id
        self.old_name = old_name
        self.new_name = new_name

    def _do(self):
        """Apply the new name."""
        self.view._rename_node_internal(self.node_id, self.new_name)

    def _undo(self):
        """Restore the old name."""
        self.view._rename_node_internal(self.node_id, self.old_name)
