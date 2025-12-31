"""
Neural Canvas Node Operations Mixin - Node add/delete operations

Contains:
- start_add_node_mode: Enter add node mode
- _add_node_from_menu: Add node from context menu
- _delete_selected_nodes: Delete selected nodes with undo
- _add_node_at_position: Add node at position with undo

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtCore import Qt, QPointF

from ...core.neural_canvas.neural_node import NodeType
from ...core.neural_canvas.node_definitions import create_node_from_type


class NeuralCanvasNodeOpsMixin:
    """Mixin providing node operations for NeuralCanvasView."""

    def start_add_node_mode(self, node_type: NodeType):
        """Enter mode to add a node of the given type."""
        self.add_node_mode = True
        self.add_node_type = node_type
        self.setCursor(Qt.CursorShape.CrossCursor)

    def _add_node_from_menu(self, node_type: NodeType, scene_pos: QPointF):
        """Add node from context menu."""
        self.add_node_type = node_type
        self._add_node_at_position(scene_pos)

    def _delete_selected_nodes(self):
        """Delete selected nodes with undo support."""
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if not selected:
            return

        from ...core.undo_manager import undo_manager
        from ...core.commands import DeleteNeuralNodeCommand

        # Use macro for multiple deletions
        if len(selected) > 1:
            undo_manager.begin_group(f"Delete {len(selected)} Nodes")

        for node_item in selected:
            # Collect connections involving this node
            connections_data = []
            for conn in self.graph.connections:
                if conn.from_node == node_item.node.id or conn.to_node == node_item.node.id:
                    connections_data.append({
                        'from_node': conn.from_node,
                        'from_port': conn.from_port,
                        'to_node': conn.to_node,
                        'to_port': conn.to_port
                    })

            # Push delete command
            cmd = DeleteNeuralNodeCommand(
                view=self,
                node_data=node_item.node.to_dict(),
                connections_data=connections_data,
                node_name=node_item.node.name
            )
            undo_manager.push(cmd)

        if len(selected) > 1:
            undo_manager.end_group()

        print(f"[Neural Canvas] Deleted {len(selected)} node(s) (undoable)")

    def _add_node_at_position(self, scene_pos: QPointF):
        """Add a new node at the given position with undo support."""
        if not self.add_node_type:
            return

        # Create node data (command will add it to graph)
        node = create_node_from_type(self.add_node_type)
        node.position = (int(scene_pos.x()), int(scene_pos.y()))

        # Push create command
        from ...core.undo_manager import undo_manager
        from ...core.commands import CreateNeuralNodeCommand

        cmd = CreateNeuralNodeCommand(
            view=self,
            node_data=node.to_dict(),
            node_name=node.name
        )
        undo_manager.push(cmd)

        print(f"[Neural Canvas] Created node (undoable): {node.name}")
