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
#   Neural Canvas Internal Mixin - Internal operations for undo/redo commands
#
#   Contains: - _set_node_position_internal: Set node positio...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.neural_canvas_internal_mixin
# PURPOSE:  Neural Canvas Internal Mixin
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralCanvasInternalMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import QGraphicsItem

from ...core.neural_canvas.neural_node import NeuralNode, Connection


class NeuralCanvasInternalMixin:
    """Mixin providing internal undo/redo operations for NeuralCanvasView."""

    def _set_node_position_internal(self, node_id: str, position: tuple):
        """
        Set node position without pushing undo command.

        Called by MoveNeuralNodeCommand during undo/redo.
        """
        # Update data model
        node = self.graph.nodes.get(node_id)
        if node:
            node.position = position

        # Update graphics
        node_item = self.node_items.get(node_id)
        if node_item:
            # Block geometry change signals to prevent recursion
            node_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, False)
            node_item.setPos(position[0], position[1])
            node_item.setFlag(QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges, True)

            # Force redraw of wires
            self.scene.update()

        self.graph_modified.emit()

    def _create_node_internal(self, node_data: dict):
        """
        Create node from serialized data without pushing undo command.

        Called by CreateNeuralNodeCommand.redo() and DeleteNeuralNodeCommand.undo().
        """
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        # Deserialize node
        node = NeuralNode.from_dict(node_data)

        # Add to graph
        self.graph.add_node(node)

        # Create graphics item
        item = NodeGraphicsItem(node)
        self.scene.addItem(item)
        self.node_items[node.id] = item

        # Re-render connections (in case restoring deleted node with connections)
        self._render_connections()
        self.graph_modified.emit()

    def _delete_node_internal(self, node_id: str):
        """
        Delete node by ID without pushing undo command.

        Called by DeleteNeuralNodeCommand.redo() and CreateNeuralNodeCommand.undo().
        """
        # Remove from graph (this also removes connections)
        self.graph.remove_node(node_id)

        # Remove graphics item
        node_item = self.node_items.get(node_id)
        if node_item:
            self.scene.removeItem(node_item)
            del self.node_items[node_id]

        # Re-render to update connections
        self._render_connections()
        self.graph_modified.emit()

    def _create_connection_internal(self, conn_data: dict):
        """
        Create connection from data without pushing undo command.

        Called by CreateNeuralConnectionCommand.redo() and DeleteNeuralConnectionCommand.undo().
        """
        # Create connection
        conn = Connection(
            from_node=conn_data['from_node'],
            from_port=conn_data['from_port'],
            to_node=conn_data['to_node'],
            to_port=conn_data['to_port']
        )

        # Add to graph
        self.graph.connections.append(conn)

        # Re-render connections
        self._render_connections()
        self.graph_modified.emit()

    def _delete_connection_internal(self, from_node: str, from_port: str,
                                    to_node: str, to_port: str):
        """
        Delete connection without pushing undo command.

        Called by DeleteNeuralConnectionCommand.redo() and CreateNeuralConnectionCommand.undo().
        """
        # Remove from graph
        self.graph.connections = [
            c for c in self.graph.connections
            if not (c.from_node == from_node and c.from_port == from_port and
                    c.to_node == to_node and c.to_port == to_port)
        ]

        # Re-render connections
        self._render_connections()
        self.graph_modified.emit()

    def delete_connection_wire(self, conn_item: 'ConnectionGraphicsItem'):
        """
        Delete a connection wire via context menu.

        Uses undo command for proper undo/redo support.

        Args:
            conn_item: The ConnectionGraphicsItem to delete
        """
        if not conn_item or not self.graph:
            return

        conn = conn_item.connection

        # Push undo command
        from ...core.undo_manager import undo_manager
        from ...core.commands import DeleteNeuralConnectionCommand

        cmd = DeleteNeuralConnectionCommand(
            self, conn.from_node, conn.from_port, conn.to_node, conn.to_port
        )
        undo_manager.push(cmd)

    def _render_connections(self):
        """Re-render all connection graphics."""
        # Import here to avoid circular imports
        from .neural_canvas_view import ConnectionGraphicsItem

        # Remove existing connection items
        for item in list(self.scene.items()):
            if isinstance(item, ConnectionGraphicsItem):
                self.scene.removeItem(item)

        # Create new connection items
        for conn in self.graph.connections:
            from_item = self.node_items.get(conn.from_node)
            to_item = self.node_items.get(conn.to_node)

            if from_item and to_item:
                conn_item = ConnectionGraphicsItem(conn, from_item, to_item)
                self.scene.addItem(conn_item)

    def _rename_node_internal(self, node_id: str, new_name: str):
        """
        Rename node without pushing undo command.

        Called by RenameNeuralNodeCommand during undo/redo.
        """
        # Update data model
        node = self.graph.nodes.get(node_id)
        if node:
            node.name = new_name

        # Update graphics
        node_item = self.node_items.get(node_id)
        if node_item:
            node_item.node.name = new_name
            node_item.update()  # Trigger repaint

        self.graph_modified.emit()

    def _set_node_param_internal(self, node_id: str, param_name: str, value):
        """
        Set a node parameter without pushing undo command.

        Called by EditNeuralNodeParamCommand during undo/redo.
        """
        from ...core.neural_canvas.neural_node import NodeType

        # Update data model
        node = self.graph.nodes.get(node_id)
        if node:
            node.params[param_name] = value

        # Update graphics
        node_item = self.node_items.get(node_id)
        if node_item:
            node_item.node.params[param_name] = value
            # Update visual dimensions for COMMENT width/height
            if param_name == 'width' and hasattr(node_item, 'width'):
                node_item.prepareGeometryChange()
                node_item.width = value
            elif param_name == 'height' and hasattr(node_item, 'height'):
                node_item.prepareGeometryChange()
                node_item.height = value
            node_item.update()  # Trigger repaint

        self.graph_modified.emit()

        # Emit param changed signal for auto-run
        self.node_param_changed.emit(node_id, param_name, value)

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
