"""
Neural Canvas Context Menu Mixin - Right-click context menu

Contains:
- contextMenuEvent: Build and show context menu

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction

from ...core.neural_canvas.neural_node import NodeType


class NeuralCanvasContextMenuMixin:
    """Mixin providing context menu for NeuralCanvasView."""

    def contextMenuEvent(self, event):
        """Handle right-click context menu (preserves selection)."""
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem, ConnectionGraphicsItem

        scene_pos = self.mapToScene(event.pos())
        items = self.scene.items(scene_pos)

        # Check if clicking on a node or connection
        clicked_node = None
        clicked_connection = None
        for item in items:
            if isinstance(item, NodeGraphicsItem):
                clicked_node = item
                break
            elif isinstance(item, ConnectionGraphicsItem):
                clicked_connection = item

        # Build context menu
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background: #2a2a2a;
                color: #ddd;
                border: 1px solid #555;
            }
            QMenu::item:selected {
                background: #4a4a4a;
            }
        """)

        # Connection-specific context menu (wire disconnect)
        if clicked_connection and not clicked_node:
            conn = clicked_connection.connection
            from_node = self.graph.nodes.get(conn.from_node)
            to_node = self.graph.nodes.get(conn.to_node)

            from_name = from_node.name if from_node else conn.from_node
            to_name = to_node.name if to_node else conn.to_node

            wire_label = f"{from_name}.{conn.from_port} -> {to_name}.{conn.to_port}"
            info_action = menu.addAction(f"Connection: {wire_label}")
            info_action.setEnabled(False)  # Just a label

            menu.addSeparator()

            delete_action = menu.addAction("Delete Connection")
            delete_action.triggered.connect(
                lambda: self.delete_connection_wire(clicked_connection)
            )

            menu.exec(event.globalPos())
            event.accept()
            return  # Early return for wire context menu

        # If clicking on empty space OR on a node, show full menu
        # (Blender shows menu everywhere, keeps selection intact)

        # Add Node submenu (only if clicking empty space)
        if not clicked_node:
            add_menu = menu.addMenu("Add")

            groups = {
                "Recurrent": [NodeType.LSTM, NodeType.GRU, NodeType.RNN],
                "Feedforward": [NodeType.LINEAR, NodeType.CONV1D],
                "Activation": [NodeType.TANH, NodeType.RELU, NodeType.SIGMOID],
                "Utility": [NodeType.STATE_CONCAT, NodeType.AFFECT_HEAD],
                "Quantum": [NodeType.QUANTUM_MICROTUBULE, NodeType.IBM_QUANTUM],
                "Audio": [NodeType.AUDIO_FILE, NodeType.AUDIO_TRIGGER, NodeType.OSCILLATOR, NodeType.AUDIO_OUTPUT],
                "Scripting": [NodeType.SCRIPTED_NODE],
                "Assets": [NodeType.CHECKPOINT],
                "Tutorial": [NodeType.NUMBER_INPUT, NodeType.THRESHOLD_OUTPUT, NodeType.CONCAT],
                "Annotation": [NodeType.COMMENT]
            }

            for group_name, node_types in groups.items():
                group_menu = add_menu.addMenu(f"{group_name}")
                for node_type in node_types:
                    action = QAction(node_type.value, self)
                    action.triggered.connect(lambda checked, nt=node_type, pos=scene_pos: self._add_node_from_menu(nt, pos))
                    group_menu.addAction(action)

            menu.addSeparator()

        # Layout submenu (always available)
        layout_menu = menu.addMenu("Layout")

        auto_arrange = QAction("Auto-Arrange (Topological)", self)
        auto_arrange.triggered.connect(self.auto_arrange_nodes)
        layout_menu.addAction(auto_arrange)

        layout_menu.addSeparator()

        align_h = QAction("Align Horizontally", self)
        align_h.triggered.connect(self.align_selected_horizontally)
        layout_menu.addAction(align_h)

        align_v = QAction("Align Vertically", self)
        align_v.triggered.connect(self.align_selected_vertically)
        layout_menu.addAction(align_v)

        # If clicking on selected node(s), add node-specific actions
        selected_nodes = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if selected_nodes:
            menu.addSeparator()
            delete_action = QAction(f"Delete ({len(selected_nodes)} node{'s' if len(selected_nodes) > 1 else ''})", self)
            delete_action.triggered.connect(self._delete_selected_nodes)
            menu.addAction(delete_action)

        menu.exec(event.globalPos())
        event.accept()  # Don't propagate to base class (prevents deselection)
