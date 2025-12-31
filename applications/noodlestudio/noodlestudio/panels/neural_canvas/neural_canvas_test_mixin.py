"""
Neural Canvas Test Mixin - Test mode display operations

Contains:
- display_test_values: Display test inference values on nodes
- clear_test_values: Clear all test values from nodes
- refresh_nodes: Refresh all node items to reflect updated data

Author: Noodlings Project
Date: December 2025
"""

from ...core.neural_canvas.neural_node import NodeType


class NeuralCanvasTestMixin:
    """Mixin providing test mode display for NeuralCanvasView."""

    def display_test_values(self, node_outputs: dict):
        """
        Display test inference values on canvas nodes.

        Args:
            node_outputs: Dict mapping node_id -> {port_name: value}
        """
        for node_id, outputs in node_outputs.items():
            node_item = self.node_items.get(node_id)
            if node_item:
                node_item.set_test_values(outputs)

        # Force scene update
        self.scene.update()

    def clear_test_values(self):
        """Clear all test values from canvas nodes."""
        for node_item in self.node_items.values():
            node_item.clear_test_values()

        self.scene.update()

    def refresh_nodes(self):
        """
        Refresh all node items to reflect updated data.

        Called when node properties are changed externally (e.g., from Inspector).
        For COMMENT nodes, recalculates size based on text.
        """
        for node_id, node_item in self.node_items.items():
            # For COMMENT nodes, recalculate dimensions based on text
            if node_item.node.type == NodeType.COMMENT:
                comment_text = node_item.node.params.get('text', '')
                comment_width = node_item.node.params.get('width', 320)
                comment_height = node_item.node.params.get('height', None)

                node_item.width = comment_width

                if comment_height:
                    node_item.height = comment_height
                else:
                    # Auto-calculate height based on text
                    from PyQt6.QtGui import QFontMetrics, QFont
                    font = QFont("Menlo", 11)
                    fm = QFontMetrics(font)

                    # Calculate wrapped text height
                    lines = comment_text.split('\n')
                    total_lines = 0
                    for line in lines:
                        if line:
                            chars_per_line = max(1, int((comment_width - 24) / fm.averageCharWidth()))
                            total_lines += max(1, (len(line) + chars_per_line - 1) // chars_per_line)
                        else:
                            total_lines += 1

                    line_height = fm.height()
                    text_height = total_lines * line_height
                    node_item.height = max(60, text_height + 40)  # Padding

                node_item.prepareGeometryChange()

            # Trigger repaint
            node_item.update()

        self.scene.update()
