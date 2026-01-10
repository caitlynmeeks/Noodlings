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
#   Neural Canvas Layout Mixin - Auto-arrange and alignment operations
#
#   Contains: - auto_arrange_nodes: Topological layering auto...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.neural_canvas_layout_mixin
# PURPOSE:  Neural Canvas Layout Mixin
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralCanvasLayoutMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

class NeuralCanvasLayoutMixin:
    """Mixin providing layout operations for NeuralCanvasView."""

    def auto_arrange_nodes(self):
        """Auto-arrange nodes using topological layering."""
        print("[Neural Canvas View] Starting auto-arrange...")

        if not self.graph or len(self.graph.nodes) == 0:
            print("[Neural Canvas View] No nodes to arrange")
            return

        try:
            # Get topological order
            node_order = self.graph.topological_sort()

            # Build layer assignment (same layer = same depth in DAG)
            layers = {}  # layer_index -> [node_ids]
            node_layer = {}  # node_id -> layer_index

            # Assign layers based on max dependency depth
            for node_id in node_order:
                # Find max layer of dependencies
                deps = self.graph.get_connections_to_node(node_id)
                if not deps:
                    layer = 0
                else:
                    max_dep_layer = max(
                        node_layer.get(conn.from_node, 0)
                        for conn in deps
                    )
                    layer = max_dep_layer + 1

                node_layer[node_id] = layer

                if layer not in layers:
                    layers[layer] = []
                layers[layer].append(node_id)

            # Layout parameters (horizontal flow, left to right)
            layer_spacing = 300  # Horizontal spacing between layers
            node_spacing = 180   # Vertical spacing within layer
            start_x = 100
            start_y = 100

            # Position nodes layer by layer (horizontal flow)
            for layer_idx in sorted(layers.keys()):
                layer_nodes = layers[layer_idx]
                x = start_x + (layer_idx * layer_spacing)  # Horizontal progression

                for node_idx, node_id in enumerate(sorted(layer_nodes)):
                    y = start_y + (node_idx * node_spacing)  # Vertical stacking within layer

                    # Update node position
                    node = self.graph.nodes[node_id]
                    node.position = (x, y)

                    print(f"[Neural Canvas View] {node.name}: ({x}, {y}) [Layer {layer_idx}]")

            # Re-render graph
            self._render_graph()
            self.graph_modified.emit()

            print(f"[Neural Canvas View] Auto-arrange complete! {len(layers)} layers")

            # Frame all nodes to show result
            self.frame_all_nodes()

        except Exception as e:
            print(f"[Neural Canvas View] Auto-arrange error: {e}")
            import traceback
            traceback.print_exc()

    def align_selected_horizontally(self):
        """Align selected nodes to same Y coordinate."""
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if len(selected) < 2:
            print("[Neural Canvas] Select at least 2 nodes to align")
            return

        # Use average Y position
        avg_y = sum(node.pos().y() for node in selected) / len(selected)

        for node in selected:
            node.setPos(node.pos().x(), avg_y)

        # Force full scene redraw (prevents smearing)
        self.scene.update()
        self.graph_modified.emit()
        print(f"[Neural Canvas] Aligned {len(selected)} nodes horizontally at y={avg_y:.0f}")

    def align_selected_vertically(self):
        """Align selected nodes to same X coordinate."""
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        selected = [item for item in self.scene.selectedItems() if isinstance(item, NodeGraphicsItem)]
        if len(selected) < 2:
            print("[Neural Canvas] Select at least 2 nodes to align")
            return

        # Use average X position
        avg_x = sum(node.pos().x() for node in selected) / len(selected)

        for node in selected:
            node.setPos(avg_x, node.pos().y())

        # Force full scene redraw (prevents smearing)
        self.scene.update()
        self.graph_modified.emit()
        print(f"[Neural Canvas] Aligned {len(selected)} nodes vertically at x={avg_x:.0f}")

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
