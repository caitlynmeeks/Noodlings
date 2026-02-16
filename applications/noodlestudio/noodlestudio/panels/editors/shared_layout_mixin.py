"""Shared layout mixin for editor views.

Provides topological auto-arrange and alignment operations.
Assumes self is a QGraphicsView subclass.
"""


class SharedLayoutMixin:
    """Auto-arrange and alignment for canvas editor views.

    Configure behavior via class attributes on the concrete view:
        LAYER_SPACING: Horizontal spacing between layers (default 300)
        NODE_SPACING: Vertical spacing within a layer (default 180)
        START_X: Left margin for auto-arrange (default 100)
        START_Y: Top margin for auto-arrange (default 100)
    """

    LAYER_SPACING = 300
    NODE_SPACING = 180
    START_X = 100
    START_Y = 100

    def auto_arrange(self):
        """Auto-arrange nodes using Kahn's topological layering.

        Horizontal flow: layers go left to right, nodes stack vertically
        within each layer. Cycles are placed in a final catch-all layer.
        """
        node_items = self.get_node_items()
        edges = self.get_graph_edges()

        if not node_items:
            return

        node_ids = set(node_items.keys())

        # Build adjacency and in-degree
        dependents = {nid: [] for nid in node_ids}
        in_degree = {nid: 0 for nid in node_ids}

        for from_id, to_id in edges:
            if from_id in node_ids and to_id in node_ids:
                dependents[from_id].append(to_id)
                in_degree[to_id] += 1

        # Kahn's algorithm: layer by layer
        layers = []
        current_layer = [nid for nid, deg in in_degree.items() if deg == 0]

        while current_layer:
            layers.append(sorted(current_layer))
            next_layer = []
            for nid in current_layer:
                for dep in dependents[nid]:
                    in_degree[dep] -= 1
                    if in_degree[dep] == 0:
                        next_layer.append(dep)
            current_layer = next_layer

        # Remaining nodes (cycles) go in a final layer
        remaining = sorted(nid for nid, deg in in_degree.items() if deg > 0)
        if remaining:
            layers.append(remaining)

        # Position nodes
        for layer_idx, layer_nodes in enumerate(layers):
            x = self.START_X + (layer_idx * self.LAYER_SPACING)
            for node_idx, node_id in enumerate(layer_nodes):
                y = self.START_Y + (node_idx * self.NODE_SPACING)
                item = node_items.get(node_id)
                if item:
                    item.setPos(x, y)

        self.on_layout_complete()

    def align_selected_horizontally(self):
        """Align selected nodes to average Y coordinate."""
        node_items = self.get_node_items()
        scene = self.scene()
        if scene is None:
            return

        selected = [
            item for item in scene.selectedItems()
            if any(item is v for v in node_items.values())
        ]
        if len(selected) < 2:
            return

        avg_y = sum(item.pos().y() for item in selected) / len(selected)
        for item in selected:
            item.setPos(item.pos().x(), avg_y)

        self.on_layout_complete()

    def align_selected_vertically(self):
        """Align selected nodes to average X coordinate."""
        node_items = self.get_node_items()
        scene = self.scene()
        if scene is None:
            return

        selected = [
            item for item in scene.selectedItems()
            if any(item is v for v in node_items.values())
        ]
        if len(selected) < 2:
            return

        avg_x = sum(item.pos().x() for item in selected) / len(selected)
        for item in selected:
            item.setPos(avg_x, item.pos().y())

        self.on_layout_complete()

    # -- Abstract: concrete view must implement --

    def get_node_items(self) -> dict:
        """Override: return dict of node_id -> QGraphicsItem."""
        raise NotImplementedError

    def get_graph_edges(self) -> list:
        """Override: return list of (from_node_id, to_node_id) tuples."""
        raise NotImplementedError

    def on_layout_complete(self):
        """Override: called after layout changes (update wires, save, etc.)."""
        raise NotImplementedError
