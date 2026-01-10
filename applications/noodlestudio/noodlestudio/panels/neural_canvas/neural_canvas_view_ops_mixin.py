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
#   Neural Canvas View Operations Mixin - Focus, frame, zoom operations
#
#   Contains: - focus_selection: Toggle focus on selected nod...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.panels.neural_canvas.neural_canvas_view_ops_mixin
# PURPOSE:  Neural Canvas View Ops Mixin
# LAYER:    Studio / Neural Canvas Panels
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   NeuralCanvasViewOpsMixin
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

from PyQt6.QtCore import Qt


class NeuralCanvasViewOpsMixin:
    """Mixin providing view operations for NeuralCanvasView."""

    def focus_selection(self):
        """
        Toggle focus on selected nodes (F key).

        Supports multi-selection: frames all selected nodes as a unit.
        First press: Zooms to selection, saves view state
        Second press: Restores pre-focus view state
        """
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        selected_items = self.scene.selectedItems()
        selected_nodes = [
            item for item in selected_items
            if isinstance(item, NodeGraphicsItem)
        ]

        if not selected_nodes:
            print("[Neural Canvas] No nodes selected to focus")
            return

        # Create selection ID (for multi-node focus tracking)
        selection_ids = tuple(sorted(node.node.id for node in selected_nodes))

        # Check if toggling focus on same selection
        if self.is_focused and self.focused_node_id == selection_ids:
            # RESTORE: Pop back to pre-focus view
            if self.pre_focus_transform:
                self.setTransform(self.pre_focus_transform)
                print(f"[Neural Canvas] Restored pre-focus view")
            self.is_focused = False
            self.focused_node_id = None
            self.pre_focus_transform = None
        else:
            # FOCUS: Save current view and zoom to selection
            self.pre_focus_transform = self.transform()
            self.focused_node_id = selection_ids
            self.is_focused = True

            # Frame selected nodes as a unit
            self._frame_nodes(selected_nodes, padding_factor=0.1)

            if len(selected_nodes) == 1:
                print(f"[Neural Canvas] Focused on {selected_nodes[0].node.name} (press F again to restore)")
            else:
                print(f"[Neural Canvas] Focused on {len(selected_nodes)} nodes (press F again to restore)")

    def frame_all_nodes(self):
        """Frame all nodes in view (A key)."""
        # Import here to avoid circular imports
        from .neural_canvas_view import NodeGraphicsItem

        all_nodes = [
            item for item in self.scene.items()
            if isinstance(item, NodeGraphicsItem)
        ]

        if not all_nodes:
            return

        self._frame_nodes(all_nodes, padding_factor=0.15)
        print(f"[Neural Canvas] Framed all {len(all_nodes)} nodes")

    def _frame_nodes(self, nodes: list, padding_factor: float = 0.1):
        """
        Frame given nodes in view with padding (centers in viewport).

        Args:
            nodes: List of NodeGraphicsItem
            padding_factor: Extra space around nodes (0.1 = 10%)
        """
        if not nodes:
            return

        # Get bounding rect of all nodes
        scene_rect = nodes[0].sceneBoundingRect()
        for node in nodes[1:]:
            scene_rect = scene_rect.united(node.sceneBoundingRect())

        # Add padding
        padding = max(scene_rect.width(), scene_rect.height()) * padding_factor
        scene_rect.adjust(-padding, -padding, padding, padding)

        # Fit in view and center
        self.fitInView(scene_rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(scene_rect.center())

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
