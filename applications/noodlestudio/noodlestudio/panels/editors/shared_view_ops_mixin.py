"""Shared view operations mixin for editor views.

Provides focus toggle (F key) and frame-all (A key).
Assumes self is a QGraphicsView subclass.
"""

from PyQt6.QtCore import Qt


class SharedViewOpsMixin:
    """Focus, frame-all, and framing utilities for canvas editor views."""

    def _init_view_ops_state(self):
        """Initialize view ops mixin state. Call from concrete view __init__."""
        self._is_focused = False
        self._focused_node_ids = None  # tuple of IDs or None
        self._pre_focus_transform = None

    def focus_selection(self):
        """Toggle focus on selected nodes (F key).

        First press: save transform, zoom to selection.
        Second press on same selection: restore pre-focus transform.
        """
        node_items = self.get_node_items()
        selected_ids = self.get_selected_node_ids()

        if not selected_ids:
            return

        selection_key = tuple(sorted(selected_ids))
        selected_items = [
            node_items[nid] for nid in selected_ids if nid in node_items
        ]
        if not selected_items:
            return

        if self._is_focused and self._focused_node_ids == selection_key:
            # Restore pre-focus view
            if self._pre_focus_transform:
                self.setTransform(self._pre_focus_transform)
            self._is_focused = False
            self._focused_node_ids = None
            self._pre_focus_transform = None
        else:
            # Save and focus
            self._pre_focus_transform = self.transform()
            self._focused_node_ids = selection_key
            self._is_focused = True
            self._frame_items(selected_items, padding_factor=0.1)

    def frame_all_nodes(self):
        """Frame all nodes in view (A key)."""
        items = list(self.get_node_items().values())
        if items:
            self._frame_items(items, padding_factor=0.15)

    def _frame_items(self, items: list, padding_factor: float = 0.1):
        """Frame given graphics items in the viewport with padding.

        Args:
            items: List of QGraphicsItem instances.
            padding_factor: Extra space around items (0.1 = 10%).
        """
        if not items:
            return

        rect = items[0].sceneBoundingRect()
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())

        padding = max(rect.width(), rect.height()) * padding_factor
        rect.adjust(-padding, -padding, padding, padding)

        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self.centerOn(rect.center())

    # -- Abstract: concrete view must implement --

    def get_node_items(self) -> dict:
        """Override: return dict of node_id -> QGraphicsItem."""
        raise NotImplementedError

    def get_selected_node_ids(self) -> list:
        """Override: return list of selected node ID strings."""
        raise NotImplementedError
