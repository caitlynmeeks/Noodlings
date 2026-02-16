"""Shared clipboard mixin for editor views.

Provides copy/paste/duplicate with internal connection preservation.
The shared mixin handles UUID remapping and connection duplication.
Domain views only implement item-level serialization/deserialization.
"""

import copy
import uuid


class SharedClipboardMixin:
    """Copy, paste, duplicate for canvas editor views.

    Internal connection preservation: when pasting a group of items,
    connections that were entirely within the copied selection are
    automatically duplicated with remapped IDs. Domain views do NOT
    need to handle this -- it lives here.
    """

    def _init_clipboard_state(self):
        """Initialize clipboard mixin state. Call from concrete view __init__."""
        self._clipboard_items = []        # list of serialized item dicts
        self._clipboard_connections = []  # list of (from_id, from_port, to_id, to_port)

    def copy_selection(self):
        """Copy selected items and their internal connections to clipboard."""
        items_data = self.serialize_selection()
        if not items_data:
            return

        self._clipboard_items = items_data

        # Capture internal connections (both endpoints in the selection)
        selected_ids = set(item["id"] for item in items_data)
        all_connections = self.get_existing_connections()
        self._clipboard_connections = [
            conn for conn in all_connections
            if conn[0] in selected_ids and conn[2] in selected_ids
        ]

    def paste_selection(self, offset_x: float = 50.0, offset_y: float = 50.0):
        """Paste clipboard items with new IDs and offset positions.

        Internal connections are automatically duplicated with remapped IDs.

        Args:
            offset_x: Horizontal offset from original positions.
            offset_y: Vertical offset from original positions.
        """
        if not self._clipboard_items:
            return

        # Build old_id -> new_id mapping
        id_mapping = {}
        pasted_items = []

        for item_data in self._clipboard_items:
            new_data = copy.deepcopy(item_data)
            old_id = new_data["id"]
            new_id = str(uuid.uuid4())
            new_data["id"] = new_id
            id_mapping[old_id] = new_id

            # Offset position
            if "position" in new_data:
                pos = new_data["position"]
                new_data["position"] = {
                    "x": pos.get("x", 0) + offset_x,
                    "y": pos.get("y", 0) + offset_y,
                }

            pasted_items.append(new_data)

        # Remap internal connections
        pasted_connections = []
        for from_id, from_port, to_id, to_port in self._clipboard_connections:
            new_from = id_mapping.get(from_id)
            new_to = id_mapping.get(to_id)
            if new_from and new_to:
                pasted_connections.append((new_from, from_port, new_to, to_port))

        # Delegate to domain view for actual creation
        self.deserialize_items(pasted_items, pasted_connections)

    def duplicate_selection(self):
        """Copy and immediately paste the selection."""
        self.copy_selection()
        if self._clipboard_items:
            self.paste_selection()

    # -- Abstract: concrete view must implement --

    def serialize_selection(self) -> list:
        """Override: return list of dicts for selected items.

        Each dict must have at minimum an "id" key and a "position" key
        (dict with "x" and "y"). Other keys are domain-specific.
        """
        raise NotImplementedError

    def deserialize_items(self, items_data: list, connections_data: list):
        """Override: create items and connections from serialized data.

        Args:
            items_data: list of item dicts (with new IDs and offset positions).
            connections_data: list of (from_id, from_port, to_id, to_port) tuples
                with already-remapped IDs.
        """
        raise NotImplementedError

    def get_existing_connections(self) -> list:
        """Override: return list of (from_node_id, from_port_name, to_node_id, to_port_name) tuples."""
        raise NotImplementedError
