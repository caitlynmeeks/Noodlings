"""
Facets Editor Wire Mixin - Wire drawing and connection operations

Contains wire/connection operations:
- start_wire_drawing: Begin drawing a wire from a pad
- can_connect: Check if two pads can be connected
- create_connection: Create a connection between pads
- delete_connection_wire: Delete a wire via context menu

Author: Noodlings Project
Date: December 2025
"""

from PyQt6.QtWidgets import QGraphicsLineItem
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor

from .facets_editor_graphics import FacetPadGraphics, ConnectionWire, get_facet_header_color
from ..core.facet_system import PadType


class FacetsEditorWireMixin:
    """Mixin providing wire drawing for FacetsEditorPanel."""

    def start_wire_drawing(self, start_pad: FacetPadGraphics):
        """Start drawing a connection wire from a pad."""
        self.wire_start_pad = start_pad

        # Create temporary line for visual feedback
        start_pos = start_pad.get_scene_position()
        self.wire_being_drawn = QGraphicsLineItem(
            start_pos.x(), start_pos.y(),
            start_pos.x(), start_pos.y()
        )
        self.wire_being_drawn.setPen(QPen(QColor("#FFFFFF"), 2, Qt.PenStyle.DashLine))
        self.scene.addItem(self.wire_being_drawn)

    def can_connect(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics) -> bool:
        """
        Check if connection between two pads is valid.

        Rules:
        - Must connect output to input (or vice versa)
        - Cannot connect to same node
        - Cannot create duplicate connection

        Args:
            from_pad: Source pad
            to_pad: Target pad

        Returns:
            True if connection is valid
        """
        # Must be different nodes
        if from_pad.facet_node == to_pad.facet_node:
            return False

        # Must connect output to input
        if from_pad.pad.pad_type == to_pad.pad.pad_type:
            return False

        # Check for duplicate connection
        from_id = from_pad.facet_node.facet.id
        to_id = to_pad.facet_node.facet.id
        from_pad_name = from_pad.pad.name
        to_pad_name = to_pad.pad.name

        # Normalize direction (always output -> input)
        if from_pad.pad.pad_type == PadType.INPUT:
            from_id, to_id = to_id, from_id
            from_pad_name, to_pad_name = to_pad_name, from_pad_name

        for conn in self.current_assembly.connections:
            if (conn.from_facet == from_id and conn.to_facet == to_id and
                conn.from_pad == from_pad_name and conn.to_pad == to_pad_name):
                return False  # Duplicate

        return True

    def create_connection(self, from_pad: FacetPadGraphics, to_pad: FacetPadGraphics):
        """
        Create a connection between two pads (with undo support).

        Args:
            from_pad: Source pad
            to_pad: Target pad
        """
        from ..core.facet_system import FacetConnection

        if not self.current_assembly:
            return

        # Normalize direction (always output -> input)
        if from_pad.pad.pad_type == PadType.INPUT:
            from_pad, to_pad = to_pad, from_pad

        # Get IDs
        from_facet = from_pad.facet_node.facet.id
        to_facet = to_pad.facet_node.facet.id
        from_pad_name = from_pad.pad.name
        to_pad_name = to_pad.pad.name

        # Push undo command
        from ..core.undo_manager import undo_manager
        from ..core.commands import CreateConnectionCommand

        cmd = CreateConnectionCommand(
            self, from_facet, from_pad_name, to_facet, to_pad_name
        )
        undo_manager.push(cmd)

    def delete_connection_wire(self, wire: 'ConnectionWire'):
        """
        Delete a connection wire via context menu or keyboard.

        Uses undo command for proper undo/redo support.

        Args:
            wire: The ConnectionWire graphics item to delete
        """
        if not wire or not self.current_assembly:
            return

        # Extract connection info from wire
        from_facet = wire.from_pad.facet_node.facet.id
        from_pad = wire.from_pad.pad.name
        to_facet = wire.to_pad.facet_node.facet.id
        to_pad = wire.to_pad.pad.name

        # Push undo command
        from ..core.undo_manager import undo_manager
        from ..core.commands import DeleteConnectionCommand

        cmd = DeleteConnectionCommand(
            self, from_facet, from_pad, to_facet, to_pad
        )
        undo_manager.push(cmd)
