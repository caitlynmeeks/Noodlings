"""Shared wire drawing mixin for editor views.

Provides temporary wire drawing, connection validation, and dispatch.
Assumes self is a QGraphicsView subclass.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPen, QColor
from PyQt6.QtWidgets import QGraphicsLineItem


class SharedWireMixin:
    """Wire drawing and connection validation for canvas editor views."""

    def _init_wire_state(self):
        """Initialize wire mixin state. Call from concrete view __init__."""
        self._wire_being_drawn = None  # QGraphicsLineItem or None
        self._wire_start_port = None   # port graphics item

    def start_wire_drawing(self, start_port):
        """Begin drawing a temporary wire from a port.

        Args:
            start_port: Port graphics item with get_scene_position() method.
        """
        self._wire_start_port = start_port
        pos = start_port.get_scene_position()
        self._wire_being_drawn = QGraphicsLineItem(
            pos.x(), pos.y(), pos.x(), pos.y()
        )
        self._wire_being_drawn.setPen(
            QPen(QColor("#FFFFFF"), 2, Qt.PenStyle.DashLine)
        )
        self.scene().addItem(self._wire_being_drawn)

    def update_wire_drawing(self, scene_pos):
        """Update the endpoint of the temporary wire being drawn.

        Args:
            scene_pos: QPointF in scene coordinates.
        """
        if self._wire_being_drawn is None:
            return
        line = self._wire_being_drawn.line()
        self._wire_being_drawn.setLine(
            line.x1(), line.y1(), scene_pos.x(), scene_pos.y()
        )

    def finish_wire_drawing(self, end_port):
        """Attempt to complete a wire connection.

        Validates the connection and dispatches to create_connection()
        if valid. Always cleans up the temporary wire.

        Args:
            end_port: Target port graphics item, or None if released on empty space.
        """
        if end_port is not None and self._wire_start_port is not None:
            if self.can_connect(self._wire_start_port, end_port):
                self.create_connection(self._wire_start_port, end_port)

        self.cancel_wire_drawing()

    def cancel_wire_drawing(self):
        """Remove the temporary wire and reset wire drawing state."""
        if self._wire_being_drawn is not None:
            scene = self.scene()
            if scene is not None:
                scene.removeItem(self._wire_being_drawn)
            self._wire_being_drawn = None
        self._wire_start_port = None

    @property
    def is_drawing_wire(self) -> bool:
        """Whether a wire is currently being drawn."""
        return self._wire_being_drawn is not None

    def can_connect(self, from_port, to_port) -> bool:
        """Validate whether two ports can be connected.

        Universal rules:
        - Ports must be on different nodes (get_parent_node_id())
        - Ports must be opposite types (is_output differs)
        - No duplicate connections (get_existing_connections())

        Then delegates to validate_connection_domain() for type-specific checks.

        Args:
            from_port: Source port graphics item.
            to_port: Target port graphics item.

        Returns:
            True if connection is valid.
        """
        # Must be different nodes
        if from_port.get_parent_node_id() == to_port.get_parent_node_id():
            return False

        # Must connect output to input (or vice versa)
        if from_port.is_output == to_port.is_output:
            return False

        # Normalize direction: output -> input
        out_port = from_port if from_port.is_output else to_port
        in_port = to_port if from_port.is_output else from_port

        # Check for duplicate
        for conn in self.get_existing_connections():
            if (conn[0] == out_port.get_parent_node_id() and
                conn[1] == out_port.get_port_name() and
                conn[2] == in_port.get_parent_node_id() and
                conn[3] == in_port.get_port_name()):
                return False

        # Domain-specific validation
        return self.validate_connection_domain(out_port, in_port)

    def validate_connection_domain(self, from_port, to_port) -> bool:
        """Override for domain-specific connection validation.

        Called after universal rules pass. NC overrides for DataType
        compatibility. Assembly view can return True if connections
        are untyped.

        Args:
            from_port: Output port (always the output side after normalization).
            to_port: Input port (always the input side after normalization).

        Returns:
            True if domain-specific rules allow the connection.
        """
        return True

    # -- Abstract: concrete view must implement --

    def create_connection(self, from_port, to_port):
        """Override: create a persistent connection between ports."""
        raise NotImplementedError

    def get_existing_connections(self) -> list:
        """Override: return list of (from_node_id, from_port_name, to_node_id, to_port_name) tuples."""
        raise NotImplementedError
