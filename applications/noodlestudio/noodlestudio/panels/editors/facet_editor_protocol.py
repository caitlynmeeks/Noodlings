"""Protocol for facet editor undo command integration.

Defines the interface that undo commands (facet_commands.py) call back
into the editor. AssemblyEditorView implements this protocol.
"""

from typing import Protocol, Dict, Any, Tuple


class FacetEditorProtocol(Protocol):
    """Interface for undo commands to call back into the editor.

    Methods are prefixed with _ because they are internal mutation
    methods that bypass undo (the commands themselves handle undo).
    """

    def _save_assembly_to_disk(self) -> None:
        """Persist the current assembly to its YAML file."""
        ...

    def _set_facet_position_internal(
        self, facet_id: str, position: Tuple[float, float]
    ) -> None:
        """Move a facet node to the given position (data + graphics)."""
        ...

    def _create_facet_internal(self, facet_data: Dict[str, Any]) -> None:
        """Create a facet from serialized data and add it to the scene."""
        ...

    def _delete_facet_internal(self, facet_id: str) -> None:
        """Remove a facet and its connections from assembly and scene."""
        ...

    def _create_connection_internal(self, conn_data: Dict[str, Any]) -> None:
        """Create a connection from serialized data and add wire to scene."""
        ...

    def _delete_connection_internal(
        self, from_facet: str, from_pad: str, to_facet: str, to_pad: str
    ) -> None:
        """Remove a connection and its wire from assembly and scene."""
        ...

    def _set_facet_property_internal(
        self, facet_id: str, property_name: str, value: Any
    ) -> None:
        """Set a property on a facet (data + graphics update)."""
        ...
