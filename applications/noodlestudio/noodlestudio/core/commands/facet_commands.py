"""
Facet Commands - Undo commands for Facets Editor operations

Commands for:
- Moving facets (with drag merging)
- Creating/deleting facets
- Editing facet properties (prompt, model, etc.)
- Creating/deleting connections
- Toggling lock state

Author: Commander Spock + Cadet Caity
Date: December 15, 2025
"""

from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING
from PyQt6.QtGui import QUndoCommand

from .base_command import StudioCommand, MergeableCommand, CommandID

if TYPE_CHECKING:
    from ..facet_system import Facet, FacetConnection, FacetAssembly
    from ...panels.facets_editor_panel import FacetsEditorPanel
    from ...panels.inspector_panel import InspectorPanel


class MoveFacetCommand(MergeableCommand):
    """
    Command for moving a facet node.

    Supports merging consecutive moves (drag = 1 undo operation).
    """

    COMMAND_ID = CommandID.MOVE_FACET

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        facet_id: str,
        old_pos: Tuple[float, float],
        new_pos: Tuple[float, float],
        facet_name: str = ""
    ):
        """
        Initialize move command.

        Args:
            editor: Reference to FacetsEditorPanel
            facet_id: UUID of facet being moved
            old_pos: Position before move (x, y)
            new_pos: Position after move (x, y)
            facet_name: Display name for undo text
        """
        text = f"Move '{facet_name}'" if facet_name else "Move Facet"
        super().__init__(text, merge_id=facet_id)

        self.editor = editor
        self.facet_id = facet_id
        self.old_pos = old_pos
        self.new_pos = new_pos

    def _do(self):
        """Move facet to new position."""
        # On first execution (when pushed), node is already at new_pos
        # Just save to disk. On re-redo, actually move the node.
        if self._first_redo:
            # Just save to disk (position already set by user drag)
            self.editor._save_assembly_to_disk()
        else:
            # Re-redo: actually move the node
            self.editor._set_facet_position_internal(self.facet_id, self.new_pos)

    def _undo(self):
        """Move facet back to old position."""
        self.editor._set_facet_position_internal(self.facet_id, self.old_pos)

    def id(self) -> int:
        """Return command ID for merge compatibility."""
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Merge consecutive moves of the same facet.

        This makes dragging a single undo operation instead of
        hundreds of tiny position updates.
        """
        if not isinstance(other, MoveFacetCommand):
            return False
        if other.facet_id != self.facet_id:
            return False

        # Merge: keep our old_pos, take their new_pos
        self.new_pos = other.new_pos
        return True


class CreateFacetCommand(StudioCommand):
    """Command for creating a new facet."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        facet_data: Dict[str, Any],
        facet_name: str = ""
    ):
        """
        Initialize create command.

        Args:
            editor: Reference to FacetsEditorPanel
            facet_data: Serialized facet data (from Facet.to_dict())
            facet_name: Display name for undo text
        """
        text = f"Create '{facet_name}'" if facet_name else "Create Facet"
        super().__init__(text)

        self.editor = editor
        self.facet_data = facet_data
        self.facet_id = facet_data.get('id', '')

    def _do(self):
        """Create the facet."""
        self.editor._create_facet_internal(self.facet_data)

    def _undo(self):
        """Delete the created facet."""
        self.editor._delete_facet_internal(self.facet_id)


class DeleteFacetCommand(StudioCommand):
    """Command for deleting a facet."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        facet_data: Dict[str, Any],
        connections_data: list,
        facet_name: str = ""
    ):
        """
        Initialize delete command.

        Args:
            editor: Reference to FacetsEditorPanel
            facet_data: Serialized facet data (for restoration)
            connections_data: List of connection dicts involving this facet
            facet_name: Display name for undo text
        """
        text = f"Delete '{facet_name}'" if facet_name else "Delete Facet"
        super().__init__(text)

        self.editor = editor
        self.facet_data = facet_data
        self.facet_id = facet_data.get('id', '')
        self.connections_data = connections_data

    def _do(self):
        """Delete the facet and its connections."""
        self.editor._delete_facet_internal(self.facet_id)

    def _undo(self):
        """Restore the facet and its connections."""
        self.editor._create_facet_internal(self.facet_data)
        for conn_data in self.connections_data:
            self.editor._create_connection_internal(conn_data)


class EditFacetPropertyCommand(StudioCommand):
    """Command for editing a facet property (prompt, model, temperature, etc.)."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        facet_id: str,
        property_name: str,
        old_value: Any,
        new_value: Any,
        facet_name: str = ""
    ):
        """
        Initialize edit command.

        Args:
            editor: Reference to FacetsEditorPanel
            facet_id: UUID of facet being edited
            property_name: Name of property (e.g., 'prompt', 'model', 'temperature')
            old_value: Value before edit
            new_value: Value after edit
            facet_name: Display name for undo text
        """
        # Create descriptive text
        prop_display = property_name.replace('_', ' ').title()
        if facet_name:
            text = f"Edit {prop_display} of '{facet_name}'"
        else:
            text = f"Edit {prop_display}"
        super().__init__(text)

        self.editor = editor
        self.facet_id = facet_id
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value

    def _do(self):
        """Apply the new value."""
        self.editor._set_facet_property_internal(
            self.facet_id, self.property_name, self.new_value
        )

    def _undo(self):
        """Restore the old value."""
        self.editor._set_facet_property_internal(
            self.facet_id, self.property_name, self.old_value
        )


class CreateConnectionCommand(StudioCommand):
    """Command for creating a connection between facets."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        from_facet: str,
        from_pad: str,
        to_facet: str,
        to_pad: str
    ):
        """
        Initialize connection create command.

        Args:
            editor: Reference to FacetsEditorPanel
            from_facet: Source facet ID
            from_pad: Source pad name
            to_facet: Destination facet ID
            to_pad: Destination pad name
        """
        super().__init__("Create Connection")

        self.editor = editor
        self.from_facet = from_facet
        self.from_pad = from_pad
        self.to_facet = to_facet
        self.to_pad = to_pad

    def _do(self):
        """Create the connection."""
        self.editor._create_connection_internal({
            'from': f"{self.from_facet}.{self.from_pad}",
            'to': f"{self.to_facet}.{self.to_pad}"
        })

    def _undo(self):
        """Delete the connection."""
        self.editor._delete_connection_internal(
            self.from_facet, self.from_pad,
            self.to_facet, self.to_pad
        )


class DeleteConnectionCommand(StudioCommand):
    """Command for deleting a connection between facets."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        from_facet: str,
        from_pad: str,
        to_facet: str,
        to_pad: str
    ):
        """
        Initialize connection delete command.

        Args:
            editor: Reference to FacetsEditorPanel
            from_facet: Source facet ID
            from_pad: Source pad name
            to_facet: Destination facet ID
            to_pad: Destination pad name
        """
        super().__init__("Delete Connection")

        self.editor = editor
        self.from_facet = from_facet
        self.from_pad = from_pad
        self.to_facet = to_facet
        self.to_pad = to_pad

    def _do(self):
        """Delete the connection."""
        self.editor._delete_connection_internal(
            self.from_facet, self.from_pad,
            self.to_facet, self.to_pad
        )

    def _undo(self):
        """Restore the connection."""
        self.editor._create_connection_internal({
            'from': f"{self.from_facet}.{self.from_pad}",
            'to': f"{self.to_facet}.{self.to_pad}"
        })


class ToggleLockCommand(StudioCommand):
    """Command for toggling facet lock state."""

    def __init__(
        self,
        editor: 'FacetsEditorPanel',
        facet_id: str,
        was_locked: bool,
        facet_name: str = ""
    ):
        """
        Initialize lock toggle command.

        Args:
            editor: Reference to FacetsEditorPanel
            facet_id: UUID of facet
            was_locked: Lock state before toggle
            facet_name: Display name for undo text
        """
        action = "Unlock" if was_locked else "Lock"
        text = f"{action} '{facet_name}'" if facet_name else f"{action} Facet"
        super().__init__(text)

        self.editor = editor
        self.facet_id = facet_id
        self.was_locked = was_locked

    def _do(self):
        """Toggle lock to new state."""
        self.editor._set_facet_property_internal(
            self.facet_id, 'locked', not self.was_locked
        )

    def _undo(self):
        """Toggle lock back to old state."""
        self.editor._set_facet_property_internal(
            self.facet_id, 'locked', self.was_locked
        )


class InspectorPropertyCommand(MergeableCommand):
    """
    Command for changing a facet property via Inspector.

    Supports merging consecutive changes to the same property
    (e.g., typing "hello" becomes one undo operation).
    """

    COMMAND_ID = CommandID.CHANGE_PROPERTY

    def __init__(
        self,
        inspector: 'InspectorPanel',
        facet_id: str,
        property_name: str,
        old_value: Any,
        new_value: Any,
        facet_name: str = ""
    ):
        """
        Initialize property change command.

        Args:
            inspector: Reference to InspectorPanel
            facet_id: UUID of facet being edited
            property_name: Name of property (e.g., 'prompt', 'temperature')
            old_value: Value before change
            new_value: Value after change
            facet_name: Display name for undo text
        """
        prop_display = property_name.replace('_', ' ').title()
        if facet_name:
            text = f"Change {prop_display} of '{facet_name}'"
        else:
            text = f"Change {prop_display}"
        super().__init__(text, merge_id=f"{facet_id}.{property_name}")

        self.inspector = inspector
        self.facet_id = facet_id
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value

    def _do(self):
        """Apply the new value."""
        # On first execution, the property is already set by the widget
        # Just save to disk
        if self._first_redo:
            self.inspector._save_facet_property_to_disk()
        else:
            self.inspector._set_facet_property_internal(
                self.facet_id, self.property_name, self.new_value
            )

    def _undo(self):
        """Restore the old value."""
        self.inspector._set_facet_property_internal(
            self.facet_id, self.property_name, self.old_value
        )

    def id(self) -> int:
        """Return command ID for merge compatibility."""
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Merge consecutive changes to the same property.

        This makes typing a word one undo operation instead of
        one per character.
        """
        if not isinstance(other, InspectorPropertyCommand):
            return False
        if other.merge_id != self.merge_id:
            return False

        # Merge: keep our old_value, take their new_value
        self.new_value = other.new_value
        # Update text to reflect final value
        return True


class GenericPropertyCommand(MergeableCommand):
    """
    Generic command for changing any object property via PropertyBinding.

    This is the universal undo command used by the PropertyBinding system.
    Works with any object that has gettable/settable attributes.

    Supports merging consecutive changes to the same property
    (e.g., typing text becomes one undo operation).
    """

    COMMAND_ID = CommandID.CHANGE_PROPERTY

    def __init__(
        self,
        inspector: 'InspectorPanel',
        obj: Any,
        property_name: str,
        old_value: Any,
        new_value: Any,
        display_name: str = "",
        obj_name: str = ""
    ):
        """
        Initialize generic property change command.

        Args:
            inspector: Reference to InspectorPanel (for refresh after undo)
            obj: Object containing the property
            property_name: Name of property being changed
            old_value: Value before change
            new_value: Value after change
            display_name: Human-readable property name for undo text
            obj_name: Name of the object for undo text
        """
        # Create descriptive text
        prop_display = display_name or property_name.replace('_', ' ').title()
        if obj_name:
            text = f"Change {prop_display} of '{obj_name}'"
        else:
            text = f"Change {prop_display}"

        # Merge ID is object ID + property name
        obj_id = getattr(obj, 'id', None) or id(obj)
        super().__init__(text, merge_id=f"{obj_id}.{property_name}")

        self.inspector = inspector
        self.obj = obj
        self.property_name = property_name
        self.old_value = old_value
        self.new_value = new_value

    def _do(self):
        """Apply the new value."""
        # On first execution, the property is already set by the widget
        # Just save to disk
        if self._first_redo:
            self._save_to_disk()
        else:
            setattr(self.obj, self.property_name, self.new_value)
            self._refresh_widget()
            self._save_to_disk()

    def _undo(self):
        """Restore the old value."""
        setattr(self.obj, self.property_name, self.old_value)
        self._refresh_widget()
        self._save_to_disk()

    def _refresh_widget(self):
        """Refresh the bound widget to show current value."""
        # Find and refresh any binding for this property
        if hasattr(self.inspector, '_binding_manager'):
            for binding in self.inspector._binding_manager.get_bindings_for_object(self.obj):
                if binding.property_name == self.property_name:
                    binding.refresh_from_model()
                    break

    def _save_to_disk(self):
        """Persist the change to disk."""
        # For facets, save via the facet assembly
        if hasattr(self.obj, 'id') and hasattr(self.inspector, '_auto_save_facet_assembly'):
            self.inspector._auto_save_facet_assembly()
        # For other objects, they should implement their own persistence

    def id(self) -> int:
        """Return command ID for merge compatibility."""
        return self.COMMAND_ID

    def mergeWith(self, other: QUndoCommand) -> bool:
        """
        Merge consecutive changes to the same property.

        This makes typing a word one undo operation instead of
        one per character.
        """
        if not isinstance(other, GenericPropertyCommand):
            return False
        if other.merge_id != self.merge_id:
            return False

        # Merge: keep our old_value, take their new_value
        self.new_value = other.new_value
        return True
