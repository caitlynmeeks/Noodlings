"""
Undo Commands for NoodleStudio

This package contains QUndoCommand subclasses for all undoable operations:
- facet_commands: Facets Editor operations
- neural_commands: Neural Canvas operations
- property_commands: Inspector property changes

Author: Commander Spock + Cadet Caity
Date: December 15, 2025
"""

from .base_command import StudioCommand, MergeableCommand, CommandID
from .facet_commands import (
    MoveFacetCommand,
    CreateFacetCommand,
    DeleteFacetCommand,
    EditFacetPropertyCommand,
    CreateConnectionCommand,
    DeleteConnectionCommand,
    ToggleLockCommand,
    InspectorPropertyCommand,
    GenericPropertyCommand
)
from .neural_commands import (
    MoveNeuralNodeCommand,
    CreateNeuralNodeCommand,
    DeleteNeuralNodeCommand,
    CreateNeuralConnectionCommand,
    DeleteNeuralConnectionCommand,
    EditNeuralNodeParamCommand,
    RenameNeuralNodeCommand
)

__all__ = [
    'StudioCommand',
    'MergeableCommand',
    'CommandID',
    # Facet commands
    'MoveFacetCommand',
    'CreateFacetCommand',
    'DeleteFacetCommand',
    'EditFacetPropertyCommand',
    'CreateConnectionCommand',
    'DeleteConnectionCommand',
    'ToggleLockCommand',
    'InspectorPropertyCommand',
    'GenericPropertyCommand',
    # Neural Canvas commands
    'MoveNeuralNodeCommand',
    'CreateNeuralNodeCommand',
    'DeleteNeuralNodeCommand',
    'CreateNeuralConnectionCommand',
    'DeleteNeuralConnectionCommand',
    'EditNeuralNodeParamCommand',
    'RenameNeuralNodeCommand'
]
