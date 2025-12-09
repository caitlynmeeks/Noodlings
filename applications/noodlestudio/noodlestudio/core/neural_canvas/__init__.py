"""
Neural Canvas - Visual editor for neural network architectures.

Provides node-based editing of CharmNetwork internals (LSTM/GRU topology).
Exports to MLX code and .nncanvas JSON format.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

from .neural_graph import NeuralGraph, ValidationResult
from .neural_node import NeuralNode, NodeType, Port, Connection, WeightInfo
from .node_definitions import NODE_DEFINITIONS, create_node_from_type

__all__ = [
    'NeuralGraph',
    'NeuralNode',
    'NodeType',
    'Port',
    'Connection',
    'WeightInfo',
    'ValidationResult',
    'NODE_DEFINITIONS',
    'create_node_from_type'
]
