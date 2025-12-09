"""
Neural Node - Core node definitions for Neural Canvas.

Defines node types, ports, connections, and parameters.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional, List
from enum import Enum
import uuid


class NodeType(Enum):
    """Neural network node types."""

    # Special nodes
    INPUT = "INPUT"
    OUTPUT = "OUTPUT"

    # Recurrent layers
    LSTM = "LSTM"
    GRU = "GRU"
    RNN = "RNN"

    # Feedforward layers
    LINEAR = "LINEAR"
    CONV1D = "CONV1D"

    # Attention
    ATTENTION = "ATTENTION"
    MULTI_HEAD_ATTENTION = "MULTI_HEAD_ATTENTION"

    # Activation functions
    TANH = "TANH"
    RELU = "RELU"
    GELU = "GELU"
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"

    # Normalization
    LAYER_NORM = "LAYER_NORM"
    BATCH_NORM = "BATCH_NORM"

    # Regularization
    DROPOUT = "DROPOUT"

    # Utility
    STATE_CONCAT = "STATE_CONCAT"
    STATE_SPLIT = "STATE_SPLIT"
    AFFECT_HEAD = "AFFECT_HEAD"

    # Quantum/Experimental
    QUANTUM_MICROTUBULE = "QUANTUM_MICROTUBULE"
    IBM_QUANTUM = "IBM_QUANTUM"
    ENTROPY_INJECTION = "ENTROPY_INJECTION"

    # Asset nodes
    CHECKPOINT = "CHECKPOINT"  # Trained weight checkpoint (.npz file)


class DataType(Enum):
    """Data types for port validation."""
    AFFECT = "AFFECT"  # 5-D affect vector
    HIDDEN_STATE = "HIDDEN_STATE"  # Recurrent hidden state
    CELL_STATE = "CELL_STATE"  # LSTM cell state
    PHENOMENAL_STATE = "PHENOMENAL_STATE"  # 40-D state
    TENSOR = "TENSOR"  # General tensor
    SCALAR = "SCALAR"  # Single value


@dataclass
class Port:
    """Input or output port on a node."""
    name: str  # Technical name (e.g., "h_out")
    data_type: DataType
    shape: Optional[Tuple[int, ...]] = None  # None = dynamic
    required: bool = True
    label: Optional[str] = None  # Human-readable label (e.g., "Hidden State")

    def get_display_label(self) -> str:
        """Get display label (falls back to name if no custom label)."""
        return self.label if self.label else self.name

    def __str__(self):
        shape_str = f"{self.shape}" if self.shape else "dynamic"
        display = self.get_display_label()
        return f"{display} ({self.data_type.value}, {shape_str})"


@dataclass
class Connection:
    """Connection between two nodes."""
    from_node: str  # Node ID
    from_port: str  # Port name
    to_node: str
    to_port: str

    def __str__(self):
        return f"{self.from_node}.{self.from_port} → {self.to_node}.{self.to_port}"


@dataclass
class WeightInfo:
    """Information about trainable weights."""
    name: str  # e.g., "weight_ih"
    shape: Tuple[int, ...]
    path: Optional[str] = None  # Path to .npy file (if pretrained)
    trainable: bool = True

    def num_parameters(self) -> int:
        """Calculate number of parameters."""
        result = 1
        for dim in self.shape:
            result *= dim
        return result


@dataclass
class NeuralNode:
    """
    A single node in the neural network graph.

    Represents a layer, operation, or special node (INPUT/OUTPUT).
    """

    id: str
    type: NodeType
    name: str
    position: Tuple[int, int] = (0, 0)

    # Node parameters (e.g., hidden_dim=16)
    params: Dict[str, Any] = field(default_factory=dict)

    # Input/output ports
    inputs: Dict[str, Port] = field(default_factory=dict)
    outputs: Dict[str, Port] = field(default_factory=dict)

    # Trainable weights
    weights: Dict[str, WeightInfo] = field(default_factory=dict)

    # Visual state
    collapsed: bool = False
    color: Optional[str] = None  # Override default color
    tags: List[str] = field(default_factory=list)
    description: str = ""

    @staticmethod
    def create_with_uuid(node_type: NodeType, name: str) -> 'NeuralNode':
        """Create node with auto-generated UUID."""
        return NeuralNode(
            id=str(uuid.uuid4()),
            type=node_type,
            name=name
        )

    def validate_params(self) -> List[str]:
        """
        Validate parameter values.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Type-specific validation
        if self.type == NodeType.LSTM:
            if 'hidden_dim' in self.params:
                if not (1 <= self.params['hidden_dim'] <= 512):
                    errors.append(f"hidden_dim must be 1-512, got {self.params['hidden_dim']}")
            else:
                errors.append("LSTM requires 'hidden_dim' parameter")

            if 'dropout' in self.params:
                if not (0 <= self.params['dropout'] <= 1):
                    errors.append(f"dropout must be 0-1, got {self.params['dropout']}")

        elif self.type == NodeType.GRU:
            if 'hidden_dim' in self.params:
                if not (1 <= self.params['hidden_dim'] <= 512):
                    errors.append(f"hidden_dim must be 1-512, got {self.params['hidden_dim']}")
            else:
                errors.append("GRU requires 'hidden_dim' parameter")

        elif self.type == NodeType.LINEAR:
            if 'out_features' not in self.params:
                errors.append("LINEAR requires 'out_features' parameter")
            elif self.params['out_features'] < 1:
                errors.append(f"out_features must be >= 1, got {self.params['out_features']}")

        elif self.type == NodeType.DROPOUT:
            if 'p' in self.params:
                if not (0 <= self.params['p'] <= 0.9):
                    errors.append(f"dropout p must be 0-0.9, got {self.params['p']}")
            else:
                errors.append("DROPOUT requires 'p' parameter")

        return errors

    def compute_output_shapes(self, input_shapes: Dict[str, Tuple[int, ...]]) -> Dict[str, Tuple[int, ...]]:
        """
        Compute output shapes based on input shapes and parameters.

        Args:
            input_shapes: Dict mapping input port names to shapes

        Returns:
            Dict mapping output port names to shapes
        """
        output_shapes = {}

        if self.type == NodeType.LSTM:
            hidden_dim = self.params.get('hidden_dim', 16)
            output_shapes['h_out'] = (hidden_dim,)
            output_shapes['c_out'] = (hidden_dim,)

        elif self.type == NodeType.GRU:
            hidden_dim = self.params.get('hidden_dim', 16)
            output_shapes['h_out'] = (hidden_dim,)

        elif self.type == NodeType.LINEAR:
            out_features = self.params.get('out_features', 1)
            output_shapes['out'] = (out_features,)

        elif self.type == NodeType.STATE_CONCAT:
            # Concatenate all input dimensions
            total_dim = sum(shape[0] for shape in input_shapes.values() if shape)
            output_shapes['state'] = (total_dim,)

        elif self.type == NodeType.AFFECT_HEAD:
            output_shapes['valence'] = (1,)
            output_shapes['arousal'] = (1,)
            output_shapes['fear'] = (1,)
            output_shapes['sorrow'] = (1,)
            output_shapes['boredom'] = (1,)

        else:
            # Default: pass through input shape
            if input_shapes:
                first_input_shape = next(iter(input_shapes.values()))
                output_shapes['out'] = first_input_shape

        return output_shapes

    def compute_num_parameters(self) -> int:
        """Calculate total trainable parameters in this node."""
        total = 0
        for weight_info in self.weights.values():
            if weight_info.trainable:
                total += weight_info.num_parameters()
        return total

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary (for .nncanvas JSON)."""
        return {
            'id': self.id,
            'type': self.type.value,
            'name': self.name,
            'position': list(self.position),
            'params': self.params,
            'inputs': {
                name: {
                    'data_type': port.data_type.value,
                    'shape': list(port.shape) if port.shape else None,
                    'required': port.required,
                    'label': port.label
                }
                for name, port in self.inputs.items()
            },
            'outputs': {
                name: {
                    'data_type': port.data_type.value,
                    'shape': list(port.shape) if port.shape else None,
                    'label': port.label
                }
                for name, port in self.outputs.items()
            },
            'weights': {
                name: {
                    'shape': list(info.shape),
                    'path': info.path,
                    'trainable': info.trainable
                }
                for name, info in self.weights.items()
            },
            'collapsed': self.collapsed,
            'color': self.color,
            'tags': self.tags,
            'description': self.description
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'NeuralNode':
        """Deserialize from dictionary."""
        node = NeuralNode(
            id=data['id'],
            type=NodeType(data['type']),
            name=data['name'],
            position=tuple(data.get('position', [0, 0])),
            params=data.get('params', {}),
            collapsed=data.get('collapsed', False),
            color=data.get('color'),
            tags=data.get('tags', []),
            description=data.get('description', '')
        )

        # Deserialize inputs
        for name, port_data in data.get('inputs', {}).items():
            node.inputs[name] = Port(
                name=name,
                data_type=DataType(port_data['data_type']),
                shape=tuple(port_data['shape']) if port_data.get('shape') else None,
                required=port_data.get('required', True),
                label=port_data.get('label')
            )

        # Deserialize outputs
        for name, port_data in data.get('outputs', {}).items():
            node.outputs[name] = Port(
                name=name,
                data_type=DataType(port_data['data_type']),
                shape=tuple(port_data['shape']) if port_data.get('shape') else None,
                label=port_data.get('label')
            )

        # Deserialize weights
        for name, weight_data in data.get('weights', {}).items():
            node.weights[name] = WeightInfo(
                name=name,
                shape=tuple(weight_data['shape']),
                path=weight_data.get('path'),
                trainable=weight_data.get('trainable', True)
            )

        return node
