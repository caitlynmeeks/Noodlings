"""
Node Definitions - Templates for all Neural Canvas node types.

Provides factory functions to create preconfigured nodes.

Author: Commander Spock + Cadet Caity
Date: December 8, 2025
"""

from typing import Dict, Any
from .neural_node import NeuralNode, NodeType, Port, DataType, WeightInfo


# Node type definitions with default parameters and ports
NODE_DEFINITIONS: Dict[NodeType, Dict[str, Any]] = {
    NodeType.INPUT: {
        'name': 'Affect Input',
        'description': 'Network entry point (5-D affect vector)',
        'params': {'output_dim': 5},
        'inputs': {},
        'outputs': {
            'affect': Port('affect', DataType.AFFECT, shape=(5,), label='Affect (5-D)')
        },
        'weights': {},
        'color': '#2A4A2A',  # Deep forest green (darker, richer)
        'icon': '🔵'
    },

    NodeType.OUTPUT: {
        'name': 'Network Output',
        'description': 'Network exit point',
        'params': {},
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='State (40-D)')
        },
        'outputs': {},
        'weights': {},
        'color': '#2A4A2A',  # Deep forest green (matches INPUT)
        'icon': '🟢'
    },

    NodeType.LSTM: {
        'name': 'LSTM Layer',
        'description': 'Long Short-Term Memory recurrent layer',
        'params': {
            'input_dim': 5,
            'hidden_dim': 16,
            'dropout': 0.0,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input'),
            'h': Port('h', DataType.HIDDEN_STATE, required=False, label='Hidden'),
            'c': Port('c', DataType.CELL_STATE, required=False, label='Cell')
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE, label='Hidden'),
            'c_out': Port('c_out', DataType.CELL_STATE, label='Cell')
        },
        'weights': {},  # Computed dynamically based on params
        'color': '#4A2A4A',  # Deeper plum (darker, more saturated)
        'icon': '📦'
    },

    NodeType.GRU: {
        'name': 'GRU Layer',
        'description': 'Gated Recurrent Unit layer',
        'params': {
            'input_dim': 16,
            'hidden_dim': 8,
            'dropout': 0.0,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input'),
            'h': Port('h', DataType.HIDDEN_STATE, required=False, label='Hidden')
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE, label='Hidden')
        },
        'weights': {},
        'color': '#4A2A4A',  # Deeper plum (matches LSTM)
        'icon': '⚙️'
    },

    NodeType.RNN: {
        'name': 'RNN Layer',
        'description': 'Simple recurrent neural network layer',
        'params': {
            'input_dim': 5,
            'hidden_dim': 16,
            'activation': 'tanh'
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True),
            'h': Port('h', DataType.HIDDEN_STATE, required=False)
        },
        'outputs': {
            'h_out': Port('h_out', DataType.HIDDEN_STATE)
        },
        'weights': {},
        'color': '#673AB7',  # Deep purple
        'icon': '🔄'
    },

    NodeType.LINEAR: {
        'name': 'Linear Layer',
        'description': 'Fully connected linear transformation',
        'params': {
            'in_features': 16,
            'out_features': 32,
            'bias': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#9C27B0',  # Purple
        'icon': '➡️'
    },

    NodeType.CONV1D: {
        'name': 'Conv1D Layer',
        'description': '1D convolutional layer',
        'params': {
            'in_channels': 1,
            'out_channels': 16,
            'kernel_size': 3,
            'stride': 1,
            'padding': 1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#E91E63',  # Pink
        'icon': '🌊'
    },

    NodeType.ATTENTION: {
        'name': 'Attention',
        'description': 'Scaled dot-product attention mechanism',
        'params': {
            'embed_dim': 64,
            'dropout': 0.1
        },
        'inputs': {
            'query': Port('query', DataType.TENSOR, required=True),
            'key': Port('key', DataType.TENSOR, required=True),
            'value': Port('value', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR),
            'weights': Port('weights', DataType.TENSOR)
        },
        'weights': {},
        'color': '#F44336',  # Red
        'icon': '👁️'
    },

    NodeType.MULTI_HEAD_ATTENTION: {
        'name': 'Multi-Head Attention',
        'description': 'Multi-head attention (Transformer)',
        'params': {
            'embed_dim': 64,
            'num_heads': 4,
            'dropout': 0.1
        },
        'inputs': {
            'query': Port('query', DataType.TENSOR, required=True),
            'key': Port('key', DataType.TENSOR, required=True),
            'value': Port('value', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#FF5722',  # Deep orange
        'icon': '👁️👁️'
    },

    NodeType.TANH: {
        'name': 'Tanh',
        'description': 'Hyperbolic tangent activation',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal (neutral, subtle)
        'icon': '〰️'
    },

    NodeType.RELU: {
        'name': 'ReLU',
        'description': 'Rectified Linear Unit activation',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '⚡'
    },

    NodeType.GELU: {
        'name': 'GELU',
        'description': 'Gaussian Error Linear Unit activation',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '🌀'
    },

    NodeType.SIGMOID: {
        'name': 'Sigmoid',
        'description': 'Sigmoid activation function',
        'params': {},
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True, label='Input')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Output')
        },
        'weights': {},
        'color': '#3A3A3A',  # Dark charcoal
        'icon': '📈'
    },

    NodeType.SOFTMAX: {
        'name': 'Softmax',
        'description': 'Softmax activation',
        'params': {
            'dim': -1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#8BC34A',  # Light green
        'icon': '🎲'
    },

    NodeType.LAYER_NORM: {
        'name': 'Layer Normalization',
        'description': 'Layer normalization',
        'params': {
            'normalized_shape': (16,),
            'eps': 1e-5,
            'elementwise_affine': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#4CAF50',  # Green
        'icon': '📏'
    },

    NodeType.BATCH_NORM: {
        'name': 'Batch Normalization',
        'description': 'Batch normalization',
        'params': {
            'num_features': 16,
            'eps': 1e-5,
            'momentum': 0.1
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#009688',  # Teal
        'icon': '📊'
    },

    NodeType.DROPOUT: {
        'name': 'Dropout',
        'description': 'Dropout regularization',
        'params': {
            'p': 0.5
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#00BCD4',  # Cyan
        'icon': '💧'
    },

    NodeType.STATE_CONCAT: {
        'name': 'State Concatenation',
        'description': 'Concatenate multiple hidden states into phenomenal state',
        'params': {},
        'inputs': {
            'fast': Port('fast', DataType.HIDDEN_STATE, shape=(16,), label='Fast (16-D)'),
            'medium': Port('medium', DataType.HIDDEN_STATE, shape=(16,), label='Medium (16-D)'),
            'slow': Port('slow', DataType.HIDDEN_STATE, shape=(8,), label='Slow (8-D)')
        },
        'outputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='Phenomenal State')
        },
        'weights': {},
        'color': '#2A4A4A',  # Deeper teal (darker, sophisticated)
        'icon': '➕'
    },

    NodeType.STATE_SPLIT: {
        'name': 'State Split',
        'description': 'Split phenomenal state into components',
        'params': {},
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,))
        },
        'outputs': {
            'fast': Port('fast', DataType.HIDDEN_STATE, shape=(16,)),
            'medium': Port('medium', DataType.HIDDEN_STATE, shape=(16,)),
            'slow': Port('slow', DataType.HIDDEN_STATE, shape=(8,))
        },
        'weights': {},
        'color': '#03A9F4',  # Light blue
        'icon': '➖'
    },

    NodeType.AFFECT_HEAD: {
        'name': 'Affect Head',
        'description': 'Maps phenomenal state to 5-D continuous affect',
        'params': {
            'state_dim': 40,
            'affect_dim': 5,
            'hidden_dim': 32
        },
        'inputs': {
            'state': Port('state', DataType.PHENOMENAL_STATE, shape=(40,), label='State (40-D)')
        },
        'outputs': {
            'valence': Port('valence', DataType.SCALAR, label='Valence'),
            'arousal': Port('arousal', DataType.SCALAR, label='Arousal'),
            'fear': Port('fear', DataType.SCALAR, label='Fear'),
            'sorrow': Port('sorrow', DataType.SCALAR, label='Sorrow'),
            'boredom': Port('boredom', DataType.SCALAR, label='Boredom')
        },
        'weights': {},
        'color': '#4A3A2A',  # Deeper tobacco brown (darker, warmer)
        'icon': '🎭'
    },

    NodeType.QUANTUM_MICROTUBULE: {
        'name': 'Quantum Microtubule',
        'description': 'Penrose-Hameroff quantum consciousness layer',
        'params': {
            'input_dim': 16,
            'hidden_dim': 16,
            'collapse_threshold': 0.5,
            'coherence_time': 10,
            'entanglement_range': 3,
            'noise_scale': 0.1,
            'use_collapse': True,
            'use_entanglement': True
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True),
            'mt_state': Port('mt_state', DataType.HIDDEN_STATE, required=False)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR),
            'new_mt_state': Port('new_mt_state', DataType.HIDDEN_STATE)
        },
        'weights': {},
        'color': '#4A2A3A',  # Deeper burgundy (darker, mysterious)
        'icon': '⚛️'
    },

    NodeType.IBM_QUANTUM: {
        'name': 'IBM Quantum',
        'description': 'Real quantum computation via IBM Quantum cloud',
        'params': {
            'num_qubits': 4,
            'shots': 100,
            'backend': 'simulator',
            'entanglement_type': 'full'
        },
        'inputs': {
            'classical_state': Port('classical_state', DataType.TENSOR, required=True)
        },
        'outputs': {
            'quantum_result': Port('quantum_result', DataType.TENSOR)
        },
        'weights': {},
        'color': '#4A2A3A',  # Deeper burgundy (matches quantum family)
        'icon': '🔮'
    },

    NodeType.ENTROPY_INJECTION: {
        'name': 'Entropy Injection',
        'description': 'Inject true quantum randomness (TrueRNG)',
        'params': {
            'noise_scale': 0.1,
            'use_hardware_rng': True,
            'distribution': 'avalanche'
        },
        'inputs': {
            'x': Port('x', DataType.TENSOR, required=True)
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR)
        },
        'weights': {},
        'color': '#FF5722',  # Deep orange
        'icon': '🎲'
    },

    NodeType.CHECKPOINT: {
        'name': 'Checkpoint',
        'description': 'Trained weight checkpoint (.npz file)',
        'params': {
            'checkpoint_path': '',
            'total_params': 0,
            'trained_epochs': 0,
            'final_loss': 0.0
        },
        'inputs': {},  # No inputs - this is a data source
        'outputs': {
            'weights': Port('weights', DataType.TENSOR)  # Provides weights to network
        },
        'weights': {},
        'color': '#795548',  # Brown (asset color)
        'icon': '💾'
    },

    # Tutorial/Interactive nodes
    NodeType.NUMBER_INPUT: {
        'name': 'Number Input',
        'description': 'Interactive scalar input with slider (0.0 to 1.0)',
        'params': {
            'value': 0.5,  # Current value
            'min_value': 0.0,
            'max_value': 1.0,
            'step': 0.1
        },
        'inputs': {},  # No inputs - this is a source node
        'outputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), label='Value')
        },
        'weights': {},
        'color': '#4A6A4A',  # Forest green (input family)
        'icon': '🎚️'
    },

    NodeType.THRESHOLD_OUTPUT: {
        'name': 'Threshold Output',
        'description': 'ON/OFF display based on threshold comparison',
        'params': {
            'threshold': 0.5,  # Activation threshold
            'show_value': True  # Show numeric value alongside ON/OFF
        },
        'inputs': {
            'value': Port('value', DataType.SCALAR, shape=(1,), required=True, label='Value')
        },
        'outputs': {},  # No outputs - this is a sink node
        'weights': {},
        'color': '#6A4A4A',  # Muted red-brown (output family)
        'icon': '💡'
    },

    NodeType.CONCAT: {
        'name': 'Concat',
        'description': 'Concatenate two inputs into one tensor',
        'params': {},
        'inputs': {
            'a': Port('a', DataType.TENSOR, required=True, label='Input A'),
            'b': Port('b', DataType.TENSOR, required=True, label='Input B')
        },
        'outputs': {
            'out': Port('out', DataType.TENSOR, label='Combined')
        },
        'weights': {},
        'color': '#4A4A6A',  # Muted blue (utility)
        'icon': '⊕'
    }
}


def create_node_from_type(node_type: NodeType, name: Optional[str] = None) -> NeuralNode:
    """
    Create a node from its type definition.

    Args:
        node_type: Type of node to create
        name: Override default name (optional)

    Returns:
        Preconfigured NeuralNode instance
    """
    if node_type not in NODE_DEFINITIONS:
        raise ValueError(f"Unknown node type: {node_type}")

    definition = NODE_DEFINITIONS[node_type]

    node = NeuralNode.create_with_uuid(
        node_type=node_type,
        name=name or definition['name']
    )

    # Copy parameters
    node.params = definition['params'].copy()

    # Copy ports (deep copy to avoid shared references)
    node.inputs = {
        name: Port(
            name=port.name,
            data_type=port.data_type,
            shape=port.shape,
            required=port.required,
            label=port.label  # Preserve label
        )
        for name, port in definition['inputs'].items()
    }

    node.outputs = {
        name: Port(
            name=port.name,
            data_type=port.data_type,
            shape=port.shape,
            label=port.label  # Preserve label
        )
        for name, port in definition['outputs'].items()
    }

    # Copy metadata
    node.color = definition.get('color')
    node.description = definition['description']

    # Compute weights if applicable (for LSTM, GRU, Linear, etc.)
    _compute_weights_for_node(node)

    return node


def _compute_weights_for_node(node: NeuralNode):
    """
    Compute weight shapes for a node based on its type and parameters.

    Modifies node.weights in place.
    """
    if node.type == NodeType.LSTM:
        input_dim = node.params.get('input_dim', 5)
        hidden_dim = node.params.get('hidden_dim', 16)
        has_bias = node.params.get('bias', True)

        # LSTM has 4 gates (input, forget, cell, output)
        # Weight shape: (4 * hidden_dim, input_dim) and (4 * hidden_dim, hidden_dim)
        node.weights['weight_ih'] = WeightInfo('weight_ih', (4 * hidden_dim, input_dim))
        node.weights['weight_hh'] = WeightInfo('weight_hh', (4 * hidden_dim, hidden_dim))

        if has_bias:
            node.weights['bias_ih'] = WeightInfo('bias_ih', (4 * hidden_dim,))
            node.weights['bias_hh'] = WeightInfo('bias_hh', (4 * hidden_dim,))

    elif node.type == NodeType.GRU:
        input_dim = node.params.get('input_dim', 16)
        hidden_dim = node.params.get('hidden_dim', 8)
        has_bias = node.params.get('bias', True)

        # GRU has 3 gates (reset, update, new)
        node.weights['weight_ih'] = WeightInfo('weight_ih', (3 * hidden_dim, input_dim))
        node.weights['weight_hh'] = WeightInfo('weight_hh', (3 * hidden_dim, hidden_dim))

        if has_bias:
            node.weights['bias_ih'] = WeightInfo('bias_ih', (3 * hidden_dim,))
            node.weights['bias_hh'] = WeightInfo('bias_hh', (3 * hidden_dim,))

    elif node.type == NodeType.LINEAR:
        in_features = node.params.get('in_features', 16)
        out_features = node.params.get('out_features', 32)
        has_bias = node.params.get('bias', True)

        node.weights['weight'] = WeightInfo('weight', (out_features, in_features))

        if has_bias:
            node.weights['bias'] = WeightInfo('bias', (out_features,))

    elif node.type == NodeType.AFFECT_HEAD:
        state_dim = node.params.get('state_dim', 40)
        affect_dim = node.params.get('affect_dim', 5)
        hidden_dim = node.params.get('hidden_dim', 32)

        # Two-layer MLP: state → hidden → affect
        node.weights['fc1_weight'] = WeightInfo('fc1_weight', (hidden_dim, state_dim))
        node.weights['fc1_bias'] = WeightInfo('fc1_bias', (hidden_dim,))
        node.weights['fc2_weight'] = WeightInfo('fc2_weight', (affect_dim, hidden_dim))
        node.weights['fc2_bias'] = WeightInfo('fc2_bias', (affect_dim,))


def get_node_icon(node_type: NodeType) -> str:
    """Get emoji icon for node type."""
    return NODE_DEFINITIONS.get(node_type, {}).get('icon', '⚙️')


def get_node_color(node_type: NodeType) -> str:
    """Get color for node type."""
    return NODE_DEFINITIONS.get(node_type, {}).get('color', '#757575')
