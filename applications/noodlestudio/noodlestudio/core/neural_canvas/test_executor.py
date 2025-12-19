"""
Neural Canvas Test Executor - Run inference directly from canvas topology.

Executes the visual graph as PyTorch operations for immediate feedback during design.
Uses PyTorch for cross-platform support (Windows, Linux, macOS).

Author: Commander Spock + Cadet Caity
Date: December 17, 2025
"""

from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from .neural_graph import NeuralGraph
from .neural_node import NeuralNode, NodeType, DataType


@dataclass
class TestResult:
    """Result of a test inference run."""
    success: bool
    outputs: Dict[str, Any] = field(default_factory=dict)
    node_outputs: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # node_id -> {port: value}
    error: Optional[str] = None
    execution_time_ms: float = 0.0


class CanvasTestExecutor:
    """
    Execute canvas topology directly using PyTorch.

    Maintains hidden states across test runs for temporal continuity.
    Cross-platform: works on Windows, Linux, and macOS.
    """

    def __init__(self, graph: NeuralGraph):
        self.graph = graph
        self.hidden_states: Dict[str, Any] = {}  # state_name -> torch.Tensor
        self.layers: Dict[str, Any] = {}  # node_id -> instantiated layer
        self._initialized = False

    def initialize(self) -> Tuple[bool, str]:
        """
        Initialize layers and hidden states from graph.

        Returns:
            (success, error_message)
        """
        if not TORCH_AVAILABLE:
            return False, "PyTorch not available. Install with: pip install torch"

        try:
            # Validate graph first
            result = self.graph.validate()
            if not result.valid:
                return False, f"Graph invalid: {'; '.join(result.errors)}"

            # Initialize layers for each node
            self.layers = {}
            self.hidden_states = {}

            for node_id, node in self.graph.nodes.items():
                if node.type in (NodeType.INPUT, NodeType.OUTPUT):
                    continue

                layer, hidden_shapes = self._create_layer(node)
                if layer is not None:
                    self.layers[node_id] = layer
                    # Set to eval mode (no dropout, etc.)
                    if hasattr(layer, 'eval'):
                        layer.eval()

                # Initialize hidden states
                for state_name, shape in hidden_shapes.items():
                    full_name = f"{node_id}_{state_name}"
                    self.hidden_states[full_name] = torch.zeros((1,) + shape)

            self._initialized = True
            return True, ""

        except Exception as e:
            return False, str(e)

    def _create_layer(self, node: NeuralNode) -> Tuple[Any, Dict[str, Tuple[int, ...]]]:
        """
        Create PyTorch layer for a node.

        Returns:
            (layer, hidden_state_shapes)
        """
        hidden_shapes = {}

        if node.type == NodeType.LSTM:
            input_dim = node.params.get('input_dim', 5)
            hidden_dim = node.params.get('hidden_dim', 16)
            # PyTorch LSTM: batch_first=True for (batch, seq, features) format
            layer = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            # PyTorch LSTM uses (h, c) tuple, each shape (num_layers, batch, hidden)
            hidden_shapes['h'] = (1, hidden_dim)  # (num_layers, hidden_dim)
            hidden_shapes['c'] = (1, hidden_dim)
            return layer, hidden_shapes

        elif node.type == NodeType.GRU:
            input_dim = node.params.get('input_dim', 5)
            hidden_dim = node.params.get('hidden_dim', 8)
            layer = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            hidden_shapes['h'] = (1, hidden_dim)
            return layer, hidden_shapes

        elif node.type == NodeType.RNN:
            input_dim = node.params.get('input_dim', 5)
            hidden_dim = node.params.get('hidden_dim', 16)
            layer = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
            hidden_shapes['h'] = (1, hidden_dim)
            return layer, hidden_shapes

        elif node.type == NodeType.LINEAR:
            in_features = node.params.get('in_features', 16)
            out_features = node.params.get('out_features', 5)
            layer = nn.Linear(in_features, out_features)
            return layer, hidden_shapes

        elif node.type == NodeType.TANH:
            return nn.Tanh(), hidden_shapes

        elif node.type == NodeType.RELU:
            return nn.ReLU(), hidden_shapes

        elif node.type == NodeType.GELU:
            return nn.GELU(), hidden_shapes

        elif node.type == NodeType.SIGMOID:
            return nn.Sigmoid(), hidden_shapes

        elif node.type == NodeType.SOFTMAX:
            return nn.Softmax(dim=-1), hidden_shapes

        elif node.type == NodeType.DROPOUT:
            p = node.params.get('p', 0.0)
            layer = nn.Dropout(p=p)
            return layer, hidden_shapes

        elif node.type == NodeType.LAYER_NORM:
            dims = node.params.get('normalized_shape', 16)
            if isinstance(dims, int):
                dims = [dims]
            layer = nn.LayerNorm(dims)
            return layer, hidden_shapes

        elif node.type == NodeType.AFFECT_HEAD:
            # Custom affect head: Linear -> Tanh -> Linear
            state_dim = node.params.get('state_dim', 40)
            hidden_dim = node.params.get('hidden_dim', 16)
            affect_dim = node.params.get('affect_dim', 5)
            # Use nn.Sequential for cleaner execution
            layer = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, affect_dim),
                nn.Tanh()  # Output in [-1, 1] range
            )
            return layer, hidden_shapes

        elif node.type == NodeType.STATE_CONCAT:
            return 'concat', hidden_shapes

        else:
            # Unsupported node type - pass through
            return None, hidden_shapes

    def reset_states(self):
        """Reset all hidden states to zeros."""
        for state_name in self.hidden_states:
            shape = self.hidden_states[state_name].shape
            self.hidden_states[state_name] = torch.zeros(shape)

    def execute(self, input_affect: Optional[List[float]] = None) -> TestResult:
        """
        Execute the graph with given input.

        Args:
            input_affect: 5-D affect vector [valence, arousal, dominance, sorrow, boredom]
                         If None, uses neutral affect [0, 0.5, 0.5, 0, 0]

        Returns:
            TestResult with outputs and per-node values
        """
        import time
        start_time = time.time()

        if not TORCH_AVAILABLE:
            return TestResult(
                success=False,
                error="PyTorch not available"
            )

        if not self._initialized:
            success, error = self.initialize()
            if not success:
                return TestResult(success=False, error=error)

        try:
            # Default neutral affect
            if input_affect is None:
                input_affect = [0.0, 0.5, 0.5, 0.0, 0.0]

            # Ensure 5-D
            while len(input_affect) < 5:
                input_affect.append(0.0)
            input_affect = input_affect[:5]

            # Convert to PyTorch tensor: shape (1, 5)
            # Use no_grad for inference (no gradient tracking needed)
            with torch.no_grad():
                x = torch.tensor([input_affect], dtype=torch.float32)

                # Get execution order
                node_order = self.graph.topological_sort()

                # Track outputs per node
                node_outputs: Dict[str, Dict[str, Any]] = {}

                # Execute each node
                for node_id in node_order:
                    node = self.graph.nodes[node_id]

                    if node.type == NodeType.INPUT:
                        # Input node outputs the input affect
                        node_outputs[node_id] = {
                            'affect': x,
                            'x': x
                        }
                        continue

                    if node.type == NodeType.NUMBER_INPUT:
                        # Number input outputs its stored value as a scalar tensor
                        value = node.params.get('value', 0.5)
                        value_tensor = torch.tensor([[value]], dtype=torch.float32)
                        node_outputs[node_id] = {
                            'value': value_tensor,
                            'x': value_tensor
                        }
                        continue

                    if node.type == NodeType.THRESHOLD_OUTPUT:
                        # Threshold output gathers inputs and computes ON/OFF
                        incoming = self.graph.get_connections_to_node(node_id)
                        output_values = {}
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                output_values[conn.to_port] = src_outputs[conn.from_port]

                        # Get input value and compare to threshold
                        input_val = output_values.get('value')
                        if input_val is not None:
                            if hasattr(input_val, 'item'):
                                scalar_val = input_val.flatten()[0].item()
                            else:
                                scalar_val = float(input_val)

                            threshold = node.params.get('threshold', 0.5)
                            is_on = scalar_val >= threshold

                            node_outputs[node_id] = {
                                'value': scalar_val,
                                'is_on': is_on,
                                'threshold': threshold
                            }
                        else:
                            node_outputs[node_id] = {
                                'value': 0.0,
                                'is_on': False,
                                'threshold': node.params.get('threshold', 0.5)
                            }
                        continue

                    if node.type == NodeType.OUTPUT:
                        # Gather inputs to output node
                        incoming = self.graph.get_connections_to_node(node_id)
                        output_values = {}
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                output_values[conn.to_port] = src_outputs[conn.from_port]
                        node_outputs[node_id] = output_values
                        continue

                    # Get layer
                    layer = self.layers.get(node_id)

                    # Gather inputs
                    incoming = self.graph.get_connections_to_node(node_id)
                    inputs = {}
                    for conn in incoming:
                        src_outputs = node_outputs.get(conn.from_node, {})
                        if conn.from_port in src_outputs:
                            inputs[conn.to_port] = src_outputs[conn.from_port]

                    # Execute based on node type
                    outputs = self._execute_node(node, layer, inputs)
                    node_outputs[node_id] = outputs

                # Get final outputs from OUTPUT node
                output_nodes = self.graph.get_output_nodes()
                final_outputs = {}
                if output_nodes:
                    output_node_id = output_nodes[0].id
                    final_outputs = node_outputs.get(output_node_id, {})

                # Convert PyTorch tensors to Python for display
                display_outputs = {}
                for key, value in final_outputs.items():
                    if hasattr(value, 'tolist'):
                        display_outputs[key] = value.tolist()
                    else:
                        display_outputs[key] = value

                # Convert node outputs for display
                display_node_outputs = {}
                for node_id, outputs in node_outputs.items():
                    display_node_outputs[node_id] = {}
                    for port, value in outputs.items():
                        if hasattr(value, 'tolist'):
                            arr = value.tolist()
                            # Flatten if needed
                            if isinstance(arr, list) and len(arr) == 1:
                                arr = arr[0]
                            display_node_outputs[node_id][port] = arr
                        else:
                            display_node_outputs[node_id][port] = value

            execution_time = (time.time() - start_time) * 1000

            return TestResult(
                success=True,
                outputs=display_outputs,
                node_outputs=display_node_outputs,
                execution_time_ms=execution_time
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            return TestResult(
                success=False,
                error=str(e)
            )

    def _execute_node(self, node: NeuralNode, layer: Any, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single node and return outputs."""
        outputs = {}

        # Get primary input (usually 'x' or first available)
        x = None
        for key in ['x', 'affect', 'input', 'state']:
            if key in inputs:
                x = inputs[key]
                break
        if x is None and inputs:
            x = list(inputs.values())[0]

        if x is None:
            # No input - return empty
            return outputs

        if node.type == NodeType.LSTM:
            # Get hidden states (PyTorch LSTM uses (h, c) tuple)
            h_name = f"{node.id}_h"
            c_name = f"{node.id}_c"
            h = self.hidden_states.get(h_name)
            c = self.hidden_states.get(c_name)

            # Ensure x has sequence dimension: (batch, seq, features)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)  # (batch, 1, features)

            # Forward pass - PyTorch LSTM takes (h, c) tuple
            if h is not None and c is not None:
                out, (h_new, c_new) = layer(x, (h, c))
            else:
                out, (h_new, c_new) = layer(x)

            # Update hidden states
            self.hidden_states[h_name] = h_new
            self.hidden_states[c_name] = c_new

            # Output is last timestep hidden state (squeezed to 2D)
            outputs['h_out'] = h_new.squeeze(0)  # Remove num_layers dim
            outputs['c_out'] = c_new.squeeze(0)
            outputs['x'] = out[:, -1, :]  # Last timestep output

        elif node.type == NodeType.GRU:
            h_name = f"{node.id}_h"
            h = self.hidden_states.get(h_name)

            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            if h is not None:
                out, h_new = layer(x, h)
            else:
                out, h_new = layer(x)
            self.hidden_states[h_name] = h_new

            outputs['h_out'] = h_new.squeeze(0)
            outputs['x'] = out[:, -1, :]

        elif node.type == NodeType.RNN:
            h_name = f"{node.id}_h"
            h = self.hidden_states.get(h_name)

            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            if h is not None:
                out, h_new = layer(x, h)
            else:
                out, h_new = layer(x)
            self.hidden_states[h_name] = h_new

            outputs['h_out'] = h_new.squeeze(0)
            outputs['x'] = out[:, -1, :]

        elif node.type == NodeType.LINEAR:
            # Ensure 2D
            if len(x.shape) == 3:
                x = x[:, -1, :]
            result = layer(x)
            outputs['x'] = result
            outputs['out'] = result  # LINEAR uses 'out' port in node_definitions

        elif node.type == NodeType.STATE_CONCAT:
            # Concatenate all inputs (for phenomenal state)
            tensors = [v for v in inputs.values() if hasattr(v, 'shape')]
            if tensors:
                # Flatten each to 2D and concat
                flat = [t.view(t.shape[0], -1) for t in tensors]
                outputs['x'] = torch.cat(flat, dim=-1)
                outputs['state'] = outputs['x']
            else:
                outputs['x'] = x

        elif node.type == NodeType.CONCAT:
            # Simple concatenation of two inputs
            a = inputs.get('a')
            b = inputs.get('b')
            if a is not None and b is not None:
                # Flatten to 2D if needed and concat
                if len(a.shape) == 1:
                    a = a.unsqueeze(0)
                if len(b.shape) == 1:
                    b = b.unsqueeze(0)
                a_flat = a.view(a.shape[0], -1)
                b_flat = b.view(b.shape[0], -1)
                result = torch.cat([a_flat, b_flat], dim=-1)
                outputs['x'] = result
                outputs['out'] = result
            elif a is not None:
                outputs['x'] = a
                outputs['out'] = a
            elif b is not None:
                outputs['x'] = b
                outputs['out'] = b

        elif node.type == NodeType.IBM_QUANTUM:
            # IBM Quantum computation (simulator mode or real hardware)
            # For Schrodinger's Cat: 1 qubit in superposition, collapse on measurement

            num_qubits = node.params.get('num_qubits', 1)
            shots = node.params.get('shots', 1)  # For single measurement, use 1
            backend = node.params.get('backend', 'simulator')

            # Get classical input to influence quantum state
            classical_input = inputs.get('classical_state', x)
            if len(classical_input.shape) == 3:
                classical_input = classical_input[:, -1, :]

            # Initialize qubit states (|0⟩ = [1,0], |1⟩ = [0,1])
            # Start in superposition: |ψ⟩ = (|0⟩ + |1⟩)/√2
            qubit_states = []

            for q in range(num_qubits):
                if backend == 'simulator':
                    # Simulate Hadamard gate: creates 50/50 superposition
                    # Then collapse based on quantum random measurement
                    # Use true random for authentic quantum behavior
                    import random
                    import time

                    # Seed with high-entropy source for "quantum" randomness
                    random.seed(int(time.time_ns()) ^ id(node) ^ hash(str(classical_input.tolist())))

                    # Each shot is a measurement
                    measurements = []
                    for _ in range(shots):
                        # 50/50 collapse - true quantum randomness
                        measurement = random.random() < 0.5
                        measurements.append(1.0 if measurement else 0.0)

                    # Result is the most common outcome (or random single shot)
                    if shots == 1:
                        qubit_state = measurements[0]
                    else:
                        qubit_state = sum(measurements) / len(measurements)

                    qubit_states.append(qubit_state)

                else:
                    # Real IBM Quantum hardware would go here
                    # For now, fall back to simulator
                    import random
                    qubit_states.append(1.0 if random.random() < 0.5 else 0.0)

            # Output as tensor
            result = torch.tensor([qubit_states], dtype=torch.float32)

            outputs['quantum_result'] = result
            outputs['x'] = result
            outputs['out'] = result
            outputs['collapsed_state'] = qubit_states[0] if qubit_states else 0.0
            outputs['is_alive'] = qubit_states[0] < 0.5 if qubit_states else True  # |0⟩ = alive

        elif node.type == NodeType.QUANTUM_MICROTUBULE:
            # Penrose-Hameroff quantum consciousness simulation
            # Models: superposition, entanglement, and objective reduction (collapse)

            input_dim = node.params.get('input_dim', 16)
            hidden_dim = node.params.get('hidden_dim', 16)
            collapse_threshold = node.params.get('collapse_threshold', 0.5)
            coherence_time = node.params.get('coherence_time', 10)
            entanglement_range = node.params.get('entanglement_range', 3)
            noise_scale = node.params.get('noise_scale', 0.1)
            use_collapse = node.params.get('use_collapse', True)
            use_entanglement = node.params.get('use_entanglement', True)

            # Get or initialize microtubule state (quantum superposition)
            mt_state_name = f"{node.id}_mt_state"
            mt_state = self.hidden_states.get(mt_state_name)

            if mt_state is None:
                # Initialize superposition state (complex amplitudes simulated as 2x hidden)
                mt_state = torch.randn(1, hidden_dim * 2) * 0.1
                self.hidden_states[mt_state_name] = mt_state

            # Ensure input dimensions match
            if len(x.shape) == 3:
                x = x[:, -1, :]
            if x.shape[-1] != input_dim:
                # Project input to expected dimension
                x = x[..., :input_dim] if x.shape[-1] > input_dim else \
                    torch.nn.functional.pad(x, (0, input_dim - x.shape[-1]))

            # Quantum processing: superposition evolution
            # Split state into "real" and "imaginary" components (simulated)
            real_part = mt_state[..., :hidden_dim]
            imag_part = mt_state[..., hidden_dim:]

            # Apply input as phase rotation (quantum gate simulation)
            x_expanded = x.view(1, -1)
            if x_expanded.shape[-1] < hidden_dim:
                x_expanded = torch.nn.functional.pad(x_expanded, (0, hidden_dim - x_expanded.shape[-1]))
            else:
                x_expanded = x_expanded[..., :hidden_dim]

            # Phase evolution (simplified quantum dynamics)
            phase = x_expanded * 0.5
            new_real = real_part * torch.cos(phase) - imag_part * torch.sin(phase)
            new_imag = real_part * torch.sin(phase) + imag_part * torch.cos(phase)

            # Entanglement: create correlations between neighboring units
            if use_entanglement and entanglement_range > 0:
                entangled_real = new_real.clone()
                entangled_imag = new_imag.clone()
                for i in range(hidden_dim):
                    for j in range(max(0, i - entanglement_range),
                                   min(hidden_dim, i + entanglement_range + 1)):
                        if i != j:
                            # Entangle: correlate amplitudes
                            coupling = 0.1 / (abs(i - j) + 1)
                            entangled_real[0, i] += coupling * new_real[0, j]
                            entangled_imag[0, i] += coupling * new_imag[0, j]
                new_real = entangled_real
                new_imag = entangled_imag

            # Add quantum noise (vacuum fluctuations)
            if noise_scale > 0:
                new_real = new_real + torch.randn_like(new_real) * noise_scale
                new_imag = new_imag + torch.randn_like(new_imag) * noise_scale

            # Compute probability amplitudes |psi|^2
            probabilities = new_real ** 2 + new_imag ** 2
            probabilities = probabilities / (probabilities.sum() + 1e-8)  # Normalize

            # Objective Reduction (collapse) - Penrose's gravity-induced collapse
            if use_collapse:
                # Collapse when coherence reaches threshold
                max_prob = probabilities.max().item()
                if max_prob > collapse_threshold:
                    # Collapse to most probable state (measurement)
                    collapsed_idx = probabilities.argmax(dim=-1)
                    collapsed_state = torch.zeros_like(new_real)
                    collapsed_state[0, collapsed_idx] = 1.0

                    # Reset quantum state after collapse (new superposition begins)
                    new_real = collapsed_state + torch.randn_like(new_real) * 0.1
                    new_imag = torch.randn_like(new_imag) * 0.1

                    # Output is the collapsed classical state
                    output = collapsed_state
                else:
                    # Still in superposition - output is probability-weighted
                    output = probabilities
            else:
                output = probabilities

            # Update microtubule state
            new_mt_state = torch.cat([new_real, new_imag], dim=-1)
            self.hidden_states[mt_state_name] = new_mt_state

            outputs['out'] = output
            outputs['x'] = output
            outputs['new_mt_state'] = new_mt_state
            outputs['probabilities'] = probabilities
            outputs['coherence'] = torch.tensor([[probabilities.max().item()]])

        elif node.type == NodeType.AFFECT_HEAD:
            # nn.Sequential handles the whole forward pass
            if len(x.shape) == 3:
                x = x[:, -1, :]
            affect = layer(x)
            outputs['affect'] = affect
            outputs['x'] = affect
            # Split into individual affect components
            if affect.shape[-1] >= 5:
                outputs['valence'] = affect[:, 0:1]
                outputs['arousal'] = affect[:, 1:2]
                outputs['dominance'] = affect[:, 2:3]
                outputs['sorrow'] = affect[:, 3:4]
                outputs['boredom'] = affect[:, 4:5]

        elif node.type == NodeType.DROPOUT:
            # Dropout disabled during test (eval mode)
            outputs['x'] = x

        elif node.type == NodeType.LAYER_NORM:
            outputs['x'] = layer(x)

        elif isinstance(layer, nn.Module):
            # Generic PyTorch module (Tanh, ReLU, GELU, Sigmoid, Softmax)
            if len(x.shape) == 3:
                x = x[:, -1, :]
            result = layer(x)
            outputs['x'] = result
            outputs['out'] = result  # Activation functions use 'out' port in node_definitions

        else:
            # Pass through for unsupported
            outputs['x'] = x

        return outputs


def text_to_affect(text: str) -> List[float]:
    """
    Simple heuristic to convert text to affect vector.

    This is a placeholder - in production, you'd use the actual
    CharmNetwork or a sentiment model.

    Returns:
        [valence, arousal, dominance, sorrow, boredom]
    """
    text_lower = text.lower()

    # Simple keyword-based heuristics
    valence = 0.0
    arousal = 0.5
    dominance = 0.5
    sorrow = 0.0
    boredom = 0.0

    # Positive words
    positive = ['happy', 'joy', 'love', 'wonderful', 'great', 'beautiful',
                'excited', 'amazing', 'good', 'nice', 'fun', 'laugh']
    # Negative words
    negative = ['sad', 'angry', 'hate', 'terrible', 'bad', 'awful',
                'horrible', 'upset', 'crying', 'pain', 'hurt', 'fear']
    # High arousal
    high_arousal = ['excited', 'angry', 'terrified', 'thrilled', 'furious',
                   'ecstatic', 'panic', 'rage', 'surprise']
    # Low arousal
    low_arousal = ['calm', 'peaceful', 'tired', 'sleepy', 'bored',
                  'relaxed', 'serene', 'quiet']
    # Sorrow
    sorrow_words = ['sad', 'crying', 'grief', 'mourning', 'loss', 'lonely',
                   'melancholy', 'tears', 'heartbreak']
    # Boredom
    boredom_words = ['bored', 'boring', 'dull', 'tedious', 'monotonous',
                    'uninteresting', 'tired']

    for word in positive:
        if word in text_lower:
            valence += 0.2
    for word in negative:
        if word in text_lower:
            valence -= 0.2
    for word in high_arousal:
        if word in text_lower:
            arousal += 0.15
    for word in low_arousal:
        if word in text_lower:
            arousal -= 0.15
    for word in sorrow_words:
        if word in text_lower:
            sorrow += 0.2
    for word in boredom_words:
        if word in text_lower:
            boredom += 0.2

    # Clamp values
    valence = max(-1.0, min(1.0, valence))
    arousal = max(0.0, min(1.0, arousal))
    dominance = max(0.0, min(1.0, dominance))
    sorrow = max(0.0, min(1.0, sorrow))
    boredom = max(0.0, min(1.0, boredom))

    return [valence, arousal, dominance, sorrow, boredom]
