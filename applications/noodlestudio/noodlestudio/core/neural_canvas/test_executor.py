# ▄▄▄    ▄▄▄   ▄▄▄▄▄     ▄▄▄▄▄   ▄▄▄▄▄▄   ▄▄▄      ▄▄▄▄▄ ▄▄▄    ▄▄▄  ▄▄▄▄▄▄▄
# ████▄  ███ ▄███████▄ ▄███████▄ ███▀▀██▄ ███       ███  ████▄  ███ ███▀▀▀▀▀
# ███▀██▄███ ███   ███ ███   ███ ███  ███ ███       ███  ███▀██▄███ ███
# ███  ▀████ ███▄▄▄███ ███▄▄▄███ ███  ███ ███       ███  ███  ▀████ ███  ███▀
# ███    ███  ▀█████▀   ▀█████▀  ██████▀  ████████ ▄███▄ ███    ███ ▀██████▀
#
#   ▄▄▄▄▄▄▄   ▄▄▄▄▄   ▄▄▄▄▄▄▄    ▄▄▄▄▄▄▄
# ███▀▀▀▀▀ ▄███████▄ ███▀▀███▄ ███▀▀▀▀▀
# ███      ███   ███ ███▄▄███▀ ███▄▄
# ███      ███▄▄▄███ ███▀▀██▄  ███
# ▀███████  ▀█████▀  ███  ▀███ ▀███████
# ──────────────────────────────────────────────────────────────
#
#   Neural Canvas Test Executor - Run inference directly from canvas topology.
#
#   Executes the visual graph as PyTorch operations for immed...
#
# ──────────────────────────────────────────────────────────────
# MODULE:   applications.noodlestudio.core.neural_canvas.test_executor
# PURPOSE:  Tests for executor
# LAYER:    Studio / Neural Canvas
# ──────────────────────────────────────────────────────────────
#
# KEY CLASSES:
#   AudioPlayer, TestResult, CanvasTestExecutor, text_to_affect()
#
# ──────────────────────────────────────────────────────────────
# SPDX-License-Identifier: AGPL-3.0-or-later
# Subject to the Noodling Ethical Covenant (NEC)
# (C) 2026 Caitlyn Meeks
# Noodling Technologies, LLC
# https://noodlings.ai
# ──────────────────────────────────────────────────────────────

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


# Audio playback support - use Qt's QSoundEffect for stability
AUDIO_AVAILABLE = False
_qt_sound_effects = {}  # Cache for QSoundEffect objects

try:
    from PyQt6.QtMultimedia import QSoundEffect
    from PyQt6.QtCore import QUrl
    AUDIO_AVAILABLE = True
except ImportError:
    QSoundEffect = None
    QUrl = None


class AudioPlayer:
    """
    Simple cross-platform audio player for Neural Canvas.

    Uses Qt's QSoundEffect for stable audio playback in Qt apps.
    """

    @classmethod
    def play_buffer(cls, audio_data: np.ndarray, sample_rate: int = 44100, volume: float = 1.0):
        """
        Play audio buffer.

        Note: Buffer playback requires saving to temp file for Qt.
        For best results, use play_file() with pre-saved WAV files.
        """
        if not AUDIO_AVAILABLE:
            print("[FACET] Audio: PyQt6.QtMultimedia not available")
            return

        try:
            import tempfile
            import wave
            import os

            # Save buffer to temp WAV file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_path = tmp.name

            # Convert float to int16
            audio_int16 = (audio_data * volume * 32767).astype(np.int16)

            with wave.open(tmp_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_int16.tobytes())

            # Play the temp file
            cls.play_file(tmp_path, volume=1.0)  # Volume already applied

            # Note: Can't delete immediately, Qt needs the file
            # It will be cleaned up on next play or app exit

        except Exception as e:
            print(f"[FACET] Audio: Buffer playback error: {e}")

    @classmethod
    def play_file(cls, file_path: str, volume: float = 1.0):
        """
        Play audio file using Qt's QSoundEffect.

        Args:
            file_path: Path to WAV file
            volume: Volume multiplier (0-1)
        """
        if not AUDIO_AVAILABLE:
            print("[FACET] Audio: PyQt6.QtMultimedia not available")
            return

        import os
        if not os.path.exists(file_path):
            print(f"[FACET] Audio: File not found: {file_path}")
            return

        try:
            # Get or create QSoundEffect for this file
            abs_path = os.path.abspath(file_path)

            if abs_path not in _qt_sound_effects:
                sound = QSoundEffect()
                sound.setSource(QUrl.fromLocalFile(abs_path))
                _qt_sound_effects[abs_path] = sound
            else:
                sound = _qt_sound_effects[abs_path]

            # Set volume and play
            sound.setVolume(volume)
            sound.play()

        except Exception as e:
            print(f"[FACET] Audio: Playback error: {e}")

    @classmethod
    def stop(cls):
        """Stop all currently playing audio."""
        for sound in _qt_sound_effects.values():
            try:
                sound.stop()
            except Exception:
                pass

    @classmethod
    def is_playing(cls) -> bool:
        """Check if any audio is currently playing."""
        for sound in _qt_sound_effects.values():
            try:
                if sound.isPlaying():
                    return True
            except Exception:
                pass
        return False


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

            # Check for initial weight values in node.weights
            if node.weights:
                weight_info = node.weights.get('weight')
                bias_info = node.weights.get('bias')

                # Load initial weight values if provided
                if weight_info and weight_info.values is not None:
                    with torch.no_grad():
                        layer.weight.copy_(torch.tensor(weight_info.values, dtype=torch.float32))

                # Load initial bias values if provided
                if bias_info and bias_info.values is not None:
                    with torch.no_grad():
                        layer.bias.copy_(torch.tensor(bias_info.values, dtype=torch.float32))

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

        elif node.type == NodeType.MULTI_HEAD_ATTENTION:
            embed_dim = node.params.get('embed_dim', 64)
            num_heads = node.params.get('num_heads', 4)
            dropout = node.params.get('dropout', 0.0)
            layer = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            return layer, hidden_shapes

        elif node.type == NodeType.TRANSFORMER_BLOCK:
            embed_dim = node.params.get('embed_dim', 64)
            num_heads = node.params.get('num_heads', 4)
            ff_dim = node.params.get('ff_dim', 256)
            dropout = node.params.get('dropout', 0.0)
            layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            )
            return layer, hidden_shapes

        elif node.type == NodeType.POSITIONAL_ENCODING:
            # Positional encoding is computed dynamically, no learnable layer
            return 'positional_encoding', hidden_shapes

        elif node.type == NodeType.ATTENTION_VIS:
            # Visualization node - no computation, just display
            return 'attention_vis', hidden_shapes

        else:
            # Unsupported node type - pass through
            return None, hidden_shapes

    def reset_states(self):
        """Reset all hidden states to zeros."""
        for state_name in self.hidden_states:
            shape = self.hidden_states[state_name].shape
            self.hidden_states[state_name] = torch.zeros(shape)
        print(f"[FACET] NeuralCanvas: Reset {len(self.hidden_states)} hidden states")

    def _format_tensor(self, t, max_vals=4) -> str:
        """Format tensor value compactly for logging."""
        if t is None:
            return "None"
        if hasattr(t, 'tolist'):
            vals = t.flatten().tolist()
            if len(vals) <= max_vals:
                return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"
            else:
                return f"[{vals[0]:.3f}, ... ({len(vals)} vals)]"
        return str(t)

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

            # Log execution start
            graph_name = self.graph.name if hasattr(self.graph, 'name') else "Canvas"
            node_count = len(self.graph.nodes)
            print(f"[FACET] NeuralCanvas: Executing '{graph_name}' ({node_count} nodes)")
            print(f"[FACET]   Input: {self._format_tensor(input_affect)}")

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

                    if node.type == NodeType.COMMENT:
                        # Comment nodes are purely decorative - no execution
                        continue

                    if node.type == NodeType.NUMBER_INPUT:
                        # Number input outputs its stored value as a scalar tensor
                        value = node.params.get('value', 0.5)
                        value_tensor = torch.tensor([[value]], dtype=torch.float32)
                        node_outputs[node_id] = {
                            'value': value_tensor,
                            'x': value_tensor
                        }
                        print(f"[FACET]   {node.name}: {value:.3f}")
                        continue

                    if node.type == NodeType.PULSE_INPUT:
                        # Pulse input outputs 1.0 if active, 0.0 otherwise
                        pulse_active = node.params.get('pulse_active', False)
                        value = 1.0 if pulse_active else 0.0
                        value_tensor = torch.tensor([[value]], dtype=torch.float32)
                        node_outputs[node_id] = {
                            'pulse': value_tensor,
                            'value': value_tensor,
                            'x': value_tensor
                        }
                        continue

                    if node.type == NodeType.TEXT_INPUT:
                        # Text input - convert text to embedding via heuristics
                        text = node.params.get('text', '')
                        # Use the text_to_affect helper to get a simple embedding
                        affect = text_to_affect(text)
                        # Extend to 8D for more flexibility
                        embedding = affect + [0.0, 0.0, 0.0]  # 8D total
                        embedding_tensor = torch.tensor([embedding], dtype=torch.float32)
                        node_outputs[node_id] = {
                            'text': embedding_tensor,
                            'embedding': embedding_tensor,
                            'x': embedding_tensor
                        }
                        continue

                    if node.type == NodeType.TIME:
                        # Time source - outputs elapsed time
                        scale = node.params.get('scale', 1.0)
                        loop_duration = node.params.get('loop_duration', 0.0)
                        elapsed = (time.time() - start_time) * scale
                        if loop_duration > 0:
                            elapsed = elapsed % loop_duration
                        time_tensor = torch.tensor([[elapsed]], dtype=torch.float32)
                        node_outputs[node_id] = {
                            'time': time_tensor,
                            'value': time_tensor,
                            'x': time_tensor
                        }
                        continue

                    if node.type == NodeType.SINE:
                        # Sine wave generator
                        incoming = self.graph.get_connections_to_node(node_id)
                        input_tensor = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                input_tensor = src_outputs[conn.from_port]
                                break

                        freq = node.params.get('frequency', 1.0)
                        amp = node.params.get('amplitude', 1.0)
                        phase = node.params.get('phase', 0.0)

                        if input_tensor is not None:
                            import math
                            output = amp * torch.sin(input_tensor * freq * 2 * math.pi + phase * 2 * math.pi)
                        else:
                            output = torch.zeros(1, 1)

                        node_outputs[node_id] = {'out': output, 'x': output}
                        continue

                    if node.type == NodeType.NOISE:
                        # Random noise generator
                        noise_type = node.params.get('noise_type', 'uniform')
                        scale = node.params.get('scale', 1.0)
                        seed = node.params.get('seed', 0)

                        if seed > 0:
                            torch.manual_seed(seed)

                        if noise_type == 'gaussian':
                            noise = torch.randn(1, 1) * scale
                        else:  # uniform
                            noise = (torch.rand(1, 1) * 2 - 1) * scale

                        node_outputs[node_id] = {'out': noise, 'x': noise}
                        continue

                    if node.type == NodeType.MULTIPLY:
                        # Element-wise multiply
                        incoming = self.graph.get_connections_to_node(node_id)
                        a_tensor = None
                        b_tensor = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                if conn.to_port == 'a':
                                    a_tensor = src_outputs[conn.from_port]
                                elif conn.to_port == 'b':
                                    b_tensor = src_outputs[conn.from_port]

                        if a_tensor is not None and b_tensor is not None:
                            output = a_tensor * b_tensor
                        elif a_tensor is not None:
                            output = a_tensor
                        elif b_tensor is not None:
                            output = b_tensor
                        else:
                            output = torch.zeros(1, 1)

                        node_outputs[node_id] = {'out': output, 'x': output}
                        continue

                    if node.type == NodeType.ADD:
                        # Element-wise add
                        incoming = self.graph.get_connections_to_node(node_id)
                        a_tensor = None
                        b_tensor = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                if conn.to_port == 'a':
                                    a_tensor = src_outputs[conn.from_port]
                                elif conn.to_port == 'b':
                                    b_tensor = src_outputs[conn.from_port]

                        if a_tensor is not None and b_tensor is not None:
                            output = a_tensor + b_tensor
                        elif a_tensor is not None:
                            output = a_tensor
                        elif b_tensor is not None:
                            output = b_tensor
                        else:
                            output = torch.zeros(1, 1)

                        node_outputs[node_id] = {'out': output, 'x': output}
                        continue

                    if node.type == NodeType.OSCILLATOR:
                        # Audio oscillator - generates waveform buffer
                        waveform = node.params.get('waveform', 'sine')
                        freq = node.params.get('frequency', 440.0)
                        sample_rate = node.params.get('sample_rate', 44100)
                        duration = node.params.get('duration', 0.1)

                        # Get frequency modulation input if connected
                        incoming = self.graph.get_connections_to_node(node_id)
                        freq_mod = 0.0
                        amp_mod = 1.0
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                val = src_outputs[conn.from_port]
                                if hasattr(val, 'item'):
                                    val = val.item()
                                if conn.to_port == 'freq_mod':
                                    freq_mod = float(val)
                                elif conn.to_port == 'amp_mod':
                                    amp_mod = float(val)

                        # Generate waveform
                        import math
                        num_samples = int(sample_rate * duration)
                        t = torch.linspace(0, duration, num_samples)
                        actual_freq = freq + freq_mod * 100  # freq_mod scales 100Hz

                        if waveform == 'sine':
                            audio = amp_mod * torch.sin(2 * math.pi * actual_freq * t)
                        elif waveform == 'square':
                            audio = amp_mod * torch.sign(torch.sin(2 * math.pi * actual_freq * t))
                        elif waveform == 'saw':
                            audio = amp_mod * (2 * (t * actual_freq % 1) - 1)
                        elif waveform == 'triangle':
                            audio = amp_mod * (2 * torch.abs(2 * (t * actual_freq % 1) - 1) - 1)
                        else:
                            audio = torch.zeros(num_samples)

                        node_outputs[node_id] = {'audio': audio.unsqueeze(0), 'x': audio.unsqueeze(0)}
                        continue

                    if node.type == NodeType.AUDIO_OUTPUT:
                        # Audio output - plays incoming audio buffer
                        incoming = self.graph.get_connections_to_node(node_id)
                        audio_buffer = None
                        sample_rate = 44100

                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                audio_buffer = src_outputs[conn.from_port]
                                # Try to get sample rate from source node
                                if 'sample_rate' in src_outputs:
                                    sample_rate = src_outputs['sample_rate']
                                break

                        volume = node.params.get('volume', 0.5)
                        if audio_buffer is not None:
                            # Convert torch tensor to numpy and play
                            if hasattr(audio_buffer, 'numpy'):
                                audio_np = audio_buffer.squeeze().numpy()
                            elif hasattr(audio_buffer, 'flatten'):
                                audio_np = np.array(audio_buffer).flatten()
                            else:
                                audio_np = np.array(audio_buffer)

                            # Play the audio
                            AudioPlayer.play_buffer(audio_np, sample_rate=sample_rate, volume=volume)
                            print(f"[FACET]   {node.name}: Playing {len(audio_np)} samples at {sample_rate}Hz")

                            node_outputs[node_id] = {
                                'played': True,
                                'samples': len(audio_np),
                                'volume': volume
                            }
                        else:
                            node_outputs[node_id] = {'played': False, 'samples': 0}
                        continue

                    if node.type == NodeType.AUDIO_FILE:
                        # Load audio file as buffer
                        file_path = node.params.get('file_path', '')
                        loop = node.params.get('loop', False)

                        audio_buffer = None
                        sample_rate = 44100
                        duration = 0.0

                        if file_path:
                            try:
                                import os
                                if os.path.exists(file_path):
                                    # Try to load audio file
                                    if file_path.lower().endswith('.wav'):
                                        import wave
                                        with wave.open(file_path, 'rb') as wf:
                                            sample_rate = wf.getframerate()
                                            n_frames = wf.getnframes()
                                            duration = n_frames / sample_rate
                                            # Read audio data
                                            audio_data = wf.readframes(n_frames)
                                            # Convert to numpy then torch
                                            audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
                                            audio_buffer = torch.tensor(audio_np).unsqueeze(0)
                                            print(f"[FACET]   {node.name}: Loaded WAV ({duration:.2f}s, {sample_rate}Hz)")
                                    else:
                                        print(f"[FACET]   {node.name}: Unsupported format (only WAV supported)")
                                else:
                                    print(f"[FACET]   {node.name}: File not found: {file_path}")
                            except Exception as e:
                                print(f"[FACET]   {node.name}: Error loading audio: {e}")

                        if audio_buffer is None:
                            # Create silent buffer as fallback
                            audio_buffer = torch.zeros(1, int(sample_rate * 0.1))

                        node_outputs[node_id] = {
                            'audio': audio_buffer,
                            'x': audio_buffer,
                            'sample_rate': sample_rate,
                            'duration': duration,
                            'loop': loop
                        }
                        continue

                    if node.type == NodeType.AUDIO_TRIGGER:
                        # Play audio when trigger value crosses threshold
                        incoming = self.graph.get_connections_to_node(node_id)
                        trigger_value = None
                        audio_on_buffer = None
                        audio_off_buffer = None

                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                val = src_outputs[conn.from_port]
                                if conn.to_port == 'trigger':
                                    trigger_value = val
                                elif conn.to_port == 'audio_on':
                                    audio_on_buffer = val
                                elif conn.to_port == 'audio_off':
                                    audio_off_buffer = val

                        threshold = node.params.get('threshold', 0.5)
                        volume = node.params.get('volume', 0.5)
                        audio_on_path = node.params.get('audio_on_path', '')
                        audio_off_path = node.params.get('audio_off_path', '')

                        # Resolve relative paths (try noodlestudio dir first)
                        import os
                        if audio_on_path and not os.path.isabs(audio_on_path):
                            # Try relative to noodlestudio directory
                            noodlestudio_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                            resolved = os.path.join(noodlestudio_dir, audio_on_path)
                            if os.path.exists(resolved):
                                audio_on_path = resolved
                        if audio_off_path and not os.path.isabs(audio_off_path):
                            noodlestudio_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
                            resolved = os.path.join(noodlestudio_dir, audio_off_path)
                            if os.path.exists(resolved):
                                audio_off_path = resolved

                        # Get scalar from trigger value
                        if trigger_value is not None:
                            if hasattr(trigger_value, 'item'):
                                scalar_val = trigger_value.flatten()[0].item()
                            else:
                                scalar_val = float(trigger_value)
                        else:
                            scalar_val = 0.0

                        # Check previous state for edge detection
                        state_key = f"{node_id}_prev_state"
                        prev_on = self.hidden_states.get(state_key, False)
                        current_on = scalar_val >= threshold

                        # Detect transitions
                        triggered_on = current_on and not prev_on
                        triggered_off = not current_on and prev_on

                        # Store current state
                        self.hidden_states[state_key] = current_on

                        # Determine which sound to play and actually play it
                        play_sound = None
                        if triggered_on:
                            play_sound = 'on'
                            if audio_on_buffer is not None:
                                # Play buffer from connected AUDIO_FILE node
                                if hasattr(audio_on_buffer, 'numpy'):
                                    audio_np = audio_on_buffer.squeeze().numpy()
                                else:
                                    audio_np = np.array(audio_on_buffer).flatten()
                                AudioPlayer.play_buffer(audio_np, volume=volume)
                                print(f"[FACET]   {node.name}: TRIGGER ON (playing audio_on input)")
                            elif audio_on_path:
                                # Play from file path
                                AudioPlayer.play_file(audio_on_path, volume=volume)
                                print(f"[FACET]   {node.name}: TRIGGER ON (playing: {audio_on_path})")
                            else:
                                print(f"[FACET]   {node.name}: TRIGGER ON (no audio configured)")
                        elif triggered_off:
                            play_sound = 'off'
                            if audio_off_buffer is not None:
                                # Play buffer from connected AUDIO_FILE node
                                if hasattr(audio_off_buffer, 'numpy'):
                                    audio_np = audio_off_buffer.squeeze().numpy()
                                else:
                                    audio_np = np.array(audio_off_buffer).flatten()
                                AudioPlayer.play_buffer(audio_np, volume=volume)
                                print(f"[FACET]   {node.name}: TRIGGER OFF (playing audio_off input)")
                            elif audio_off_path:
                                # Play from file path
                                AudioPlayer.play_file(audio_off_path, volume=volume)
                                print(f"[FACET]   {node.name}: TRIGGER OFF (playing: {audio_off_path})")
                            else:
                                print(f"[FACET]   {node.name}: TRIGGER OFF (no audio configured)")

                        node_outputs[node_id] = {
                            'is_on': current_on,
                            'triggered_on': triggered_on,
                            'triggered_off': triggered_off,
                            'value': scalar_val,
                            'threshold': threshold,
                            'play_sound': play_sound
                        }
                        continue

                    if node.type == NodeType.SCRIPTED_NODE:
                        # Execute user-defined JavaScript logic
                        incoming = self.graph.get_connections_to_node(node_id)
                        script_inputs = {}

                        # Gather all inputs
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                val = src_outputs[conn.from_port]
                                # Convert tensor to scalar for script
                                if hasattr(val, 'item'):
                                    script_inputs[conn.to_port] = val.flatten()[0].item()
                                elif hasattr(val, 'tolist'):
                                    script_inputs[conn.to_port] = val.flatten().tolist()
                                else:
                                    script_inputs[conn.to_port] = val

                        script = node.params.get('script', 'return { out: 0 };')
                        script_params = {k: v for k, v in node.params.items() if k != 'script'}

                        try:
                            # Execute script in a restricted environment
                            # Using Python eval with a custom namespace (safer than exec)
                            # Convert JS-style script to Python
                            result = self._execute_script(script, script_inputs, script_params)

                            # Convert results to tensors
                            outputs_dict = {}
                            if isinstance(result, dict):
                                for key, val in result.items():
                                    if isinstance(val, (int, float)):
                                        outputs_dict[key] = torch.tensor([[val]], dtype=torch.float32)
                                    elif isinstance(val, list):
                                        outputs_dict[key] = torch.tensor([val], dtype=torch.float32)
                                    else:
                                        outputs_dict[key] = val
                            else:
                                # Single return value goes to 'out'
                                if isinstance(result, (int, float)):
                                    outputs_dict['out'] = torch.tensor([[result]], dtype=torch.float32)
                                else:
                                    outputs_dict['out'] = result

                            # Ensure 'x' output exists
                            if 'x' not in outputs_dict and 'out' in outputs_dict:
                                outputs_dict['x'] = outputs_dict['out']

                            node_outputs[node_id] = outputs_dict
                            print(f"[FACET]   {node.name}: Script executed, outputs: {list(outputs_dict.keys())}")

                        except Exception as e:
                            print(f"[FACET]   {node.name}: Script error: {e}")
                            node_outputs[node_id] = {
                                'out': torch.tensor([[0.0]], dtype=torch.float32),
                                'x': torch.tensor([[0.0]], dtype=torch.float32),
                                'error': str(e)
                            }
                        continue

                    if node.type == NodeType.SHADER_VIS:
                        # Shader visualization - gather uniform inputs
                        incoming = self.graph.get_connections_to_node(node_id)
                        uniform_value = 0.0
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                val = src_outputs[conn.from_port]
                                if hasattr(val, 'item'):
                                    uniform_value = val.item()
                                elif hasattr(val, 'flatten'):
                                    uniform_value = float(val.flatten()[0])
                                else:
                                    uniform_value = float(val)
                                break

                        # Store shader state for rendering
                        node_outputs[node_id] = {
                            'u_value': uniform_value,
                            'shader_code': node.params.get('shader_code', ''),
                            'preset': node.params.get('preset', 'custom')
                        }
                        continue

                    if node.type == NodeType.SIMPLE_EMBED:
                        # Simple embed - pass through with dimension adjustment
                        incoming = self.graph.get_connections_to_node(node_id)
                        input_tensor = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                input_tensor = src_outputs[conn.from_port]
                                break

                        if input_tensor is not None:
                            # Just pass through - TEXT_INPUT already does embedding
                            output_dim = node.params.get('output_dim', 8)
                            if input_tensor.shape[-1] < output_dim:
                                # Pad to output_dim
                                padding = output_dim - input_tensor.shape[-1]
                                input_tensor = torch.nn.functional.pad(input_tensor, (0, padding))
                            elif input_tensor.shape[-1] > output_dim:
                                # Truncate
                                input_tensor = input_tensor[..., :output_dim]
                            node_outputs[node_id] = {
                                'embedding': input_tensor,
                                'x': input_tensor
                            }
                        else:
                            # No input - output zeros
                            output_dim = node.params.get('output_dim', 8)
                            node_outputs[node_id] = {
                                'embedding': torch.zeros(1, output_dim),
                                'x': torch.zeros(1, output_dim)
                            }
                        continue

                    if node.type == NodeType.AFFECT_VIS:
                        # Affect visualizer - gather 5D affect input
                        incoming = self.graph.get_connections_to_node(node_id)
                        affect_values = {}
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                affect_values[conn.to_port] = src_outputs[conn.from_port]

                        # Get affect input
                        affect_input = affect_values.get('affect')
                        if affect_input is not None:
                            if hasattr(affect_input, 'tolist'):
                                affect_list = affect_input.flatten().tolist()
                            else:
                                affect_list = list(affect_input)
                            # Ensure 5D
                            while len(affect_list) < 5:
                                affect_list.append(0.0)
                            affect_list = affect_list[:5]
                        else:
                            affect_list = [0.0, 0.5, 0.5, 0.0, 0.0]

                        node_outputs[node_id] = {
                            'valence': affect_list[0],
                            'arousal': affect_list[1],
                            'dominance': affect_list[2],
                            'sorrow': affect_list[3],
                            'boredom': affect_list[4],
                            'affect': affect_list
                        }
                        continue

                    if node.type == NodeType.ATTENTION_VIS:
                        # Attention visualizer - gather attention weights
                        incoming = self.graph.get_connections_to_node(node_id)
                        attn_weights = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                attn_weights = src_outputs[conn.from_port]
                                break

                        if attn_weights is not None:
                            if hasattr(attn_weights, 'tolist'):
                                weights_list = attn_weights.squeeze().tolist()
                            else:
                                weights_list = attn_weights
                        else:
                            weights_list = [[1.0]]  # Placeholder

                        node_outputs[node_id] = {
                            'weights': weights_list,
                            'display': weights_list
                        }
                        continue

                    if node.type == NodeType.TOKEN_INPUT:
                        # Token input outputs token ID as tensor
                        token_id = node.params.get('token_id', 0)
                        vocab = node.params.get('vocab', [])
                        token_tensor = torch.tensor([[token_id]], dtype=torch.long)
                        node_outputs[node_id] = {
                            'token_id': token_tensor,
                            'x': token_tensor.float()
                        }
                        continue

                    if node.type == NodeType.EMBEDDING:
                        # Embedding lookup
                        incoming = self.graph.get_connections_to_node(node_id)
                        token_id = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                token_id = src_outputs[conn.from_port]
                                break

                        vocab_size = node.params.get('vocab_size', 100)
                        embed_dim = node.params.get('embed_dim', 16)

                        # Create or get embedding layer
                        embed_key = f"{node_id}_embed"
                        if embed_key not in self.hidden_states:
                            self.hidden_states[embed_key] = torch.nn.Embedding(vocab_size, embed_dim)

                        embed_layer = self.hidden_states[embed_key]

                        if token_id is not None:
                            if hasattr(token_id, 'long'):
                                idx = token_id.long().flatten()[0]
                            else:
                                idx = int(token_id)
                            idx = max(0, min(idx, vocab_size - 1))
                            embedding = embed_layer(torch.tensor([idx]))
                        else:
                            embedding = torch.zeros(1, embed_dim)

                        node_outputs[node_id] = {
                            'embedding': embedding,
                            'x': embedding
                        }
                        continue

                    if node.type == NodeType.SAMPLING:
                        # Temperature-controlled sampling from logits
                        incoming = self.graph.get_connections_to_node(node_id)
                        logits = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                logits = src_outputs[conn.from_port]
                                break

                        temperature = node.params.get('temperature', 1.0)
                        top_k = node.params.get('top_k', 0)

                        if logits is not None:
                            # Apply temperature
                            if temperature > 0:
                                scaled_logits = logits / temperature
                            else:
                                scaled_logits = logits

                            # Softmax to get probabilities
                            probs = torch.nn.functional.softmax(scaled_logits, dim=-1)

                            # Top-k filtering
                            if top_k > 0 and top_k < probs.shape[-1]:
                                topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
                                probs = torch.zeros_like(probs)
                                probs.scatter_(-1, topk_indices, topk_probs)
                                probs = probs / probs.sum(dim=-1, keepdim=True)

                            # Sample
                            sampled_idx = torch.multinomial(probs.flatten(), 1)
                        else:
                            probs = torch.ones(1, 10) / 10
                            sampled_idx = torch.tensor([0])

                        node_outputs[node_id] = {
                            'token_id': sampled_idx,
                            'probs': probs,
                            'x': sampled_idx.float()
                        }
                        continue

                    if node.type == NodeType.TOKEN_OUTPUT:
                        # Display token as text
                        incoming = self.graph.get_connections_to_node(node_id)
                        token_id = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                token_id = src_outputs[conn.from_port]
                                break

                        vocab = node.params.get('vocab', [])

                        if token_id is not None:
                            if hasattr(token_id, 'item'):
                                idx = int(token_id.flatten()[0].item())
                            else:
                                idx = int(token_id)
                            if 0 <= idx < len(vocab):
                                token_text = vocab[idx]
                            else:
                                token_text = f"[{idx}]"
                        else:
                            token_text = "[none]"
                            idx = -1

                        node_outputs[node_id] = {
                            'token_id': idx,
                            'token_text': token_text,
                            'display': token_text
                        }
                        continue

                    if node.type == NodeType.PROB_VIS:
                        # Probability distribution visualizer
                        incoming = self.graph.get_connections_to_node(node_id)
                        probs = None
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                probs = src_outputs[conn.from_port]
                                break

                        vocab = node.params.get('vocab', [])
                        top_k = node.params.get('top_k', 10)

                        if probs is not None:
                            probs_list = probs.flatten().tolist()
                            # Get top k
                            indexed = list(enumerate(probs_list))
                            indexed.sort(key=lambda x: x[1], reverse=True)
                            top_items = indexed[:top_k]

                            top_probs = []
                            for idx, prob in top_items:
                                label = vocab[idx] if idx < len(vocab) else f"[{idx}]"
                                top_probs.append({'token': label, 'prob': prob, 'idx': idx})
                        else:
                            top_probs = []
                            probs_list = []

                        node_outputs[node_id] = {
                            'probs': probs_list,
                            'top_probs': top_probs,
                            'display': top_probs
                        }
                        continue

                    if node.type == NodeType.OUTPUT_CHART:
                        # Output chart gathers input and stores in history
                        incoming = self.graph.get_connections_to_node(node_id)
                        output_values = {}
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                output_values[conn.to_port] = src_outputs[conn.from_port]

                        # Get input value
                        input_val = output_values.get('value')
                        if input_val is not None:
                            if hasattr(input_val, 'item'):
                                scalar_val = input_val.flatten()[0].item()
                            else:
                                scalar_val = float(input_val)
                        else:
                            scalar_val = 0.0

                        # Store in history (managed externally for UI)
                        history_key = f"{node_id}_history"
                        if history_key not in self.hidden_states:
                            self.hidden_states[history_key] = []
                        history = self.hidden_states[history_key]
                        history.append(scalar_val)

                        # Trim to history_length
                        max_len = node.params.get('history_length', 50)
                        if len(history) > max_len:
                            self.hidden_states[history_key] = history[-max_len:]

                        node_outputs[node_id] = {
                            'value': scalar_val,
                            'history': self.hidden_states[history_key].copy()
                        }
                        continue

                    if node.type == NodeType.COUNTER_OUTPUT:
                        # Counter output displays rounded integer
                        incoming = self.graph.get_connections_to_node(node_id)
                        output_values = {}
                        for conn in incoming:
                            src_outputs = node_outputs.get(conn.from_node, {})
                            if conn.from_port in src_outputs:
                                output_values[conn.to_port] = src_outputs[conn.from_port]

                        # Get input value
                        input_val = output_values.get('value')
                        if input_val is not None:
                            if hasattr(input_val, 'item'):
                                scalar_val = input_val.flatten()[0].item()
                            else:
                                scalar_val = float(input_val)
                        else:
                            scalar_val = 0.0

                        # Apply scale and offset
                        scale = node.params.get('scale', 1.0)
                        offset = node.params.get('offset', 0.0)
                        display_val = scalar_val * scale + offset
                        count = int(round(display_val))

                        node_outputs[node_id] = {
                            'value': scalar_val,
                            'count': count,
                            'display': count
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
                            status = "ON" if is_on else "OFF"
                            print(f"[FACET]   {node.name}: {status} (value={scalar_val:.3f}, threshold={threshold})")
                        else:
                            node_outputs[node_id] = {
                                'value': 0.0,
                                'is_on': False,
                                'threshold': node.params.get('threshold', 0.5)
                            }
                            print(f"[FACET]   {node.name}: OFF (no input)")
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

            # Log execution results
            print(f"[FACET] NeuralCanvas: Execution complete ({execution_time:.2f}ms)")

            # Log key node outputs (skip nodes that already log themselves)
            already_logged = (NodeType.INPUT, NodeType.COMMENT, NodeType.OUTPUT,
                              NodeType.NUMBER_INPUT, NodeType.THRESHOLD_OUTPUT)
            for node_id in node_order:
                node = self.graph.nodes[node_id]
                if node.type in already_logged:
                    continue
                if node_id in display_node_outputs:
                    out = display_node_outputs[node_id]
                    # Get the primary output value
                    primary_val = None
                    for key in ['out', 'x', 'value', 'affect', 'is_on']:
                        if key in out:
                            primary_val = out[key]
                            break
                    if primary_val is not None:
                        print(f"[FACET]   {node.name} ({node.type.value}): {self._format_tensor(primary_val)}")

            # Log final output
            if display_outputs:
                for key, val in display_outputs.items():
                    print(f"[FACET]   OUTPUT.{key}: {self._format_tensor(val)}")

            return TestResult(
                success=True,
                outputs=display_outputs,
                node_outputs=display_node_outputs,
                execution_time_ms=execution_time
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[FACET] NeuralCanvas: EXECUTION FAILED - {str(e)}")
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

        elif node.type == NodeType.STACK:
            # Stack tensors along sequence dimension for transformers
            # Output: (batch, seq_len, embed_dim)
            a = inputs.get('a')
            b = inputs.get('b')
            if a is not None and b is not None:
                # Ensure both are at least 2D
                if len(a.shape) == 1:
                    a = a.unsqueeze(0)  # (embed,) -> (1, embed)
                if len(b.shape) == 1:
                    b = b.unsqueeze(0)

                # If already 3D (batch, seq, embed), concat along seq
                if len(a.shape) == 3 and len(b.shape) == 3:
                    result = torch.cat([a, b], dim=1)
                elif len(a.shape) == 3:
                    # a is (batch, seq, embed), b is (batch, embed)
                    b = b.unsqueeze(1)  # -> (batch, 1, embed)
                    result = torch.cat([a, b], dim=1)
                elif len(b.shape) == 3:
                    # a is (batch, embed), b is (batch, seq, embed)
                    a = a.unsqueeze(1)
                    result = torch.cat([a, b], dim=1)
                else:
                    # Both are 2D (batch, embed) - stack into (batch, 2, embed)
                    a = a.unsqueeze(1)  # (batch, 1, embed)
                    b = b.unsqueeze(1)  # (batch, 1, embed)
                    result = torch.cat([a, b], dim=1)  # (batch, 2, embed)

                outputs['x'] = result
                outputs['out'] = result
            elif a is not None:
                if len(a.shape) == 2:
                    a = a.unsqueeze(1)  # Make it a 1-token sequence
                outputs['x'] = a
                outputs['out'] = a
            elif b is not None:
                if len(b.shape) == 2:
                    b = b.unsqueeze(1)
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

        elif node.type == NodeType.MULTI_HEAD_ATTENTION:
            # Multi-head attention: Q, K, V inputs
            query = inputs.get('query', x)
            key = inputs.get('key', x)
            value = inputs.get('value', x)

            # Ensure 3D: (batch, seq, embed)
            if len(query.shape) == 2:
                query = query.unsqueeze(1)
            if len(key.shape) == 2:
                key = key.unsqueeze(1)
            if len(value.shape) == 2:
                value = value.unsqueeze(1)

            # Forward pass with attention weights
            out, attn_weights = layer(query, key, value, need_weights=True)
            outputs['out'] = out
            outputs['x'] = out
            outputs['attn_weights'] = attn_weights

        elif node.type == NodeType.TRANSFORMER_BLOCK:
            # Complete transformer encoder layer
            # Ensure 3D: (batch, seq, embed)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            # TransformerEncoderLayer forward
            out = layer(x)
            outputs['out'] = out
            outputs['x'] = out

            # Try to get attention weights (requires hook or manual computation)
            # For now, compute attention manually for visualization
            embed_dim = node.params.get('embed_dim', 64)
            if x.shape[-1] == embed_dim:
                # Simplified attention weight computation for visualization
                # Real attention is inside the layer, this is an approximation
                attn_scores = torch.bmm(x, x.transpose(1, 2)) / (embed_dim ** 0.5)
                attn_weights = torch.softmax(attn_scores, dim=-1)
                outputs['attn_weights'] = attn_weights

        elif node.type == NodeType.POSITIONAL_ENCODING:
            # Add sinusoidal positional encoding
            max_seq_len = node.params.get('max_seq_len', 512)
            embed_dim = node.params.get('embed_dim', 64)

            # Ensure 3D: (batch, seq, embed)
            if len(x.shape) == 2:
                x = x.unsqueeze(1)

            batch_size, seq_len, d_model = x.shape

            # Generate positional encoding
            position = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                                 -(np.log(10000.0) / d_model))

            pe = torch.zeros(seq_len, d_model)
            pe[:, 0::2] = torch.sin(position * div_term[:d_model // 2 + d_model % 2])
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

            # Add positional encoding to input
            out = x + pe.unsqueeze(0)
            outputs['out'] = out
            outputs['x'] = out

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

    def _execute_script(self, script: str, inputs: Dict[str, Any], params: Dict[str, Any]) -> Any:
        """
        Execute a user-defined script.

        Supports a simple JavaScript-like syntax that gets converted to Python.
        For safety, uses a restricted eval environment.

        Args:
            script: JavaScript-style code
            inputs: Input values from connected nodes
            params: Node parameters (excluding script itself)

        Returns:
            Script result (typically a dict of outputs)
        """
        # Convert simple JS patterns to Python
        py_script = script

        # Handle 'return { ... }' -> result assignment
        # JS: return { out: value };
        # Python: result = { 'out': value }
        py_script = py_script.replace('return {', 'result = {')
        py_script = py_script.replace('return', 'result =')

        # Convert JS object syntax to Python dict
        # { key: value } -> { 'key': value }
        import re
        # Match unquoted keys in object literals
        py_script = re.sub(r'(\{|\,)\s*(\w+)\s*:', r"\1 '\2':", py_script)

        # Handle var/let/const declarations
        py_script = re.sub(r'\b(var|let|const)\s+', '', py_script)

        # Handle semicolons (make optional in Python)
        py_script = py_script.replace(';', '')

        # Handle Math.* functions
        py_script = py_script.replace('Math.sin', 'math.sin')
        py_script = py_script.replace('Math.cos', 'math.cos')
        py_script = py_script.replace('Math.abs', 'abs')
        py_script = py_script.replace('Math.sqrt', 'math.sqrt')
        py_script = py_script.replace('Math.pow', 'pow')
        py_script = py_script.replace('Math.exp', 'math.exp')
        py_script = py_script.replace('Math.log', 'math.log')
        py_script = py_script.replace('Math.floor', 'math.floor')
        py_script = py_script.replace('Math.ceil', 'math.ceil')
        py_script = py_script.replace('Math.min', 'min')
        py_script = py_script.replace('Math.max', 'max')
        py_script = py_script.replace('Math.PI', 'math.pi')
        py_script = py_script.replace('Math.random()', 'random.random()')

        # Handle inputs.a -> inputs['a']
        py_script = re.sub(r'inputs\.(\w+)', r"inputs['\1']", py_script)

        # Handle params.x -> params['x']
        py_script = re.sub(r'params\.(\w+)', r"params['\1']", py_script)

        # Build safe namespace
        import math
        import random

        safe_namespace = {
            'inputs': inputs,
            'params': params,
            'math': math,
            'random': random,
            'abs': abs,
            'min': min,
            'max': max,
            'pow': pow,
            'sum': sum,
            'len': len,
            'range': range,
            'True': True,
            'False': False,
            'None': None,
            'result': None
        }

        # Execute the converted script
        exec(py_script, {"__builtins__": {}}, safe_namespace)

        return safe_namespace.get('result', {'out': 0})


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

# ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡ ～ ♡
# જ⁀➴ ♡ Made with love. Use with love.
# Caitlyn Meeks 2026
