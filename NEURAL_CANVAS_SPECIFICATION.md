# Neural Canvas Specification

**Status:** Design Phase
**Author:** Commander Spock + Cadet Caity
**Date:** December 8, 2025
**Purpose:** Visual node-based editor for CharmNetwork neural architecture

---

## 🎯 Vision

**Neural Canvas** is a visual editor for designing and modifying the CharmNetwork's internal neural topology. When you double-click a CharmNetwork facet in the Facets Editor, Neural Canvas opens and displays the LSTM/GRU hierarchy as an editable node graph.

Think: **Blender's shader nodes, but for recurrent neural networks.**

---

## 📐 Architecture Integration

### How It Fits Into NoodleStudio

```
NoodleStudio
├─ Facets Editor (existing)
│  ├─ INCOMING node
│  ├─ CharmNetwork facet ← Double-click opens Neural Canvas
│  ├─ Context Intelligence
│  └─ OUTGOING node
│
└─ Neural Canvas (new tab/panel)
   ├─ Fast LSTM (16-D)
   ├─ Medium LSTM (16-D)
   ├─ Slow GRU (8-D)
   ├─ Affect Head (5-D output)
   └─ Custom layers...
```

**Interaction Flow:**
1. User double-clicks CharmNetwork facet in Facets Editor
2. Neural Canvas tab opens (or switches to if already open)
3. Canvas displays current network topology loaded from `.nncanvas` file
4. User edits topology visually (add/remove/configure layers)
5. Save → Generates Python code + `.nncanvas` JSON
6. Reload facet → New topology active

---

## 🧩 Node Types

### Core Recurrent Layers

| Node Type | Icon | Inputs | Outputs | Configurable Parameters |
|-----------|------|--------|---------|------------------------|
| **LSTM** | 📦 | `x` (features)<br>`h` (hidden)<br>`c` (cell) | `h_out`<br>`c_out` | `hidden_dim` (8-128)<br>`dropout` (0-0.5)<br>`bias` (bool) |
| **GRU** | ⚙️ | `x`<br>`h` | `h_out` | `hidden_dim`<br>`dropout`<br>`bias` |
| **RNN** | 🔄 | `x`<br>`h` | `h_out` | `hidden_dim`<br>`activation` (tanh/relu) |

### Feedforward Layers

| Node Type | Icon | Inputs | Outputs | Parameters |
|-----------|------|--------|---------|-----------|
| **Linear** | ➡️ | `x` | `out` | `in_features`<br>`out_features`<br>`bias` |
| **Conv1D** | 🌊 | `x` | `out` | `in_channels`<br>`out_channels`<br>`kernel_size`<br>`stride`<br>`padding` |
| **Attention** | 👁️ | `query`<br>`key`<br>`value` | `out`<br>`weights` | `num_heads`<br>`embed_dim`<br>`dropout` |

### Activation Functions

| Node Type | Icon | Inputs | Outputs | Parameters |
|-----------|------|--------|---------|-----------|
| **Tanh** | 〰️ | `x` | `out` | None |
| **ReLU** | ⚡ | `x` | `out` | None |
| **GELU** | 🌀 | `x` | `out` | None |
| **Sigmoid** | 📈 | `x` | `out` | None |
| **Softmax** | 🎲 | `x` | `out` | `dim` (axis) |

### Normalization

| Node Type | Icon | Inputs | Outputs | Parameters |
|-----------|------|--------|---------|-----------|
| **LayerNorm** | 📏 | `x` | `out` | `normalized_shape`<br>`eps`<br>`elementwise_affine` |
| **BatchNorm** | 📊 | `x` | `out` | `num_features`<br>`eps`<br>`momentum` |

### Dropout & Regularization

| Node Type | Icon | Inputs | Outputs | Parameters |
|-----------|------|--------|---------|-----------|
| **Dropout** | 💧 | `x` | `out` | `p` (0.0-0.9) |

### Special Nodes

| Node Type | Icon | Inputs | Outputs | Purpose |
|-----------|------|--------|---------|---------|
| **INPUT** | 🔵 | — | `affect` (5-D) | Network entry point |
| **OUTPUT** | 🟢 | `state` (40-D) | — | Phenomenal state output |
| **AFFECT_HEAD** | 🎭 | `state` (40-D) | `affect` (5-D) | Maps state → continuous affect |
| **STATE_CONCAT** | ➕ | `fast` (16-D)<br>`medium` (16-D)<br>`slow` (8-D) | `state` (40-D) | Concatenates hidden states |
| **STATE_SPLIT** | ➖ | `state` (40-D) | `fast` (16-D)<br>`medium` (16-D)<br>`slow` (8-D) | Splits phenomenal state |

### Quantum/Experimental

| Node Type | Icon | Inputs | Outputs | Parameters |
|-----------|------|--------|---------|-----------|
| **QuantumMicrotubule** | ⚛️ | `x`<br>`mt_state` | `out`<br>`new_mt_state` | `hidden_dim`<br>`collapse_threshold`<br>`coherence_time`<br>`noise_scale`<br>`use_collapse`<br>`use_entanglement` |
| **IBMQuantum** | 🔮 | `classical_state` | `quantum_result` | `num_qubits`<br>`shots`<br>`backend` (simulator/hardware) |
| **EntropyInjection** | 🎲 | `x` | `out` | `noise_scale`<br>`use_hardware_rng` (bool)<br>`distribution` (uniform/avalanche) |

---

## 📄 File Format: `.nncanvas`

Neural Canvas saves network topologies in a JSON format with full type safety and validation.

### Example: Current CharmNetwork Topology

```json
{
  "version": "1.0",
  "name": "CharmNetwork Default",
  "description": "Hierarchical temporal affect processor (Fast LSTM + Medium LSTM + Slow GRU)",
  "metadata": {
    "created": "2025-12-08T19:30:00Z",
    "modified": "2025-12-08T19:30:00Z",
    "author": "Caity",
    "total_parameters": 132500
  },
  "nodes": [
    {
      "id": "input",
      "type": "INPUT",
      "name": "Affect Input",
      "position": [100, 300],
      "params": {
        "output_dim": 5
      },
      "outputs": {
        "affect": {"shape": [5], "dtype": "float32"}
      }
    },
    {
      "id": "fast_lstm",
      "type": "LSTM",
      "name": "Fast LSTM (seconds)",
      "position": [300, 200],
      "params": {
        "input_dim": 5,
        "hidden_dim": 16,
        "dropout": 0.0,
        "bias": true
      },
      "inputs": {
        "x": {"from": "input", "port": "affect"},
        "h": {"internal": "h_fast"},
        "c": {"internal": "c_fast"}
      },
      "outputs": {
        "h_out": {"shape": [16], "dtype": "float32"},
        "c_out": {"shape": [16], "dtype": "float32"}
      },
      "weights": {
        "weight_ih": {"shape": [64, 5], "path": "checkpoints/fast_lstm_ih.npy"},
        "weight_hh": {"shape": [64, 16], "path": "checkpoints/fast_lstm_hh.npy"},
        "bias_ih": {"shape": [64], "path": "checkpoints/fast_lstm_bias_ih.npy"},
        "bias_hh": {"shape": [64], "path": "checkpoints/fast_lstm_bias_hh.npy"}
      }
    },
    {
      "id": "medium_lstm",
      "type": "LSTM",
      "name": "Medium LSTM (minutes)",
      "position": [300, 400],
      "params": {
        "input_dim": 16,
        "hidden_dim": 16,
        "dropout": 0.0,
        "bias": true
      },
      "inputs": {
        "x": {"from": "fast_lstm", "port": "h_out"},
        "h": {"internal": "h_medium"},
        "c": {"internal": "c_medium"}
      },
      "outputs": {
        "h_out": {"shape": [16], "dtype": "float32"},
        "c_out": {"shape": [16], "dtype": "float32"}
      },
      "weights": {
        "weight_ih": {"shape": [64, 16], "path": "checkpoints/medium_lstm_ih.npy"},
        "weight_hh": {"shape": [64, 16], "path": "checkpoints/medium_lstm_hh.npy"},
        "bias_ih": {"shape": [64], "path": "checkpoints/medium_lstm_bias_ih.npy"},
        "bias_hh": {"shape": [64], "path": "checkpoints/medium_lstm_bias_hh.npy"}
      }
    },
    {
      "id": "slow_gru",
      "type": "GRU",
      "name": "Slow GRU (hours/days)",
      "position": [300, 600],
      "params": {
        "input_dim": 16,
        "hidden_dim": 8,
        "dropout": 0.0,
        "bias": true
      },
      "inputs": {
        "x": {"from": "medium_lstm", "port": "h_out"},
        "h": {"internal": "h_slow"}
      },
      "outputs": {
        "h_out": {"shape": [8], "dtype": "float32"}
      },
      "weights": {
        "weight_ih": {"shape": [24, 16], "path": "checkpoints/slow_gru_ih.npy"},
        "weight_hh": {"shape": [24, 8], "path": "checkpoints/slow_gru_hh.npy"},
        "bias_ih": {"shape": [24], "path": "checkpoints/slow_gru_bias_ih.npy"},
        "bias_hh": {"shape": [24], "path": "checkpoints/slow_gru_bias_hh.npy"}
      }
    },
    {
      "id": "state_concat",
      "type": "STATE_CONCAT",
      "name": "Phenomenal State",
      "position": [500, 400],
      "inputs": {
        "fast": {"from": "fast_lstm", "port": "h_out"},
        "medium": {"from": "medium_lstm", "port": "h_out"},
        "slow": {"from": "slow_gru", "port": "h_out"}
      },
      "outputs": {
        "state": {"shape": [40], "dtype": "float32"}
      }
    },
    {
      "id": "affect_head",
      "type": "AFFECT_HEAD",
      "name": "Affect Predictor",
      "position": [700, 400],
      "params": {
        "state_dim": 40,
        "affect_dim": 5,
        "hidden_dim": 32
      },
      "inputs": {
        "state": {"from": "state_concat", "port": "state"}
      },
      "outputs": {
        "valence": {"shape": [1], "dtype": "float32"},
        "arousal": {"shape": [1], "dtype": "float32"},
        "fear": {"shape": [1], "dtype": "float32"},
        "sorrow": {"shape": [1], "dtype": "float32"},
        "boredom": {"shape": [1], "dtype": "float32"}
      },
      "weights": {
        "fc1_weight": {"shape": [32, 40], "path": "checkpoints/affect_head_fc1.npy"},
        "fc1_bias": {"shape": [32], "path": "checkpoints/affect_head_fc1_bias.npy"},
        "fc2_weight": {"shape": [5, 32], "path": "checkpoints/affect_head_fc2.npy"},
        "fc2_bias": {"shape": [5], "path": "checkpoints/affect_head_fc2_bias.npy"}
      }
    },
    {
      "id": "output",
      "type": "OUTPUT",
      "name": "Network Output",
      "position": [900, 400],
      "inputs": {
        "state": {"from": "state_concat", "port": "state"},
        "affect": {"from": "affect_head", "port": "valence"}
      }
    }
  ],
  "connections": [
    {"from_node": "input", "from_port": "affect", "to_node": "fast_lstm", "to_port": "x"},
    {"from_node": "fast_lstm", "from_port": "h_out", "to_node": "medium_lstm", "to_port": "x"},
    {"from_node": "medium_lstm", "from_port": "h_out", "to_node": "slow_gru", "to_port": "x"},
    {"from_node": "fast_lstm", "from_port": "h_out", "to_node": "state_concat", "to_port": "fast"},
    {"from_node": "medium_lstm", "from_port": "h_out", "to_node": "state_concat", "to_port": "medium"},
    {"from_node": "slow_gru", "from_port": "h_out", "to_node": "state_concat", "to_port": "slow"},
    {"from_node": "state_concat", "from_port": "state", "to_node": "affect_head", "to_port": "state"},
    {"from_node": "state_concat", "from_port": "state", "to_node": "output", "to_port": "state"},
    {"from_node": "affect_head", "from_port": "valence", "to_node": "output", "to_port": "affect"}
  ],
  "hidden_states": [
    {"id": "h_fast", "shape": [16], "initial_value": "zeros"},
    {"id": "c_fast", "shape": [16], "initial_value": "zeros"},
    {"id": "h_medium", "shape": [16], "initial_value": "zeros"},
    {"id": "c_medium", "shape": [16], "initial_value": "zeros"},
    {"id": "h_slow", "shape": [8], "initial_value": "zeros"}
  ],
  "training_config": {
    "optimizer": "adamw",
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "batch_size": 32,
    "max_epochs": 100
  },
  "export_targets": {
    "mlx": true,
    "pytorch": false,
    "onnx": false
  }
}
```

---

## 🎨 UI/UX Design

### Panel Layout (New Tab in NoodleStudio)

```
╔══════════════════════════════════════════════════════════════╗
║ Neural Canvas - CharmNetwork Default                    [X] ║
╟──────────────────────────────────────────────────────────────╢
║ [File ▾] [Edit ▾] [View ▾] [Validate ▾] [Export ▾]         ║
╟──────────────────────────────────────────────────────────────╢
║ 🔍 100% | Grid: ON | Snap: ON | ● INPUT ● LSTM ● OUTPUT    ║
╠════════════╦═════════════════════════════════════╦══════════╣
║            ║                                     ║          ║
║  Node      ║         Canvas Area                 ║  Inspec- ║
║  Palette   ║                                     ║  tor     ║
║            ║   ┌─────┐                          ║          ║
║ ● INPUT    ║   │INPUT│                          ║ Selected ║
║ ● LSTM     ║   └──┬──┘                          ║ Node:    ║
║ ● GRU      ║      │                             ║          ║
║ ● Linear   ║   ┌──▼────┐                        ║ LSTM     ║
║ ● Tanh     ║   │ LSTM  │  ← SELECTED           ║          ║
║ ● Dropout  ║   │ Fast  │                        ║ hidden_  ║
║ ● Concat   ║   └───┬───┘                        ║ dim: 16  ║
║ ...        ║       │                             ║          ║
║            ║   ┌───▼────┐                       ║ dropout: ║
║ [Quantum]  ║   │  LSTM  │                       ║ 0.0      ║
║ ⚛️ Micro-  ║   │ Medium │                       ║          ║
║   tubule   ║   └───┬────┘                       ║ [Apply]  ║
║ 🔮 IBM     ║       │                             ║          ║
║   Quantum  ║       ...                           ║          ║
║            ║                                     ║          ║
╠════════════╩═════════════════════════════════════╩══════════╣
║ Parameters: 132,500 | Layers: 7 | Trainable: Yes           ║
╚══════════════════════════════════════════════════════════════╝
```

### Interaction Patterns

**Adding Nodes:**
- Drag from Node Palette → Canvas
- OR: Right-click canvas → "Add Node" → Select type
- Automatically snaps to grid (optional)

**Connecting Nodes:**
- Click output port (right side of node) → Drag → Drop on input port (left side)
- Wire routing: Orthogonal (Manhattan) with 90° angles only (matches Facets Editor)
- Invalid connections show red preview (type mismatch)
- Valid connections show green preview

**Configuring Nodes:**
- Click node → Inspector panel shows editable parameters
- Real-time validation (e.g., hidden_dim must be 1-512)
- Weight paths editable (browse to .npy files)

**Node Visual States:**
- **Normal:** Gray border
- **Selected:** Cyan border (matches Facets Editor selection color)
- **Error:** Red border (validation failed)
- **Training:** Yellow pulsing border (when training in progress)

### Wire Coloring (Data Type Indication)

| Data Type | Wire Color | Example |
|-----------|------------|---------|
| Affect (5-D) | 🟢 Green | INPUT → LSTM |
| Hidden State | 🔵 Blue | LSTM h_out → LSTM h |
| Phenomenal State (40-D) | 🟣 Purple | STATE_CONCAT → AFFECT_HEAD |
| Scalar | 🟡 Yellow | Loss, metrics |
| Error/Invalid | 🔴 Red | Type mismatch |

---

## 🔧 Implementation Architecture

### Tech Stack

- **Language:** Python 3.11+
- **GUI Framework:** PyQt6 (matches existing NoodleStudio)
- **Graphics:** QPainter for custom node rendering (matches Facets Editor approach)
- **Neural Framework:** MLX (primary), PyTorch (export option)
- **Serialization:** JSON (.nncanvas format)

### Key Classes

```python
# neural_canvas_panel.py
class NeuralCanvasPanel(QWidget):
    """Main panel for Neural Canvas editor."""
    def __init__(self, parent=None):
        self.canvas = NeuralCanvasView()
        self.palette = NodePalettePanel()
        self.inspector = NodeInspectorPanel()
        self.graph = NeuralGraph()  # Data model

    def open_from_nncanvas(self, path: str):
        """Load topology from .nncanvas JSON."""
        pass

    def save_to_nncanvas(self, path: str):
        """Save topology to .nncanvas JSON."""
        pass

    def export_to_mlx(self, path: str):
        """Generate MLX Python code from graph."""
        pass

# neural_graph.py
class NeuralGraph:
    """Data model for neural network topology."""
    def __init__(self):
        self.nodes: List[NeuralNode] = []
        self.connections: List[Connection] = []
        self.hidden_states: Dict[str, HiddenState] = {}

    def validate(self) -> ValidationResult:
        """Check for cycles, type mismatches, dimension errors."""
        pass

    def compute_total_parameters(self) -> int:
        """Count trainable parameters."""
        pass

    def to_json(self) -> dict:
        """Serialize to .nncanvas format."""
        pass

    @staticmethod
    def from_json(data: dict) -> 'NeuralGraph':
        """Deserialize from .nncanvas format."""
        pass

# neural_node.py
class NeuralNode:
    """A single node in the network (LSTM, Linear, etc.)."""
    def __init__(self, node_type: NodeType, name: str):
        self.id: str = str(uuid.uuid4())
        self.type: NodeType = node_type
        self.name: str = name
        self.position: Tuple[int, int] = (0, 0)
        self.params: Dict[str, Any] = {}
        self.inputs: Dict[str, Port] = {}
        self.outputs: Dict[str, Port] = {}
        self.weights: Dict[str, WeightInfo] = {}

    def validate_params(self) -> List[str]:
        """Validate parameter values (e.g., hidden_dim > 0)."""
        pass

    def compute_output_shapes(self) -> Dict[str, Tuple]:
        """Calculate output tensor shapes from input shapes."""
        pass

# mlx_codegen.py
class MLXCodeGenerator:
    """Generate MLX Python code from NeuralGraph."""
    def __init__(self, graph: NeuralGraph):
        self.graph = graph

    def generate_model_class(self) -> str:
        """Generate nn.Module subclass code."""
        pass

    def generate_forward_method(self) -> str:
        """Generate forward() pass logic."""
        pass

    def generate_checkpoint_loader(self) -> str:
        """Generate weight loading code."""
        pass
```

### File Structure

```
applications/noodlestudio/noodlestudio/
├── panels/
│   ├── neural_canvas_panel.py       # Main panel (new)
│   ├── neural_canvas_view.py        # Canvas rendering (new)
│   ├── node_palette_panel.py        # Node type picker (new)
│   └── node_inspector_panel.py      # Property editor (extends existing)
│
├── core/
│   ├── neural_graph.py              # Graph data model (new)
│   ├── neural_node.py               # Node definitions (new)
│   ├── neural_connection.py         # Wire/connection logic (new)
│   ├── mlx_codegen.py               # MLX code generation (new)
│   ├── pytorch_codegen.py           # PyTorch export (new)
│   └── graph_validator.py           # Validation logic (new)
│
└── formats/
    ├── nncanvas_loader.py           # .nncanvas JSON parser (new)
    └── nncanvas_saver.py            # .nncanvas JSON writer (new)

facet_assemblies/
└── charm_networks/
    ├── default.nncanvas             # Default CharmNetwork (new)
    ├── minimal.nncanvas             # Minimal test network (new)
    └── experimental_quantum.nncanvas # With quantum layers (new)
```

---

## 🚀 Workflow Examples

### Example 1: Modify Existing CharmNetwork

**Goal:** Increase Fast LSTM hidden dimension from 16 → 32

1. Open NoodleStudio → Facets Editor
2. Double-click CharmNetwork facet
3. Neural Canvas opens, loads `default.nncanvas`
4. Click "Fast LSTM" node
5. Inspector shows `hidden_dim: 16`
6. Change to `32` → Press Enter
7. Network auto-validates (updates downstream shapes)
8. Save → `charm_network_32d.nncanvas`
9. Export → Generates `charm_network_32d.py` (MLX code)
10. Update CharmNetwork facet config to use new topology

### Example 2: Add Quantum Microtubule Layer

**Goal:** Insert quantum layer between Medium LSTM and Slow GRU

1. Open Neural Canvas (double-click CharmNetwork)
2. Drag "QuantumMicrotubule" node from palette to canvas
3. Position between Medium LSTM and Slow GRU nodes
4. Delete existing wire: Medium LSTM → Slow GRU
5. Create new wires:
   - Medium LSTM `h_out` → QuantumMicrotubule `x`
   - QuantumMicrotubule `out` → Slow GRU `x`
6. Configure QuantumMicrotubule in Inspector:
   - `hidden_dim: 16`
   - `collapse_threshold: 0.5`
   - `use_entanglement: true`
7. Validate → All green
8. Save → `charm_network_quantum.nncanvas`
9. Export → Generates code with quantum layer integrated

### Example 3: Design New Architecture from Scratch

**Goal:** Build a simple 2-layer LSTM affect classifier

1. Create new canvas (File → New)
2. Add nodes:
   - INPUT (affect 5-D)
   - LSTM (hidden=32)
   - LSTM (hidden=16)
   - Linear (out=5)
   - Softmax
   - OUTPUT
3. Connect:
   - INPUT → LSTM1
   - LSTM1 → LSTM2
   - LSTM2 → Linear → Softmax → OUTPUT
4. Configure training:
   - Optimizer: AdamW
   - LR: 0.001
   - Epochs: 50
5. Save → `simple_classifier.nncanvas`
6. Export → `simple_classifier.py`
7. Train using generated code

---

## 🧪 Validation Rules

Neural Canvas validates graphs in real-time:

### Structural Validation

- ✅ **No cycles:** Network must be a DAG (Directed Acyclic Graph)
- ✅ **Single input/output:** Must have exactly 1 INPUT and 1 OUTPUT node
- ✅ **All ports connected:** No dangling wires
- ✅ **Type compatibility:** Wire data types must match

### Dimensional Validation

- ✅ **Shape propagation:** Output shapes computed from input shapes
- ✅ **Dimension matching:** e.g., LSTM `input_dim` must match incoming wire dimension
- ✅ **Recurrent state consistency:** Hidden states must match across timesteps

### Parameter Validation

- ✅ **Range checks:** e.g., `dropout` in [0, 1], `hidden_dim` > 0
- ✅ **Compatibility:** e.g., `num_heads` must divide `embed_dim` evenly
- ✅ **File existence:** Weight paths point to valid .npy files

### Training Validation

- ✅ **Differentiability:** All operations support backpropagation
- ✅ **Numerical stability:** No exp() on unbounded inputs, etc.

**Validation UI:**
- Green checkmark icon: ✅ Valid
- Red X icon: ❌ Errors (hover shows details)
- Yellow warning icon: ⚠️ Warnings (e.g., "Untrained weights")

---

## 📤 Export Targets

### MLX (Primary)

**Output:** Python file with `nn.Module` subclass

```python
# Generated by Neural Canvas
# Architecture: CharmNetwork Default
# Date: 2025-12-08

import mlx.core as mx
import mlx.nn as nn

class CharmNetworkDefault(nn.Module):
    def __init__(self):
        super().__init__()

        # Fast LSTM (seconds timescale)
        self.fast_lstm = nn.LSTM(input_size=5, hidden_size=16)

        # Medium LSTM (minutes timescale)
        self.medium_lstm = nn.LSTM(input_size=16, hidden_size=16)

        # Slow GRU (hours/days timescale)
        self.slow_gru = nn.GRU(input_size=16, hidden_size=8)

        # Affect head
        self.affect_fc1 = nn.Linear(40, 32)
        self.affect_fc2 = nn.Linear(32, 5)

        # Hidden states
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))

    def forward(self, affect_input):
        # Fast layer
        fast_out, (self.h_fast, self.c_fast) = self.fast_lstm(
            affect_input, (self.h_fast, self.c_fast)
        )

        # Medium layer
        medium_out, (self.h_medium, self.c_medium) = self.medium_lstm(
            fast_out, (self.h_medium, self.c_medium)
        )

        # Slow layer
        slow_out, self.h_slow = self.slow_gru(
            medium_out, self.h_slow
        )

        # Concatenate phenomenal state
        state = mx.concatenate([
            self.h_fast, self.h_medium, self.h_slow
        ], axis=-1)  # (1, 40)

        # Affect prediction
        affect_hidden = mx.tanh(self.affect_fc1(state))
        affect_out = self.affect_fc2(affect_hidden)  # (1, 5)

        return {
            'phenomenal_state': state,
            'affect': affect_out,
            'valence': affect_out[0, 0],
            'arousal': affect_out[0, 1],
            'fear': affect_out[0, 2],
            'sorrow': affect_out[0, 3],
            'boredom': affect_out[0, 4]
        }

    def load_weights(self, checkpoint_path: str):
        """Load weights from .npz checkpoint."""
        weights = mx.load(checkpoint_path)
        self.load_weights_dict(weights)

    def reset_state(self):
        """Reset hidden states."""
        self.h_fast = mx.zeros((1, 16))
        self.c_fast = mx.zeros((1, 16))
        self.h_medium = mx.zeros((1, 16))
        self.c_medium = mx.zeros((1, 16))
        self.h_slow = mx.zeros((1, 8))
```

### PyTorch (Optional)

Similar structure but using `torch.nn` modules.

### ONNX (Optional)

Export to ONNX for deployment on non-MLX platforms.

---

## 🎓 Training Integration

Neural Canvas can optionally integrate training:

### Phase 1: Export Only (MVP)
- Generate code, user trains manually

### Phase 2: Training Panel (Future)
```
╔══════════════════════════════════════════╗
║ Training - CharmNetwork Default          ║
╟──────────────────────────────────────────╢
║ Dataset: affect_corpus_2025.npz          ║
║ Optimizer: AdamW | LR: 0.0003            ║
║ Batch: 32 | Epochs: 100                  ║
║                                          ║
║ Epoch: 23/100 ████████░░░░░░░ 65%       ║
║ Loss: 0.0234 ↓                           ║
║                                          ║
║ [📈 View Metrics] [⏸️ Pause] [💾 Save]   ║
╚══════════════════════════════════════════╝
```

**Features:**
- Live loss plots (valence, arousal, etc.)
- Early stopping
- Checkpoint saving
- TensorBoard integration

---

## 🔮 Future Enhancements

### v1.0 (MVP)
- [x] Visual node editor
- [x] .nncanvas JSON format
- [x] MLX code generation
- [x] Real-time validation
- [x] Quantum layer support

### v1.5
- [ ] Training panel integration
- [ ] Live gradient visualization
- [ ] Architecture search (auto-generate topologies)
- [ ] PyTorch export

### v2.0
- [ ] Community canvas library (Asset Store integration)
- [ ] Pretrained model zoo (one-click add BERT, GPT layers)
- [ ] Collaborative editing (multi-user)
- [ ] Visual debugging (step through forward pass)

### v3.0
- [ ] AutoML: "Build me a classifier" → generates architecture
- [ ] Neural Architecture Search (NAS) integration
- [ ] Deployment targets (iOS, edge devices)

---

## 📋 Implementation Checklist

### Phase 1: Core Editor (2 weeks)
- [ ] Create `NeuralCanvasPanel` class
- [ ] Implement node palette (draggable node types)
- [ ] Implement canvas rendering (QPainter-based)
- [ ] Implement wire routing (orthogonal)
- [ ] Implement node inspector (property editing)
- [ ] Add zoom/pan controls
- [ ] Add grid snapping

### Phase 2: Data Model (1 week)
- [ ] Define `NeuralGraph` class
- [ ] Define `NeuralNode` class
- [ ] Define `Connection` class
- [ ] Implement .nncanvas JSON serialization
- [ ] Implement .nncanvas JSON deserialization
- [ ] Add validation logic (cycles, types, shapes)

### Phase 3: Code Generation (1 week)
- [ ] Implement `MLXCodeGenerator`
- [ ] Generate `__init__()` method
- [ ] Generate `forward()` method
- [ ] Generate weight loading code
- [ ] Generate checkpoint saving code
- [ ] Test generated code executes correctly

### Phase 4: Integration (3 days)
- [ ] Add "Neural Canvas" tab to NoodleStudio
- [ ] Hook up double-click on CharmNetwork facet
- [ ] Create default.nncanvas from existing CharmNetwork
- [ ] Test round-trip: .nncanvas → MLX code → execution
- [ ] Update CharmNetworkFacet to load from .nncanvas

### Phase 5: Polish (3 days)
- [ ] Add keyboard shortcuts (Cmd+S, Cmd+Z, etc.)
- [ ] Add validation error tooltips
- [ ] Add parameter count display
- [ ] Add example .nncanvas files
- [ ] Write user documentation

---

## 🎯 Success Criteria

Neural Canvas is successful when:

1. ✅ Caity can double-click CharmNetwork facet → Canvas opens
2. ✅ Canvas accurately displays current 3-layer hierarchy
3. ✅ Caity can modify hidden dimensions → Validation passes
4. ✅ Export generates working MLX code
5. ✅ Generated code loads existing checkpoints
6. ✅ Red Fire Anklebiter runs with modified CharmNetwork
7. ✅ Quantum layers can be visually added/configured
8. ✅ Complete workflow takes < 5 minutes (from idea to deployed)

---

## 🧠 Philosophy

Neural Canvas embodies the same principles as NoodleStudio:

- **Visual-first:** See the architecture, understand the flow
- **Unity prefab model:** Networks are shareable .nncanvas files
- **No black boxes:** Every connection, every parameter visible
- **Production-grade:** Generated code is clean, readable, maintainable
- **Monochrome aesthetic:** Gray gradients, circuit-board elegance

This is not a toy. This is how professionals will design consciousness architectures.

---

**End of Specification**

**Next Steps:**
1. Review with Caity
2. Create IBM Quantum integration strategy (separate doc)
3. Begin Phase 1 implementation (core editor)

*Ordnung muss sein.*
