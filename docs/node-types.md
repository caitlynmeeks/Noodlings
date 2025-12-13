# Neural Canvas Node Types

Complete reference for all 26 node types available in the Neural Canvas.

## I/O Nodes

### Input
Entry point for neural networks. Accepts 5-D affect vectors.

**Parameters:**

- `output_dim` (int) - Output dimension (default: 5)

**Ports:**

- Output: `affect` - Affect (5-D)

**Color:** Deep forest green (#2A4A2A)

---

### Output
Exit point for neural networks.

**Parameters:** None

**Ports:**

- Input: `state` - State (40-D)

**Color:** Deep forest green (#2A4A2A)

---

## Recurrent Layers

### LSTM
Long Short-Term Memory recurrent layer. Best for learning long-range dependencies.

**Parameters:**

- `input_dim` (int) - Input dimension (default: 5)
- `hidden_dim` (int) - Hidden layer size (default: 16)
- `dropout` (float) - Dropout rate (default: 0.0)
- `bias` (bool) - Use bias (default: true)

**Ports:**

- Inputs: `x` (Input), `h` (Hidden), `c` (Cell)
- Outputs: `h_out` (Hidden), `c_out` (Cell)

**Weights:** `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` (4 gates)

**Color:** Deeper plum (#4A2A4A)

**Usage:**
```javascript
var lstm = network.create_node("LSTM", {
    hidden_dim: 32,
    position: [100, 200]
});
```

---

### GRU
Gated Recurrent Unit layer. Simpler than LSTM, often just as effective.

**Parameters:**

- `input_dim` (int) - Input dimension (default: 16)
- `hidden_dim` (int) - Hidden layer size (default: 8)
- `dropout` (float) - Dropout rate (default: 0.0)
- `bias` (bool) - Use bias (default: true)

**Ports:**

- Inputs: `x` (Input), `h` (Hidden)
- Outputs: `h_out` (Hidden)

**Weights:** `weight_ih`, `weight_hh`, `bias_ih`, `bias_hh` (3 gates)

**Color:** Deeper plum (#4A2A4A)

---

### RNN
Simple recurrent neural network layer.

**Parameters:**

- `input_dim` (int) - Input dimension (default: 5)
- `hidden_dim` (int) - Hidden layer size (default: 16)
- `activation` (string) - Activation function (default: "tanh")

**Ports:**

- Inputs: `x` (Input), `h` (Hidden)
- Outputs: `h_out` (Hidden)

**Color:** Deep purple (#673AB7)

---

## Feedforward Layers

### Linear
Fully connected linear transformation (Dense layer).

**Parameters:**

- `in_features` (int) - Input features (default: 16)
- `out_features` (int) - Output features (default: 32)
- `bias` (bool) - Use bias (default: true)

**Ports:**

- Input: `x` (Input)
- Output: `out` (Output)

**Weights:** `weight`, `bias`

**Color:** Purple (#9C27B0)

---

### Conv1D
1D convolutional layer for sequential data.

**Parameters:**

- `in_channels` (int) - Input channels (default: 1)
- `out_channels` (int) - Output channels (default: 16)
- `kernel_size` (int) - Kernel size (default: 3)
- `stride` (int) - Stride (default: 1)
- `padding` (int) - Padding (default: 1)

**Ports:**

- Input: `x` (Input)
- Output: `out` (Output)

**Color:** Pink (#E91E63)

---

## Attention Mechanisms

### Attention
Scaled dot-product attention mechanism.

**Parameters:**

- `embed_dim` (int) - Embedding dimension (default: 64)
- `dropout` (float) - Dropout rate (default: 0.1)

**Ports:**

- Inputs: `query`, `key`, `value`
- Outputs: `out`, `weights` (attention weights)

**Color:** Red (#F44336)

---

### Multi-Head Attention
Multi-head attention (Transformer architecture).

**Parameters:**

- `embed_dim` (int) - Embedding dimension (default: 64)
- `num_heads` (int) - Number of attention heads (default: 4)
- `dropout` (float) - Dropout rate (default: 0.1)

**Ports:**

- Inputs: `query`, `key`, `value`
- Output: `out`

**Color:** Deep orange (#FF5722)

---

## Activation Functions

### Tanh
Hyperbolic tangent activation. Output range: [-1, 1]

**Parameters:** None

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Dark charcoal (#3A3A3A)

---

### ReLU
Rectified Linear Unit. Most common activation.

**Parameters:** None

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Dark charcoal (#3A3A3A)

---

### GELU
Gaussian Error Linear Unit. Used in modern transformers.

**Parameters:** None

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Dark charcoal (#3A3A3A)

---

### Sigmoid
Sigmoid activation. Output range: [0, 1]

**Parameters:** None

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Dark charcoal (#3A3A3A)

---

### Softmax
Softmax activation for probability distributions.

**Parameters:**

- `dim` (int) - Dimension to apply softmax (default: -1)

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Light green (#8BC34A)

---

## Normalization

### Layer Normalization
Layer normalization. Normalizes across features.

**Parameters:**

- `normalized_shape` (tuple) - Shape to normalize (default: (16,))
- `eps` (float) - Epsilon for numerical stability (default: 1e-5)
- `elementwise_affine` (bool) - Learnable affine parameters (default: true)

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Green (#4CAF50)

---

### Batch Normalization
Batch normalization. Normalizes across batch.

**Parameters:**

- `num_features` (int) - Number of features (default: 16)
- `eps` (float) - Epsilon (default: 1e-5)
- `momentum` (float) - Momentum for running stats (default: 0.1)

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Teal (#009688)

---

## Regularization

### Dropout
Dropout regularization. Randomly zeros elements during training.

**Parameters:**

- `p` (float) - Dropout probability (default: 0.5)

**Ports:**

- Input: `x`
- Output: `out`

**Color:** Cyan (#00BCD4)

---

## CharmNetwork Special Nodes

### State Concatenation
Concatenate multiple hidden states into phenomenal state (40-D).

**Parameters:** None

**Ports:**

- Inputs: `fast` (16-D), `medium` (16-D), `slow` (8-D)
- Output: `state` (Phenomenal State 40-D)

**Color:** Deeper teal (#2A4A4A)

**Usage:** Combines Fast LSTM (16-D) + Medium LSTM (16-D) + Slow GRU (8-D) → 40-D state

---

### State Split
Split phenomenal state into components.

**Parameters:** None

**Ports:**

- Input: `state` (Phenomenal State 40-D)
- Outputs: `fast` (16-D), `medium` (16-D), `slow` (8-D)

**Color:** Light blue (#03A9F4)

---

### Affect Head
Maps phenomenal state to 5-D continuous affect (PAD + boredom + sorrow).

**Parameters:**

- `state_dim` (int) - State dimension (default: 40)
- `affect_dim` (int) - Affect dimension (default: 5)
- `hidden_dim` (int) - Hidden layer size (default: 32)

**Ports:**

- Input: `state` (State 40-D)
- Outputs: `valence`, `arousal`, `fear`, `sorrow`, `boredom` (scalars)

**Weights:** Two-layer MLP: `fc1_weight`, `fc1_bias`, `fc2_weight`, `fc2_bias`

**Color:** Deeper tobacco brown (#4A3A2A)

**Usage:** Final layer of CharmNetwork, produces continuous affect values

---

## Quantum Nodes

### Quantum Microtubule
Penrose-Hameroff quantum consciousness layer. Simulated quantum collapse in microtubules.

**Parameters:**

- `input_dim` (int) - Input dimension (default: 16)
- `hidden_dim` (int) - Hidden dimension (default: 16)
- `collapse_threshold` (float) - Threshold for quantum collapse (default: 0.5)
- `coherence_time` (int) - Coherence time steps (default: 10)
- `entanglement_range` (int) - Range for entanglement (default: 3)
- `noise_scale` (float) - Quantum noise scale (default: 0.1)
- `use_collapse` (bool) - Enable collapse dynamics (default: true)
- `use_entanglement` (bool) - Enable entanglement (default: true)

**Ports:**

- Inputs: `x` (Input), `mt_state` (MT State)
- Outputs: `out`, `new_mt_state`

**Color:** Deeper burgundy (#4A2A3A)

**Notes:** Experimental. Requires TrueRNG for authentic quantum randomness.

---

### IBM Quantum
Real quantum computation via IBM Quantum cloud.

**Parameters:**

- `num_qubits` (int) - Number of qubits (default: 4)
- `shots` (int) - Number of measurement shots (default: 100)
- `backend` (string) - IBM backend (default: "simulator")
- `entanglement_type` (string) - Entanglement type (default: "full")

**Ports:**

- Input: `classical_state` (Classical State)
- Output: `quantum_result` (Quantum Result)

**Color:** Deeper burgundy (#4A2A3A)

**Notes:** Requires IBM Quantum API key and account.

---

### Entropy Injection
Inject true quantum randomness using TrueRNG hardware.

**Parameters:**

- `noise_scale` (float) - Noise scale (default: 0.1)
- `use_hardware_rng` (bool) - Use hardware RNG (default: true)
- `distribution` (string) - Distribution type (default: "avalanche")

**Ports:**

- Input: `x` (Input)
- Output: `out`

**Color:** Deep orange (#FF5722)

**Notes:** Requires TrueRNG USB device.

---

## Asset Nodes

### Checkpoint
Trained weight checkpoint (.npz file). Loads pretrained weights.

**Parameters:**

- `checkpoint_path` (string) - Path to .npz file (default: "")
- `total_params` (int) - Total parameters (default: 0)
- `trained_epochs` (int) - Epochs trained (default: 0)
- `final_loss` (float) - Final loss (default: 0.0)

**Ports:**

- Output: `weights` (Provides weights to network)

**Color:** Brown (#795548)

**Usage:** Drag .npz file into canvas to create checkpoint node, then wire to network layers

---

## Node Type Summary

| Category | Node Types | Count |
|----------|-----------|-------|
| **I/O** | Input, Output | 2 |
| **Recurrent** | LSTM, GRU, RNN | 3 |
| **Feedforward** | Linear, Conv1D | 2 |
| **Attention** | Attention, Multi-Head Attention | 2 |
| **Activation** | Tanh, ReLU, GELU, Sigmoid, Softmax | 5 |
| **Normalization** | Layer Norm, Batch Norm | 2 |
| **Regularization** | Dropout | 1 |
| **CharmNetwork** | State Concat, State Split, Affect Head | 3 |
| **Quantum** | Quantum Microtubule, IBM Quantum, Entropy Injection | 3 |
| **Assets** | Checkpoint | 1 |
| **Total** | | **26** |

## Common Patterns

### Multi-Timescale Recurrent Network
```
Input → Fast LSTM → Medium LSTM → Slow GRU → State Concat → Affect Head
```

### Attention-Based Sequence Processing
```
Input → Multi-Head Attention → Layer Norm → Linear → Output
```

### Quantum-Enhanced Processing
```
Input → LSTM → Quantum Microtubule → Linear → Output
```

## See Also

- [NeuralAPI Reference](api/neural-api.md) - How to create and manipulate nodes
- [Complete Examples](examples.md) - Real-world usage patterns
- [Quick Start](quick-start.md) - Get started with Neural Canvas
