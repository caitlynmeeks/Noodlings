# Neural Canvas Format

YAML format for neural network graphs.

---

## Overview

`.nncanvas` files store visual neural network architectures designed in
the Neural Canvas panel. They can be exported to executable MLX or PyTorch code.

## Structure

```yaml
version: 1
name: "my_network"

nodes:
  - id: input_1
    type: Input
    position: [100, 200]
    properties:
      dimensions: [32]

  - id: linear_1
    type: Linear
    position: [300, 200]
    properties:
      in_features: 32
      out_features: 64
      bias: true

  - id: relu_1
    type: ReLU
    position: [500, 200]

  - id: output_1
    type: Output
    position: [700, 200]

connections:
  - from: input_1
    from_port: output
    to: linear_1
    to_port: input

  - from: linear_1
    from_port: output
    to: relu_1
    to_port: input

  - from: relu_1
    from_port: output
    to: output_1
    to_port: input

metadata:
  created: "2025-12-30T12:00:00Z"
  parameter_count: 2112
```

## Node Types

### Input
```yaml
type: Input
properties:
  dimensions: [batch, features]  # or [batch, channels, height, width]
```

### Linear
```yaml
type: Linear
properties:
  in_features: 32
  out_features: 64
  bias: true
```

### LSTM
```yaml
type: LSTM
properties:
  input_size: 32
  hidden_size: 64
  num_layers: 1
  bidirectional: false
```

### Activation
```yaml
type: ReLU  # or GELU, Tanh, Sigmoid, Softmax
```

### Output
```yaml
type: Output
```

## Connections

```yaml
connections:
  - from: node_id
    from_port: output    # or specific port name
    to: other_node_id
    to_port: input
```

## Loading in Code

```python
from noodlestudio.core.neural_canvas.canvas_model import CanvasModel

model = CanvasModel()
model.load("network.nncanvas")

# Generate MLX code
mlx_code = model.generate_mlx_code()
```
