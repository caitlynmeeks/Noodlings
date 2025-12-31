# Neural Canvas

Visual programming for neural networks.

---

## What is Neural Canvas?

Neural Canvas is a node-based editor for designing neural network architectures.
It generates executable MLX/PyTorch code from visual graphs.

## Node Types

### Input/Output
- **Input**: Network entry point (specify dimensions)
- **Output**: Network exit point

### Layers
- **Linear**: Fully connected layer
- **Conv2d**: 2D convolution
- **LSTM**: Long short-term memory
- **GRU**: Gated recurrent unit
- **Attention**: Multi-head attention

### Activations
- **ReLU**, **GELU**, **Tanh**, **Sigmoid**, **Softmax**

### Operations
- **Add**: Element-wise addition
- **Concat**: Concatenate tensors
- **Reshape**: Change tensor shape
- **Dropout**: Regularization

### Normalization
- **LayerNorm**: Layer normalization
- **BatchNorm**: Batch normalization

## Creating a Network

1. Open Neural Canvas panel
2. Right-click canvas: Add node
3. Connect output ports to input ports
4. Set node properties in Inspector
5. View parameter count in status bar

## Example: Simple MLP

```
[Input 32] → [Linear 64] → [ReLU] → [Linear 10] → [Output]
```

Parameters: 32*64 + 64 + 64*10 + 10 = 2,762

## Code Generation

Neural Canvas can export to:

- **MLX**: Apple Silicon native
- **PyTorch**: Cross-platform

```python
# Generated MLX code
import mlx.core as mx
import mlx.nn as nn

class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(32, 64)
        self.linear2 = nn.Linear(64, 10)

    def __call__(self, x):
        x = nn.relu(self.linear1(x))
        x = self.linear2(x)
        return x
```

## Saving/Loading

Networks save to `.nncanvas` files (YAML format).

```yaml
nodes:
  - id: input_1
    type: Input
    position: [100, 200]
    properties:
      dimensions: [32]
  # ...
connections:
  - from: input_1
    to: linear_1
```

## Use in Facet Assemblies

Neural Canvas networks can be referenced from `CharmNetworkFacet` nodes
in facet assemblies, enabling learned components in cognitive architectures.
