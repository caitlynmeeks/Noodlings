# generate_mlx_code()

Generate MLX Python code from the visual topology.

**Class**: NeuralNetworkProxy

**Access**: `network.generate_mlx_code()`

## Parameters

None

## Returns

Python source code string or `null` on failure

## Example

```javascript
var code = network.generate_mlx_code();

if (code) {
    context.log("Generated " + code.length + " characters of MLX code");
    // Code is a complete Python class with forward() method
}
```

## Generated Code Structure

```python
import mlx.core as mx
import mlx.nn as nn

class GeneratedNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.fast_lstm = nn.LSTM(input_dim=..., hidden_dim=16)
        # ... etc

    def forward(self, x):
        # Generated forward pass
        return output
```

## See Also

- [save()](save.md) - Save topology to file
- [get_parameter_count()](get_parameter_count.md) - Count parameters
