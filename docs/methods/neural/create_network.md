# create_network()

Create a new empty neural network.

**Class**: NeuralAPI

**Access**: `context.noodle.neural.create_network(name)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `name` | string | Network name (optional, default: "Untitled") |

## Returns

NeuralNetworkProxy instance

## Example

```javascript
var network = context.noodle.neural.create_network("MyNetwork");

// Add nodes
var lstm = network.create_node("LSTM", {hidden_dim: 32});

context.log("Created network: " + network);
```

## See Also

- [load()](load.md) - Load network from file
- [get_network()](get_network.md) - Get network by ID
- [NeuralAPI](../../api/neural-api.md) - Complete class reference
