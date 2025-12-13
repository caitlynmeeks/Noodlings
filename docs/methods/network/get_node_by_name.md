# get_node_by_name()

Find node ID by name.

**Class**: NeuralNetworkProxy

**Access**: `network.get_node_by_name(name)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `name` | string | Node name (e.g., "Fast_LSTM", "Slow_GRU") |

## Returns

Node ID (UUID) or `null` if not found

## Example

```javascript
var fast_lstm_id = network.get_node_by_name("Fast_LSTM");

if (fast_lstm_id) {
    var node = network.get_node(fast_lstm_id);
    context.log("Found Fast LSTM with hidden_dim: " + node.properties.hidden_dim);
}
```

## Note

Node names can be changed in Neural Canvas by double-clicking.

## See Also

- [get_node()](get_node.md) - Get node by ID
- [create_node()](create_node.md) - Create nodes
