# set_node_property()

Set a node property.

**Class**: NeuralNetworkProxy

**Access**: `network.set_node_property(node_id, property_name, value)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `node_id` | string | Node UUID |
| `property_name` | string | Property name (e.g., "hidden_dim") |
| `value` | any | New value |

## Returns

`true` if set successfully, `false` on error

## Example

```javascript
// Double the hidden dimension
var node = network.get_node(lstm_id);
var new_dim = node.properties.hidden_dim * 2;

var success = network.set_node_property(lstm_id, "hidden_dim", new_dim);

if (success) {
    context.log("Increased hidden_dim to " + new_dim);
}
```

## Common Properties

- `hidden_dim` - Hidden layer size (LSTM, GRU)
- `num_heads` - Attention heads (MultiHeadAttention)
- `dropout` - Dropout rate
- `activation` - Activation function

## See Also

- [get_node()](get_node.md) - Get node properties
- [set_node_position()](set_node_position.md) - Set position
