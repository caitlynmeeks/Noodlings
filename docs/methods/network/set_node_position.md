# set_node_position()

Set node position in canvas.

**Class**: NeuralNetworkProxy

**Access**: `network.set_node_position(node_id, x, y)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `node_id` | string | Node UUID |
| `x` | number | X coordinate |
| `y` | number | Y coordinate |

## Returns

`true` if set successfully, `false` on error

## Example

```javascript
network.set_node_position(lstm_id, 150, 250);
context.log("Moved LSTM to (150, 250)");
```

## See Also

- [get_node()](get_node.md) - Get current position
- [create_node()](create_node.md) - Set position during creation
