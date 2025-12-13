# connect()

Connect two nodes via their ports.

**Class**: NeuralNetworkProxy

**Access**: `network.connect(from_node, from_port, to_node, to_port)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `from_node` | string | Source node ID |
| `from_port` | string | Source port name (e.g., "out") |
| `to_node` | string | Target node ID |
| `to_port` | string | Target port name (e.g., "input") |

## Returns

`true` if connected successfully, `false` on error

## Example

```javascript
// Connect LSTM output to GRU input
var success = network.connect(lstm_id, "out", gru_id, "input");

if (success) {
    context.log("Connected LSTM → GRU");
}
```

## Common Port Names

- Recurrent nodes (LSTM/GRU): `input`, `out`
- Linear nodes: `input`, `out`
- Affect Head: `hidden_state`, `affect_output`

## See Also

- [disconnect()](disconnect.md) - Disconnect nodes
- [create_node()](create_node.md) - Create nodes
- [get_node()](get_node.md) - Get node info
