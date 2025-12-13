# disconnect()

Disconnect two nodes.

**Class**: NeuralNetworkProxy

**Access**: `network.disconnect(from_node, from_port, to_node, to_port)`

## Parameters

Same as [connect()](connect.md)

| Name | Type | Description |
|------|------|-------------|
| `from_node` | string | Source node ID |
| `from_port` | string | Source port name |
| `to_node` | string | Target node ID |
| `to_port` | string | Target port name |

## Returns

`true` if disconnected successfully, `false` on error

## Example

```javascript
network.disconnect(lstm_id, "out", gru_id, "input");
context.log("Disconnected LSTM from GRU");
```

## See Also

- [connect()](connect.md) - Connect nodes
- [remove_node()](remove_node.md) - Remove node
