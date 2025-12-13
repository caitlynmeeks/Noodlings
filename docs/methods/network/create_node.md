# create_node()

Create a new node in the network.

**Class**: NeuralNetworkProxy

**Access**: `network.create_node(node_type, properties)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `node_type` | string | Node type (e.g., "LSTM", "GRU", "Linear") |
| `properties` | object | Node properties (see below) |

### Common Properties

| Property | Type | Description |
|----------|------|-------------|
| `hidden_dim` | number | Hidden dimension size |
| `position` | array | Canvas position `[x, y]` |

## Returns

Node ID (UUID string) or `null` on failure

## Example

```javascript
var lstm_id = network.create_node("LSTM", {
    hidden_dim: 32,
    position: [100, 200]
});

var gru_id = network.create_node("GRU", {
    hidden_dim: 16,
    position: [300, 200]
});

context.log("Created LSTM: " + lstm_id);
```

## Supported Node Types

See [Node Types Reference](../../node-types.md) for complete list (26 types)

## See Also

- [remove_node()](remove_node.md) - Remove node
- [connect()](connect.md) - Connect nodes
- [get_node()](get_node.md) - Get node info
