# remove_node()

Remove a node from the network.

**Class**: NeuralNetworkProxy

**Access**: `network.remove_node(node_id)`

## Parameters

| Name | Type | Description |
|------|------|-------------|
| `node_id` | string | Node UUID |

## Returns

`true` if removed successfully, `false` on error

## Example

```javascript
var removed = network.remove_node(lstm_id);

if (removed) {
    context.log("Node removed");
}
```

## Warning

Removing nodes will also remove all connections to/from that node.

## See Also

- [create_node()](create_node.md) - Create nodes
- [disconnect()](disconnect.md) - Disconnect before removing
